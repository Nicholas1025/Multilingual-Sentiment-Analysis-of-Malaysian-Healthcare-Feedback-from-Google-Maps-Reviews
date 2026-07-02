"""
Prediction service for the FastAPI app.

Loads the trained sentiment model(s) and exposes a single function:
    predict(text) -> dict
The dict contains: label, confidence, language, emojis, aspects, cleaned text.

Tries the fine-tuned XLM-RoBERTa transformer first (best quality);
falls back to the TF-IDF + Logistic Regression baseline if not available.
"""

import os
import sys
import pickle
from typing import Dict, List, Optional

import numpy as np

# Inject the project's own modules so we can reuse preprocessing + aspect logic
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Source Codes", "2_preprocessing"))
sys.path.insert(0, os.path.join(ROOT, "Source Codes", "5_advanced"))

import preprocess              # noqa: E402
import aspect_sentiment        # noqa: E402

# Make sure NLTK corpora (stopwords / punkt / wordnet) are present.
# First-run users won't have them — download once at startup so /api/predict
# does not crash on the first request.
preprocess.ensure_nltk_data()

MODELS_DIR        = os.path.join(ROOT, "models")
TRANSFORMER_DIR   = os.path.join(MODELS_DIR, "transformer")
TFIDF_VEC_PATH    = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
BASELINE_PATH     = os.path.join(MODELS_DIR, "best_baseline.pkl")

LABELS = ["positive", "neutral", "negative"]


# ── singletons ─────────────────────────────────────────────────────────

_vectorizer = None
_baseline   = None
_baseline_name = None
_transformer = None
_tokenizer   = None
_id2label    = None


def _load_baseline():
    global _vectorizer, _baseline, _baseline_name
    if _baseline is not None:
        return
    if not (os.path.exists(TFIDF_VEC_PATH) and os.path.exists(BASELINE_PATH)):
        return
    with open(TFIDF_VEC_PATH, "rb") as f:
        _vectorizer = pickle.load(f)
    with open(BASELINE_PATH, "rb") as f:
        payload = pickle.load(f)
    _baseline = payload["model"]
    _baseline_name = payload["name"]
    print(f"[predict_service] loaded baseline: {_baseline_name}")


def _load_transformer():
    global _transformer, _tokenizer, _id2label
    if _transformer is not None:
        return
    if not os.path.isdir(TRANSFORMER_DIR):
        return
    try:
        import torch                                                  # noqa: F401
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification)
        _tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        _transformer = AutoModelForSequenceClassification.from_pretrained(
            TRANSFORMER_DIR
        )
        _transformer.eval()
        _id2label = _transformer.config.id2label
        print("[predict_service] loaded transformer (XLM-RoBERTa)")
    except Exception as e:
        print(f"[predict_service] transformer not loaded: {e}")
        _transformer = None


# ── inference primitives ───────────────────────────────────────────────

def _predict_baseline(text_cleaned: str) -> Dict:
    """Return label, confidence and full score distribution from baseline."""
    X = _vectorizer.transform([text_cleaned])
    pred = _baseline.predict(X)[0]
    if hasattr(_baseline, "predict_proba"):
        probs = _baseline.predict_proba(X)[0]
        classes = list(_baseline.classes_)
        scores = {c: float(probs[i]) for i, c in enumerate(classes)}
        conf = float(probs.max())
    else:                                                     # LinearSVM path
        margins = _baseline.decision_function(X)[0]
        if margins.ndim == 0:
            margins = np.array([margins])
        exp = np.exp(margins - margins.max())
        soft = exp / exp.sum()
        classes = list(_baseline.classes_)
        scores = {c: float(soft[i]) for i, c in enumerate(classes)}
        conf = float(soft.max())
    return {"label": pred, "confidence": conf, "scores": scores}


def _predict_transformer(text: str) -> Dict:
    import torch
    enc = _tokenizer(text, truncation=True, max_length=192,
                     return_tensors="pt")
    with torch.no_grad():
        out = _transformer(**enc)
    logits = out.logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    label_id = int(probs.argmax())
    label = _id2label[label_id]
    scores = {_id2label[i]: float(probs[i]) for i in range(len(probs))}
    return {"label": label, "confidence": float(probs[label_id]),
            "scores": scores}


# ── single sentence classifier (used by Aspect-Based SA) ───────────────

def _classify_sentence(s: str) -> str:
    """Lightweight classifier for short aspect-level sentences."""
    if _transformer is not None:
        return _predict_transformer(s)["label"]
    if _baseline is not None:
        result = preprocess.process_text(s)
        return _predict_baseline(result["text_cleaned"])["label"]
    return "neutral"


# ── public API ─────────────────────────────────────────────────────────

def init_models():
    _load_transformer()
    _load_baseline()
    if _transformer is None and _baseline is None:
        print("[predict_service] WARNING: no model is loaded - "
              "train one first (Stage 4).")


def _top_words_baseline(cleaned: str, label: str, k: int = 5):
    """Return the k tokens in `cleaned` that contributed most to the
    predicted label according to the LR coefficients. Works with both
    LogisticRegression (coef_) and LinearSVC (coef_)."""
    if _vectorizer is None or _baseline is None:
        return None
    if not hasattr(_baseline, "coef_"):
        return None
    try:
        classes = list(_baseline.classes_)
        if label not in classes:
            return None
        cls_idx = classes.index(label)
        coefs = _baseline.coef_[cls_idx]   # (n_features,)
        vec = _vectorizer.transform([cleaned])
        cols, vals = vec.indices, vec.data
        if len(cols) == 0:
            return []
        contributions = vals * coefs[cols]    # element-wise contribution
        vocab = _vectorizer.get_feature_names_out()
        scored = [(vocab[c], float(contributions[i]))
                  for i, c in enumerate(cols)]
        scored.sort(key=lambda kv: -kv[1])
        return [{"word": w, "weight": round(s, 4)} for w, s in scored[:k]]
    except Exception:
        return None


def predict(text: str) -> Dict:
    """Full prediction pipeline used by /api/predict."""
    proc = preprocess.process_text(text)
    cleaned = proc["text_cleaned"]
    lang    = proc["language"]
    emojis  = proc["emojis_found"]

    if _transformer is not None:
        core = _predict_transformer(text)
        model_used = "transformer"
    elif _baseline is not None:
        core = _predict_baseline(cleaned)
        model_used = "baseline"
    else:
        core = {"label": "neutral", "confidence": 0.0, "scores": {}}
        model_used = "none"

    aspects = aspect_sentiment.aspect_sentiments(text, _classify_sentence)

    # Explainability via baseline coefficients (always available since LR is loaded).
    top_words = _top_words_baseline(cleaned, core["label"])

    return {
        "label":         core["label"],
        "confidence":    core["confidence"],
        "class_scores":  core["scores"],
        "language":      lang,
        "emojis":        emojis,
        "aspects":       aspects,
        "text_cleaned":  cleaned,
        "model_used":    model_used,
        "top_words":     top_words,
    }


# ── module-level init when imported by FastAPI ─────────────────────────

init_models()
