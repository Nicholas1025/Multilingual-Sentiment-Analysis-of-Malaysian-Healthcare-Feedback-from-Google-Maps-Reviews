"""
Stage 1+ : AI weak-supervision labelling.

Uses the publicly released 2025 multilingual sentiment model
    tabularisai/multilingual-sentiment-analysis
(5 classes, 23 languages including English, Malay, Chinese)
to give every review an initial label + confidence. Output is then
narrowed to the ambiguous cases for human verification by
label_content.py.

Methodology follows R15 (Sim et al., 2025, JMIR Formative Research):
"Large-Language-Model-Assisted Content Analysis" — two-stage labelling
where a pre-trained model proposes labels and humans verify only the
uncertain ones, instead of labelling 1,000+ reviews from scratch.

Inputs:
    Data/raw/healthcare_raw.csv

Outputs:
    Data/raw/healthcare_ai_labeled.csv
        original columns + ai_label_5 + ai_label + ai_confidence + needs_review

Run once. Re-running just refreshes the AI labels (resumes naturally).
"""

import os
import sys
from collections import Counter

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE  = os.path.join(os.path.dirname(__file__), "..", "..")
RAW   = os.path.join(BASE, "Data", "raw", "healthcare_raw.csv")
OUT   = os.path.join(BASE, "Data", "raw", "healthcare_ai_labeled.csv")

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
BATCH_SIZE = 16          # CPU-safe; on GPU you can bump to 64
MAX_LEN    = 256
LOW_CONF_THRESHOLD = 0.70   # below this -> mark for human review

# tabularisai outputs 5 labels; collapse to our 3-class scheme.
COLLAPSE = {
    "Very Negative": "negative",
    "Negative":      "negative",
    "Neutral":       "neutral",
    "Positive":      "positive",
    "Very Positive": "positive",
}


def main():
    # --- 1. load model -------------------------------------------------
    print(f"[1] Loading model {MODEL_NAME} ...", flush=True)
    print("    (first run downloads ~400MB; cached afterwards)")

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    id2label = model.config.id2label
    print(f"    label set: {list(id2label.values())}")

    # --- 2. load data --------------------------------------------------
    if not os.path.exists(RAW):
        sys.exit(f"[!] Not found: {RAW}  - run the scraper first.")
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 10].reset_index(drop=True)
    n = len(df)
    print(f"[2] {n} reviews to label\n")

    # --- 3. batched inference -----------------------------------------
    five_labels   = []
    three_labels  = []
    confidences   = []

    for start in range(0, n, BATCH_SIZE):
        batch_texts = df["text"].iloc[start:start + BATCH_SIZE].tolist()
        enc = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        for row in probs:
            top_id = int(row.argmax())
            top_label = id2label[top_id]
            five_labels.append(top_label)
            three_labels.append(COLLAPSE.get(top_label, "neutral"))
            confidences.append(float(row[top_id]))

        done = min(start + BATCH_SIZE, n)
        if done % 80 == 0 or done == n:
            print(f"    {done}/{n}", flush=True)

    # --- 4. assemble output -------------------------------------------
    df["ai_label_5"]   = five_labels
    df["ai_label"]     = three_labels
    df["ai_confidence"] = confidences
    df["needs_review"]  = (
        (df["ai_label"] == "neutral") |
        (df["ai_confidence"] < LOW_CONF_THRESHOLD)
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n[3] Saved -> {OUT}\n")

    # --- 5. summary report --------------------------------------------
    line = "=" * 60
    print(f"{line}\n  AI LABELLING REPORT\n{line}")

    print("\n  5-class distribution:")
    for lab, c in Counter(df["ai_label_5"]).most_common():
        print(f"    {lab:<15} {c:>5}  ({100*c/n:.1f}%)")

    print("\n  3-class distribution (collapsed):")
    for lab, c in Counter(df["ai_label"]).most_common():
        print(f"    {lab:<10} {c:>5}  ({100*c/n:.1f}%)")

    if "label" in df.columns:
        agree = int((df["ai_label"] == df["label"]).sum())
        print(f"\n  AI vs star-label agreement: {agree}/{n} ({100*agree/n:.1f}%)")

    print("\n  Confidence:")
    print(f"    mean={df['ai_confidence'].mean():.3f}  "
          f"median={df['ai_confidence'].median():.3f}  "
          f"min={df['ai_confidence'].min():.3f}")
    low_conf = int((df["ai_confidence"] < LOW_CONF_THRESHOLD).sum())
    print(f"    below {LOW_CONF_THRESHOLD}: {low_conf}  "
          f"({100*low_conf/n:.1f}%)")

    review_count = int(df["needs_review"].sum())
    print(f"\n  Reviews flagged for human verification: "
          f"{review_count}  ({100*review_count/n:.1f}%)")
    print(f"  (these are: all AI-Neutral + everything below "
          f"confidence {LOW_CONF_THRESHOLD})")

    print(f"\n  Next: run  label_content.py  to verify the "
          f"{review_count} flagged reviews.")
    print(line)


if __name__ == "__main__":
    main()
