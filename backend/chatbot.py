"""
Retrieval-augmented question-answering orchestrator.

Pipeline:
  1. [R] Retrieve top-K most relevant reviews via TF-IDF cosine similarity
         (see retriever.py), with optional filters: location, hospital,
         aspect (via multilingual lexicon), and sentiment label.
  2. [A] Assemble a grounded prompt: system rules + hospital-level statistics
         + numbered retrieved reviews + user question.
  3. [G] Generate the answer with an LLM. The provider is chosen at load
         time from LLM_PROVIDER in .env, with automatic fallback to any
         other configured provider if the primary one errors.
  4. If no LLM is available (missing keys, network error, quota), gracefully
     fall back to pattern_qa - a deterministic rule-based Q&A over the same
     precomputed statistics.

Supported providers:
  - groq    (llama-3.3-70b-versatile)         [preferred]
  - gemini  (gemini-flash-latest)             [backup]

Design: the interface `answer(question) -> dict` is identical across all
paths; only the `model_used` field differs.
"""

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

from backend import hospital_stats, retriever, pattern_qa

load_dotenv()

# ─── PROVIDER CONFIG ────────────────────────────────────────────────────────

PRIMARY_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

GROQ_KEY   = os.getenv("GROQ_API_KEY",   "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GROQ_MODEL   = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-flash-latest"

_groq_client   = None
_gemini_model  = None
_available: List[str] = []   # provider names in order of preference


def _placeholder(key: str) -> bool:
    return (not key) or key.lower().startswith(("your_", "paste_"))


def _init_groq() -> bool:
    global _groq_client
    if _groq_client is not None:
        return True
    if _placeholder(GROQ_KEY):
        return False
    try:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_KEY)
        print(f"[chatbot] Groq configured: {GROQ_MODEL}")
        return True
    except Exception as e:
        print(f"[chatbot] Groq init failed: {e}")
        return False


def _init_gemini() -> bool:
    global _gemini_model
    if _gemini_model is not None:
        return True
    if _placeholder(GEMINI_KEY):
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        print(f"[chatbot] Gemini configured: {GEMINI_MODEL}")
        return True
    except Exception as e:
        print(f"[chatbot] Gemini init failed: {e}")
        return False


def _init_providers():
    """Populate _available in the order we should try providers."""
    global _available
    if _available:
        return
    order = ["groq", "gemini"]
    if PRIMARY_PROVIDER in order:
        order.remove(PRIMARY_PROVIDER)
        order.insert(0, PRIMARY_PROVIDER)
    for name in order:
        ok = (_init_groq() if name == "groq" else _init_gemini())
        if ok:
            _available.append(name)
    if not _available:
        print("[chatbot] no LLM provider configured - will use pattern-based fallback")


# ─── PROMPT ASSEMBLY ────────────────────────────────────────────────────────

SYSTEM_RULES = """You are a healthcare-feedback analyst for Malaysian hospitals.
You have access to a curated dataset of ~2,161 public reviews from 14 hospitals
in KL / Klang Valley and Melaka.

STRICT RULES:
- Answer ONLY based on the DATA below (aggregate statistics + retrieved reviews).
- Do NOT invent hospital names, statistics, or reviews. If the data is
  insufficient, say so explicitly.
- For "best" questions, name the hospital with the HIGHEST positive_rate for
  the relevant aspect. For "worst" questions, name the hospital with the
  LOWEST positive_rate for the relevant aspect. Double-check by scanning ALL
  hospitals in the statistics before deciding.
- Cite specific hospitals when making a claim. Reference retrieved reviews
  by their number [1], [2], etc. when quoting evidence.
- Be concise: 3-6 sentences. Use bullets for comparisons.
- Do NOT give medical advice. Only report what the data shows.
- Do NOT wrap your answer in code blocks or JSON. Reply in plain text
  (markdown allowed).
"""


def _stats_summary(stats: Dict[str, Dict]) -> str:
    """Compact text summary of hospital-level stats for the LLM context.
    Deliberately not JSON: prose is more token-efficient here."""
    lines = []
    for name, s in stats.items():
        overall = s["overall_sentiment"]
        aspect_parts = []
        for a, v in s["aspects"].items():
            aspect_parts.append(
                f"{a}={v['positive_rate']:.0%} pos ({v['mentions']}n)")
        lines.append(
            f"- {name} ({s['location']}, {s['total_reviews']} reviews): "
            f"overall {overall['positive']:.0%} pos / "
            f"{overall['neutral']:.0%} neu / {overall['negative']:.0%} neg. "
            f"Aspects: {', '.join(aspect_parts)}"
        )
    return "\n".join(lines)


def _reviews_block(reviews: List[Dict]) -> str:
    if not reviews:
        return "(No reviews retrieved above similarity threshold.)"
    lines = []
    for i, r in enumerate(reviews, 1):
        text = (r["text"][:280] + "...") if len(r["text"]) > 280 else r["text"]
        lines.append(f"[{i}] Hospital: {r['hospital']} | Date: {r['date']} "
                     f"| Stars: {r['stars']} | Similarity: {r['similarity']:.2f}\n"
                     f"    \"{text}\"")
    return "\n".join(lines)


def _build_prompt(question: str, reviews: List[Dict]) -> str:
    stats_txt = _stats_summary(hospital_stats.load())
    return (
        f"{SYSTEM_RULES}\n\n"
        f"HOSPITAL AGGREGATE STATISTICS:\n{stats_txt}\n\n"
        f"RETRIEVED REVIEWS (top-{len(reviews)} by cosine similarity to the query):\n"
        f"{_reviews_block(reviews)}\n\n"
        f"USER QUESTION: {question}\n\nANSWER:"
    )


# ─── LLM CALL DISPATCH ──────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    resp = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=700,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_gemini(prompt: str) -> str:
    resp = _gemini_model.generate_content(prompt)
    return (resp.text or "").strip()


_CALLERS = {"groq": _call_groq, "gemini": _call_gemini}
_MODEL_NAMES = {"groq": GROQ_MODEL, "gemini": GEMINI_MODEL}


# ─── RETRIEVAL WITH PROGRESSIVE RELAXATION ─────────────────────────────────

def _retrieve_with_progressive_relaxation(q, location, hospital_filter,
                                          aspect_filter, label_filter):
    """Try the strictest filter combination first; relax progressively
    (drop the label constraint, then the aspect constraint) if too few
    reviews come back. This preserves precision when it exists and
    prevents empty citations when the query is too narrow."""
    tiers = [
        (location, hospital_filter, aspect_filter, label_filter),
        (location, hospital_filter, aspect_filter, None),
        (location, hospital_filter, None,          None),
        (None,     hospital_filter, None,          None),
        (None,     None,            None,          None),
    ]
    seen = set()
    results: List[Dict] = []
    for loc, hosp, asp_f, lab in tiers:
        args = (loc, hosp, asp_f, lab)
        if args in seen:
            continue
        seen.add(args)
        results = retriever.retrieve(
            q, k=10,
            filter_location=loc, filter_hospital=hosp,
            filter_aspect=asp_f, filter_label=lab,
        )
        if len(results) >= 3:
            return results
    return results


# ─── PUBLIC ENTRY ───────────────────────────────────────────────────────────

def answer(question: str) -> Dict:
    """Main entry. Returns dict with:
        answer, cited_reviews, model_used, sources_count, intent, fallback_reason
    """
    q = question.strip()
    if not q:
        return {
            "answer":          "Please ask a question.",
            "cited_reviews":   [],
            "model_used":      "pattern-based (fallback)",
            "sources_count":   0,
            "intent":          "EMPTY",
            "fallback_reason": "empty question",
        }

    # 1. Extract soft filters from question for smarter retrieval
    intent    = pattern_qa.detect_intent(q)
    location  = pattern_qa.extract_location(q)
    aspect    = pattern_qa.extract_aspect(q)
    hospitals = pattern_qa.extract_hospitals(q)
    hospital_filter = hospitals[0] if len(hospitals) == 1 else None

    # Sentiment-intent alignment
    label_filter = None
    ql = q.lower()
    if intent == "BEST":
        label_filter = "positive"
    elif intent == "WORST":
        label_filter = "negative"
    elif any(w in ql for w in ("negative", "complain", "bad review",
                                "worst review", "unhappy")):
        label_filter = "negative"
    elif any(w in ql for w in ("positive", "praise", "good review",
                                "best review", "happy")):
        label_filter = "positive"

    # 2. Retrieve with progressive relaxation
    retrieved = _retrieve_with_progressive_relaxation(
        q, location, hospital_filter, aspect, label_filter,
    )

    # 3. Try each configured LLM provider in preference order
    _init_providers()
    prompt = None
    fb_reasons = []
    for provider in _available:
        try:
            if prompt is None:
                prompt = _build_prompt(q, retrieved)
            text = _CALLERS[provider](prompt)
            if text:
                return {
                    "answer":        text,
                    "cited_reviews": retrieved[:5],
                    "model_used":    f"{provider} ({_MODEL_NAMES[provider]})",
                    "sources_count": len(retrieved),
                    "intent":        intent,
                    "fallback_reason": None,
                }
            fb_reasons.append(f"{provider}: empty response")
        except Exception as e:
            reason = f"{provider}: {type(e).__name__}"
            fb_reasons.append(reason)
            print(f"[chatbot] {reason}: {str(e)[:150]}")

    # 4. Fallback
    fb = "; ".join(fb_reasons) if fb_reasons else "no LLM configured"
    return pattern_qa.answer(q, retrieved=retrieved, fallback_reason=fb)


# Initialize at import time so the first request is fast.
_init_providers()


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    tests = [
        "Which KL hospital has the best doctors?",
        "Which Melaka hospital has the worst waiting time?",
        "Compare Sunway Medical and Gleneagles",
        "What do people say about cleanliness at Hospital Alor Gajah?",
        "Show me negative reviews about waiting time",
    ]
    for q in tests:
        print("=" * 70)
        print("Q:", q)
        r = answer(q)
        print(f"[{r['model_used']}] intent={r.get('intent')}")
        print(r["answer"])
        print()
        print("SOURCES:")
        for i, c in enumerate(r["cited_reviews"][:3], 1):
            print(f"  [{i}] {c['hospital'][:35]:35s} sim={c['similarity']:.2f}")
            print(f'      "{c["text"][:110]}"')
        print()
