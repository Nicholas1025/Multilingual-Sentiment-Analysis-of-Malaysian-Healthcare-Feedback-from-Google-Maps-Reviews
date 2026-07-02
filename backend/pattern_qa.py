"""
Pattern-based question-answering fallback.

Used when the Gemini API is unavailable (missing key, quota, network).
Handles 4 core intents (BEST / WORST / COMPARE / INFO) plus a friendly
unknown-intent fallback with suggested questions.

All answers are grounded in the precomputed hospital_stats.STATS - so this
module is completely offline and cannot hallucinate.
"""

import re
from typing import Dict, List, Optional

from backend import hospital_stats

# ─── ENTITY DICTIONARIES ────────────────────────────────────────────────────

# Aspect keywords (align with 5_advanced/aspect_sentiment.py's ASPECTS keys)
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "doctor":       ["doctor", "doctors", "dr", "physician", "specialist",
                     "consultant", "surgeon", "doktor", "pakar"],
    "nurse_staff":  ["nurse", "nurses", "staff", "receptionist",
                     "jururawat", "kakitangan"],
    "waiting_time": ["wait", "waiting", "queue", "long", "slow",
                     "delay", "hours", "tunggu", "lambat"],
    "facilities":   ["facility", "facilities", "room", "ward",
                     "parking", "kemudahan"],
    "cost":         ["cost", "price", "expensive", "cheap", "affordable",
                     "fee", "charge", "bill", "mahal", "murah"],
    "treatment":    ["treatment", "medicine", "medication", "therapy",
                     "operation", "surgery", "rawatan", "ubat"],
    "cleanliness":  ["clean", "cleanliness", "dirty", "hygiene",
                     "hygienic", "bersih", "kotor"],
}

# Hospital short-name aliases (case-insensitive substring match)
HOSPITAL_ALIASES: Dict[str, str] = {
    "sunway":            "Sunway Medical Centre",
    "gleneagles":        "Gleneagles Hospital Kuala Lumpur",
    "mahkota":           "Mahkota Medical Centre Melaka",
    "pantai":            "Pantai Hospital Ayer Keroh Melaka",
    "ayer keroh":        "Pantai Hospital Ayer Keroh Melaka",
    "columbia pj":       "Columbia Asia Hospital Petaling Jaya",
    "columbia petaling": "Columbia Asia Hospital Petaling Jaya",
    "columbia bukit":    "Columbia Asia Hospital Bukit Rambai Melaka",
    "columbia rambai":   "Columbia Asia Hospital Bukit Rambai Melaka",
    "columbia melaka":   "Columbia Asia Hospital Bukit Rambai Melaka",
    "oriental":          "Oriental Melaka Straits Medical Centre",
    "putra":             "Putra Specialist Hospital Melaka",
    "jasin":             "Hospital Jasin Melaka",
    "alor gajah":        "Hospital Alor Gajah Melaka",
    "hospital melaka":   "Hospital Melaka",
    "peringgit":         "Klinik Kesihatan Peringgit Melaka",
    "pergigian":         "Klinik Pergigian Melaka",
    "qhc":               "QHC Medical Centre Melaka",
}

LOCATION_KEYWORDS: Dict[str, List[str]] = {
    "KL":     ["kl", "kuala lumpur", "klang valley", "petaling jaya",
               "pj", "selangor"],
    "Melaka": ["melaka", "malacca"],
}

BEST_MARKERS  = ["best", "top", "highest", "leading",
                 "most positive", "recommend"]
WORST_MARKERS = ["worst", "lowest", "weakest", "poorest",
                 "most negative", "longest", "avoid"]


# ─── ENTITY EXTRACTORS ──────────────────────────────────────────────────────

def _match_word(text: str, kw: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])" + re.escape(kw) + r"(?![A-Za-z])",
                          text, flags=re.IGNORECASE))


def extract_aspect(q: str) -> Optional[str]:
    for aspect, kws in ASPECT_KEYWORDS.items():
        if any(_match_word(q, k) for k in kws):
            return aspect
    return None


def extract_hospitals(q: str) -> List[str]:
    """Return full hospital names mentioned in the query."""
    found = set()
    ql = q.lower()
    # Try full names first (longest match)
    for h in sorted(hospital_stats.load().keys(),
                    key=lambda x: -len(x)):
        if h.lower() in ql:
            found.add(h)
    # Then aliases
    for alias, full in HOSPITAL_ALIASES.items():
        if alias in ql:
            found.add(full)
    return sorted(found)


def extract_location(q: str) -> Optional[str]:
    ql = q.lower()
    for loc, kws in LOCATION_KEYWORDS.items():
        if any(kw in ql for kw in kws):
            return loc
    return None


def detect_intent(q: str) -> str:
    ql = q.lower()
    hospitals = extract_hospitals(q)
    if len(hospitals) >= 2 or "compare" in ql or "vs" in ql or "versus" in ql:
        return "COMPARE"
    if any(m in ql for m in WORST_MARKERS):
        return "WORST"
    if any(m in ql for m in BEST_MARKERS):
        return "BEST"
    if len(hospitals) == 1:
        return "INFO"
    if ql.startswith(("how many", "count")):
        return "STATS"
    if any(w in ql for w in ["show", "give me", "list", "example"]):
        return "REVIEWS"
    return "UNKNOWN"


# ─── ANSWER TEMPLATES ───────────────────────────────────────────────────────

_ASPECT_PRETTY = {
    "doctor": "doctors", "nurse_staff": "nurses/staff",
    "waiting_time": "waiting time", "facilities": "facilities",
    "cost": "cost/affordability", "treatment": "treatment",
    "cleanliness": "cleanliness",
}


def _pretty(aspect: str) -> str:
    return _ASPECT_PRETTY.get(aspect, aspect.replace("_", " "))


def _fmt_pct(x: float) -> str:
    return f"{x*100:.0f}%"


def answer_best(q: str, retrieved: List[Dict]) -> Dict:
    aspect = extract_aspect(q)
    loc    = extract_location(q)

    ranked = hospital_stats.rank_by(aspect=aspect, location=loc,
                                    metric="positive_rate", reverse=True)
    if not ranked:
        return _no_data(loc, aspect)
    ranked = [r for r in ranked if r["_mentions"] >= 5][:3]
    if not ranked:
        return _no_data(loc, aspect)

    what = _pretty(aspect) + " sentiment" if aspect else "overall positive rate"
    where = f" in {loc}" if loc else ""
    lines = [f"Based on our dataset, here are the top hospitals for "
             f"**{what}**{where}:", ""]
    for i, h in enumerate(ranked, 1):
        lines.append(f"{i}. **{h['hospital']}** — "
                     f"{_fmt_pct(h['_score'])} positive "
                     f"(n={h['_mentions']} reviews)")
    return {
        "answer": "\n".join(lines),
        "cited_reviews": retrieved[:3],
        "intent": "BEST",
    }


def answer_worst(q: str, retrieved: List[Dict]) -> Dict:
    aspect = extract_aspect(q)
    loc    = extract_location(q)

    ranked = hospital_stats.rank_by(aspect=aspect, location=loc,
                                    metric="positive_rate", reverse=False)
    if not ranked:
        return _no_data(loc, aspect)
    ranked = [r for r in ranked if r["_mentions"] >= 5][:3]
    if not ranked:
        return _no_data(loc, aspect)

    what = _pretty(aspect) + " sentiment" if aspect else "overall positive rate"
    where = f" in {loc}" if loc else ""
    lines = [f"The following hospitals rank lowest on **{what}**{where}:", ""]
    for i, h in enumerate(ranked, 1):
        lines.append(f"{i}. **{h['hospital']}** — only "
                     f"{_fmt_pct(h['_score'])} positive "
                     f"(n={h['_mentions']} reviews)")
    return {
        "answer": "\n".join(lines),
        "cited_reviews": retrieved[:3],
        "intent": "WORST",
    }


def answer_compare(q: str, retrieved: List[Dict]) -> Dict:
    hospitals = extract_hospitals(q)[:3]
    if len(hospitals) < 2:
        return {
            "answer": "Please name at least two hospitals to compare.\n"
                      "Example: *Compare Sunway and Gleneagles*.",
            "cited_reviews": retrieved[:3],
            "intent": "COMPARE",
        }

    stats = [hospital_stats.get_hospital(h) for h in hospitals]
    stats = [s for s in stats if s]

    lines = ["**Comparison** (positive sentiment rate; higher is better):", ""]
    lines.append("| Metric | " + " | ".join(f"**{s['hospital'][:25]}**" for s in stats) + " |")
    lines.append("| --- | " + " | ".join(":---:" for _ in stats) + " |")
    lines.append("| Overall | " +
                 " | ".join(_fmt_pct(s["positive_rate"]) for s in stats) + " |")

    for aspect in ASPECT_KEYWORDS.keys():
        row = []
        any_data = False
        for s in stats:
            if aspect in s["aspects"]:
                any_data = True
                v = s["aspects"][aspect]
                row.append(f"{_fmt_pct(v['positive_rate'])} (n={v['mentions']})")
            else:
                row.append("—")
        if any_data:
            lines.append(f"| {_pretty(aspect).capitalize()} | " +
                         " | ".join(row) + " |")

    lines.append("")
    lines.append("Sample sizes: " +
                 ", ".join(f"{s['hospital'].split()[0]} n={s['total_reviews']}"
                           for s in stats))
    return {
        "answer": "\n".join(lines),
        "cited_reviews": retrieved[:3],
        "intent": "COMPARE",
    }


def answer_info(q: str, retrieved: List[Dict]) -> Dict:
    hospitals = extract_hospitals(q)
    if not hospitals:
        return _unknown(retrieved)
    h = hospital_stats.get_hospital(hospitals[0])
    if not h:
        return _unknown(retrieved)

    aspect = extract_aspect(q)

    lines = [f"**{h['hospital']}** ({h['location']}, "
             f"{h['total_reviews']} reviews)", ""]

    if aspect and aspect in h["aspects"]:
        v = h["aspects"][aspect]
        lines.append(f"On **{_pretty(aspect)}**: "
                     f"{_fmt_pct(v['positive_rate'])} positive, "
                     f"{_fmt_pct(v['negative_rate'])} negative "
                     f"({v['mentions']} mentions).")
        if v.get("sample_positive"):
            lines.append("")
            lines.append("*Positive example:* \"" +
                         v["sample_positive"][0][:180] + "\"")
        if v.get("sample_negative"):
            lines.append("*Negative example:* \"" +
                         v["sample_negative"][0][:180] + "\"")
    else:
        s = h["overall_sentiment"]
        lines.append(f"Overall sentiment: "
                     f"{_fmt_pct(s['positive'])} positive, "
                     f"{_fmt_pct(s['neutral'])} neutral, "
                     f"{_fmt_pct(s['negative'])} negative.")
        # Top strengths / weaknesses
        aspects = h["aspects"]
        if aspects:
            top = sorted(aspects.items(),
                         key=lambda kv: -kv[1]["positive_rate"])
            best = top[0]
            worst = top[-1]
            lines.append("")
            lines.append(f"Strongest aspect: **{_pretty(best[0])}** "
                         f"({_fmt_pct(best[1]['positive_rate'])} positive).")
            lines.append(f"Weakest aspect: **{_pretty(worst[0])}** "
                         f"({_fmt_pct(worst[1]['positive_rate'])} positive).")

    return {
        "answer": "\n".join(lines),
        "cited_reviews": retrieved[:3],
        "intent": "INFO",
    }


def answer_stats(q: str, retrieved: List[Dict]) -> Dict:
    hospitals = extract_hospitals(q)
    if hospitals:
        h = hospital_stats.get_hospital(hospitals[0])
        if h:
            return {
                "answer": f"**{h['hospital']}** has "
                          f"**{h['total_reviews']} reviews** in our dataset "
                          f"({h['location']}).",
                "cited_reviews": retrieved[:3],
                "intent": "STATS",
            }
    # dataset-wide
    all_h = hospital_stats.all_hospitals()
    total = sum(h["total_reviews"] for h in all_h)
    return {
        "answer": f"Our dataset contains **{total} reviews** across "
                  f"**{len(all_h)} hospitals** (KL and Melaka).",
        "cited_reviews": retrieved[:3],
        "intent": "STATS",
    }


def answer_reviews(q: str, retrieved: List[Dict]) -> Dict:
    if not retrieved:
        return _unknown(retrieved)
    lines = [f"Here are the top {min(len(retrieved), 3)} reviews matching "
             f"your query:", ""]
    for i, r in enumerate(retrieved[:3], 1):
        text = r["text"][:220] + ("..." if len(r["text"]) > 220 else "")
        lines.append(f"**{i}.** \"{text}\"")
        lines.append(f"  — {r['hospital']} · {r['date']} "
                     f"· similarity {r['similarity']:.2f}")
        lines.append("")
    return {
        "answer": "\n".join(lines),
        "cited_reviews": retrieved[:5],
        "intent": "REVIEWS",
    }


def _no_data(loc: Optional[str], aspect: Optional[str]) -> Dict:
    where  = f" in {loc}" if loc else ""
    what   = f" for {_pretty(aspect)}" if aspect else ""
    return {
        "answer": f"The dataset does not contain enough reviews{what}{where} "
                  f"to give a reliable answer. Try broadening the location "
                  f"or asking about a different aspect.",
        "cited_reviews": [],
        "intent": "UNKNOWN",
    }


def _unknown(retrieved: List[Dict]) -> Dict:
    return {
        "answer":
            "I can only answer questions about the 14 Malaysian hospitals "
            "in our dataset. Try asking:\n\n"
            "  • *Which KL hospital has the best doctors?*\n"
            "  • *Compare Sunway and Gleneagles*\n"
            "  • *What do people say about cleanliness at Hospital Melaka?*\n"
            "  • *Show me reviews about waiting time*",
        "cited_reviews": retrieved[:3],
        "intent": "UNKNOWN",
    }


# ─── PUBLIC ENTRY ───────────────────────────────────────────────────────────

def answer(question: str, retrieved: Optional[List[Dict]] = None,
           fallback_reason: str = "") -> Dict:
    """Main entry point. Returns dict with keys:
        answer, cited_reviews, intent, model_used, sources_count, fallback_reason
    """
    retrieved = retrieved or []
    intent = detect_intent(question)

    router = {
        "BEST":    answer_best,
        "WORST":   answer_worst,
        "COMPARE": answer_compare,
        "INFO":    answer_info,
        "STATS":   answer_stats,
        "REVIEWS": answer_reviews,
    }
    result = router.get(intent, lambda q, r: _unknown(r))(question, retrieved)

    return {
        "answer":          result["answer"],
        "cited_reviews":   result["cited_reviews"],
        "intent":          result.get("intent", intent),
        "model_used":      "pattern-based (fallback)",
        "sources_count":   len(retrieved),
        "fallback_reason": fallback_reason,
    }


if __name__ == "__main__":
    tests = [
        "Which hospital has the best doctors?",
        "Which KL hospital has best doctors?",
        "Which Melaka hospital has worst waiting time?",
        "Compare Sunway and Gleneagles",
        "What do people say about cleanliness at Hospital Melaka?",
        "How many reviews for Sunway?",
        "What's the weather?",
    ]
    for q in tests:
        print("=" * 70)
        print("Q:", q)
        r = answer(q, retrieved=[])
        print(r["answer"])
        print()
