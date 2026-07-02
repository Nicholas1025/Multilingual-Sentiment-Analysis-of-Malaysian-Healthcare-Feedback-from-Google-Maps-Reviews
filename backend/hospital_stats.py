"""
Precomputed per-hospital aggregate statistics.

Reads Data/raw/healthcare_raw.csv (2,170 reviews across 15 hospitals),
groups by hospital, and produces a rich structured summary used by:
  - /api/ranking       (leaderboard page)
  - chatbot.py         (grounded context for Gemini + pattern fallback)
  - /api/chat          (question answering)

Cached at module load - build cost is ~2 seconds.
Hospitals with < MIN_REVIEWS reviews are excluded (only UMMC, n=9).
"""

import os
import sys
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Source Codes", "5_advanced"))

import aspect_sentiment as asp   # noqa: E402

RAW_CSV     = os.path.join(ROOT, "Data", "raw", "healthcare_raw.csv")
MIN_REVIEWS = 10

# ─── LOCATION CLASSIFICATION ────────────────────────────────────────────────

# Hospitals we know are in the Klang Valley (KL / Selangor).
# Everything else in our dataset is in Melaka.
KL_HOSPITALS = {
    "Sunway Medical Centre",
    "Gleneagles Hospital Kuala Lumpur",
    "Columbia Asia Hospital Petaling Jaya",
    "Universiti Malaya Medical Centre",
}


def classify_location(hospital_name: str) -> str:
    if hospital_name in KL_HOSPITALS:
        return "KL"
    if "melaka" in hospital_name.lower() or "malacca" in hospital_name.lower():
        return "Melaka"
    return "Other"


# ─── AGGREGATION HELPERS ────────────────────────────────────────────────────

def _sentiment_dist(labels) -> Dict[str, float]:
    counter = Counter(labels)
    total = sum(counter.values())
    if total == 0:
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    return {k: round(counter.get(k, 0) / total, 3)
            for k in ["positive", "neutral", "negative"]}


def _pick_sample_reviews(group: pd.DataFrame, label: str,
                         k: int = 3) -> List[Dict]:
    """Pick up to k readable-length reviews with the given label."""
    subset = group[group["label"] == label].copy()
    if subset.empty:
        return []
    subset["_len"] = subset["text"].astype(str).str.len()
    # Prefer medium-length (40-300 chars) for readability
    filtered = subset[(subset["_len"] >= 40) & (subset["_len"] <= 300)]
    if not filtered.empty:
        subset = filtered
    subset = subset.sort_values("_len", ascending=False).head(k)
    return [
        {
            "text":  str(r["text"]),
            "date":  str(r.get("review_date", "")),
            "stars": int(r["stars"]) if pd.notna(r.get("stars")) else 0,
        }
        for _, r in subset.iterrows()
    ]


def _aspect_stats_for_group(group: pd.DataFrame) -> Dict[str, Dict]:
    """Detect aspects once per review, then aggregate."""
    per_row = []
    for _, row in group.iterrows():
        text = str(row.get("text", ""))
        per_row.append((row, text, asp.detect_aspects(text)))

    result: Dict[str, Dict] = {}
    for aspect in asp.ASPECTS.keys():
        labels: List[str] = []
        pos_examples: List[str] = []
        neg_examples: List[str] = []
        for row, text, aspects in per_row:
            if aspect not in aspects:
                continue
            lab = row.get("label", "neutral")
            labels.append(lab)
            if lab == "positive" and len(pos_examples) < 2 and 40 <= len(text) <= 250:
                pos_examples.append(text)
            elif lab == "negative" and len(neg_examples) < 2 and 40 <= len(text) <= 250:
                neg_examples.append(text)
        if not labels:
            continue
        result[aspect] = {
            "mentions":      len(labels),
            "positive_rate": round(labels.count("positive") / len(labels), 3),
            "negative_rate": round(labels.count("negative") / len(labels), 3),
            "neutral_rate":  round(labels.count("neutral")  / len(labels), 3),
            "sample_positive": pos_examples,
            "sample_negative": neg_examples,
        }
    return result


# ─── MAIN BUILD ─────────────────────────────────────────────────────────────

def build_hospital_stats() -> Dict:
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=["hospital", "text", "label"])
    stats: Dict[str, Dict] = {}
    for hospital, group in df.groupby("hospital"):
        if len(group) < MIN_REVIEWS:
            continue
        overall = _sentiment_dist(group["label"])
        stats[hospital] = {
            "hospital":          hospital,
            "location":          classify_location(hospital),
            "total_reviews":     int(len(group)),
            "overall_sentiment": overall,
            "positive_rate":     overall["positive"],
            "negative_rate":     overall["negative"],
            "aspects":           _aspect_stats_for_group(group),
            "sample_positive":   _pick_sample_reviews(group, "positive", k=3),
            "sample_negative":   _pick_sample_reviews(group, "negative", k=3),
        }
    return stats


# ─── MODULE-LEVEL CACHE ─────────────────────────────────────────────────────

STATS: Dict[str, Dict] = {}


def load() -> Dict:
    global STATS
    if not STATS:
        try:
            STATS = build_hospital_stats()
            print(f"[hospital_stats] loaded {len(STATS)} hospitals "
                  f"(min={MIN_REVIEWS} reviews)")
        except Exception as e:
            print(f"[hospital_stats] WARN: failed to load: {e}")
            STATS = {}
    return STATS


# ─── QUERY HELPERS (used by chatbot / ranking) ──────────────────────────────

def all_hospitals(location: Optional[str] = None) -> List[Dict]:
    """Return list of hospital stat dicts, optionally filtered by location."""
    load()
    out = list(STATS.values())
    if location:
        out = [h for h in out if h["location"] == location]
    return out


def get_hospital(name_or_alias: str) -> Optional[Dict]:
    """Find a hospital by exact name or lowercase substring alias."""
    load()
    if name_or_alias in STATS:
        return STATS[name_or_alias]
    q = name_or_alias.lower()
    for name, s in STATS.items():
        if q in name.lower():
            return s
    return None


def rank_by(aspect: Optional[str] = None,
            location: Optional[str] = None,
            metric: str = "positive_rate",
            reverse: bool = True) -> List[Dict]:
    """Return hospitals ranked by overall positive_rate or by an aspect's rate.

    metric: 'positive_rate' or 'negative_rate'
    reverse=True means best first; reverse=False means worst first
    """
    load()
    hospitals = all_hospitals(location=location)
    out = []
    for h in hospitals:
        if aspect and aspect in h["aspects"]:
            score = h["aspects"][aspect].get(metric, 0.0)
            mentions = h["aspects"][aspect]["mentions"]
        elif aspect:
            # no mentions of this aspect at all
            continue
        else:
            score    = h.get(metric, 0.0)
            mentions = h["total_reviews"]
        out.append({
            **h,
            "_score":        score,
            "_mentions":     mentions,
        })
    out.sort(key=lambda x: x["_score"], reverse=reverse)
    return out


# Auto-load at import
load()


if __name__ == "__main__":
    import json
    stats = load()
    print(f"Loaded {len(stats)} hospitals:")
    for name, s in stats.items():
        print(f"  {name:50s}  n={s['total_reviews']:3d}  "
              f"pos={s['positive_rate']:.1%}  loc={s['location']}")
    print()
    print("Sample (first hospital):")
    first = next(iter(stats.values()))
    print(json.dumps({
        "hospital":       first["hospital"],
        "total_reviews":  first["total_reviews"],
        "aspects_keys":   list(first["aspects"].keys()),
        "sample_positive": first["sample_positive"][:1],
    }, indent=2, ensure_ascii=False)[:800])
