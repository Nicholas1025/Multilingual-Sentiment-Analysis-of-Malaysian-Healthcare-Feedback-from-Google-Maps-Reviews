"""
Stage 1+ : Build the final balanced 120 / 120 / 120 training dataset.

Reads (in priority order):
    1.  Data/raw/healthcare_verified.csv   (after label_content.py)
    2.  Data/raw/healthcare_ai_labeled.csv (AI labels only, fallback)
    3.  Data/raw/healthcare_raw.csv        (star labels only, last resort)

Uses the highest-quality label column available:
    final_label  >  human_label  >  ai_label  >  label (star-based)

Output: Data/raw/healthcare_final.csv  with a `verified` column so the
report can honestly say how many rows are human-verified vs AI-proposed
vs star-derived.
"""

import os
import sys
from collections import Counter

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE       = os.path.join(os.path.dirname(__file__), "..", "..")
CANDIDATES = [
    os.path.join(BASE, "Data", "raw", "healthcare_verified.csv"),
    os.path.join(BASE, "Data", "raw", "healthcare_ai_labeled.csv"),
    os.path.join(BASE, "Data", "raw", "healthcare_raw.csv"),
]
OUT        = os.path.join(BASE, "Data", "raw", "healthcare_final.csv")

PER_CLASS = 189
SEED      = 42


def pick_input() -> tuple[str, pd.DataFrame]:
    for p in CANDIDATES:
        if os.path.exists(p):
            df = pd.read_csv(p, encoding="utf-8-sig")
            print(f"[1] Source: {os.path.basename(p)}  ({len(df)} rows)")
            return p, df
    sys.exit("[!] No labelled CSV found. Run label_with_ai.py first.")


def resolve_label(row) -> str:
    """Pick the best label available on this row."""
    for col in ("final_label", "human_label", "ai_label", "label"):
        if col in row.index:
            v = row.get(col)
            if isinstance(v, str) and v in {"positive", "neutral", "negative"}:
                return v
    return ""


def main():
    _, df = pick_input()

    df["chosen_label"] = df.apply(resolve_label, axis=1)
    df = df[df["chosen_label"].isin(["positive", "neutral", "negative"])].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 10].drop_duplicates(subset="text")

    # carry verified flag (default False if missing)
    if "verified" not in df.columns:
        df["verified"] = False
    df["verified"] = df["verified"].fillna(False).astype(bool)

    # ── Language detection (English preferred; non-English allowed only for neutral) ──
    print("\n[2] Detecting language per row...")
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(BASE, "Source Codes", "2_preprocessing"))
    from preprocess import detect_language as _detect
    df["lang"] = df["text"].astype(str).map(_detect)

    print("\n[3] Pool by chosen label (English vs non-English split):")
    pool_en  = {}
    pool_non = {}
    for lab in ("positive", "neutral", "negative"):
        sub = df[df["chosen_label"] == lab]
        pool_en[lab]  = sub[sub["lang"] == "english"]
        pool_non[lab] = sub[sub["lang"] != "english"]
        print(f"    {lab:<9} english={len(pool_en[lab]):>4}  "
              f"non-english={len(pool_non[lab]):>4}")

    # rubric check — positive/negative MUST come from English; neutral may borrow
    shortages = []
    if len(pool_en["positive"]) < PER_CLASS:
        shortages.append(f"positive: only {len(pool_en['positive'])} English (need {PER_CLASS})")
    if len(pool_en["negative"]) < PER_CLASS:
        shortages.append(f"negative: only {len(pool_en['negative'])} English (need {PER_CLASS})")
    neu_available = len(pool_en["neutral"]) + len(pool_non["neutral"])
    if neu_available < PER_CLASS:
        shortages.append(f"neutral: only {neu_available} total (need {PER_CLASS})")

    if shortages:
        print(f"\n[!] Not enough rows for {PER_CLASS} per class:")
        for s in shortages:
            print(f"    {s}")
        return

    print(f"\n[4] Sampling {PER_CLASS} per class:")
    parts = []
    # positive: English only
    pos = pool_en["positive"].sort_values("verified", ascending=False).head(PER_CLASS)
    parts.append(pos)
    print(f"    positive  {len(pos)}  (english {len(pos)})")

    # negative: English only
    neg = pool_en["negative"].sort_values("verified", ascending=False).head(PER_CLASS)
    parts.append(neg)
    print(f"    negative  {len(neg)}  (english {len(neg)})")

    # neutral: English first, fill remainder with non-English
    neu_en  = pool_en["neutral"].sort_values("verified", ascending=False)
    neu_non = pool_non["neutral"].sort_values("verified", ascending=False)
    take_en  = neu_en.head(PER_CLASS)
    need_non = PER_CLASS - len(take_en)
    take_non = neu_non.head(max(0, need_non))
    neu = pd.concat([take_en, take_non])
    parts.append(neu)
    from collections import Counter
    print(f"    neutral   {len(neu)}  "
          f"(english {len(take_en)}, non-english {len(take_non)} "
          f"-> {dict(Counter(neu['lang']))})")

    final = pd.concat(parts, ignore_index=True)
    final = final.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # use 'label' as the standard column name downstream
    final["label"] = final["chosen_label"]

    cols = [c for c in [
        "source", "hospital", "text", "stars",
        "label", "verified",
        "ai_label", "ai_confidence",
        "human_label", "review_date",
    ] if c in final.columns]
    final = final[cols]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    final.to_csv(OUT, index=False, encoding="utf-8-sig")

    n_ver = int(final["verified"].sum())
    print(f"\n[4] Saved -> {OUT}")
    print(f"    {len(final)} reviews  ({PER_CLASS} per class)")
    print(f"    human-verified: {n_ver}   AI/star-derived: {len(final) - n_ver}")
    print("\nNext: re-run preprocess.py with the new healthcare_final.csv.")


if __name__ == "__main__":
    main()
