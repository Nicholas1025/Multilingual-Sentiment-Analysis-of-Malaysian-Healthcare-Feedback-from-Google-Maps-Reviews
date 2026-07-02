"""
Tighten the needs_review flag without re-running the AI model.

Original rule (label_with_ai.py): flag every AI-Neutral + every row with
confidence < 0.70 -> ~670 flagged for a 1,400-row dataset, which is more
than the team needs to verify.

This script re-computes a SMARTER rule on the existing CSV:
    flag = ai_label == "neutral"              (always - we need neutrals)
          OR ai_confidence < strict_threshold (truly ambiguous only)
          OR ai_label disagrees with star label AND confidence < 0.85
                                              (cases the AI itself wobbled on)

Output: rewrites Data/raw/healthcare_ai_labeled.csv in place
"""

import os
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
FILE = os.path.join(BASE, "Data", "raw", "healthcare_ai_labeled.csv")

STRICT_LOW_CONF = 0.55      # really uncertain cases
DISAGREE_CONF   = 0.85      # AI vs star disagreement is interesting if conf < this


def main():
    if not os.path.exists(FILE):
        sys.exit(f"[!] Not found: {FILE}")

    df = pd.read_csv(FILE, encoding="utf-8-sig")

    is_neutral   = df["ai_label"] == "neutral"
    low_conf     = df["ai_confidence"] < STRICT_LOW_CONF
    if "label" in df.columns:
        star_disagree = (
            (df["ai_label"] != df["label"]) &
            (df["ai_confidence"] < DISAGREE_CONF)
        )
    else:
        star_disagree = False

    df["needs_review"] = is_neutral | low_conf | star_disagree

    df.to_csv(FILE, index=False, encoding="utf-8-sig")

    total = len(df)
    flagged = int(df["needs_review"].sum())
    print(f"[OK] Refined needs_review flag in {FILE}")
    print(f"     Total reviews : {total}")
    print(f"     Now flagged   : {flagged}  ({100*flagged/total:.1f}%)")
    print(f"\n  Breakdown:")
    print(f"    AI-Neutral             : {int(is_neutral.sum())}")
    print(f"    Low confidence (<{STRICT_LOW_CONF}) : {int(low_conf.sum())}")
    if "label" in df.columns:
        print(f"    Star-AI disagreement   : {int(star_disagree.sum())}")
    print(f"\nNext: run  label_content.py  to verify {flagged} reviews "
          f"(~{flagged // 4} per person if 4 people).")


if __name__ == "__main__":
    main()
