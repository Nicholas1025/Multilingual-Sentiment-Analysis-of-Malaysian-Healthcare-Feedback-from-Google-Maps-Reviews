"""
Interactive tool to manually add reviews to healthcare_raw.csv.

Use this to plug language gaps that the scraper missed (especially
Malay positive and Chinese reviews). Auto-detects language, prevents
duplicates, and shows running stats so you can target the right gaps.

Each new row is appended to Data/raw/healthcare_raw.csv with
    source = "Manual"
so it's distinguishable from scraped data.

Run again any time - it picks up where you left off.

Run the rest of the pipeline AFTER you finish adding:
    build_final_dataset.py -> preprocess.py -> split_data.py -> vectorize.py
"""

import os
import re
import sys
import csv
from collections import Counter
from datetime import datetime

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
RAW  = os.path.join(BASE, "Data", "raw", "healthcare_raw.csv")

sys.path.insert(0, os.path.join(BASE, "Source Codes", "2_preprocessing"))
from preprocess import detect_language          # noqa: E402

FIELDS = ["source", "hospital", "text", "stars", "label",
          "date_scraped", "review_date"]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def stars_to_label(s: int) -> str:
    if s >= 4: return "positive"
    if s == 3: return "neutral"
    return "negative"


def load_existing() -> pd.DataFrame:
    if os.path.exists(RAW):
        return pd.read_csv(RAW, encoding="utf-8-sig")
    return pd.DataFrame(columns=FIELDS)


def show_stats_and_gaps(df: pd.DataFrame):
    print("=" * 64)
    print(f"  Current dataset: {len(df)} reviews")
    print("=" * 64)

    if len(df) == 0:
        print("  (empty — anything you add is welcome!)")
        return

    df = df.copy()
    df["lang"] = df["text"].astype(str).map(detect_language)
    pivot = pd.crosstab(df["lang"], df["label"]).reindex(
        columns=["positive", "neutral", "negative"], fill_value=0
    )
    print("\n  Language x label counts:")
    print(pivot.to_string().replace("\n", "\n  "))

    print("\n  Suggested gaps to fill (rough heuristic):")
    suggestions = []
    if pivot.get("positive", pd.Series()).get("malay", 0) < 80:
        suggestions.append("• Malay positive reviews")
    if pivot.get("positive", pd.Series()).get("chinese", 0) < 30:
        suggestions.append("• Chinese positive reviews")
    if pivot.get("neutral", pd.Series()).get("chinese", 0) < 30:
        suggestions.append("• Chinese neutral reviews")
    if pivot.get("neutral", pd.Series()).get("malay", 0) < 80:
        suggestions.append("• Malay neutral reviews")
    if pivot.get("positive", pd.Series()).get("mixed", 0) < 40:
        suggestions.append("• Manglish/Rojak positive reviews")
    if not suggestions:
        suggestions.append("• Anything in any language — the data is balanced!")
    for s in suggestions:
        print(f"    {s}")


def get_multiline(prompt: str) -> str:
    print(prompt)
    print("  (press Enter on an empty line to finish)")
    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if not line.strip() and lines:
            break
        if line.strip():
            lines.append(line)
    return " ".join(lines).strip()


def main():
    df = load_existing()
    show_stats_and_gaps(df)

    # build dedup set
    seen = set(df["text"].astype(str).map(normalize))
    added = 0
    today = datetime.now().strftime("%Y-%m-%d")

    print("\n" + "=" * 64)
    print("  ADD NEW REVIEWS  (type 'q' at hospital prompt to quit)")
    print("=" * 64)

    while True:
        print()
        hospital = input("Hospital name (or 'q' to quit): ").strip()
        if hospital.lower() == "q":
            break
        if not hospital:
            print("  [!] hospital required")
            continue

        stars_str = input("Star rating 1-5: ").strip()
        try:
            stars = int(stars_str)
            assert 1 <= stars <= 5
        except (ValueError, AssertionError):
            print("  [!] invalid star rating - must be 1, 2, 3, 4, or 5")
            continue

        text = get_multiline("Review text:")
        if len(text) < 10:
            print("  [!] text too short (<10 chars) - not added")
            continue

        key = normalize(text)
        if key in seen:
            print("  [!] duplicate (same text already exists) - not added")
            continue
        seen.add(key)

        lang  = detect_language(text)
        label = stars_to_label(stars)

        row = {
            "source": "Manual",
            "hospital": hospital,
            "text": text,
            "stars": stars,
            "label": label,
            "date_scraped": today,
            "review_date": today,
        }

        write_header = not os.path.exists(RAW)
        with open(RAW, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

        added += 1
        print(f"  [OK] added.  Language: {lang}   Label: {label}   "
              f"Total: {len(df) + added}")

    print("\n" + "=" * 64)
    if added == 0:
        print("  No new reviews added.")
    else:
        print(f"  Added {added} new reviews.  Total now: {len(df) + added}")
        print()
        print("  Next: re-run the AI labelling + balanced sampling so the new")
        print("  reviews enter the training set:")
        print()
        print('     py -3.13 "Source Codes\\1_scraper\\label_with_ai.py"')
        print('     py -3.13 "Source Codes\\1_scraper\\refine_flags.py"')
        print('     py -3.13 "Source Codes\\1_scraper\\build_final_dataset.py"')
        print('     py -3.13 "Source Codes\\2_preprocessing\\preprocess.py"')
        print('     py -3.13 "Source Codes\\3_features\\split_data.py"')
        print('     py -3.13 "Source Codes\\3_features\\vectorize.py"')
    print("=" * 64)


if __name__ == "__main__":
    main()
