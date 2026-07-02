"""
Stage 3 (Knowledge Representation) - Stratified train / val / test split.

Reads:  Data/processed/healthcare_cleaned.csv
Writes: Data/splits/train.csv  val.csv  test.csv  (70 / 15 / 15, stratified)

Stratified splitting keeps each sentiment class proportionally represented
in every split - essential for fair training and evaluation.
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE     = os.path.join(os.path.dirname(__file__), "..", "..")
INPUT    = os.path.join(BASE, "Data", "processed", "healthcare_cleaned.csv")
OUT_DIR  = os.path.join(BASE, "Data", "splits")
SEED     = 42

VALID_LABELS = ("positive", "neutral", "negative")


def main():
    if not os.path.exists(INPUT):
        sys.exit(f"[!] Not found: {INPUT}  - run preprocess.py first.")

    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    print(f"[1] Loaded {len(df)} rows from healthcare_cleaned.csv")

    # drop rows with empty cleaned text or invalid label
    before = len(df)
    df = df[df["text_cleaned"].notna()]
    df["text_cleaned"] = df["text_cleaned"].astype(str).str.strip()
    df = df[df["text_cleaned"] != ""]
    df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)
    print(f"    {len(df)} usable rows (dropped {before - len(df)})")

    # 70 / 15 / 15  stratified by label
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=SEED
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False,
                 encoding="utf-8-sig")
    val.to_csv(os.path.join(OUT_DIR, "val.csv"),     index=False,
                 encoding="utf-8-sig")
    test.to_csv(os.path.join(OUT_DIR, "test.csv"),   index=False,
                 encoding="utf-8-sig")

    print(f"[2] Saved splits to {OUT_DIR}/")
    print(f"    train: {len(train):>4}  |  val: {len(val):>4}  "
          f"|  test: {len(test):>4}")

    print("\n[3] Class distribution per split:")
    for name, s in (("train", train), ("val", val), ("test", test)):
        dist = s["label"].value_counts().to_dict()
        line = "  ".join(f"{k}={dist.get(k, 0)}" for k in VALID_LABELS)
        print(f"    {name:<6} {line}")

    print("\nNext: vectorize.py (TF-IDF features)")


if __name__ == "__main__":
    main()
