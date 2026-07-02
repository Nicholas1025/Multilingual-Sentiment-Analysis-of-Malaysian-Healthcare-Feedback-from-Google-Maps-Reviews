"""
Stage 3 (Knowledge Representation) - TF-IDF vectorisation.

Reads:  Data/splits/train.csv  val.csv  test.csv
Writes: models/tfidf_vectorizer.pkl
        models/tfidf_train.npz  tfidf_val.npz  tfidf_test.npz
        models/tfidf_labels_train.pkl  val.pkl  test.pkl

TF-IDF with unigrams + bigrams, sublinear TF, and a min/max document
frequency filter to drop both noise and over-common terms.
The vectorizer is *fit on training data only* (no peeking) and then applied
to val / test - the standard, leakage-free protocol.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
SPL  = os.path.join(BASE, "Data", "splits")
MDL  = os.path.join(BASE, "models")


def main():
    for f in ("train.csv", "val.csv", "test.csv"):
        if not os.path.exists(os.path.join(SPL, f)):
            sys.exit(f"[!] Missing {f}  - run split_data.py first.")

    train = pd.read_csv(os.path.join(SPL, "train.csv"), encoding="utf-8-sig")
    val   = pd.read_csv(os.path.join(SPL, "val.csv"),   encoding="utf-8-sig")
    test  = pd.read_csv(os.path.join(SPL, "test.csv"),  encoding="utf-8-sig")

    print(f"[1] Loaded train={len(train)}, val={len(val)}, test={len(test)}")

    train_text = train["text_cleaned"].fillna("")
    val_text   = val["text_cleaned"].fillna("")
    test_text  = test["text_cleaned"].fillna("")

    vec = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=2,                # drop tokens that appear in only 1 doc
        max_df=0.95,             # drop tokens that appear in >95% of docs
        sublinear_tf=True,       # log-scaled term frequency
    )

    print("[2] Fitting TF-IDF on training set...")
    X_train = vec.fit_transform(train_text)
    X_val   = vec.transform(val_text)
    X_test  = vec.transform(test_text)

    os.makedirs(MDL, exist_ok=True)
    with open(os.path.join(MDL, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    sp.save_npz(os.path.join(MDL, "tfidf_train.npz"), X_train)
    sp.save_npz(os.path.join(MDL, "tfidf_val.npz"),   X_val)
    sp.save_npz(os.path.join(MDL, "tfidf_test.npz"),  X_test)

    # also save labels alongside, so model code can load both halves cleanly
    for name, df in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(MDL, f"labels_{name}.pkl"), "wb") as f:
            pickle.dump(df["label"].tolist(), f)

    print(f"[3] Saved -> {MDL}/")
    nnz = X_train.nnz
    n, m = X_train.shape
    density = 100 * nnz / (n * m) if n and m else 0
    print(f"    vocab size: {len(vec.vocabulary_)}")
    print(f"    train shape: {X_train.shape}  "
          f"({nnz} nonzero, density {density:.3f}%)")
    print(f"    val shape:   {X_val.shape}")
    print(f"    test shape:  {X_test.shape}")

    print("\nNext: Stage 4 - model training (baseline ML + Transformer)")


if __name__ == "__main__":
    main()
