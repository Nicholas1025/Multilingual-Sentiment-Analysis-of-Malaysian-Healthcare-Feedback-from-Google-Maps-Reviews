"""
Stage 4 - Baseline ML training on the TF-IDF features built in Stage 3.

Trains three classic classifiers and selects the best by macro-F1 on the
validation set:

    Logistic Regression     class_weight='balanced'
    Linear SVM              class_weight='balanced'
    Multinomial Naive Bayes  (no class weighting)

Outputs:
    models/best_baseline.pkl              best classifier (pickle)
    models/baseline_results.txt           full evaluation report
    models/baseline_confusion_<model>.txt confusion matrices

Run AFTER vectorize.py.
"""

import os
import sys
import pickle
import warnings
from io import StringIO

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
MDL  = os.path.join(BASE, "models")

LABELS = ["positive", "neutral", "negative"]


def load_features():
    X_train = sp.load_npz(os.path.join(MDL, "tfidf_train.npz"))
    X_val   = sp.load_npz(os.path.join(MDL, "tfidf_val.npz"))
    X_test  = sp.load_npz(os.path.join(MDL, "tfidf_test.npz"))
    with open(os.path.join(MDL, "labels_train.pkl"), "rb") as f:
        y_train = np.array(pickle.load(f))
    with open(os.path.join(MDL, "labels_val.pkl"), "rb") as f:
        y_val = np.array(pickle.load(f))
    with open(os.path.join(MDL, "labels_test.pkl"), "rb") as f:
        y_test = np.array(pickle.load(f))
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(name, model, X, y, report_lines):
    pred = model.predict(X)
    acc  = accuracy_score(y, pred)
    f1m  = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
    cls  = classification_report(y, pred, labels=LABELS, zero_division=0,
                                  digits=3)
    cm   = confusion_matrix(y, pred, labels=LABELS)

    report_lines.append(f"\n--- {name} ---")
    report_lines.append(f"accuracy: {acc:.4f}   macro-F1: {f1m:.4f}")
    report_lines.append(cls)
    report_lines.append("confusion matrix (rows=true, cols=pred):")
    header = "         " + "  ".join(f"{l[:3]:>6}" for l in LABELS)
    report_lines.append(header)
    for lab, row in zip(LABELS, cm):
        report_lines.append(f"  {lab[:3]:<6} " + "  ".join(f"{v:>6}" for v in row))
    return acc, f1m


def main():
    print("[1] Loading features...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_features()
    print(f"    train: {X_train.shape}   val: {X_val.shape}   test: {X_test.shape}")
    print(f"    train class dist: "
          f"{dict(zip(*np.unique(y_train, return_counts=True)))}")

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "LinearSVM":          LinearSVC(class_weight="balanced", random_state=42),
        "MultinomialNB":      MultinomialNB(),
    }

    report = ["=" * 64, " BASELINE ML EVALUATION", "=" * 64,
              f"\nValidation set: {len(y_val)} reviews",
              f"Test set:       {len(y_test)} reviews"]

    val_scores = {}
    trained = {}
    for name, clf in candidates.items():
        print(f"[2] Training {name}...")
        clf.fit(X_train, y_train)
        trained[name] = clf
        report.append(f"\n\n###  {name}  (validation)")
        acc, f1m = evaluate(name, clf, X_val, y_val, report)
        val_scores[name] = f1m
        print(f"    {name:<22}  acc={acc:.3f}  macroF1={f1m:.3f}")

    # pick best by macro-F1 on val, then evaluate on test
    best_name = max(val_scores, key=val_scores.get)
    best = trained[best_name]
    print(f"\n[3] Best on validation: {best_name}  (F1={val_scores[best_name]:.3f})")
    report.append(f"\n\n=== BEST MODEL: {best_name} ===")
    report.append("\n###  Best model on TEST set")
    test_acc, test_f1 = evaluate(best_name, best, X_test, y_test, report)
    print(f"    Test accuracy: {test_acc:.4f}   macro-F1: {test_f1:.4f}")

    # save
    with open(os.path.join(MDL, "best_baseline.pkl"), "wb") as f:
        pickle.dump({"name": best_name, "model": best}, f)
    out_txt = os.path.join(MDL, "baseline_results.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"[4] Saved best model -> models/best_baseline.pkl")
    print(f"    Saved report      -> {out_txt}")

    print("\nNext: train_transformer.py (XLM-RoBERTa fine-tuning)")


if __name__ == "__main__":
    main()
