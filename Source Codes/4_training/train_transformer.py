"""
Stage 4 / Advanced Feature - Fine-tune XLM-RoBERTa-base.

A multilingual transformer is the right choice for our domain because the
dataset is Malay + English + Chinese + Rojak code-switching. XLM-RoBERTa
was pretrained on 100 languages including all of those, so it handles them
in a single model - directly addressing R16's English-only lexicon limit.

Inputs  : Data/splits/{train,val,test}.csv  (uses text_original, not cleaned)
Outputs : models/transformer/                fine-tuned model + tokenizer
          models/transformer_results.txt     evaluation report

Run AFTER split_data.py.
Tip: GPU strongly recommended. On CPU expect ~30-60 min per epoch.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    set_seed,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE        = os.path.join(os.path.dirname(__file__), "..", "..")
SPL         = os.path.join(BASE, "Data", "splits")
MDL_OUT     = os.path.join(BASE, "models", "transformer")
REPORT_OUT  = os.path.join(BASE, "models", "transformer_results.txt")
MODEL_NAME  = "xlm-roberta-base"

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABELS   = list(LABEL2ID.keys())

SEED   = 42
set_seed(SEED)

# ── hyperparameters ─────────────────────────────────────────────────────
MAX_LEN     = 128
BATCH_SIZE  = 8
EPOCHS      = 3
LR          = 2e-5


# ── dataset wrapper ─────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.as_tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.as_tensor(self.labels[idx])
        return item


def load_split(name, tokenizer):
    df = pd.read_csv(os.path.join(SPL, f"{name}.csv"), encoding="utf-8-sig")
    # transformer should see the raw text - tokenizer handles cleaning
    text = df["text_original"].astype(str).tolist()
    labels = [LABEL2ID[l] for l in df["label"].tolist()]
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
    return ReviewDataset(enc, labels), labels


def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1_macro": f1_score(p.label_ids, preds,
                             labels=list(LABEL2ID.values()),
                             average="macro", zero_division=0),
    }


# ── main ────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[0] Device: {device}")
    if device == "cpu":
        print("    [!] Running on CPU - this will be slow. "
              "Reinstall torch with CUDA for ~20x speed-up.")

    print(f"[1] Loading tokenizer / model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("[2] Loading data splits")
    train_ds, _      = load_split("train", tokenizer)
    val_ds, _        = load_split("val",   tokenizer)
    test_ds, y_test  = load_split("test",  tokenizer)
    print(f"    train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    args = TrainingArguments(
        output_dir=MDL_OUT,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        seed=SEED,
        report_to=[],
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("[3] Fine-tuning XLM-RoBERTa...")
    trainer.train()

    print("[4] Evaluating on test set")
    test_pred = trainer.predict(test_ds)
    preds = test_pred.predictions.argmax(axis=-1)
    y_true = np.array(y_test)

    acc = accuracy_score(y_true, preds)
    f1m = f1_score(y_true, preds, average="macro", zero_division=0)
    cls = classification_report(
        y_true, preds, target_names=LABELS, zero_division=0, digits=3
    )
    cm = confusion_matrix(y_true, preds, labels=list(LABEL2ID.values()))

    lines = ["=" * 64,
             " XLM-RoBERTa FINE-TUNED  - test set",
             "=" * 64,
             f"\nDevice: {device}",
             f"Model : {MODEL_NAME}",
             f"Epochs: {EPOCHS}  BatchSize: {BATCH_SIZE}  LR: {LR}",
             f"\nTest accuracy : {acc:.4f}",
             f"Test macro-F1 : {f1m:.4f}",
             "\nClassification report:",
             cls,
             "Confusion matrix (rows=true, cols=pred):",
             "         " + "  ".join(f"{l[:3]:>6}" for l in LABELS)]
    for lab, row in zip(LABELS, cm):
        lines.append(f"  {lab[:3]:<6} " + "  ".join(f"{v:>6}" for v in row))

    os.makedirs(os.path.dirname(REPORT_OUT), exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # save the best fine-tuned model + tokenizer for the FastAPI app
    trainer.save_model(MDL_OUT)
    tokenizer.save_pretrained(MDL_OUT)
    with open(os.path.join(MDL_OUT, "label_map.json"), "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)

    print(f"\nTest accuracy: {acc:.4f}   macro-F1: {f1m:.4f}")
    print(f"Model saved   -> {MDL_OUT}")
    print(f"Report saved  -> {REPORT_OUT}")
    print("\nNext: Stage 5 (Aspect-based SA)  +  Stage 6 (FastAPI deployment)")


if __name__ == "__main__":
    main()
