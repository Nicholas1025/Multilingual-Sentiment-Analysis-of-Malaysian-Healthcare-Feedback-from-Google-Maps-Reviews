"""
COLAB VERSION — Fine-tune XLM-RoBERTa on Google Colab's free T4 GPU.

How to use:
  1. Upload Data/splits/{train,val,test}.csv to your Google Drive at
     /MyDrive/healthcare_sa/data/
  2. Open https://colab.research.google.com -> New Notebook
  3. Runtime -> Change runtime type -> GPU (T4)
  4. Paste this whole file into a single cell, run it.
  5. After it finishes (~5 min), download the transformer/ folder from
     /MyDrive/healthcare_sa/models/transformer/ and put it in your local
     PROJECT_CODE/models/transformer/.

Local FastAPI app auto-loads it on next startup. No code change needed.
"""

# ============================================================
#  STEP 1: install + mount Drive (run this cell first in Colab)
# ============================================================
# Run these lines IN Colab (not on your PC):
#
# !pip -q install transformers accelerate datasets
# from google.colab import drive
# drive.mount('/content/drive')


# ============================================================
#  STEP 2: training script (paste this in the next Colab cell)
# ============================================================

import os
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

# ── paths (adjust if you put your folder somewhere else) ────────────────
DATA_DIR  = "/content/drive/MyDrive/healthcare_sa/data"
MODEL_DIR = "/content/drive/MyDrive/healthcare_sa/models/transformer"
REPORT    = "/content/drive/MyDrive/healthcare_sa/models/transformer_results.txt"

MODEL_NAME = "xlm-roberta-base"
LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABELS   = list(LABEL2ID.keys())

SEED   = 42
MAX_LEN     = 128
BATCH_SIZE  = 16          # T4 handles 16 comfortably
EPOCHS      = 4
LR          = 2e-5

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
assert device == "cuda", "Switch Runtime -> Change runtime type -> GPU"


# ── dataset wrapper ────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.as_tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.as_tensor(self.labels[i])
        return item


def load_split(name, tokenizer):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv", encoding="utf-8-sig")
    # transformer sees raw text; preprocessing column is for the baseline only
    text_col = "text_original" if "text_original" in df.columns else "text"
    text = df[text_col].astype(str).tolist()
    labels = [LABEL2ID[l] for l in df["label"].tolist()]
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
    return ReviewDataset(enc, labels), labels


def metrics_fn(p):
    preds = p.predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1_macro": f1_score(p.label_ids, preds,
                             labels=list(LABEL2ID.values()),
                             average="macro", zero_division=0),
    }


# ── load model + data ──────────────────────────────────────────────────
print(f"Loading tokenizer + model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
)

print("Loading splits...")
train_ds, _      = load_split("train", tokenizer)
val_ds,   _      = load_split("val",   tokenizer)
test_ds,  y_test = load_split("test",  tokenizer)
print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")


# ── train ──────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="/content/checkpoints",
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
    logging_steps=10,
    seed=SEED,
    fp16=True,                  # T4 supports fp16
    report_to=[],
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=metrics_fn,
)

print("Fine-tuning XLM-RoBERTa...")
trainer.train()


# ── evaluate on test ───────────────────────────────────────────────────
print("\nEvaluating on test set...")
pred = trainer.predict(test_ds).predictions.argmax(axis=-1)
y_true = np.array(y_test)
acc = accuracy_score(y_true, pred)
f1m = f1_score(y_true, pred, average="macro", zero_division=0)
cls = classification_report(y_true, pred, target_names=LABELS, digits=3)
cm  = confusion_matrix(y_true, pred, labels=list(LABEL2ID.values()))

lines = ["=" * 60,
         " XLM-RoBERTa FINE-TUNED  -  test set", "=" * 60,
         f"\nDevice: {device}", f"Model: {MODEL_NAME}",
         f"Epochs={EPOCHS}  BatchSize={BATCH_SIZE}  LR={LR}",
         f"\nTest accuracy: {acc:.4f}",
         f"Test macro-F1: {f1m:.4f}",
         "\nClassification report:", cls,
         "Confusion matrix (rows=true, cols=pred):",
         "         " + "  ".join(f"{l[:3]:>6}" for l in LABELS)]
for lab, row in zip(LABELS, cm):
    lines.append(f"  {lab[:3]:<6} " + "  ".join(f"{v:>6}" for v in row))

print("\n".join(lines))


# ── save model to Drive ────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
with open(f"{MODEL_DIR}/label_map.json", "w") as f:
    json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nModel saved -> {MODEL_DIR}")
print(f"Report saved -> {REPORT}")
print("Download the transformer/ folder to your local PROJECT_CODE/models/")
