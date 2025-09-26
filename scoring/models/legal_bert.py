# file: train_legalbert_clause_regressor.py
import os
import json
from typing import Dict, List, Any

import numpy as np
import torch
from torch import nn
from datasets import load_dataset, Dataset, DatasetDict, load_metric
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    default_data_collator,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# Config / hyperparameters
# ---------------------------
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"   # Legal-BERT checkpoint
MAX_LEN = 384          # token length for clauses (increase if memory allows)
BATCH_SIZE = 8
EPOCHS = 4
LR = 2e-5
OUTPUT_DIR = "./legalbert_clause_regressor"
LABEL_KEYS = ["A", "M", "R", "T", "I"]   # order of outputs

# ---------------------------
# Model definition
# ---------------------------
class LegalBertRegressor(nn.Module):
    def __init__(self, model_name: str, n_outputs: int = 5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        # small head: pooled output -> FC -> ReLU -> Dropout -> outputs
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, n_outputs)
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # Use pooled_output (CLS) for clause-level regression
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:,0]
        preds = self.head(pooled)  # shape (batch, n_outputs)
        loss = None
        if labels is not None:
            loss_f = nn.MSELoss()
            loss = loss_f(preds.float(), labels.float())
        return {"loss": loss, "logits": preds}

# ---------------------------
# Data helpers
# ---------------------------
def load_json_clauses(path: str) -> Dataset:
    """
    Expect JSONL or JSON array where each example is:
    { "id": ..., "clause": "...", "A": 0.0, "M":0.0, ... } 
    Label fields may be missing (set to None/omit) if using heuristic labelling.
    """
    # detect JSONL vs array
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if text.startswith("["):
        arr = json.loads(text)
    else:
        # assume jsonl
        arr = [json.loads(l) for l in text.splitlines() if l.strip()]
    # ensure labels present (set to None if missing)
    for ex in arr:
        for k in LABEL_KEYS:
            if k not in ex:
                ex[k] = None
    return Dataset.from_list(arr)

def preprocess_function(examples, tokenizer):
    # tokenizes clause; returns input_ids, attention_mask and label array
    toks = tokenizer(examples["clause"], truncation=True, padding=False, max_length=MAX_LEN)
    # build label array (float), replace None with np.nan for trainer filtering
    labels = []
    for i in range(len(examples["clause"])):
        lab = []
        for k in LABEL_KEYS:
            v = examples.get(k)[i] if k in examples else None
            lab.append(None if v is None else float(v))
        labels.append(lab)
    toks["labels_raw"] = labels
    return toks

def prepare_dataset_for_training(ds: Dataset, tokenizer):
    # map tokenization
    # we need to filter examples that have labels for training; for Trainer, we'll convert labels to tensor with NaNs replaced by 0 and mask them in loss if needed.
    tokenized = ds.map(lambda ex: preprocess_function(ex, tokenizer), batched=True, remove_columns=ds.column_names)
    # Convert labels_raw to label arrays and store label_mask indicating which labels exist
    def attach_labels(batch):
        labs = batch.pop("labels_raw")
        label_array = []
        label_mask = []
        for l in labs:
            arr = []
            mask = []
            for v in l:
                if v is None:
                    arr.append(0.0)
                    mask.append(0)   # missing
                else:
                    arr.append(float(v))
                    mask.append(1)
            label_array.append(arr)
            label_mask.append(mask)
        batch["labels"] = label_array
        batch["label_mask"] = label_mask
        return batch
    tokenized = tokenized.map(attach_labels, batched=True)
    return tokenized

# ---------------------------
# Training metric
# ---------------------------
def compute_metrics(pred):
    preds = pred.predictions
    labels = pred.label_ids  # shape (N, 5)
    # labels may contain zeros for missing; but our dataset step should ensure we only include examples that have at least one label
    maes = []
    rmses = []
    for i in range(preds.shape[1]):
        # build mask: if label equals a special sentinel (we used 0 for missing), but 0 may be valid label.
        # Better approach: we avoided training on rows with missing labels; still compute metrics on all rows with non-nan in the dataset if possible.
        pred_col = preds[:, i]
        lab_col = labels[:, i]
        # compute metrics treating all entries as present (user should create eval sets with all labels)
        mae = mean_absolute_error(lab_col, pred_col)
        rmse = mean_squared_error(lab_col, pred_col, squared=False)
        maes.append(mae)
        rmses.append(rmse)
    metrics = {}
    for idx, k in enumerate(LABEL_KEYS):
        metrics[f"MAE_{k}"] = maes[idx]
        metrics[f"RMSE_{k}"] = rmses[idx]
    # overall
    metrics["MAE_mean"] = float(np.mean(maes))
    metrics["RMSE_mean"] = float(np.mean(rmses))
    return metrics

# ---------------------------
# Training entrypoint
# ---------------------------
def main(train_json: str, val_json: str = None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = LegalBertRegressor(MODEL_NAME, n_outputs=len(LABEL_KEYS))

    # load datasets
    train_ds = load_json_clauses(train_json)
    if val_json:
        val_ds = load_json_clauses(val_json)
        ds = DatasetDict({"train": train_ds, "validation": val_ds})
    else:
        # simple split
        ds = train_ds.train_test_split(test_size=0.1)
        ds = DatasetDict({"train": ds["train"], "validation": ds["test"]})

    train_tok = prepare_dataset_for_training(ds["train"], tokenizer)
    val_tok = prepare_dataset_for_training(ds["validation"], tokenizer)

    # data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="MAE_mean",
        greater_is_better=False,
        fp16=True,  # enable if GPU supports it
    )

    # Custom Trainer wrapper so that Trainer knows how to call our model (since model returns dict)
    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # labels is a list of lists -> convert to tensor
            labels = torch.tensor(inputs.pop("labels"), dtype=torch.float32).to(model.head[0].weight.device)
            outputs = model(**inputs, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            return (loss, outputs) if return_outputs else loss

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete. Model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="path to train json")
    p.add_argument("--val", default=None, help="path to val json (optional)")
    args = p.parse_args()
    main(args.train, args.val)
