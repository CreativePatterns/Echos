# main_emotion_pipeline.py

import os
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

from insight_generator import InsightGenerator  # <-- newly added module

# ================================
# SETTINGS
# ================================
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
TRAIN_SAMPLES = 10000
VAL_SAMPLES = 2000
NUM_EPOCHS = 3
BATCH_SIZE = 8

# ================================
# Load and Preprocess Dataset
# ================================
train_df = pd.read_csv("emotion_dataset/train.csv")
val_df = pd.read_csv("emotion_dataset/val.csv")

# Convert label strings to lists of ints
train_df["labels"] = train_df["emotion_multihot"].apply(lambda x: [int(i) for i in x.split(",")])
val_df["labels"] = val_df["emotion_multihot"].apply(lambda x: [int(i) for i in x.split(",")])

# Subsample
train_df = train_df.sample(n=TRAIN_SAMPLES, random_state=42)
val_df = val_df.sample(n=VAL_SAMPLES, random_state=42)

# Convert to HF datasets
train_dataset = Dataset.from_pandas(train_df[["text", "labels"]])
val_dataset = Dataset.from_pandas(val_df[["text", "labels"]])

# ================================
# Tokenization
# ================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

# Format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ================================
# Model
# ================================
NUM_LABELS = len(train_df["labels"].iloc[0])
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

# ================================
# Custom Data Collator
# ================================
class CustomCollator:
    def __init__(self, tokenizer):
        self.collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        batch = self.collator(features)
        batch["labels"] = batch["labels"].float()
        return batch

# ================================
# Metrics
# ================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    return {"f1_macro": f1_score(labels, preds, average="macro")}

# ================================
# Training Arguments
# ================================
training_args = TrainingArguments(
    output_dir="emotion_model",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_dir="logs",
    logging_steps=10,
    save_total_limit=1,
    fp16=False
)

# ================================
# Trainer
# ================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=CustomCollator(tokenizer),
    compute_metrics=compute_metrics,
)

# ================================
# Train and Evaluate
# ================================
trainer.train()
eval_results = trainer.evaluate()
trainer.save_model("emotion_model")
print(eval_results)

# ================================
# Use Insight Generator on Validation Text
# ================================
insight_gen = InsightGenerator(device=0 if torch.cuda.is_available() else -1)

print("\nGenerating insights for a few validation samples:")

sample_val = val_df.head(5)
for idx, row in sample_val.iterrows():
    text = row["text"]
    insights = insight_gen.analyze(text)
    print(f"\nText: {text}")
    print(f"Emotion (multi-hot): {row['labels']}")
    print("Reframe:", insights["reframe"])
    print("Tone Summary:", insights["tone"])
    print("Reflection Prompt:", insights["reflection_prompt"])
