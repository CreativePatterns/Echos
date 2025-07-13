import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# --- Load Dataset ---
emotion_dataset = pd.read_csv('emotion_sentiment_dataset.csv')

# --- Get unique emotions and create a label-to-index mapping ---
emotion_list = sorted(emotion_dataset["Emotion"].unique())  # ensure consistent order
emotion_to_idx = {label: idx for idx, label in enumerate(emotion_list)}

# --- Config ---
NUM_EMOTION_LABELS = len(emotion_list)
SAVE_DIR = "emotion_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Convert to Multi-Hot Vectors for Single-Label ---
def labels_to_multihot(emotion_column):
    num_samples = len(emotion_column)
    num_labels = len(emotion_to_idx)
    multi_hot = np.zeros((num_samples, num_labels), dtype=int)

    for i, label in enumerate(emotion_column):
        idx = emotion_to_idx.get(label)
        if idx is not None:
            multi_hot[i, idx] = 1

    return multi_hot

# --- Apply Multi-Hot Encoding ---
emotion_dataset["emotion_multihot"] = labels_to_multihot(emotion_dataset["Emotion"]).tolist()

# --- Split into Train and Validation ---
train_df, val_df = train_test_split(emotion_dataset, test_size=0.1, random_state=42)

# --- Convert list to string for saving ---
def listify_and_stringify(df):
    df = df.copy()
    df["emotion_multihot"] = df["emotion_multihot"].apply(lambda x: ",".join(map(str, x)))
    return df

train_df = listify_and_stringify(train_df)
val_df = listify_and_stringify(val_df)

# --- Save to CSV ---
train_df[["text", "emotion_multihot"]].to_csv(f"{SAVE_DIR}/train.csv", index=False)
val_df[["text", "emotion_multihot"]].to_csv(f"{SAVE_DIR}/val.csv", index=False)

print(emotion_dataset.columns)
print(emotion_dataset.head())
print(emotion_dataset.info)
print(emotion_dataset.describe())
