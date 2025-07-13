import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from insight_generator import generate_insights  # Make sure this file is in the same directory
import numpy as np

# ================================
# Load Emotion Classification Model
# ================================
MODEL_PATH = "emotion_model/checkpoint-3750"
BASE_MODEL = "distilbert-base-uncased"  # <- used during training
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# List your emotion labels in the correct order
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral"
]

# ================================
# Emotion Prediction Function
# ================================
def predict_emotions(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    predicted_indices = np.where(probs > threshold)[0]
    predicted_emotions = [EMOTION_LABELS[i] for i in predicted_indices]
    return predicted_emotions, probs

# ================================
# Main Function
# ================================
def run_pipeline(text):
    predicted_emotions, raw_scores = predict_emotions(text)
    print("\n📊 Detected Emotions:")
    for emotion in predicted_emotions:
        print(f"- {emotion}")

    print("\n🧠 Insight Generator:")
    insights = generate_insights(text)
    for key, value in insights.items():
        print(f"\n[{key.upper()}]\n{value}")


# ================================
# Interactive Loop
# ================================
if __name__ == "__main__":
    print("✨ Emotion Insight Generator ✨")
    print("Type your text below (or type 'exit' to quit):\n")
    while True:
        user_input = input("🔹 Your text: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        run_pipeline(user_input)
