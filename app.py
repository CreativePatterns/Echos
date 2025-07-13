import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from insight_generator import InsightGenerator

# Load emotion model
MODEL_PATH = "emotion_model/checkpoint-3750"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Emotion labels
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral"
]

# Insight Generator
insight_generator = InsightGenerator()

# Emotion prediction
def predict_emotions(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    predicted_indices = np.where(probs > threshold)[0]
    predicted_emotions = [EMOTION_LABELS[i] for i in predicted_indices]
    return predicted_emotions, probs

# Streamlit UI
st.set_page_config(page_title="Emotion + Insight Analyzer", layout="centered")
st.title("🧠 Emotion + Insight Analyzer")
st.write("Enter any text to analyze its emotional tone and receive a personalized reflection.")

user_input = st.text_area("📝 Your input text", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing emotions..."):
            emotions, raw_scores = predict_emotions(user_input)
            insights = insight_generator.analyze(user_input)

        st.subheader("📊 Detected Emotions")
        if emotions:
            for e in emotions:
                st.markdown(f"- {e}")
        else:
            st.write("No strong emotions detected.")

        st.subheader("🧠 Insight")
        st.markdown(f"**Tone:** {insights['tone'].capitalize()}")
        st.markdown(f"**Reframe:** _{insights['reframe']}_")
        st.markdown(f"**Reflection Prompt:** _{insights['reflection_prompt']}_")
