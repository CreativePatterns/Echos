# 🧠 Emotion + Insight Analyzer

Welcome to **Echoes**, a personal and public-facing project that explores the emotional intelligence of text through machine learning. This web app classifies emotions in user-written text and then reflects back a meaningful insight — tone summary, a reframe, and a gentle prompt — the way a thoughtful friend or therapist might.

Built with 🧠 **Transformers**, 🤗 **Sentence Transformers**, and 🌐 **Streamlit**.

---

## ✨ What It Does

- ✅ **Multi-label Emotion Classification** using a fine-tuned DistilBERT model trained on 28 emotional categories.
- 💬 **Insight Generation**: Based on emotional tone, the app returns a:
  - Tone summary (positive, negative, mixed)
  - Reframe (helpful re-interpretation)
  - Reflection prompt (for journaling or introspection)
- GPU/CPU aware and fast!
- Simple and accessible **web interface** via Streamlit.

---

## 🌱 Why This Project?

I built this during a period of personal reflection — while diving deeper into AI, emotional literacy, and building things that **talk back meaningfully**.

It’s meant to be a personal healing tool, a portfolio piece, and a stepping stone toward emotionally intelligent software.

> _"What if your diary could understand what you were trying to say before even you did?"_

---

## 🛠️ Technologies Used

| Component           | Stack / Tool                          |
|--------------------|----------------------------------------|
| Emotion Classifier | `transformers`, `distilbert-base-uncased` |
| Insight Engine     | `sentence-transformers`, `MiniLM-L6-v2` |
| UI                 | `streamlit`                          |
| Training           |  Custom script with `Trainer` API     |
| Deployment         | GitHub + Streamlit       |

---

## 📦 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/emotion-insight-analyzer.git
   cd emotion-insight-analyzer

