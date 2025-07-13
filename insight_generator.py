import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

class InsightGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Select device automatically
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model (DO NOT pass device here!)
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

        # Tone categories
        self.tones = {
            'positive': ["I feel hopeful.", "There is lightness in this text.", "A sense of peace emerges."],
            'negative': ["There is deep frustration here.", "It feels like pain is surfacing.", "A heavy burden is expressed."],
            'mixed': ["There’s an emotional conflict here.", "Contradictory feelings appear together.", "Hope and despair coexist."]
        }

        # Reframe examples
        self.reframes = {
            'negative': [
                "What if this struggle is actually a turning point?",
                "This might be hard now, but growth often begins in chaos.",
                "Could this pain be asking you to listen to a deeper need?"
            ],
            'mixed': [
                "There’s tension here—what might these emotions be trying to tell you?",
                "Perhaps both feelings are valid—can you hold them with compassion?",
                "Could this be a sign of inner transformation beginning?"
            ],
            'positive': [
                "Celebrate this strength—you’ve earned this peace.",
                "Hold on to this hope—it’s a guide forward.",
                "What you feel now is proof of healing."
            ]
        }

        # Reflection prompts
        self.reflections = [
            "What is this emotion trying to protect?",
            "What’s the story beneath the surface feeling?",
            "Is there something you haven’t yet said out loud?",
            "Where in your body do you feel this most?",
            "What would you say to a younger version of yourself feeling this way?"
        ]

    def detect_tone(self, text):
        # Encode text and tone examples to the correct device
        embeddings = self.model.encode([text] + sum(self.tones.values(), []), convert_to_tensor=True, device=self.device)
        text_emb = embeddings[0]
        tone_embs = embeddings[1:]

        tone_scores = []
        idx = 0
        for tone, examples in self.tones.items():
            sims = [util.cos_sim(text_emb, tone_embs[idx + i]).item() for i in range(len(examples))]
            tone_scores.append((tone, np.mean(sims)))
            idx += len(examples)

        tone_scores.sort(key=lambda x: x[1], reverse=True)
        top_tone = tone_scores[0][0]

        return top_tone

    def suggest_reframe(self, tone):
        return random.choice(self.reframes.get(tone, ["This moment matters."]))

    def suggest_reflection(self):
        return random.choice(self.reflections)

    def analyze(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                "tone": "unknown",
                "reframe": "It's okay to start with silence.",
                "reflection_prompt": "What do you need most right now?"
            }
        tone = self.detect_tone(text)
        return {
            "tone": tone,
            "reframe": self.suggest_reframe(tone),
            "reflection_prompt": self.suggest_reflection()
        }

# Global instance
_insight_generator = InsightGenerator()

def generate_insights(text):
    return _insight_generator.analyze(text)
