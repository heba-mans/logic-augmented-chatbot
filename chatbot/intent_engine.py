import json
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


class IntentEngine:
    def __init__(self, intents_path: str | Path, model_name: str = "all-MiniLM-L6-v2"):
        self.intents_path = Path(intents_path)
        self.intents = self._load_intents()
        self.model = SentenceTransformer(model_name)
        self._pairs = self._build_pattern_embeddings()

    def _load_intents(self):
        with open(self.intents_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_pattern_embeddings(self):
        pairs = []
        for intent in self.intents["intents"]:
            for pattern in intent.get("patterns", []):
                pairs.append((intent, self.model.encode(pattern)))
        return pairs

    def match(self, query: str, threshold: float = 0.70):
        q_emb = self.model.encode(query)

        best_intent = None
        best_score = 0.0

        for intent, p_emb in self._pairs:
            score = util.pytorch_cos_sim(q_emb, p_emb).item()
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_intent and best_score >= threshold:
            return {
                "type": "intent",
                "tag": best_intent.get("tag"),
                "score": best_score,
                "text": random.choice(best_intent.get("responses", ["OK."])),
            }

        return {"type": "none", "tag": None, "score": best_score, "text": None}