import os
import json
import random
from pathlib import Path

import gradio as gr
from sentence_transformers import SentenceTransformer, util

# Optional OpenAI fallback
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

try:
    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI()
except Exception:
    USE_OPENAI = False

BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "data" / "intents.json"

# ---------- Load intents ----------
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

# ---------- Embedding model ----------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

intent_embeddings = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        intent_embeddings.append((intent, embedding_model.encode(pattern)))

def classify_intent(query: str, threshold: float = 0.70):
    query_emb = embedding_model.encode(query)
    best_intent = None
    best_score = 0.0

    for intent, pat_emb in intent_embeddings:
        score = util.pytorch_cos_sim(query_emb, pat_emb).item()
        if score > best_score:
            best_score = score
            best_intent = intent

    if best_intent and best_score >= threshold:
        return random.choice(best_intent["responses"]), best_score, best_intent["tag"]

    return None, best_score, None

def llm_fallback(query: str) -> str:
    if not USE_OPENAI:
        return (
            "I can answer common questions using intent matching, but the LLM fallback is disabled.\n\n"
            "To enable it, set an OPENAI_API_KEY environment variable."
        )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a chatbot demo."},
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content

def chat_fn(message, history):
    intent_resp, score, tag = classify_intent(message)

    if intent_resp:
        return f"{intent_resp}\n\n_(intent: {tag}, similarity: {score:.2f})_"

    answer = llm_fallback(message)
    return f"{answer}\n\n_(fallback: LLM)_"

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Hybrid AI Chatbot (Intent + LLM Fallback)",
    description="Uses semantic intent matching via SentenceTransformers. Falls back to an LLM when confidence is low.",
    examples=[
        "Hello",
        "What services do you offer?",
        "What are your working hours?",
        "How do I contact support?",
        "Explain what this chatbot does",
    ],
)

if __name__ == "__main__":
    demo.launch()