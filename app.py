from pathlib import Path

import gradio as gr

from chatbot.intent_engine import IntentEngine
from chatbot.llm_engine import llm_reply
from chatbot.rules_engine import rules_reply


BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "data" / "intents.json"

intent_engine = IntentEngine(INTENTS_PATH)


def chat_fn(message, history):
    # 1) Deterministic rule layer
    rule = rules_reply(message)
    if rule:
        return f"{rule}\n\n_(route: rules)_"

    # 2) Intent matching layer
    match = intent_engine.match(message, threshold=0.70)
    if match["type"] == "intent":
        return f'{match["text"]}\n\n_(route: intent={match["tag"]}, similarity: {match["score"]:.2f})_'

    # 3) LLM fallback
    return f"{llm_reply(message)}\n\n_(route: LLM fallback)_"


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Logic-Augmented Chatbot",
    description="Hybrid routing: rules → semantic intent matching → LLM fallback",
    examples=[
        "Hello",
        "What are your working hours?",
        "How do I contact support?",
        "Where are you located?",
        "Explain what this chatbot does",
    ],
)


if __name__ == "__main__":
    demo.launch(inbrowser=True)