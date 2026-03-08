from pathlib import Path
import json

import gradio as gr

from chatbot.intent_engine import IntentEngine
from chatbot.llm_engine import llm_reply, USE_OPENAI
from chatbot.rules_engine import rules_reply

BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "data" / "intents.json"


def validate_intents_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Missing file: {path}\n\nFix: create data/intents.json (see README)."

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Invalid JSON in {path}.\n\nError: {e}"

    if "intents" not in data or not isinstance(data["intents"], list) or len(data["intents"]) == 0:
        return (
            False,
            f"{path} must contain a top-level key 'intents' with a non-empty list.\n\n"
            "Example:\n"
            '{ "intents": [ { "tag": "greeting", "patterns": ["hi"], "responses": ["hello"] } ] }'
        )

    return True, "OK"


# -------- Startup checks --------
ok, msg = validate_intents_file(INTENTS_PATH)

intent_engine = None
startup_banner = None

if ok:
    try:
        intent_engine = IntentEngine(INTENTS_PATH)
    except Exception as e:
        startup_banner = (
            "⚠️ Startup error: failed to load the embedding model or intent engine.\n\n"
            f"Error: {e}\n\n"
            "Fix:\n"
            "- Check internet connection (first run downloads model)\n"
            "- Recreate venv + reinstall requirements\n"
            "- Run: python app.py from repo root"
        )
else:
    startup_banner = f"⚠️ Startup error:\n\n{msg}"


def route_and_reply(message: str, threshold: float) -> str:
    if startup_banner:
        return startup_banner

    # 1) Rule layer (deterministic)
    rule = rules_reply(message)
    if rule:
        return f"✅ **{rule}**\n\n`route: RULES`"

    # 2) Intent layer (semantic)
    match = intent_engine.match(message, threshold=threshold)
    if match["type"] == "intent":
        return f"{match['text']}\n\n`route: INTENT • tag={match['tag']} • similarity={match['score']:.2f}`"

    # 3) LLM fallback
    answer = llm_reply(message)
    if not USE_OPENAI:
        answer += "\n\n💡 Tip: Add `OPENAI_API_KEY` in a local `.env` to enable LLM fallback."
    return f"{answer}\n\n`route: LLM FALLBACK`"


def chat_fn(message, history, threshold):
    """
    Gradio 'messages' format:
    history is a list of dicts like {"role": "user"/"assistant", "content": "..."}
    """
    if history is None:
        history = []

    reply = route_and_reply(message, threshold)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


HOW_IT_WORKS_MD = f"""
### How it works (Hybrid Routing)

This demo routes each message through three layers:

1. **Rules (deterministic)**  
   Fast, reliable answers for structured questions (hours, contact, location).

2. **Semantic Intent Matching**  
   Uses **SentenceTransformers** embeddings to match user input to intent patterns in `data/intents.json`.

3. **LLM Fallback**  
   For open-ended prompts, the system optionally calls an LLM.  
   **Status:** {"✅ Enabled" if USE_OPENAI else "⚠️ Disabled (no OPENAI_API_KEY)"}.

---

### Interview talk-track (30 seconds)

> “I built a hybrid chatbot that combines deterministic rule routing, semantic intent classification using embeddings, and an LLM fallback for open-ended queries. The routing logic is transparent, adjustable via a confidence threshold, and packaged as a reproducible app.”
""".strip()

EXAMPLES = [
    "Hello",
    "What are your working hours?",
    "How do I contact support?",
    "Where are you located?",
    "Explain what this chatbot does",
    "Give me a short summary of transformers in NLP",
]

DESCRIPTION = "Hybrid routing: rules → semantic intent matching → LLM fallback"
if not USE_OPENAI:
    DESCRIPTION += "  •  (LLM fallback disabled: no OPENAI_API_KEY)"


with gr.Blocks(
    title="Logic-Augmented Chatbot",
    css="""
    /* Make chat bubbles a bit narrower for a more polished look */
    .message { max-width: 78% !important; }
    """,
) as demo:
    # Header
    gr.Markdown(
        """
# Logic-Augmented Chatbot
**Hybrid routing:** rules → semantic intent matching → LLM fallback  
A polished demo app built for interview storytelling.
""".strip()
    )

    with gr.Row():
        # Left panel: how it works + controls
        with gr.Column(scale=4):
            gr.Markdown(HOW_IT_WORKS_MD)

            threshold = gr.Slider(
                minimum=0.50,
                maximum=0.90,
                value=0.70,
                step=0.01,
                label="Intent confidence threshold",
                info="Higher = fewer intent matches, more LLM fallback. Lower = more intent matches.",
            )

            gr.Markdown(
                f"**LLM status:** {'✅ enabled' if USE_OPENAI else '⚠️ disabled (no OPENAI_API_KEY)'}"
            )

            clear_btn = gr.Button("🧹 Clear chat", variant="secondary")

        # Right panel: chat UI
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                label="Chat",
                height=520,
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type a message…",
                    scale=10,
                )
                send = gr.Button("Send", variant="primary", scale=2)

            gr.Examples(
                examples=EXAMPLES,
                inputs=msg,
                label="Try these examples",
            )

    # Events
    send.click(chat_fn, inputs=[msg, chatbot, threshold], outputs=[chatbot, msg])
    msg.submit(chat_fn, inputs=[msg, chatbot, threshold], outputs=[chatbot, msg])
    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch(inbrowser=True)