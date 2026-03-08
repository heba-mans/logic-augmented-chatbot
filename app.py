from pathlib import Path
import json
import re

import gradio as gr

from chatbot.intent_engine import IntentEngine
from chatbot.llm_engine import llm_reply, USE_OPENAI
from chatbot.rules_engine import rules_reply
from chatbot.router import route_and_reply

BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "data" / "intents.json"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant for a chatbot demo. "
    "Be concise, friendly, and practical. If the user asks about the project, "
    "explain that it uses rules, semantic intent matching, and an LLM fallback."
)

# CSS must be passed to launch() in Gradio 6+
APP_CSS = """
/* Make bubbles look normal */
.message { max-width: 72% !important; }

/* User bubble (right) */
.message-row.user-row .message {
  margin-left: auto !important;
  max-width: 48% !important;
  min-width: 260px !important;
  padding: 12px 14px !important;
  border-radius: 16px !important;
}

/* Bot bubble (left) - wider so it doesn't wrap every word */
.message-row.bot-row .message {
  margin-right: auto !important;
  max-width: 62% !important;
  min-width: 340px !important;
  padding: 12px 14px !important;
  border-radius: 16px !important;
}

/* Text behavior */
.message-content {
  white-space: normal !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
  line-height: 1.35 !important;
  font-size: 15px !important;
}

/* Hide the weird Gradio action overlays (from your HTML) */
div[class*="message-buttons-right"],
div[class*="message-buttons-left"],
div[class*="message-buttons"][class*="bubble"],
div[class*="icon-button-wrapper"][class*="top-panel"] {
  display: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}
"""


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


def _details_line(text: str, show_debug: bool) -> str:
    # No backticks — avoids Gradio rendering as broken code blocks
    return f"\n\n_( {text} )_" if show_debug else ""

def chat_fn(message, history, threshold, show_debug, system_prompt, memory):
    if history is None:
        history = []

    reply, memory = route_and_reply(
        message,
        intent_engine=intent_engine,
        threshold=threshold,
        show_debug=show_debug,
        system_prompt=system_prompt,
        memory=memory,
        startup_banner=startup_banner,
    )

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, "", memory


def reset_session(reset_prompt: bool):
    """Clears the conversation + clears memory. Optionally resets prompt to default."""
    if reset_prompt:
        return [], "", DEFAULT_SYSTEM_PROMPT, {"name": None}
    return [], "", gr.update(), {"name": None}  # keep current prompt, reset memory


HOW_IT_WORKS_MD = f"""
### How it works (Hybrid Routing)

This demo routes each message through three layers:

1. Rules (deterministic)  
   Fast, reliable answers for structured questions (hours, contact, location).

2. Semantic Intent Matching  
   Uses SentenceTransformers embeddings to match user input to intent patterns in data/intents.json.

3. LLM Fallback  
   For open-ended prompts, the system optionally calls an LLM.  
   Status: {"✅ Enabled" if USE_OPENAI else "⚠️ Disabled (no OPENAI_API_KEY)"}.

---

### Interview talk-track (30 seconds)

> “I built a hybrid chatbot that combines deterministic rule routing, semantic intent classification using embeddings, and an LLM fallback for open-ended queries. The routing logic is transparent, adjustable via a confidence threshold, and packaged as a reproducible app.”
""".strip()

EXAMPLES = [
    "My name is Hiba",
    "Hello",
    "What are your working hours?",
    "How do I contact support?",
    "Where are you located?",
    "Explain what this chatbot does",
    "Give me a short summary of transformers in NLP",
    "Are you open now?",
    "Are you open today?",
]

DESCRIPTION = "Hybrid routing: rules → semantic intent matching → LLM fallback"
if not USE_OPENAI:
    DESCRIPTION += "  •  (LLM fallback disabled: no OPENAI_API_KEY)"


with gr.Blocks(title="Logic-Augmented Chatbot") as demo:
    gr.Markdown(
        """
# Logic-Augmented Chatbot
Hybrid routing: rules → semantic intent matching → LLM fallback  
A polished demo app built for interview storytelling.
""".strip()
    )

    # Session memory (per browser session)
    memory = gr.State({"name": None})

    with gr.Row():
        # Left panel
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

            show_debug = gr.Checkbox(
                value=False,
                label="Show routing details",
                info="Turn on to show routing + similarity."
            )

            gr.Markdown(
                f"LLM status: {'✅ enabled' if USE_OPENAI else '⚠️ disabled (no OPENAI_API_KEY)'}"
            )

            with gr.Accordion("System prompt (LLM)", open=False):
                system_prompt = gr.Textbox(
                    value=DEFAULT_SYSTEM_PROMPT,
                    lines=6,
                    label="System prompt",
                    placeholder="Controls assistant behavior for LLM fallback.",
                )

            reset_prompt = gr.Checkbox(value=False, label="Reset prompt to default on reset")
            reset_btn = gr.Button("🔄 Reset conversation", variant="secondary")

        # Right panel
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(label="Chat", height=520)

            with gr.Row():
                msg = gr.Textbox(label=None, placeholder="Type a message…", scale=10)
                send = gr.Button("Send", variant="primary", scale=2)

            gr.Examples(examples=EXAMPLES, inputs=msg, label="Try these examples")

    # Events
    send.click(
        chat_fn,
        inputs=[msg, chatbot, threshold, show_debug, system_prompt, memory],
        outputs=[chatbot, msg, memory]
    )
    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, threshold, show_debug, system_prompt, memory],
        outputs=[chatbot, msg, memory]
    )

    reset_btn.click(
        reset_session,
        inputs=[reset_prompt],
        outputs=[chatbot, msg, system_prompt, memory]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=APP_CSS)