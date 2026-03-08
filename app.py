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
            "- Check your internet connection (first run downloads model)\n"
            "- Recreate venv and reinstall requirements\n"
            "- Run: python app.py from repo root"
        )
else:
    startup_banner = f"⚠️ Startup error:\n\n{msg}"


def chat_fn(message, history):
    # If startup failed, return the banner instead of crashing
    if startup_banner:
        return startup_banner

    # 1) Deterministic rule layer
    rule = rules_reply(message)
    if rule:
        return f"{rule}\n\n_(route: rules)_"

    # 2) Intent matching layer
    match = intent_engine.match(message, threshold=0.70)
    if match["type"] == "intent":
        return f'{match["text"]}\n\n_(route: intent={match["tag"]}, similarity: {match["score"]:.2f})_'

    # 3) LLM fallback
    answer = llm_reply(message)
    if not USE_OPENAI:
        answer += "\n\nTip: Add OPENAI_API_KEY in a local `.env` to enable LLM fallback."
    return f"{answer}\n\n_(route: LLM fallback)_"


description = "Hybrid routing: rules → semantic intent matching → LLM fallback"
if not USE_OPENAI:
    description += "  •  (LLM fallback disabled: no OPENAI_API_KEY)"

demo = gr.ChatInterface(
    fn=chat_fn,
    title="Logic-Augmented Chatbot",
    description=description,
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