import os
from dotenv import load_dotenv

load_dotenv()

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI

        _client = OpenAI()
    except Exception:
        USE_OPENAI = False
        _client = None


def llm_reply(user_message: str) -> str:
    if not USE_OPENAI or _client is None:
        return (
            "LLM fallback is disabled (no OPENAI_API_KEY found).\n\n"
            "For the interview demo: set OPENAI_API_KEY in a local .env (do not commit it)."
        )

    resp = _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a chatbot demo."},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.choices[0].message.content