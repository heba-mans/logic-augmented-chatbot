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


def llm_reply(user_message: str, system_prompt: str | None = None) -> str:
    """
    Returns an LLM-generated reply if OpenAI is enabled.
    If the key is missing/invalid or a transient API error happens, returns a friendly message
    instead of crashing the app.
    """
    if not USE_OPENAI or _client is None:
        return (
            "LLM fallback is disabled (no OPENAI_API_KEY found).\n\n"
            "For the interview demo: set OPENAI_API_KEY in a local .env (do not commit it)."
        )

    sys_prompt = (system_prompt or "").strip() or "You are a helpful assistant for a chatbot demo."

    try:
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content

    except Exception as e:
        # Try to provide a helpful, non-technical message for common OpenAI errors
        err_name = type(e).__name__

        # Newer OpenAI SDK exposes these error types; import lazily to avoid issues if missing
        try:
            from openai import AuthenticationError, RateLimitError, APIConnectionError, APIError  # type: ignore

            if isinstance(e, AuthenticationError):
                return (
                    "⚠️ LLM fallback is enabled, but your OpenAI API key is invalid.\n\n"
                    "Fix: update OPENAI_API_KEY in your local `.env` (or Hugging Face Space secret) and restart."
                )
            if isinstance(e, RateLimitError):
                return (
                    "⚠️ LLM fallback hit a rate limit.\n\n"
                    "Please try again in a moment, or use a different API key/plan."
                )
            if isinstance(e, APIConnectionError):
                return (
                    "⚠️ LLM fallback couldn’t reach OpenAI (network/connectivity issue).\n\n"
                    "Please try again in a moment."
                )
            if isinstance(e, APIError):
                return (
                    "⚠️ OpenAI returned an API error.\n\n"
                    "Please try again in a moment."
                )
        except Exception:
            # If importing OpenAI error classes fails, fall back to generic message below
            pass

        # Generic fallback (keeps your app running)
        return (
            f"⚠️ LLM fallback failed ({err_name}).\n\n"
            "The chatbot will still work via rules + intent routing. "
            "If you want LLM fallback, verify your OPENAI_API_KEY and try again."
        )