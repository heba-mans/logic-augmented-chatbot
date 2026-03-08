# chatbot/router.py
from typing import Dict, Tuple, Optional

from chatbot.intent_engine import IntentEngine
from chatbot.rules_engine import rules_reply
from chatbot.llm_engine import llm_reply, USE_OPENAI
from chatbot.memory import ensure_memory, extract_and_store_name


def _details_line(text: str, show_debug: bool) -> str:
    return f"\n\n_( {text} )_" if show_debug else ""


def route_and_reply(
    message: str,
    *,
    intent_engine: IntentEngine,
    threshold: float,
    show_debug: bool,
    system_prompt: str,
    memory: Optional[Dict],
    startup_banner: Optional[str] = None,
) -> Tuple[str, Dict]:
    """
    Orchestrates: memory → rules → intent → LLM
    Returns (reply, updated_memory).
    """
    memory = ensure_memory(memory)

    if startup_banner:
        return startup_banner, memory

    # Memory rule: capture name
    mem_reply, memory = extract_and_store_name(message, memory)
    if mem_reply:
        return mem_reply, memory

    # 1) Rule layer
    rule = rules_reply(message)
    if rule:
        return f"{rule}{_details_line('Route: RULES', show_debug)}", memory

    # 2) Intent layer
    match = intent_engine.match(message, threshold=threshold)
    if match["type"] == "intent":
        route_text = f"Route: INTENT | tag={match['tag']} | similarity={match['score']:.2f}"
        reply_text = match["text"]

        # Personalize greeting if name is known
        if memory.get("name") and match.get("tag") == "greeting":
            reply_text = f"Hey {memory['name']}! 👋 How can I help you today?"

        return f"{reply_text}{_details_line(route_text, show_debug)}", memory

    # 3) LLM fallback
    answer = llm_reply(message, system_prompt=system_prompt)
    if not USE_OPENAI:
        answer += "\n\n💡 Tip: Add OPENAI_API_KEY in a local .env to enable LLM fallback."

    return f"{answer}{_details_line('Route: LLM FALLBACK', show_debug)}", memory