from typing import Dict, Tuple, Optional

from chatbot.intent_engine import IntentEngine
from chatbot.rules_engine import rules_reply
from chatbot.llm_engine import llm_reply, USE_OPENAI
from chatbot.memory import ensure_memory, extract_and_store_name


def format_footer(meta: Dict) -> str:
    """
    Compact explainability footer shown only in Demo Mode.
    """
    route = meta.get("route", "unknown")
    parts = [f"Route: {route}"]
    if meta.get("rule"):
        parts.append(f"rule={meta['rule']}")
    if meta.get("tag"):
        parts.append(f"tag={meta['tag']}")
    if meta.get("score") is not None:
        parts.append(f"score={meta['score']:.2f}")
    return " | ".join(parts)


def route_and_reply(
    message: str,
    *,
    intent_engine: IntentEngine,
    threshold: float,
    system_prompt: str,
    memory: Optional[Dict],
    startup_banner: Optional[str] = None,
) -> Tuple[str, Dict, Dict]:
    """
    Orchestrates: memory → rules → intent → LLM
    Returns (reply, updated_memory, meta)
    meta is used for Demo Mode explainability.
    """
    memory = ensure_memory(memory)

    if startup_banner:
        return startup_banner, memory, {"route": "startup_error"}

    # Memory rule
    mem_reply, memory = extract_and_store_name(message, memory)
    if mem_reply:
        return mem_reply, memory, {"route": "memory", "rule": "capture_name"}

    # Rules layer
    rule_text = rules_reply(message)
    if rule_text:
        return rule_text, memory, {"route": "rules", "rule": "rules_reply"}

    # Intent layer
    match = intent_engine.match(message, threshold=threshold)
    if match["type"] == "intent":
        reply_text = match["text"]
        if memory.get("name") and match.get("tag") == "greeting":
            reply_text = f"Hey {memory['name']}! 👋 How can I help you today?"

        return reply_text, memory, {
            "route": "intent",
            "tag": match.get("tag"),
            "score": match.get("score"),
        }

    # LLM fallback
    answer = llm_reply(message, system_prompt=system_prompt)
    if not USE_OPENAI:
        answer += "\n\n💡 Tip: Add OPENAI_API_KEY in a local .env to enable LLM fallback."

    return answer, memory, {"route": "llm_fallback"}