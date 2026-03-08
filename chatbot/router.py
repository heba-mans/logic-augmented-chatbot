from __future__ import annotations

import os
import time
from typing import Dict, Tuple, Optional

from chatbot.intent_engine import IntentEngine
from chatbot.rules_engine import rules_reply
from chatbot.llm_engine import llm_reply, USE_OPENAI
from chatbot.memory import ensure_memory, extract_and_store_name
from chatbot.logger import get_logger, log_event

LOG_FILE = os.getenv("APP_LOG_FILE")  # e.g. logs/app.log
logger = get_logger("logic_chatbot", log_file=LOG_FILE)


def format_footer(meta: Dict) -> str:
    route = meta.get("route", "unknown")
    parts = [f"Route: {route}"]
    if meta.get("rule"):
        parts.append(f"rule={meta['rule']}")
    if meta.get("tag"):
        parts.append(f"tag={meta['tag']}")
    if meta.get("score") is not None:
        parts.append(f"score={meta['score']:.2f}")
    if meta.get("latency_ms") is not None:
        parts.append(f"latency_ms={meta['latency_ms']}")
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
    Returns (reply, updated_memory, meta) where meta includes routing details.
    Also logs a structured event with route/score/latency.
    """
    t0 = time.perf_counter()
    memory = ensure_memory(memory)

    # A few safe fields to log (avoid storing full user message in logs unless you want)
    base_log = {
        "threshold": float(threshold),
        "has_name": bool(memory.get("name")),
        "llm_enabled": bool(USE_OPENAI),
    }

    if startup_banner:
        meta = {"route": "startup_error"}
        latency_ms = int((time.perf_counter() - t0) * 1000)
        meta["latency_ms"] = latency_ms
        log_event(logger, "chat.route", {**base_log, **meta})
        return startup_banner, memory, meta

    # Memory rule
    mem_reply, memory = extract_and_store_name(message, memory)
    if mem_reply:
        meta = {"route": "memory", "rule": "capture_name"}
        latency_ms = int((time.perf_counter() - t0) * 1000)
        meta["latency_ms"] = latency_ms
        log_event(logger, "chat.route", {**base_log, **meta})
        return mem_reply, memory, meta

    # Rules layer
    rule_text = rules_reply(message)
    if rule_text:
        meta = {"route": "rules", "rule": "rules_reply"}
        latency_ms = int((time.perf_counter() - t0) * 1000)
        meta["latency_ms"] = latency_ms
        log_event(logger, "chat.route", {**base_log, **meta})
        return rule_text, memory, meta

    # Intent layer
    match = intent_engine.match(message, threshold=threshold)
    if match["type"] == "intent":
        reply_text = match["text"]
        if memory.get("name") and match.get("tag") == "greeting":
            reply_text = f"Hey {memory['name']}! 👋 How can I help you today?"

        meta = {
            "route": "intent",
            "tag": match.get("tag"),
            "score": float(match.get("score") or 0.0),
        }
        latency_ms = int((time.perf_counter() - t0) * 1000)
        meta["latency_ms"] = latency_ms
        log_event(logger, "chat.route", {**base_log, **meta})
        return reply_text, memory, meta

    # LLM fallback
    answer = llm_reply(message, system_prompt=system_prompt)
    if not USE_OPENAI:
        answer += "\n\n💡 Tip: Add OPENAI_API_KEY in a local .env to enable LLM fallback."

    meta = {"route": "llm_fallback"}
    latency_ms = int((time.perf_counter() - t0) * 1000)
    meta["latency_ms"] = latency_ms
    log_event(logger, "chat.route", {**base_log, **meta})
    return answer, memory, meta
