from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure repo root is on PYTHONPATH when running `python scripts/smoke_test.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chatbot.intent_engine import IntentEngine  # noqa: E402
from chatbot.router import route_and_reply  # noqa: E402


def fail(msg: str) -> None:
    print(f"❌ SMOKE TEST FAILED: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"✅ {msg}")


def main() -> None:
    intents_path = REPO_ROOT / "data" / "intents.json"

    # 1) Intents file exists + valid JSON
    if not intents_path.exists():
        fail(f"Missing intents file: {intents_path}")

    try:
        intents_data = json.loads(intents_path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"Invalid intents.json JSON: {e}")

    if "intents" not in intents_data or not isinstance(intents_data["intents"], list) or len(intents_data["intents"]) == 0:
        fail("intents.json must contain a non-empty top-level 'intents' list")

    ok("Loaded data/intents.json")

    # 2) Intent engine loads
    try:
        t0 = time.time()
        engine = IntentEngine(intents_path)
        ok(f"IntentEngine loaded ({time.time() - t0:.2f}s)")
    except Exception as e:
        fail(f"IntentEngine failed to load: {e}")

    # Shared inputs
    threshold = 0.70
    system_prompt = "You are a helpful assistant."
    startup_banner = None
    show_debug = False

    # 3) Memory capture
    memory = {"name": None}
    reply, memory = route_and_reply(
        "my name is Hiba",
        intent_engine=engine,
        threshold=threshold,
        show_debug=show_debug,
        system_prompt=system_prompt,
        memory=memory,
        startup_banner=startup_banner,
    )
    if "Hiba" not in reply or memory.get("name") != "Hiba":
        fail("Memory rule failed: name not captured or reply not personalized")
    ok("Memory rule works (name captured)")

    # 4) Rules sanity (contact / open now)
    reply, memory = route_and_reply(
        "How do I contact support?",
        intent_engine=engine,
        threshold=threshold,
        show_debug=show_debug,
        system_prompt=system_prompt,
        memory=memory,
        startup_banner=startup_banner,
    )
    if "support" not in reply.lower():
        fail("Rules sanity failed: contact rule did not trigger")
    ok("Rules layer works (contact)")

    reply, memory = route_and_reply(
        "Are you open now?",
        intent_engine=engine,
        threshold=threshold,
        show_debug=show_debug,
        system_prompt=system_prompt,
        memory=memory,
        startup_banner=startup_banner,
    )
    if "open" not in reply.lower() and "closed" not in reply.lower():
        fail("Rules sanity failed: open-now rule did not trigger")
    ok("Rules layer works (open now inference)")

    # 5) Intent sanity (greeting)
    reply, memory = route_and_reply(
        "hello",
        intent_engine=engine,
        threshold=threshold,
        show_debug=show_debug,
        system_prompt=system_prompt,
        memory=memory,
        startup_banner=startup_banner,
    )
    if not isinstance(reply, str) or len(reply.strip()) == 0:
        fail("Intent sanity failed: empty reply")
    ok("Intent routing works (hello → response)")

    print("\n🎉 SMOKE TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
