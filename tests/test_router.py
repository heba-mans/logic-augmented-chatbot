from chatbot.router import route_and_reply

def test_router_memory_rule_precedence(intent_engine):
    memory = {"name": None}
    reply, memory, meta = route_and_reply(
        "my name is Hiba",
        intent_engine=intent_engine,
        threshold=0.70,
        system_prompt="You are helpful",
        memory=memory,
        startup_banner=None,
    )
    assert "Hiba" in reply
    assert memory["name"] == "Hiba"
    assert meta["route"] in {"memory", "rules", "intent", "llm_fallback", "startup_error"}
    assert meta["route"] == "memory"

def test_router_rules_precedence_over_intent(intent_engine):
    memory = {"name": None}
    reply, memory, meta = route_and_reply(
        "How do I contact support?",
        intent_engine=intent_engine,
        threshold=0.10,  # low threshold shouldn't matter; rules should win
        system_prompt="You are helpful",
        memory=memory,
        startup_banner=None,
    )
    assert "support" in reply.lower() or "contact" in reply.lower()
    assert meta["route"] == "rules"

def test_router_startup_banner_short_circuit(intent_engine):
    memory = {"name": None}
    banner = "Startup failed"
    reply, memory, meta = route_and_reply(
        "hello",
        intent_engine=intent_engine,
        threshold=0.70,
        system_prompt="You are helpful",
        memory=memory,
        startup_banner=banner,
    )
    assert reply == banner
    assert meta["route"] == "startup_error"
