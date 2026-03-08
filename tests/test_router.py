from chatbot.router import route_and_reply

def test_router_memory_rule_precedence(intent_engine):
    memory = {"name": None}
    reply, memory = route_and_reply(
        "my name is Hiba",
        intent_engine=intent_engine,
        threshold=0.70,
        show_debug=False,
        system_prompt="You are helpful",
        memory=memory,
        startup_banner=None,
    )
    assert "Hiba" in reply
    assert memory["name"] == "Hiba"

def test_router_rules_precedence_over_intent(intent_engine):
    # Even if intent engine might match something, rules should win for contact
    memory = {"name": None}
    reply, memory = route_and_reply(
        "How do I contact support?",
        intent_engine=intent_engine,
        threshold=0.10,  # low threshold shouldn't matter; rules should still win
        show_debug=False,
        system_prompt="You are helpful",
        memory=memory,
        startup_banner=None,
    )
    # We don't assert exact text, just that it looks like rules response
    assert "support" in reply.lower() or "contact" in reply.lower()

def test_router_startup_banner_short_circuit(intent_engine):
    memory = {"name": None}
    banner = "Startup failed"
    reply, memory = route_and_reply(
        "hello",
        intent_engine=intent_engine,
        threshold=0.70,
        show_debug=False,
        system_prompt="You are helpful",
        memory=memory,
        startup_banner=banner,
    )
    assert reply == banner
