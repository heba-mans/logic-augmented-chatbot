from chatbot.rules_engine import rules_reply

def test_rules_contact():
    r = rules_reply("How do I contact support?")
    assert r is not None
    assert "support" in r.lower() or "contact" in r.lower()

def test_rules_open_now():
    r = rules_reply("Are you open now?")
    assert r is not None
    assert ("open" in r.lower()) or ("closed" in r.lower())
