from chatbot.memory import ensure_memory, extract_and_store_name

def test_ensure_memory_default():
    mem = ensure_memory(None)
    assert isinstance(mem, dict)
    assert mem.get("name") is None

def test_extract_and_store_name_my_name_is():
    mem = {"name": None}
    reply, mem2 = extract_and_store_name("my name is Hiba", mem)
    assert reply is not None
    assert "Hiba" in reply
    assert mem2["name"] == "Hiba"

def test_extract_and_store_name_im():
    mem = {"name": None}
    reply, mem2 = extract_and_store_name("I'm Karim", mem)
    assert reply is not None
    assert "Karim" in reply
    assert mem2["name"] == "Karim"

def test_extract_and_store_name_no_match():
    mem = {"name": None}
    reply, mem2 = extract_and_store_name("hello there", mem)
    assert reply is None
    assert mem2["name"] is None
