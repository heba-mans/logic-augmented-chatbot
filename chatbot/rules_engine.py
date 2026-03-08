import re


def rules_reply(message: str) -> str | None:
    """
    Simple deterministic rules to make the project feel 'logic-augmented'.
    Return a string if a rule matches, otherwise None.
    """
    m = message.lower().strip()

    if re.search(r"\b(hours|opening|open|close)\b", m):
        return "We’re open Monday–Friday, 9am–5pm."

    if re.search(r"\b(contact|email|phone|support)\b", m):
        return "You can contact support at support@example.com."

    if re.search(r"\b(location|address|where are you)\b", m):
        return "We’re based online for this demo, but I can help route your request."

    return None