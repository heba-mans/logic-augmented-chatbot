import re
from datetime import datetime


HOURS = {
    "timezone_note": "Local time on the machine running the app",
    "days_open": {0, 1, 2, 3, 4},  # Mon-Fri (Mon=0)
    "open_hour": 9,               # 9am
    "close_hour": 17,             # 5pm (17:00)
}

CONTACT_EMAIL = "support@example.com"


def _is_open_now(now: datetime) -> bool:
    if now.weekday() not in HOURS["days_open"]:
        return False
    return HOURS["open_hour"] <= now.hour < HOURS["close_hour"]


def rules_reply(message: str) -> str | None:
    """
    Deterministic rules layer.
    Return a string if a rule matches, otherwise None.
    """
    m = message.lower().strip()

    # Rule 1: Hours (static)
    if re.search(r"\b(hours|opening hours|open hours|when are you open)\b", m):
        return "We’re open Monday–Friday, 9am–5pm."

    # Rule 2: Contact (static)
    if re.search(r"\b(contact|email|phone|support)\b", m):
        return f"You can contact support at {CONTACT_EMAIL}."

    # Rule 3: Location (static)
    if re.search(r"\b(location|address|where are you)\b", m):
        return "We’re based online for this demo, but I can help route your request."

    # Rule 4 (Inference): "open now?" based on local time
    if re.search(r"\b(open now|are you open right now|are you open now|open today)\b", m):
        now = datetime.now()
        open_now = _is_open_now(now)
        if open_now:
            return f"Yes — we’re currently open. (Local time: {now.strftime('%a %H:%M')})"
        return f"No — we’re currently closed. (Local time: {now.strftime('%a %H:%M')})"

    return None