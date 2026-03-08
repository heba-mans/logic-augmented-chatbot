# chatbot/memory.py
import re
from typing import Tuple, Dict, Optional

NAME_PATTERN = re.compile(r"\b(my name is|i am|i'm)\s+([A-Za-z][A-Za-z\-']{1,30})\b", re.IGNORECASE)

def ensure_memory(memory: Optional[Dict]) -> Dict:
    return memory if isinstance(memory, dict) else {"name": None}

def extract_and_store_name(message: str, memory: Dict) -> Tuple[Optional[str], Dict]:
    """
    If message contains a name introduction, store it in memory and return a response.
    Otherwise return (None, memory).
    """
    m = (message or "").strip()
    match = NAME_PATTERN.search(m)
    if not match:
        return None, memory

    name = match.group(2)
    memory = {**memory, "name": name}
    return f"Nice to meet you, {name}! 👋", memory