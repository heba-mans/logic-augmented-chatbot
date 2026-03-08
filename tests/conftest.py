from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add repo root so `import chatbot...` works when running pytest
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chatbot.intent_engine import IntentEngine  # noqa: E402

INTENTS_PATH = REPO_ROOT / "data" / "intents.json"


@pytest.fixture(scope="session")
def intent_engine():
    return IntentEngine(INTENTS_PATH)