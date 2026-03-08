from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


def get_logger(name: str = "app", log_file: Optional[str] = None) -> logging.Logger:
    """
    Creates a logger that emits JSON lines. Logs to stdout and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers in reload scenarios
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_event(logger: logging.Logger, event: str, payload: Dict[str, Any]) -> None:
    record = {"event": event, **payload}
    logger.info(json.dumps(record, ensure_ascii=False))
