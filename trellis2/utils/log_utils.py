# File: trellis2/utils/log_utils.py
import os
from typing import Literal


LogLevel = Literal["quiet", "normal", "verbose", "debug"]

_LEVEL_TO_INT = {
    "quiet": 0,
    "normal": 1,
    "verbose": 2,
    "debug": 3,
}


def get_log_level() -> LogLevel:
    raw = os.environ.get("TRELLIS_LOG_LEVEL", "").strip().lower()
    if raw in _LEVEL_TO_INT:
        return raw  # type: ignore[return-value]

    # Backward compatibility:
    # TRELLIS_RENDER_DIAG=1 is treated as verbose.
    if os.environ.get("TRELLIS_RENDER_DIAG", "0") == "1":
        return "verbose"

    return "normal"


def should_log(level: LogLevel) -> bool:
    return _LEVEL_TO_INT[get_log_level()] >= _LEVEL_TO_INT[level]


def log(level: LogLevel, message: str) -> None:
    if should_log(level):
        print(message)

