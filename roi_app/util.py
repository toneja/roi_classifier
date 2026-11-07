"""Small helpers shared across the app."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

Log = Callable[[str], None]


def silent(_message: str) -> None:
    """A log function that discards its messages."""


def timestamp() -> str:
    """A sortable timestamp for folder and file names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
