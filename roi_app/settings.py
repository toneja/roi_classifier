"""Turn the window's text fields into checked values.

Each parser raises InputError with a message that can be shown to the user as is.
"""

from __future__ import annotations

import math
from pathlib import Path


class InputError(ValueError):
    """An input the user needs to fix. The message is shown as it is."""


def clean_path(text: str) -> Path:
    """A path typed or pasted into a field, without surrounding quotes or spaces."""
    return Path(text.strip().strip('"'))


def parse_dataset(text: str) -> Path:
    path = clean_path(text)
    if not path.is_file():
        raise InputError("Select a valid model dataset CSV.")
    return path


def parse_images(text: str) -> Path:
    path = clean_path(text)
    if not path.is_dir():
        raise InputError("Select a valid image folder.")
    return path


def parse_test_size(text: str) -> float:
    try:
        value = float(text.strip())
    except ValueError:
        raise InputError("Test fraction must be a number between 0 and 1.") from None
    if not 0 < value < 1:
        raise InputError("Test fraction must be between 0 and 1.")
    return value


def parse_target(text: str) -> str:
    target = text.strip()
    if not target:
        raise InputError("Enter the name of the target (classification) column.")
    return target


def parse_n_features(text: str) -> int:
    try:
        value = int(text.strip())
    except ValueError:
        raise InputError("Features to select must be a whole number.") from None
    if value < 1:
        raise InputError("Features to select must be at least 1.")
    return value


def parse_size_bounds(low_text: str, high_text: str) -> tuple[float, float]:
    try:
        low = float(low_text.strip())
        high = float(high_text.strip())
    except ValueError:
        raise InputError("Particle size bounds must be numbers.") from None
    if not (math.isfinite(low) and math.isfinite(high)):
        raise InputError("Particle size bounds must be finite numbers.")
    if low < 0 or high < low:
        raise InputError("Particle size bounds need 0 <= min <= max.")
    return low, high
