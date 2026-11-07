"""Class names: typed in the app, read from class_labels.txt, or generic."""

from __future__ import annotations

import ast

from . import config
from .util import Log

LABELS_FORMAT_HINT = '{ -1: "debris", 0: "ungerminated", 1: "germinated" }'


def label_problem(names: dict[int, str]) -> str | None:
    """Describe what is wrong with a set of class names, or return None if they are fine."""
    blank = sorted(cls for cls, name in names.items() if not name.strip())
    if blank:
        return f"Every class needs a name. Missing: {blank}"
    counts: dict[str, int] = {}
    for name in names.values():
        key = name.strip()
        counts[key] = counts.get(key, 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        return f"Each class needs its own name. Repeated: {duplicates}"
    return None


def parse_class_labels(text: str, source: str) -> dict[int, str]:
    """Parse a {int: str} dict written as a Python literal and check it."""
    try:
        raw = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"{source} could not be parsed: {exc}") from None

    valid = isinstance(raw, dict) and all(
        isinstance(k, int) and not isinstance(k, bool) and isinstance(v, str)
        for k, v in raw.items()
    )
    if not valid:
        raise ValueError(
            f"{source} must hold a dict of {{int: str}}, e.g. {LABELS_FORMAT_HINT}"
        )

    labels = {int(k): v for k, v in raw.items()}
    problem = label_problem(labels)
    if problem:
        raise ValueError(f"{source}: {problem}")
    return labels


def resolve_class_labels(
    expected: list[int], typed: dict[int, str] | None, log: Log
) -> dict[int, str]:
    """
    Name every class in the model. Priority: names typed in the app, then
    class_labels.txt in the project folder, then generic 'class <n>' names.
    Raises ValueError when the names do not match the model's classes exactly.
    """
    if typed:
        labels, source = dict(typed), "set in the app"
    elif config.CLASS_LABELS_FILE.is_file():
        labels = parse_class_labels(
            config.CLASS_LABELS_FILE.read_text(encoding="utf-8"), "class_labels.txt"
        )
        source = "class_labels.txt"
    else:
        log("No class labels given; using generic class names.")
        return {cls: f"class {cls}" for cls in expected}

    model_classes = set(expected)
    label_classes = set(labels)
    missing = sorted(model_classes - label_classes)
    extra = sorted(label_classes - model_classes)
    if missing or extra:
        raise ValueError(
            f"The class labels ({source}) do not match the classes in the model dataset.\n"
            f"  Model classes:            {sorted(model_classes)}\n"
            f"  Classes in labels:        {sorted(label_classes)}\n"
            f"  Missing from labels:      {missing or 'none'}\n"
            f"  Not in the model dataset: {extra or 'none'}"
        )

    log(
        f"Class labels from {source}: "
        f"{', '.join(f'{k} = {labels[k]}' for k in sorted(labels))}"
    )
    return labels


def label_source_text(typed: dict[int, str] | None) -> str:
    """One line for the window saying where the class names come from."""
    if typed:
        return ", ".join(f"{k} = {v}" for k, v in sorted(typed.items()))
    if config.CLASS_LABELS_FILE.is_file():
        return "Using class_labels.txt"
    return "Using generic class names"
