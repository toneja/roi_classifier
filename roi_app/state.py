"""Settings remembered between sessions, stored in data/state.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from . import config


@dataclass
class AppState:
    dataset_path: str = ""
    images_dir: str = ""
    test_size: float = config.DEFAULT_TEST_SIZE
    target_column: str = config.DEFAULT_TARGET_COLUMN
    size_min: float = config.DEFAULT_SIZE_MIN
    size_max: float = config.DEFAULT_SIZE_MAX
    n_features: int = config.DEFAULT_N_FEATURES
    class_names: dict[int, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "AppState":
        """Load saved settings. Missing or wrongly typed values fall back to defaults."""
        try:
            saved = json.loads(config.STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(saved, dict):
            return cls()
        defaults = cls()
        return cls(
            dataset_path=_text(saved.get("dataset_path"), defaults.dataset_path),
            images_dir=_text(saved.get("images_dir"), defaults.images_dir),
            test_size=_number(saved.get("test_size"), defaults.test_size),
            target_column=_text(saved.get("target_column"), defaults.target_column)
            or defaults.target_column,
            size_min=_number(saved.get("size_min"), defaults.size_min),
            size_max=_number(saved.get("size_max"), defaults.size_max),
            n_features=_integer(saved.get("n_features"), defaults.n_features),
            class_names=_class_names(saved.get("class_names")),
        )

    def save(self) -> None:
        data = asdict(self)
        data["class_names"] = {str(k): v for k, v in self.class_names.items()}
        try:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            config.STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass  # state is a convenience; never block the app on it


def _text(value: Any, default: str) -> str:
    return value if isinstance(value, str) else default


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _integer(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _class_names(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        return {}
    names: dict[int, str] = {}
    for key, name in value.items():
        try:
            cls = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(name, str) and name.strip():
            names[cls] = name
    return names
