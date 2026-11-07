"""Saved models: one per dataset file.

Each model is stored under a name made from the dataset's file name plus a short
hash of its location, so two datasets with the same file name never share a model.
A model is only used with the exact dataset file it was trained on (SHA-256 check).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import joblib

from . import config
from .datasets import ModelDataset, sha256_file
from .modelling import TrainedModel

REQUIRED_SUMMARY_KEYS = {"dataset_sha256", "features", "classes", "n_features", "trained_at"}


class ModelError(RuntimeError):
    """There is no usable saved model for a dataset. The message says what to do."""


def _stem(dataset_path: Path) -> str:
    location = hashlib.sha256(str(dataset_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{dataset_path.stem}_{location}"


def model_file(dataset_path: Path) -> Path:
    return config.MODELS_DIR / f"{_stem(dataset_path)}.joblib"


def summary_file(dataset_path: Path) -> Path:
    return config.MODELS_DIR / f"{_stem(dataset_path)}.json"


def _replace_atomically(target: Path, write: Callable[[Path], None]) -> None:
    temp = target.with_name(target.name + ".tmp")
    write(temp)
    os.replace(temp, target)  # overwrites the old file; works on Windows too


def save_model(trained: TrainedModel, dataset: ModelDataset) -> Path:
    """Save the fitted pipeline and a JSON summary. The summary is written last."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_name": dataset.path.name,
        "dataset_path": str(dataset.path.resolve()),
        "dataset_sha256": dataset.sha256,
        "target": dataset.target,
        "features": trained.features,
        "classes": trained.classes,
        "n_features": trained.n_features,
        "trained_at": trained.trained_at,
    }
    _replace_atomically(model_file(dataset.path), lambda p: joblib.dump(trained.model, p))
    _replace_atomically(
        summary_file(dataset.path),
        lambda p: p.write_text(json.dumps(summary, indent=2), encoding="utf-8"),
    )
    return model_file(dataset.path)


def read_model_summary(dataset: Path) -> dict:
    """
    The summary of the saved model for this dataset. Raises ModelError with a
    message for the user if there is no model, or it was trained on another file.
    """
    if not (summary_file(dataset).is_file() and model_file(dataset).is_file()):
        raise ModelError(
            f"No model has been trained for {dataset.name} yet. "
            "Run B (Select features & train)."
        )
    try:
        summary = json.loads(summary_file(dataset).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = None
    if not isinstance(summary, dict) or not REQUIRED_SUMMARY_KEYS <= summary.keys():
        raise ModelError(
            f"The saved model for {dataset.name} is unreadable. "
            "Run B (Select features & train) again."
        )
    if summary["dataset_sha256"] != sha256_file(dataset):
        raise ModelError(
            f"The saved model for {dataset.name} was trained on an earlier version of "
            "that file. Run B (Select features & train) again."
        )
    return summary


def load_model(dataset: Path) -> TrainedModel:
    """Load the saved model for this dataset, checking it matches the current file."""
    summary = read_model_summary(dataset)
    try:
        model = joblib.load(model_file(dataset))
    except Exception as exc:  # corrupt or incompatible file
        raise ModelError(
            f"The saved model for {dataset.name} could not be loaded ({exc}). "
            "Run B (Select features & train) again."
        ) from None
    return TrainedModel(
        model=model,
        features=list(summary["features"]),
        classes=[int(c) for c in summary["classes"]],
        n_features=int(summary["n_features"]),
        trained_at=str(summary["trained_at"]),
    )
