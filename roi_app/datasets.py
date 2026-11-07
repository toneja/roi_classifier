"""Labelled model datasets, and the per-image ROI tables that get classified."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config
from .util import Log


@dataclass(frozen=True)
class ModelDataset:
    path: Path
    sha256: str  # fingerprint of the exact bytes the data was read from
    target: str
    features: list[str]
    X: pd.DataFrame
    y: pd.Series

    @property
    def classes(self) -> list[int]:
        return sorted(int(c) for c in self.y.unique())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_table(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [str(c).strip() for c in df.columns]
    return df


def select_features(columns: list[str], log: Log) -> list[str]:
    """The wanted feature columns that are present. Missing ones are reported and skipped."""
    available = set(columns)
    found = [f for f in config.WANTED_FEATURES if f in available]
    missing = [f for f in config.WANTED_FEATURES if f not in available]
    if missing:
        log(f"Feature columns not found and skipped: {missing}")
    if not found:
        raise ValueError("None of the wanted feature columns are present in the data.")
    return found


def load_model_dataset(path: Path, target_column: str, log: Log) -> ModelDataset:
    """
    Read the labelled dataset. The fingerprint and the training data both come from
    the same bytes, so a file edited during a run can't be paired with the wrong model.
    """
    raw = path.read_bytes()
    df = _read_table(raw)
    target = target_column.strip()
    if target not in df.columns:
        raise ValueError(
            f"Dataset has no column named '{target}'. "
            f"Set the target column to match {path.name}."
        )

    labels = pd.to_numeric(df[target], errors="coerce")
    not_numbers = df.loc[labels.isna() & df[target].notna(), target]
    if not not_numbers.empty:
        examples = not_numbers.head(3).tolist()
        raise ValueError(
            f"Target column '{target}' has values that are not numbers, e.g. {examples}. "
            "Classes must be whole numbers."
        )
    numeric = labels.dropna()
    if (numeric != numeric.round()).any():
        raise ValueError(
            f"Target column '{target}' has values that are not whole numbers. "
            "Classes must be integers."
        )

    features = select_features([c for c in df.columns if c != target], log)
    frame = pd.DataFrame({f: pd.to_numeric(df[f], errors="coerce") for f in features})
    frame[target] = labels
    valid = frame.dropna()
    if len(valid) < len(frame):
        log(f"Dropped {len(frame) - len(valid)} row(s) with missing values.")

    X = valid[features].astype(float)
    y = valid[target].round().astype(int)
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError(
            f"The dataset needs at least two classes; '{target}' has {len(counts)}."
        )
    too_small = sorted(int(c) for c in counts[counts < config.MIN_ROIS_PER_CLASS].index)
    if too_small:
        raise ValueError(
            f"Classes with fewer than {config.MIN_ROIS_PER_CLASS} labelled ROIs "
            f"cannot be modelled: {too_small}"
        )

    classes = sorted(int(c) for c in counts.index)
    log(
        f"Loaded {len(X)} labelled ROIs, {len(features)} candidate features, "
        f"{len(classes)} classes: {classes}"
    )
    return ModelDataset(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        target=target,
        features=features,
        X=X,
        y=y,
    )


def load_roi_tables(csv_files: list[Path], features: list[str], log: Log) -> pd.DataFrame:
    """Combine the per-image ROI tables, tagged with the image name in 'source'."""
    parts = []
    for csv_path in csv_files:
        try:
            df = _read_table(csv_path.read_bytes())
        except pd.errors.EmptyDataError:
            log(f"{csv_path.name} has no ROIs.")  # an image with no particles
            continue
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"{csv_path.name} is missing model features: {missing}")
        part = pd.DataFrame({f: pd.to_numeric(df[f], errors="coerce") for f in features})
        part.insert(0, "source", csv_path.stem)
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["source", *features])
    combined = pd.concat(parts, ignore_index=True)
    valid = combined.dropna(subset=features)
    if len(valid) < len(combined):
        log(f"Dropped {len(combined) - len(valid)} ROI(s) with missing feature values.")
    return valid
