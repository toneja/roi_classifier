"""Logistic regression with forward feature selection, evaluation and per-image classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from . import config
from .datasets import ModelDataset, load_roi_tables
from .util import Log, now_iso

NOT_AVAILABLE = "n/a"  # shown instead of a percentage when an image has no counted ROIs


def build_pipeline() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def forward_select(X: pd.DataFrame, y: pd.Series, n_features: int, log: Log) -> list[str]:
    """Forward Sequential Feature Selection, scored by cross-validated accuracy."""
    if n_features > X.shape[1]:
        raise ValueError(
            f"Features to select ({n_features}) is more than the {X.shape[1]} "
            "candidate features in the dataset."
        )
    log(f"Running forward feature selection for {n_features} features (this can take a moment) ...")
    selector = SequentialFeatureSelector(
        build_pipeline(),
        n_features_to_select=n_features,
        direction="forward",
        scoring="accuracy",
        cv=config.SFS_CV_FOLDS,
    )
    selector.fit(X, y)
    selected = [name for name, keep in zip(X.columns, selector.get_support()) if keep]
    log(f"Selected features: {selected}")
    return selected


@dataclass
class TrainedModel:
    model: Pipeline
    features: list[str]
    classes: list[int]
    n_features: int
    trained_at: str


def train_model(dataset: ModelDataset, n_features: int, log: Log) -> TrainedModel:
    """Select features on the whole dataset and fit the final model."""
    selected = forward_select(dataset.X, dataset.y, n_features, log)
    model = build_pipeline().fit(dataset.X[selected], dataset.y)
    return TrainedModel(
        model=model,
        features=selected,
        classes=dataset.classes,
        n_features=n_features,
        trained_at=now_iso(),
    )


@dataclass
class Evaluation:
    selected: list[str]
    accuracy: float
    report: str
    model: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def summary(self) -> str:
        return (
            f"Selected features: {self.selected}\n"
            f"Test accuracy: {self.accuracy:.4f}\n\n{self.report}"
        )


def evaluate_split(
    dataset: ModelDataset,
    test_size: float,
    n_features: int,
    labels: dict[int, str],
    log: Log,
) -> Evaluation:
    """Train/test evaluation. Feature selection sees only the training split."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X,
            dataset.y,
            test_size=test_size,
            stratify=dataset.y,
            random_state=config.RANDOM_STATE,
        )
    except ValueError as exc:
        raise ValueError(f"The test fraction does not suit this dataset: {exc}") from None

    selected = forward_select(X_train, y_train, n_features, log)
    model = build_pipeline().fit(X_train[selected], y_train)
    y_pred = model.predict(X_test[selected])

    classes = sorted(labels)
    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=[labels[c] for c in classes],
        zero_division=0,
    )
    return Evaluation(
        selected=selected,
        accuracy=accuracy_score(y_test, y_pred),
        report=report,
        model=model,
        X_test=X_test[selected],
        y_test=y_test,
    )


def build_roc_figure(evaluation: Evaluation, labels: dict[int, str]) -> Figure:
    """ROC curve for two classes; one-vs-rest curves plus a micro-average otherwise."""
    model, X_test, y_test = evaluation.model, evaluation.X_test, evaluation.y_test
    classes = list(model.classes_)
    proba = model.predict_proba(X_test)

    fig = Figure(figsize=(7, 6), dpi=100)
    ax = fig.add_subplot(111)
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_test == classes[1], proba[:, 1])
        ax.plot(fpr, tpr, label=f"{labels[classes[1]]} (AUC = {auc(fpr, tpr):.3f})")
    else:
        y_bin = label_binarize(y_test, classes=classes)
        for idx, cls in enumerate(classes):
            if y_bin[:, idx].sum() == 0:
                continue  # no positives in the test set for this class
            fpr, tpr, _ = roc_curve(y_bin[:, idx], proba[:, idx])
            ax.plot(fpr, tpr, label=f"{labels[cls]} (AUC = {auc(fpr, tpr):.3f})")
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        ax.plot(
            fpr, tpr, linestyle="--", label=f"micro-average (AUC = {auc(fpr, tpr):.3f})"
        )

    ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curve (test set)",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


@dataclass
class ClassificationResult:
    per_image: pd.DataFrame
    features: list[str]
    n_images: int
    total: int

    @property
    def summary(self) -> str:
        return "\n".join(
            [
                f"ROI classification: {self.total} ROIs across {self.n_images} image(s)",
                f"Model features: {self.features}",
                "",
                self.per_image.to_string(index=False),
            ]
        )


def classify_rois(
    trained: TrainedModel,
    csv_files: list[Path],
    labels: dict[int, str],
    log: Log,
) -> ClassificationResult:
    """Predict every ROI, then tabulate per image: the total and each class's count and %."""
    images = [csv.stem for csv in csv_files]
    if len(set(images)) != len(images):
        raise ValueError("Image file names must be unique to report per image.")

    rois = load_roi_tables(csv_files, trained.features, log)
    if rois.empty:
        raise RuntimeError("No ROIs were found in any image.")
    predicted = pd.Series(
        trained.model.predict(rois[trained.features]), index=rois.index
    )
    counts = pd.crosstab(rois["source"], predicted).reindex(
        index=images, columns=trained.classes, fill_value=0
    )

    # Negative classes (e.g. -1 debris) are left out of totals and percentages,
    # but their counts are still reported.
    included = [cls for cls in trained.classes if cls >= 0]
    rows: list[dict[str, object]] = []
    for image in images:
        total = int(sum(int(counts.loc[image, cls]) for cls in included))
        row: dict[str, object] = {"image": image, "total_rois": total}
        for cls in trained.classes:
            n = int(counts.loc[image, cls])
            name = labels[cls]
            if cls < 0:
                row[f"{name} count (excluded)"] = n
            else:
                row[f"{name} count"] = n
                row[f"{name} %"] = round(100 * n / total, 2) if total else NOT_AVAILABLE
        rows.append(row)

    return ClassificationResult(
        per_image=pd.DataFrame(rows),
        features=trained.features,
        n_images=len(images),
        total=sum(int(row["total_rois"]) for row in rows),
    )
