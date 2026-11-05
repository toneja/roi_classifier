#!/usr/bin/env python3
"""
roi_classifier_gui.py - Tkinter GUI for classifying ImageJ ROIs with logistic regression.

Workflow
--------
1. Batch-run an ImageJ macro over a folder of fresh images -> one CSV per image.
2. Import the labelled model dataset (CSV with a 'class' column).
3. Use the features chosen by forward Sequential Feature Selection to classify ROIs.
4. Report, per image, the ROI total and the count and percentage of each class.

Buttons
-------
A. Test accuracy        - train/test split, forward SFS on the training split,
                          classification report, ROC curve.
B. Select features      - forward SFS on the full dataset; trains and saves the model.
C. Classify images      - ImageJ batch run -> predictions -> per-image report.

Class labels
------------
Optional file: class_labels.txt in the project folder, containing a Python dict, e.g.
    { -1: "debris", 0: "ungerminated", 1: "germinated" }
It must list exactly the classes found in the model dataset; mismatches stop the run.
Without the file, classes are shown as "class <n>".

Requirements
------------
    pip install numpy pandas scikit-learn matplotlib joblib
    Tkinter ships with most Python installs (on Linux: install python3-tk).
    Java 8+ is needed only if the downloaded ImageJ bundle has no bundled Java
    (the cross-platform bundle does not).

ImageJ batch-macro contract
---------------------------
Run as:  java -Droi.runDir=<run_dir> -Droi.mainMacro=<AnalyzeSporesAndGermlings.ijm>
         -jar ij.jar -batch <BatchProcess.ijm> "<image_dir>"

- The image folder path is the batch macro's single argument (getArgument()).
- The run folder (data/runs/<timestamp>) is passed as the JVM property roi.runDir.
  Macros read it with: call("java.lang.System.getProperty", "roi.runDir")
- The batch macro runs the main macro on each image. The main macro writes its
  CSV tables into the run folder. The app collects every CSV under the run folder.
- Both macros live in the project's macros folder.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import threading
import traceback
import urllib.request
import zipfile
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MACROS_DIR = PROJECT_DIR / "macros"  # the custom ImageJ macros
CLASS_LABELS_FILE = PROJECT_DIR / "class_labels.txt"
APP_DIR = PROJECT_DIR / "data"  # everything the app generates
IMAGEJ_DIR = APP_DIR / "ImageJ"
STATE_FILE = APP_DIR / "state.json"
MODELS_DIR = APP_DIR / "models"  # one saved model per dataset
REPORTS_DIR = APP_DIR / "reports"
RUNS_DIR = APP_DIR / "runs"
BATCH_MACRO_FILE = MACROS_DIR / "BatchProcess.ijm"
MAIN_MACRO_FILE = MACROS_DIR / "AnalyzeSporesAndGermlings.ijm"

# Column holding the manual classification identifier (integer).
TARGET_COLUMN = "manual_class"

# Only these ImageJ Analyze Particles columns are used as features. Any other
# columns in the data are ignored. Missing ones are reported and skipped.
WANTED_FEATURES = [
    "Area",
    "Perim.",
    "Major",
    "Minor",
    "Circ.",
    "Feret",
    "MinFeret",
    "AR",
    "Round",
    "Solidity",
]

# ImageJ 1.x distributions (version 1.54). Platform bundles include a Java runtime.
IMAGEJ_URLS = {
    "windows": "https://wsr.imagej.net/distros/win/ij154-win-java8.zip",
    "macos-intel": "https://wsr.imagej.net/distros/osx/ij154-osx-java8.zip",
    "macos-arm": "https://wsr.imagej.net/distros/osx/ij154-osx-arm-java13.zip",
    "linux": "https://wsr.imagej.net/distros/linux/ij154-linux64-java8.zip",
    "cross-platform": "https://wsr.imagej.net/distros/cross-platform/ij154.zip",
}

RUN_DIR_PROPERTY = "roi.runDir"  # JVM property: run folder the macros write into
MAIN_MACRO_PROPERTY = "roi.mainMacro"  # JVM property: full path to the main macro
RANDOM_STATE = 42
SFS_CV_FOLDS = 5
SFS_TOLERANCE = 1e-3
DEFAULT_TEST_SIZE = 0.25

POLL_INTERVAL_MS = 100  # how often the UI checks for worker events
MAX_EVENTS_PER_TICK = 200  # cap per poll so a chatty ImageJ can't freeze the window

# Only these ImageJ output lines are shown in the app log; everything is still
# written to imagej.log in the run folder.
LOG_KEEP_PATTERN = re.compile(
    r"error|exception|warning|failed|not found|cannot|unable|^Processing ",
    re.IGNORECASE,
)

Log = Callable[[str], None]


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    dataset_path: str = ""
    images_dir: str = ""
    test_size: float = DEFAULT_TEST_SIZE

    @classmethod
    def load(cls) -> "AppState":
        try:
            saved = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {k: v for k, v in saved.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self) -> None:
        try:
            APP_DIR.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError:
            pass  # state is a convenience; never block the app on it


# ---------------------------------------------------------------------------
# ImageJ installation and batch execution
# ---------------------------------------------------------------------------


def get_imagej_url() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        return IMAGEJ_URLS["windows"]
    if system == "Darwin":
        is_arm = machine in ("arm64", "aarch64")
        return IMAGEJ_URLS["macos-arm" if is_arm else "macos-intel"]
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return IMAGEJ_URLS["linux"]
    return IMAGEJ_URLS["cross-platform"]


def _find_ij_jar() -> Path | None:
    if not IMAGEJ_DIR.exists():
        return None
    return next(IMAGEJ_DIR.rglob("ij.jar"), None)


def _extract_zip_safely(zip_path: Path, dest: Path) -> None:
    """Extract a zip, refusing any member that would land outside dest."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (dest / member.filename).resolve()
            if target != dest and dest not in target.parents:
                raise RuntimeError(f"Unsafe path in ImageJ archive: {member.filename}")
        archive.extractall(dest)
    if os.name != "nt":
        # zipfile does not restore execute bits; the bundled java needs them.
        for path in dest.rglob("*"):
            if path.is_file() and path.parent.name == "bin":
                path.chmod(path.stat().st_mode | 0o111)


def ensure_imagej(log: Log) -> Path:
    """Return the path to ij.jar, downloading and installing ImageJ if needed."""
    jar = _find_ij_jar()
    if jar:
        return jar

    url = get_imagej_url()
    log(f"ImageJ not found. Downloading from {url}")
    IMAGEJ_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = APP_DIR / "imagej_download.zip"
    urllib.request.urlretrieve(url, zip_path)
    try:
        log("Extracting ImageJ ...")
        _extract_zip_safely(zip_path, IMAGEJ_DIR)
    finally:
        zip_path.unlink(missing_ok=True)

    jar = _find_ij_jar()
    if jar is None:
        raise RuntimeError("ImageJ installed but ij.jar was not found in the archive.")
    log(f"ImageJ installed at {jar.parent}")
    return jar


def _find_java() -> str:
    """Prefer the Java runtime bundled with ImageJ, then fall back to the system Java."""
    exe_name = "java.exe" if os.name == "nt" else "java"
    if IMAGEJ_DIR.exists():
        for candidate in IMAGEJ_DIR.rglob(exe_name):
            if candidate.parent.name == "bin":
                return str(candidate)
    system_java = shutil.which("java")
    if system_java:
        return system_java
    raise RuntimeError(
        "No Java runtime found. Install Java 8 or newer, or use a platform ImageJ bundle."
    )


def run_imagej_batch(input_dir: Path, output_dir: Path, log: Log) -> list[Path]:
    """Run the batch macro over input_dir and return the CSV files it produced."""
    for macro in (BATCH_MACRO_FILE, MAIN_MACRO_FILE):
        if not macro.is_file():
            raise FileNotFoundError(f"Macro not found: {macro}")
    ij_jar = ensure_imagej(log)
    java = _find_java()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        java,
        f"-D{RUN_DIR_PROPERTY}={output_dir}",
        f"-D{MAIN_MACRO_PROPERTY}={MAIN_MACRO_FILE}",
        "-jar",
        str(ij_jar),
        "-batch",
        str(BATCH_MACRO_FILE),
        str(input_dir),  # the single macro argument: the image folder
    ]
    log(f"Batch macro: {BATCH_MACRO_FILE}")
    log(f"Image folder: {input_dir}")
    log("Starting ImageJ ...")

    log_path = output_dir / "imagej.log"
    recent: deque[str] = deque(maxlen=20)
    with log_path.open("w", encoding="utf-8") as ij_log:
        process = subprocess.Popen(
            command,
            cwd=IMAGEJ_DIR,  # keeps ImageJ's preferences and temp files inside data/
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            ij_log.write(line)
            recent.append(text)
            if LOG_KEEP_PATTERN.search(text):
                log(f"[ImageJ] {text}")
        returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(
            f"ImageJ exited with code {returncode}. Last output:\n" + "\n".join(recent)
        )

    csv_files = sorted(output_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(
            f"ImageJ finished but wrote no CSV files to {output_dir}. "
            f"Last output:\n" + "\n".join(recent)
        )
    log(f"ImageJ produced {len(csv_files)} CSV file(s).")
    return csv_files


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def select_features(
    columns, log: Log, wanted: list[str] = WANTED_FEATURES
) -> list[str]:
    """Return the wanted feature columns that exist, reporting any that are missing."""
    available = set(columns)
    found = [f for f in wanted if f in available]
    missing = [f for f in wanted if f not in available]
    if missing:
        log(f"Feature columns not found and skipped: {missing}")
    if not found:
        raise ValueError("None of the wanted feature columns are present in the data.")
    return found


def load_model_dataset(
    csv_path: Path, log: Log
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the labelled model dataset. Returns (X, y, feature_names)."""
    df = _clean_columns(pd.read_csv(csv_path))
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset has no '{TARGET_COLUMN}' column: {csv_path.name}")

    features = select_features(df.columns, log)
    data = df[features + [TARGET_COLUMN]].dropna()
    if len(data) < len(df):
        log(f"Dropped {len(df) - len(data)} row(s) with missing values.")

    X = data[features].astype(float)
    y = data[TARGET_COLUMN].round().astype(int)
    log(
        f"Loaded {len(X)} labelled ROIs, {len(features)} candidate features, "
        f"{y.nunique()} classes: {sorted(y.unique().tolist())}"
    )
    return X, y, features


def load_roi_tables(
    csv_files: list[Path], features: list[str], log: Log
) -> pd.DataFrame:
    """Combine per-image ROI CSVs into one frame, tagged with the source image name."""
    frames = []
    for csv_path in csv_files:
        try:
            df = _clean_columns(pd.read_csv(csv_path))
        except pd.errors.EmptyDataError:
            log(f"{csv_path.name} has no ROIs.")  # an image with no particles
            continue
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"{csv_path.name} is missing model features: {missing}")
        part = df[features].astype(float)
        part.insert(0, "source", csv_path.stem)
        frames.append(part)

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.dropna(subset=features)
    if len(combined) < before:
        log(f"Dropped {before - len(combined)} ROI(s) with missing feature values.")
    return combined


def load_class_labels(expected_classes: list[int], log: Log) -> dict[int, str]:
    """
    Read class_labels.txt (a Python dict of {int: str}) and check it against the
    model's classes. Without the file, generic 'class <n>' names are used.
    """
    if not CLASS_LABELS_FILE.is_file():
        log("No class_labels.txt found; using generic class names.")
        return {cls: f"class {cls}" for cls in expected_classes}

    try:
        raw = ast.literal_eval(CLASS_LABELS_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"class_labels.txt could not be parsed: {exc}") from None

    valid = isinstance(raw, dict) and all(
        isinstance(k, int) and not isinstance(k, bool) and isinstance(v, str)
        for k, v in raw.items()
    )
    if not valid:
        raise ValueError(
            "class_labels.txt must hold a dict of {int: str}, e.g. "
            '{ -1: "debris", 0: "ungerminated", 1: "germinated" }'
        )

    empty = [k for k, v in raw.items() if not v.strip()]
    if empty:
        raise ValueError(f"class_labels.txt has empty label names for classes: {empty}")

    names = list(raw.values())
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"class_labels.txt uses the same label for more than one class: {duplicates}"
        )

    model_classes = set(expected_classes)
    label_classes = set(raw)
    missing = sorted(model_classes - label_classes)
    extra = sorted(label_classes - model_classes)
    if missing or extra:
        raise ValueError(
            "class_labels.txt does not match the classes in the model dataset.\n"
            f"  Model classes:            {sorted(model_classes)}\n"
            f"  Classes in labels file:   {sorted(label_classes)}\n"
            f"  Missing from labels file: {missing or 'none'}\n"
            f"  Not in the model dataset: {extra or 'none'}"
        )

    labels = {int(k): v for k, v in raw.items()}
    log(f"Class labels: {', '.join(f'{k} = {labels[k]}' for k in sorted(labels))}")
    return labels


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------


def build_pipeline() -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def forward_select(X: pd.DataFrame, y: pd.Series, log: Log) -> list[str]:
    """Forward Sequential Feature Selection scored by cross-validated accuracy."""
    log("Running forward feature selection (this can take a moment) ...")
    selector = SequentialFeatureSelector(
        build_pipeline(),
        n_features_to_select="auto",
        tol=SFS_TOLERANCE,
        direction="forward",
        scoring="accuracy",
        cv=SFS_CV_FOLDS,
    )
    selector.fit(X, y)
    selected = [name for name, keep in zip(X.columns, selector.get_support()) if keep]
    log(f"Selected features: {selected}")
    return selected


@dataclass
class EvaluationResult:
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
    X: pd.DataFrame, y: pd.Series, test_size: float, labels: dict[int, str], log: Log
) -> EvaluationResult:
    """Train/test evaluation. Feature selection sees only the training split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    selected = forward_select(X_train, y_train, log)
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
    return EvaluationResult(
        selected=selected,
        accuracy=accuracy_score(y_test, y_pred),
        report=report,
        model=model,
        X_test=X_test[selected],
        y_test=y_test,
    )


def build_roc_figure(result: EvaluationResult, labels: dict[int, str]) -> Figure:
    """ROC curve for binary problems; one-vs-rest plus micro-average otherwise."""
    model, X_test, y_test = result.model, result.X_test, result.y_test
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


def train_final_model(
    X: pd.DataFrame, y: pd.Series, dataset_path: str, log: Log
) -> dict:
    selected = forward_select(X, y, log)
    model = build_pipeline().fit(X[selected], y)
    return {
        "model": model,
        "features": selected,
        "classes": sorted(y.unique().tolist()),
        "trained_on": dataset_path,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_path_for(dataset: Path) -> Path:
    """Each dataset gets its own saved model, so models never mix across datasets."""
    return MODELS_DIR / f"{dataset.stem}.joblib"


def save_model_bundle(bundle: dict, dataset: Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path_for(dataset))


def load_model_for_dataset(dataset: Path) -> dict:
    """Return the model trained from this exact dataset file, or explain why there isn't one."""
    path = model_path_for(dataset)
    if not path.is_file():
        raise FileNotFoundError(
            f"No model has been trained for {dataset.name} yet. "
            "Run B (Select features & train)."
        )
    bundle = joblib.load(path)
    if bundle.get("dataset_sha256") != file_sha256(dataset):
        raise ValueError(
            f"The saved model for {dataset.name} was trained on an earlier version of "
            "that file. Run B (Select features & train) again."
        )
    return bundle


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
    bundle: dict, csv_files: list[Path], labels: dict[int, str], log: Log
) -> ClassificationResult:
    """Predict every ROI and tabulate, per image, the total and each class's count and %."""
    features = bundle["features"]
    classes = bundle["classes"]
    images = [csv.stem for csv in csv_files]

    rois = load_roi_tables(csv_files, features, log)
    if rois.empty:
        raise RuntimeError("No ROIs available for classification.")
    predictions = pd.Series(bundle["model"].predict(rois[features]), index=rois.index)

    counts = pd.crosstab(rois["source"], predictions).reindex(
        index=images, columns=classes, fill_value=0
    )
    totals = counts.sum(axis=1)

    rows = []
    for image in images:
        total = int(totals[image])
        row: dict[str, object] = {"image": image, "total_rois": total}
        for cls in classes:
            n = int(counts.loc[image, cls])
            row[f"{labels[cls]} count"] = n
            row[f"{labels[cls]} %"] = round(100 * n / total, 2) if total else 0.0
        rows.append(row)

    return ClassificationResult(
        per_image=pd.DataFrame(rows),
        features=features,
        n_images=len(images),
        total=int(totals.sum()),
    )


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _require_file(text: str, what: str) -> Path:
    path = Path(text.strip().strip('"'))
    if not path.is_file():
        raise ValueError(f"Select a valid {what}.")
    return path


def _require_dir(text: str, what: str) -> Path:
    path = Path(text.strip().strip('"'))
    if not path.is_dir():
        raise ValueError(f"Select a valid {what}.")
    return path


def _tree_with_scrollbars(parent: tk.Misc) -> ttk.Treeview:
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.BOTH, expand=True)
    tree = ttk.Treeview(frame, show="headings")
    yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return tree


def _fill_tree(tree: ttk.Treeview, df: pd.DataFrame) -> None:
    tree.delete(*tree.get_children())
    columns = [str(c) for c in df.columns]
    tree["columns"] = columns
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=110, anchor=tk.CENTER, stretch=False)
    for _, row in df.iterrows():
        tree.insert("", tk.END, values=[str(v) for v in row])


class ROIClassifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ROI Classifier")
        self.minsize(960, 680)

        self.state_data = AppState.load()
        MACROS_DIR.mkdir(parents=True, exist_ok=True)
        self.events: queue.Queue = queue.Queue()
        self.busy = False

        self.dataset_var = tk.StringVar(value=self.state_data.dataset_path)
        self.images_var = tk.StringVar(value=self.state_data.images_dir)
        self.test_size_var = tk.StringVar(value=str(self.state_data.test_size))
        self.status_var = tk.StringVar(value="")

        self._build_widgets()
        self.dataset_var.trace_add("write", lambda *_: self._refresh_model_status())
        self._refresh_model_status()
        self.after(POLL_INTERVAL_MS, self._poll_events)

    # -- layout ------------------------------------------------------------

    def _build_widgets(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        inputs = ttk.LabelFrame(self, text="Inputs", padding=10)
        inputs.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        inputs.columnconfigure(1, weight=1)
        self._path_row(
            inputs, 0, "Model dataset (CSV)", self.dataset_var, self._browse_dataset
        )
        self._path_row(inputs, 1, "Image folder", self.images_var, self._browse_images)
        ttk.Label(inputs, text="Test fraction").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(inputs, textvariable=self.test_size_var, width=8).grid(
            row=2, column=1, sticky="w", pady=(6, 0)
        )

        actions = ttk.Frame(self, padding=(10, 4))
        actions.grid(row=1, column=0, sticky="ew")
        actions.columnconfigure(3, weight=1)
        self.run_buttons = [
            ttk.Button(actions, text="A. Test accuracy", command=self._on_accuracy),
            ttk.Button(
                actions,
                text="B. Select features & train",
                command=self._on_select_features,
            ),
            ttk.Button(actions, text="C. Classify images", command=self._on_classify),
        ]
        for col, button in enumerate(self.run_buttons):
            button.grid(row=0, column=col, padx=(0, 8))
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=160)
        self.progress.grid(row=0, column=3, sticky="e")

        ttk.Label(
            self,
            textvariable=self.status_var,
            padding=(10, 2),
            wraplength=900,
            justify="left",
        ).grid(row=2, column=0, sticky="ew")

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)

        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="Log")
        self.log_text = ScrolledText(log_tab, wrap="word", state="disabled", height=12)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.roc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.roc_frame, text="ROC curve")
        ttk.Label(
            self.roc_frame,
            text="Run A (Test accuracy) to see the ROC curve here.",
            padding=20,
        ).pack()

        self.counts_tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(self.counts_tab, text="ROI counts per image")
        ttk.Label(
            self.counts_tab,
            text="Per image: total ROIs, then each class's count and % of that image's total.",
        ).pack(anchor="w")
        self.per_image_tree = _tree_with_scrollbars(self.counts_tab)

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        browse: Callable[[], None],
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=2
        )
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(parent, text="Browse...", command=browse).grid(
            row=row, column=2, padx=(8, 0), pady=2
        )

    # -- browsing and input handling ----------------------------------------

    def _browse_dataset(self) -> None:
        path = filedialog.askopenfilename(
            title="Select the labelled model dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.dataset_var.set(path)
            self._save_inputs()

    def _browse_images(self) -> None:
        path = filedialog.askdirectory(title="Choose the folder of images to analyze")
        if path:
            self.images_var.set(path)
            self._save_inputs()

    def _save_inputs(self) -> None:
        self.state_data.dataset_path = self.dataset_var.get()
        self.state_data.images_dir = self.images_var.get()
        try:
            self.state_data.test_size = self._parse_test_size()
        except ValueError:
            pass
        self.state_data.save()

    def _parse_test_size(self) -> float:
        try:
            value = float(self.test_size_var.get().strip())
        except ValueError:
            raise ValueError(
                "Test fraction must be a number between 0 and 1."
            ) from None
        if not 0 < value < 1:
            raise ValueError("Test fraction must be between 0 and 1.")
        return value

    # -- background task plumbing -------------------------------------------

    def _start_task(
        self,
        name: str,
        job: Callable[[Log], object],
        on_success: Callable[[object], None],
    ) -> None:
        if self.busy:
            return
        self._set_busy(True)
        self._log(f"--- {name} started ---")

        def log(message: str) -> None:
            self.events.put(("log", message))

        def worker() -> None:
            try:
                result = job(log)
                self.events.put(("success", on_success, result))
            except Exception as exc:
                self.events.put(("log", traceback.format_exc()))
                self.events.put(("error", name, exc))
            finally:
                self.events.put(("done",))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_events(self) -> None:
        """Handle a bounded batch of worker events, then yield back to Tk."""
        pending: list[str] = []
        try:
            for _ in range(MAX_EVENTS_PER_TICK):
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    pending.append(event[1])
                    continue
                self._write_log(pending)
                pending = []
                if kind == "success":
                    event[1](event[2])
                elif kind == "error":
                    self._write_log([f"ERROR: {event[2]}"])
                    messagebox.showerror(event[1], str(event[2]))
                elif kind == "done":
                    self._set_busy(False)
        except queue.Empty:
            pass
        except Exception as exc:  # a UI callback failed; keep polling
            self._write_log([f"ERROR in UI update: {exc}"])
        finally:
            self._write_log(pending)
        self.after(POLL_INTERVAL_MS, self._poll_events)

    def _set_busy(self, flag: bool) -> None:
        self.busy = flag
        for button in self.run_buttons:
            button.state(["disabled"] if flag else ["!disabled"])
        if flag:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _log(self, message: str) -> None:
        self._write_log([message])

    def _write_log(self, messages: list[str]) -> None:
        """Append several messages in one widget update (much cheaper than one per line)."""
        if not messages:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        text = "".join(
            f"[{stamp}] {line.rstrip()}\n"
            for message in messages
            for line in (message.splitlines() or [""])
        )
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    # -- actions ------------------------------------------------------------

    def _on_accuracy(self) -> None:
        try:
            dataset = _require_file(self.dataset_var.get(), "model dataset CSV")
            test_size = self._parse_test_size()
        except ValueError as exc:
            messagebox.showwarning("Missing input", str(exc))
            return
        self._save_inputs()

        def job(log: Log):
            X, y, _ = load_model_dataset(dataset, log)
            labels = load_class_labels(sorted(y.unique().tolist()), log)
            result = evaluate_split(X, y, test_size, labels, log)
            log(result.summary)
            stamp = _timestamp()
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_path = REPORTS_DIR / f"classification_report_{stamp}.txt"
            report_path.write_text(result.summary, encoding="utf-8")
            fig = build_roc_figure(result, labels)
            fig.savefig(REPORTS_DIR / f"roc_{stamp}.png", dpi=150)
            log(f"Report and ROC curve saved to {REPORTS_DIR}")
            return fig

        self._start_task("Accuracy test", job, self._show_accuracy)

    def _show_accuracy(self, fig: Figure) -> None:
        for child in self.roc_frame.winfo_children():
            child.destroy()
        toolbar_host = ttk.Frame(self.roc_frame)
        toolbar_host.pack(side=tk.TOP, fill=tk.X)
        canvas = FigureCanvasTkAgg(fig, master=self.roc_frame)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_host, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        self.notebook.select(self.roc_frame)

    def _on_select_features(self) -> None:
        try:
            dataset = _require_file(self.dataset_var.get(), "model dataset CSV")
        except ValueError as exc:
            messagebox.showwarning("Missing input", str(exc))
            return
        self._save_inputs()

        def job(log: Log):
            X, y, _ = load_model_dataset(dataset, log)
            load_class_labels(
                sorted(y.unique().tolist()), log
            )  # validate before training
            bundle = train_final_model(X, y, str(dataset), log)
            bundle["dataset_sha256"] = file_sha256(dataset)
            save_model_bundle(bundle, dataset)
            log(f"Model saved to {model_path_for(dataset)}")
            return bundle

        self._start_task(
            "Feature selection", job, lambda _bundle: self._refresh_model_status()
        )

    def _refresh_model_status(self) -> None:
        """Show the saved model for the dataset currently selected, if there is one."""
        dataset = Path(self.dataset_var.get().strip().strip('"'))
        if not dataset.is_file():
            self.status_var.set("Select a model dataset to see its saved model.")
            return
        try:
            bundle = load_model_for_dataset(dataset)
        except Exception as exc:  # missing, out-of-date or unreadable model
            self.status_var.set(str(exc))
            return
        self.status_var.set(
            f"Model for {dataset.name}: {len(bundle['features'])} features "
            f"{bundle['features']} | classes {bundle['classes']} | "
            f"trained {bundle['trained_at']}"
        )

    def _on_classify(self) -> None:
        try:
            dataset = _require_file(self.dataset_var.get(), "model dataset CSV")
            images = _require_dir(self.images_var.get(), "image folder")
            bundle = load_model_for_dataset(dataset)
        except (ValueError, FileNotFoundError) as exc:
            messagebox.showwarning("Model not ready", str(exc))
            return
        self._save_inputs()

        def job(log: Log):
            labels = load_class_labels(bundle["classes"], log)
            run_dir = RUNS_DIR / _timestamp()
            csv_files = run_imagej_batch(images, run_dir, log)
            if not csv_files:
                raise RuntimeError("The batch macro produced no CSV files.")
            result = classify_rois(bundle, csv_files, labels, log)
            result.per_image.to_csv(run_dir / "per_image_counts.csv", index=False)
            (run_dir / "classification_report.txt").write_text(
                result.summary, encoding="utf-8"
            )
            log(result.summary)
            log(f"Results saved to {run_dir}")
            return result

        self._start_task("Classification", job, self._show_classification)

    def _show_classification(self, result: ClassificationResult) -> None:
        _fill_tree(self.per_image_tree, result.per_image)
        self.notebook.select(self.counts_tab)


if __name__ == "__main__":
    ROIClassifierApp().mainloop()
