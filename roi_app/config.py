"""Paths and settings shared by the whole app.

Everything the app creates or downloads lives under PROJECT_DIR: the macros
folder, the optional class_labels.txt and the data/ folder.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent  # the folder holding roi_classifier_gui.py
MACROS_DIR = PROJECT_DIR / "macros"  # the custom ImageJ macros
CLASS_LABELS_FILE = PROJECT_DIR / "class_labels.txt"

DATA_DIR = PROJECT_DIR / "data"  # everything the app generates
IMAGEJ_DIR = DATA_DIR / "ImageJ"
IMAGEJ_DOWNLOAD_ZIP = DATA_DIR / "imagej_download.zip"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
RUNS_DIR = DATA_DIR / "runs"
STATE_FILE = DATA_DIR / "state.json"

BATCH_MACRO_FILE = MACROS_DIR / "BatchProcess.ijm"
MAIN_MACRO_FILE = MACROS_DIR / "AnalyzeSporesAndGermlings.ijm"

# ImageJ 1.x distributions (version 1.54). Platform bundles include a Java runtime.
IMAGEJ_URLS = {
    "windows": "https://wsr.imagej.net/distros/win/ij154-win-java8.zip",
    "macos-intel": "https://wsr.imagej.net/distros/osx/ij154-osx-java8.zip",
    "macos-arm": "https://wsr.imagej.net/distros/osx/ij154-osx-arm-java13.zip",
    "linux": "https://wsr.imagej.net/distros/linux/ij154-linux64-java8.zip",
    "cross-platform": "https://wsr.imagej.net/distros/cross-platform/ij154.zip",
}
DOWNLOAD_TIMEOUT_S = 120
IMAGEJ_TIMEOUT_S = 4 * 60 * 60  # a batch run is stopped after this many seconds

# JVM properties the ImageJ macros read with call("java.lang.System.getProperty", ...)
RUN_DIR_PROPERTY = "roi.runDir"  # run folder the macros write into
MAIN_MACRO_PROPERTY = "roi.mainMacro"  # full path to the main macro
SIZE_MIN_PROPERTY = "roi.sizeMin"  # Analyze Particles minimum size
SIZE_MAX_PROPERTY = "roi.sizeMax"  # Analyze Particles maximum size

DEFAULT_TARGET_COLUMN = "class"  # column holding the integer class labels
DEFAULT_SIZE_MIN = 50.0
DEFAULT_SIZE_MAX = 500.0
DEFAULT_N_FEATURES = 8
DEFAULT_TEST_SIZE = 0.25

# Only these Analyze Particles columns are used as features; others are ignored.
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

MIN_ROIS_PER_CLASS = 2  # each class needs this many labelled ROIs to be modelled
RANDOM_STATE = 42
SFS_CV_FOLDS = 5

POLL_INTERVAL_MS = 100  # how often the window checks for worker events
MAX_EVENTS_PER_TICK = 200  # cap per check so a chatty job can't freeze the window
DATASET_REFRESH_DELAY_MS = 400  # wait this long after typing before re-reading a dataset

# ImageJ output lines shown in the app log. Everything is still kept in imagej.log.
LOG_KEEP_PATTERN = re.compile(
    r"error|exception|warning|failed|not found|cannot|unable|^Processing ",
    re.IGNORECASE,
)
