"""ImageJ: download it on first use, then run the batch macro over an image folder.

Batch contract (see the macros folder):
    java -Droi.runDir=<run_dir> -Droi.mainMacro=<main.ijm> -Droi.sizeMin=.. -Droi.sizeMax=..
         -jar ij.jar -batch BatchProcess.ijm "<image_dir>"
The batch macro runs the main macro on each image, which writes CSV tables into run_dir.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.request
import zipfile
from collections import deque
from pathlib import Path

from . import config
from .util import Log, timestamp


class ImageJError(RuntimeError):
    """ImageJ could not be installed, or did not finish successfully."""


class JobCancelled(Exception):
    """The user cancelled the running job."""


def get_imagej_url() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        return config.IMAGEJ_URLS["windows"]
    if system == "Darwin":
        is_arm = machine in ("arm64", "aarch64")
        return config.IMAGEJ_URLS["macos-arm" if is_arm else "macos-intel"]
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return config.IMAGEJ_URLS["linux"]
    return config.IMAGEJ_URLS["cross-platform"]


def find_ij_jar() -> Path | None:
    if not config.IMAGEJ_DIR.exists():
        return None
    return next(config.IMAGEJ_DIR.rglob("ij.jar"), None)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=config.DOWNLOAD_TIMEOUT_S) as response:
        with dest.open("wb") as out:
            shutil.copyfileobj(response, out)


def _extract_zip_safely(zip_path: Path, dest: Path) -> None:
    """Extract a zip, refusing any member that would land outside dest."""
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (dest / member.filename).resolve()
            if target != dest and dest not in target.parents:
                raise ImageJError(f"Unsafe path in ImageJ archive: {member.filename}")
        archive.extractall(dest)
    if os.name != "nt":
        # zipfile does not keep execute bits; the bundled java launcher needs them.
        for path in dest.rglob("*"):
            if path.is_file() and path.parent.name == "bin":
                path.chmod(path.stat().st_mode | 0o111)


def ensure_imagej(log: Log) -> Path:
    """Return the path to ij.jar, downloading and installing ImageJ first if needed."""
    jar = find_ij_jar()
    if jar:
        return jar

    url = get_imagej_url()
    log(f"ImageJ not found. Downloading from {url}")
    zip_path = config.IMAGEJ_DOWNLOAD_ZIP
    try:
        _download(url, zip_path)
        log("Extracting ImageJ ...")
        _extract_zip_safely(zip_path, config.IMAGEJ_DIR)
    except zipfile.BadZipFile:
        raise ImageJError("The ImageJ download is not a valid zip file. Try again.") from None
    finally:
        zip_path.unlink(missing_ok=True)

    jar = find_ij_jar()
    if jar is None:
        raise ImageJError("ImageJ was installed but ij.jar was not found in the archive.")
    log(f"ImageJ installed at {jar.parent}")
    return jar


def find_java() -> str:
    """Prefer the Java runtime bundled with ImageJ, then fall back to the system Java."""
    exe_name = "java.exe" if os.name == "nt" else "java"
    if config.IMAGEJ_DIR.exists():
        for candidate in config.IMAGEJ_DIR.rglob(exe_name):
            if candidate.parent.name == "bin":
                return str(candidate)
    system_java = shutil.which("java")
    if system_java:
        return system_java
    raise ImageJError(
        "No Java runtime found. Install Java 8 or newer, or use a platform ImageJ bundle."
    )


def new_run_dir() -> Path:
    """A fresh run folder under data/runs, never reusing an existing one."""
    base = config.RUNS_DIR / timestamp()
    run_dir, n = base, 2
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}_{n}")
        n += 1
    return run_dir


def build_batch_command(
    java: str,
    ij_jar: Path,
    run_dir: Path,
    size_min: float,
    size_max: float,
    input_dir: Path,
) -> list[str]:
    return [
        java,
        f"-D{config.RUN_DIR_PROPERTY}={run_dir}",
        f"-D{config.MAIN_MACRO_PROPERTY}={config.MAIN_MACRO_FILE}",
        f"-D{config.SIZE_MIN_PROPERTY}={size_min:g}",
        f"-D{config.SIZE_MAX_PROPERTY}={size_max:g}",
        "-jar",
        str(ij_jar),
        "-batch",
        str(config.BATCH_MACRO_FILE),
        str(input_dir),  # the batch macro's single argument: the image folder
    ]


def _watch(
    process: subprocess.Popen,
    cancel: threading.Event,
    deadline: float,
    finished: threading.Event,
    stopped: list[str],
) -> None:
    """Kill ImageJ if the user cancels or the time limit passes."""
    while not finished.wait(0.25):
        if cancel.is_set():
            stopped.append("cancelled")
        elif time.monotonic() > deadline:
            stopped.append("timed out")
        else:
            continue
        process.kill()
        return


def _run_process(
    command: list[str],
    cwd: Path,
    log_path: Path,
    log: Log,
    cancel: threading.Event,
    timeout_s: float,
) -> list[str]:
    """
    Run ImageJ. Useful output goes to the app log; everything goes to log_path.
    Returns the last few output lines. Raises JobCancelled, or ImageJError on
    timeout or a non-zero exit.
    """
    recent: deque[str] = deque(maxlen=20)
    stopped: list[str] = []
    finished = threading.Event()
    with log_path.open("w", encoding="utf-8", errors="replace") as ij_log, subprocess.Popen(
        command,
        cwd=cwd,  # keeps ImageJ's preferences and temp files inside data/
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",  # stray bytes in ImageJ output must not stop the run
        bufsize=1,
    ) as process:
        watcher = threading.Thread(
            target=_watch,
            args=(process, cancel, time.monotonic() + timeout_s, finished, stopped),
            daemon=True,
        )
        watcher.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                text = line.rstrip()
                ij_log.write(line)
                recent.append(text)
                if config.LOG_KEEP_PATTERN.search(text):
                    log(f"[ImageJ] {text}")
            returncode = process.wait()
        finally:
            finished.set()
            if process.poll() is None:
                process.kill()  # never leave ImageJ running after an error
                process.wait()

    if "cancelled" in stopped:
        raise JobCancelled()
    if stopped:
        raise ImageJError(
            f"ImageJ did not finish within {timeout_s:g} seconds and was stopped."
        )
    if returncode != 0:
        raise ImageJError(
            f"ImageJ exited with code {returncode}. Last output:\n" + "\n".join(recent)
        )
    return list(recent)


def run_imagej_batch(
    input_dir: Path,
    run_dir: Path,
    size_min: float,
    size_max: float,
    log: Log,
    cancel: threading.Event,
    timeout_s: float = config.IMAGEJ_TIMEOUT_S,
) -> list[Path]:
    """Run the batch macro over input_dir. Returns the CSV files written into run_dir."""
    for macro in (config.BATCH_MACRO_FILE, config.MAIN_MACRO_FILE):
        if not macro.is_file():
            raise ImageJError(f"Macro not found: {macro}")
    ij_jar = ensure_imagej(log)
    java = find_java()
    run_dir.mkdir(parents=True, exist_ok=True)

    command = build_batch_command(java, ij_jar, run_dir, size_min, size_max, input_dir)
    log(f"Batch macro: {config.BATCH_MACRO_FILE}")
    log(f"Image folder: {input_dir}")
    log("Starting ImageJ ...")
    tail = _run_process(
        command, config.IMAGEJ_DIR, run_dir / "imagej.log", log, cancel, timeout_s
    )

    csv_files = sorted(run_dir.rglob("*.csv"))
    if not csv_files:
        raise ImageJError(
            f"ImageJ finished but wrote no CSV files to {run_dir}. "
            "Last output:\n" + "\n".join(tail)
        )
    log(f"ImageJ produced {len(csv_files)} CSV file(s).")
    return csv_files
