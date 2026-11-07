"""The main window.

Buttons
-------
A. Test accuracy     train/test split, forward selection on the training split,
                     classification report, ROC curve.
B. Select features   forward selection on the full dataset; trains and saves the model.
C. Classify images   ImageJ batch run over the image folder, then per-image counts.
                     Can be cancelled while ImageJ runs.

Long jobs run on a background thread and report back through a queue that the window
checks every POLL_INTERVAL_MS, so the window never blocks on ImageJ or training.
"""

from __future__ import annotations

import queue
import threading
import traceback
from contextlib import suppress
from datetime import datetime
from typing import Callable

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from . import config, store
from .datasets import load_model_dataset
from .dialogs import ClassLabelsDialog
from .imagej import JobCancelled, new_run_dir, run_imagej_batch
from .labels import label_source_text, resolve_class_labels
from .modelling import ClassificationResult, build_roc_figure, classify_rois, evaluate_split, train_model
from .settings import (
    InputError,
    clean_path,
    parse_dataset,
    parse_images,
    parse_n_features,
    parse_size_bounds,
    parse_target,
    parse_test_size,
)
from .state import AppState
from .store import ModelError
from .tables import SortableTable
from .util import Log, silent, timestamp

# Problems the user can act on. These are shown without a traceback in the log.
EXPECTED_ERRORS = (ValueError, RuntimeError, OSError)


def _field(parent: tk.Misc, row: int, text: str, widget: tk.Misc) -> None:
    """A label and a widget on one grid row of the inputs frame."""
    ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=(6, 0))
    widget.grid(row=row, column=1, sticky="w", pady=(6, 0))


class ROIClassifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ROI Classifier")
        self.minsize(960, 680)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.state_data = AppState.load()
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.cancel_event: threading.Event | None = None
        self.class_names: dict[int, str] = dict(self.state_data.class_names)
        self.dataset_classes: list[int] | None = None
        self._refresh_after_id: str | None = None

        s = self.state_data
        self.dataset_var = tk.StringVar(value=s.dataset_path)
        self.images_var = tk.StringVar(value=s.images_dir)
        self.test_size_var = tk.StringVar(value=str(s.test_size))
        self.target_var = tk.StringVar(value=s.target_column)
        self.size_min_var = tk.StringVar(value=f"{s.size_min:g}")
        self.size_max_var = tk.StringVar(value=f"{s.size_max:g}")
        self.n_features_var = tk.StringVar(value=str(s.n_features))
        self.dataset_info_var = tk.StringVar(value="")
        self.class_summary_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")

        self._build_widgets()
        self.dataset_var.trace_add("write", lambda *_: self._schedule_dataset_refresh())
        self.target_var.trace_add("write", lambda *_: self._schedule_dataset_refresh())
        self._refresh_dataset_info()
        self.after(config.POLL_INTERVAL_MS, self._poll_events)

    # -- layout -------------------------------------------------------------

    def _build_widgets(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        self._build_inputs()
        self._build_actions()
        ttk.Label(
            self,
            textvariable=self.status_var,
            padding=(10, 2),
            wraplength=900,
            justify="left",
        ).grid(row=2, column=0, sticky="ew")
        self._build_notebook()

    def _build_inputs(self) -> None:
        inputs = ttk.LabelFrame(self, text="Inputs", padding=10)
        inputs.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        inputs.columnconfigure(1, weight=1)
        self._path_row(inputs, 0, "Model dataset (CSV)", self.dataset_var, self._browse_dataset)
        self._path_row(inputs, 1, "Image folder", self.images_var, self._browse_images)
        _field(inputs, 2, "Test fraction", ttk.Entry(inputs, textvariable=self.test_size_var, width=8))
        _field(inputs, 3, "Target column", ttk.Entry(inputs, textvariable=self.target_var, width=20))

        size_frame = ttk.Frame(inputs)
        ttk.Entry(size_frame, textvariable=self.size_min_var, width=8).pack(side=tk.LEFT)
        ttk.Label(size_frame, text="  to  ").pack(side=tk.LEFT)
        ttk.Entry(size_frame, textvariable=self.size_max_var, width=8).pack(side=tk.LEFT)
        _field(inputs, 4, "Particle size min / max", size_frame)

        _field(inputs, 5, "Features to select", ttk.Entry(inputs, textvariable=self.n_features_var, width=8))

        labels_frame = ttk.Frame(inputs)
        self.labels_button = ttk.Button(
            labels_frame, text="Edit class labels...", command=self._open_class_labels_dialog
        )
        self.labels_button.pack(side=tk.LEFT)
        ttk.Label(labels_frame, textvariable=self.class_summary_var).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        _field(inputs, 6, "Class labels", labels_frame)

        ttk.Label(inputs, textvariable=self.dataset_info_var).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

    def _build_actions(self) -> None:
        actions = ttk.Frame(self, padding=(10, 4))
        actions.grid(row=1, column=0, sticky="ew")
        actions.columnconfigure(4, weight=1)
        self.run_buttons = [
            ttk.Button(actions, text="A. Test accuracy", command=self._on_accuracy),
            ttk.Button(
                actions,
                text="B. Select features & (re)train",
                command=self._on_select_features,
            ),
            ttk.Button(actions, text="C. Classify images", command=self._on_classify),
        ]
        for col, button in enumerate(self.run_buttons):
            button.grid(row=0, column=col, padx=(0, 8))
        self.cancel_button = ttk.Button(actions, text="Cancel run", command=self._cancel_job)
        self.cancel_button.grid(row=0, column=3, padx=(0, 8))
        self.cancel_button.state(["disabled"])
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=160)
        self.progress.grid(row=0, column=4, sticky="e")

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)

        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="Log")
        ttk.Button(log_tab, text="Clear log", command=self._clear_log).pack(
            anchor="e", padx=4, pady=(4, 0)
        )
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
            text=(
                "Per image: total ROIs, then each class's count and % of that total. "
                "Negative classes are excluded from totals and %, shown as '(excluded)'. "
                "Click a heading to sort."
            ),
        ).pack(anchor="w")
        self.per_image_table = SortableTable(self.counts_tab)
        self.per_image_table.pack(fill=tk.BOTH, expand=True)

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        browse: Callable[[], None],
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=2)
        ttk.Button(parent, text="Browse...", command=browse).grid(
            row=row, column=2, padx=(8, 0), pady=2
        )

    # -- inputs -------------------------------------------------------------

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
        """Remember the current inputs. Invalid numbers keep their last good value."""
        s = self.state_data
        s.dataset_path = self.dataset_var.get()
        s.images_dir = self.images_var.get()
        s.target_column = self.target_var.get().strip() or config.DEFAULT_TARGET_COLUMN
        s.class_names = dict(self.class_names)
        with suppress(InputError):
            s.test_size = parse_test_size(self.test_size_var.get())
        with suppress(InputError):
            s.size_min, s.size_max = parse_size_bounds(
                self.size_min_var.get(), self.size_max_var.get()
            )
        with suppress(InputError):
            s.n_features = parse_n_features(self.n_features_var.get())
        s.save()

    # -- dataset and class names ---------------------------------------------

    def _schedule_dataset_refresh(self) -> None:
        """Re-read the dataset once typing pauses, rather than on every keystroke."""
        if self._refresh_after_id is not None:
            self.after_cancel(self._refresh_after_id)
        self._refresh_after_id = self.after(
            config.DATASET_REFRESH_DELAY_MS, self._refresh_dataset_info
        )

    def _refresh_dataset_info(self) -> None:
        self._refresh_after_id = None
        target = self.target_var.get().strip() or config.DEFAULT_TARGET_COLUMN
        path = clean_path(self.dataset_var.get())
        self.dataset_classes = None
        if not path.is_file():
            info = "Select a model dataset to load its classes."
        else:
            try:
                dataset = load_model_dataset(path, target, silent)
            except (ValueError, OSError) as exc:
                info = f"Dataset not loaded: {exc}"
            else:
                self.dataset_classes = dataset.classes
                info = (
                    f"Loaded {path.name}: {len(dataset.X)} labelled ROIs, "
                    f"classes {dataset.classes}"
                )
        self.dataset_info_var.set(info)
        self.labels_button.state(["!disabled"] if self.dataset_classes else ["disabled"])
        self._update_class_summary()
        self._refresh_model_status()

    def _update_class_summary(self) -> None:
        self.class_summary_var.set(label_source_text(self.class_names))

    def _open_class_labels_dialog(self) -> None:
        if not self.dataset_classes:
            messagebox.showwarning("Load a dataset", "Load a model dataset first.")
            return
        ClassLabelsDialog(self, self.dataset_classes, self.class_names, self._set_class_names)

    def _set_class_names(self, names: dict[int, str]) -> None:
        self.class_names = names
        self._save_inputs()
        self._update_class_summary()

    def _refresh_model_status(self) -> None:
        """Show the saved model for the dataset currently selected, if there is one."""
        path = clean_path(self.dataset_var.get())
        if not path.is_file():
            self.status_var.set("Select a model dataset to see its saved model.")
            return
        try:
            summary = store.read_model_summary(path)
        except ModelError as exc:
            self.status_var.set(str(exc))
            return
        self.status_var.set(
            f"Model for {path.name}: {len(summary['features'])} features "
            f"{summary['features']} | classes {summary['classes']} | "
            f"trained {summary['trained_at']}"
        )

    # -- background jobs ----------------------------------------------------

    def _start_job(
        self,
        name: str,
        job: Callable[[Log, threading.Event], object],
        on_success: Callable[[object], None],
        cancellable: bool = False,
    ) -> None:
        if self.busy:
            return
        cancel = threading.Event()
        self.cancel_event = cancel
        self._set_busy(True, cancellable)
        self._log(f"--- {name} started ---")

        def log(message: str) -> None:
            self.events.put(("log", message))

        def worker() -> None:
            try:
                result = job(log, cancel)
            except JobCancelled:
                self.events.put(("log", f"{name} cancelled."))
            except Exception as exc:
                if not isinstance(exc, EXPECTED_ERRORS):
                    self.events.put(("log", traceback.format_exc()))
                self.events.put(("error", name, exc))
            else:
                self.events.put(("success", on_success, result))
            finally:
                self.events.put(("done",))

        threading.Thread(target=worker, daemon=True).start()

    def _cancel_job(self) -> None:
        if self.cancel_event is not None:
            self.cancel_event.set()
            self._log("Cancel requested ...")

    def _poll_events(self) -> None:
        """Handle a bounded batch of worker events, then yield back to Tk."""
        pending: list[str] = []
        for _ in range(config.MAX_EVENTS_PER_TICK):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "log":
                pending.append(event[1])
                continue
            self._write_log(pending)
            pending = []
            self._handle_event(event)
        self._write_log(pending)
        self.after(config.POLL_INTERVAL_MS, self._poll_events)

    def _handle_event(self, event: tuple) -> None:
        kind = event[0]
        if kind == "success":
            _, callback, result = event
            try:
                callback(result)
            except Exception as exc:  # a window update failed; the app keeps running
                self._write_log([f"ERROR in UI update: {exc}"])
        elif kind == "error":
            _, title, exc = event
            self._write_log([f"ERROR: {exc}"])
            messagebox.showerror(title, str(exc))
        elif kind == "done":
            self.cancel_event = None
            self._set_busy(False)

    def _set_busy(self, flag: bool, cancellable: bool = False) -> None:
        self.busy = flag
        for button in self.run_buttons:
            button.state(["disabled"] if flag else ["!disabled"])
        self.cancel_button.state(["!disabled"] if flag and cancellable else ["disabled"])
        if flag:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _on_close(self) -> None:
        if self.cancel_event is not None:
            self.cancel_event.set()  # stops ImageJ so it isn't left running
        self.destroy()

    # -- log ----------------------------------------------------------------

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _log(self, message: str) -> None:
        self._write_log([message])

    def _write_log(self, messages: list[str]) -> None:
        """Append several messages in one widget update."""
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
            dataset_path = parse_dataset(self.dataset_var.get())
            target = parse_target(self.target_var.get())
            test_size = parse_test_size(self.test_size_var.get())
            n_features = parse_n_features(self.n_features_var.get())
        except InputError as exc:
            messagebox.showwarning("Missing input", str(exc))
            return
        self._save_inputs()
        names = dict(self.class_names)

        def job(log: Log, _cancel: threading.Event) -> Figure:
            dataset = load_model_dataset(dataset_path, target, log)
            class_labels = resolve_class_labels(dataset.classes, names, log)
            evaluation = evaluate_split(dataset, test_size, n_features, class_labels, log)
            log(evaluation.summary)
            stamp = timestamp()
            config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            (config.REPORTS_DIR / f"classification_report_{stamp}.txt").write_text(
                evaluation.summary, encoding="utf-8"
            )
            figure = build_roc_figure(evaluation, class_labels)
            figure.savefig(config.REPORTS_DIR / f"roc_{stamp}.png", dpi=150)
            log(f"Report and ROC curve saved to {config.REPORTS_DIR}")
            return figure

        self._start_job("Accuracy test", job, self._show_roc)

    def _show_roc(self, figure: Figure) -> None:
        for child in self.roc_frame.winfo_children():
            child.destroy()
        toolbar_host = ttk.Frame(self.roc_frame)
        toolbar_host.pack(side=tk.TOP, fill=tk.X)
        canvas = FigureCanvasTkAgg(figure, master=self.roc_frame)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_host, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        self.notebook.select(self.roc_frame)

    def _on_select_features(self) -> None:
        try:
            dataset_path = parse_dataset(self.dataset_var.get())
            target = parse_target(self.target_var.get())
            n_features = parse_n_features(self.n_features_var.get())
        except InputError as exc:
            messagebox.showwarning("Missing input", str(exc))
            return
        self._save_inputs()
        names = dict(self.class_names)

        def job(log: Log, _cancel: threading.Event) -> None:
            dataset = load_model_dataset(dataset_path, target, log)
            resolve_class_labels(dataset.classes, names, log)  # check names before training
            if store.model_file(dataset_path).is_file():
                log("Replacing the existing model for this dataset.")
            trained = train_model(dataset, n_features, log)
            log(f"Model saved to {store.save_model(trained, dataset)}")

        self._start_job(
            "Feature selection / model training",
            job,
            lambda _result: self._refresh_model_status(),
        )

    def _on_classify(self) -> None:
        try:
            dataset_path = parse_dataset(self.dataset_var.get())
            images = parse_images(self.images_var.get())
            size_min, size_max = parse_size_bounds(
                self.size_min_var.get(), self.size_max_var.get()
            )
            store.read_model_summary(dataset_path)  # quick check before starting ImageJ
        except (InputError, ModelError) as exc:
            messagebox.showwarning("Not ready", str(exc))
            return
        self._save_inputs()
        names = dict(self.class_names)

        def job(log: Log, cancel: threading.Event) -> ClassificationResult:
            trained = store.load_model(dataset_path)
            class_labels = resolve_class_labels(trained.classes, names, log)
            run_dir = new_run_dir()
            csv_files = run_imagej_batch(
                images, run_dir, size_min, size_max, log, cancel
            )
            result = classify_rois(trained, csv_files, class_labels, log)
            result.per_image.to_csv(run_dir / "per_image_counts.csv", index=False)
            (run_dir / "classification_report.txt").write_text(
                result.summary, encoding="utf-8"
            )
            log(result.summary)
            log(f"Results saved to {run_dir}")
            return result

        self._start_job("Classification", job, self._show_classification, cancellable=True)

    def _show_classification(self, result: ClassificationResult) -> None:
        self.per_image_table.show(result.per_image)
        self.notebook.select(self.counts_tab)


def main() -> None:
    ROIClassifierApp().mainloop()
