"""Dialog for naming the classes found in the loaded model dataset."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from .labels import label_problem


class ClassLabelsDialog(tk.Toplevel):
    """Ask for a name per class. Calls on_save with {class: name} when saved."""

    def __init__(
        self,
        parent: tk.Misc,
        classes: list[int],
        current: dict[int, str],
        on_save: Callable[[dict[int, str]], None],
    ) -> None:
        super().__init__(parent)
        self.title("Class labels")
        self.transient(parent)
        self.on_save = on_save
        self.vars: dict[int, tk.StringVar] = {}

        ttk.Label(
            self,
            text="Name each class found in the model dataset:",
            padding=(10, 10, 10, 4),
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        for row, cls in enumerate(classes, start=1):
            ttk.Label(self, text=f"Class {cls}").grid(
                row=row, column=0, sticky="w", padx=(10, 8), pady=3
            )
            var = tk.StringVar(value=current.get(cls, ""))
            self.vars[cls] = var
            ttk.Entry(self, textvariable=var, width=30).grid(
                row=row, column=1, sticky="ew", padx=(0, 10), pady=3
            )

        buttons = ttk.Frame(self, padding=10)
        buttons.grid(row=len(classes) + 1, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT, padx=(6, 0)
        )
        ttk.Button(buttons, text="Save", command=self._save).pack(side=tk.RIGHT)
        self.columnconfigure(1, weight=1)
        self.bind("<Return>", lambda _event: self._save())
        self.bind("<Escape>", lambda _event: self.destroy())

    def _save(self) -> None:
        names = {cls: var.get().strip() for cls, var in self.vars.items()}
        problem = label_problem(names)
        if problem:
            messagebox.showwarning("Class labels", problem, parent=self)
            return
        self.on_save(names)
        self.destroy()
