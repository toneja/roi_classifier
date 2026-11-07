"""A results table that sorts when a column heading is clicked."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pandas as pd

UP_ARROW = "▲"
DOWN_ARROW = "▼"


def _sort_key(text: str) -> tuple[int, float | str]:
    """Numbers sort numerically (a trailing % is ignored); other text sorts alphabetically."""
    try:
        return (0, float(text.rstrip("%").strip()))
    except ValueError:
        return (1, text.lower())


class SortableTable(ttk.Frame):
    """A Treeview with scrollbars. Clicking a heading sorts by that column."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.tree = ttk.Treeview(self, show="headings")
        yscroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        xscroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._columns: list[str] = []
        self._sort_column: str | None = None
        self._descending = False

    def show(self, df: pd.DataFrame) -> None:
        """Replace the table's contents with df."""
        self.tree.delete(*self.tree.get_children())
        self._columns = [str(c) for c in df.columns]
        self._sort_column = None
        self._descending = False
        self.tree["columns"] = self._columns
        for col in self._columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by(c))
            self.tree.column(col, width=110, anchor=tk.CENTER, stretch=False)
        for row in df.itertuples(index=False):
            self.tree.insert("", tk.END, values=[str(v) for v in row])

    def sort_by(self, column: str) -> None:
        """The first click sorts ascending; clicking the same heading again reverses it."""
        descending = column == self._sort_column and not self._descending
        self._sort_column, self._descending = column, descending

        items = [(self.tree.set(item, column), item) for item in self.tree.get_children("")]
        items.sort(key=lambda pair: _sort_key(pair[0]), reverse=descending)
        for position, (_, item) in enumerate(items):
            self.tree.move(item, "", position)

        for col in self._columns:
            arrow = f" {DOWN_ARROW if descending else UP_ARROW}" if col == column else ""
            self.tree.heading(col, text=col + arrow)
