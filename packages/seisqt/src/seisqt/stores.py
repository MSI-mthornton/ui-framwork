"""Shared GUI state: events, the current selection, view preferences.

Panels read from stores and listen to their signals; they never talk to each
other. Only controllers write to stores.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from PySide6.QtCore import QObject, QSettings, Signal

from seiscore.model import Event, EventTable


class EventStore(QObject):
    """Holds the current :class:`EventTable`."""

    changed = Signal(object)  # EventTable (whole table replaced, e.g. a new load)
    rowChanged = Signal(int)  # one row's QC changed

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._table = EventTable.empty()

    @property
    def table(self) -> EventTable:
        return self._table

    def __len__(self) -> int:
        return len(self._table)

    def set_table(self, table: EventTable) -> None:
        self._table = table
        self.changed.emit(table)

    def set_qc(self, index: int, code: str, group: str = "event") -> tuple[str, str]:
        """Set one row's QC; returns the previous ``(code, group)`` for rollback."""
        old = self._table.event(index)
        self._table = self._table.with_qc(index, code, group)
        self.rowChanged.emit(index)
        return old.qc_code, old.qc_group


class Selection(QObject):
    """The current event index (-1 for none)."""

    changed = Signal(int)

    def __init__(self, events: EventStore, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.events = events
        self._index = -1
        events.changed.connect(self._on_table)

    @property
    def index(self) -> int:
        return self._index

    @property
    def event(self) -> Event | None:
        return self.events.table.event(self._index) if self._index >= 0 else None

    def set(self, index: int) -> None:
        n = len(self.events)
        index = -1 if n == 0 else max(0, min(int(index), n - 1))
        if index != self._index:
            self._index = index
            self.changed.emit(index)

    def step(self, delta: int) -> None:
        self.set(self._index + delta if self._index >= 0 else 0)

    def select_id(self, event_id: str) -> bool:
        i = self.events.table.index_of(event_id)
        if i is None:
            return False
        self.set(i)
        return True

    def _on_table(self, table: EventTable) -> None:
        old = self._index
        self._index = -1
        if len(table):
            self.set(0)
        elif old != -1:
            self.changed.emit(-1)


@dataclass(frozen=True)
class ViewPrefs:
    """Display settings shared by panels and remembered between sessions."""

    component: str = "P"
    sort: str = "offset"
    clip_percentile: float = 99.0
    gain: float = 1.0
    colormap: str = "seismic"
    show_stack: bool = True
    max_display_stations: int = 1500
    max_display_samples: int = 2000


class PrefsStore(QObject):
    changed = Signal(object)  # ViewPrefs

    def __init__(self, prefs: ViewPrefs | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._prefs = prefs or ViewPrefs()

    @property
    def prefs(self) -> ViewPrefs:
        return self._prefs

    def update(self, **changes: Any) -> None:
        new = replace(self._prefs, **changes)
        if new != self._prefs:
            self._prefs = new
            self.changed.emit(new)

    def save(self, settings: QSettings, group: str = "view") -> None:
        settings.beginGroup(group)
        for f in fields(self._prefs):
            settings.setValue(f.name, getattr(self._prefs, f.name))
        settings.endGroup()

    def restore(self, settings: QSettings, group: str = "view") -> None:
        settings.beginGroup(group)
        changes: dict[str, Any] = {}
        for f in fields(self._prefs):
            if not settings.contains(f.name):
                continue
            raw = settings.value(f.name)
            default = getattr(self._prefs, f.name)
            try:
                if isinstance(default, bool):
                    changes[f.name] = str(raw).lower() in {"1", "true"}
                else:
                    changes[f.name] = type(default)(raw)
            except (TypeError, ValueError):
                pass
        settings.endGroup()
        if changes:
            self.update(**changes)
