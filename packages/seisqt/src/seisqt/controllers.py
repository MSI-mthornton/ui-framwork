"""Controllers: the only code that turns user intent into source calls.

* :class:`EventsController` loads the event table.
* :class:`GatherController` fetches the gather for the selected event, through
  the cache, and prefetches neighbours.
* :class:`QcController` tags events optimistically and rolls back on failure.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace

from PySide6.QtCore import QObject, Signal

from seiscore.model import EventTable, Gather
from seiscore.sources.base import GatherRequest, Sources, TimeWindow
from seisqt.cache import GatherCache, Prefetcher
from seisqt.stores import EventStore, Selection
from seisqt.tasks import Result, TaskRunner

log = logging.getLogger(__name__)


class EventsController(QObject):
    loading = Signal(bool)
    failed = Signal(str)

    def __init__(
        self,
        runner: TaskRunner,
        sources: Sources,
        events: EventStore,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.runner, self.sources, self.events = runner, sources, events
        self.window = TimeWindow()

    def load(self, window: TimeWindow | None = None) -> None:
        if self.sources.events is None:
            self.failed.emit("no event source configured")
            return
        if window is not None:
            self.window = window
        src, win = self.sources.events, self.window
        self.loading.emit(True)
        self.runner.submit(
            lambda: src.fetch_events(win),
            self._done,
            key="events",
            label="load events",
            timeout=None,
        )

    def _done(self, res: Result) -> None:
        self.loading.emit(False)
        if res.ok:
            table: EventTable = res.value
            log.info("loaded %d events", len(table))
            self.events.set_table(table)
        else:
            self.failed.emit(f"loading events failed: {res.message}")


class GatherController(QObject):
    """Keeps the gather in sync with the selection."""

    loading = Signal(str)  # event id
    ready = Signal(object)  # Gather
    failed = Signal(str, str)  # event id, message
    cleared = Signal()

    def __init__(
        self,
        runner: TaskRunner,
        sources: Sources,
        selection: Selection,
        cache: GatherCache,
        request: GatherRequest | None = None,
        *,
        prefetch_ahead: int = 2,
        prefetch_behind: int = 1,
        timeout: float | None = 60.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.runner, self.sources, self.selection, self.cache = runner, sources, selection, cache
        self.template = request or GatherRequest(event_id="")
        self.timeout = timeout
        self.current: Gather | None = None
        self.prefetcher = Prefetcher(
            runner,
            cache,
            self._key_for,
            self._fetch_index,
            ahead=prefetch_ahead,
            behind=prefetch_behind,
        )
        selection.changed.connect(self._on_selection)

    # --- requests --------------------------------------------------------
    def request_for(self, event_id: str) -> GatherRequest:
        return replace(self.template, event_id=event_id)

    def set_template(self, **changes) -> None:
        """Change window/alignment/phase for all later fetches and refetch."""
        self.template = replace(self.template, **changes)
        self.refresh()

    def _key_for(self, index: int) -> tuple:
        return self.request_for(str(self.selection.events.table.ids[index])).key()

    def _fetch_index(self, index: int) -> tuple[tuple, Gather]:
        event = self.selection.events.table.event(index)
        req = self.request_for(event.id)
        return req.key(), self.sources.gathers.fetch_gather(event, req)

    # --- flow ------------------------------------------------------------
    def refresh(self) -> None:
        self._on_selection(self.selection.index)

    def _on_selection(self, index: int) -> None:
        if index < 0 or self.sources.gathers is None:
            self.runner.cancel("gather")
            self.current = None
            self.cleared.emit()
            return
        event = self.selection.events.table.event(index)
        req = self.request_for(event.id)
        cached = self.cache.get(req.key())
        if cached is not None:
            self.runner.cancel("gather")
            self._show(cached)
        else:
            self.loading.emit(event.id)
            src = self.sources.gathers
            self.runner.submit(
                lambda: src.fetch_gather(event, req),
                lambda res, req=req: self._done(req, res),
                key="gather",
                label=f"gather {event.id}",
                timeout=self.timeout,
            )
        self.prefetcher.around(index, len(self.selection.events))

    def _done(self, req: GatherRequest, res: Result) -> None:
        if res.ok:
            self.cache.put(req.key(), res.value)
            self._show(res.value)
        else:
            self.failed.emit(req.event_id, res.message)

    def _show(self, gather: Gather) -> None:
        self.current = gather
        self.ready.emit(gather)


class QcController(QObject):
    """Optimistic QC tagging with rollback and optional auto-advance."""

    written = Signal(str, str)  # event id, code
    failed = Signal(str, str)  # event id, message

    def __init__(
        self,
        runner: TaskRunner,
        sources: Sources,
        events: EventStore,
        selection: Selection,
        *,
        codes: Sequence[str] = ("keeper", "noise"),
        group: str = "event",
        auto_advance: bool = True,
        reviewed_by: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.runner, self.sources, self.events, self.selection = runner, sources, events, selection
        self.codes = tuple(codes)
        self.group = group
        self.auto_advance = auto_advance
        self.reviewed_by = reviewed_by
        self.pending: dict[str, int] = {}  # event id -> writes in flight

    def tag(self, code: str, index: int | None = None) -> None:
        index = self.selection.index if index is None else index
        if index < 0:
            return
        event = self.events.table.event(index)
        old_code, old_group = self.events.set_qc(index, code, self.group)
        if self.auto_advance and index == self.selection.index:
            self.selection.step(1)
        sink = self.sources.qc
        if sink is None:  # local-only tagging
            self.written.emit(event.id, code)
            return
        self.pending[event.id] = self.pending.get(event.id, 0) + 1
        self.runner.submit(
            lambda: sink.write_qc(event.id, code, reviewed_by=self.reviewed_by),
            self._make_done(event.id, code, old_code, old_group),
            label=f"qc {event.id}={code}",
        )

    def _make_done(
        self, event_id: str, code: str, old_code: str, old_group: str
    ) -> Callable[[Result], None]:
        def done(res: Result) -> None:
            left = self.pending.get(event_id, 1) - 1
            if left:
                self.pending[event_id] = left
            else:
                self.pending.pop(event_id, None)
            if res.ok:
                self.written.emit(event_id, code)
                return
            index = self.events.table.index_of(event_id)
            if index is not None and self.events.table.event(index).qc_code == code:
                self.events.set_qc(index, old_code, old_group)
            self.failed.emit(event_id, f"QC write {code!r} failed: {res.message}")

        return done
