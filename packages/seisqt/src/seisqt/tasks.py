"""Background work for the GUI: one runner, keyed supersession, timeouts.

Every slow call (Pulse, file reads, rotation) goes through :class:`TaskRunner`.
The callback always runs on the GUI thread with a :class:`Result`, never an
exception, so panels cannot hang on a failure.

Supersession: submitting with a ``key`` that already has a task in flight
marks the older task stale. Its result is dropped when it arrives. This is
what keeps fast scrolling through events from piling up stale redraws.

Timeouts: Python threads cannot be killed, so a timed-out task keeps running
on its worker, but its callback fires immediately with ``timed_out=True`` and
its late result is discarded.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

log = logging.getLogger(__name__)
_DEFAULT = object()


@dataclass(frozen=True)
class Result:
    ok: bool
    value: Any = None
    error: BaseException | None = None
    traceback: str = ""
    elapsed: float = 0.0
    timed_out: bool = False
    key: str | None = None

    @property
    def message(self) -> str:
        if self.timed_out:
            return f"timed out after {self.elapsed:.1f}s"
        if self.error is None:
            return ""
        return str(self.error) or type(self.error).__name__


class Task:
    """Handle to submitted work. ``cancel()`` drops its result."""

    __slots__ = ("id", "key", "label", "_cancelled", "_done", "started")

    def __init__(self, task_id: int, key: str | None, label: str) -> None:
        self.id = task_id
        self.key = key
        self.label = label
        self.started = time.perf_counter()
        self._cancelled = threading.Event()
        self._done = False

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def done(self) -> bool:
        return self._done


class _Bridge(QObject):
    finished = Signal(int, object)  # task id, Result


class _Job(QRunnable):
    def __init__(self, task: Task, fn: Callable[[], Any], bridge: _Bridge) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.task = task
        self.fn = fn
        self.bridge = bridge

    def run(self) -> None:  # worker thread
        t0 = time.perf_counter()
        if self.task.cancelled:  # superseded before it started
            self.bridge.finished.emit(self.task.id, Result(False, key=self.task.key))
            return
        try:
            value = self.fn()
            res = Result(True, value=value, elapsed=time.perf_counter() - t0, key=self.task.key)
        except BaseException as exc:  # noqa: BLE001 - delivered as a Result
            res = Result(
                False,
                error=exc,
                traceback=traceback.format_exc(),
                elapsed=time.perf_counter() - t0,
                key=self.task.key,
            )
        self.bridge.finished.emit(self.task.id, res)


class TaskRunner(QObject):
    """Runs callables on a thread pool and reports back on the GUI thread."""

    busyChanged = Signal(int)  # number of tasks in flight
    taskFailed = Signal(str, object)  # label, Result

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        max_threads: int = 4,
        default_timeout: float | None = 60.0,
    ) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max_threads)
        self.default_timeout = default_timeout
        self._ids = itertools.count(1)
        self._bridge = _Bridge()
        self._bridge.finished.connect(self._on_finished)
        self._pending: dict[int, tuple[Task, Callable[[Result], None] | None, QTimer | None]] = {}
        self._latest: dict[str, int] = {}

    # --- API -------------------------------------------------------------
    def submit(
        self,
        fn: Callable[[], Any],
        on_done: Callable[[Result], None] | None = None,
        *,
        key: str | None = None,
        label: str = "",
        timeout: float | None | object = _DEFAULT,
        priority: int = 0,
    ) -> Task:
        task = Task(next(self._ids), key, label or key or getattr(fn, "__name__", "task"))
        if key is not None:
            older = self._latest.get(key)
            if older is not None and older in self._pending:
                self._pending[older][0].cancel()
            self._latest[key] = task.id
        limit = self.default_timeout if timeout is _DEFAULT else timeout
        timer = None
        if limit:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda tid=task.id: self._on_timeout(tid))
            timer.start(int(limit * 1000))
        self._pending[task.id] = (task, on_done, timer)
        self.pool.start(_Job(task, fn, self._bridge), priority)
        self.busyChanged.emit(self.in_flight)
        return task

    def cancel(self, key: str) -> None:
        tid = self._latest.get(key)
        if tid is not None and tid in self._pending:
            self._pending[tid][0].cancel()

    @property
    def in_flight(self) -> int:
        return sum(1 for t, _, _ in self._pending.values() if not t.cancelled)

    def wait(self, msecs: int = 5000) -> bool:
        """Block until the pool is idle (tests and shutdown only)."""
        return self.pool.waitForDone(msecs)

    def shutdown(self, msecs: int = 2000) -> None:
        for task, _, timer in self._pending.values():
            task.cancel()
            if timer is not None:
                timer.stop()
        self.pool.clear()
        self.pool.waitForDone(msecs)

    # --- internals (GUI thread) -----------------------------------------
    @Slot(int, object)
    def _on_finished(self, task_id: int, result: Result) -> None:
        entry = self._pending.pop(task_id, None)
        if entry is None:  # already timed out
            return
        task, on_done, timer = entry
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        task._done = True
        self._forget(task)
        self.busyChanged.emit(self.in_flight)
        if task.cancelled:
            log.debug("dropped stale result of %s", task.label)
            return
        self._deliver(task, on_done, result)

    def _on_timeout(self, task_id: int) -> None:
        entry = self._pending.pop(task_id, None)
        if entry is None:
            return
        task, on_done, timer = entry
        if timer is not None:
            timer.deleteLater()
        task.cancel()  # its late result is dropped
        self._forget(task)
        self.busyChanged.emit(self.in_flight)
        elapsed = time.perf_counter() - task.started
        log.warning("%s timed out after %.1fs", task.label, elapsed)
        self._deliver(task, on_done, Result(False, elapsed=elapsed, timed_out=True, key=task.key))

    def _forget(self, task: Task) -> None:
        if task.key is not None and self._latest.get(task.key) == task.id:
            del self._latest[task.key]

    def _deliver(self, task: Task, on_done: Callable[[Result], None] | None, res: Result) -> None:
        if not res.ok:
            if res.traceback:
                log.error("%s failed: %s\n%s", task.label, res.message, res.traceback)
            self.taskFailed.emit(task.label, res)
        if on_done is None:
            return
        try:
            on_done(res)
        except Exception:
            log.exception("callback for %s raised", task.label)
