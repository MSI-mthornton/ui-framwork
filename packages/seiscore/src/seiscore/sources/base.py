"""Data-source protocols.

Panels and controllers never call Pulse or read files directly; they go
through these protocols, run by the task runner on worker threads. Any
implementation must be safe to call from a worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from seiscore.model import Event, EventTable, Gather

Alignment = Literal["origin", "arrival"]


class SourceError(RuntimeError):
    """A data source failed in an expected way (missing data, bad response)."""


@dataclass(frozen=True)
class TimeWindow:
    """Half-open UTC window ``[start, end)``; ``None`` is open-ended."""

    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("TimeWindow end must be after start")


@dataclass(frozen=True)
class GatherRequest:
    """What to fetch for one event. Hashable, so it doubles as a cache key."""

    event_id: str
    pre: float = 1.0
    post: float = 2.0
    alignment: Alignment = "arrival"
    phase: str = "P"
    travel_time_id: str | None = None
    rotate: bool = True

    def key(self) -> tuple:
        return (
            self.event_id,
            round(self.pre, 6),
            round(self.post, 6),
            self.alignment,
            self.phase,
            self.travel_time_id,
            self.rotate,
        )


@runtime_checkable
class EventSource(Protocol):
    def fetch_events(self, window: TimeWindow) -> EventTable: ...


@runtime_checkable
class GatherSource(Protocol):
    def fetch_gather(self, event: Event, request: GatherRequest) -> Gather: ...


@runtime_checkable
class QcSink(Protocol):
    def write_qc(
        self,
        event_id: str,
        outcome_code: str,
        *,
        reviewed_by: str | None = None,
        notes: str = "",
    ) -> None: ...


@dataclass(frozen=True)
class Sources:
    """The sources an app runs with. Any may be None if the app does not need it."""

    events: EventSource | None = None
    gathers: GatherSource | None = None
    qc: QcSink | None = None
