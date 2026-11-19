"""Pulse-backed sources (events, gathers, QC writes).

Ported from qc-keepers ``events.py``, ``gathers.py`` and ``qc.py``. The Pulse
packages (``pulse_sdk``, ``pulse_schemas``) come from a private repo, so they
are imported lazily: importing :mod:`seiscore` never needs them.

Threading: one synchronous ``PulseClient`` is shared by all three sources. The
sync client is thread-safe; do not swap in the async client without adding a
lock, because the task runner calls these from worker threads.

Status: written against the qc-keepers code paths; not yet run against a live
Pulse server from this package.
"""

from __future__ import annotations

import getpass
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from seiscore import dsp
from seiscore.conventions import RAW_COMPONENTS, canonical_component
from seiscore.model import Event, EventTable, Gather
from seiscore.sources.base import GatherRequest, SourceError, TimeWindow

log = logging.getLogger(__name__)

PAGE_SIZE = 5000
RUN_ID_CHUNK = 200

EVENT_FIELDS = (
    "id",
    "origin_time",
    "x",
    "y",
    "z",
    "snr",
    "pset_run_id",
    "latest_qc_outcome_code",
    "latest_qc_outcome_group",
    "latest_qc_state",
)


@dataclass(frozen=True)
class PulseSettings:
    host: str = "hou-dok-1"
    port: int = 8000
    project: str = ""
    page_size: int = PAGE_SIZE
    run_id_chunk: int = RUN_ID_CHUNK
    apply_statics: bool = True


def _pulse_error() -> type[Exception]:
    try:
        from pulse_sdk.exceptions import PulseError
    except ImportError:  # pragma: no cover - only without pulse installed
        return RuntimeError
    return PulseError


class PulseConnection:
    """Owns the one shared sync client. Opened on first use (on a worker)."""

    def __init__(
        self, settings: PulseSettings, *, factory: Callable[[], Any] | None = None
    ) -> None:
        self.settings = settings
        self._factory = factory
        self._client: Any = None
        self._lock = threading.Lock()

    @property
    def client(self) -> Any:
        with self._lock:
            if self._client is None:
                self._client = self._factory() if self._factory else self._open()
            return self._client

    def _open(self) -> Any:
        try:
            from pulse_sdk import PulseClient
        except ImportError as exc:
            raise SourceError("pulse_sdk is not installed in this environment") from exc
        if not self.settings.project:
            raise SourceError("Pulse project is not configured (pulse.project)")
        log.info(
            "connecting to Pulse %s:%s project=%s",
            self.settings.host,
            self.settings.port,
            self.settings.project,
        )
        return PulseClient(
            project_name=self.settings.project, host=self.settings.host, port=self.settings.port
        )

    def call(self, fn: Callable[[Any], Any], what: str) -> Any:
        """Run ``fn(client)``, turning Pulse errors into :class:`SourceError`."""
        try:
            return fn(self.client)
        except SourceError:
            raise
        except _pulse_error() as exc:
            detail = getattr(exc, "response_data", None) or exc
            raise SourceError(f"{what}: {detail}") from exc

    def close(self) -> None:
        with self._lock:
            client, self._client = self._client, None
        closer = getattr(client, "close", None)
        if callable(closer):
            closer()


def _iso(stamp: datetime) -> str:
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC).isoformat()


def _pages(fetch: Callable[[int], Any]):
    page_number = 1
    while True:
        page = fetch(page_number)
        if not page.data:
            return
        yield page
        if not page.has_next_page:
            return
        page_number += 1


# --- events ------------------------------------------------------------------


class PulseEventSource:
    """Events with a latest QC outcome, minus events from superseded PSET runs."""

    def __init__(self, conn: PulseConnection, *, require_qc: bool = True) -> None:
        self.conn = conn
        self.require_qc = require_qc

    def fetch_events(self, window: TimeWindow) -> EventTable:
        return self.conn.call(lambda c: self._fetch(c, window), "fetching events")

    def _fetch(self, client: Any, window: TimeWindow) -> EventTable:
        s = self.conn.settings
        filters: list[dict[str, Any]] = []
        if self.require_qc:
            filters.append({"field": "latest_qc_outcome_code", "op": "is_not_null"})
        if window.start is not None:
            filters.append({"field": "origin_time", "op": "ge", "value": _iso(window.start)})
        if window.end is not None:
            filters.append({"field": "origin_time", "op": "lt", "value": _iso(window.end)})

        records: list[dict[str, Any]] = []
        for page in _pages(
            lambda n: client.event.search_select(
                fields=list(EVENT_FIELDS),
                filters=filters,
                sort=[{"field": "origin_time", "direction": "asc"}],
                page_size=s.page_size,
                page_number=n,
            )
        ):
            for row in page.data:
                run_id = row.get("pset_run_id")
                records.append(
                    {
                        "id": str(row["id"]),
                        "origin_time": row.get("origin_time"),
                        "x": row.get("x"),
                        "y": row.get("y"),
                        "z": row.get("z"),
                        "snr": row.get("snr"),
                        "pset_run_id": str(run_id) if run_id is not None else None,
                        "qc_code": row.get("latest_qc_outcome_code") or "",
                        "qc_group": row.get("latest_qc_outcome_group") or "",
                        "qc_state": row.get("latest_qc_state") or "",
                    }
                )
            log.info("events: fetched %d/%s", len(records), page.total)
        if not records:
            return EventTable.empty()
        frame = pd.DataFrame.from_records(records)
        frame["origin_time"] = pd.to_datetime(frame["origin_time"], utc=True, format="ISO8601")
        for col in ("x", "y", "z", "snr"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["x", "y", "origin_time"]).reset_index(drop=True)
        active = self._active_run_ids(client, frame["pset_run_id"].dropna().unique().tolist())
        frame = drop_superseded(frame, active)
        return EventTable(frame)

    def _active_run_ids(self, client: Any, run_ids: list[str]) -> set[str]:
        s = self.conn.settings
        merge_ids: set[str] = set()
        for i in range(0, len(run_ids), s.run_id_chunk):
            chunk = run_ids[i : i + s.run_id_chunk]
            for page in _pages(
                lambda n, chunk=chunk: client.pset_run.search_select(
                    fields=["id", "merge_seismic_id"],
                    filters=[{"field": "id", "op": "in", "value": chunk}],
                    page_size=s.page_size,
                    page_number=n,
                )
            ):
                merge_ids.update(
                    str(r["merge_seismic_id"]) for r in page.data if r.get("merge_seismic_id")
                )
        active: set[str] = set()
        ordered = sorted(merge_ids)
        for i in range(0, len(ordered), s.run_id_chunk):
            chunk = ordered[i : i + s.run_id_chunk]
            for page in _pages(
                lambda n, chunk=chunk: client.merge_seismic.search_select(
                    fields=["id", "active_run_id"],
                    filters=[{"field": "id", "op": "in", "value": chunk}],
                    page_size=s.page_size,
                    page_number=n,
                )
            ):
                active.update(
                    str(r["active_run_id"]) for r in page.data if r.get("active_run_id") is not None
                )
        return active


def drop_superseded(frame: pd.DataFrame, active_run_ids: set[str]) -> pd.DataFrame:
    """Keep events with no run, or whose run is the active run of its merge."""
    if frame.empty or not active_run_ids:
        return frame
    run_id = frame["pset_run_id"].astype("string")
    keep = (run_id.isna() | run_id.isin(active_run_ids)).to_numpy()
    if (~keep).any():
        log.info("dropped %d event(s) from superseded PSET runs", int((~keep).sum()))
    return frame.loc[keep].reset_index(drop=True)


# --- gathers -----------------------------------------------------------------


def resolve_travel_time_id(
    conn: PulseConnection, *, travel_time_id: str | None = None, file_path: str | None = None
) -> str | None:
    """Explicit id, else the row matching ``file_path``, else the project's first table."""
    if travel_time_id:
        return travel_time_id

    def run(client: Any) -> str | None:
        if file_path:
            page = client.travel_time.search(
                filters=[{"field": "travel_time_file_path", "op": "eq", "value": file_path}],
                page_size=1,
                page_number=1,
            )
        else:
            page = client.travel_time.get_list(page_size=1)
        return str(page.data[0].id) if page.data else None

    return conn.call(run, "resolving travel-time table")


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def channels_to_gather_arrays(
    samples: np.ndarray, channels: list[Any]
) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Group manifest channels into 3C stations: ``(E/N/Z data, stations, skipped)``."""
    by_station: dict[tuple, dict[str, Any]] = {}
    for ch in channels:
        comp = canonical_component(str(_value(ch, "component", "")))
        if comp not in RAW_COMPONENTS:
            continue
        loc = _value(ch, "geophone_location_id")
        gline, gpoint = _value(ch, "gline"), _value(ch, "gpoint")
        if loc is not None:
            key: tuple = ("id", str(loc))
        elif gline is not None and gpoint is not None:
            key = ("grid", gline, gpoint)
        else:
            continue
        by_station.setdefault(key, {})[comp] = ch

    rows, traces, skipped = [], [], 0
    for group in by_station.values():
        if set(RAW_COMPONENTS) - group.keys():
            skipped += 1
            continue
        ref = group["Z"]
        code = _value(ref, "station_code") or f"{_value(ref, 'gline')}/{_value(ref, 'gpoint')}"
        tt = _value(ref, "travel_time")
        rows.append(
            {
                "code": str(code),
                "x": float(_value(ref, "x", np.nan)),
                "y": float(_value(ref, "y", np.nan)),
                "z": float(_value(ref, "z", np.nan)),
                "gline": float(gline) if (gline := _value(ref, "gline")) is not None else np.nan,
                "gpoint": float(gp) if (gp := _value(ref, "gpoint")) is not None else np.nan,
                "travel_time": float(tt) if tt is not None else np.nan,
            }
        )
        traces.append([samples[int(_value(group[c], "trace_index"))] for c in RAW_COMPONENTS])
    if not rows:
        raise SourceError("no complete 3C stations in the trace manifest")
    data = np.asarray(traces, dtype=np.float32).transpose(1, 0, 2)  # (3, station, sample)
    return data, pd.DataFrame(rows), skipped


def collapse_envelope(samples: np.ndarray) -> np.ndarray:
    """Turn a width-decimated ``(trace, pixel, 2)`` envelope into 2-D (display only)."""
    if samples.ndim == 2:
        return samples
    if samples.ndim == 3 and samples.shape[-1] == 2:
        lo, hi = samples[..., 0], samples[..., 1]
        return np.where(np.abs(lo) >= np.abs(hi), lo, hi)
    raise ValueError(f"unexpected sample shape {samples.shape}")


class PulseGatherSource:
    """Full-rate E/N/Z traces from Pulse, rotated in seiscore (not by Pulse).

    ``width=None`` asks Pulse for every sample. Processing (stacks, clip
    levels) must run on full-rate data; display decimation happens in the panel.
    """

    def __init__(self, conn: PulseConnection, *, width: int | None = None) -> None:
        self.conn = conn
        self.width = width

    def fetch_gather(self, event: Event, request: GatherRequest) -> Gather:
        return self.conn.call(lambda c: self._fetch(c, event, request), f"gather {event.id}")

    def _fetch(self, client: Any, event: Event, request: GatherRequest) -> Gather:
        statics = self.conn.settings.apply_statics
        aligned: dict[str, Any] = {
            "pre": request.pre,
            "post": request.post,
            "alignment": request.alignment,
        }
        if request.alignment == "arrival":
            if request.travel_time_id is None:
                raise SourceError("arrival alignment needs a travel_time_id")
            aligned.update(travel_time_id=request.travel_time_id, phase=request.phase)
        manifest = client.event.get_trace_manifest(event.id, apply_statics=statics, **aligned)
        traces = client.event.get_traces(
            event.id, apply_statics=statics, width=self.width, **aligned
        )
        channels = manifest.trace_channels
        if channels is None and request.travel_time_id is not None:
            # Origin-aligned manifests omit inline channels; arrival ones carry them.
            channels = client.event.get_trace_manifest(
                event.id,
                apply_statics=statics,
                pre=request.pre,
                post=request.post,
                alignment="arrival",
                travel_time_id=request.travel_time_id,
                phase=request.phase,
            ).trace_channels
        if channels is None:
            raise SourceError("trace manifest has no channel metadata")
        samples = collapse_envelope(np.asarray(traces.dequantized_samples()))
        data, stations, skipped = channels_to_gather_arrays(samples, list(channels))
        header = getattr(traces, "header", {}) or {}
        n = data.shape[2]
        dt = header.get("sample_interval") or header.get("dt")
        dt = float(dt) if dt else (request.pre + request.post) / max(n - 1, 1)
        src = event.xyz
        stations["offset"] = np.hypot(stations["x"] - src[0], stations["y"] - src[1])
        gather = Gather(
            data=data,
            components=RAW_COMPONENTS,
            stations=stations,
            dt=dt,
            t0=-request.pre,
            reference="origin" if request.alignment == "origin" else f"arrival:{request.phase}",
            event_id=event.id,
            source_xyz=(float(src[0]), float(src[1]), float(src[2])),
            meta={
                "source": "pulse",
                "skipped_incomplete": skipped,
                "proc_seismic_id": header.get("source", {}).get("proc_seismic_id", "?"),
                "decimated": self.width is not None,
            },
        )
        return dsp.rotate_gather(gather) if request.rotate else gather


# --- QC writes ---------------------------------------------------------------


class PulseQcSink:
    def __init__(
        self, conn: PulseConnection, *, app_name: str = "seisqt", outcome_group: str = "event"
    ) -> None:
        self.conn = conn
        self.app_name = app_name
        self.outcome_group = outcome_group

    def write_qc(
        self, event_id: str, outcome_code: str, *, reviewed_by: str | None = None, notes: str = ""
    ) -> None:
        from pulse_schemas.qc import QCCreate, QCReviewSource, QCReviewState

        try:
            payload = QCCreate(
                event_id=uuid.UUID(event_id),
                review_source=QCReviewSource.MANUAL,
                review_state=QCReviewState.COMPLETED,
                completed_at=datetime.now(UTC),
                outcome_code=outcome_code,
                outcome_group=self.outcome_group,
                reviewed_by=reviewed_by or getpass.getuser(),
                notes=notes or f"Tagged {outcome_code} from {self.app_name}",
                details={"import_kind": self.app_name},
            )
        except ValueError as exc:
            raise SourceError(f"bad QC payload for {event_id}: {exc}") from exc
        self.conn.call(lambda c: c.qc.create(payload), f"writing QC for {event_id}")


def pulse_sources(settings: PulseSettings, *, app_name: str = "seisqt"):
    """Build events/gathers/qc sources sharing one connection."""
    from seiscore.sources.base import Sources

    conn = PulseConnection(settings)
    return conn, Sources(
        events=PulseEventSource(conn),
        gathers=PulseGatherSource(conn),
        qc=PulseQcSink(conn, app_name=app_name),
    )
