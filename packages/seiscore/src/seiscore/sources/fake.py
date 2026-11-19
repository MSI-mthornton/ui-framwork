"""Synthetic sources for tests, demos and offline development.

The fake gather is physically sensible: a P pulse polarised along the ray and
an S pulse polarised along SH, both on E/N/Z with noise. Rotating it with
:func:`seiscore.dsp.rotate_gather` should put the P pulse on P and the S pulse
on SH, which makes it a test of the conventions as well as demo data.
"""

from __future__ import annotations

import threading
import time
import zlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from seiscore import dsp
from seiscore.conventions import RAW_COMPONENTS, ray_frames
from seiscore.model import Event, EventTable, Gather
from seiscore.sources.base import GatherRequest, SourceError, TimeWindow


class FakeEventSource:
    def __init__(
        self,
        n_events: int = 2000,
        *,
        seed: int = 0,
        start: str = "2026-01-01",
        days: float = 30.0,
        qc_codes: tuple[str, ...] = ("", "keeper", "noise"),
        latency_s: float = 0.0,
    ) -> None:
        self.n_events = n_events
        self.seed = seed
        self.start = pd.Timestamp(start, tz="UTC")
        self.days = days
        self.qc_codes = qc_codes
        self.latency_s = latency_s

    def fetch_events(self, window: TimeWindow) -> EventTable:
        time.sleep(self.latency_s)
        rng = np.random.default_rng(self.seed)
        n = self.n_events
        offsets = np.sort(rng.random(n)) * self.days * 86400.0
        origin = self.start + pd.to_timedelta(offsets, unit="s")
        frame = pd.DataFrame(
            {
                "id": [f"evt-{i:06d}" for i in range(n)],
                "origin_time": origin,
                "x": rng.normal(0.0, 400.0, n),
                "y": rng.normal(0.0, 250.0, n),
                "z": rng.normal(-2500.0, 60.0, n),
                "snr": rng.lognormal(1.2, 0.6, n),
                "qc_code": rng.choice(np.array(self.qc_codes, dtype=object), n),
            }
        )
        if window.start is not None:
            frame = frame[frame["origin_time"] >= pd.Timestamp(window.start)]
        if window.end is not None:
            frame = frame[frame["origin_time"] < pd.Timestamp(window.end)]
        return EventTable(frame)


def grid_array(n_lines: int = 12, n_points: int = 25, spacing: float = 100.0) -> pd.DataFrame:
    """A surface grid of 3C stations, centred on (0, 0), z = 0."""
    lines, points = np.meshgrid(np.arange(n_lines), np.arange(n_points), indexing="ij")
    x = (points.ravel() - (n_points - 1) / 2) * spacing
    y = (lines.ravel() - (n_lines - 1) / 2) * spacing
    return pd.DataFrame(
        {
            "code": [f"{g}/{p}" for g, p in zip(lines.ravel(), points.ravel(), strict=True)],
            "x": x,
            "y": y,
            "z": np.zeros(x.size),
            "gline": lines.ravel().astype(float),
            "gpoint": points.ravel().astype(float),
        }
    )


class FakeGatherSource:
    """E/N/Z gathers with a P and an S pulse at straight-ray arrival times."""

    def __init__(
        self,
        stations: pd.DataFrame | None = None,
        *,
        dt: float = 0.0005,
        vp: float = 4000.0,
        vs: float = 2300.0,
        noise: float = 0.05,
        latency_s: float = 0.0,
        fail_ids: Iterable[str] = (),
        seed: int = 0,
    ) -> None:
        self.stations = grid_array() if stations is None else stations
        self.dt = dt
        self.vp = vp
        self.vs = vs
        self.noise = noise
        self.latency_s = latency_s
        self.fail_ids = set(fail_ids)
        self.seed = seed
        self.calls = 0
        self._lock = threading.Lock()

    def fetch_gather(self, event: Event, request: GatherRequest) -> Gather:
        with self._lock:
            self.calls += 1
        time.sleep(self.latency_s)
        if event.id in self.fail_ids:
            raise SourceError(f"synthetic failure for {event.id}")
        src = event.xyz
        rec = self.stations[["x", "y", "z"]].to_numpy(dtype=float)
        dist = np.linalg.norm(rec - src, axis=1)
        tp, ts = dist / self.vp, dist / self.vs
        n = int(round((request.pre + request.post) / self.dt)) + 1
        rel = -request.pre + self.dt * np.arange(n)
        # Arrival alignment puts each trace's chosen phase at t = 0.
        if request.alignment == "arrival":
            shift = ts if request.phase.upper().startswith("S") else tp
        else:
            shift = np.zeros_like(tp)
        t = rel[None, :] + shift[:, None]  # absolute time since origin per station
        frames, _ = ray_frames(src, rec)
        p_hat, sh_hat = frames[:, 0, :], frames[:, 1, :]
        pulse_p = _ricker(t - tp[:, None], 60.0) / np.maximum(dist, 1.0)[:, None] * 1000.0
        pulse_s = _ricker(t - ts[:, None], 35.0) / np.maximum(dist, 1.0)[:, None] * 2000.0
        rng = np.random.default_rng((self.seed, zlib.crc32(event.id.encode())))
        data = (
            p_hat.T[:, :, None] * pulse_p[None]
            + sh_hat.T[:, :, None] * pulse_s[None]
            + self.noise
            * rng.standard_normal((3, len(rec), n))
            * np.median(np.abs(pulse_p).max(axis=1))
        ).astype(np.float32)
        stations = self.stations.copy()
        stations["offset"] = np.hypot(rec[:, 0] - src[0], rec[:, 1] - src[1])
        stations["travel_time"] = tp
        gather = Gather(
            data=data,
            components=RAW_COMPONENTS,
            stations=stations.reset_index(drop=True),
            dt=self.dt,
            t0=-request.pre,
            reference="origin" if request.alignment == "origin" else f"arrival:{request.phase}",
            event_id=event.id,
            source_xyz=(float(src[0]), float(src[1]), float(src[2])),
            meta={"source": "fake"},
        )
        return dsp.rotate_gather(gather) if request.rotate else gather


class FakeQcSink:
    """Records writes; ``fail_codes`` makes chosen outcomes raise."""

    def __init__(self, *, fail_codes: Iterable[str] = (), latency_s: float = 0.0) -> None:
        self.fail_codes = set(fail_codes)
        self.latency_s = latency_s
        self.writes: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def write_qc(
        self,
        event_id: str,
        outcome_code: str,
        *,
        reviewed_by: str | None = None,
        notes: str = "",
    ) -> None:
        time.sleep(self.latency_s)
        if outcome_code in self.fail_codes:
            raise SourceError(f"synthetic QC failure for {outcome_code}")
        with self._lock:
            self.writes.append((event_id, outcome_code))


def _ricker(t: np.ndarray, f: float) -> np.ndarray:
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)
