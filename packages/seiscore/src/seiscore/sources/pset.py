"""PSET trace files (SEG-Y layout: 3200-byte text, 400-byte binary header,
240-byte trace headers, IEEE float32 samples).

This replaces event3CQC's ``seistrc.py`` / ``FileSet`` / ``TraceSet`` with a
memory-mapped, vectorised reader:

- A file with fixed-length traces is mapped once as a structured array, so
  reading a window is one slice, not a Python loop with a ``deepcopy`` per trace.
- :class:`PsetFileSet` indexes each file's time span from its first header and
  keeps the index, so finding the files for an event is a lookup, not a scan.

Conventions carried over from event3CQC (check these against your data):

- Samples per trace come from ``fldr`` (PSET stores the big sample count
  there), falling back to ``ns``.
- ``dt`` is in microseconds.
- A trace's first sample is at ``year/day/hour/minute/sec/micro``.
- ``trid`` 12/13/14 are down/east/north; 15/16/17 are already-rotated SV/SH/P.
- ``gx``, ``gy``, ``gelev`` are used as stored (``scalco`` is ignored, as in
  event3CQC); ``gelev`` is elevation, i.e. z up.

IBM-float files (format code 1) are not supported.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from seiscore import dsp
from seiscore.conventions import RAW_COMPONENTS, RAY_COMPONENTS
from seiscore.model import Event, Gather
from seiscore.sources.base import GatherRequest, SourceError

TEXT_HEADER_BYTES = 3200
BINARY_HEADER_BYTES = 400
TRACE_HEADER_BYTES = 240

_HEADER_FORMAT = "7l4h8l2h4l13h2H31h6fl2h4l6h"
HEADER_FIELDS: tuple[str, ...] = (
    "tracl",
    "tracr",
    "fldr",
    "tracf",
    "ep",
    "cdp",
    "cdpt",
    "trid",
    "nvs",
    "nhs",
    "duse",
    "offset",
    "gelev",
    "selev",
    "sdepth",
    "gdel",
    "sdel",
    "swdep",
    "gwdep",
    "scalel",
    "scalco",
    "sx",
    "sy",
    "gx",
    "gy",
    "counit",
    "wevel",
    "swevel",
    "sut",
    "gut",
    "sstat",
    "gstat",
    "tstat",
    "laga",
    "lagb",
    "delrt",
    "muts",
    "mute",
    "ns",
    "dt",
    "gain",
    "igc",
    "igi",
    "corr",
    "sfs",
    "sfe",
    "slen",
    "styp",
    "stas",
    "stae",
    "tatyp",
    "afilf",
    "afils",
    "nofilf",
    "nofils",
    "lcf",
    "hcf",
    "lcs",
    "hcs",
    "year",
    "day",
    "hour",
    "minute",
    "sec",
    "timbas",
    "trwf",
    "grnors",
    "grnofr",
    "grnlof",
    "gaps",
    "otrav",
    "d1",
    "f1",
    "d2",
    "f2",
    "ungpow",
    "unscale",
    "ntr",
    "mark",
    "shortpad",
    "nsbig",
    "gline",
    "gpoint",
    "micro",
    "unass1",
    "unass2",
    "unass3",
    "unass4",
    "unass5",
    "unass6",
)
_CODE = {"l": "i4", "h": "i2", "H": "u2", "f": "f4"}
VALID_FORMAT_CODES = {1, 2, 3, 4, 5, 8}

#: trace id -> component, after converting down to up.
TRID_COMPONENT = {12: "Z", 13: "E", 14: "N", 15: "SV", 16: "SH", 17: "P"}


def _header_dtype(byteorder: str) -> np.dtype:
    types: list[str] = []
    for count, code in _parse_format(_HEADER_FORMAT):
        types.extend([byteorder + _CODE[code]] * count)
    dtype = np.dtype(list(zip(HEADER_FIELDS, types, strict=True)))
    assert dtype.itemsize == TRACE_HEADER_BYTES, dtype.itemsize
    return dtype


def _parse_format(fmt: str) -> list[tuple[int, str]]:
    out, digits = [], ""
    for ch in fmt:
        if ch.isdigit():
            digits += ch
        else:
            out.append((int(digits or 1), ch))
            digits = ""
    return out


HEADER_DTYPE_LE = _header_dtype("<")
HEADER_DTYPE_BE = _header_dtype(">")


class PsetFile:
    """One PSET file, memory-mapped. Safe to share between threads for reading."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        size = self.path.stat().st_size
        if size < TEXT_HEADER_BYTES + BINARY_HEADER_BYTES + TRACE_HEADER_BYTES:
            raise SourceError(f"{self.path}: too small to be a PSET file")
        with self.path.open("rb") as handle:
            handle.seek(TEXT_HEADER_BYTES)
            binary = handle.read(BINARY_HEADER_BYTES)
            first = handle.read(TRACE_HEADER_BYTES)
        fmt_le = int(np.frombuffer(binary, dtype="<u2")[12])
        fmt_be = int(np.frombuffer(binary, dtype=">u2")[12])
        if fmt_le in VALID_FORMAT_CODES:
            self.byteorder, fmt = "<", fmt_le
        elif fmt_be in VALID_FORMAT_CODES:
            self.byteorder, fmt = ">", fmt_be
        else:
            raise SourceError(f"{self.path}: not a valid PSET/SEG-Y file (format code {fmt_le})")
        if fmt == 1:
            raise SourceError(f"{self.path}: IBM float samples are not supported")
        hdr_dtype = HEADER_DTYPE_LE if self.byteorder == "<" else HEADER_DTYPE_BE
        h0 = np.frombuffer(first, dtype=hdr_dtype)[0]
        ns = int(h0["fldr"]) if int(h0["fldr"]) > 0 else int(h0["ns"])
        if ns <= 0:
            raise SourceError(f"{self.path}: invalid sample count {ns}")
        trace_bytes = TRACE_HEADER_BYTES + 4 * ns
        body = size - TEXT_HEADER_BYTES - BINARY_HEADER_BYTES
        if body % trace_bytes:
            raise SourceError(
                f"{self.path}: trace length is not constant; variable-length files are not supported"
            )
        self.n_samples = ns
        self.dt = float(h0["dt"]) * 1e-6
        if not self.dt > 0:
            raise SourceError(f"{self.path}: invalid dt {h0['dt']}")
        self._dtype = np.dtype([("hdr", hdr_dtype), ("data", self.byteorder + "f4", (ns,))])
        self._records = np.memmap(
            self.path,
            dtype=self._dtype,
            mode="r",
            offset=TEXT_HEADER_BYTES + BINARY_HEADER_BYTES,
            shape=(body // trace_bytes,),
        )
        self._start_ns: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self._records.shape[0])

    @property
    def headers(self) -> np.ndarray:
        """Structured array of all trace headers (a view on the map)."""
        return self._records["hdr"]

    @property
    def start_ns(self) -> np.ndarray:
        """First-sample time of every trace, as int64 nanoseconds since the epoch."""
        if self._start_ns is None:
            self._start_ns = _start_times_ns(self.headers)
        return self._start_ns

    @property
    def span_ns(self) -> tuple[int, int]:
        start = self.start_ns
        end = start + int(round(self.n_samples * self.dt * 1e9))
        return int(start.min()), int(end.max())

    def samples(self, rows: np.ndarray, i0: int, i1: int) -> np.ndarray:
        """Float32 samples ``[i0, i1)`` for trace ``rows`` (native byte order)."""
        return np.ascontiguousarray(self._records["data"][rows, i0:i1], dtype=np.float32)


def _start_times_ns(hdr: np.ndarray) -> np.ndarray:
    year = hdr["year"].astype("int64")
    years = (year - 1970).astype("timedelta64[Y]") + np.datetime64("1970", "Y")
    base = years.astype("datetime64[D]").astype("datetime64[ns]")
    offset = (
        (hdr["day"].astype("int64") - 1) * 86_400_000_000_000
        + hdr["hour"].astype("int64") * 3_600_000_000_000
        + hdr["minute"].astype("int64") * 60_000_000_000
        + hdr["sec"].astype("int64") * 1_000_000_000
        + hdr["micro"].astype("int64") * 1_000
    )
    return base.astype("int64") + offset


@dataclass(frozen=True)
class _FileSpan:
    path: Path
    start_ns: int
    end_ns: int


class PsetFileSet:
    """A set of PSET files indexed by time span. Files are opened on demand."""

    def __init__(self, paths: Iterable[str | os.PathLike[str]], *, max_open: int = 32) -> None:
        self._spans: list[_FileSpan] = []
        self._open: dict[Path, PsetFile] = {}
        self._max_open = max_open
        for path in paths:
            pset = PsetFile(path)
            start, end = pset.span_ns
            self._spans.append(_FileSpan(Path(path), start, end))
            self._remember(pset)
        self._spans.sort(key=lambda s: s.start_ns)

    @classmethod
    def from_list_file(cls, list_path: str | os.PathLike[str]) -> PsetFileSet:
        """Paths one per line, or a CSV with a ``PATH`` column (event3CQC file sets)."""
        list_path = Path(list_path)
        text = list_path.read_text().splitlines()
        if text and text[0].strip().upper().startswith("PATH"):
            paths = pd.read_csv(list_path)["PATH"].astype(str).str.strip().tolist()
        else:
            paths = [line.strip() for line in text if line.strip() and not line.startswith("#")]
        base = list_path.parent
        return cls([p if Path(p).is_absolute() else base / p for p in paths])

    def __len__(self) -> int:
        return len(self._spans)

    def files_overlapping(self, start_ns: int, end_ns: int) -> list[PsetFile]:
        return [
            self._get(span.path)
            for span in self._spans
            if span.start_ns < end_ns and span.end_ns > start_ns
        ]

    def _get(self, path: Path) -> PsetFile:
        pset = self._open.get(path)
        if pset is None:
            pset = PsetFile(path)
            self._remember(pset)
        return pset

    def _remember(self, pset: PsetFile) -> None:
        if len(self._open) >= self._max_open:
            self._open.pop(next(iter(self._open)))
        self._open[pset.path] = pset


TravelTimeFn = Callable[[Event, pd.DataFrame, str], np.ndarray]
"""``(event, stations, phase) -> travel time in seconds per station row``."""


class PsetGatherSource:
    """Event gathers cut from a :class:`PsetFileSet`.

    ``travel_times`` is needed for arrival alignment; without it only origin
    alignment is available.
    """

    def __init__(self, files: PsetFileSet, *, travel_times: TravelTimeFn | None = None) -> None:
        self.files = files
        self.travel_times = travel_times

    def fetch_gather(self, event: Event, request: GatherRequest) -> Gather:
        origin_ns = int(event.origin_time.value)
        start_ns = origin_ns - int(round(request.pre * 1e9))
        post_ns = int(round(request.post * 1e9))
        # Pass 1 (headers only): the stations, so travel times are known before
        # any samples are read. Arrival alignment then reads origin-pre up to
        # origin+post+max(travel time) and cuts one window per station.
        files = self.files.files_overlapping(start_ns, origin_ns + post_ns)
        stations, _, _, dt = _collect(files, start_ns, start_ns)
        if not stations:
            raise SourceError(f"no PSET traces cover event {event.id} at {event.origin_time}")
        station_df = pd.DataFrame(stations)
        shifts = np.zeros(len(station_df))
        reference = "origin"
        if request.alignment == "arrival":
            if self.travel_times is None:
                raise SourceError("arrival alignment needs travel times; configure travel_times")
            shifts = np.asarray(self.travel_times(event, station_df, request.phase), dtype=float)
            if shifts.shape != (len(station_df),):
                raise SourceError("travel_times returned the wrong number of values")
            shifts = np.where(np.isfinite(shifts), shifts, 0.0)
            station_df["travel_time"] = shifts
            reference = f"arrival:{request.phase}"
        end_ns = origin_ns + post_ns + int(np.ceil(max(shifts.max(), 0.0) * 1e9)) + int(dt * 2e9)
        files = self.files.files_overlapping(start_ns, end_ns)
        stations2, comps, data, dt = _collect(files, start_ns, end_ns)
        if len(stations2) != len(stations):
            raise SourceError("PSET station set changed between reads")
        n_out = int(round((request.pre + request.post) / dt)) + 1
        start_idx = np.round(shifts / dt).astype(int)
        has_raw = set(comps) >= set(RAW_COMPONENTS)
        components = (
            RAY_COMPONENTS if set(comps) >= set(RAY_COMPONENTS) and not has_raw else RAW_COMPONENTS
        )
        out = np.zeros((3, len(station_df), n_out), dtype=np.float32)
        for c, name in enumerate(components):
            block = data.get(name)
            if block is None:
                continue
            for s in range(len(station_df)):
                i0 = start_idx[s]
                i1 = min(i0 + n_out, block.shape[1])
                if i0 < block.shape[1] and i1 > max(i0, 0):
                    lo = max(i0, 0)
                    out[c, s, lo - i0 : i1 - i0] = block[s, lo:i1]
        gather = Gather(
            data=out,
            components=components,
            stations=station_df,
            dt=dt,
            t0=-request.pre,
            reference=reference,
            event_id=event.id,
            source_xyz=(event.x, event.y, event.z),
            meta={"source": "pset", "files": [str(f.path) for f in files]},
        )
        if request.rotate and components == RAW_COMPONENTS:
            gather = dsp.rotate_gather(gather)
        return gather


def _collect(
    files: Sequence[PsetFile], start_ns: int, end_ns: int
) -> tuple[list[dict], set[str], dict[str, np.ndarray], float]:
    """Traces in ``[start_ns, end_ns)`` grouped by (gline, gpoint) station.

    Returns station rows, components seen, ``{component: (station, sample)}``
    starting at ``start_ns``, and dt.
    """
    if not files:
        return [], set(), {}, 0.0
    dt = files[0].dt
    if any(abs(f.dt - dt) > 1e-12 for f in files):
        raise SourceError("PSET files in one window have different sample intervals")
    n_win = int(round((end_ns - start_ns) / (dt * 1e9)))
    station_index: dict[tuple[int, int], int] = {}
    stations: list[dict] = []
    blocks: dict[str, list[tuple[int, int, np.ndarray]]] = {}
    for pset in files:
        hdr = pset.headers
        starts = pset.start_ns
        trid = hdr["trid"].astype(int)
        keep = np.isin(trid, list(TRID_COMPONENT))
        # first output sample index for each trace's first sample
        offset = np.round((starts - start_ns) / (dt * 1e9)).astype(np.int64)
        for row in np.flatnonzero(keep):
            key = (int(hdr["gline"][row]), int(hdr["gpoint"][row]))
            s = station_index.get(key)
            if s is None:
                s = station_index[key] = len(stations)
                stations.append(
                    {
                        "code": f"{key[0]}/{key[1]}",
                        "x": float(hdr["gx"][row]),
                        "y": float(hdr["gy"][row]),
                        "z": float(hdr["gelev"][row]),
                        "gline": float(key[0]),
                        "gpoint": float(key[1]),
                    }
                )
            j0 = int(offset[row])  # output index of this trace's sample 0
            i0 = max(0, -j0)
            i1 = min(pset.n_samples, n_win - j0)
            if i1 <= i0:
                continue
            samples = pset.samples(np.array([row]), i0, i1)[0]
            comp = TRID_COMPONENT[int(trid[row])]
            if int(trid[row]) == 12:
                samples = -samples  # down -> up
            blocks.setdefault(comp, []).append((s, j0 + i0, samples))
    data: dict[str, np.ndarray] = {}
    for comp, pieces in blocks.items():
        arr = np.zeros((len(stations), n_win), dtype=np.float32)
        for s, j, samples in pieces:
            arr[s, j : j + samples.size] = samples
        data[comp] = arr
    return stations, set(blocks), data, dt


def write_pset(
    path: str | os.PathLike[str],
    headers: pd.DataFrame,
    data: np.ndarray,
    *,
    dt: float,
) -> None:
    """Write a fixed-length little-endian PSET file. ``headers`` columns are
    trace header field names; missing fields are zero. ``dt`` is in seconds."""
    data = np.asarray(data, dtype="<f4")
    n_traces, ns = data.shape
    if len(headers) != n_traces:
        raise ValueError("one header row per trace is required")
    hdr = np.zeros(n_traces, dtype=HEADER_DTYPE_LE)
    for name in headers.columns:
        if name in HEADER_DTYPE_LE.names:
            hdr[name] = headers[name].to_numpy()
    hdr["fldr"] = ns
    hdr["ns"] = ns if ns <= 32767 else 0
    hdr["dt"] = int(round(dt * 1e6))
    binary = np.zeros(200, dtype="<i2")
    binary[8] = int(round(dt * 1e6))
    binary[10] = ns if ns <= 32767 else 0
    binary[12] = 5
    rec = np.zeros(n_traces, dtype=[("hdr", HEADER_DTYPE_LE), ("data", "<f4", (ns,))])
    rec["hdr"] = hdr
    rec["data"] = data
    with Path(path).open("wb") as handle:
        handle.write(b"\0" * TEXT_HEADER_BYTES)
        handle.write(binary.tobytes())
        handle.write(rec.tobytes())
