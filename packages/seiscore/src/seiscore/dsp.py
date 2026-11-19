"""Signal processing on gathers. Pure numpy, no Qt, no I/O.

Rule: processing runs on full-rate samples. Display decimation
(:func:`decimate_for_display`) is the last step and its output is only drawn,
never processed further.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from seiscore.conventions import RAW_COMPONENTS, RAY_COMPONENTS, ray_frames
from seiscore.model import Gather, ModelError

SortMode = Literal["offset", "grid", "traveltime", "elevation", "input"]
SORT_MODES: tuple[str, ...] = ("offset", "grid", "traveltime", "elevation", "input")


# --- rotation ---------------------------------------------------------------
def rotate_enz_to_ray(
    enz: np.ndarray,
    source_xyz: np.ndarray,
    receivers_xyz: np.ndarray,
    *,
    block: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ``(3, station, sample)`` E/N/Z data into P/SH/SV.

    Works in station blocks so a 5000-station gather never needs a second
    full-size temporary. Returns ``(rotated, valid)``; stations with an
    undefined ray (see :func:`seiscore.conventions.ray_frames`) are zero and
    flagged False in ``valid``.
    """
    enz = np.asarray(enz)
    if enz.ndim != 3 or enz.shape[0] != 3:
        raise ModelError(f"expected (3, station, sample) E/N/Z data, got {enz.shape}")
    frames, valid = ray_frames(source_xyz, receivers_xyz)
    if frames.shape[0] != enz.shape[1]:
        raise ModelError(f"{frames.shape[0]} receivers for {enz.shape[1]} stations")
    out_dtype = np.result_type(enz.dtype, np.float32)
    out = np.empty(enz.shape, dtype=out_dtype)
    for start in range(0, enz.shape[1], block):
        stop = min(start + block, enz.shape[1])
        # out[c, s, t] = sum_k frames[s, c, k] * enz[k, s, t]
        out[:, start:stop, :] = np.einsum(
            "sck,kst->cst", frames[start:stop].astype(out_dtype), enz[:, start:stop, :]
        )
    return out, valid


def rotate_gather(gather: Gather, *, drop_invalid: bool = True) -> Gather:
    """P/SH/SV gather from an E/N/Z gather that has ``source_xyz``."""
    if gather.source_xyz is None:
        raise ModelError("gather has no source_xyz; cannot rotate")
    try:
        idx = [gather.components.index(c) for c in RAW_COMPONENTS]
    except ValueError:
        raise ModelError(f"rotation needs E, N, Z components, got {gather.components}") from None
    receivers = gather.stations[["x", "y", "z"]].to_numpy(dtype=float)
    rotated, valid = rotate_enz_to_ray(gather.data[idx], np.asarray(gather.source_xyz), receivers)
    out = gather.with_data(rotated, RAY_COMPONENTS)
    if drop_invalid and not valid.all():
        out = out.take_stations(np.flatnonzero(valid))
        out = out.with_data(
            out.data, out.components, meta={**gather.meta, "skipped_stations": int((~valid).sum())}
        )
    return out


# --- station order ----------------------------------------------------------
def station_order(
    stations: pd.DataFrame,
    mode: str,
    *,
    source_xyz: np.ndarray | None = None,
) -> np.ndarray | None:
    """Permutation that sorts stations by ``mode``, or None if not possible.

    Non-finite keys sort last. ``offset`` uses the ``offset`` column if present,
    else horizontal distance from ``source_xyz``.
    """
    n = len(stations)
    if mode == "input":
        return np.arange(n)
    if mode == "grid":
        if "gline" not in stations or "gpoint" not in stations:
            return None
        primary = stations["gline"].to_numpy(dtype=float)
        secondary = stations["gpoint"].to_numpy(dtype=float)
        if not (np.isfinite(primary).any() or np.isfinite(secondary).any()):
            return None
        return np.lexsort((_last(secondary), _last(primary)))
    if mode == "offset":
        if "offset" in stations:
            keys = stations["offset"].to_numpy(dtype=float)
        elif source_xyz is not None:
            src = np.asarray(source_xyz, dtype=float)
            keys = np.hypot(
                stations["x"].to_numpy(dtype=float) - src[0],
                stations["y"].to_numpy(dtype=float) - src[1],
            )
        else:
            return None
    elif mode == "traveltime":
        if "travel_time" not in stations:
            return None
        keys = stations["travel_time"].to_numpy(dtype=float)
    elif mode == "elevation":
        keys = -stations["z"].to_numpy(dtype=float)  # shallowest first
    else:
        raise ValueError(f"unknown station sort {mode!r}; expected one of {SORT_MODES}")
    if not np.isfinite(keys).any():
        return None
    return np.argsort(_last(keys), kind="stable")


def sort_gather(gather: Gather, mode: str) -> tuple[Gather, bool]:
    """Sorted copy and whether the sort could be applied."""
    order = station_order(
        gather.stations,
        mode,
        source_xyz=None if gather.source_xyz is None else np.asarray(gather.source_xyz),
    )
    if order is None:
        return gather, False
    return gather.take_stations(order), True


def _last(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, np.inf)


# --- amplitude --------------------------------------------------------------
def time_window(gather: Gather, tmin: float | None, tmax: float | None) -> slice:
    """Sample slice covering ``[tmin, tmax]`` seconds (clipped to the data)."""
    i0 = 0 if tmin is None else int(np.floor((tmin - gather.t0) / gather.dt))
    i1 = gather.n_samples if tmax is None else int(np.ceil((tmax - gather.t0) / gather.dt)) + 1
    i0 = min(max(i0, 0), gather.n_samples)
    i1 = min(max(i1, i0), gather.n_samples)
    return slice(i0, i1)


def clip_level(
    data: np.ndarray,
    *,
    window: slice | None = None,
    percentile: float = 99.0,
    max_values: int = 200_000,
) -> float:
    """Display clip from ``|data|`` in a sample window, on a strided subsample.

    Scaling on a window around the arrival keeps pre-event noise bursts and
    the S coda from setting the gain. Returns 1.0 when nothing is finite.
    """
    sub = data[..., window] if window is not None else data
    flat = np.abs(np.asarray(sub, dtype=float)).ravel()
    if flat.size > max_values:
        flat = flat[:: int(np.ceil(flat.size / max_values))]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.0
    level = float(np.percentile(flat, percentile))
    return level if level > 0 else 1.0


def normalize_traces(data: np.ndarray, *, window: slice | None = None) -> np.ndarray:
    """Scale each trace (last axis) to unit max |amplitude| inside ``window``."""
    sub = data[..., window] if window is not None else data
    peak = np.nanmax(np.abs(sub), axis=-1, keepdims=True)
    peak = np.where(np.isfinite(peak) & (peak > 0), peak, 1.0)
    return data / peak


def stack(traces: np.ndarray, *, normalize: bool = True, window: slice | None = None) -> np.ndarray:
    """Mean over stations of ``(station, sample)`` traces.

    With ``normalize`` each trace is first scaled to unit peak, so near
    stations do not dominate.
    """
    work = normalize_traces(traces, window=window) if normalize else traces
    with np.errstate(invalid="ignore"):
        return np.nanmean(work, axis=0)


# --- display decimation -----------------------------------------------------
def decimate_minmax(data: np.ndarray, max_len: int, axis: int = -1) -> np.ndarray:
    """Reduce ``axis`` to at most ``max_len`` bins, keeping the signed value
    with the largest magnitude in each bin. For display only."""
    data = np.asarray(data)
    axis = axis % data.ndim
    n = data.shape[axis]
    if max_len <= 0 or n <= max_len:
        return data
    factor = int(np.ceil(n / max_len))
    pad = (-n) % factor
    moved = np.moveaxis(data, axis, -1)
    if pad:
        widths = [(0, 0)] * (moved.ndim - 1) + [(0, pad)]
        moved = np.pad(moved, widths, mode="constant", constant_values=0)
    blocks = moved.reshape(*moved.shape[:-1], -1, factor)
    pick = np.nanargmax(np.abs(np.nan_to_num(blocks, nan=0.0)), axis=-1)
    out = np.take_along_axis(blocks, pick[..., None], axis=-1)[..., 0]
    return np.moveaxis(out, -1, axis)


def decimate_for_display(traces: np.ndarray, max_stations: int, max_samples: int) -> np.ndarray:
    """``(station, sample)`` traces reduced to fit a panel of the given size."""
    out = decimate_minmax(traces, max_samples, axis=1)
    return decimate_minmax(out, max_stations, axis=0)
