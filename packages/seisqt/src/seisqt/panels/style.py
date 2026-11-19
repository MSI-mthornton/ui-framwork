"""Shared colours so every app tags and plots the same way."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

# One look for every app: light background, no antialiasing on big scatters.
pg.setConfigOptions(background="w", foreground="k", antialias=False)

QC_COLORS: dict[str, tuple[int, int, int]] = {
    "": (150, 150, 150),
    "keeper": (30, 160, 60),
    "real_event": (30, 160, 60),
    "reviewed_event": (30, 160, 60),
    "not_keeper": (200, 120, 40),
    "noise": (200, 50, 50),
}
_FALLBACK = [(70, 110, 200), (150, 80, 180), (20, 150, 160), (180, 160, 30)]


def qc_color(code: str) -> tuple[int, int, int]:
    if code in QC_COLORS:
        return QC_COLORS[code]
    return _FALLBACK[sum(map(ord, code)) % len(_FALLBACK)]


def colormap(name: str) -> pg.ColorMap:
    if name == "seismic":
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        colors = np.array(
            [[0, 0, 80], [0, 0, 255], [255, 255, 255], [255, 0, 0], [128, 0, 0]], dtype=np.ubyte
        )
        return pg.ColorMap(pos, colors)
    if name == "gray":
        return pg.ColorMap(
            np.array([0.0, 1.0]), np.array([[0, 0, 0], [255, 255, 255]], dtype=np.ubyte)
        )
    try:
        return pg.colormap.get(name)
    except Exception:  # noqa: BLE001 - unknown name falls back
        return colormap("seismic")
