"""Gather panel: one component as an image, with a normalised stack below.

The full-rate gather is never drawn directly. :func:`prepare_display` sorts,
picks the component, computes the clip level and stack on full-rate samples,
then min/max-decimates stations and samples to the panel budget. It runs on a
worker through the task runner, so a 5e3-station gather does not freeze the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from seiscore import dsp
from seiscore.model import Gather
from seisqt.panels.style import colormap
from seisqt.stores import ViewPrefs


@dataclass(frozen=True)
class GatherDisplay:
    image: np.ndarray  # (station_bins, sample_bins)
    stack: np.ndarray | None  # full-rate
    times: np.ndarray
    t0: float
    t1: float
    n_stations: int
    clip: float
    component: str
    sort_applied: bool
    event_id: str
    reference: str


def prepare_display(
    gather: Gather, prefs: ViewPrefs, *, clip_window: tuple[float, float] = (-0.05, 0.15)
) -> GatherDisplay:
    """Everything the panel draws, computed off the GUI thread."""
    g, applied = dsp.sort_gather(gather, prefs.sort)
    comp = prefs.component if prefs.component in g.components else g.components[0]
    traces = g.component(comp)
    window = dsp.time_window(g, *clip_window) if g.reference.startswith("arrival") else None
    clip = dsp.clip_level(traces, window=window, percentile=prefs.clip_percentile)
    stacked = dsp.stack(traces) if prefs.show_stack else None
    image = dsp.decimate_for_display(traces, prefs.max_display_stations, prefs.max_display_samples)
    return GatherDisplay(
        image=np.ascontiguousarray(image, dtype=np.float32),
        stack=None if stacked is None else stacked.astype(np.float32),
        times=g.times,
        t0=float(g.t0),
        t1=float(g.t0 + g.dt * g.n_samples),
        n_stations=g.n_stations,
        clip=clip,
        component=comp,
        sort_applied=applied,
        event_id=g.event_id,
        reference=g.reference,
    )


class GatherPanel(QWidget):
    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.display: GatherDisplay | None = None
        self._gather: Gather | None = None

        self.component = QComboBox()
        self.sort = QComboBox()
        self.sort.addItems(list(dsp.SORT_MODES))
        self.clip = QDoubleSpinBox()
        self.clip.setRange(50.0, 100.0)
        self.clip.setDecimals(1)
        self.clip.setSuffix(" %")
        self.gain = QDoubleSpinBox()
        self.gain.setRange(0.05, 50.0)
        self.gain.setSingleStep(0.25)
        self.gain.setPrefix("gain ")
        self.status = QLabel("")
        bar = QHBoxLayout()
        for label, w in (
            ("Component", self.component),
            ("Sort", self.sort),
            ("Clip", self.clip),
            ("", self.gain),
        ):
            if label:
                bar.addWidget(QLabel(label))
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.status)

        self.graphics = pg.GraphicsLayoutWidget()
        self.image_plot = self.graphics.addPlot(row=0, col=0)
        self.image_plot.invertY(True)
        self.image_plot.setLabel("left", "Time (s)")
        self.image_plot.getAxis("left").enableAutoSIPrefix(False)
        self.image_plot.setLabel("bottom", "Station")
        self.image = pg.ImageItem(axisOrder="col-major")
        self.image_plot.addItem(self.image)
        self.zero_line = pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen((0, 160, 0), width=1))
        self.image_plot.addItem(self.zero_line)
        self.stack_plot = self.graphics.addPlot(row=0, col=1)
        self.stack_plot.invertY(True)
        self.stack_plot.setYLink(self.image_plot)
        self.stack_plot.setLabel("bottom", "Stack")
        self.stack_plot.hideAxis("left")
        self.stack_curve = self.stack_plot.plot(pen=pg.mkPen("k", width=1))
        self.graphics.ci.layout.setColumnStretchFactor(0, 5)
        self.graphics.ci.layout.setColumnStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.graphics, 1)

        self._sync_controls(ctx.prefs.prefs)
        self.component.currentTextChanged.connect(lambda v: v and ctx.prefs.update(component=v))
        self.sort.currentTextChanged.connect(lambda v: ctx.prefs.update(sort=v))
        self.clip.valueChanged.connect(lambda v: ctx.prefs.update(clip_percentile=float(v)))
        self.gain.valueChanged.connect(lambda v: ctx.prefs.update(gain=float(v)))
        ctx.prefs.changed.connect(self._on_prefs)
        ctx.gathers_ctl.ready.connect(self.set_gather)
        ctx.gathers_ctl.loading.connect(lambda eid: self.status.setText(f"loading {eid}…"))
        ctx.gathers_ctl.failed.connect(lambda eid, msg: self.status.setText(f"failed: {msg}"))
        ctx.gathers_ctl.cleared.connect(self.clear)

    # --- controls ----------------------------------------------------------
    def _sync_controls(self, prefs: ViewPrefs) -> None:
        for w in (self.sort, self.clip, self.gain):
            w.blockSignals(True)
        self.sort.setCurrentText(prefs.sort)
        self.clip.setValue(prefs.clip_percentile)
        self.gain.setValue(prefs.gain)
        for w in (self.sort, self.clip, self.gain):
            w.blockSignals(False)

    def _on_prefs(self, prefs: ViewPrefs) -> None:
        self._sync_controls(prefs)
        if self.display is not None and self._only_levels_changed(prefs):
            self._apply_levels()
        else:
            self._prepare()

    def _only_levels_changed(self, prefs: ViewPrefs) -> bool:
        d = self.display
        return (
            d is not None
            and d.component == prefs.component
            and prefs.colormap == self._cmap_name
            and self._prepared_with is not None
            and {k: v for k, v in vars(self._prepared_with).items() if k != "gain"}
            == {k: v for k, v in vars(prefs).items() if k != "gain"}
        )

    # --- data --------------------------------------------------------------
    _prepared_with: ViewPrefs | None = None
    _cmap_name: str = ""

    def set_gather(self, gather: Gather) -> None:
        self._gather = gather
        comps = list(gather.components)
        if [self.component.itemText(i) for i in range(self.component.count())] != comps:
            self.component.blockSignals(True)
            self.component.clear()
            self.component.addItems(comps)
            self.component.blockSignals(False)
        want = self.ctx.prefs.prefs.component
        self.component.blockSignals(True)
        self.component.setCurrentText(want if want in comps else comps[0])
        self.component.blockSignals(False)
        self._prepare()

    def _prepare(self) -> None:
        if self._gather is None:
            return
        gather, prefs = self._gather, self.ctx.prefs.prefs
        self.ctx.runner.submit(
            lambda: (prefs, prepare_display(gather, prefs)),
            self._prepared,
            key="gather-display",
            label="draw gather",
        )

    def _prepared(self, res) -> None:
        if not res.ok:
            self.status.setText(f"draw failed: {res.message}")
            return
        self._prepared_with, display = res.value
        self.show_display(display)

    def show_display(self, d: GatherDisplay) -> None:
        self.display = d
        prefs = self.ctx.prefs.prefs
        self._cmap_name = prefs.colormap
        self.image.setImage(d.image, autoLevels=False)
        self.image.setColorMap(colormap(prefs.colormap))
        self.image.setRect(QRectF(0.0, d.t0, float(d.n_stations), d.t1 - d.t0))
        self._apply_levels()
        if d.stack is not None:
            self.stack_curve.setData(d.stack, d.times)
            self.stack_plot.show()
        else:
            self.stack_curve.setData([], [])
            self.stack_plot.hide()
        self.zero_line.setVisible(d.reference.startswith("arrival"))
        self.image_plot.setTitle(f"{d.event_id}  {d.component}  ({d.reference})")
        note = "" if d.sort_applied else " (sort unavailable)"
        self.status.setText(
            f"{d.n_stations} stations, image {d.image.shape[0]}×{d.image.shape[1]}{note}"
        )
        self.image_plot.setXRange(0, d.n_stations, padding=0)
        self.image_plot.setYRange(d.t0, d.t1, padding=0)

    def _apply_levels(self) -> None:
        if self.display is None:
            return
        level = self.display.clip / max(self.ctx.prefs.prefs.gain, 1e-6)
        self.image.setLevels((-level, level))

    def clear(self) -> None:
        self._gather = None
        self.display = None
        self.image.clear()
        self.stack_curve.setData([], [])
        self.image_plot.setTitle("")
        self.status.setText("")
