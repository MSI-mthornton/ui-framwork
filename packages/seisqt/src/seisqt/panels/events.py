"""Event scatter panels (map view and time view).

Sized for 1e5 events: points are grouped into one ScatterPlotItem per QC code
(uniform brush, no per-point styling), the selection is a separate one-point
overlay so moving it never redraws the cloud, and a QC change only rebuilds
the two affected groups. Clicks pick the nearest event in screen space.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from seiscore.model import EventTable
from seisqt.panels.style import qc_color

Axis = Callable[[EventTable], np.ndarray]


class EventScatterPanel(QWidget):
    def __init__(
        self,
        ctx,
        x: Axis,
        y: Axis,
        *,
        x_label: str = "",
        y_label: str = "",
        lock_aspect: bool = False,
        invert_y: bool = False,
        point_size: float = 4.0,
        axis_items: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.x_of, self.y_of = x, y
        self.point_size = point_size
        self.plot = pg.PlotWidget(axisItems=axis_items or {})
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        if lock_aspect:
            self.plot.setAspectLocked(True)
        if invert_y:
            self.plot.invertY(True)
        self.legend = self.plot.addLegend(offset=(5, 5))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

        self._x = np.empty(0)
        self._y = np.empty(0)
        self._codes = np.empty(0, dtype=object)
        self._groups: dict[str, pg.ScatterPlotItem] = {}
        self._marker = pg.ScatterPlotItem(
            size=14, symbol="o", pen=pg.mkPen("k", width=2), brush=pg.mkBrush(255, 220, 0, 220)
        )
        self._marker.setZValue(10)
        self.plot.addItem(self._marker)

        ctx.events.changed.connect(self.set_table)
        ctx.events.rowChanged.connect(self._on_row)
        ctx.selection.changed.connect(self._on_selection)
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.set_table(ctx.events.table)

    # --- data ----------------------------------------------------------------
    def set_table(self, table: EventTable) -> None:
        self._x = np.asarray(self.x_of(table), dtype=float)
        self._y = np.asarray(self.y_of(table), dtype=float)
        self._codes = np.asarray(table.qc_code, dtype=object)
        for item in self._groups.values():
            self.plot.removeItem(item)
            self.legend.removeItem(item)
        self._groups.clear()
        for code in sorted(set(self._codes.tolist())):
            self._rebuild(code)
        self.plot.enableAutoRange()
        self._on_selection(self.ctx.selection.index)

    def _rebuild(self, code: str) -> None:
        mask = self._codes == code
        item = self._groups.get(code)
        if not mask.any():
            if item is not None:
                self.plot.removeItem(item)
                self.legend.removeItem(item)
                del self._groups[code]
            return
        if item is None:
            item = pg.ScatterPlotItem(
                pen=None,
                size=self.point_size,
                name=code or "untagged",
                brush=pg.mkBrush(*qc_color(code), 200),
            )
            item.setZValue(1 if code else 0)
            self._groups[code] = item
            self.plot.addItem(item)
        item.setData(self._x[mask], self._y[mask])

    def _on_row(self, index: int) -> None:
        old = self._codes[index]
        new = self.ctx.events.table.qc_code[index]
        if old == new:
            return
        self._codes[index] = new
        self._rebuild(old)
        self._rebuild(new)

    def _on_selection(self, index: int) -> None:
        if 0 <= index < self._x.size:
            self._marker.setData([self._x[index]], [self._y[index]])
        else:
            self._marker.setData([], [])

    # --- picking -------------------------------------------------------------
    def _on_click(self, ev) -> None:
        if self._x.size == 0 or ev.button() != pg.QtCore.Qt.MouseButton.LeftButton:
            return
        vb = self.plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        index = self.nearest(ev.scenePos())
        if index is not None:
            self.ctx.selection.set(index)

    def nearest(self, scene_pos, max_px: float = 15.0) -> int | None:
        vb = self.plot.getPlotItem().vb
        p = vb.mapSceneToView(scene_pos)
        (xr0, xr1), (yr0, yr1) = vb.viewRange()
        rect = vb.sceneBoundingRect()
        sx = rect.width() / max(xr1 - xr0, 1e-12)
        sy = rect.height() / max(yr1 - yr0, 1e-12)
        d2 = ((self._x - p.x()) * sx) ** 2 + ((self._y - p.y()) * sy) ** 2
        d2 = np.where(np.isfinite(d2), d2, np.inf)
        i = int(np.argmin(d2))
        return i if d2[i] <= max_px**2 else None


class EventMapPanel(EventScatterPanel):
    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(
            ctx,
            lambda t: t.x,
            lambda t: t.y,
            x_label="Easting (m)",
            y_label="Northing (m)",
            lock_aspect=True,
            parent=parent,
        )


class EventTimePanel(EventScatterPanel):
    """Origin time against a value (SNR by default)."""

    def __init__(
        self,
        ctx,
        value: Axis | None = None,
        value_label: str = "SNR",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            ctx,
            lambda t: t.origin_seconds,
            value or (lambda t: t.snr),
            x_label="Origin time (UTC)",
            y_label=value_label,
            axis_items={"bottom": pg.DateAxisItem(utcOffset=0)},
            parent=parent,
        )
