"""demo-viewer: the reference app. Synthetic data, no Pulse needed.

Copy this file to start a new app: change the config, swap the Fake sources
for Pulse/PSET ones, and pick panels.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from demo_viewer.config import PACKAGED, DemoConfig
from seiscore.sources.base import GatherRequest, Sources
from seiscore.sources.fake import FakeEventSource, FakeGatherSource, FakeQcSink, grid_array
from seisqt.app import AppContext, BaseApp, BaseMainWindow, Command
from seisqt.panels import EventMapPanel, EventTimePanel, GatherPanel, LogPanel, QcTagBar


class DemoViewer(BaseApp):
    name = "demo-viewer"
    config_cls = DemoConfig
    packaged_config = PACKAGED
    env_prefix = "DEMO"

    def sources(self, config: DemoConfig) -> Sources:
        f = config.fake
        stations = grid_array(f.n_lines, f.n_points, f.spacing)
        return Sources(
            events=FakeEventSource(f.n_events, latency_s=f.latency_s),
            gathers=FakeGatherSource(stations, noise=f.noise, latency_s=f.latency_s),
            qc=FakeQcSink(latency_s=f.latency_s / 2),
        )

    def gather_request(self, config: DemoConfig) -> GatherRequest:
        g = config.gather
        return GatherRequest(
            event_id="", pre=g.pre, post=g.post, alignment=g.alignment, phase=g.phase
        )

    def make_context(self, config: DemoConfig, log_handler=None) -> AppContext:
        ctx = AppContext.create(
            self.name,
            config,
            self.sources(config),
            gather_request=self.gather_request(config),
            qc_codes=config.qc.codes,
            auto_advance=config.qc.auto_advance,
            log_handler=log_handler,
        )
        return ctx

    def build(self, ctx: AppContext) -> BaseMainWindow:
        win = BaseMainWindow(ctx, organization=self.organization)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        tags = QcTagBar(ctx)
        layout.addWidget(tags)
        layout.addWidget(GatherPanel(ctx), 1)
        win.set_central(central)
        win.add_panel(EventMapPanel(ctx), "Map", area=Qt.DockWidgetArea.LeftDockWidgetArea)
        win.add_panel(EventTimePanel(ctx), "Timeline", area=Qt.DockWidgetArea.LeftDockWidgetArea)
        win.add_panel(LogPanel(ctx), "Log", area=Qt.DockWidgetArea.BottomDockWidgetArea)
        for cmd in tags.commands():
            win.add_command(cmd)
        prefs = ctx.prefs
        for n, comp in enumerate(("P", "SH", "SV")):
            win.add_command(
                Command(
                    f"view.{comp}",
                    f"Show {comp}",
                    lambda c=comp: prefs.update(component=c),
                    f"Ctrl+{n + 1}",
                    "&View",
                )
            )
        win.add_command(
            Command(
                "view.gain_up",
                "More gain",
                lambda: prefs.update(gain=prefs.prefs.gain * 1.5),
                "Up",
                "&View",
            )
        )
        win.add_command(
            Command(
                "view.gain_down",
                "Less gain",
                lambda: prefs.update(gain=prefs.prefs.gain / 1.5),
                "Down",
                "&View",
            )
        )
        win.resize(1400, 900)
        return win


def main(argv: list[str] | None = None) -> int:
    return DemoViewer().main(argv)


if __name__ == "__main__":
    sys.exit(main())
