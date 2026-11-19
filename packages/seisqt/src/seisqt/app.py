"""App skeleton: context, commands, base window and ``BaseApp.main``.

An app subclasses :class:`BaseApp` and supplies three things: its config
dataclass, its sources, and its panels/commands. Everything else (threads,
caching, logging, settings, shutdown) is shared.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import QApplication, QDockWidget, QLabel, QMainWindow, QMessageBox, QWidget

from seiscore.config import ConfigError, load_config
from seiscore.sources.base import GatherRequest, Sources
from seisqt.cache import GatherCache
from seisqt.controllers import EventsController, GatherController, QcController
from seisqt.diagnostics import DisplayError, require_display
from seisqt.logs import QtLogHandler, install_excepthook, setup_logging
from seisqt.stores import EventStore, PrefsStore, Selection
from seisqt.tasks import Result, TaskRunner

log = logging.getLogger(__name__)


@dataclass
class Command:
    """A user action: one place for its text, shortcut, menu and callback."""

    id: str
    text: str
    callback: Callable[[], Any]
    shortcut: str = ""
    menu: str = "&View"
    tip: str = ""


@dataclass
class AppContext:
    """Everything a panel or controller may need. Built once per app run."""

    name: str
    config: Any
    sources: Sources
    runner: TaskRunner
    cache: GatherCache
    events: EventStore
    selection: Selection
    prefs: PrefsStore
    events_ctl: EventsController
    gathers_ctl: GatherController
    qc_ctl: QcController
    log_handler: QtLogHandler | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        config: Any,
        sources: Sources,
        *,
        gather_request: GatherRequest | None = None,
        qc_codes: Sequence[str] = ("keeper", "noise"),
        cache_bytes: int = 1_500_000_000,
        max_threads: int = 4,
        gather_timeout: float | None = 60.0,
        auto_advance: bool = True,
        log_handler: QtLogHandler | None = None,
    ) -> AppContext:
        runner = TaskRunner(max_threads=max_threads)
        cache = GatherCache(cache_bytes)
        events = EventStore()
        selection = Selection(events)
        return cls(
            name=name,
            config=config,
            sources=sources,
            runner=runner,
            cache=cache,
            events=events,
            selection=selection,
            prefs=PrefsStore(),
            events_ctl=EventsController(runner, sources, events),
            gathers_ctl=GatherController(
                runner, sources, selection, cache, gather_request, timeout=gather_timeout
            ),
            qc_ctl=QcController(
                runner, sources, events, selection, codes=qc_codes, auto_advance=auto_advance
            ),
            log_handler=log_handler,
        )


class BaseMainWindow(QMainWindow):
    """Docked panels, a command registry, remembered layout and a status bar."""

    def __init__(self, ctx: AppContext, *, organization: str = "MicroSeismic") -> None:
        super().__init__()
        self.ctx = ctx
        self.settings = QSettings(organization, ctx.name)
        self.commands: dict[str, QAction] = {}
        self._menus: dict[str, Any] = {}
        self.setWindowTitle(ctx.name)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks | QMainWindow.DockOption.AllowTabbedDocks
        )
        self._busy = QLabel("")
        self._where = QLabel("")
        self.statusBar().addPermanentWidget(self._where)
        self.statusBar().addPermanentWidget(self._busy)
        ctx.runner.busyChanged.connect(self._on_busy)
        ctx.runner.taskFailed.connect(self._on_task_failed)
        ctx.selection.changed.connect(self._on_selection)
        ctx.events_ctl.failed.connect(self.show_error)
        ctx.gathers_ctl.failed.connect(lambda eid, msg: self.flash(f"gather {eid}: {msg}"))
        ctx.qc_ctl.failed.connect(lambda eid, msg: self.flash(msg))
        self._standard_commands()

    # --- building ----------------------------------------------------------
    def set_central(self, widget: QWidget) -> None:
        self.setCentralWidget(widget)

    def add_panel(
        self,
        widget: QWidget,
        title: str,
        *,
        area: Qt.DockWidgetArea = Qt.DockWidgetArea.LeftDockWidgetArea,
        name: str | None = None,
    ) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(name or f"dock-{title.lower().replace(' ', '-')}")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        self.menu("&Panels").addAction(dock.toggleViewAction())
        return dock

    def menu(self, name: str):
        if name not in self._menus:
            self._menus[name] = self.menuBar().addMenu(name)
        return self._menus[name]

    def add_command(self, cmd: Command) -> QAction:
        action = QAction(cmd.text, self)
        action.setObjectName(cmd.id)
        if cmd.shortcut:
            action.setShortcut(QKeySequence(cmd.shortcut))
            action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        if cmd.tip:
            action.setStatusTip(cmd.tip)
        action.triggered.connect(lambda _=False: cmd.callback())
        self.menu(cmd.menu).addAction(action)
        self.addAction(action)
        self.commands[cmd.id] = action
        return action

    def _standard_commands(self) -> None:
        sel, ctx = self.ctx.selection, self.ctx
        for cmd in (
            Command("event.next", "Next event", lambda: sel.step(1), "Right", "&Event"),
            Command("event.prev", "Previous event", lambda: sel.step(-1), "Left", "&Event"),
            Command("event.first", "First event", lambda: sel.set(0), "Home", "&Event"),
            Command(
                "event.last", "Last event", lambda: sel.set(len(ctx.events) - 1), "End", "&Event"
            ),
            Command("events.reload", "Reload events", ctx.events_ctl.load, "Ctrl+R", "&File"),
            Command("app.quit", "Quit", self.close, "Ctrl+Q", "&File"),
        ):
            self.add_command(cmd)

    # --- state -------------------------------------------------------------
    def restore_layout(self) -> None:
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)
        self.ctx.prefs.restore(self.settings)

    def save_layout(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.ctx.prefs.save(self.settings)
        self.settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        pending = sum(self.ctx.qc_ctl.pending.values())
        if pending:
            log.warning("closing with %d QC write(s) in flight", pending)
        self.save_layout()
        self.ctx.runner.shutdown()
        super().closeEvent(event)

    # --- feedback ----------------------------------------------------------
    def flash(self, message: str, msecs: int = 8000) -> None:
        self.statusBar().showMessage(message, msecs)

    def show_error(self, message: str) -> None:
        log.error(message)
        QMessageBox.warning(self, self.ctx.name, message)

    def _on_busy(self, n: int) -> None:
        self._busy.setText(f"working ({n})" if n else "")

    def _on_task_failed(self, label: str, res: Result) -> None:
        self.flash(f"{label}: {res.message}")

    def _on_selection(self, index: int) -> None:
        ev = self.ctx.selection.event
        self._where.setText(
            ""
            if ev is None
            else f"{index + 1}/{len(self.ctx.events)}  {ev.id}  {ev.qc_code or '-'}"
        )


class BaseApp:
    """Subclass and set ``name``/``config_cls``; implement ``sources`` and ``build``."""

    name: ClassVar[str] = "seisqt-app"
    organization: ClassVar[str] = "MicroSeismic"
    config_cls: ClassVar[type | None] = None
    packaged_config: ClassVar[Path | None] = None
    env_prefix: ClassVar[str | None] = None
    qc_codes: ClassVar[tuple[str, ...]] = ("keeper", "noise")

    # --- hooks for apps ------------------------------------------------------
    def sources(self, config: Any) -> Sources:
        raise NotImplementedError

    def gather_request(self, config: Any) -> GatherRequest:
        return GatherRequest(event_id="")

    def build(self, ctx: AppContext) -> BaseMainWindow:
        raise NotImplementedError

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Extra CLI flags; map them into config in :meth:`overrides`."""

    def overrides(self, args: argparse.Namespace) -> dict[str, Any]:
        return {}

    # --- shared plumbing -----------------------------------------------------
    def parse_args(self, argv: Sequence[str] | None) -> argparse.Namespace:
        p = argparse.ArgumentParser(prog=self.name)
        p.add_argument("--project", type=Path, help="project TOML file")
        p.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a config value, e.g. --set pulse.host=hou-dok-2",
        )
        p.add_argument("--log-dir", type=Path, default=None)
        p.add_argument("--debug", action="store_true")
        self.add_arguments(p)
        return p.parse_args(argv)

    def load_config(self, args: argparse.Namespace) -> Any:
        if self.config_cls is None:
            return None
        overrides = self.overrides(args)
        for item in args.set:
            key, sep, value = item.partition("=")
            if not sep:
                raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
            overrides[key.strip()] = value
        return load_config(
            self.config_cls,
            packaged=self.packaged_config,
            project=args.project,
            env_prefix=self.env_prefix,
            overrides=overrides,
        )

    def make_context(self, config: Any, log_handler: QtLogHandler | None = None) -> AppContext:
        return AppContext.create(
            self.name,
            config,
            self.sources(config),
            gather_request=self.gather_request(config),
            qc_codes=self.qc_codes,
            log_handler=log_handler,
        )

    def main(self, argv: Sequence[str] | None = None) -> int:
        args = self.parse_args(argv)
        try:
            require_display()
        except DisplayError as exc:
            print(f"{self.name}: cannot open a window.\n\n{exc}", file=sys.stderr)
            return 2
        try:
            config = self.load_config(args)
        except ConfigError as exc:
            print(f"{self.name}: configuration error: {exc}", file=sys.stderr)
            return 2
        qapp = QApplication.instance() or QApplication(sys.argv[:1])
        qapp.setOrganizationName(self.organization)
        qapp.setApplicationName(self.name)
        handler = setup_logging(
            self.name, level=logging.DEBUG if args.debug else logging.INFO, log_dir=args.log_dir
        )
        install_excepthook()
        ctx = self.make_context(config, handler)
        window = self.build(ctx)
        window.restore_layout()
        window.show()
        ctx.events_ctl.load()
        return int(qapp.exec())
