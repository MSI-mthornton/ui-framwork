"""Logging setup, a Qt-aware log handler, and an excepthook that never kills the app."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QObject, Signal

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class QtLogBridge(QObject):
    record = Signal(int, str)  # level, formatted message


class QtLogHandler(logging.Handler):
    """Forwards records to a signal; safe from any thread (queued delivery)."""

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.bridge = QtLogBridge()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(RuntimeError):  # bridge deleted during shutdown
            self.bridge.record.emit(record.levelno, self.format(record))


def setup_logging(
    app_name: str, *, level: int = logging.INFO, log_dir: Path | None = None
) -> QtLogHandler:
    """Console + rotating file + Qt handler. Returns the Qt handler for the log panel."""
    root = logging.getLogger()
    root.setLevel(min(level, logging.DEBUG))
    for h in list(root.handlers):
        if getattr(h, "_seisqt", False):
            root.removeHandler(h)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(FORMAT))
    handlers: list[logging.Handler] = [console]
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                log_dir / f"{app_name}.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(FORMAT))
            handlers.append(fh)
        except OSError as exc:
            print(f"warning: cannot write logs to {log_dir}: {exc}", file=sys.stderr)
    qt = QtLogHandler(level)
    handlers.append(qt)
    for h in handlers:
        h._seisqt = True  # type: ignore[attr-defined]
        root.addHandler(h)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return qt


def install_excepthook() -> None:
    """Log uncaught exceptions (GUI and worker threads) instead of aborting.

    PySide6 calls ``sys.excepthook`` for exceptions raised in slots; the
    default prints and carries on, but an unhandled one in some code paths
    terminates. Logging them keeps the session alive and the log panel honest.
    """
    log = logging.getLogger("seisqt.uncaught")

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.error("uncaught %s: %s\n%s", exc_type.__name__, exc, "".join(traceback.format_tb(tb)))

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        hook(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = hook
    threading.excepthook = thread_hook
