"""Reusable panels. Each takes the :class:`~seisqt.app.AppContext` and only
reads stores / calls controllers, so any app can mix and match them."""

from seisqt.panels.events import EventMapPanel, EventScatterPanel, EventTimePanel
from seisqt.panels.gather import GatherPanel, prepare_display
from seisqt.panels.log import LogPanel
from seisqt.panels.qc_tags import QcTagBar

__all__ = [
    "EventMapPanel",
    "EventScatterPanel",
    "EventTimePanel",
    "GatherPanel",
    "LogPanel",
    "QcTagBar",
    "prepare_display",
]
