"""Data sources: protocols plus Fake, Pulse and PSET implementations.

Pulse and PSET modules are imported on demand (``seiscore.sources.pulse``,
``seiscore.sources.pset``) so the core does not need their dependencies.
"""

from seiscore.sources.base import (
    EventSource,
    GatherRequest,
    GatherSource,
    QcSink,
    SourceError,
    Sources,
    TimeWindow,
)

__all__ = [
    "EventSource",
    "GatherRequest",
    "GatherSource",
    "QcSink",
    "SourceError",
    "Sources",
    "TimeWindow",
]
