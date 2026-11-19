"""Qt-free core for MSI seismic apps.

Layers (each may import only the ones above it):

- ``seiscore.model``        domain types: EventTable, Event, Gather
- ``seiscore.conventions``  coordinate frame, component names, ray frame
- ``seiscore.dsp``          rotation, scaling, stacking, sorting, display decimation
- ``seiscore.config``       layered TOML/env/CLI config into frozen dataclasses
- ``seiscore.sources``      data-source protocols and Fake / Pulse / PSET implementations
"""

from seiscore.model import Event, EventTable, Gather

__all__ = ["Event", "EventTable", "Gather"]
__version__ = "0.1.0"
