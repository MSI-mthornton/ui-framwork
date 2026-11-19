"""Demo settings. Every app declares its config the same way: frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FakeSettings:
    n_events: int = 2000
    n_lines: int = 12
    n_points: int = 25
    spacing: float = 100.0
    latency_s: float = 0.15
    noise: float = 0.05


@dataclass(frozen=True)
class GatherSettings:
    pre: float = 0.3
    post: float = 0.7
    alignment: str = "arrival"
    phase: str = "P"


@dataclass(frozen=True)
class QcSettings:
    codes: tuple[str, ...] = ("keeper", "noise", "review")
    auto_advance: bool = True


@dataclass(frozen=True)
class DemoConfig:
    fake: FakeSettings = field(default_factory=FakeSettings)
    gather: GatherSettings = field(default_factory=GatherSettings)
    qc: QcSettings = field(default_factory=QcSettings)


PACKAGED = Path(__file__).with_name("defaults.toml")
