from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest

from seiscore.config import ConfigError, load_config


@dataclass(frozen=True)
class Pulse:
    host: str = "hou-dok-1"
    port: int = 8000
    verify: bool = True


@dataclass(frozen=True)
class Cfg:
    pulse: Pulse = field(default_factory=Pulse)
    codes: tuple[str, ...] = ("keeper",)
    start: datetime | None = None
    pset_list: Path | None = None


def test_layers_and_precedence(tmp_path):
    packaged = tmp_path / "defaults.toml"
    packaged.write_text('[pulse]\nhost = "a"\nport = 1\n')
    project = tmp_path / "proj" / "project.toml"
    project.parent.mkdir()
    project.write_text('pset_list = "files.csv"\nstart = 2026-01-02\n[pulse]\nhost = "b"\n')
    env = {"APP_PULSE__PORT": "9000", "APP_PULSE__VERIFY": "no", "OTHER_X": "1"}
    cfg = load_config(
        Cfg,
        packaged=packaged,
        project=project,
        env_prefix="APP",
        environ=env,
        overrides={"codes": "keeper, noise", "pulse.host": None},
    )
    assert cfg.pulse == Pulse(host="b", port=9000, verify=False)
    assert cfg.codes == ("keeper", "noise")
    assert cfg.pset_list == (project.parent / "files.csv").resolve()
    assert cfg.start.isoformat() == "2026-01-02T00:00:00+00:00"


def test_unknown_keys_fail(tmp_path):
    with pytest.raises(ConfigError, match="hostt"):
        load_config(Cfg, overrides={"pulse.hostt": "x"}, environ={})
    with pytest.raises(ConfigError):
        load_config(Cfg, overrides={"pulse.port": "eighty"})


def test_defaults_only():
    assert load_config(Cfg) == Cfg()
