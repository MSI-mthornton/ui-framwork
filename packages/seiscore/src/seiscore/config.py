"""Layered configuration into frozen dataclasses.

Precedence, lowest to highest::

    dataclass defaults < packaged TOML < project TOML < environment < overrides

An app declares its settings as a frozen dataclass whose fields may be other
frozen dataclasses (one per TOML table). :func:`load_config` merges the layers,
converts types, resolves relative paths against the project file, and rejects
unknown keys so typos fail loudly instead of being ignored.

Environment variables use ``<PREFIX>_<SECTION>__<KEY>``, e.g.
``QC_KEEPERS_PULSE__HOST=hou-dok-1``.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
import types
import typing
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class ConfigError(ValueError):
    """Configuration could not be loaded or did not match the schema."""


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def env_layer(prefix: str, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Nested dict from ``<PREFIX>_SECTION__KEY`` variables (values as strings)."""
    environ = os.environ if environ is None else environ
    head = prefix.upper().rstrip("_") + "_"
    out: dict[str, Any] = {}
    for name, value in environ.items():
        if not name.startswith(head):
            continue
        parts = [p.lower() for p in name[len(head) :].split("__") if p]
        if not parts:
            continue
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def dotted_layer(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Nested dict from ``{"pulse.host": "x"}``; ``None`` values are skipped."""
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def merge(*layers: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer in layers:
        for key, value in layer.items():
            if isinstance(value, Mapping) and isinstance(out.get(key), dict):
                out[key] = merge(out[key], value)
            elif isinstance(value, Mapping):
                out[key] = merge(value)
            else:
                out[key] = value
    return out


def load_config(
    cls: type[T],
    *,
    packaged: Path | None = None,
    project: Path | None = None,
    env_prefix: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> T:
    """Build ``cls`` from the layers. ``project`` paths anchor relative paths."""
    layers: list[Mapping[str, Any]] = []
    if packaged is not None:
        layers.append(read_toml(packaged))
    if project is not None:
        layers.append(read_toml(project))
    if env_prefix:
        layers.append(env_layer(env_prefix, environ))
    if overrides:
        layers.append(dotted_layer(overrides))
    base = Path(project).resolve().parent if project is not None else Path.cwd()
    return build(cls, merge(*layers), base_dir=base)


def build(cls: type[T], data: Mapping[str, Any], *, base_dir: Path, where: str = "") -> T:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in [{where or 'top level'}]")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        label = f"{where}.{name}" if where else name
        kwargs[name] = _convert(value, hints[name], base_dir=base_dir, where=label)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"[{where or 'top level'}]: {exc}") from None


def _convert(value: Any, hint: Any, *, base_dir: Path, where: str) -> Any:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if type(None) in args and (value is None or value == ""):
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _convert(value, arg, base_dir=base_dir, where=where)
            except (ConfigError, ValueError, TypeError) as exc:
                errors.append(str(exc))
        raise ConfigError(f"{where}: cannot convert {value!r} ({'; '.join(errors)})")
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a table, got {value!r}")
        return build(hint, value, base_dir=base_dir, where=where)
    if origin is tuple:
        items = _as_list(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0], base_dir=base_dir, where=where) for v in items)
        if len(items) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} values, got {len(items)}")
        return tuple(
            _convert(v, a, base_dir=base_dir, where=where) for v, a in zip(items, args, strict=True)
        )
    if origin is list:
        return [_convert(v, args[0], base_dir=base_dir, where=where) for v in _as_list(value)]
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a table")
        return {
            str(k): _convert(v, args[1], base_dir=base_dir, where=f"{where}.{k}")
            for k, v in value.items()
        }
    if origin is typing.Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} is not one of {args}")
        return value
    if hint is Any:
        return value
    if hint is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"{where}: {value!r} is not a boolean")
    if hint is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: {value!r} is not an integer") from None
    if hint is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: {value!r} is not a number") from None
    if hint is str:
        return str(value)
    if hint is Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()
    if hint is datetime:
        return _as_datetime(value, where)
    raise ConfigError(f"{where}: unsupported config type {hint!r}")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list | tuple):
        return list(value)
    raise ConfigError(f"expected a list, got {value!r}")


def _as_datetime(value: Any, where: str) -> datetime:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, date):
        stamp = datetime(value.year, value.month, value.day)
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            raise ConfigError(f"{where}: {value!r} is not an ISO date or datetime") from None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC)
