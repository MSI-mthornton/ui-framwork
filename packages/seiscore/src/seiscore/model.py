"""Domain types shared by every app.

Objects here are treated as immutable: operations return new objects instead
of changing arrays in place. That keeps the GUI thread and worker threads from
racing on the same data, and makes caching safe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

#: Required event columns and their dtypes.
EVENT_COLUMNS: dict[str, str] = {
    "id": "string",
    "origin_time": "datetime64[ns, UTC]",
    "x": "float64",
    "y": "float64",
    "z": "float64",
    "snr": "float64",
}
#: Optional event columns, filled with these defaults when absent.
EVENT_OPTIONAL: dict[str, Any] = {
    "qc_code": "",
    "qc_group": "",
}


class ModelError(ValueError):
    """A domain object was built from inconsistent data."""


@dataclass(frozen=True)
class Event:
    """One event, as a plain value."""

    index: int
    id: str
    origin_time: pd.Timestamp
    x: float
    y: float
    z: float
    snr: float
    qc_code: str = ""
    qc_group: str = ""

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


class EventTable:
    """Events sorted by origin time with a validated schema.

    Wraps a DataFrame. Read columns through the properties (numpy arrays);
    change QC with :meth:`with_qc`, which returns a new table.
    """

    __slots__ = ("_df", "_index_by_id")

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = [c for c in EVENT_COLUMNS if c not in frame.columns]
        if missing:
            raise ModelError(f"event table is missing columns: {missing}")
        df = frame.copy()
        df["id"] = df["id"].astype("string")
        df["origin_time"] = pd.to_datetime(df["origin_time"], utc=True, format="ISO8601")
        for col in ("x", "y", "z", "snr"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        for col, default in EVENT_OPTIONAL.items():
            if col not in df.columns:
                df[col] = default
            df[col] = df[col].fillna(default).astype(str).str.strip().str.lower()
        if df["id"].isna().any():
            raise ModelError("event ids must not be empty")
        if df["id"].duplicated().any():
            dupes = df.loc[df["id"].duplicated(), "id"].head(3).tolist()
            raise ModelError(f"duplicate event ids, e.g. {dupes}")
        df = df.sort_values("origin_time", kind="stable").reset_index(drop=True)
        self._df = df
        self._index_by_id: dict[str, int] | None = None

    # --- construction -------------------------------------------------
    @classmethod
    def empty(cls) -> EventTable:
        frame = pd.DataFrame({c: pd.Series(dtype=d) for c, d in EVENT_COLUMNS.items()})
        return cls(frame)

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]]) -> EventTable:
        if not records:
            return cls.empty()
        return cls(pd.DataFrame.from_records(list(records)))

    # --- access -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._df)

    @property
    def frame(self) -> pd.DataFrame:
        """A copy of the underlying frame."""
        return self._df.copy()

    @property
    def ids(self) -> np.ndarray:
        return self._df["id"].to_numpy(dtype=object)

    @property
    def x(self) -> np.ndarray:
        return self._df["x"].to_numpy()

    @property
    def y(self) -> np.ndarray:
        return self._df["y"].to_numpy()

    @property
    def z(self) -> np.ndarray:
        return self._df["z"].to_numpy()

    @property
    def snr(self) -> np.ndarray:
        return self._df["snr"].to_numpy()

    @property
    def qc_code(self) -> np.ndarray:
        return self._df["qc_code"].to_numpy(dtype=object)

    @property
    def origin_seconds(self) -> np.ndarray:
        """Origin times as float seconds since the Unix epoch (for plotting)."""
        ns = self._df["origin_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        return ns / 1e9

    def event(self, index: int) -> Event:
        if not 0 <= index < len(self):
            raise IndexError(f"event index {index} out of range 0..{len(self) - 1}")
        row = self._df.iloc[index]
        return Event(
            index=index,
            id=str(row["id"]),
            origin_time=row["origin_time"],
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            snr=float(row["snr"]),
            qc_code=str(row["qc_code"]),
            qc_group=str(row["qc_group"]),
        )

    def index_of(self, event_id: str) -> int | None:
        if self._index_by_id is None:
            self._index_by_id = {str(v): i for i, v in enumerate(self._df["id"])}
        return self._index_by_id.get(str(event_id))

    # --- derived tables -----------------------------------------------
    def with_qc(self, index: int, code: str, group: str = "event") -> EventTable:
        """New table with event ``index`` set to QC ``code``."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        df = self._df.copy()
        df.loc[index, "qc_code"] = str(code).strip().lower()
        df.loc[index, "qc_group"] = str(group).strip().lower()
        out = EventTable.__new__(EventTable)
        out._df = df
        out._index_by_id = self._index_by_id
        return out

    def qc_counts(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self._df["qc_code"].value_counts().items()}


@dataclass(frozen=True, eq=False)
class Gather:
    """Traces for one event, one array per component.

    ``data`` has shape ``(component, station, sample)``. ``stations`` has one
    row per station in the same order, with at least ``code``, ``x``, ``y``,
    ``z`` (metres, z up). Time of sample ``k`` is ``t0 + k * dt`` seconds
    relative to ``reference`` (``"origin"`` or ``"arrival:<phase>"``).
    """

    data: np.ndarray
    components: tuple[str, ...]
    stations: pd.DataFrame
    dt: float
    t0: float
    reference: str = "origin"
    event_id: str = ""
    source_xyz: tuple[float, float, float] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim != 3:
            raise ModelError(
                f"gather data must be 3-D (component, station, sample), got {data.shape}"
            )
        if data.shape[0] != len(self.components):
            raise ModelError(f"{data.shape[0]} component arrays but components={self.components}")
        if data.shape[1] != len(self.stations):
            raise ModelError(f"{data.shape[1]} stations in data but {len(self.stations)} rows")
        for col in ("code", "x", "y", "z"):
            if col not in self.stations.columns:
                raise ModelError(f"stations table is missing column {col!r}")
        if not (self.dt > 0):
            raise ModelError(f"dt must be positive, got {self.dt}")
        object.__setattr__(self, "data", data)

    @property
    def n_stations(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[2])

    @property
    def times(self) -> np.ndarray:
        return self.t0 + self.dt * np.arange(self.n_samples)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes + self.stations.memory_usage(deep=True).sum())

    def component(self, name: str) -> np.ndarray:
        try:
            return self.data[self.components.index(name)]
        except ValueError:
            raise KeyError(f"component {name!r} not in {self.components}") from None

    def take_stations(self, order: np.ndarray) -> Gather:
        order = np.asarray(order, dtype=int)
        return replace(
            self,
            data=self.data[:, order, :],
            stations=self.stations.iloc[order].reset_index(drop=True),
        )

    def with_data(self, data: np.ndarray, components: tuple[str, ...], **changes: Any) -> Gather:
        return replace(self, data=data, components=tuple(components), **changes)
