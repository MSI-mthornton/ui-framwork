import pandas as pd
import pytest

from seiscore.model import EventTable, ModelError


def frame(**extra):
    base = {
        "id": ["b", "a"],
        "origin_time": ["2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"],
        "x": [1, 2],
        "y": [3, 4],
        "z": [-5, -6],
        "snr": [1.5, 2.5],
    }
    base.update(extra)
    return pd.DataFrame(base)


def test_sorted_by_time_and_qc_normalised():
    t = EventTable(frame(qc_code=[" Keeper", None]))
    assert list(t.ids) == ["a", "b"]
    assert list(t.qc_code) == ["", "keeper"]
    assert t.index_of("b") == 1


def test_with_qc_returns_new_table():
    t = EventTable(frame())
    t2 = t.with_qc(0, "NOISE")
    assert t.event(0).qc_code == ""
    assert t2.event(0).qc_code == "noise"
    assert t2.qc_counts() == {"noise": 1, "": 1}


def test_validation():
    with pytest.raises(ModelError):
        EventTable(frame().drop(columns="snr"))
    with pytest.raises(ModelError):
        EventTable(frame(id=["a", "a"]))
    assert len(EventTable.empty()) == 0
