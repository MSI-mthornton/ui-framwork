"""Pulse sources against a stand-in client (no pulse_sdk needed)."""

from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from seiscore.model import Event
from seiscore.sources.base import GatherRequest, SourceError, TimeWindow
from seiscore.sources.fake import FakeGatherSource, grid_array
from seiscore.sources.pulse import (
    PulseConnection,
    PulseEventSource,
    PulseGatherSource,
    PulseSettings,
    drop_superseded,
)


class Table:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search_select(self, *, fields, filters, page_size, page_number, sort=None):
        self.calls.append(filters)
        rows = self.rows
        for f in filters:
            if f["op"] == "in":
                rows = [r for r in rows if str(r[f["field"]]) in set(map(str, f["value"]))]
        start = (page_number - 1) * page_size
        chunk = rows[start : start + page_size]
        return NS(data=chunk, has_next_page=start + page_size < len(rows), total=len(rows))


def event_rows():
    return [
        {
            "id": "e1",
            "origin_time": "2026-01-01T00:00:01Z",
            "x": 1,
            "y": 2,
            "z": -3,
            "snr": 4,
            "pset_run_id": "r-old",
            "latest_qc_outcome_code": "keeper",
        },
        {
            "id": "e2",
            "origin_time": "2026-01-01T00:00:02Z",
            "x": 1,
            "y": 2,
            "z": -3,
            "snr": 4,
            "pset_run_id": "r-new",
            "latest_qc_outcome_code": "Noise",
        },
        {
            "id": "e3",
            "origin_time": "2026-01-01T00:00:03Z",
            "x": 1,
            "y": 2,
            "z": -3,
            "snr": 4,
            "pset_run_id": None,
            "latest_qc_outcome_code": "keeper",
        },
    ]


def test_events_drop_superseded_runs_and_page():
    client = NS(
        event=Table(event_rows()),
        pset_run=Table(
            [{"id": "r-old", "merge_seismic_id": "m"}, {"id": "r-new", "merge_seismic_id": "m"}]
        ),
        merge_seismic=Table([{"id": "m", "active_run_id": "r-new"}]),
    )
    conn = PulseConnection(PulseSettings(page_size=2), factory=lambda: client)
    table = PulseEventSource(conn).fetch_events(TimeWindow())
    assert list(table.ids) == ["e2", "e3"]
    assert list(table.qc_code) == ["noise", "keeper"]
    assert len(client.event.calls) == 2  # two pages


def test_drop_superseded_without_active_ids_keeps_all():
    df = pd.DataFrame({"pset_run_id": ["a", None]})
    assert len(drop_superseded(df, set())) == 2


def fake_pulse_traces(ev, request):
    """Manifest + samples shaped like Pulse, from the fake gather source."""
    st = grid_array(2, 3)
    g = FakeGatherSource(st, noise=0.0).fetch_gather(
        ev,
        GatherRequest(
            ev.id, pre=request["pre"], post=request["post"], alignment="origin", rotate=False
        ),
    )
    channels, samples = [], []
    for s, row in st.iterrows():
        for c, name in enumerate(("east", "north", "vertical")):
            channels.append(
                NS(
                    component=name,
                    geophone_location_id=None,
                    gline=row.gline,
                    gpoint=row.gpoint,
                    station_code=row.code,
                    x=row.x,
                    y=row.y,
                    z=row.z,
                    trace_index=len(samples),
                    travel_time=None,
                )
            )
            samples.append(g.data[c, s])
    return g, channels, np.asarray(samples)


def test_gather_from_manifest_rotates_like_fake():
    ev = Event(0, "e1", pd.Timestamp("2026-01-01", tz="UTC"), 50.0, 20.0, -2000.0, 1.0)
    g, channels, samples = fake_pulse_traces(ev, {"pre": 0.2, "post": 1.0})
    client = NS(
        event=NS(
            get_traces=lambda event_id, **kw: NS(
                dequantized_samples=lambda: samples, header={"sample_interval": g.dt}
            ),
            get_trace_manifest=lambda event_id, **kw: NS(trace_channels=channels),
        )
    )
    conn = PulseConnection(PulseSettings(), factory=lambda: client)
    req = GatherRequest("e1", pre=0.2, post=1.0, alignment="origin")
    got = PulseGatherSource(conn).fetch_gather(ev, req)
    want = FakeGatherSource(grid_array(2, 3), noise=0.0).fetch_gather(ev, req)
    assert got.components == ("P", "SH", "SV")
    assert got.dt == want.dt
    np.testing.assert_allclose(got.data, want.data, atol=1e-5)


def test_arrival_without_travel_time_is_a_source_error():
    conn = PulseConnection(PulseSettings(), factory=lambda: NS())
    ev = Event(0, "e1", pd.Timestamp("2026-01-01", tz="UTC"), 0, 0, -1, 1)
    with pytest.raises(SourceError):
        PulseGatherSource(conn).fetch_gather(ev, GatherRequest("e1"))
