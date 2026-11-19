import numpy as np
import pandas as pd

from seiscore import dsp
from seiscore.model import Event
from seiscore.sources.base import GatherRequest
from seiscore.sources.fake import FakeGatherSource, grid_array


def event(x=0.0, y=0.0, z=-2500.0):
    return Event(0, "evt-1", pd.Timestamp("2026-01-01", tz="UTC"), x, y, z, 5.0)


def test_fake_gather_rotates_p_onto_p_and_s_onto_sh():
    src = FakeGatherSource(grid_array(4, 6), noise=0.0)
    ev = event(130.0, -70.0)
    g = src.fetch_gather(ev, GatherRequest(ev.id, pre=0.2, post=1.2, alignment="origin"))
    assert g.components == ("P", "SH", "SV")
    p, sh, sv = (np.abs(g.component(c)).max() for c in g.components)
    assert p > 0 and sh > 0
    assert sv < 1e-4 * max(p, sh)
    tp = g.stations["travel_time"].to_numpy()
    k = np.argmax(np.abs(g.component("P")), axis=1)
    np.testing.assert_allclose(g.times[k], tp, atol=2 * g.dt)


def test_arrival_alignment_puts_p_at_zero():
    src = FakeGatherSource(grid_array(3, 3), noise=0.0)
    g = src.fetch_gather(event(), GatherRequest("evt-1", pre=0.1, post=0.3))
    k = np.argmax(np.abs(g.component("P")), axis=1)
    np.testing.assert_allclose(g.times[k], 0.0, atol=g.dt)


def test_station_order_modes():
    st = grid_array(2, 3)
    src = np.array([150.0, 0.0, -1000.0])
    off = dsp.station_order(st, "offset", source_xyz=src)
    d = np.hypot(st.x - src[0], st.y - src[1]).to_numpy()
    assert np.all(np.diff(d[off]) >= 0)
    grid = dsp.station_order(st, "grid", source_xyz=src)
    assert list(grid) == list(range(6))
    assert dsp.station_order(st.drop(columns=["gline", "gpoint"]), "grid", source_xyz=src) is None


def test_decimate_minmax_keeps_peaks():
    x = np.zeros((2, 1000))
    x[0, 503] = -7.0
    x[1, 10] = 3.0
    out = dsp.decimate_minmax(x, 100)
    assert out.shape == (2, 100)
    assert out[0].min() == -7.0 and out[1].max() == 3.0
    shape = dsp.decimate_for_display(np.ones((5000, 4000)), 500, 300).shape
    assert shape[0] <= 500 and shape[1] <= 300 and shape[1] > 250


def test_clip_level_and_stack():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((50, 2000))
    level = dsp.clip_level(data, percentile=99)
    assert 2.0 < level < 3.5
    traces = np.vstack([np.sin(np.linspace(0, 6, 100)) * a for a in (1, 100)])
    s = dsp.stack(traces)
    np.testing.assert_allclose(s, traces[0] / np.abs(traces[0]).max(), atol=1e-12)


def test_time_window():
    src = FakeGatherSource(grid_array(1, 2))
    g = src.fetch_gather(event(), GatherRequest("evt-1", pre=0.1, post=0.1))
    w = dsp.time_window(g, -0.05, 0.05)
    assert abs(g.times[w.start] + 0.05) < g.dt and abs(g.times[w.stop - 1] - 0.05) < g.dt
