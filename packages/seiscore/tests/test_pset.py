import numpy as np
import pandas as pd
import pytest

from seiscore.model import Event
from seiscore.sources.base import GatherRequest, SourceError
from seiscore.sources.fake import FakeGatherSource, grid_array
from seiscore.sources.pset import PsetFile, PsetFileSet, PsetGatherSource, write_pset

ORIGIN = pd.Timestamp("2026-03-04T05:06:07.250", tz="UTC")


def headers_for(stations, trid, start: pd.Timestamp, dt):
    n = len(stations)
    return pd.DataFrame(
        {
            "trid": np.full(n, trid),
            "gline": stations["gline"].astype(int),
            "gpoint": stations["gpoint"].astype(int),
            "gx": stations["x"].round().astype(int),
            "gy": stations["y"].round().astype(int),
            "gelev": stations["z"].round().astype(int),
            "year": start.year,
            "day": start.day_of_year,
            "hour": start.hour,
            "minute": start.minute,
            "sec": start.second,
            "micro": start.microsecond,
        }
    )


@pytest.fixture
def pset_dir(tmp_path):
    """Two files split in time, holding the fake E/N/Z gather with Z stored as down."""
    stations = grid_array(3, 4)
    src = FakeGatherSource(stations, noise=0.0)
    ev = Event(0, "e1", ORIGIN, 120.0, -40.0, -2000.0, 3.0)
    g = src.fetch_gather(
        ev, GatherRequest("e1", pre=1.0, post=3.0, alignment="origin", rotate=False)
    )
    half = g.n_samples // 2
    paths = []
    for k, (a, b) in enumerate(((0, half), (half, g.n_samples))):
        start = ORIGIN + pd.Timedelta(seconds=g.t0 + a * g.dt)
        hdr, data = [], []
        for comp, trid, sign in (("E", 13, 1), ("N", 14, 1), ("Z", 12, -1)):
            hdr.append(headers_for(stations, trid, start, g.dt))
            data.append(sign * g.component(comp)[:, a:b])
        path = tmp_path / f"part{k}.pset"
        write_pset(path, pd.concat(hdr, ignore_index=True), np.vstack(data), dt=g.dt)
        paths.append(path)
    return paths, ev, src


def test_header_roundtrip(pset_dir):
    paths, _, _ = pset_dir
    f = PsetFile(paths[0])
    assert len(f) == 36 and f.byteorder == "<"
    assert f.start_ns[0] == (ORIGIN - pd.Timedelta(seconds=1.0)).value
    assert set(np.unique(f.headers["trid"])) == {12, 13, 14}


def test_origin_gather_matches_fake(pset_dir):
    paths, ev, fake = pset_dir
    source = PsetGatherSource(PsetFileSet(paths))
    req = GatherRequest("e1", pre=0.5, post=1.0, alignment="origin")
    got = source.fetch_gather(ev, req)
    want = fake.fetch_gather(ev, req)
    assert got.components == ("P", "SH", "SV")
    order = [list(got.stations.code).index(c) for c in want.stations.code]
    # integer coordinates in the headers cost a little accuracy in the rotation
    np.testing.assert_allclose(got.data[:, order], want.data, atol=2e-3 * np.abs(want.data).max())


def test_arrival_alignment_with_travel_times(pset_dir):
    paths, ev, fake = pset_dir

    def tt(event, stations, phase):
        rec = stations[["x", "y", "z"]].to_numpy(float)
        return np.linalg.norm(rec - event.xyz, axis=1) / fake.vp

    source = PsetGatherSource(PsetFileSet(paths), travel_times=tt)
    g = source.fetch_gather(ev, GatherRequest("e1", pre=0.1, post=0.2))
    k = np.argmax(np.abs(g.component("P")), axis=1)
    np.testing.assert_allclose(g.times[k], 0.0, atol=2 * g.dt)
    with pytest.raises(SourceError):
        PsetGatherSource(PsetFileSet(paths)).fetch_gather(ev, GatherRequest("e1"))


def test_list_file_and_missing_window(pset_dir, tmp_path):
    paths, ev, _ = pset_dir
    listing = tmp_path / "files.csv"
    listing.write_text("PATH\n" + "\n".join(p.name for p in paths) + "\n")
    files = PsetFileSet.from_list_file(listing)
    assert len(files) == 2
    late = Event(0, "late", ORIGIN + pd.Timedelta(hours=1), 0, 0, -2000, 1)
    with pytest.raises(SourceError):
        PsetGatherSource(files).fetch_gather(late, GatherRequest("late", alignment="origin"))


def test_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.pset"
    bad.write_bytes(b"x" * 5000)
    with pytest.raises(SourceError):
        PsetFile(bad)
