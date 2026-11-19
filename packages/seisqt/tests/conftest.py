import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Keep QSettings written by window tests out of the real user profile.
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="seisqt-tests-")

import pytest

from seiscore.sources.base import GatherRequest, Sources, TimeWindow
from seiscore.sources.fake import FakeEventSource, FakeGatherSource, FakeQcSink, grid_array


@pytest.fixture
def sources():
    return Sources(
        events=FakeEventSource(50),
        gathers=FakeGatherSource(grid_array(3, 4)),
        qc=FakeQcSink(fail_codes={"bad"}),
    )


@pytest.fixture
def ctx(qtbot, sources):
    from seisqt.app import AppContext

    c = AppContext.create(
        "test-app",
        None,
        sources,
        gather_request=GatherRequest("", pre=0.1, post=0.3),
        qc_codes=("keeper", "noise", "bad"),
    )
    yield c
    c.runner.shutdown()


@pytest.fixture
def loaded(ctx, qtbot):
    ctx.events.set_table(ctx.sources.events.fetch_events(TimeWindow()))
    return ctx
