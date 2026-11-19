from seiscore.sources.base import TimeWindow


def test_events_load_selects_first(ctx, qtbot):
    with qtbot.waitSignal(ctx.events.changed, timeout=3000):
        ctx.events_ctl.load(TimeWindow())
    assert len(ctx.events) == 50
    assert ctx.selection.index == 0


def test_gather_follows_selection_and_uses_cache(loaded, qtbot):
    ctx = loaded
    with qtbot.waitSignal(ctx.gathers_ctl.ready, timeout=3000) as sig:
        ctx.selection.set(3)
    assert sig.args[0].event_id == ctx.events.table.event(3).id
    ctx.runner.wait()
    qtbot.wait(50)
    calls = ctx.sources.gathers.calls
    with qtbot.waitSignal(ctx.gathers_ctl.ready, timeout=1000):
        ctx.selection.set(4)  # prefetched
    ctx.runner.wait()
    qtbot.wait(50)
    with qtbot.waitSignal(ctx.gathers_ctl.ready, timeout=1000):
        ctx.selection.set(3)  # cached
    assert ctx.sources.gathers.calls <= calls + 2


def test_qc_tag_is_optimistic_and_advances(loaded, qtbot):
    ctx = loaded
    ctx.selection.set(0)
    eid = ctx.events.table.event(0).id
    with qtbot.waitSignal(ctx.qc_ctl.written, timeout=3000):
        ctx.qc_ctl.tag("keeper")
    assert ctx.events.table.event(0).qc_code == "keeper"
    assert ctx.selection.index == 1
    assert (eid, "keeper") in ctx.sources.qc.writes


def test_qc_failure_rolls_back(loaded, qtbot):
    ctx = loaded
    ctx.selection.set(2)
    before = ctx.events.table.event(2).qc_code
    with qtbot.waitSignal(ctx.qc_ctl.failed, timeout=3000):
        ctx.qc_ctl.tag("bad")
    assert ctx.events.table.event(2).qc_code == before
    assert not ctx.qc_ctl.pending
