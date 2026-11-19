"""Build a full window offscreen, load data, draw a gather, tag an event."""

from seisqt.app import BaseMainWindow
from seisqt.panels import EventMapPanel, EventTimePanel, GatherPanel, LogPanel, QcTagBar


def test_window_end_to_end(ctx, qtbot):
    win = BaseMainWindow(ctx)
    gather = GatherPanel(ctx)
    tags = QcTagBar(ctx)
    win.set_central(gather)
    win.add_panel(EventMapPanel(ctx), "Map")
    win.add_panel(EventTimePanel(ctx), "Timeline")
    win.add_panel(LogPanel(ctx), "Log")
    win.add_panel(tags, "QC")
    for cmd in tags.commands():
        win.add_command(cmd)
    qtbot.addWidget(win)
    win.show()
    with qtbot.waitSignal(ctx.events.changed, timeout=3000):
        ctx.events_ctl.load()
    qtbot.waitUntil(lambda: gather.display is not None, timeout=5000)
    assert gather.display.component == "P"
    ctx.prefs.update(component="SH", sort="grid")
    qtbot.waitUntil(lambda: gather.display.component == "SH", timeout=5000)
    win.commands["qc.tag.keeper"].trigger()
    qtbot.waitUntil(lambda: bool(ctx.sources.qc.writes), timeout=3000)
    win.commands["event.last"].trigger()
    assert ctx.selection.index == len(ctx.events) - 1
    win.close()
