import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from demo_viewer.app import DemoViewer


def test_config_layers(monkeypatch):
    app = DemoViewer()
    monkeypatch.setenv("DEMO_FAKE__N_EVENTS", "120")
    cfg = app.load_config(app.parse_args(["--set", "gather.phase=S"]))
    assert cfg.fake.n_events == 120
    assert cfg.gather.phase == "S"
    assert cfg.qc.codes == ("keeper", "noise", "review")


def test_demo_window_builds_and_runs(qtbot):
    app = DemoViewer()
    cfg = app.load_config(
        app.parse_args(["--set", "fake.latency_s=0", "--set", "fake.n_events=40"])
    )
    ctx = app.make_context(cfg)
    win = app.build(ctx)
    qtbot.addWidget(win)
    win.show()
    with qtbot.waitSignal(ctx.gathers_ctl.ready, timeout=5000):
        ctx.events_ctl.load()
    assert {"qc.tag.keeper", "view.SH", "event.next"} <= set(win.commands)
    win.commands["view.SH"].trigger()
    assert ctx.prefs.prefs.component == "SH"
    win.close()
