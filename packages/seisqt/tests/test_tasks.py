import threading
import time

from seisqt.tasks import TaskRunner


def wait_for(qtbot, results, n=1, timeout=3000):
    qtbot.waitUntil(lambda: len(results) >= n, timeout=timeout)


def test_success_and_failure_are_results(qtbot):
    runner = TaskRunner()
    out = []
    runner.submit(lambda: 42, out.append)
    runner.submit(lambda: 1 / 0, out.append)
    wait_for(qtbot, out, 2)
    by_ok = {r.ok: r for r in out}
    assert by_ok[True].value == 42
    assert isinstance(by_ok[False].error, ZeroDivisionError)
    assert threading.current_thread() is threading.main_thread()
    runner.shutdown()


def test_newer_task_with_same_key_supersedes(qtbot):
    runner = TaskRunner()
    out = []
    gate = threading.Event()
    runner.submit(lambda: (gate.wait(2), "old")[1], out.append, key="gather")
    runner.submit(lambda: "new", out.append, key="gather")
    wait_for(qtbot, out)
    gate.set()
    runner.wait()
    qtbot.wait(50)
    assert [r.value for r in out] == ["new"]
    runner.shutdown()


def test_timeout_reports_and_drops_late_result(qtbot):
    runner = TaskRunner()
    out = []
    runner.submit(lambda: (time.sleep(0.5), "late")[1], out.append, timeout=0.05)
    wait_for(qtbot, out)
    assert out[0].timed_out and not out[0].ok
    runner.wait()
    qtbot.wait(50)
    assert len(out) == 1
    assert runner.in_flight == 0
    runner.shutdown()


def test_callback_exception_does_not_escape(qtbot):
    runner = TaskRunner()
    seen = []

    def bad(res):
        seen.append(res)
        raise RuntimeError("boom")

    runner.submit(lambda: 1, bad)
    wait_for(qtbot, seen)
    runner.shutdown()
