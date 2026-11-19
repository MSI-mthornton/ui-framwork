"""Fail fast, with a fix, when Qt cannot open a window.

Ported from qc-keepers ``display_check.py``. The usual failures on Linux
servers are a missing system library for the PySide6 wheel, no DISPLAY, a
stale DISPLAY in a long-lived terminal, or no X authority cookie after su/sudo.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys

QT_SYSTEM_PACKAGES = (
    "mesa-libGL libxkbcommon-x11 libxcb xcb-util-cursor xcb-util-image "
    "xcb-util-keysyms xcb-util-renderutil dbus-libs fontconfig xorg-x11-xauth"
)


class DisplayError(RuntimeError):
    pass


def _qt_import_error() -> str | None:
    try:
        import PySide6.QtWidgets  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def _run(cmd: list[str]) -> subprocess.CompletedProcess | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None


def diagnose() -> str | None:
    """A human-readable reason Qt will not show a window, or None if it should."""
    error = _qt_import_error()
    if error is not None:
        return (
            f"PySide6 could not be imported:\n    {error}\n\n"
            "The PySide6 wheel needs system libraries a minimal server lacks. On RHEL/Rocky 9:\n"
            f"    sudo dnf install -y {QT_SYSTEM_PACKAGES}"
        )
    if not sys.platform.startswith("linux"):
        return None
    platform = os.environ.get("QT_QPA_PLATFORM", "")
    if platform in {"offscreen", "minimal"}:
        return None
    display = os.environ.get("DISPLAY")
    if not (display or os.environ.get("WAYLAND_DISPLAY")):
        return "No DISPLAY or WAYLAND_DISPLAY is set. Reconnect with ssh -X (or -Y)."
    if display:
        live = _run(["xdpyinfo", "-display", display])
        if live is not None and live.returncode != 0:
            return (
                f"DISPLAY is {display}, but that X server is not reachable. Long-lived\n"
                "terminals often keep a stale display number. In a shell where X works,\n"
                "run `echo $DISPLAY` and export that value here."
            )
        auth = _run(["xauth", "-n", "list", display])
        if auth is not None and auth.returncode == 0 and not auth.stdout.strip():
            user = getpass.getuser()
            return (
                f"DISPLAY is {display}, but {user} has no X authority cookie for it (usual\n"
                "after su/sudo). In the original login shell run:\n"
                f"    xauth nlist \"$DISPLAY\" | sed -e 's/^..../ffff/' | sudo -u {user} -H xauth nmerge -"
            )
    return None


def require_display() -> None:
    reason = diagnose()
    if reason:
        raise DisplayError(reason)
