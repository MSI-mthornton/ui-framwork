"""QC tag bar: one button per outcome code, numbered shortcuts 1..9."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from seisqt.panels.style import qc_color


class QcTagBar(QWidget):
    def __init__(self, ctx, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.buttons: dict[str, QPushButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(QLabel("QC:"))
        for n, code in enumerate(ctx.qc_ctl.codes, start=1):
            label = f"{code} ({n})" if n <= 9 else code
            btn = QPushButton(label)
            btn.setCheckable(True)
            r, g, b = qc_color(code)
            btn.setStyleSheet(
                f"QPushButton:checked {{ background: rgb({r},{g},{b}); color: white; }}"
            )
            btn.clicked.connect(lambda _=False, c=code: ctx.qc_ctl.tag(c))
            layout.addWidget(btn)
            self.buttons[code] = btn
        layout.addStretch(1)
        self.pending = QLabel("")
        layout.addWidget(self.pending)
        ctx.selection.changed.connect(lambda _i: self.refresh())
        ctx.events.rowChanged.connect(lambda _i: self.refresh())
        ctx.qc_ctl.written.connect(lambda *_: self.refresh())
        ctx.qc_ctl.failed.connect(lambda *_: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        ev = self.ctx.selection.event
        current = ev.qc_code if ev is not None else None
        for code, btn in self.buttons.items():
            btn.setChecked(code == current)
            btn.setEnabled(ev is not None)
        n = sum(self.ctx.qc_ctl.pending.values())
        self.pending.setText(f"saving {n}…" if n else "")

    def commands(self):
        """Commands for the window's registry (shortcuts 1..9)."""
        from seisqt.app import Command

        return [
            Command(
                f"qc.tag.{code}",
                f"Tag {code}",
                lambda c=code: self.ctx.qc_ctl.tag(c),
                str(n),
                "&QC",
            )
            for n, code in enumerate(self.ctx.qc_ctl.codes, start=1)
            if n <= 9
        ]
