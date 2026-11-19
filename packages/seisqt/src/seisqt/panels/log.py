"""Log panel: shows records from the Qt log handler."""

from __future__ import annotations

import logging

from PySide6.QtGui import QColor, QTextCharFormat
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

_COLORS = {logging.WARNING: QColor(170, 110, 0), logging.ERROR: QColor(200, 0, 0)}


class LogPanel(QWidget):
    def __init__(self, ctx, *, max_lines: int = 5000, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(max_lines)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text)
        if ctx.log_handler is not None:
            ctx.log_handler.bridge.record.connect(self.append)

    def append(self, level: int, message: str) -> None:
        fmt = QTextCharFormat()
        color = _COLORS.get(logging.ERROR if level >= logging.ERROR else level)
        if color is not None:
            fmt.setForeground(color)
        cursor = self.text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        if not self.text.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(message, fmt)
        self.text.ensureCursorVisible()
