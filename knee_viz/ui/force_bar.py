"""Vertical force bar, sited left of the 3D view.

Placeholder: the widget, its layout slot and its call site in the main window
all exist and run, but nothing drives it yet. :class:`Frame` already carries
``force_n`` (parsed from CSV columns 1-3), so switching it on is a one-line
change in ``MainWindow._on_frame`` plus ``AppConfig.show_force_bar = True``.
"""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from knee_viz.theme import PALETTE, Palette

BAR_WIDTH_PX = 46
TICK_COUNT = 5


class ForceBarWidget(QtWidgets.QWidget):
    """A track that fills with applied force, green at low load through to red."""

    def __init__(
        self,
        f_min: float = 0.0,
        f_max: float = 120.0,
        palette: Palette = PALETTE,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette
        self._min = f_min
        self._max = f_max
        self._value: float | None = None
        self._label = "FORCE"
        self.setFixedWidth(BAR_WIDTH_PX)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding
        )

    def set_range(self, f_min: float, f_max: float) -> None:
        self._min, self._max = f_min, f_max
        self.update()

    def set_value(self, newtons: float | None) -> None:
        """Set the fill. ``None`` leaves the bar empty -- the placeholder state."""
        self._value = newtons if newtons is not None and np.isfinite(newtons) else None
        self.update()

    def set_label(self, text: str) -> None:
        self._label = text
        self.update()

    def _fill_colour(self, fraction: float) -> QtGui.QColor:
        low = self._palette.qcolor("force_low")
        high = self._palette.qcolor("force_high")
        return QtGui.QColor.fromRgbF(
            low.redF() + (high.redF() - low.redF()) * fraction,
            low.greenF() + (high.greenF() - low.greenF()) * fraction,
            low.blueF() + (high.blueF() - low.blueF()) * fraction,
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        margin = 8.0
        caption_height = 22.0
        track = QtCore.QRectF(
            margin, margin, self.width() - 2 * margin,
            self.height() - 2 * margin - caption_height,
        )
        radius = track.width() / 2.0

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self._palette.qcolor("force_track"))
        painter.drawRoundedRect(track, radius, radius)

        if self._value is not None and self._max > self._min:
            fraction = float(np.clip((self._value - self._min) / (self._max - self._min), 0.0, 1.0))
            fill_height = track.height() * fraction
            fill = QtCore.QRectF(
                track.x(), track.bottom() - fill_height, track.width(), fill_height
            )
            painter.setBrush(self._fill_colour(fraction))
            painter.drawRoundedRect(fill, radius, radius)

        painter.setPen(self._palette.qcolor("plot_grid"))
        for index in range(TICK_COUNT):
            y = track.bottom() - track.height() * index / (TICK_COUNT - 1)
            painter.drawLine(QtCore.QPointF(track.right() - 4, y), QtCore.QPointF(track.right(), y))

        painter.setPen(self._palette.qcolor("text_muted"))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRectF(0, self.height() - caption_height, self.width(), caption_height),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            self._label,
        )
        painter.end()
