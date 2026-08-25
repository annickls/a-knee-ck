"""Rounded translucent readouts drawn over the 3D view, as in the reference."""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from knee_viz.theme import PALETTE, Palette


class PillLabel(QtWidgets.QWidget):
    """A rounded pill showing a large value over a smaller caption.

    Painted rather than styled so the corner radius follows the widget height
    and the background can be translucent over the GL surface.
    """

    def __init__(
        self,
        caption: str = "",
        *,
        value_pt: int = 26,
        caption_pt: int = 11,
        palette: Palette = PALETTE,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette
        self._value = "--"
        self._caption = caption
        self._value_pt = value_pt
        self._caption_pt = caption_pt
        # Must not intercept mouse events or it would swallow camera drags.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_value(self, value: str, caption: str | None = None) -> None:
        self._value = value
        if caption is not None:
            self._caption = caption
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(150, 78 if self._caption else 54)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2.2
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self._palette.qcolor("pill_bg", 0.82))
        painter.drawRoundedRect(rect, radius, radius)

        painter.setPen(self._palette.qcolor("text_on_pill"))
        if self._caption:
            value_rect = QtCore.QRectF(rect.x(), rect.y() + rect.height() * 0.08,
                                       rect.width(), rect.height() * 0.55)
            caption_rect = QtCore.QRectF(rect.x(), rect.y() + rect.height() * 0.60,
                                         rect.width(), rect.height() * 0.34)
        else:
            value_rect = rect
            caption_rect = None

        font = painter.font()
        font.setPointSize(self._value_pt)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(value_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self._value)

        if caption_rect is not None:
            font.setPointSize(self._caption_pt)
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(caption_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self._caption)
        painter.end()
