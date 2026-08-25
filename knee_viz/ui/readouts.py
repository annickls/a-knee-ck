"""Compact panel showing the remaining five degrees of freedom plus the gaps."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtWidgets

from knee_viz.core.state import KneeState
from knee_viz.theme import PALETTE, Palette

# (attribute path, caption, unit, decimals)
_ROWS = (
    ("dof.adduction_deg", "VAR / VAL", "°", 1),
    ("dof.rotation_deg", "INT / EXT ROT", "°", 1),
    ("dof.anterior_mm", "ANT / POST", "mm", 1),
    ("dof.medial_mm", "MED / LAT", "mm", 1),
    ("dof.proximal_mm", "PROX / DIST", "mm", 1),
    ("gaps.medial_mm", "GAP MED", "mm", 1),
    ("gaps.lateral_mm", "GAP LAT", "mm", 1),
)


def _resolve(state: KneeState, path: str) -> float:
    value = state
    for part in path.split("."):
        value = getattr(value, part)
    return float(value)


class ReadoutPanel(QtWidgets.QFrame):
    """A two-column value list, styled as a raised panel."""

    def __init__(self, palette: Palette = PALETTE, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._palette = palette

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(6)

        self._values: dict[str, QtWidgets.QLabel] = {}
        for row, (path, caption, unit, _) in enumerate(_ROWS):
            name = QtWidgets.QLabel(caption)
            name.setObjectName("SectionTitle")
            value = QtWidgets.QLabel("--")
            value.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            font = value.font()
            font.setPointSize(13)
            font.setBold(True)
            value.setFont(font)

            layout.addWidget(name, row, 0)
            layout.addWidget(value, row, 1)
            layout.addWidget(QtWidgets.QLabel(unit), row, 2)
            self._values[path] = value

    def update_state(self, state: KneeState) -> None:
        for path, _, _, decimals in _ROWS:
            value = _resolve(state, path)
            # Explicit sign and a decimal: the original used int(), so -0.7 and
            # +0.7 both displayed as "0" and a NaN raised.
            text = "--" if not np.isfinite(value) else f"{value:+.{decimals}f}"
            self._values[path].setText(text)

    def clear(self) -> None:
        for label in self._values.values():
            label.setText("--")
