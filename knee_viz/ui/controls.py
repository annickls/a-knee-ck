"""Button strip: plot mode selection and the session actions."""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from knee_viz.core.modes import MODES, PlotMode
from knee_viz.ui.plot2d import RenderStyle


class ControlBar(QtWidgets.QWidget):
    """Emits typed signals; owns no application state beyond its own toggles."""

    modeChanged = QtCore.pyqtSignal(PlotMode)
    styleChanged = QtCore.pyqtSignal(RenderStyle)
    runningChanged = QtCore.pyqtSignal(bool)
    clearRequested = QtCore.pyqtSignal()
    zeroRequested = QtCore.pyqtSignal()
    debugToggled = QtCore.pyqtSignal(bool)

    def __init__(self, debug: bool = False, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        mode_label = QtWidgets.QLabel("PLOT")
        mode_label.setObjectName("SectionTitle")
        layout.addWidget(mode_label)

        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for mode in PlotMode:
            button = QtWidgets.QPushButton(MODES[mode].label)
            button.setCheckable(True)
            button.setChecked(mode is PlotMode.JOINT_GAPS)
            button.clicked.connect(lambda _, m=mode: self.modeChanged.emit(m))
            self._mode_group.addButton(button)
            layout.addWidget(button)

        layout.addSpacing(10)
        action_label = QtWidgets.QLabel("SESSION")
        action_label.setObjectName("SectionTitle")
        layout.addWidget(action_label)

        self.run_button = QtWidgets.QPushButton("start plot")
        self.run_button.setCheckable(True)
        self.run_button.toggled.connect(self._on_run_toggled)
        layout.addWidget(self.run_button)

        self.clear_button = QtWidgets.QPushButton("clear plot")
        self.clear_button.clicked.connect(self.clearRequested)
        layout.addWidget(self.clear_button)

        self.zero_button = QtWidgets.QPushButton("zero angles && translations")
        self.zero_button.clicked.connect(self.zeroRequested)
        layout.addWidget(self.zero_button)

        self.bars_button = QtWidgets.QPushButton("show bars")
        self.bars_button.setCheckable(True)
        self.bars_button.toggled.connect(
            lambda on: self.styleChanged.emit(RenderStyle.BARS if on else RenderStyle.POINTS)
        )
        layout.addWidget(self.bars_button)

        self.debug_button = QtWidgets.QPushButton("debug overlay")
        self.debug_button.setCheckable(True)
        self.debug_button.setChecked(debug)
        self.debug_button.toggled.connect(self.debugToggled)
        layout.addWidget(self.debug_button)

        layout.addStretch(1)

    def _on_run_toggled(self, running: bool) -> None:
        self.run_button.setText("stop plot" if running else "start plot")
        self.runningChanged.emit(running)
