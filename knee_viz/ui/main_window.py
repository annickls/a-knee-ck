"""The application window: 3D scene on the left, live plot on the right."""

from __future__ import annotations

import logging

from PyQt6 import QtCore, QtGui, QtWidgets

from knee_viz.config import AppConfig
from knee_viz.core.bones import load_knee
from knee_viz.core.modes import FORCE_BAR_CHANNELS
from knee_viz.core.pipeline import KneePipeline
from knee_viz.core.state import KneeState
from knee_viz.data import make_source
from knee_viz.data.driver import FrameDriver
from knee_viz.theme import PALETTE, Palette
from knee_viz.ui.controls import ControlBar
from knee_viz.ui.force_bar import ForceBarWidget
from knee_viz.ui.plot2d import KneePlot2D
from knee_viz.ui.readouts import ReadoutPanel
from knee_viz.ui.scene3d import Scene3D

log = logging.getLogger(__name__)

SCENE_MIN_WIDTH_PX = 460
PLOT_MIN_WIDTH_PX = 420
CONTROLS_WIDTH_PX = 210


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: AppConfig, palette: Palette = PALETTE) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle("Knee joint visualisation")
        self.resize(1500, 950)

        femur, tibia = load_knee(config.model_dir)
        self.pipeline = KneePipeline(
            femur,
            tibia,
            side=config.side,
            gap_crop_radius_mm=config.gap_crop_radius_mm,
            gap_margin_mm=config.gap_margin_mm,
        )
        self._state: KneeState | None = None
        self._plot_running = False

        self.scene = Scene3D(femur, tibia, config, palette)
        self.plot = KneePlot2D(max_points=config.plot_max_points, palette=palette)
        self.readouts = ReadoutPanel(palette)
        self.force_bar = ForceBarWidget(palette=palette)
        self.force_bar.setVisible(config.show_force_bar)
        self.controls = ControlBar(debug=config.debug)

        self._build_layout()
        self._connect()

        self.driver = FrameDriver(make_source(config), interval_ms=config.poll_interval_ms)
        self.driver.frameReady.connect(self._on_frame)
        self.driver.errorRaised.connect(self._on_error)
        self.driver.start()

        self.statusBar().showMessage(
            f"{config.source_kind}: {config.csv_path.name}   model: {config.model_dir.name}"
        )

    # ---------------------------------------------------------------- assembly

    def _build_layout(self) -> None:
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QHBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.force_bar)
        left_layout.addWidget(self.scene, 1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QHBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(self.plot, 1)

        side = QtWidgets.QWidget()
        side.setFixedWidth(CONTROLS_WIDTH_PX)
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)
        side_layout.addWidget(self.readouts)
        side_layout.addWidget(self.controls, 1)
        right_layout.addWidget(side)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        # A GLViewWidget reports a tiny size hint, so stretch factors alone let
        # the plot claim the whole window. Pin both panes explicitly.
        splitter.setChildrenCollapsible(False)
        left.setMinimumWidth(SCENE_MIN_WIDTH_PX)
        self.plot.setMinimumWidth(PLOT_MIN_WIDTH_PX)
        splitter.setSizes([760, 740])

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _connect(self) -> None:
        self.controls.modeChanged.connect(self.plot.set_mode)
        self.controls.styleChanged.connect(self.plot.set_render_style)
        self.controls.clearRequested.connect(self.plot.clear)
        self.controls.zeroRequested.connect(self._on_zero)
        self.controls.runningChanged.connect(self._on_running_changed)
        self.controls.debugToggled.connect(self.scene.set_debug)

        QtGui.QShortcut(QtGui.QKeySequence("D"), self, activated=self._toggle_debug)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self.scene.request_recentre)

    # ------------------------------------------------------------------ events

    def _on_frame(self, frame) -> None:
        state = self.pipeline.step(frame)
        self._state = state

        self.scene.update_state(state)
        self.readouts.update_state(state)
        self._update_force_bar(frame)

        if self._plot_running:
            self.plot.push(state)

    def _update_force_bar(self, frame) -> None:
        attr, index, limit = FORCE_BAR_CHANNELS[self.plot.mode]
        vector = getattr(frame, attr)
        if vector is None:
            self.force_bar.set_value(None)
            return
        self.force_bar.set_range(0.0, limit)
        self.force_bar.set_label("TORQUE" if attr == "torque_nm" else "FORCE")
        self.force_bar.set_value(abs(float(vector[index])))

    def _on_running_changed(self, running: bool) -> None:
        self._plot_running = running

    def _on_zero(self) -> None:
        if self._state is None:
            self.statusBar().showMessage("no frame yet -- nothing to zero", 4000)
            return
        self.pipeline.capture_zero(self._state)
        self.statusBar().showMessage("angles and translations zeroed", 4000)

    def _toggle_debug(self) -> None:
        self.controls.debug_button.toggle()

    def _on_error(self, message: str) -> None:
        log.warning("data source: %s", message)
        self.statusBar().showMessage(message, 8000)

    def closeEvent(self, event) -> None:
        self.driver.stop()
        super().closeEvent(event)
