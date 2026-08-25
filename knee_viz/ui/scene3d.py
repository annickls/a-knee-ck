"""The live 3D bone view."""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector
from PyQt6 import QtWidgets

from knee_viz.config import AppConfig
from knee_viz.core.bones import BoneModel
from knee_viz.core.state import KneeState
from knee_viz.theme import PALETTE, Palette
from knee_viz.ui.actors import BoneActor
from knee_viz.ui.debug_layer import DebugLayer
from knee_viz.ui.overlays import PillLabel

CAMERA_DISTANCE_MM = 260.0
CAMERA_ELEVATION_DEG = 6.0
CAMERA_AZIMUTH_DEG = 90.0
PILL_MARGIN_PX = 16
MIN_VIEW_WIDTH_PX = 420
MIN_VIEW_HEIGHT_PX = 480


class Scene3D(QtWidgets.QWidget):
    """A GL view of the two bones plus the flexion pill and the debug overlay."""

    def __init__(
        self,
        femur: BoneModel,
        tibia: BoneModel,
        config: AppConfig,
        palette: Palette = PALETTE,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(palette.qcolor("scene_bg"))
        # GLViewWidget's own size hint is tiny, which lets neighbouring widgets
        # squeeze it to nothing inside a splitter.
        self.view.setMinimumSize(MIN_VIEW_WIDTH_PX, MIN_VIEW_HEIGHT_PX)
        self.view.setCameraPosition(
            distance=CAMERA_DISTANCE_MM,
            elevation=CAMERA_ELEVATION_DEG,
            azimuth=CAMERA_AZIMUTH_DEG,
        )
        # Lighting lives entirely in knee_viz.ui.shading -- GLViewWidget's
        # lightPosition/ambient/diffuse options are inert in pyqtgraph.
        bone_colour = palette.qcolor("bone")
        self.femur_actor = BoneActor(self.view, femur, bone_colour)
        self.tibia_actor = BoneActor(self.view, tibia, bone_colour)
        self.debug = DebugLayer(self.view, config.debug, palette)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self.flexion_pill = PillLabel("FLEX", palette=palette, parent=self.view)
        self.flexion_pill.resize(self.flexion_pill.sizeHint())
        self.flexion_pill.move(PILL_MARGIN_PX, PILL_MARGIN_PX)
        self.flexion_pill.show()

        self._recentre_pending = True

    def update_state(self, state: KneeState) -> None:
        self.femur_actor.set_pose(state.femur_pose)
        self.tibia_actor.set_pose(state.tibia_pose)
        self.debug.update(state)

        flexion = state.dof.flexion_deg
        self.flexion_pill.set_value("--" if not np.isfinite(flexion) else f"{flexion:+.0f}°")

        if self._recentre_pending:
            self.centre_on(state)
            self._recentre_pending = False

    def centre_on(self, state: KneeState) -> None:
        """Point the camera at the joint line."""
        centre = (state.frames.femur_origin + state.frames.tibia_origin) / 2.0
        self.view.setCameraPosition(pos=Vector(*centre))

    def request_recentre(self) -> None:
        self._recentre_pending = True

    def set_debug(self, enabled: bool) -> None:
        self.debug.set_enabled(enabled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.flexion_pill.move(PILL_MARGIN_PX, PILL_MARGIN_PX)

    def grab_png(self, path) -> None:
        self.grab().save(str(path))
