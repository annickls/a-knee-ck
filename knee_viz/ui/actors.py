"""Thin wrappers over pyqtgraph.opengl items.

Every actor is created once and updated with ``setData``/``setTransform``
thereafter. The original removed and re-added line items on every frame, which
churned GL buffers 50 times a second.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from OpenGL.GL import GL_BLEND, GL_DEPTH_TEST, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA
from PyQt6 import QtGui

from knee_viz.core.bones import BoneModel
from knee_viz.ui.shading import register as register_bone_shader

# Diagnostic geometry is drawn with depth testing off so landmarks and axes
# stay visible where they sit inside the bone -- which is most of the time.
OVERLAY_GL_OPTIONS = {
    GL_DEPTH_TEST: False,
    GL_BLEND: True,
    "glBlendFunc": (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
}


class BoneActor:
    """A registered bone mesh, posed each frame by its tracker transform."""

    def __init__(self, view: gl.GLViewWidget, model: BoneModel, colour: QtGui.QColor) -> None:
        # STL vertices are unshared, so normals are per-face however they are
        # computed; flat shading is both cheaper and what the reference shows.
        self.item = gl.GLMeshItem(
            vertexes=model.vertices,
            faces=model.faces,
            smooth=False,
            drawEdges=False,
            computeNormals=True,
            color=colour,
            shader=register_bone_shader(),
            glOptions="opaque",
        )
        view.addItem(self.item)

    def set_pose(self, world_from_ref: np.ndarray) -> None:
        self.item.setTransform(QtGui.QMatrix4x4(*world_from_ref.flatten()))

    def set_visible(self, visible: bool) -> None:
        self.item.setVisible(visible)


class PointCloud:
    """A single scatter item standing in for many small spheres."""

    def __init__(
        self, view: gl.GLViewWidget, colour: tuple, size: float = 12.0, on_top: bool = False
    ) -> None:
        self.item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), color=colour, size=size, pxMode=True
        )
        self.item.setGLOptions(OVERLAY_GL_OPTIONS if on_top else "translucent")
        view.addItem(self.item)

    def set_points(self, points: np.ndarray) -> None:
        if points is None or len(points) == 0:
            self.item.setVisible(False)
            return
        self.item.setVisible(True)
        self.item.setData(pos=np.asarray(points, dtype=float))

    def set_visible(self, visible: bool) -> None:
        self.item.setVisible(visible)


class LineBundle:
    """Many independent line segments in one item, each with its own colour."""

    def __init__(
        self,
        view: gl.GLViewWidget,
        colours: np.ndarray,
        width: float = 2.0,
        on_top: bool = False,
    ) -> None:
        # 'lines' mode consumes the vertex array as disjoint pairs, so one item
        # can carry every axis and ray in the scene.
        self._colours = np.repeat(np.asarray(colours, dtype=float), 2, axis=0)
        self.item = gl.GLLinePlotItem(
            pos=np.zeros((len(self._colours), 3)),
            color=self._colours,
            width=width,
            mode="lines",
            antialias=True,
        )
        if on_top:
            self.item.setGLOptions(OVERLAY_GL_OPTIONS)
        view.addItem(self.item)

    def set_segments(self, starts: np.ndarray, ends: np.ndarray) -> None:
        """Set every segment at once; ``starts`` and ``ends`` are ``(n, 3)``."""
        starts = np.asarray(starts, dtype=float)
        ends = np.asarray(ends, dtype=float)
        points = np.empty((2 * len(starts), 3))
        points[0::2] = starts
        points[1::2] = ends
        self.item.setData(pos=points, color=self._colours[: len(points)])

    def set_visible(self, visible: bool) -> None:
        self.item.setVisible(visible)
