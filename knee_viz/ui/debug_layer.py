"""Every optional diagnostic visual, behind a single flag.

These are the small elements the original scattered through the GUI as
commented-out blocks and dead helpers: landmark spheres, the femoral and tibial
anatomical axes, the floating axis, and the world triad. The joint-gap rays are
new -- they make the gap measurement inspectable rather than a bare number.

``cfg.debug`` is read here and nowhere else in the codebase; callers invoke
:meth:`update` unconditionally.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl

from knee_viz.core.state import KneeState
from knee_viz.theme import PALETTE, Palette
from knee_viz.ui.actors import LineBundle, PointCloud

AXIS_LENGTH_MM = 60.0
WORLD_AXIS_LENGTH_MM = 100.0

# One entry per anatomical axis: (attribute on KneeFrames, origin, palette colour).
_AXES = (
    ("e1f", "femur_origin", "debug_axis_x"),
    ("e3f", "femur_origin", "debug_axis_z"),
    ("e1t", "tibia_origin", "debug_axis_x"),
    ("e2t", "tibia_origin", "debug_axis_y"),
    ("e3t", "tibia_origin", "debug_axis_z"),
    ("floating", "tibia_origin", "accent"),
)


class DebugLayer:
    """Owns the diagnostic overlay for one 3D view."""

    def __init__(
        self,
        view: gl.GLViewWidget,
        enabled: bool,
        palette: Palette = PALETTE,
    ) -> None:
        self._view = view
        self._palette = palette
        self._built = False
        self.enabled = enabled
        if enabled:
            self._build()

    def _build(self) -> None:
        if self._built:
            return
        palette = self._palette

        self._landmarks = PointCloud(self._view, palette.gl("accent"), size=13.0, on_top=True)
        self._gap_hits = PointCloud(self._view, palette.gl("debug_hit"), size=11.0, on_top=True)

        axis_colours = np.array([palette.gl(name) for _, _, name in _AXES])
        self._axes = LineBundle(self._view, axis_colours, width=2.0, on_top=True)

        world_colours = np.array(
            [palette.gl("debug_axis_x"), palette.gl("debug_axis_y"), palette.gl("debug_axis_z")]
        )
        self._world = LineBundle(self._view, world_colours, width=1.0)
        self._world.set_segments(
            np.zeros((3, 3)), np.eye(3) * WORLD_AXIS_LENGTH_MM
        )

        self._rays = LineBundle(
            self._view, np.array([palette.gl("debug_ray")] * 2), width=2.0, on_top=True
        )
        self._built = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if enabled:
            self._build()
        for actor in self._actors():
            actor.set_visible(enabled)

    def _actors(self):
        if not self._built:
            return ()
        return (self._landmarks, self._gap_hits, self._axes, self._world, self._rays)

    def update(self, state: KneeState) -> None:
        if not self.enabled:
            return

        landmarks = state.landmarks
        self._landmarks.set_points(np.stack(list(landmarks.values())))

        frames = state.frames
        origins = np.stack([getattr(frames, origin) for _, origin, _ in _AXES])
        directions = np.stack([getattr(frames, axis) for axis, _, _ in _AXES])
        self._axes.set_segments(origins, origins + directions * AXIS_LENGTH_MM)

        # Joint-gap rays: from each plateau landmark to its surface hit, or a
        # stub along the axis when the ray misses.
        starts = np.stack([landmarks["tibia_medial"], landmarks["tibia_lateral"]])
        gaps = state.gaps
        ends = np.stack(
            [
                gaps.medial_hit if gaps.medial_hit is not None else starts[0] + frames.e3t * AXIS_LENGTH_MM,
                gaps.lateral_hit if gaps.lateral_hit is not None else starts[1] + frames.e3t * AXIS_LENGTH_MM,
            ]
        )
        self._rays.set_segments(starts, ends)

        hits = [h for h in (gaps.medial_hit, gaps.lateral_hit) if h is not None]
        self._gap_hits.set_points(np.stack(hits) if hits else None)
