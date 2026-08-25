"""Turns one tracker frame into one :class:`KneeState`. Pure numpy, no Qt.

This is the whole measurement chain in one place, which makes it runnable
headless over a recorded CSV without constructing a single widget.
"""

from __future__ import annotations

import numpy as np

from knee_viz.core.bones import REQUIRED_LANDMARKS, BoneModel
from knee_viz.core.filters import PoseFilter
from knee_viz.core.joint_gap import JointGapCalculator
from knee_viz.core.kinematics import Dof6, GroodSuntayEngine, Side, ZeroOffsets
from knee_viz.core.state import KneeState
from knee_viz.core.transforms import pose_from_metres


class KneePipeline:
    """Owns the bone models and the measurement objects for one session."""

    def __init__(
        self,
        femur: BoneModel,
        tibia: BoneModel,
        *,
        side: Side = Side.LEFT,
        gap_crop_radius_mm: float = 80.0,
        gap_margin_mm: float = 1.0,
    ) -> None:
        self.femur = femur
        self.tibia = tibia
        self.engine = GroodSuntayEngine(side)
        self.zero = ZeroOffsets()
        self.pose_filter = PoseFilter()
        self.gap = JointGapCalculator(
            femur, crop_radius_mm=gap_crop_radius_mm, margin_mm=gap_margin_mm
        )

        # Fixed ordering lets the per-frame landmark transform be a single matmul.
        self._femur_names = REQUIRED_LANDMARKS["femur"]
        self._tibia_names = REQUIRED_LANDMARKS["tibia"]
        self._femur_ref = femur.landmark_array(self._femur_names)
        self._tibia_ref = tibia.landmark_array(self._tibia_names)

    def step(self, frame) -> KneeState:
        """Measure one frame. ``frame`` is a :class:`knee_viz.data.source.Frame`."""
        frame = self.pose_filter.apply(frame)
        femur_pose = pose_from_metres(frame.femur_quat_wxyz, frame.femur_pos_m)
        tibia_pose = pose_from_metres(frame.tibia_quat_wxyz, frame.tibia_pos_m)

        femur_world = self._femur_ref @ femur_pose[:3, :3].T + femur_pose[:3, 3]
        tibia_world = self._tibia_ref @ tibia_pose[:3, :3].T + tibia_pose[:3, 3]
        femur_landmarks = dict(zip(self._femur_names, femur_world))
        tibia_landmarks = dict(zip(self._tibia_names, tibia_world))

        frames = self.engine.build_frames(femur_landmarks, tibia_landmarks)
        raw = self.engine.solve(frames)

        gaps = self.gap.measure(
            np.stack([tibia_landmarks["tibia_medial"], tibia_landmarks["tibia_lateral"]]),
            frames.e3t,
            femur_pose,
        )

        return KneeState(
            dof=self.zero.apply(raw),
            raw_dof=raw,
            frames=frames,
            gaps=gaps,
            femur_landmarks=femur_landmarks,
            tibia_landmarks=tibia_landmarks,
            femur_pose=femur_pose,
            tibia_pose=tibia_pose,
        )

    def capture_zero(self, state: KneeState) -> None:
        """Make the supplied pose read as zero on every subsequent frame."""
        self.zero.capture(state.raw_dof)

    def clear_zero(self) -> None:
        self.zero.clear()
