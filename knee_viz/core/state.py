"""The per-frame result object. This is the only thing that crosses into the UI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from knee_viz.core.joint_gap import GapResult
from knee_viz.core.kinematics import Dof6, KneeFrames


@dataclass(frozen=True, slots=True)
class KneeState:
    """Everything the UI needs for one frame, already reduced to numbers."""

    dof: Dof6  # zeroed, for display and plotting
    raw_dof: Dof6  # un-zeroed, for capturing a new baseline
    frames: KneeFrames
    gaps: GapResult
    femur_landmarks: dict[str, np.ndarray]
    tibia_landmarks: dict[str, np.ndarray]
    femur_pose: np.ndarray  # (4, 4) world from femur reference frame, mm
    tibia_pose: np.ndarray  # (4, 4) world from tibia reference frame, mm

    @property
    def landmarks(self) -> dict[str, np.ndarray]:
        return {**self.femur_landmarks, **self.tibia_landmarks}
