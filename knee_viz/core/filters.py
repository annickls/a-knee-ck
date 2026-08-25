"""PT1 (first-order) low-pass smoothing for jittery tracked pose data."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from knee_viz.data.source import Frame

# Time constant, in seconds. Larger = smoother but more visible lag.
# Kept deliberately small -- this only needs to take the edge off marker
# jitter, not damp real motion.
POSE_FILTER_TAU_S = 0.08


class _Pt1Vector:
    """First-order low-pass on a fixed-size vector, driven by true elapsed time."""

    def __init__(self, tau_s: float) -> None:
        self._tau = tau_s
        self._state: np.ndarray | None = None
        self._t_prev: float | None = None

    def apply(self, value: np.ndarray, t: float) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if self._state is None:
            self._state = value.copy()
            self._t_prev = t
            return self._state
        dt = max(t - self._t_prev, 0.0)
        self._t_prev = t
        alpha = dt / (self._tau + dt)
        self._state = self._state + alpha * (value - self._state)
        return self._state


class _Pt1Quaternion(_Pt1Vector):
    """Same, but keeps consecutive quaternions on one hemisphere before
    blending and renormalises after -- a plain component-wise low-pass on a
    quaternion that flips sign between frames would blend through the wrong
    rotation."""

    def apply(self, value: np.ndarray, t: float) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if self._state is not None and np.dot(self._state, value) < 0.0:
            value = -value
        blended = super().apply(value, t)
        self._state = blended / np.linalg.norm(blended)
        return self._state


class PoseFilter:
    """Smooths a Frame's tracked position/orientation. Force and torque pass
    through untouched -- the load cell isn't affected by marker jitter."""

    def __init__(self, tau_s: float = POSE_FILTER_TAU_S) -> None:
        self._femur_pos = _Pt1Vector(tau_s)
        self._femur_quat = _Pt1Quaternion(tau_s)
        self._tibia_pos = _Pt1Vector(tau_s)
        self._tibia_quat = _Pt1Quaternion(tau_s)

    def apply(self, frame: Frame) -> Frame:
        return replace(
            frame,
            femur_pos_m=self._femur_pos.apply(frame.femur_pos_m, frame.t),
            femur_quat_wxyz=self._femur_quat.apply(frame.femur_quat_wxyz, frame.t),
            tibia_pos_m=self._tibia_pos.apply(frame.tibia_pos_m, frame.t),
            tibia_quat_wxyz=self._tibia_quat.apply(frame.tibia_quat_wxyz, frame.t),
        )
