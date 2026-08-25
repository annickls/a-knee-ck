"""Rigid-body transform helpers.

All geometry in this package is in **millimetres**. Tracker positions arrive from
the CSV in metres; :func:`pose_from_metres` is the only place that conversion
happens.
"""

from __future__ import annotations

import numpy as np

MM_PER_M = 1000.0


def normalize(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled to unit length. Raises on a zero-length vector."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return v / norm


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a scalar-first quaternion ``(w, x, y, z)`` to a 3x3 rotation."""
    q = np.asarray(quat_wxyz, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("cannot build a rotation from a zero-length quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def pose(quat_wxyz: np.ndarray, position_mm: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a quaternion and a position in mm."""
    matrix = np.eye(4)
    matrix[:3, :3] = quat_to_matrix(quat_wxyz)
    matrix[:3, 3] = np.asarray(position_mm, dtype=float)
    return matrix


def pose_from_metres(quat_wxyz: np.ndarray, position_m: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a tracker sample.

    The tracking system reports positions in metres while every mesh, landmark
    and gap measurement is in millimetres. This function owns that conversion.
    """
    return pose(quat_wxyz, np.asarray(position_m, dtype=float) * MM_PER_M)


def rigid_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform using the transpose rather than a solve."""
    rotation = matrix[:3, :3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverse


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an ``(..., 3)`` array of points."""
    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_direction(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Apply only the rotation part of a 4x4 transform to a direction vector."""
    return np.asarray(direction, dtype=float) @ matrix[:3, :3].T
