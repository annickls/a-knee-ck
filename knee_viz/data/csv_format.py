"""The recorded CSV column layout. The only module that knows column indices.

Header (note the stray spaces, and that every row ends with a trailing comma)::

    timestamp,force_x,force_y,force_z,torque_x, torque_y, torque_z,
    tibia_x, tibia_y, tibia_z,tibia_qx, tibia_qy, tibia_qz, tibia_qw,
    femur_x, femur_y, femur_z,femur_qx, femur_qy, femur_qz, femur_qw,
    sensor_x, sensor_y, sensor_z,sensor_qx, sensor_qy, sensor_qz, sensor_qw

Quaternions are stored scalar-last and are reordered to scalar-first here, once.
"""

from __future__ import annotations

import numpy as np

from knee_viz.data.source import Frame

MIN_COLUMNS = 28

TIMESTAMP = 0
FORCE = slice(1, 4)
TORQUE = slice(4, 7)
TIBIA_POS = slice(7, 10)
TIBIA_QUAT_XYZW = slice(10, 14)
FEMUR_POS = slice(14, 17)
FEMUR_QUAT_XYZW = slice(17, 21)
SENSOR_POS = slice(21, 24)
SENSOR_QUAT_XYZW = slice(24, 28)

_MIN_QUAT_NORM = 1e-6


def _to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray | None:
    """Reorder to scalar-first and normalise; None if the tracker dropped out."""
    norm = np.linalg.norm(quat_xyzw)
    if not np.isfinite(norm) or norm < _MIN_QUAT_NORM:
        return None
    x, y, z, w = quat_xyzw / norm
    return np.array([w, x, y, z])


def parse_row(line: str, index: int = -1) -> Frame | None:
    """Parse one CSV data line into a :class:`Frame`, or None if unusable.

    Returns None rather than raising: a live file is appended to while it is
    read, so truncated and partially-flushed lines are routine.
    """
    fields = line.strip().split(",")
    if len(fields) < MIN_COLUMNS:
        return None

    try:
        values = np.array([float(v) for v in fields[:MIN_COLUMNS]])
    except ValueError:
        return None
    if not np.isfinite(values).all():
        return None

    tibia_quat = _to_wxyz(values[TIBIA_QUAT_XYZW])
    femur_quat = _to_wxyz(values[FEMUR_QUAT_XYZW])
    if tibia_quat is None or femur_quat is None:
        return None

    return Frame(
        t=float(values[TIMESTAMP]),
        tibia_pos_m=values[TIBIA_POS],
        tibia_quat_wxyz=tibia_quat,
        femur_pos_m=values[FEMUR_POS],
        femur_quat_wxyz=femur_quat,
        force_n=values[FORCE],
        torque_nm=values[TORQUE],
        index=index,
    )


def is_header(line: str) -> bool:
    return line.lstrip().lower().startswith("timestamp")
