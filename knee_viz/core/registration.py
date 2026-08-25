"""Kabsch registration from the CT/Slicer frame into a bone's tracker frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from knee_viz.core import assets
from knee_viz.core.transforms import rigid_inverse


@dataclass(frozen=True)
class Registration:
    """A rigid map ``target = rotation @ source + translation``."""

    rotation: np.ndarray
    translation: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        """The equivalent 4x4 homogeneous transform."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map an ``(..., 3)`` array of points from the source into the target frame."""
        return np.asarray(points, dtype=float) @ self.rotation.T + self.translation

    def inverse(self) -> "Registration":
        matrix = rigid_inverse(self.matrix)
        return Registration(rotation=matrix[:3, :3], translation=matrix[:3, 3])

    @property
    def rmsd(self) -> float:
        """Residual set by :func:`kabsch`; NaN when built by other means."""
        return float(getattr(self, "_rmsd", np.nan))


def kabsch(source: np.ndarray, target: np.ndarray) -> Registration:
    """Least-squares rigid transform taking ``source`` points onto ``target``.

    Reflections are rejected by flipping the least-significant singular vector,
    so the result is always a proper rotation (``det == +1``).
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"expected matching (N, 3) arrays, got {source.shape} and {target.shape}")

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)

    covariance = (source - source_centroid).T @ (target - target_centroid)
    u, _, vt = np.linalg.svd(covariance)
    if np.linalg.det(vt.T @ u.T) < 0:
        vt[-1, :] *= -1
    rotation = vt.T @ u.T

    registration = Registration(rotation=rotation, translation=target_centroid - rotation @ source_centroid)
    residual = registration.apply(source) - target
    object.__setattr__(registration, "_rmsd", float(np.sqrt((residual ** 2).sum(axis=1).mean())))
    return registration


def registration_from_yaml(yaml_path: Path, bone: str) -> Registration:
    """Build the CT-frame -> tracker-frame registration for ``bone``.

    ``<bone>_slicer`` holds the tracker's marker spheres as digitised in the CT
    scan; ``<bone>_ref`` holds the same markers' coordinates in the tracker body
    frame. Correspondence between the two sets is established upstream by
    ``kabsch/convert_tracker_csv.py``.
    """
    marker_sets = assets.read_marker_yaml(yaml_path)
    for suffix in ("slicer", "ref"):
        if f"{bone}_{suffix}" not in marker_sets:
            raise KeyError(f"{yaml_path} has no '{bone}_{suffix}' marker set")
    return kabsch(marker_sets[f"{bone}_slicer"], marker_sets[f"{bone}_ref"])
