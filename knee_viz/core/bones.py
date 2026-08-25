"""Bone models: an STL plus its landmarks, both pre-registered into the tracker frame.

Registration is baked in once at load. From then on a bone is posed purely by the
per-frame tracker transform, so the vertex and triangle arrays never move — which
is what lets :mod:`knee_viz.core.joint_gap` precompute against them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from knee_viz.core import assets
from knee_viz.core.registration import Registration, registration_from_yaml

MARKER_YAML_NAME = "marker_coordinates.yaml"

REQUIRED_LANDMARKS = {
    "femur": ("femur_medial", "femur_lateral", "femur_proximal", "femur_distal"),
    "tibia": ("tibia_medial", "tibia_lateral", "tibia_proximal", "tibia_distal"),
}

_STL_NAMES = {"femur": "Femur.stl", "tibia": "Tibia.stl"}
_LANDMARK_NAMES = {"femur": "femur_landmarks.fcsv", "tibia": "tibia_landmarks.fcsv"}


@dataclass(frozen=True)
class BoneModel:
    """One bone, with all geometry expressed in its own tracker-reference frame."""

    name: str
    triangles: np.ndarray  # (n_triangles, 3, 3)
    vertices: np.ndarray  # (3 * n_triangles, 3) float32, for GLMeshItem
    faces: np.ndarray  # (n_triangles, 3) uint32
    landmarks: dict[str, np.ndarray]
    registration: Registration

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def landmark_array(self, names: tuple[str, ...]) -> np.ndarray:
        """Stack the named landmarks into an ``(N, 3)`` array in a fixed order."""
        return np.stack([self.landmarks[name] for name in names])


def load_bone(model_dir: Path, name: str) -> BoneModel:
    """Load ``name`` ('femur' or 'tibia') from a model directory and register it."""
    if name not in REQUIRED_LANDMARKS:
        raise ValueError(f"unknown bone {name!r}; expected one of {sorted(REQUIRED_LANDMARKS)}")

    registration = registration_from_yaml(model_dir / MARKER_YAML_NAME, name)

    triangles = registration.apply(assets.load_stl_triangles(model_dir / _STL_NAMES[name]))
    vertices, faces = assets.triangles_to_mesh_arrays(triangles)

    landmarks_ct = assets.read_fcsv(model_dir / _LANDMARK_NAMES[name])
    missing = [key for key in REQUIRED_LANDMARKS[name] if key not in landmarks_ct]
    if missing:
        raise KeyError(f"{model_dir / _LANDMARK_NAMES[name]} is missing landmarks: {missing}")
    landmarks = {label: registration.apply(xyz) for label, xyz in landmarks_ct.items()}

    return BoneModel(
        name=name,
        triangles=triangles,
        vertices=vertices,
        faces=faces,
        landmarks=landmarks,
        registration=registration,
    )


def load_knee(model_dir: Path) -> tuple[BoneModel, BoneModel]:
    """Load ``(femur, tibia)`` from a model directory."""
    return load_bone(model_dir, "femur"), load_bone(model_dir, "tibia")
