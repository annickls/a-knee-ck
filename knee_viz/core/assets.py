"""Readers for the on-disk model assets: STL meshes, Slicer fiducials, marker YAML.

Everything here returns plain numpy — no Qt, no OpenGL — so the whole geometry
pipeline can be exercised headless.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml
from stl import mesh as stl_mesh

# 3D Slicer Markups .fcsv files carry three '#'-prefixed header lines, then one
# row per fiducial: id, x, y, z, ow, ox, oy, oz, vis, sel, lock, label, ...
_FCSV_HEADER_LINES = 3
_FCSV_XYZ_COLUMNS = slice(1, 4)
_FCSV_LABEL_COLUMN = 11


def read_fcsv(path: Path) -> dict[str, np.ndarray]:
    """Read a Slicer fiducial file into ``{label: xyz}`` (millimetres, CT frame)."""
    landmarks: dict[str, np.ndarray] = {}
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(_FCSV_HEADER_LINES):
            next(reader, None)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            label = row[_FCSV_LABEL_COLUMN]
            landmarks[label] = np.array([float(v) for v in row[_FCSV_XYZ_COLUMNS]])
    if not landmarks:
        raise ValueError(f"no fiducials found in {path}")
    return landmarks


def read_marker_yaml(path: Path) -> dict[str, np.ndarray]:
    """Read marker_coordinates.yaml into ``{set_name: (N, 3) array}``.

    Each set (``femur_slicer``, ``femur_ref``, ``tibia_slicer``, ``tibia_ref``)
    is a list of ``{x, y, z}`` mappings. Unlike the original implementation this
    reads however many points a set actually contains instead of assuming four.
    """
    with open(path, "r") as handle:
        content = yaml.safe_load(handle)
    return {
        name: np.array([[p["x"], p["y"], p["z"]] for p in points], dtype=float)
        for name, points in content.items()
    }


def load_stl_triangles(path: Path) -> np.ndarray:
    """Load an STL as an ``(n_triangles, 3, 3)`` array of vertex positions."""
    return np.asarray(stl_mesh.Mesh.from_file(str(path)).vectors, dtype=np.float64)


def triangles_to_mesh_arrays(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten triangles into the ``(vertices, faces)`` pair GLMeshItem expects.

    STL has no vertex sharing, so each triangle keeps its own three vertices.
    That means shading is flat regardless of any smoothing request downstream.
    """
    vertices = triangles.reshape(-1, 3).astype(np.float32)
    faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
    return vertices, faces
