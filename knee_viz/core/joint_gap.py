"""Tibiofemoral joint gap, measured along the tibial anatomical axis.

A ray is cast from each tibial plateau centre landmark along the tibial
anatomical axis (``e3t``, pointing distal -> proximal, i.e. towards the femur)
and the gap is the distance to the first intersection with the femoral surface.

This replaces an older nearest-neighbour query against femoral face centroids,
which measured in an arbitrary direction rather than along an anatomical axis.

Both rays are transformed into the femur's own reference frame rather than
transforming the femur, so the triangle arrays stay static and everything
derived from them can be precomputed once:

* the search region is cropped to the distal condyles at construction time;
* per frame, triangles whose centroid lies further from the ray than their own
  circumradius are rejected before any intersection test runs.

The surviving few hundred triangles then go through a vectorised
Moeller-Trumbore test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from knee_viz.core.bones import BoneModel
from knee_viz.core.transforms import rigid_inverse

_PARALLEL_EPSILON = 1e-9
_MIN_DISTANCE_MM = 1e-6


@dataclass(frozen=True, slots=True)
class GapResult:
    """Medial and lateral gaps in mm; NaN where the ray misses the femur."""

    medial_mm: float
    lateral_mm: float
    medial_hit: np.ndarray | None = None  # world mm, for the debug ray overlay
    lateral_hit: np.ndarray | None = None

    @property
    def is_valid(self) -> bool:
        return bool(np.isfinite(self.medial_mm) and np.isfinite(self.lateral_mm))


class JointGapCalculator:
    """Casts plateau-to-condyle rays against a static femoral surface."""

    def __init__(
        self,
        femur: BoneModel,
        *,
        crop_radius_mm: float = 80.0,
        margin_mm: float = 1.0,
        max_distance_mm: float = 60.0,
    ) -> None:
        condyle_midpoint = (femur.landmarks["femur_medial"] + femur.landmarks["femur_lateral"]) / 2.0

        triangles = femur.triangles
        centroids = triangles.mean(axis=1)
        keep = np.linalg.norm(centroids - condyle_midpoint, axis=1) < crop_radius_mm
        if not keep.any():
            raise ValueError(
                f"no femoral triangles within {crop_radius_mm} mm of the condyle midpoint"
            )

        self._triangles = triangles[keep]
        self._centroids = centroids[keep]
        self._v0 = self._triangles[:, 0]
        self._edge1 = self._triangles[:, 1] - self._triangles[:, 0]
        self._edge2 = self._triangles[:, 2] - self._triangles[:, 0]

        # Per-triangle circumradius about its centroid: any triangle further from
        # the ray than this cannot possibly be hit by it.
        self._circumradii = np.linalg.norm(
            self._triangles - self._centroids[:, None, :], axis=2
        ).max(axis=1)
        self._reach = self._circumradii + margin_mm
        self._max_reach = float(self._reach.max())
        self.max_distance_mm = max_distance_mm

    @property
    def n_triangles(self) -> int:
        return len(self._triangles)

    @property
    def max_circumradius_mm(self) -> float:
        return float(self._circumradii.max())

    def measure(
        self,
        ray_origins_world: np.ndarray,
        ray_direction_world: np.ndarray,
        femur_world_from_ref: np.ndarray,
    ) -> GapResult:
        """Measure the medial and lateral gaps.

        Args:
            ray_origins_world: ``(2, 3)`` mm, ordered ``[medial, lateral]``.
            ray_direction_world: ``(3,)`` unit vector, the tibial anatomical axis
                pointing towards the femur.
            femur_world_from_ref: ``(4, 4)`` placing the femur reference frame in
                the world this frame.
        """
        ref_from_world = rigid_inverse(femur_world_from_ref)
        origins = np.asarray(ray_origins_world, dtype=float) @ ref_from_world[:3, :3].T + ref_from_world[:3, 3]
        direction = np.asarray(ray_direction_world, dtype=float) @ ref_from_world[:3, :3].T
        direction = direction / np.linalg.norm(direction)

        medial, medial_hit = self._cast(origins[0], direction)
        lateral, lateral_hit = self._cast(origins[1], direction)

        to_world = femur_world_from_ref
        return GapResult(
            medial_mm=medial,
            lateral_mm=lateral,
            medial_hit=None if medial_hit is None else to_world[:3, :3] @ medial_hit + to_world[:3, 3],
            lateral_hit=None if lateral_hit is None else to_world[:3, :3] @ lateral_hit + to_world[:3, 3],
        )

    def _cast(self, origin: np.ndarray, direction: np.ndarray) -> tuple[float, np.ndarray | None]:
        """Distance along ``direction`` to the nearest forward surface hit."""
        offset = self._centroids - origin
        along = offset @ direction
        perpendicular_sq = np.einsum("ij,ij->i", offset, offset) - along ** 2

        candidates = (
            (perpendicular_sq <= self._reach ** 2)
            & (along > -self._max_reach)
            & (along < self.max_distance_mm + self._max_reach)
        )
        if not candidates.any():
            return float("nan"), None

        v0 = self._v0[candidates]
        edge1 = self._edge1[candidates]
        edge2 = self._edge2[candidates]

        pvec = np.cross(direction, edge2)
        determinant = np.einsum("ij,ij->i", edge1, pvec)
        usable = np.abs(determinant) > _PARALLEL_EPSILON
        if not usable.any():
            return float("nan"), None
        inverse_determinant = np.where(usable, 1.0 / np.where(usable, determinant, 1.0), 0.0)

        tvec = origin - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inverse_determinant
        qvec = np.cross(tvec, edge1)
        v = (qvec @ direction) * inverse_determinant
        distance = np.einsum("ij,ij->i", edge2, qvec) * inverse_determinant

        hits = usable & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (distance > _MIN_DISTANCE_MM)
        if not hits.any():
            return float("nan"), None

        nearest = float(distance[hits].min())
        return nearest, origin + direction * nearest
