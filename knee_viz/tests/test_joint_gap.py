import numpy as np
import pytest

from knee_viz.core.joint_gap import JointGapCalculator
from knee_viz.core.transforms import normalize, pose_from_metres


def _brute_force(origin, direction, triangles):
    """Unaccelerated Moeller-Trumbore over every triangle, as ground truth."""
    v0, e1, e2 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > 1e-9
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = (qvec @ direction) * inv
    t = np.einsum("ij,ij->i", e2, qvec) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    return t[hit].min() if hit.any() else float("nan")


@pytest.fixture(scope="module")
def rays(femur, tibia):
    """Both plateau rays, expressed in the femur reference frame."""
    origins = np.stack([tibia.landmarks["tibia_medial"], tibia.landmarks["tibia_lateral"]])
    direction = normalize(tibia.landmarks["tibia_proximal"] - tibia.landmarks["tibia_distal"])
    to_femur = femur.registration.matrix @ np.linalg.inv(tibia.registration.matrix)
    return (
        origins @ to_femur[:3, :3].T + to_femur[:3, 3],
        to_femur[:3, :3] @ direction,
    )


def test_crop_keeps_the_condylar_region(femur):
    gap = JointGapCalculator(femur)
    assert 0 < gap.n_triangles < femur.n_triangles
    assert gap.max_circumradius_mm < 15.0


def test_matches_brute_force_over_the_whole_mesh(femur, rays):
    """The crop plus the perpendicular-distance prefilter must be lossless."""
    gap = JointGapCalculator(femur)
    origins, direction = rays
    result = gap.measure(origins, direction, np.eye(4))
    for measured, origin in ((result.medial_mm, origins[0]), (result.lateral_mm, origins[1])):
        expected = _brute_force(origin, direction / np.linalg.norm(direction), femur.triangles)
        if np.isnan(expected):
            assert np.isnan(measured)
        else:
            assert abs(measured - expected) < 1e-9


def test_gaps_are_physiologically_plausible(femur, rays):
    gap = JointGapCalculator(femur)
    result = gap.measure(*rays, np.eye(4))
    assert result.is_valid
    for value in (result.medial_mm, result.lateral_mm):
        assert 0.0 < value < 40.0


def test_hit_points_lie_on_the_rays(femur, rays):
    gap = JointGapCalculator(femur)
    origins, direction = rays
    result = gap.measure(origins, direction, np.eye(4))
    unit = direction / np.linalg.norm(direction)
    for hit, origin, distance in (
        (result.medial_hit, origins[0], result.medial_mm),
        (result.lateral_hit, origins[1], result.lateral_mm),
    ):
        assert np.allclose(hit, origin + unit * distance, atol=1e-6)


def test_a_missing_ray_yields_nan_not_an_exception(femur, rays):
    gap = JointGapCalculator(femur)
    origins, direction = rays
    far_away = origins + np.array([0.0, 0.0, 10_000.0])
    result = gap.measure(far_away, direction, np.eye(4))
    assert np.isnan(result.medial_mm) and np.isnan(result.lateral_mm)
    assert not result.is_valid


def test_result_is_invariant_to_the_femur_pose(femur, rays):
    """Measuring in the femur frame must give the same answer wherever it sits."""
    gap = JointGapCalculator(femur)
    origins, direction = rays
    baseline = gap.measure(origins, direction, np.eye(4))

    pose = pose_from_metres(np.array([0.31, -0.22, 0.47, 0.79]), np.array([1.3, -0.8, 2.2]))
    moved = gap.measure(
        origins @ pose[:3, :3].T + pose[:3, 3],
        pose[:3, :3] @ direction,
        pose,
    )
    assert abs(moved.medial_mm - baseline.medial_mm) < 1e-6
    assert abs(moved.lateral_mm - baseline.lateral_mm) < 1e-6
