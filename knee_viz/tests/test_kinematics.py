import math

import numpy as np
import pytest

from knee_viz.core.kinematics import Dof6, GroodSuntayEngine, KneeFrames, Side, ZeroOffsets


def _neutral_landmarks():
    """A textbook aligned knee: femur above the tibia, axes mutually orthogonal."""
    femur = {
        "femur_medial": np.array([-30.0, 0.0, 10.0]),
        "femur_lateral": np.array([30.0, 0.0, 10.0]),
        "femur_distal": np.array([0.0, 0.0, 10.0]),
        "femur_proximal": np.array([0.0, 0.0, 410.0]),
    }
    tibia = {
        "tibia_medial": np.array([-28.0, 0.0, 0.0]),
        "tibia_lateral": np.array([28.0, 0.0, 0.0]),
        "tibia_proximal": np.array([0.0, 0.0, 0.0]),
        "tibia_distal": np.array([0.0, 0.0, -400.0]),
    }
    return femur, tibia


def test_frames_are_orthonormal_triads():
    frames = GroodSuntayEngine.build_frames(*_neutral_landmarks())
    for triad in ((frames.e1f, frames.e2f, frames.e3f), (frames.e1t, frames.e2t, frames.e3t)):
        for axis in triad:
            assert np.isclose(np.linalg.norm(axis), 1.0)
        a, b, c = triad
        assert abs(np.dot(a, b)) < 1e-12
        assert abs(np.dot(b, c)) < 1e-12
        assert abs(np.dot(a, c)) < 1e-12


def test_neutral_pose_reads_near_zero():
    dof = GroodSuntayEngine().solve(GroodSuntayEngine.build_frames(*_neutral_landmarks()))
    assert abs(dof.flexion_deg) < 1e-9
    assert abs(dof.adduction_deg) < 1e-9
    assert abs(dof.rotation_deg) < 1e-9


@pytest.mark.parametrize("angle_deg", [-25.0, -5.0, 15.0, 45.0, 90.0])
def test_imposed_flexion_is_recovered(angle_deg):
    """Rotate the tibia about the femoral flexion axis and read the angle back."""
    femur, tibia = _neutral_landmarks()
    theta = math.radians(angle_deg)
    # Rotation about +x (the femoral medial-lateral axis).
    rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(theta), -math.sin(theta)],
        [0.0, math.sin(theta), math.cos(theta)],
    ])
    rotated = {name: rot @ xyz for name, xyz in tibia.items()}
    dof = GroodSuntayEngine().solve(GroodSuntayEngine.build_frames(femur, rotated))
    assert abs(abs(dof.flexion_deg) - abs(angle_deg)) < 1e-6


def test_arc_arguments_are_clipped():
    """A dot product of unit vectors can exceed 1.0 in floating point.

    ``[1/sqrt(3)] * 3`` dotted with itself gives 1.0000000000000002, which the
    unclipped original would have handed straight to ``math.acos``.
    """
    axis = np.full(3, 1.0 / math.sqrt(3.0))
    assert float(np.dot(axis, axis)) > 1.0, "precondition: this must overflow the domain"

    other = np.array([0.0, 0.0, 1.0])
    frames = KneeFrames(
        e1f=axis, e2f=other, e3f=other,
        e1t=other, e2t=other, e3t=axis,
        floating=other,
        femur_origin=np.zeros(3), tibia_origin=np.zeros(3),
    )
    dof = GroodSuntayEngine().solve(frames)
    assert all(np.isfinite(v) for v in dof.as_tuple())
    assert np.isclose(abs(dof.adduction_deg), 90.0)


def test_side_flips_the_mediolateral_signs():
    femur, tibia = _neutral_landmarks()
    tibia = dict(tibia)
    tibia["tibia_distal"] = np.array([40.0, 0.0, -400.0])
    frames = GroodSuntayEngine.build_frames(femur, tibia)
    left = GroodSuntayEngine(Side.LEFT).solve(frames)
    right = GroodSuntayEngine(Side.RIGHT).solve(frames)
    assert np.isclose(left.adduction_deg, -right.adduction_deg)
    assert np.isclose(left.rotation_deg, -right.rotation_deg)
    assert np.isclose(left.flexion_deg, right.flexion_deg)


def test_zeroing_nulls_every_degree_of_freedom():
    """The original only ever zeroed the three translations."""
    raw = Dof6(11.0, -3.0, 7.0, 4.5, -2.5, 19.0)
    offsets = ZeroOffsets()
    offsets.capture(raw)
    assert offsets.apply(raw).as_tuple() == (0.0,) * 6


def test_zeroing_is_a_pure_offset():
    offsets = ZeroOffsets()
    offsets.capture(Dof6(10.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert offsets.apply(Dof6(25.0, 0.0, 0.0, 0.0, 0.0, 0.0)).flexion_deg == 15.0
    offsets.clear()
    assert offsets.apply(Dof6(25.0, 0.0, 0.0, 0.0, 0.0, 0.0)).flexion_deg == 25.0
