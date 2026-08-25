import numpy as np
import pytest

from knee_viz.core import transforms as tf


def test_quat_to_matrix_is_a_rotation():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    r = tf.quat_to_matrix(q)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_quat_is_normalised_before_use():
    unit = tf.quat_to_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    scaled = tf.quat_to_matrix(np.array([7.0, 0.0, 0.0, 0.0]))
    assert np.allclose(unit, scaled)


def test_zero_quaternion_raises_rather_than_producing_nan():
    with pytest.raises(ValueError):
        tf.quat_to_matrix(np.zeros(4))


def test_pose_from_metres_converts_to_millimetres():
    pose = tf.pose_from_metres(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.1, -0.2, 0.3]))
    assert np.allclose(pose[:3, 3], [100.0, -200.0, 300.0])


def test_rigid_inverse_round_trips():
    pose = tf.pose_from_metres(np.array([0.2, 0.3, -0.4, 0.85]), np.array([1.0, 2.0, -3.0]))
    assert np.allclose(tf.rigid_inverse(pose) @ pose, np.eye(4), atol=1e-12)


def test_transform_points_matches_explicit_maths():
    pose = tf.pose_from_metres(np.array([0.2, 0.3, -0.4, 0.85]), np.array([1.0, 2.0, -3.0]))
    pts = np.random.default_rng(0).normal(size=(17, 3)) * 50
    assert np.allclose(tf.transform_points(pose, pts), pts @ pose[:3, :3].T + pose[:3, 3])


def test_transform_direction_ignores_translation():
    pose = tf.pose_from_metres(np.array([0.2, 0.3, -0.4, 0.85]), np.array([9.0, 9.0, 9.0]))
    d = np.array([0.0, 0.0, 1.0])
    assert np.isclose(np.linalg.norm(tf.transform_direction(pose, d)), 1.0)
