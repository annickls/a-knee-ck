import numpy as np

from knee_viz.core.filters import PoseFilter, _Pt1Quaternion, _Pt1Vector
from knee_viz.data.source import Frame


def test_first_call_returns_raw_value_unchanged():
    f = _Pt1Vector(tau_s=0.1)
    value = np.array([1.0, 2.0, 3.0])
    result = f.apply(value, t=0.0)
    assert np.array_equal(result, value)


def test_converges_toward_a_held_target():
    f = _Pt1Vector(tau_s=0.1)
    target = np.array([1.0, 0.0, 0.0])
    f.apply(np.zeros(3), t=0.0)

    errors = []
    for step in range(1, 21):
        out = f.apply(target, t=step * 0.02)
        errors.append(np.linalg.norm(target - out))

    assert all(a >= b - 1e-9 for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 0.05


def test_zero_or_negative_dt_holds_state():
    f = _Pt1Vector(tau_s=0.1)
    f.apply(np.array([0.0]), t=0.0)
    held = f.apply(np.array([1.0]), t=0.0)  # duplicate timestamp
    assert held[0] == 0.0


def test_quaternion_filter_stays_unit_norm():
    f = _Pt1Quaternion(tau_s=0.1)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.9, 0.1, 0.0, 0.0])
    q1 /= np.linalg.norm(q1)
    f.apply(q0, t=0.0)
    out = f.apply(q1, t=0.02)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_quaternion_filter_handles_hemisphere_flip():
    f = _Pt1Quaternion(tau_s=0.1)
    q = np.array([0.9689, 0.2474, 0.0, 0.0])
    q /= np.linalg.norm(q)
    f.apply(q, t=0.0)

    q_next = np.array([0.9613, 0.2756, 0.0, 0.0])
    q_next /= np.linalg.norm(q_next)
    flipped = -q_next  # same rotation, opposite hemisphere

    out = f.apply(flipped, t=0.02)
    # Output should stay close to the previous state, not jump toward -q_next.
    assert np.dot(out, q) > 0.9


def test_pose_filter_leaves_force_and_torque_untouched():
    filt = PoseFilter(tau_s=0.1)
    frame = Frame(
        t=0.0,
        tibia_pos_m=np.array([0.0, 0.0, 0.0]),
        tibia_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        femur_pos_m=np.array([0.0, 0.0, 0.0]),
        femur_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        force_n=np.array([1.0, 2.0, 3.0]),
        torque_nm=np.array([0.1, 0.2, 0.3]),
        index=5,
    )
    out = filt.apply(frame)
    assert out.t == frame.t
    assert out.index == frame.index
    assert np.array_equal(out.force_n, frame.force_n)
    assert np.array_equal(out.torque_nm, frame.torque_nm)


def test_pose_filter_smooths_position_over_repeated_frames():
    filt = PoseFilter(tau_s=0.1)
    quat = np.array([1.0, 0.0, 0.0, 0.0])

    def make(t, tibia_x):
        return Frame(
            t=t,
            tibia_pos_m=np.array([tibia_x, 0.0, 0.0]),
            tibia_quat_wxyz=quat,
            femur_pos_m=np.zeros(3),
            femur_quat_wxyz=quat,
        )

    first = filt.apply(make(0.0, 0.0))
    assert first.tibia_pos_m[0] == 0.0

    jumped = filt.apply(make(0.02, 1.0))
    assert 0.0 < jumped.tibia_pos_m[0] < 1.0  # eased, not snapped

    settled = jumped
    for step in range(2, 40):
        settled = filt.apply(make(step * 0.02, 1.0))
    assert settled.tibia_pos_m[0] > 0.95
