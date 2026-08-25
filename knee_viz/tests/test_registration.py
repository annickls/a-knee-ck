import numpy as np

from knee_viz.core.bones import REQUIRED_LANDMARKS
from knee_viz.core.registration import kabsch, registration_from_yaml
from knee_viz.core.transforms import quat_to_matrix


def test_kabsch_recovers_a_known_rigid_transform():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(8, 3)) * 40
    rotation = quat_to_matrix(np.array([0.3, -0.2, 0.5, 0.78]))
    translation = np.array([12.0, -45.0, 7.5])

    reg = kabsch(source, source @ rotation.T + translation)

    assert np.allclose(reg.rotation, rotation, atol=1e-10)
    assert np.allclose(reg.translation, translation, atol=1e-9)
    assert reg.rmsd < 1e-9


def test_kabsch_never_returns_a_reflection():
    rng = np.random.default_rng(11)
    source = rng.normal(size=(6, 3))
    mirrored = source * np.array([1.0, 1.0, -1.0])
    assert np.linalg.det(kabsch(source, mirrored).rotation) > 0


def test_registration_inverse_round_trips(femur):
    reg = femur.registration
    pts = np.random.default_rng(1).normal(size=(20, 3)) * 100
    assert np.allclose(reg.inverse().apply(reg.apply(pts)), pts, atol=1e-9)


def test_marker_registration_fits_tightly(model_dir):
    """A loose fit here means the digitised markers do not match the CAD body."""
    for bone in ("femur", "tibia"):
        reg = registration_from_yaml(model_dir / "marker_coordinates.yaml", bone)
        assert reg.rmsd < 0.05, f"{bone} marker RMSD {reg.rmsd:.4f} mm"
        assert np.isclose(np.linalg.det(reg.rotation), 1.0)


def test_bones_expose_their_required_landmarks(femur, tibia):
    for bone in (femur, tibia):
        for name in REQUIRED_LANDMARKS[bone.name]:
            assert name in bone.landmarks
        assert bone.landmark_array(REQUIRED_LANDMARKS[bone.name]).shape == (4, 3)


def test_mesh_arrays_are_consistent(femur):
    assert femur.vertices.shape == (3 * femur.n_triangles, 3)
    assert femur.faces.shape == (femur.n_triangles, 3)
    assert femur.faces.max() == len(femur.vertices) - 1
