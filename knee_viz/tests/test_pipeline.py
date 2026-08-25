import numpy as np

from knee_viz.core.modes import FORCE_BAR_CHANNELS, MODES, PlotMode, sample_from_state
from knee_viz.core.pipeline import KneePipeline
from knee_viz.data import CsvReplaySource


def test_pipeline_runs_a_recording_end_to_end(femur, tibia, motion_csv):
    pipeline = KneePipeline(femur, tibia)
    now = [0.0]
    source = CsvReplaySource(motion_csv, rate_hz=50.0, loop=False, clock=lambda: now[0])
    source.open()

    states = []
    for step in range(400):
        now[0] = step / 50.0
        frame = source.poll()
        if frame is not None:
            states.append(pipeline.step(frame))

    assert len(states) > 100
    for state in states:
        assert all(np.isfinite(v) for v in state.dof.as_tuple())
        assert state.femur_pose.shape == (4, 4)
        assert set(state.landmarks) >= {"femur_medial", "tibia_medial"}


def test_capture_zero_nulls_the_current_pose(femur, tibia, motion_csv):
    pipeline = KneePipeline(femur, tibia)
    source = CsvReplaySource(motion_csv, rate_hz=50.0, clock=lambda: 0.0)
    source.open()
    state = pipeline.step(source.poll())

    pipeline.capture_zero(state)
    zeroed = pipeline.zero.apply(state.raw_dof)
    assert all(abs(v) < 1e-9 for v in zeroed.as_tuple())

    pipeline.clear_zero()
    assert pipeline.zero.apply(state.raw_dof).as_tuple() == state.raw_dof.as_tuple()


def test_every_mode_can_read_a_sample(femur, tibia, motion_csv):
    pipeline = KneePipeline(femur, tibia)
    source = CsvReplaySource(motion_csv, rate_hz=50.0, clock=lambda: 0.0)
    source.open()
    sample = sample_from_state(pipeline.step(source.poll()))

    for mode, spec in MODES.items():
        for column in spec.columns:
            assert column in sample, f"{mode} plots unknown signal {column}"
        assert spec.gate_signal in sample
        assert len(spec.columns) == len(spec.signs)


def test_joint_gaps_mode_yields_two_mirrored_columns():
    """One push per frame must produce both the medial and lateral column."""
    spec = MODES[PlotMode.JOINT_GAPS]
    assert spec.columns == ("gap_medial", "gap_lateral")
    assert spec.signs == (-1.0, 1.0)


def test_force_bar_channel_defined_for_every_mode():
    for mode in PlotMode:
        assert mode in FORCE_BAR_CHANNELS
