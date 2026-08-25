import numpy as np
import pytest

from knee_viz.data import CsvReplaySource, CsvTailSource, DataSourceError
from knee_viz.data.csv_format import MIN_COLUMNS, parse_row


def _good_row() -> str:
    values = ["1785340219.27"] + ["0.1"] * 6
    values += ["0.08", "0.5", "-0.36", "0.0", "0.0", "0.0", "1.0"]  # tibia
    values += ["0.07", "0.47", "0.04", "0.0", "0.0", "0.0", "1.0"]  # femur
    values += ["0.0"] * 3 + ["0.0", "0.0", "0.0", "1.0"]  # sensor
    return ",".join(values) + ","


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "not,a,row",
        ",".join(["1.0"] * (MIN_COLUMNS - 1)),  # too few columns
        ",".join(["nan"] * MIN_COLUMNS),  # non-finite
        ",".join(["abc"] * MIN_COLUMNS),  # unparseable
        ",".join(["1.0"] * 10 + ["0", "0", "0", "0"] + ["1.0"] * 14),  # tracker dropout
    ],
)
def test_malformed_rows_are_dropped_not_raised(line):
    assert parse_row(line) is None


def test_quaternions_are_reordered_to_scalar_first_and_normalised():
    frame = parse_row(_good_row())
    assert frame is not None
    for quat in (frame.tibia_quat_wxyz, frame.femur_quat_wxyz):
        assert np.isclose(np.linalg.norm(quat), 1.0)
        assert np.isclose(quat[0], 1.0)  # CSV stores w last; it must land first


def test_force_and_torque_are_carried_for_the_future_force_bar():
    frame = parse_row(_good_row())
    assert frame.force_n.shape == (3,)
    assert frame.torque_nm.shape == (3,)


def test_replay_emits_consecutive_frames_without_drops(motion_csv):
    """Guards a float-truncation bug: 0.58 s * 50 Hz evaluates to 28.999999."""
    now = [0.0]
    source = CsvReplaySource(motion_csv, rate_hz=50.0, loop=True, clock=lambda: now[0])
    source.open()
    assert source.poll().index == 0
    assert source.poll() is None, "the same clock tick must not re-emit"

    seen = []
    for step in range(1, 501):
        now[0] = step / 50.0
        frame = source.poll()
        assert frame is not None, f"dropped frame at step {step}"
        seen.append(frame.index)
    assert seen == list(range(1, 501))


def test_replay_seek_and_loop(motion_csv):
    now = [0.0]
    source = CsvReplaySource(motion_csv, rate_hz=50.0, loop=True, clock=lambda: now[0])
    source.open()
    source.seek(len(source) - 2)
    now[0] = 100.0
    assert source.poll() is not None


def test_replay_on_a_missing_file_raises_once(tmp_path):
    with pytest.raises(DataSourceError):
        CsvReplaySource(tmp_path / "nope.csv").open()


def test_tail_follows_appends_and_ignores_partial_lines(tmp_path, motion_csv):
    rows = motion_csv.read_text().splitlines()
    path = tmp_path / "live.csv"
    path.write_text(rows[0] + "\n")

    source = CsvTailSource(path)
    source.open()
    assert source.poll() is None, "a header alone is not a frame"

    with open(path, "a") as handle:
        handle.write(rows[1] + "\n")
    first = source.poll()
    assert first is not None
    assert source.poll() is None, "an unchanged file must yield nothing"

    with open(path, "a") as handle:
        handle.write(rows[2][:40])
    assert source.poll() is None, "a half-written line must be ignored"

    with open(path, "a") as handle:
        handle.write(rows[2][40:] + "\n")
    second = source.poll()
    assert second is not None and second.t != first.t


def test_tail_on_a_missing_file_raises_once(tmp_path):
    with pytest.raises(DataSourceError):
        CsvTailSource(tmp_path / "nope.csv").open()
