"""The plot modes, and the neutrality gate that decides which samples count.

A sample is only meaningful for the quantity being plotted if the *other*
degrees of freedom are near neutral -- a varus/valgus reading taken while the
tibia is rotated 20 degrees is not comparable with one taken in neutral
rotation. The original expressed this as a chain of ``if`` statements inside the
plot widget; here each mode declares its own gate, so adding a mode is one
table entry rather than an edit in three places.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Mapping

from knee_viz.core.state import KneeState

# Half-width of the "near neutral" corridor, in degrees or mm depending on the
# gated signal. Both were 4 in the original constants.
NEUTRAL_ANGLE_DEG = 4.0
NEUTRAL_TRANSLATION_MM = 10.0

# Names of the per-sample signals kept in the plot's ring buffer.
SIGNALS = ("flexion", "adduction", "rotation", "anterior", "medial", "gap_medial", "gap_lateral")

# Upper end of the force bar's fill, in the channel's own unit. The bar shows
# green at 0 and red at the limit.
TORQUE_LIMIT_NM = 2.0
FORCE_LIMIT_N = 50.0


class PlotMode(enum.Enum):
    JOINT_GAPS = "joint_gaps"
    ADDUCTION = "adduction"
    ROTATION = "rotation"
    ANTERIOR = "anterior"
    MEDIAL = "medial"


@dataclass(frozen=True)
class ModeSpec:
    """Everything that differs between one plot mode and another."""

    label: str  # button text
    title: str  # plot title
    left_label: str  # caption at -x
    right_label: str  # caption at +x
    unit: str
    x_limit: float
    columns: tuple[str, ...]  # buffer signals drawn, in order
    signs: tuple[float, ...]  # sign applied to each column
    gate_signal: str  # signal that must be near neutral
    gate_limit: float


MODES: Mapping[PlotMode, ModeSpec] = {
    PlotMode.JOINT_GAPS: ModeSpec(
        label="joint gaps",
        title="medial / lateral joint gap",
        left_label="MED",
        right_label="LAT",
        unit="mm",
        x_limit=25.0,
        columns=("gap_medial", "gap_lateral"),
        signs=(-1.0, 1.0),
        gate_signal="rotation",
        gate_limit=NEUTRAL_ANGLE_DEG,
    ),
    PlotMode.ADDUCTION: ModeSpec(
        label="var / val",
        title="adduction / abduction angle",
        left_label="VAL",
        right_label="VAR",
        unit="°",
        x_limit=25.0,
        columns=("adduction",),
        signs=(1.0,),
        gate_signal="rotation",
        gate_limit=NEUTRAL_ANGLE_DEG,
    ),
    PlotMode.ROTATION: ModeSpec(
        label="rotation",
        title="internal / external rotation",
        left_label="INT",
        right_label="EXT",
        unit="°",
        x_limit=35.0,
        columns=("rotation",),
        signs=(1.0,),
        gate_signal="adduction",
        gate_limit=NEUTRAL_ANGLE_DEG,
    ),
    PlotMode.ANTERIOR: ModeSpec(
        label="ant / post",
        title="anterior / posterior translation",
        left_label="POST",
        right_label="ANT",
        unit="mm",
        x_limit=30.0,
        columns=("anterior",),
        signs=(1.0,),
        gate_signal="medial",
        gate_limit=NEUTRAL_TRANSLATION_MM,
    ),
    PlotMode.MEDIAL: ModeSpec(
        label="med / lat",
        title="medial / lateral translation",
        left_label="LAT",
        right_label="MED",
        unit="mm",
        x_limit=30.0,
        columns=("medial",),
        signs=(1.0,),
        gate_signal="anterior",
        gate_limit=NEUTRAL_TRANSLATION_MM,
    ),
}

# (Frame attribute, index into the xyz vector, limit) for the force bar,
# keyed by the plot mode currently shown.
FORCE_BAR_CHANNELS: Mapping[PlotMode, tuple[str, int, float]] = {
    PlotMode.ROTATION: ("torque_nm", 2, TORQUE_LIMIT_NM),  # torque_z
    PlotMode.ADDUCTION: ("torque_nm", 0, TORQUE_LIMIT_NM),  # torque_x
    PlotMode.JOINT_GAPS: ("force_n", 2, FORCE_LIMIT_N),  # force_z
    PlotMode.ANTERIOR: ("force_n", 0, FORCE_LIMIT_N),  # force_x
    PlotMode.MEDIAL: ("force_n", 1, FORCE_LIMIT_N),  # force_y
}


def sample_from_state(state: KneeState) -> dict[str, float]:
    """Flatten a :class:`KneeState` into the signals the plot stores."""
    return {
        "flexion": state.dof.flexion_deg,
        "adduction": state.dof.adduction_deg,
        "rotation": state.dof.rotation_deg,
        "anterior": state.dof.anterior_mm,
        "medial": state.dof.medial_mm,
        "gap_medial": state.gaps.medial_mm,
        "gap_lateral": state.gaps.lateral_mm,
    }
