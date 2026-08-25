"""Grood & Suntay joint coordinate system for the tibiofemoral joint.

Ported from the "new implementation from paper" blocks of the original
``update_visualization.calculate_grood_suntay_angles``. The original computed
every angle twice (a legacy formulation, then the paper one overwriting it);
only the paper formulation survives here.

Two deliberate simplifications, both exactly equivalent to the original:

* The ``np.isclose`` sign-recovery hacks are ``atan2``. ``floating`` is the cross
  product of ``e1f`` with ``e3t`` and so is perpendicular to ``e1f``, which puts
  it in the ``e2f``/``e3f`` plane; therefore ``cos_alpha**2 + sin_alpha**2 == 1``
  and the sign test reduces to the sign of ``sin_alpha``. Using ``atan2`` drops a
  float-tolerance comparison that could flip sign near zero. Same for rotation.
* The original fed ``math.cos(adduction)`` into the translations, where
  ``adduction = acos(e1f . e3t)``. That is just ``e1f . e3t`` again, so the whole
  legacy block collapses to ``cos_beta``.

All arc functions clip their argument; the original relied on dot products of
unit vectors never drifting past 1.0, which is not guaranteed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from knee_viz.core.transforms import normalize


class Side(enum.Enum):
    """Which knee. Sign conventions in the source data were derived for a left knee."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class KneeFrames:
    """The femoral and tibial anatomical triads plus the floating axis, in world mm."""

    e1f: np.ndarray  # femoral flexion axis, medial -> lateral
    e2f: np.ndarray  # femoral anterior-posterior axis
    e3f: np.ndarray  # femoral long axis, distal -> proximal
    e1t: np.ndarray  # tibial medial-lateral axis
    e2t: np.ndarray  # tibial anterior-posterior axis
    e3t: np.ndarray  # tibial anatomical axis, distal -> proximal
    floating: np.ndarray  # common perpendicular to e1f and e3t
    femur_origin: np.ndarray  # midpoint of the femoral condyle landmarks
    tibia_origin: np.ndarray  # midpoint of the tibial plateau landmarks


@dataclass(frozen=True, slots=True)
class Dof6:
    """The six tibiofemoral degrees of freedom. Angles in degrees, translations in mm."""

    flexion_deg: float = 0.0
    adduction_deg: float = 0.0
    rotation_deg: float = 0.0
    anterior_mm: float = 0.0
    medial_mm: float = 0.0
    proximal_mm: float = 0.0

    @classmethod
    def zeros(cls) -> "Dof6":
        return cls()

    def __sub__(self, other: "Dof6") -> "Dof6":
        return Dof6(*(a - b for a, b in zip(self.as_tuple(), other.as_tuple())))

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.flexion_deg,
            self.adduction_deg,
            self.rotation_deg,
            self.anterior_mm,
            self.medial_mm,
            self.proximal_mm,
        )


class GroodSuntayEngine:
    """Turns landmark positions into the six degrees of freedom.

    ``solve`` always returns **raw** values. Zeroing is applied afterwards by
    :class:`ZeroOffsets`, so no offset can be silently overwritten by a later
    assignment — which is how the original lost its rotation and adduction offsets.
    """

    def __init__(self, side: Side = Side.LEFT) -> None:
        self.side = side

    @property
    def _handedness(self) -> float:
        """Sign applied to the mediolateral quantities.

        Only ``Side.LEFT`` is validated against the recorded data; the right-knee
        flip follows the Grood & Suntay convention but is untested here.
        """
        return 1.0 if self.side is Side.LEFT else -1.0

    @staticmethod
    def build_frames(
        femur: Mapping[str, np.ndarray],
        tibia: Mapping[str, np.ndarray],
    ) -> KneeFrames:
        """Build both anatomical triads from the eight landmarks."""
        e1f = normalize(femur["femur_lateral"] - femur["femur_medial"])
        temp_femur = normalize(femur["femur_proximal"] - femur["femur_distal"])
        e2f = normalize(np.cross(e1f, temp_femur))
        e3f = normalize(np.cross(e2f, e1f))

        e3t = normalize(tibia["tibia_proximal"] - tibia["tibia_distal"])
        temp_tibia = normalize(tibia["tibia_lateral"] - tibia["tibia_medial"])
        e2t = normalize(np.cross(temp_tibia, e3t))
        e1t = normalize(np.cross(e3t, e2t))

        return KneeFrames(
            e1f=e1f,
            e2f=e2f,
            e3f=e3f,
            e1t=e1t,
            e2t=e2t,
            e3t=e3t,
            floating=normalize(np.cross(e1f, e3t)),
            femur_origin=(femur["femur_medial"] + femur["femur_lateral"]) / 2.0,
            tibia_origin=(tibia["tibia_medial"] + tibia["tibia_lateral"]) / 2.0,
        )

    def solve(self, frames: KneeFrames) -> Dof6:
        """Compute the raw, un-zeroed six degrees of freedom."""
        flexion = np.degrees(
            np.arctan2(-np.dot(frames.floating, frames.e3f), np.dot(frames.e2f, frames.floating))
        )

        # cos of the angle between the femoral flexion axis and the tibial long
        # axis; also the factor coupling the two in-plane translations below.
        cos_beta = float(np.clip(np.dot(frames.e1f, frames.e3t), -1.0, 1.0))
        adduction = -np.degrees(np.arcsin(cos_beta))

        rotation = np.degrees(
            np.arctan2(-np.dot(frames.floating, frames.e1t), np.dot(frames.e2t, frames.floating))
        )

        offset = frames.femur_origin - frames.tibia_origin
        along_e1f = float(np.dot(offset, frames.e1f))
        along_e3t = float(np.dot(offset, frames.e3t))

        handedness = self._handedness
        return Dof6(
            flexion_deg=float(flexion),
            adduction_deg=float(adduction) * handedness,
            rotation_deg=float(rotation) * handedness,
            anterior_mm=-float(np.dot(offset, frames.floating)),
            medial_mm=(along_e1f + along_e3t * cos_beta) * handedness,
            proximal_mm=-along_e3t - along_e1f * cos_beta,
        )


class ZeroOffsets:
    """Baseline subtracted from the raw degrees of freedom, for all six uniformly."""

    def __init__(self) -> None:
        self.baseline = Dof6.zeros()

    def capture(self, raw: Dof6) -> None:
        """Make the supplied pose read as zero from now on."""
        self.baseline = raw

    def clear(self) -> None:
        self.baseline = Dof6.zeros()

    def apply(self, raw: Dof6) -> Dof6:
        return raw - self.baseline
