"""The frame contract between data acquisition and the measurement pipeline."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


class DataSourceError(RuntimeError):
    """Raised by :meth:`DataSource.open` when a source cannot be used at all."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One tracker sample.

    Positions are in metres exactly as the tracking system reports them;
    conversion to millimetres happens in :mod:`knee_viz.core.transforms`.
    Quaternions are scalar-first and unit norm.
    """

    t: float
    tibia_pos_m: np.ndarray
    tibia_quat_wxyz: np.ndarray
    femur_pos_m: np.ndarray
    femur_quat_wxyz: np.ndarray
    force_n: np.ndarray | None = None
    torque_nm: np.ndarray | None = None
    index: int = -1


class DataSource(abc.ABC):
    """A pollable source of tracker frames."""

    @abc.abstractmethod
    def open(self) -> None:
        """Validate the source. Raises :class:`DataSourceError` once, not per poll."""

    @abc.abstractmethod
    def poll(self) -> Frame | None:
        """Return the newest unseen frame, or None.

        Must not block, and must not raise on a malformed, partial or duplicate
        row -- such rows are dropped by returning None. A live acquisition can
        produce any of those at any moment and none of them are exceptional.
        """

    @abc.abstractmethod
    def close(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def nominal_hz(self) -> float:
        """Expected sample rate, used to size the polling interval."""

    def __enter__(self) -> "DataSource":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
