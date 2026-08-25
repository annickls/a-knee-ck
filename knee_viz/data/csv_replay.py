"""Replays a finished recording at a chosen rate, for demos and testing.

The clock is injectable so tests can step through frames deterministically
instead of sleeping.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

from knee_viz.data import csv_format
from knee_viz.data.source import DataSource, DataSourceError, Frame

_FRAME_EPSILON = 1e-6


class CsvReplaySource(DataSource):
    """Plays a recorded CSV back as if it were arriving live."""

    def __init__(
        self,
        path: Path,
        *,
        rate_hz: float = 50.0,
        loop: bool = True,
        start_index: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = Path(path)
        self.loop = loop
        self._rate_hz = rate_hz
        self._clock = clock
        self._start_index = start_index
        self._rows: list[str] = []
        self._started_at: float | None = None
        self._last_emitted = -1

    def open(self) -> None:
        if not self.path.exists():
            raise DataSourceError(f"CSV not found: {self.path}")
        with open(self.path, "r") as handle:
            lines = [line for line in handle if line.strip()]
        if lines and csv_format.is_header(lines[0]):
            lines = lines[1:]
        if not lines:
            raise DataSourceError(f"no data rows in {self.path}")
        self._rows = lines
        self._started_at = None
        self._last_emitted = self._start_index - 1

    def close(self) -> None:
        self._rows = []
        self._started_at = None

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def nominal_hz(self) -> float:
        return self._rate_hz

    def seek(self, index: int) -> None:
        """Jump to a row; the next poll resumes timing from there."""
        self._start_index = max(0, min(index, len(self._rows) - 1))
        self._started_at = None
        self._last_emitted = self._start_index - 1

    def poll(self) -> Frame | None:
        if not self._rows:
            return None

        now = self._clock()
        if self._started_at is None:
            self._started_at = now

        # The epsilon absorbs double-rounding: at 50 Hz an elapsed time of 0.58 s
        # evaluates to 28.999999999999996 frames, which would truncate to 28 and
        # silently drop frame 29.
        elapsed = now - self._started_at
        target = self._start_index + math.floor(elapsed * self._rate_hz + _FRAME_EPSILON)
        if self.loop:
            target %= len(self._rows)
        elif target >= len(self._rows):
            target = len(self._rows) - 1

        if target == self._last_emitted:
            return None
        self._last_emitted = target

        return csv_format.parse_row(self._rows[target], index=target)
