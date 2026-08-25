"""Follows a CSV that another process (the ROS2 recorder) is appending to.

Only the final line is ever parsed: the GUI wants the newest pose, not the
history, so seeking back from EOF keeps the cost independent of file size.
"""

from __future__ import annotations

import os
from pathlib import Path

from knee_viz.data import csv_format
from knee_viz.data.source import DataSource, DataSourceError, Frame

_READ_BACK_BYTES = 4096


class CsvTailSource(DataSource):
    """Polls a growing CSV and yields its most recent complete row."""

    def __init__(self, path: Path, *, nominal_hz: float = 100.0) -> None:
        self.path = Path(path)
        self._nominal_hz = nominal_hz
        self._signature: tuple[float, int] | None = None
        self._last_timestamp: float | None = None

    def open(self) -> None:
        if not self.path.exists():
            raise DataSourceError(f"CSV not found: {self.path}")

    def close(self) -> None:
        self._signature = None
        self._last_timestamp = None

    @property
    def nominal_hz(self) -> float:
        return self._nominal_hz

    def poll(self) -> Frame | None:
        try:
            stat = self.path.stat()
        except OSError:
            return None

        signature = (stat.st_mtime, stat.st_size)
        if signature == self._signature:
            return None
        self._signature = signature

        line = self._read_last_line(stat.st_size)
        if line is None or csv_format.is_header(line):
            return None

        frame = csv_format.parse_row(line)
        if frame is None or frame.t == self._last_timestamp:
            return None
        self._last_timestamp = frame.t
        return frame

    def _read_last_line(self, size: int) -> str | None:
        """Return the last newline-terminated line, or None if there isn't one yet."""
        try:
            with open(self.path, "rb") as handle:
                handle.seek(max(0, size - _READ_BACK_BYTES))
                chunk = handle.read()
        except OSError:
            return None

        # A row still being written has no trailing newline; drop it and take the
        # last complete one instead of parsing half a sample.
        lines = [line for line in chunk.split(b"\n") if line.strip()]
        if not lines:
            return None
        if not chunk.endswith(b"\n"):
            lines = lines[:-1]
        if not lines:
            return None

        try:
            return lines[-1].decode("utf-8")
        except UnicodeDecodeError:
            return None
