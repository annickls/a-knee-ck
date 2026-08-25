"""Data acquisition. Everything except :mod:`knee_viz.data.driver` is Qt-free."""

from __future__ import annotations

from knee_viz.data.csv_replay import CsvReplaySource
from knee_viz.data.csv_tail import CsvTailSource
from knee_viz.data.source import DataSource, DataSourceError, Frame

__all__ = ["CsvReplaySource", "CsvTailSource", "DataSource", "DataSourceError", "Frame", "make_source"]


def make_source(config) -> DataSource:
    """Build the source named by ``config.source_kind``."""
    if config.source_kind == "replay":
        return CsvReplaySource(config.csv_path, rate_hz=config.replay_hz, loop=config.replay_loop)
    if config.source_kind == "tail":
        return CsvTailSource(config.csv_path)
    raise ValueError(f"unknown source kind {config.source_kind!r}")
