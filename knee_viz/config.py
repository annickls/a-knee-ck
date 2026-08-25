"""Application configuration, built once at startup and injected everywhere.

Being a frozen dataclass rather than module-level globals means a second window,
a test, or a headless run can each use their own settings without touching
process state -- which is what the original's class-level mutables prevented.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from knee_viz.core.kinematics import Side

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "data_for_gui" / "Model_demo"
DEFAULT_LIVE_CSV = REPO_ROOT / "data_new.csv"

SourceKind = Literal["tail", "replay"]


@dataclass(frozen=True)
class AppConfig:
    model_dir: Path = DEFAULT_MODEL_DIR
    csv_path: Path = DEFAULT_LIVE_CSV
    source_kind: SourceKind = "tail"

    replay_hz: float = 50.0
    replay_loop: bool = True
    poll_interval_ms: int = 20

    debug: bool = False
    side: Side = Side.LEFT
    show_force_bar: bool = True

    gap_crop_radius_mm: float = 80.0
    gap_margin_mm: float = 1.0

    plot_max_points: int = 10_000
    repo_root: Path = field(default=REPO_ROOT)

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "AppConfig":
        parser = argparse.ArgumentParser(prog="knee_viz", description="Live knee joint visualisation")
        parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                            help="folder holding Femur.stl, Tibia.stl, the landmark fcsv files and marker_coordinates.yaml")
        parser.add_argument("--csv", dest="csv_path", type=Path, default=None,
                            help="CSV to read (tail) or replay")
        parser.add_argument("--source", dest="source_kind", choices=("tail", "replay"), default="tail")
        parser.add_argument("--hz", dest="replay_hz", type=float, default=50.0, help="replay rate")
        parser.add_argument("--no-loop", dest="replay_loop", action="store_false")
        parser.add_argument("--side", choices=("left", "right"), default="left")
        parser.add_argument("--debug", action="store_true",
                            default=os.environ.get("KNEEVIZ_DEBUG", "") not in ("", "0"),
                            help="show landmarks, anatomical axes and joint-gap rays")
        args = parser.parse_args(argv)

        csv_path = args.csv_path
        if csv_path is None:
            csv_path = DEFAULT_LIVE_CSV

        return cls(
            model_dir=args.model_dir,
            csv_path=csv_path,
            source_kind=args.source_kind,
            replay_hz=args.replay_hz,
            replay_loop=args.replay_loop,
            side=Side(args.side),
            debug=args.debug,
        )
