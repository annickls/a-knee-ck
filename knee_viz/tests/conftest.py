"""Shared fixtures. Model loading is session-scoped: the STLs are ~300k triangles."""

from __future__ import annotations

from pathlib import Path

import pytest

from knee_viz.config import REPO_ROOT
from knee_viz.core.bones import load_knee

MODEL_DEMO = REPO_ROOT / "data_for_gui" / "Model_demo"
# P6_pre registers correctly (0.3 mm bone apposition), so it is the fixture to
# use whenever a test needs an anatomically plausible pose.
MODEL_P6 = REPO_ROOT / "data_for_gui" / "P6_pre"
MOTION_CSV = REPO_ROOT / "data_20260729_175019.csv"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"missing test asset: {path}")
    return path


@pytest.fixture(scope="session")
def model_dir() -> Path:
    return _require(MODEL_DEMO)


@pytest.fixture(scope="session")
def knee(model_dir):
    return load_knee(model_dir)


@pytest.fixture(scope="session")
def femur(knee):
    return knee[0]


@pytest.fixture(scope="session")
def tibia(knee):
    return knee[1]


@pytest.fixture(scope="session")
def motion_csv() -> Path:
    return _require(MOTION_CSV)
