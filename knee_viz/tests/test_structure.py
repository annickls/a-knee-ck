"""Structural guards that keep the layering and the theming honest."""

from __future__ import annotations

import ast
import re
import subprocess
import sys

from knee_viz.config import REPO_ROOT

PACKAGE = REPO_ROOT / "knee_viz"

# Hex colours, rgb()/rgba() and QColor built from numbers. Named-attribute
# lookups such as palette.qcolor("bone") are the sanctioned form.
_COLOUR_PATTERNS = (
    re.compile(r"#[0-9A-Fa-f]{3,8}\b"),
    re.compile(r"\brgba?\s*\("),
    re.compile(r"QColor\s*\(\s*[\d.]"),
)


def test_core_and_data_import_no_qt():
    """The measurement chain must stay runnable headless.

    Run in a subprocess so an earlier UI test in the same session cannot mask a
    real dependency by having already imported Qt.
    """
    code = (
        "import knee_viz.core.pipeline, knee_viz.core.joint_gap, knee_viz.core.modes;"
        "import knee_viz.data;"
        "import sys;"
        "bad=[m for m in sys.modules if m.split('.')[0] in ('PyQt5','PyQt6','PySide2','PySide6','pyqtgraph','matplotlib')];"
        "print(','.join(sorted(bad)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"GUI libraries leaked into core/data: {result.stdout}"


def test_colours_live_only_in_theme():
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "theme.py" or "tests" in path.parts:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for pattern in _COLOUR_PATTERNS:
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert not offenders, "colour literals outside theme.py:\n" + "\n".join(offenders)


def test_debug_flag_is_only_forwarded_never_branched_on():
    """`if config.debug` scattered through the widgets is what this replaced.

    Parsed rather than grepped so prose in a docstring cannot pass or fail it.
    The flag may be *read* where it is handed on, but the only place that acts
    on it is DebugLayer, which stores it as ``self.enabled``.
    """
    readers: dict[str, list[int]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "debug"
                and isinstance(node.value, ast.Name)
                and node.value.id in {"cfg", "config"}
            ):
                readers.setdefault(path.relative_to(PACKAGE).as_posix(), []).append(node.lineno)

    assert set(readers) <= {"ui/scene3d.py", "ui/main_window.py"}, readers


def test_debug_layer_owns_the_only_enabled_branch():
    source = (PACKAGE / "ui" / "debug_layer.py").read_text()
    assert "if not self.enabled:" in source, "DebugLayer must short-circuit when disabled"
