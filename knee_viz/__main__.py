"""Entry point: ``python -m knee_viz [--source replay --csv ... --debug]``."""

from __future__ import annotations

import logging
import os
import sys

# Must precede any pyqtgraph import so it binds to PyQt6 rather than walking its
# own preference order.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

from PyQt6 import QtWidgets  # noqa: E402

from knee_viz.config import AppConfig  # noqa: E402
from knee_viz.theme import stylesheet  # noqa: E402
from knee_viz.ui.main_window import MainWindow  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = AppConfig.from_cli(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyleSheet(stylesheet())

    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
