"""Qt widgets.

Importing this package pins pyqtgraph to PyQt6 before it can auto-select a
binding. pyqtgraph checks ``sys.modules`` first and only then walks its own
preference order, so this must run before any ``import pyqtgraph``.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")
