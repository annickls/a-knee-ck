"""The single source of colour for the whole application.

Every value here was sampled from ``cropped_GUI.png``. No other module in
``knee_viz`` may contain a colour literal -- ``tests/test_no_colour_literals.py``
enforces that -- so restyling the app means editing this file and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PyQt6 import QtGui


@dataclass(frozen=True)
class Palette:
    """Named colours as hex strings."""

    # Surfaces
    page_bg: str = "#2D2D2D"
    panel_bg: str = "#323232"
    panel_raised: str = "#434343"
    separator: str = "#1F1F1F"

    # Text
    text: str = "#F5F1F0"
    text_muted: str = "#9A9A9A"
    text_on_pill: str = "#FFFFFF"

    # 3D scene
    scene_bg: str = "#2D2D2D"
    bone: str = "#E3C6B6"
    bone_shadow: str = "#9D897E"
    accent: str = "#538882"  # landmarks and anatomical axes
    guide_line: str = "#FFFFFF"

    # Overlay pills
    pill_bg: str = "#585858"
    pill_border: str = "#707070"

    # 2D plot
    plot_bg: str = "#2D2D2D"
    plot_band_a: str = "#484848"
    plot_band_b: str = "#3E3E3E"
    plot_neutral_band: str = "#466270"
    plot_grid: str = "#707070"
    plot_axis_text: str = "#F5F1F0"

    # Plot series
    history: str = "#707070"  # every sample, unfiltered
    valid: str = "#90BF21"  # samples passing the neutrality gate
    current_ok: str = "#FFFFFF"  # live sample, other DOF near neutral
    current_out_of_range: str = "#D9534F"  # live sample, gated out

    # Force bar (placeholder, not yet driven)
    force_low: str = "#90BF21"
    force_high: str = "#D9534F"
    force_track: str = "#434343"

    # Debug-only visuals
    debug_axis_x: str = "#E06C5A"
    debug_axis_y: str = "#90BF21"
    debug_axis_z: str = "#4FA3D1"
    debug_ray: str = "#E8C547"
    debug_hit: str = "#E06C5A"

    def qcolor(self, name: str, alpha: float = 1.0) -> QtGui.QColor:
        """A ``QColor`` for widget and QPainter use."""
        colour = QtGui.QColor(getattr(self, name))
        colour.setAlphaF(alpha)
        return colour

    def gl(self, name: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
        """An RGBA float tuple for pyqtgraph.opengl items."""
        colour = QtGui.QColor(getattr(self, name))
        return (colour.redF(), colour.greenF(), colour.blueF(), alpha)

    def pen(self, name: str, width: float = 1.0, **kwargs) -> pg.functions.mkPen:
        return pg.mkPen(color=getattr(self, name), width=width, **kwargs)

    def brush(self, name: str, alpha: float = 1.0) -> pg.functions.mkBrush:
        return pg.mkBrush(self.qcolor(name, alpha))


PALETTE = Palette()


def stylesheet(palette: Palette = PALETTE) -> str:
    """The whole-application QSS, applied once to the QApplication."""
    return f"""
    QWidget {{
        background-color: {palette.page_bg};
        color: {palette.text};
        font-family: "DejaVu Sans", "Segoe UI", Arial, sans-serif;
        font-size: 12px;
    }}
    QLabel#SectionTitle {{
        color: {palette.text_muted};
        font-size: 11px;
        letter-spacing: 1px;
    }}
    QPushButton {{
        background-color: {palette.panel_raised};
        color: {palette.text};
        border: none;
        border-radius: 6px;
        padding: 8px 12px;
    }}
    QPushButton:hover {{ background-color: {palette.pill_border}; }}
    QPushButton:pressed {{ background-color: {palette.panel_bg}; }}
    QPushButton:checked {{
        background-color: {palette.accent};
        color: {palette.text_on_pill};
        font-weight: bold;
    }}
    QPushButton:disabled {{ color: {palette.text_muted}; }}
    QFrame#Panel {{
        background-color: {palette.panel_bg};
        border-radius: 8px;
    }}
    QSplitter::handle {{ background-color: {palette.separator}; }}
    """
