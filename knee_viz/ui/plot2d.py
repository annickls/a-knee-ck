"""Live flexion-versus-quantity plot.

Flexion runs down the vertical axis (extension at the top, as in the reference)
and the selected quantity runs left/right about a neutral centre band.

Replaces a hand-rolled numpy rasteriser that drew every point and grid line with
per-pixel Python loops. Here pyqtgraph owns the drawing and this class only
maintains a ring buffer and calls ``setData``.
"""

from __future__ import annotations

import enum

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

from knee_viz.core.modes import MODES, SIGNALS, PlotMode, sample_from_state
from knee_viz.core.state import KneeState
from knee_viz.theme import PALETTE, Palette

FLEXION_MIN_DEG = -10.0
FLEXION_MAX_DEG = 120.0

# Half-width of the highlighted centre corridor, in the plotted unit. Fixed
# rather than proportional so it means the same thing in every mode: within
# 3 mm / 3 degrees of symmetric.
NEUTRAL_BAND = 3.0
OUTER_BANDS = 3  # bands either side of the corridor, filling the rest of the axis


class RenderStyle(enum.Enum):
    POINTS = "points"
    BARS = "bars"


class MirroredAxis(pg.AxisItem):
    """Ticks labelled by magnitude, with direction carried by the end captions.

    The x axis is a mirrored pair -- medial to the left of zero, lateral to the
    right -- so a tick at -15 means "15 mm, medially", not "minus 15 mm". The
    same holds for varus/valgus and internal/external rotation, which are named
    directions rather than signed quantities.
    """

    def tickStrings(self, values, scale, spacing):
        return [f"{abs(v) * scale:.0f}" for v in values]


class KneePlot2D(QtWidgets.QWidget):
    """Scatter of past and current samples against flexion angle."""

    def __init__(
        self,
        max_points: int = 10_000,
        palette: Palette = PALETTE,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette
        self._mode = PlotMode.JOINT_GAPS
        self._style = RenderStyle.POINTS

        self._buffer = {name: np.zeros(max_points, dtype=np.float32) for name in SIGNALS}
        self._max_points = max_points
        self._write = 0
        self._count = 0

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(
            background=palette.qcolor("plot_bg"),
            axisItems={"bottom": MirroredAxis(orientation="bottom"),
                       "top": MirroredAxis(orientation="top")},
        )
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()

        item = self.plot.getPlotItem()
        item.invertY(True)  # extension at the top, flexion downwards
        item.showGrid(x=False, y=True, alpha=0.25)
        item.setYRange(FLEXION_MIN_DEG, FLEXION_MAX_DEG, padding=0)
        for side in ("left", "bottom", "top", "right"):
            axis = item.getAxis(side)
            axis.setPen(palette.pen("plot_grid"))
            axis.setTextPen(palette.pen("plot_axis_text"))
        item.showAxis("right")
        item.showAxis("top")
        item.getAxis("left").setLabel("flexion", units="°")

        self._bands: list[pg.LinearRegionItem] = []
        self._build_bands()

        self._history = self._scatter("history", size=4, alpha=0.55)
        self._valid = self._scatter("valid", size=7)
        self._current = self._scatter("current_ok", size=13)
        self._bars = pg.BarGraphItem(x0=[], y=[], height=0.8, width=[], brush=palette.brush("valid"))
        self._bars.setVisible(False)
        self.plot.addItem(self._bars)

        self._zero_line = pg.InfiniteLine(pos=0, angle=90, pen=palette.pen("guide_line", 1.0))
        self.plot.addItem(self._zero_line)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        self.set_mode(self._mode)

    # ---------------------------------------------------------------- building

    def _scatter(self, colour: str, size: int, alpha: float = 1.0) -> pg.ScatterPlotItem:
        item = pg.ScatterPlotItem(
            size=size,
            pen=None,
            brush=self._palette.brush(colour, alpha),
            pxMode=True,
        )
        self.plot.addItem(item)
        return item

    @staticmethod
    def _band_edges(x_limit: float) -> tuple[float, ...]:
        """Corridor edge at a fixed width, then equal bands out to the limit."""
        step = (x_limit - NEUTRAL_BAND) / OUTER_BANDS
        return (0.0, NEUTRAL_BAND) + tuple(NEUTRAL_BAND + step * (i + 1) for i in range(OUTER_BANDS))

    def _build_bands(self) -> None:
        """Alternating vertical bands, with the neutral corridor highlighted."""
        edges = self._band_edges(MODES[self._mode].x_limit)
        for index in range(len(edges) - 1):
            for sign in (-1.0, 1.0):
                low, high = sorted((sign * edges[index], sign * edges[index + 1]))
                colour = "plot_neutral_band" if index == 0 else (
                    "plot_band_a" if index % 2 else "plot_band_b"
                )
                band = pg.LinearRegionItem(
                    values=(low, high),
                    orientation="vertical",
                    brush=self._palette.brush(colour, 0.55),
                    pen=pg.mkPen(None),
                    movable=False,
                )
                band.setZValue(-100)
                for line in band.lines:
                    line.setPen(pg.mkPen(None))
                self.plot.addItem(band)
                self._bands.append(band)

    # ------------------------------------------------------------------- state

    @property
    def mode(self) -> PlotMode:
        return self._mode

    @property
    def sample_count(self) -> int:
        return self._count

    def set_mode(self, mode: PlotMode) -> None:
        self._mode = mode
        spec = MODES[mode]
        item = self.plot.getPlotItem()
        item.setXRange(-spec.x_limit, spec.x_limit, padding=0)
        item.setTitle(spec.title, color=self._palette.text, size="11pt")
        item.getAxis("bottom").setLabel(
            f"{spec.left_label}  ←  {spec.unit}  →  {spec.right_label}"
        )
        self._rescale_bands(spec.x_limit)
        self._redraw()

    def _rescale_bands(self, x_limit: float) -> None:
        """Keep the outermost band flush with the axis limit."""
        edges = self._band_edges(x_limit)
        placements = [(i, s) for i in range(len(edges) - 1) for s in (-1.0, 1.0)]
        for band, (index, sign) in zip(self._bands, placements):
            low, high = sorted((sign * edges[index], sign * edges[index + 1]))
            band.setRegion((low, high))

    def set_render_style(self, style: RenderStyle) -> None:
        self._style = style
        self._redraw()

    def push(self, state: KneeState) -> None:
        """Store one sample. Every signal is kept, so switching mode is free."""
        sample = sample_from_state(state)
        if not np.isfinite(sample["flexion"]):
            return
        for name, value in sample.items():
            self._buffer[name][self._write] = value
        self._write = (self._write + 1) % self._max_points
        self._count = min(self._count + 1, self._max_points)
        self._redraw()

    def clear(self) -> None:
        for array in self._buffer.values():
            array.fill(0.0)
        self._write = 0
        self._count = 0
        self._redraw()

    # ----------------------------------------------------------------- drawing

    def _ordered(self, name: str) -> np.ndarray:
        """Ring-buffer contents in insertion order."""
        array = self._buffer[name]
        if self._count < self._max_points:
            return array[: self._count]
        return np.concatenate((array[self._write :], array[: self._write]))

    def _redraw(self) -> None:
        spec = MODES[self._mode]
        if self._count == 0:
            for item in (self._history, self._valid, self._current):
                item.setData([], [])
            self._bars.setVisible(False)
            return

        flexion = self._ordered("flexion")
        near_neutral = np.abs(self._ordered(spec.gate_signal)) <= spec.gate_limit

        xs, ys, valid_mask = [], [], []
        for column, sign in zip(spec.columns, spec.signs):
            values = self._ordered(column) * sign
            finite = np.isfinite(values)
            xs.append(values[finite])
            ys.append(flexion[finite])
            valid_mask.append(near_neutral[finite])
        x = np.concatenate(xs) if xs else np.array([])
        y = np.concatenate(ys) if ys else np.array([])
        keep = np.concatenate(valid_mask) if valid_mask else np.array([], dtype=bool)

        self._history.setData(x, y)

        show_bars = self._style is RenderStyle.BARS
        self._bars.setVisible(show_bars)
        if show_bars:
            self._valid.setData([], [])
            self._bars.setOpts(x0=np.zeros(keep.sum()), y=y[keep], width=x[keep], height=0.8)
        else:
            self._valid.setData(x[keep], y[keep])

        self._draw_current(spec)

    def _draw_current(self, spec) -> None:
        """Highlight the newest sample, coloured by whether it passed the gate."""
        latest = (self._write - 1) % self._max_points
        flexion = float(self._buffer["flexion"][latest])
        in_range = abs(float(self._buffer[spec.gate_signal][latest])) <= spec.gate_limit

        points_x, points_y = [], []
        for column, sign in zip(spec.columns, spec.signs):
            value = float(self._buffer[column][latest]) * sign
            if np.isfinite(value):
                points_x.append(value)
                points_y.append(flexion)

        colour = "current_ok" if in_range else "current_out_of_range"
        self._current.setBrush(self._palette.brush(colour))
        self._current.setData(points_x, points_y)
