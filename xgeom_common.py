"""
xgeom_common.py

What the design tool's geometry adapters (xgeom_rudder.py, xgeom_blade.py)
share: the left panel's row spec, the colours and the fit of a 3D box to its
panel. No tkinter here, so the adapters run without a display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a9994"
SECTION_COLORS = ("#1f4e9a", "#2a78d6", "#3fa0c8", "#1baf7a", "#0f7a55", "#0b5e43", "#08452f")


@dataclass
class Row:
    """One line of a curve's rows: a design variable (kind 'var') or a pinned
    control-point value shown for information (kind 'pinned')."""
    kind: str
    label: str
    name: str = ""                 # slot name for 'var'
    digits: int = 4
    text: str = ""                 # for 'pinned'


def fit_box(ax, fig, box):
    """Fit a 3D box and its labels to the panel at the current angle: zoom so
    that they fill it, and shift the view so that they sit in its middle.

    ax is the 3D axes with its limits set, box its aspect (set_box_aspect).
    mplot3d puts the tick and axis labels outside the box by a fraction of its
    size (at the sides, and below the box when seen from above), so the box
    grown by that fraction, plus room for the text, has to fit."""
    if box is None:
        return
    from mpl_toolkits.mplot3d import proj3d
    dpi = fig.dpi
    ax.apply_aspect()
    panel, square, view = ax.get_position(original=True).transformed(fig.transFigure), ax.bbox, ax.viewLim
    lims = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    grow = 1.33 * dpi / (square.width + square.height) * np.diff(lims, axis=1) * [-1.0, 1.0]   # the labels' offset
    boxes = [np.array(np.meshgrid(*b)).reshape(3, -1) for b in (lims, lims + 0.5 * grow, lims + grow)]
    text, gap = 0.25 * dpi, 0.1 * dpi
    top = min(panel.y1, square.y1)                            # the title is above the square
    x0, x1, y0, y1 = panel.x0 + text, panel.x1 - text, panel.y0 + text, top - gap
    if ax.elev < 0.0:                                         # seen from below, the labels are above the box
        y0, y1 = panel.y0 + gap, top - text
    px = square.width / view.width                            # px per unit of the projection, in x and y
    zoom = 1.0
    for _ in range(3):                                        # the perspective is not quite linear in the zoom
        ax.set_box_aspect(box, zoom=zoom)
        (u, v), (_, v_half), (u_out, _) = (proj3d.proj_transform(*b, ax.get_proj())[:2] for b in boxes)
        lo, hi = (v_half.min(), v.max()) if ax.elev >= 0.0 else (v.min(), v_half.max())
        fit = min((x1 - x0) / (np.ptp(u_out) * px), (y1 - y0) / ((hi - lo) * px))
        zoom = float(np.clip(zoom * fit, 0.3, 3.0))
    ax.set_box_aspect(box, zoom=zoom)
    (u, v), (_, v_half), (u_out, _) = (proj3d.proj_transform(*b, ax.get_proj())[:2] for b in boxes)
    lo, hi = (v_half.min(), v.max()) if ax.elev >= 0.0 else (v.min(), v_half.max())
    mid_u, mid_v = 0.5 * (u_out.min() + u_out.max()), 0.5 * (lo + hi)
    fx = (0.5 * (x0 + x1) - square.x0) / square.width        # where the middle goes, as a fraction of the square
    fy = (0.5 * (y0 + y1) - square.y0) / square.height
    w, h = view.width, view.height
    view.intervalx = (mid_u - fx * w, mid_u + (1.0 - fx) * w)
    view.intervaly = (mid_v - fy * h, mid_v + (1.0 - fy) * h)


def keep_fitted(fig, ax, get_box):
    """Refit the box while the axes is turned with the mouse (left drag) and
    when the window is resized; get_box() returns the box's aspect (None
    before the first draw)."""
    def refit(_event=None):
        fit_box(ax, fig, get_box())

    def on_move(_event):
        if ax.button_pressed == 1:
            refit()
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("resize_event", refit)
