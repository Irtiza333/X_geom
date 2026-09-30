"""
Non-interactive Bezier plotter (no sliders).

- Plots a continuous curve that goes through the 15 Bezier sample points
- Reads optimized Bezier parameters from `chord_opt_bez_final.txt`
  - first 6 values: pitch params  [p1y, y4, p4x, d1, d2, p7y]
  - remaining values: chord params [p1y, p4x, d1, y4, w56]
- Renders the Bezier approximation as 15 hollow blue sample points
- Renders the control/anchor points as filled red dots
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

FONT_FAMILY = "Times New Roman"

# Make everything bigger (for reports / papers)
FIGSIZE = (22, 16)
TITLE_FONTSIZE = 60
LABEL_FONTSIZE = 60
TICK_FONTSIZE = 52
LEGEND_FONTSIZE = 52

CURVE_LINEWIDTH = 5.0
GRID_LINEWIDTH = 2.0
AXIS_MARGIN_X = 0.01
AXIS_MARGIN_Y = 0.03

# Reduce outer whitespace while keeping labels visible
SUBPLOT_ADJUST = dict(left=0.11, right=0.99, bottom=0.13, top=0.92)

BEZIER_MARKERSIZE = 32
BEZIER_MARKEREDGEWIDTH = 6.0
CONTROL_MARKERSIZE = 28
CONTROL_LABEL_FONTSIZE = int(LABEL_FONTSIZE * 0.95)
CONTROL_LABEL_BOX_ALPHA = 0.70

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": LABEL_FONTSIZE,
    }
)

try:
    # Optional (smooth curve through the 15 points)
    from scipy.interpolate import PchipInterpolator, make_interp_spline  # type: ignore
except Exception:  # pragma: no cover
    PchipInterpolator = None  # type: ignore
    make_interp_spline = None  # type: ignore


# Original functions (copied from `run_interactive_bezier.py`)
Pitch = lambda x: 1 * (
    19344.5071 * x**12
    + -114044.8587 * x**11
    + 280789.2801 * x**10
    + -357377.6146 * x**9
    + 207947.2705 * x**8
    + 43330.8173 * x**7
    + -173099.4797 * x**6
    + 143116.5772 * x**5
    + -65570.9523 * x**4
    + 18410.2055 * x**3
    + -3128.7530 * x**2
    + 294.0671 * x
    + -10.4419
)

ChordLength = lambda x: 1 * (
    -143202.4761 * x**12
    + 978274.9902 * x**11
    + -2992184.0323 * x**10
    + 5408923.9625 * x**9
    + -6424276.8851 * x**8
    + 5271614.6993 * x**7
    + -3058632.5267 * x**6
    + 1261908.7720 * x**5
    + -366745.1423 * x**4
    + 73096.4455 * x**3
    + -9470.1908 * x**2
    + 716.1619 * x
    + -23.7157
)


def eval_rational_bezier(CP: np.ndarray, w: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate one 4-point rational cubic Bézier.

    CP: (2,4) control point array [[x1..x4],[y1..y4]]
    w:  (4,) weights
    u:  (N,) parameter in [0,1]
    """
    B0 = (1 - u) ** 3
    B1 = 3 * (1 - u) ** 2 * u
    B2 = 3 * (1 - u) * u**2
    B3 = u**3

    num = (
        CP[:, 0:1] * (w[0] * B0)
        + CP[:, 1:2] * (w[1] * B1)
        + CP[:, 2:3] * (w[2] * B2)
        + CP[:, 3:4] * (w[3] * B3)
    )
    den = w[0] * B0 + w[1] * B1 + w[2] * B2 + w[3] * B3

    X = num[0, :] / den
    Y = num[1, :] / den
    return X, Y


def compute_chord_w23(R1: float, p4x: float, d1: float) -> float:
    """
    Compute chord w23 from p4x and d1 using the relationship in `interactive_bezier_control.py`.
    """
    a = 0.995
    x2 = p4x - d1
    r_exact = x2 + 0.78 * (p4x - x2)
    denom = (1 - a) * d1
    if abs(denom) < 1e-9:
        return 1.0
    s3 = (r_exact - a * (p4x - d1) - (1 - a) * R1) / denom
    s = np.cbrt(s3)
    denom2 = 3.0 * s * (1.0 + s)
    if abs(denom2) < 1e-9 or not np.isfinite(denom2):
        return 1.0
    W = (a / (1 - a) - s3) / denom2
    if not np.isfinite(W) or W <= 0:
        return 1.0
    return float(W)


def read_bezier_params(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    raw = np.fromstring(txt, sep=" ")
    raw = raw[np.isfinite(raw)]
    if raw.size < 11:
        raise ValueError(f"Expected at least 11 numeric values in {path}, got {raw.size}.")
    pitch_params = raw[:6].astype(float)
    chord_params = raw[6:].astype(float)
    if chord_params.size < 5:
        raise ValueError(f"Expected at least 5 chord values after the first 6 pitch values, got {chord_params.size}.")
    chord_params = chord_params[:5]
    return pitch_params, chord_params


def sample_stitched_curve(
    seg1: tuple[np.ndarray, np.ndarray],
    seg2: tuple[np.ndarray, np.ndarray],
    n_points: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    r1, y1 = seg1
    r2, y2 = seg2
    r = np.concatenate([r1, r2[1:]])
    y = np.concatenate([y1, y2[1:]])
    idx = np.linspace(0, r.size - 1, n_points).astype(int)
    return r[idx], y[idx]


def build_chord_bezier_points(R1: float, R7: float, chord_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # chord_params: [p1y, p4x, d1, y4, w56]
    p1y, p4x, d1, y4, w56 = chord_params.tolist()

    p1x = R1
    if p4x <= p1x + d1:
        d1 = max(0.01, p4x - p1x - 0.01)

    p2x = p4x - d1
    P1 = np.array([p1x, p1y])
    P2 = np.array([p2x, y4])
    P3 = P2
    P4 = np.array([p4x, y4])
    P5 = np.array([R7, y4])
    P6 = P5
    P7 = np.array([R7, float(ChordLength(R7))])

    w23 = compute_chord_w23(R1, p4x, d1)
    w_seg1 = np.array([1.0, w23, w23, 1.0], dtype=float)
    w_seg2 = np.array([1.0, w56, w56, 1.0], dtype=float)

    u = np.linspace(0.0, 1.0, 600)
    seg1 = eval_rational_bezier(np.column_stack([P1, P2, P3, P4]), w_seg1, u)
    seg2 = eval_rational_bezier(np.column_stack([P4, P5, P6, P7]), w_seg2, u)
    r_pts, y_pts = sample_stitched_curve(seg1, seg2, n_points=15)

    control_points = np.vstack([P1, P2, P3, P4, P5, P6, P7])
    return r_pts, y_pts, control_points


def build_pitch_bezier_points(R1: float, R7: float, pitch_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # pitch_params: [p1y, y4, p4x, d1, d2, p7y]
    p1y, y4, p4x, d1, d2, p7y = pitch_params.tolist()

    if p4x <= R1 + d1:
        d1 = max(0.01, p4x - R1 - 0.01)
    if p4x + d2 >= R7:
        d2 = max(0.01, R7 - p4x - 0.01)

    p2x = p4x - d1
    p5x = p4x + d2

    P1 = np.array([R1, p1y])
    P2 = np.array([p2x, y4])
    P3 = P2
    P4 = np.array([p4x, y4])
    P5 = np.array([p5x, y4])
    P6 = P5
    P7 = np.array([R7, p7y])

    w_seg1 = np.ones(4, dtype=float)
    w_seg2 = np.ones(4, dtype=float)

    u = np.linspace(0.0, 1.0, 600)
    seg1 = eval_rational_bezier(np.column_stack([P1, P2, P3, P4]), w_seg1, u)
    seg2 = eval_rational_bezier(np.column_stack([P4, P5, P6, P7]), w_seg2, u)
    r_pts, y_pts = sample_stitched_curve(seg1, seg2, n_points=15)

    control_points = np.vstack([P1, P2, P3, P4, P5, P6, P7])
    return r_pts, y_pts, control_points


def plot_single(
    title: str,
    y_label: str,
    r_pts: np.ndarray,
    y_pts: np.ndarray,
    control_points: np.ndarray,
    spline_order: int,
    curve_method: str,
    control_prefix: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Continuous curve through the 15 points (replaces original polynomial curve)
    r_pts = np.asarray(r_pts, dtype=float)
    y_pts = np.asarray(y_pts, dtype=float)
    order = np.argsort(r_pts)
    r_sorted = r_pts[order]
    y_sorted = y_pts[order]

    r_unique, unique_idx = np.unique(r_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]

    r_dense = np.linspace(float(r_unique[0]), float(r_unique[-1]), 400)
    if curve_method.lower() == "pchip" and PchipInterpolator is not None and r_unique.size >= 2:
        f = PchipInterpolator(r_unique, y_unique)
        y_dense = f(r_dense)
    elif make_interp_spline is not None and r_unique.size >= (spline_order + 1):
        spline = make_interp_spline(r_unique, y_unique, k=spline_order)
        y_dense = spline(r_dense)
    else:
        y_dense = np.interp(r_dense, r_unique, y_unique)

    ax.plot(r_dense, y_dense, "k--", linewidth=CURVE_LINEWIDTH, label="Original Curve")

    # 15 sampled points along the stitched Bezier curve (hollow blue circles)
    ax.plot(
        r_pts,
        y_pts,
        linestyle="None",
        marker="o",
        markersize=BEZIER_MARKERSIZE,
        markerfacecolor="none",
        markeredgecolor="b",
        markeredgewidth=BEZIER_MARKEREDGEWIDTH,
        label="Bezier Curve Fit",
    )

    # Filled red anchor/control points (bigger)
    ax.plot(
        control_points[:, 0],
        control_points[:, 1],
        linestyle="None",
        marker="o",
        markersize=CONTROL_MARKERSIZE,
        markerfacecolor="r",
        markeredgecolor="r",
        label="Control points",
    )

    # Control point labels.
    # Double (coincident) points are labeled as: P_{c1,2} and P_{c4,5} (and similarly for pitch).
    control_points = np.asarray(control_points, dtype=float)
    # Offsets in "points" (scaled to match big fonts)
    base_dx = int(CONTROL_LABEL_FONTSIZE * 1.6)
    base_dy = int(CONTROL_LABEL_FONTSIZE * 1.0)

    label_groups: list[tuple[int, ...]] = [(0,), (1, 2), (3,), (4, 5), (6,)]
    if control_prefix.lower() == "p":
        label_offsets: dict[tuple[int, ...], tuple[int, int]] = {
            (0,): (int(0.2 * base_dx), int(-1.0 * base_dy)),         # down-right
            (1, 2): (-int(0.05 * base_dx), int(-1.0 * base_dy)),     # between Pp1 and Pp2
            (3,): (int(0.4 * base_dx), int(-0.8 * base_dy)),         # down-right
            (4, 5): (int(0.5 * base_dx), int(-0.5 * base_dy)),       # between Pp4 and Pp5
            (6,): (-int(1.2 * base_dx), int(0.4 * base_dy)),         # left / slightly up
        }
    else:
        label_offsets = {
            (0,): (int(0.7 * base_dx), int(-1.2 * base_dy)),         # down-right
            (1, 2): (-0.5*base_dx, int(-0.5 * base_dy)),                 # between Pc1 and Pc2
            (3,): (int(0.5 * base_dx), int(-0.8 * base_dy)),         # down-right
            (4, 5): (-int(0.7 * base_dx), int(-0.8 * base_dy)),      # between Pc4 and Pc5
            (6,): (-int(1.2 * base_dx), int(0.4 * base_dy)),         # left / slightly up
        }

    # Nudge labels back inside if near axes edges
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr = max(1e-12, x1 - x0)
    yr = max(1e-12, y1 - y0)
    for group in label_groups:
        i0 = group[0]
        x, y = control_points[i0]
        dx, dy = label_offsets.get(group, (8, 8))
        x_norm = (float(x) - x0) / xr
        y_norm = (float(y) - y0) / yr
        if x_norm > 0.92:
            dx = -abs(dx)
        elif x_norm < 0.08:
            dx = abs(dx)
        if y_norm > 0.92:
            dy = -abs(dy)
        elif y_norm < 0.08:
            dy = abs(dy)

        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        if len(group) == 1:
            label = f"P$_{{{control_prefix}{group[0]}}}$"
        else:
            label = f"P$_{{{control_prefix}{group[0]},{group[1]}}}$"
        ax.annotate(
            label,
            xy=(float(x), float(y)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontname=FONT_FAMILY,
            fontsize=CONTROL_LABEL_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=CONTROL_LABEL_BOX_ALPHA, pad=0.2),
            clip_on=False,
        )

    ax.set_title(title, fontname=FONT_FAMILY, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Normalized Radius [-]", fontname=FONT_FAMILY, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontname=FONT_FAMILY, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.margins(x=AXIS_MARGIN_X, y=AXIS_MARGIN_Y)
    ax.grid(True, linewidth=GRID_LINEWIDTH)
    ax.legend(prop={"family": FONT_FAMILY, "size": LEGEND_FONTSIZE})
    fig.subplots_adjust(**SUBPLOT_ADJUST)


def main() -> None:
    R_values = np.linspace(0.15, 1.0, 100)
    R1, R7 = float(R_values[0]), float(R_values[-1])

    pitch_params, chord_params = read_bezier_params("chord_opt_bez_final.txt")

    # Build 15-point Bezier samples + anchors
    r_chord, c_pts, chord_anchors = build_chord_bezier_points(R1, R7, chord_params)
    r_pitch, p_pts, pitch_anchors = build_pitch_bezier_points(R1, R7, pitch_params)

    plot_single(
        title="Chord-Length Over Diameter (original vs Bezier points)",
        y_label="Chord-Length Over Diameter",
        r_pts=r_chord,
        y_pts=c_pts,
        control_points=chord_anchors,
        spline_order=3,
        curve_method="pchip",
        control_prefix="c",
    )
    plot_single(
        title="Pitch Over Diameter (original vs Bezier points)",
        y_label="Pitch Over Diameter",
        r_pts=r_pitch,
        y_pts=p_pts,
        control_points=pitch_anchors,
        spline_order=4,
        curve_method="spline",
        control_prefix="p",
    )

    plt.show()


if __name__ == "__main__":
    main()

