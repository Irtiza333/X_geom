"""
Compare original vs optimized Bezier-fitted pitch/chord curves.

- "Original" curve comes from `chord_opt_bez_final.txt` (same format used by `run_bezier_points_from_file.py`)
- "Optimized" blade comes from `Surrogate/Prop_hull/opt_blade.txt` which stores 11 control-point (con-point) values:
    pitch: [p1y, y4, p4x, p2x, p5x, p7y]
    chord: [p1y, p4x, p2x, y4, w56]
  These are converted into Bezier "var" parameters:
    pitch_vars: [p1y, y4, p4x, d1, d2, p7y]  where d1=p4x-p2x, d2=p5x-p4x
    chord_vars: [p1y, p4x, d1, y4, w56]      where d1=p4x-p2x

The plotted curves are the same "continuous curve through the 15 Bezier sample points"
logic used in `run_bezier_points_from_file.py`:
  - chord: PCHIP through 15 points
  - pitch: 4th-order spline through 15 points

No points/control-points are shown; only the two curves on each plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import PchipInterpolator, make_interp_spline  # type: ignore
except Exception:  # pragma: no cover
    PchipInterpolator = None  # type: ignore
    make_interp_spline = None  # type: ignore


# ---------- Styling (match the "big" plots used elsewhere) ----------

FONT_FAMILY = "Times New Roman"
FIGSIZE = (22, 16)
TITLE_FONTSIZE = 60
LABEL_FONTSIZE = 60
TICK_FONTSIZE = 52
LEGEND_FONTSIZE = 52
CURVE_LINEWIDTH = 6.0
GRID_LINEWIDTH = 2.0
SUBPLOT_ADJUST = dict(left=0.11, right=0.99, bottom=0.13, top=0.92)

plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": LABEL_FONTSIZE})


# ---------- Original polynomials (used only to set chord tip point P7y) ----------

Pitch_poly = lambda x: 1 * (
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

Chord_poly = lambda x: 1 * (
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


# ---------- Rational cubic Bezier helpers ----------

def eval_rational_bezier(CP: np.ndarray, w: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one 4-point rational cubic Bézier."""
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
    """Compute chord w23 from p4x and d1 (same relationship as interactive code)."""
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


def sample_stitched_curve(seg1: tuple[np.ndarray, np.ndarray], seg2: tuple[np.ndarray, np.ndarray], n_points: int = 15) -> tuple[np.ndarray, np.ndarray]:
    r1, y1 = seg1
    r2, y2 = seg2
    r = np.concatenate([r1, r2[1:]])
    y = np.concatenate([y1, y2[1:]])
    idx = np.linspace(0, r.size - 1, int(n_points)).astype(int)
    return r[idx], y[idx]


def build_chord_15pts(R1: float, R7: float, chord_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 15 (r,y) points for the chord stitched curve."""
    p1y, p4x, d1, y4, w56 = chord_params.astype(float).tolist()

    p1x = float(R1)
    if p4x <= p1x + d1:
        d1 = max(0.01, p4x - p1x - 0.01)

    p2x = p4x - d1
    P1 = np.array([p1x, p1y])
    P2 = np.array([p2x, y4])
    P3 = P2
    P4 = np.array([p4x, y4])
    P5 = np.array([float(R7), y4])
    P6 = P5
    P7 = np.array([float(R7), float(Chord_poly(R7))])

    w23 = compute_chord_w23(R1, p4x, d1)
    w_seg1 = np.array([1.0, w23, w23, 1.0], dtype=float)
    w_seg2 = np.array([1.0, w56, w56, 1.0], dtype=float)

    u = np.linspace(0.0, 1.0, 600)
    seg1 = eval_rational_bezier(np.column_stack([P1, P2, P3, P4]), w_seg1, u)
    seg2 = eval_rational_bezier(np.column_stack([P4, P5, P6, P7]), w_seg2, u)
    return sample_stitched_curve(seg1, seg2, n_points=15)


def build_pitch_15pts(R1: float, R7: float, pitch_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 15 (r,y) points for the pitch stitched curve."""
    p1y, y4, p4x, d1, d2, p7y = pitch_params.astype(float).tolist()

    if p4x <= R1 + d1:
        d1 = max(0.01, p4x - R1 - 0.01)
    if p4x + d2 >= R7:
        d2 = max(0.01, R7 - p4x - 0.01)

    p2x = p4x - d1
    p5x = p4x + d2

    P1 = np.array([float(R1), p1y])
    P2 = np.array([p2x, y4])
    P3 = P2
    P4 = np.array([p4x, y4])
    P5 = np.array([p5x, y4])
    P6 = P5
    P7 = np.array([float(R7), p7y])

    w_seg1 = np.ones(4, dtype=float)
    w_seg2 = np.ones(4, dtype=float)

    u = np.linspace(0.0, 1.0, 600)
    seg1 = eval_rational_bezier(np.column_stack([P1, P2, P3, P4]), w_seg1, u)
    seg2 = eval_rational_bezier(np.column_stack([P4, P5, P6, P7]), w_seg2, u)
    return sample_stitched_curve(seg1, seg2, n_points=15)


# ---------- Curve-through-points (same logic as run_bezier_points_from_file.py) ----------

def curve_through_points(
    r_pts: np.ndarray,
    y_pts: np.ndarray,
    *,
    method: str,
    spline_order: int,
    spline_bc_type=None,
    n_dense: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    r_pts = np.asarray(r_pts, dtype=float)
    y_pts = np.asarray(y_pts, dtype=float)

    order = np.argsort(r_pts)
    r_sorted = r_pts[order]
    y_sorted = y_pts[order]

    r_unique, unique_idx = np.unique(r_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]

    r_dense = np.linspace(float(r_unique[0]), float(r_unique[-1]), int(n_dense))
    m = method.lower().strip()
    if m == "pchip" and PchipInterpolator is not None and r_unique.size >= 2:
        f = PchipInterpolator(r_unique, y_unique)
        y_dense = f(r_dense)
    elif m == "spline" and make_interp_spline is not None and r_unique.size >= (spline_order + 1):
        spline = make_interp_spline(r_unique, y_unique, k=int(spline_order), bc_type=spline_bc_type)
        y_dense = spline(r_dense)
    else:
        # Fallback: linear interpolation
        y_dense = np.interp(r_dense, r_unique, y_unique)
    return r_dense, np.asarray(y_dense, dtype=float)


# ---------- I/O + conversions ----------

def read_bezier_params(path: Path) -> tuple[np.ndarray, np.ndarray]:
    txt = path.read_text(encoding="utf-8")
    raw = np.fromstring(txt, sep=" ")
    raw = raw[np.isfinite(raw)]
    if raw.size < 11:
        raise ValueError(f"Expected at least 11 numeric values in {path}, got {raw.size}.")
    pitch_params = raw[:6].astype(float)
    chord_params = raw[6:][:5].astype(float)
    return pitch_params, chord_params


def read_opt_blade_conpoints(path: Path) -> np.ndarray:
    raw = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")
    raw = raw[np.isfinite(raw)]
    if raw.size != 11:
        raise ValueError(f"Expected exactly 11 values in {path}, got {raw.size}.")
    return raw.astype(float)


def conpoints_to_bezier_vars(x_con: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 11 con-points -> (pitch_vars, chord_vars) in the format expected by Bezier plotting."""
    x = np.asarray(x_con, dtype=float).ravel()
    p1y, y4, p4x, p2x, p5x, p7y = x[:6]
    c_p1y, c_p4x, c_p2x, c_y4, c_w56 = x[6:]

    pitch_vars = np.array([p1y, y4, p4x, (p4x - p2x), (p5x - p4x), p7y], dtype=float)
    chord_vars = np.array([c_p1y, c_p4x, (c_p4x - c_p2x), c_y4, c_w56], dtype=float)
    return pitch_vars, chord_vars


# ---------- Plotting ----------

def plot_comparison(
    *,
    title: str,
    y_label: str,
    r_orig: np.ndarray,
    y_orig: np.ndarray,
    r_opt: np.ndarray,
    y_opt: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(r_orig, y_orig, "k--", linewidth=CURVE_LINEWIDTH, label="Original")
    # Optimized shown as red circles (no solid line)
    ax.plot(
        r_opt,
        y_opt,
        linestyle="None",
        marker="o",
        markersize=14,
        markerfacecolor="none",
        markeredgecolor="r",
        markeredgewidth=3.0,
        markevery=8,
        label="Optimized",
    )

    ax.set_title(title, fontname=FONT_FAMILY, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Normalized Radius [-]", fontname=FONT_FAMILY, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontname=FONT_FAMILY, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=0.35)
    ax.legend(prop={"family": FONT_FAMILY, "size": LEGEND_FONTSIZE})
    fig.subplots_adjust(**SUBPLOT_ADJUST)


def main() -> None:
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent

    original_param_path = this_dir / "chord_opt_bez_final.txt"
    opt_blade_path = repo_root / "Surrogate" / "Prop_hull" / "opt_blade.txt"

    if not original_param_path.exists():
        raise FileNotFoundError(f"Missing {original_param_path}")
    if not opt_blade_path.exists():
        raise FileNotFoundError(f"Missing {opt_blade_path}")

    # Same radius range used by `run_bezier_points_from_file.py`
    R1 = 0.15
    R7 = 1.0

    # Original (from chord_opt_bez_final)
    pitch_params_orig, chord_params_orig = read_bezier_params(original_param_path)
    r15_p_o, y15_p_o = build_pitch_15pts(R1, R7, pitch_params_orig)
    r15_c_o, y15_c_o = build_chord_15pts(R1, R7, chord_params_orig)

    r_p_o, y_p_o = curve_through_points(r15_p_o, y15_p_o, method="spline", spline_order=4)
    r_c_o, y_c_o = curve_through_points(r15_c_o, y15_c_o, method="pchip", spline_order=3)

    # Optimized (from opt_blade con-points)
    x_opt_con = read_opt_blade_conpoints(opt_blade_path)
    pitch_params_opt, chord_params_opt = conpoints_to_bezier_vars(x_opt_con)

    print("Optimized blade (con-points) from opt_blade.txt:")
    print("  ", x_opt_con)
    print("Converted to Bezier vars:")
    print("  pitch_vars [p1y, y4, p4x, d1, d2, p7y] =", pitch_params_opt)
    print("  chord_vars [p1y, p4x, d1, y4, w56]     =", chord_params_opt)

    r15_p_x, y15_p_x = build_pitch_15pts(R1, R7, pitch_params_opt)
    r15_c_x, y15_c_x = build_chord_15pts(R1, R7, chord_params_opt)

    r_p_x, y_p_x = curve_through_points(r15_p_x, y15_p_x, method="spline", spline_order=4)
    r_c_x, y_c_x = curve_through_points(r15_c_x, y15_c_x, method="pchip", spline_order=3)

    plot_comparison(
        title="Pitch Over Diameter (original fit vs optimized blade)",
        y_label="Pitch Over Diameter",
        r_orig=r_p_o,
        y_orig=y_p_o,
        r_opt=r_p_x,
        y_opt=y_p_x,
    )
    plot_comparison(
        title="Chord Over Diameter (original fit vs optimized blade)",
        y_label="Chord Over Diameter",
        r_orig=r_c_o,
        y_orig=y_c_o,
        r_opt=r_c_x,
        y_opt=y_c_x,
    )

    plt.show()


if __name__ == "__main__":
    main()

