"""
plot_fig3_five_surfaces.py

Regenerates explainer_figures/fig3_five_surfaces.png.

Left  : the five DRDC wrap surfaces on the whole blade (grid rows = cut curves).
Right : a zoom on the tip showing that the cut rows wrap AROUND the edges
        rather than shrinking into rings with a cap.

Two things matter for these plots and are easy to get wrong:

1. Matplotlib's 3D axes do NOT clip artists to the axis limits.  Setting
   xlim/ylim/zlim to a zoom window therefore leaves every out-of-window curve
   floating around the box.  The data must be cropped explicitly, which is what
   _crop_rows does (points outside the window become NaN, so the polyline is
   simply broken there).

2. A 3D axes is a cube by default, so a blade that is long in one direction is
   drawn small and distorted.  set_box_aspect must be given the true data
   ranges, and the limits must be padded equally, or the geometry will not fill
   the frame.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

SURFACES = [
    ("te_strip",         "TE strip (Blade 1)",       "tab:blue"),
    ("le_strip",         "LE strip (Blade 3)",       "tab:red"),
    ("tip",              "tip surface (Blade 5)",    "tab:green"),
    ("central_pressure", "central pressure (Blade 2)", "0.25"),
    ("central_suction",  "central suction (Blade 4)",  "tab:purple"),
]


def load_grids(npz_path=None):
    """Grids from a cached npz if given, else built from the pipeline."""
    if npz_path:
        d = np.load(npz_path)
        return {k: d[k] for k in d.files}
    from x_blade_new import X_blade
    from tip_surfaces_new import build_drdc_grids, TipConfig
    pitch_con = np.array([1.025, 0.525, 0.55, 0.325, 0.325, 0.55])
    chord_con = np.array([0.25, 0.65, 0.325, 0.55, 0.2])
    blade = X_blade(pitch_con, chord_con, 99999,
                    return_blade_surface=True, write_dat=False)[-1]
    cfg = TipConfig(delta_c=0.05, n_wrap=61, n_iter=20,
                    n_outline=1500, max_curves=16)
    g = build_drdc_grids(blade, cfg, verbose=False)
    return {k: v for k, v in g.items() if isinstance(v, np.ndarray)}


def _bounds(arrays):
    pts = np.vstack([a.reshape(-1, 3) for a in arrays])
    return pts.min(axis=0), pts.max(axis=0)


def _frame(ax, lo, hi, pad=0.04):
    """Equal-scale box sized to the data, with a small uniform margin."""
    span = hi - lo
    span = np.where(span < 1e-12, 1e-12, span)
    lo, hi = lo - pad * span, hi + pad * span
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)          # true proportions, no cube distortion


def _crop_radius(grid, r_min):
    """
    NaN out points inboard of r_min so nothing is drawn outside the zoom.

    The crop is by radius rather than by a bounding box because the tip patch
    already spans the outer third of the span: its bounding box is almost the
    whole blade, so framing on it produces no zoom at all.
    """
    g = grid.astype(float).copy()
    r = np.hypot(g[..., 1], g[..., 2])
    inside = r >= r_min
    g[~inside] = np.nan
    return g, inside


def _draw(ax, grid, colour, rows=True, cols=True, lw=0.6, alpha=0.9):
    for i in range(grid.shape[0]) if rows else []:
        ax.plot(*grid[i].T, color=colour, lw=lw, alpha=alpha)
    for j in range(grid.shape[1]) if cols else []:
        ax.plot(*grid[:, j].T, color=colour, lw=lw * 0.6, alpha=alpha * 0.55)


def main(npz_path=None, out="../explainer_figures/fig3_five_surfaces.png"):
    grids = load_grids(npz_path)
    all_g = [grids[k] for k, _, _ in SURFACES]

    fig = plt.figure(figsize=(15, 7.2), constrained_layout=True)

    # ---------------- left: whole blade ----------------
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    for key, label, colour in SURFACES:
        _draw(axL, grids[key], colour)
        axL.plot([], [], color=colour, lw=1.6, label=label)   # legend proxy
    lo, hi = _bounds(all_g)
    _frame(axL, lo, hi)
    axL.view_init(elev=22, azim=-58)
    axL.set_title("The five surfaces (grid rows = wrap curves)", pad=4)
    axL.legend(loc="upper left", fontsize=8, framealpha=0.9)
    axL.set_xlabel("x"); axL.set_ylabel("y"); axL.set_zlabel("z")
    axL.tick_params(labelsize=7)

    # ---------------- right: tip zoom ----------------
    # Keep the outer FRAC of the span, which is enough to show the tip rows
    # sweeping over the fold and handing over to the LE and TE strips.
    FRAC = 0.28
    allpts = np.vstack([g.reshape(-1, 3) for g in all_g])
    r_all = np.hypot(allpts[:, 1], allpts[:, 2])
    r_min = r_all.max() - FRAC * (r_all.max() - r_all.min())
    # Look roughly down the radius, from outboard, so the wrap curves are seen
    # as arcs going around the section instead of edge-on.
    tip_pt = allpts[np.argmax(r_all)]
    elev_r = np.degrees(np.arcsin(tip_pt[2] / r_all.max()))
    azim_r = np.degrees(np.arctan2(tip_pt[1], 0.0))

    axR = fig.add_subplot(1, 2, 2, projection="3d")
    kept = []
    for key, label, colour in SURFACES:
        g, inside = _crop_radius(grids[key], r_min)
        if not inside.any():
            continue
        kept.append(g[inside])
        _draw(axR, g, colour, lw=1.1, alpha=0.95)
    _frame(axR, *_bounds([np.vstack(kept)]), pad=0.06)
    axR.view_init(elev=elev_r - 30, azim=azim_r + 40)
    axR.set_title("Zoom at the tip: cuts wrap around the edges,\n"
                  "no shrinking rings, no cap", pad=4)
    axR.set_xlabel("x"); axR.set_ylabel("y"); axR.set_zlabel("z")
    axR.tick_params(labelsize=7)

    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else None
    main(npz)
