"""
plot_fig1_parameter_space.py

Regenerates explainer_figures/fig1_parameter_space.png: the five-surface split
drawn in blade parameter space (xi, eta).

The side rails c5..c8 are computed with the same build_side_curve used by the
pipeline, so the figure shows the split the code actually produces rather than
an idealised sketch.  The upper corners follow TipConfig.solve_corners: solved
from the TM 2013-178 Sec. 5 width conditions when True (the default), else the
prescribed constants.  The title reports the resulting strip width at both ends,
which should be nearly equal when the solve succeeds.
The tip-strip boundaries c11/c12 are true blade cuts; they are drawn here as
straight parameter-space segments between their three defining points, which
is accurate at the end-points and schematic in between.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from x_blade_new import X_blade
from tip_surfaces_new import (OutlineCurve, TipConfig, build_side_curve,
                              solve_upper_corner)

PITCH = np.array([1.025, 0.525, 0.55, 0.325, 0.325, 0.55])
CHORD = np.array([0.25, 0.65, 0.325, 0.55, 0.2])


def main(out="../explainer_figures/fig1_parameter_space.png"):
    blade = X_blade(PITCH, CHORD, 99999, return_blade_surface=True,
                    write_dat=False)[-1]
    cfg = TipConfig()
    e0 = blade.eta_min
    o = OutlineCurve(blade, cfg.n_outline)
    s_tip = o.s_tip
    p_te, _ = o.param_at_s(cfg.tip_extent * s_tip)
    p_le, _ = o.param_at_s(1.0 - cfg.tip_extent * (1.0 - s_tip))

    p_ll = np.array([cfg.xi_ll, e0])
    p_lr = np.array([cfg.xi_lr, e0])
    B = lambda p: blade.b(np.array([float(p[0])]), np.array([float(p[1])]))[0]
    if cfg.solve_corners:
        ur, w_le, dc_le = solve_upper_corner(blade, cfg.xi_lr, 0.5, p_le, 0.10, 0.49)
        ul, w_te, dc_te = solve_upper_corner(blade, cfg.xi_ll, 0.0, p_te, 0.01, 0.40)
        ur = ur or cfg.ur
        ul = ul or cfg.ul
        note = (f"upper corners solved from the Sec. 5 width conditions: "
                f"x_UR=({ur[0]:.3f}, {ur[1]:.3f}), x_UL=({ul[0]:.3f}, {ul[1]:.3f})")
    else:
        ur, ul = cfg.ur, cfg.ul
        note = (f"upper corners prescribed: x_UR=({ur[0]:.3f}, {ur[1]:.3f}), "
                f"x_UL=({ul[0]:.3f}, {ul[1]:.3f})")
    p_ul, p_ur = np.array(ul), np.array(ur)
    w_root = np.linalg.norm(B(p_lr) - B((0.5, e0)))
    w_up = np.linalg.norm(B(p_ur) - B(p_le))

    u = np.linspace(0, 1, 61)
    c6 = build_side_curve(blade, p_lr, p_ur, 0.5, p_le)(u)      # LE side rail
    c5 = build_side_curve(blade, p_ll, p_ul, 0.0, p_te)(u)      # TE side rail
    mir = lambda c: np.column_stack([1.0 - c[:, 0], c[:, 1]])   # suction mirror
    c7, c8 = mir(c6), mir(c5)
    M = lambda p: np.array([1.0 - p[0], p[1]])

    fig, ax = plt.subplots(figsize=(13, 6.4), constrained_layout=True)
    top_p = np.array([p_ul, p_ur])                     # c9
    top_s = np.array([M(p_ur), M(p_ul)])               # c10

    def poly(pts, **kw):
        P = np.vstack(pts); ax.fill(P[:, 0], P[:, 1], **kw)

    # central patches
    poly([[p_ll], c5, [p_ul, p_ur], c6[::-1], [p_lr]],
         color="0.75", alpha=.55, label="central pressure (Blade 2)")
    poly([c7, [M(p_ul)], c8[::-1]],
         color="violet", alpha=.35, label="central suction (Blade 4)")
    # LE strip: c6 -> tip-strip boundary -> c7
    poly([c6, [p_le], c7[::-1], [M(p_lr)], [p_lr]],
         color="salmon", alpha=.45, label="LE strip (Blade 3, wraps xi=0.5)")
    # TE strip wraps xi = 0, so it is drawn as two pieces
    poly([c5, [p_te], [[0.0, e0]], [p_ll]],
         color="cornflowerblue", alpha=.40, label="TE strip (Blade 1, wraps xi=0/1)")
    poly([c8[::-1], [[1.0, e0]], [[1.0, p_te[1]]]],
         color="cornflowerblue", alpha=.40)
    # tip surface: everything above the top curves / cut boundaries
    tip_bnd = np.vstack([[[0.0, p_te[1]]], [p_te], [p_ul], top_p, [p_ur],
                         [p_le], [M(p_ur)], top_s, [M(p_ul)],
                         [[1.0, p_te[1]]], [[1.0, 1.0]], [[0.0, 1.0]]])
    ax.fill(tip_bnd[:, 0], tip_bnd[:, 1], color="mediumseagreen", alpha=.35,
            label="tip surface (Blade 5, wraps over the tip)")

    for c, lab in ((c5, None), (c6, None), (c7, None), (c8, None)):
        ax.plot(c[:, 0], c[:, 1], "k-", lw=1.4)
    ax.plot(top_p[:, 0], top_p[:, 1], "k-", lw=1.4)
    ax.plot(top_s[:, 0], top_s[:, 1], "k-", lw=1.4)
    ax.axhline(1.0, color="darkgreen", lw=2.5)
    ax.text(.99, 1.005, "closed tip: b(xi,1) = b(1-xi,1)", ha="right",
            color="darkgreen", fontsize=9)
    ax.axhline(e0, color="k", lw=2.5)
    ax.text(.99, e0 - .022, "root ring", ha="right", fontsize=9)
    for x, lab, col in ((0.0, "trailing edge xi=0", "b"), (0.5, "leading edge xi=0.5", "r"),
                        (1.0, "trailing edge xi=1", "b")):
        ax.axvline(x, color=col, ls="--", lw=1)
        ax.text(x + .006, .30, lab, rotation=90, color=col, fontsize=8)

    pts = {"x_LL": p_ll, "x_LR": p_lr, "x_UL": p_ul, "x_UR": p_ur,
           "x_TE": p_te, "x_LE": p_le}
    for nm, p in pts.items():
        ax.plot(*p, "o", ms=7, mfc="yellow", mec="k", zorder=5)
        ax.annotate(nm, p, textcoords="offset points", xytext=(7, 6),
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("xi  (around the section: TE -> pressure -> LE -> suction -> TE)")
    ax.set_ylabel("eta  (root -> tip),   r = sin(pi*eta/2)")
    ax.set_title("Blade parameter space and the five-surface split\n" + note +
                 f"\nLE strip width: {w_root:.3f} m at the root, {w_up:.3f} m at the top",
                 fontsize=10)
    ax.set_xlim(-.02, 1.02); ax.set_ylim(e0 - .05, 1.05)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.34), ncol=3, fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
