"""
plot_fig2_tip_closure.py

Regenerates explainer_figures/fig2_tip_closure.png.

Left  : canonical section shapes approaching the tip, showing that they both
        shrink and collapse onto the camber line (the DRDC fold closure)
        rather than funnelling to a point.
Right : section thickness at mid-chord against r, showing that it reaches zero
        smoothly with no overshoot, and that the modification is confined to
        [r_close, r_tip].

Note on the left panel: the normal axis is exaggerated relative to the
chordwise axis.  At true scale these sections are ~30:1 slivers and the
collapse onto the camber line - the whole point of the panel - is invisible.
The exaggeration factor is stated on the axis label.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from x_blade_new import X_blade

PITCH = np.array([1.025, 0.525, 0.55, 0.325, 0.325, 0.55])
CHORD = np.array([0.25, 0.65, 0.325, 0.55, 0.2])


def main(out="../explainer_figures/fig2_tip_closure.png"):
    blade = X_blade(PITCH, CHORD, 99999, return_blade_surface=True,
                    write_dat=False)[-1]
    rt, rc = blade.r_tip, blade.r_close

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4),
                                   constrained_layout=True)

    # ---------------- left: section shapes ----------------
    fracs = [0.90, 0.95, 0.975, 0.99, 0.997, 0.9999]
    cols = plt.cm.viridis(np.linspace(0, .88, len(fracs)))
    xi = np.linspace(0, 1, 601)
    for f, c in zip(fracs, cols):
        r = f * rt
        can = blade._canonical(xi, np.full_like(xi, r))
        chord = float(np.maximum(blade.ChordLength(r), blade.chord_floor)) * blade.d
        axL.plot(can[:, 0] * chord, can[:, 1] * chord, color=c, lw=1.5,
                 label=f"r/R = {f:g}")
    axL.axhline(0, color="0.7", lw=.7, ls=":")
    axL.set_xlabel("chordwise (m)")
    xs, ys = axL.get_xlim(), axL.get_ylim()
    exag = (xs[1] - xs[0]) / (ys[1] - ys[0])
    axL.set_ylabel(f"normal (m)   [scale exaggerated {exag:.0f}x]")
    axL.set_title("Sections approaching the tip:\nthey shrink AND collapse onto "
                  "the camber line", fontsize=11)
    axL.legend(fontsize=8, loc="upper right")
    axL.grid(alpha=.25)

    # ---------------- right: mid-chord thickness ----------------
    rr = np.linspace(0.90 * rt, rt, 400)
    up = blade._canonical(np.full_like(rr, 0.25), rr)
    lo = blade._canonical(np.full_like(rr, 0.75), rr)
    chord = np.maximum(np.asarray(blade.ChordLength(rr)),
                       blade.chord_floor) * blade.d
    thick_mm = np.abs(up[:, 1] - lo[:, 1]) * chord * 1e3
    axR.plot(rr / rt, thick_mm, "k-", lw=2)
    axR.axvline(rc / rt, color="b", ls="--", lw=1.5)
    axR.annotate(f"closure interval starts\n(r_close = {rc/rt:.3f} R)",
                 xy=(rc / rt, thick_mm.max() * .88),
                 xytext=(-14, 0), textcoords="offset points",
                 ha="right", color="b", fontsize=9)
    axR.axhline(0, color="0.7", lw=.7, ls=":")
    axR.set_xlabel("r / R")
    axR.set_ylabel("section thickness at mid-chord (mm)")
    axR.set_title("Thickness reaches zero smoothly:\nno overshoot, modification "
                  "confined to [r_close, R]", fontsize=11)
    axR.grid(alpha=.25)
    axR.set_xlim(0.90, 1.001)
    axR.set_ylim(-0.4, thick_mm.max() * 1.06)

    fig.savefig(out, dpi=200)
    print(f"wrote {out}   (exaggeration {exag:.0f}x, r_close={rc:.4f} m, r_tip={rt:.4f} m)")


if __name__ == "__main__":
    main()
