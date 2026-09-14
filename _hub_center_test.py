"""
_hub_center_test.py

Checks that the hub sector is CENTRED width-wise on the blade root section:
that the sector's two side edges sit equally far from the root foil's two
surfaces at every axial station.

The measurement is deliberately INDEPENDENT of hub_new's own diagnostic:

  * the edge positions are read back out of the exported hub_sector GRID,
    not from the centre curve p_h;
  * the blade's two surfaces are re-sampled from the analytic blade at the
    design hub radius, with a crossing test written from scratch below.

So agreement between this script and hub_new's reported numbers is real
corroboration rather than the same code run twice. The script also builds
the hub BOTH ways (doc Eqs. 51-53 and centred) so the improvement is
measured, and re-checks the invariants that centring must not disturb:
sector width, tiling over Z copies, cap seams and hub radius.

Run:  python _hub_center_test.py          (numpy/scipy only, no OCC)
"""

import warnings

import numpy as np
from scipy.interpolate import CubicSpline

warnings.filterwarnings("ignore")

from x_blade_new import X_blade                       # noqa: E402
from tip_surfaces_new import build_drdc_grids, TipConfig   # noqa: E402
from hub_new import hub_grids                         # noqa: E402

PITCH = np.array([1.025, 0.525, 0.55, 0.325, 0.325, 0.55])
CHORD = np.array([0.25, 0.65, 0.325, 0.55, 0.2])
Z = 5

# Which sweep setting the PASS/FAIL is judged on: must match
# hub_new.CENTRE_SMOOTH_KNOTS, since that is what the pipeline will build.
DEFAULT_TAG = "centred, 6 knots"

# Pass thresholds. The doc construction measures 23.65 mm of asymmetry, so
# anything under ~0.5 mm is already 50x better and is about 1.5% of the
# tightest gap (29 mm of arc): invisible. Chasing the last tenth of a mm
# is NOT free -- interpolating the mid-line exactly got to 0.018 mm but
# multiplied the strip edge's curvature by 140. So the curvature bound is
# a real criterion here, not a formality, and asymmetry is allowed to be
# looser than in earlier runs.
#
# The max is taken over the WHOLE footprint, and it is dominated by the two
# axial extremes, where the section turns around and the edge gaps are
# ~35 deg: 1 mm there is under 1% of the local gap. Through the blade body,
# where the gaps close to 14 deg and a viewer would actually notice, the
# two sides agree to ~0.01 deg. So 1.0 mm on this metric is generous-
# looking but genuinely invisible, and is still 24x better than the doc.
MAX_ASYM_MM = 1.00        # max |gap_above - gap_below| as arc on the hub
MIN_IMPROVEMENT = 20.0    # centred must beat the doc construction by this
MAX_CURV_RATIO = 5.0      # strip-edge curvature, relative to the doc's


def wrap(t):
    return np.arctan2(np.sin(t), np.cos(t))


def envelope(x_fp, th_fp, xq):
    """min / max theta of the closed polyline at each station in xq.

    Written independently of hub_new._theta_envelope: plain loop over
    segments, explicit wrap-around, no numpy roll tricks.
    """
    n = len(x_fp)
    lo = np.full(len(xq), np.nan)
    hi = np.full(len(xq), np.nan)
    for k, x in enumerate(xq):
        vals = []
        for i in range(n):
            j = i + 1 if i + 1 < n else 0
            a, b = x_fp[i], x_fp[j]
            if a == b:
                continue
            if (a - x) * (b - x) <= 0.0:
                w = (x - a) / (b - a)
                vals.append(th_fp[i] + w * (th_fp[j] - th_fp[i]))
        if vals:
            lo[k] = min(vals)
            hi[k] = max(vals)
    return lo, hi


def measure(G, blade, r_hub, th_ref, n_stations=401):
    """Edge gaps, taken from the exported sector grid.

    Sampled DENSELY in x, not just at the grid's own columns. Querying only
    at the columns was the first version of this test and it under-reported
    the asymmetry by 40% (0.77 mm against a true 1.26 mm), because the
    worst off-centring sits between columns near the ends of the footprint.
    Between columns the edge is modelled with a cubic through the column
    values, which is what the exported B-spline surface does, so this also
    picks up any error from the axial resolution of the grid itself.
    """
    def th_of(P):
        return wrap(np.arctan2(P[:, 1], P[:, 2]) - th_ref)

    x_col = G[0, :, 0]
    edge_lo = CubicSpline(x_col, th_of(G[0, :, :]))     # s=0 -> p_h - pi/Z
    edge_hi = CubicSpline(x_col, th_of(G[-1, :, :]))    # s=1 -> p_h + pi/Z

    xi = np.linspace(0.0, 1.0, 1501)[:-1]
    eta = blade.eta_of_r(r_hub / (blade.d / 2.0))
    fp = blade.b(xi, np.full(xi.shape, eta))
    x_fp, th_fp = fp[:, 0], th_of(fp)

    xq = np.linspace(float(x_fp.min()), float(x_fp.max()), n_stations)[1:-1]
    lo, hi = envelope(x_fp, th_fp, xq)
    good = np.isfinite(lo)
    xq, lo, hi = xq[good], lo[good], hi[good]

    gap_above = edge_hi(xq) - hi
    gap_below = lo - edge_lo(xq)
    return gap_above, gap_below


def edge_shape(G, th_ref):
    """How hard the strip edge turns, independent of column count.

    Centring buys equal gaps by making p_h follow the footprint mid-line,
    which curves hardest at the two ends of the section. If that made the
    edge turn far more sharply than the doc's near-straight curve did, the
    hub sector would be harder to mesh -- the trade would not be worth it.
    Reported as max slope and max curvature of theta(x).
    """
    def th_of(P):
        return wrap(np.arctan2(P[:, 1], P[:, 2]) - th_ref)

    x_col = G[0, :, 0]
    sp = CubicSpline(x_col, th_of(G[-1, :, :]))
    xq = np.linspace(float(x_col[0]), float(x_col[-1]), 2001)
    return (float(np.degrees(np.abs(sp(xq, 1)).max())),
            float(np.degrees(np.abs(sp(xq, 2)).max())))


def main():
    print("building blade and grids ...")
    blade = X_blade(PITCH, CHORD, 99999,
                    return_blade_surface=True, write_dat=False)[-1]
    cfg = TipConfig(delta_c=0.05, n_wrap=61, n_outline=1500, max_curves=16)
    grids = build_drdc_grids(blade, cfg, verbose=False)

    r_hub = grids["meta"]["hub_radius"]
    if grids["meta"].get("root_footprint") is None:
        print("FAIL: meta['root_footprint'] missing; tip_surfaces_new is "
              "not the centring-aware version")
        return 1
    print(f"design hub radius {r_hub:.6f} m, Z = {Z}")

    # Equal gaps are only worth having if the strip edge stays meshable, so
    # sweep the smoothing and print both numbers for each setting. "interp"
    # follows the mid-line exactly; fewer knots = smoother but less centred.
    cases = [("doc Eqs. 51-53", dict(center_on_footprint=False)),
             ("centred, interp", dict(centre_smooth=None)),
             ("centred, 24 knots", dict(centre_smooth=24)),
             ("centred, 16 knots", dict(centre_smooth=16)),
             ("centred, 10 knots", dict(centre_smooth=10)),
             ("centred, 6 knots", dict(centre_smooth=6))]

    out = {}
    for tag, kw in cases:
        hg, info = hub_grids(grids, n_blades=Z, verbose=False, **kw)
        G = hg["hub_sector"]
        above, below = measure(G, blade, r_hub, info["theta_ref"])
        asym_mm = np.abs(above - below) * r_hub * 1000.0
        slope, curv = edge_shape(G, info["theta_ref"])
        out[tag] = dict(info=info, hg=hg, G=G, above=above, below=below,
                        asym_mm=asym_mm, slope=slope, curv=curv)

        print(f"\n--- {tag}")
        print(f"  gap above blade : {np.degrees(above.min()):8.3f} .. "
              f"{np.degrees(above.max()):8.3f} deg")
        print(f"  gap below blade : {np.degrees(below.min()):8.3f} .. "
              f"{np.degrees(below.max()):8.3f} deg")
        print(f"  |above - below| : max {asym_mm.max():.4f} mm of arc "
              f"({np.degrees(np.abs(above - below)).max():.4f} deg)")
        print(f"  hub_new reports : {info['offcentre_mm']:.4f} mm "
              f"({info['offcentre_deg']:.4f} deg)")
        print(f"  edge turn       : max slope {slope:.1f} deg/m, "
              f"max curvature {curv:.0f} deg/m^2")
        # Asymmetry where it is actually visible: the station where the
        # blade comes closest to an edge. The max above is dominated by the
        # two axial extremes, where the gaps are huge and nobody can see it.
        it = int(np.argmin(np.minimum(above, below)))
        print(f"  at tightest gap : {asym_mm[it]:.4f} mm asymmetry, gap "
              f"{np.degrees(min(above[it], below[it])):.3f} deg")
        print(f"  sector grid     : {G.shape[0]} x {G.shape[1]}")

    print("\n=== tradeoff (want low asymmetry AND curvature near the doc's)")
    doc_curv = out["doc Eqs. 51-53"]["curv"]
    print(f"  {'setting':<20} {'asym mm':>9} {'curv deg/m^2':>14} "
          f"{'vs doc':>8}")
    for tag, _ in cases:
        d = out[tag]
        print(f"  {tag:<20} {d['asym_mm'].max():9.4f} {d['curv']:14.0f} "
              f"{d['curv'] / doc_curv:7.1f}x")

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    base = out["doc Eqs. 51-53"]["asym_mm"].max()
    d = out[DEFAULT_TAG]
    cent = d["asym_mm"].max()
    info_c = d["info"]

    print(f"\n=== centring (judged on the default: {DEFAULT_TAG})")
    check("centred edges equidistant", cent <= MAX_ASYM_MM,
          f"max asymmetry {cent:.4f} mm (limit {MAX_ASYM_MM})")
    check("improves on the doc construction",
          base > 0 and base / max(cent, 1e-12) >= MIN_IMPROVEMENT,
          f"{base:.4f} -> {cent:.4f} mm "
          f"({base / max(cent, 1e-12):.0f}x)")
    check("strip edge stays meshable",
          d["curv"] <= MAX_CURV_RATIO * doc_curv,
          f"curvature {d['curv']:.0f} vs doc {doc_curv:.0f} deg/m^2 "
          f"({d['curv'] / doc_curv:.1f}x, limit {MAX_CURV_RATIO}x)")
    check("blade still inside its sector",
          info_c["min_edge_gap_deg"] > 0.0,
          f"smallest edge gap {info_c['min_edge_gap_deg']:.2f} deg")

    print("\n=== invariants (must be unchanged by centring)")
    G = d["G"]
    rad = np.hypot(G[:, :, 1], G[:, :, 2])
    check("sector width exactly 360/Z",
          abs(info_c["sector_deg"] - 360.0 / Z) < 1e-9,
          f"{info_c['sector_deg']:.6f} deg")
    check("hub radius exact",
          float(np.abs(rad - r_hub).max()) < 1e-12,
          f"max dev {float(np.abs(rad - r_hub).max()):.2e} m")
    check("tiles over Z copies", info_c["tile_err"] < 1e-12,
          f"{info_c['tile_err']:.2e} m")
    check("cap seams closed", info_c["cap_seam"] < 1e-12,
          f"{info_c['cap_seam']:.2e} m")
    check("caps present",
          {"hub_cap_lo", "hub_cap_hi"} <= set(d["hg"]))

    print("\n== RESULT:", "ALL PASS ==" if ok else "FAILURES ABOVE ==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
