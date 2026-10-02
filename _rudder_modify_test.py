"""
_rudder_modify_test.py

Self-check for rudder_modify.py (the shape change below H), bezier_section.py
(the Bezier half-section of any degree it builds its sections with) and the
pieces of replica_funcs it added (fillet_geometry, sharp_chord,
rounded_section's num_pts layout, the faster cut_for_radius).

Most checks run on a synthetic rudder written to a temporary folder: a
parabolic LE, a straight TE and section parameters that are polynomials in
the height, built with the same sections, so every fit and every baseline
must come back exactly. When the extraction's outputs are in outputs/ (the
X_geom repo), the real rudder is checked too.

    python _rudder_modify_test.py
"""

import importlib.util
import os
import tempfile

import numpy as np

import bezier_section as BS
import replica_funcs as RF
import rudder_modify as RM

_FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))
REAL = {"corner": "outputs/corner_points.dat", "table": "outputs/thickness_params_H80_info.dat",
        "xcad": "outputs/sections_xcad_with_cap_y0_y200_m.dat",
        "stack": "outputs/sections_stack_y0_y200_raw_mm.dat"}


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<62s} max error {err:.3e}   (tol {tol:.0e})")


def raises(name, fn, exc=ValueError):
    try:
        fn()
    except exc:
        check(name, 0, 0, 0)
        return
    check(name + " (did not raise)", 1, 0, 0)


# --------------------------------------------------------------------------
# a synthetic rudder
# --------------------------------------------------------------------------

Z_OPT, R_TE = 80.0, 0.75
x_le_true = lambda y: 0.07 * y + 2.0e-4 * y**2
x_te_true = lambda y: 180.0 - 0.14 * y
LAW_TRUE = {"X1": lambda y: 0.2177 + 2.0e-4 * (y / 80.0),
            "X2": lambda y: 0.30 + 0.02 * (y / 80.0)**2,
            "T1": lambda y: 0.0734 - 0.004 * (y / 80.0) + 0.001 * (y / 80.0)**3,
            "T2": lambda y: 0.1235 + 0.002 * (y / 80.0)**2}


def true_section(y, num_pts=200):
    """The synthetic original section at height y: (x_le, chord, cs, r, selig)."""
    X1, X2, T1, T2 = (LAW_TRUE[c](y) for c in ("X1", "X2", "T1", "T2"))
    chord = x_te_true(y) - x_le_true(y)
    cs, r = RF.sharp_chord(X1, X2, T1, T2, chord, R_TE)
    return x_le_true(y), chord, cs, r, RF.rounded_section(X1, X2, T1, T2, r, chord=chord,
                                                          chord_type="actual", num_pts=num_pts)


def write_synthetic(tmp):
    """corner_points.dat, a section table up to Z_OPT, an XCAD file and a
    stack file of the synthetic rudder, in the extraction's formats."""
    y_rows = np.arange(0.0, 151.0, 2.0)
    p = {k: os.path.join(tmp, n) for k, n in (("corner", "corner_points.dat"), ("table", "table_info.dat"),
                                              ("xcad", "xcad_m.dat"), ("stack", "stack_raw_mm.dat"))}
    with open(p["corner"], "w") as fh:
        fh.write("# synthetic corner points\n#   x1 y1 z1 x2 y2 z2\n")
        for y in y_rows:
            fh.write(f"{x_le_true(y):20.12f} {y:20.12f} 0.0 {x_te_true(y):20.12f} {y:20.12f} 0.0\n")
        fh.write(f"{x_le_true(151.0):20.12f} 151.0 0.0 {x_te_true(151.0):20.12f} 157.0 0.0\n")  # tilted row
    rows = []
    for k, y in enumerate(np.linspace(0.0, Z_OPT, 50), start=1):
        _, chord, cs, r, _ = true_section(y)
        rows.append({"section": k, "y_mm": y, "chord_mm": chord, "cs_mm": cs,
                     **{c: LAW_TRUE[c](y) for c in ("X1", "X2", "T1", "T2")}, "r": r, "r_mm": R_TE,
                     "R_orig_mm": R_TE, "fit_um": 0.0, "gap_rms_um": 0.0, "gap_max_um": 0.0,
                     "rle_fit_mm": 0.0, "rle_orig_mm": 0.0})
    RM.write_section_table(p["table"], rows, ["source       : synthetic"])
    loops, stack = [], []
    for y in y_rows:
        x_le, chord, _, _, selig = true_section(y)
        loops.append(RM.xcad_lines(RM.xcad_loop(selig, x_le, y), "m"))
        stack.append((y, chord, x_le, selig))
    top = RM.xcad_loop(true_section(150.0)[4], x_le_true(150.0), 150.0)
    for dz in (3.0, 6.0):                                  # two 'cap' loops, tilted
        cap = top.copy()
        cap[:, 1] += dz + 0.01 * (cap[:, 0] - cap[0, 0])
        loops.append(RM.xcad_lines(cap, "m"))
    RM.write_xcad_lines(p["xcad"], loops)
    with open(p["stack"], "w") as fh:
        fh.write("# synthetic stack, top first\n")
        for k, (y, chord, x_le, selig) in enumerate(stack[::-1], start=1):
            fh.write(f"# SECTION {k} / {len(stack)}   y = {y:.6f}   chord = {chord:.6f}   x_le = {x_le:.6f}"
                     f"   x_te = {x_le + chord:.6f}\n#   x_mm y_mm\n")
            fh.writelines(f"{a:20.9f} {b:18.9f}\n" for a, b in selig)
    return p


# --------------------------------------------------------------------------

def test_bezier():
    print("\nBezier curves")
    rng = np.random.default_rng(3)
    ctrl = rng.uniform(-1.0, 1.0, (6, 2))
    w = rng.uniform(0.3, 2.0, 6)
    b = RM.Bezier(ctrl, w)
    t = np.linspace(0.0, 1.0, 13)
    hom = np.column_stack((ctrl * w[:, None], w))[None].repeat(len(t), 0)
    for k in range(5, 0, -1):                                           # de Casteljau
        hom = (1 - t[:, None, None]) * hom[:, :k] + t[:, None, None] * hom[:, 1:k + 1]
    check("rational order 5 against de Casteljau", b(t), hom[:, 0, :2] / hom[:, 0, 2:], 1e-14)
    path = os.path.join(HERE, "Bezier code", "para_control_bez.py")
    c4, w4 = ctrl[:4], w[:4]
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("para_control_bez", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        X, Y = mod.eval_rational_bezier(c4.T, w4, t)
        check("cubic: same as eval_rational_bezier (para_control_bez.py)", RM.Bezier(c4, w4)(t),
              np.column_stack((X, Y)), 1e-14)
    else:
        print("       (Bezier code/para_control_bez.py not here: eval_rational_bezier not compared)")
    h = 1e-6
    fd = (b(t[1:-1] + h) - b(t[1:-1] - h)) / (2 * h)
    check("derivative against central differences", b.derivative(t[1:-1]), fd, 1e-7)
    left, right = b.split(0.37)
    check("split: left piece is [0, 0.37]", left(t), b(0.37 * t), 1e-13)
    check("split: right piece is [0.37, 1]", right(t), b(0.37 + 0.63 * t), 1e-13)
    check("elevate by 3: same curve", b.elevate(3)(t), b(t), 1e-13)
    heights = RM.Bezier(np.column_stack(([80.0, 80.0, 50.0, 10.0, 0.0], [1, 2, 3, 4, 5])), [1, 2, 1, .5, 1])
    yq = np.array([80.0, 79.999, 55.5, 3.0, 0.0])
    check("solve: heights with a flat start, rational", heights(heights.solve(yq))[:, 0], yq, 1e-11)
    check("solve: ends exactly at t = 0 and 1", heights.solve([80.0, 0.0]), [0.0, 1.0], 0.0)
    raises("solve: a height outside the curve raises", lambda: heights.solve([80.5]))
    s = RM.heights_from_fractions(rng.uniform(0.0, 1.0, 6))
    check("d fractions: heights in order from 1 to 0", [s[0], s[-1], np.any(np.diff(s) > 0)], [1, 0, 0], 0.0)
    check("d fractions: round trip", RM.heights_from_fractions(RM.fractions_from_heights(s)), s, 1e-15)
    check("even heights from their fractions", RM.heights_from_fractions(
        RM.fractions_from_heights(RM.uniform_heights(5))), RM.uniform_heights(5), 1e-15)


def test_fits():
    print("\nfits of laws")
    y = np.linspace(0.0, 100.0, 60)
    v = 0.3 + 0.01 * y - 2e-4 * y**2 + 1e-6 * y**3
    law, res = RM.fit_law(y, v, 3)
    check("order 3 through a cubic: exact", res, 0.0, 1e-13)
    noisy = v + 1e-3 * np.sin(y)
    law, res = RM.fit_law(y, noisy, 4)
    check("even heights: the least-squares polynomial (np.polyfit)", law.at(y)[:, 1],
          np.polyval(np.polyfit(y, noisy, 4), y), 1e-11)
    vals, res = RM.fit_values(RM.uniform_heights(3) * 100.0, y, v, pinned={0: 1.0, 3: 0.3})
    check("pinned values held", vals[[0, 3]], [1.0, 0.3], 0.0)
    curved = (y / 100.0)**0.3
    _, r_even = RM.fit_law(y, curved, 3)
    _, r_free = RM.fit_law(y, curved, 3, free_heights=True)
    rms = lambda r: np.sqrt(np.mean(r**2))
    print(f"       curved law, order 3: RMS {rms(r_even):.2e} with even heights, {rms(r_free):.2e} searched")
    check("free heights (Main_PSO) fit a curved law at least 3x better", rms(r_free) < rms(r_even) / 3, 1, 0)
    w = np.array([1.0, 0.5, 2.0, 0.8])
    h = RM.uniform_heights(3) * 100.0
    true = RM.Bezier(np.column_stack((h, [0.2, 0.35, 0.1, 0.3])), w)
    vals, res = RM.fit_values(h, y, true.at(y)[:, 1], w)
    check("rational law (weights held): values recovered from its points", vals, [0.2, 0.35, 0.1, 0.3], 1e-12)


def test_replica():
    print("\nreplica_funcs: TE rounding and the extraction layout")
    X1, X2, T1, T2 = 0.2177, 0.30, 0.0734, 0.1235
    xi = [T1, X1, X2, T2]
    cut, center, r, x_te = RF.fillet_geometry(X1, X2, T1, T2, 0.004)
    y, s = RF.y_and_slope(xi, cut[0])
    check("fillet_geometry: cut on the curve, circle tangent there",
          [cut[1] - y[0], np.hypot(*(cut - center)) - r, np.dot(cut - center, [1.0, s[0]]), center[1]],
          [0, 0, 0, 0], 1e-13)
    check("fillet_geometry: y sqrt(1 + s^2) = r at the cut", y[0] * np.sqrt(1 + s[0]**2), 0.004, 1e-15)
    cs, rf = RF.sharp_chord(X1, X2, T1, T2, 163.3312, 0.75)
    xy = RF.rounded_section(X1, X2, T1, T2, rf, chord=cs, chord_type="sharp")
    check("sharp_chord: rounded TE at the chord, r * cs = r_abs", [xy[0][0], rf * cs], [163.3312, 0.75], 1e-9)
    for n in (200, 201):
        xy = RF.rounded_section(X1, X2, T1, T2, rf, chord=163.3312, chord_type="actual", num_pts=n)
        nh = (n + 1) // 2
        stations = lambda m: 0.5 * (1 - np.cos(np.linspace(0, np.pi, m)))
        check(f"num_pts {n}: LE at {nh - 1}, TE first and last, cosine stations per side",
              [len(xy) - n, *xy[nh - 1], *(xy[0] - [163.3312, 0]), *(xy[-1] - [163.3312, 0]),
               np.abs(xy[:nh][::-1, 0] / 163.3312 - stations(nh)).max(),
               np.abs(np.r_[0, xy[nh:, 0]] / 163.3312 - stations(n - nh + 1)).max()], 0.0, 1e-12)
    up = xy[:nh][::-1] / (163.3312 / x_te_actual(X1, X2, T1, T2, rf))
    cut, center, r, _ = RF.fillet_geometry(X1, X2, T1, T2, rf)
    arc = up[:, 0] > cut[0]
    check("num_pts: points on the Bezier, then on the arc",
          [np.abs(RF.half_airfoil_y(xi, up[~arc, 0])[0] - up[~arc, 1]).max(),
           np.abs(np.hypot(*(up[arc] - center).T) - r).max()], [0, 0], 1e-12)
    raises("n_per_side and num_pts together raise",
           lambda: RF.rounded_section(X1, X2, T1, T2, rf, n_per_side=10, num_pts=20))


def test_bezier_section():
    print("\nbezier_section: the half-section of any degree")
    X1, X2, T1, T2 = 0.2177, 0.30, 0.0734, 0.1235
    q = RM.quartic_polygons(X1, X2, T1, T2)
    cs_old, r_old = RF.sharp_chord(X1, X2, T1, T2, 163.3312, 0.75)
    old = RF.rounded_section(X1, X2, T1, T2, r_old, chord=163.3312, chord_type="actual", num_pts=200)
    new, cs, r = BS.sections(q, 163.3312, 0.75, 200)
    check("quartic: the same section as replica_funcs (mm), same sharp chord",
          [np.abs(new[0] - old).max(), cs[0] - cs_old], 0.0, 1e-11)
    e = BS.elevate(q, 3)
    new7, cs7, _ = BS.sections(e, 163.3312, 0.75, 200)
    check("raised to degree 7: the same section, reduced back: the same polygon",
          [np.abs(new7[0] - old).max(), cs7[0] - cs_old, np.abs(BS.reduce(e, 4) - q).max()], 0.0, 1e-11)
    check("LE radius = 4 T1^2 / (3 X1) for the quartic, unchanged by raising the degree",
          [BS.le_radius(q)[0] - 4 * T1**2 / (3 * X1), BS.le_radius(e)[0] - 4 * T1**2 / (3 * X1)], 0.0, 1e-12)
    g = BS.polygon([[0.02, 0.05], [0.15, 0.09], [0.45, 0.07], [0.8, 0.03]])        # degree 5, P1x not 0
    f = BS.fillet(g, 0.004)
    p = BS.points(g, [f["t"][0]])[0, 0]
    dp = BS.derivative(g, [f["t"][0]])[0, 0]
    check("degree 5: TE circle tangent to the curve, centre on the chord line, radius r",
          [p[0] - f["x"][0], np.dot(p - [f["cx"][0], 0.0], dp), np.hypot(p[0] - f["cx"][0], p[1]) - 0.004],
          0.0, 1e-12)
    sel, cs5, r5 = BS.sections(g, 120.0, 0.6, 201)
    nh = 101
    check("degree 5: layout of the extraction (LE at 100, TE first and last on the chord), r * cs = 0.6",
          [*sel[0, nh - 1], sel[0, 0, 0] - 120.0, sel[0, -1, 0] - 120.0, sel[0, 0, 1], r5[0] * cs5[0] - 0.6],
          0.0, 1e-9)
    check("x_monotone: a polygon folding back is caught",
          [BS.x_monotone(q)[0], BS.x_monotone(BS.polygon([[0.0, 0.07], [0.9, 0.07], [-0.6, 0.1]]))[0]], [1, 0], 0)
    check("names of the inner coordinates", BS.coord_names(5) == ["P1x", "P1z", "P2x", "P2z", "P3x", "P3z", "P4x", "P4z"],
          1, 0)
    vals = BS.values_from_polygon(e)
    check("polygon <-> {name: value}", np.abs(BS.polygon_from_values(vals, 7) - e).max(), 0.0, 0.0)


def x_te_actual(X1, X2, T1, T2, r):
    return RF.fillet_geometry(X1, X2, T1, T2, r)[3]


SECTION = {"P1z": "T1", "P2x": "X1", "P2z": "T1", "P3x": "X2", "P3z": "T2"}   # degree-4 coordinates of the table


def test_original_and_designs(p):
    print("\noriginal shape, designs and sections (synthetic rudder)")
    orig = RM.fit_original(p["corner"], p["table"], Z_OPT)
    check("the curves: LE and the x, z of P1 .. P3",
          orig.names == ["LE", "P1x", "P1z", "P2x", "P2z", "P3x", "P3z"], 1, 0)
    check("fits: the parabolic LE comes back exactly (mm)", np.abs(orig.residuals["LE"]).max(), 0.0, 1e-10)
    check("fits: the cubic coordinates too, to the table's 7 decimals",
          [np.abs(orig.residuals[c]).max() for c in orig.names[1:]], 0.0, 1e-7)
    yy = np.linspace(0.0, Z_OPT, 17)
    check("original LE and its slope at any height", [*orig.value("LE", yy), *orig.slope("LE", yy)],
          [*x_le_true(yy), *(0.07 + 4e-4 * yy)], 1e-10)
    check("original coordinates = the laws they come from (P1x = 0, P1z = P2z = T1, ...)",
          [np.abs(orig.value("P1x", yy)).max(), *(np.abs(orig.value(c, yy) - LAW_TRUE[t](yy)).max()
                                                 for c, t in SECTION.items())], 0.0, 1e-7)
    check("TE line from the rows", orig.x_te(yy), x_te_true(yy), 1e-10)
    raises("a height above z_top raises", lambda: orig.value("P1z", [Z_OPT + 1.0]))
    raises("a degree below the table's quartics raises", lambda: orig.at_degree(3))

    base = RM.baseline_design(orig, Z_OPT)
    secs = RM.modified_sections(base, orig, 50, 200, R_TE)
    worst = 0.0
    for s in secs:
        x_le, chord, cs, r, selig = true_section(s.y)
        worst = max(worst, abs(s.x_le - x_le), abs(s.chord - chord), abs(s.cs - cs), abs(s.r_mm - R_TE),
                    np.abs(s.selig - selig).max())
    check("baseline H = Z_opt: every section is the original one (mm)", worst, 0.0, 2e-5)
    check("section heights from the root to H", [secs[0].y, secs[-1].y, len(secs)], [0.0, Z_OPT, 50], 0.0)
    join = []
    for s in secs:
        f = BS.fillet(s.ctrl, s.r)
        pt = BS.points(s.ctrl, [f["t"][0]])[0, 0]
        join.append([np.arctan(f["slope"][0]) - np.arctan(-(pt[0] - f["cx"][0]) / pt[1]),
                     (pt[1] - np.sqrt(f["r"][0]**2 - (pt[0] - f["cx"][0])**2)) * s.cs, pt[0] * s.cs / s.chord])
    join = np.array(join)
    check("TE circle meets the Bezier tangentially in every section (rad, mm)", join[:, :2], 0.0, 1e-12)
    check("the cut moves smoothly along the span (2nd difference of x_cut / c)", np.diff(join[:, 2], 2), 0.0, 1e-6)

    b50 = RM.baseline_design(orig, 50.0)
    s50 = RM.modified_sections(b50, orig, 26, 200, R_TE)
    worst = max(np.abs(s.selig - true_section(s.y)[4]).max() + abs(s.x_le - x_le_true(s.y)) for s in s50)
    check("baseline H = 50: the original below 50 too (split fits)", worst, 0.0, 2e-5)

    d = base.copy()
    d.curves["LE"].v[-1] -= 20.0
    d.curves["P2z"].v[2] += 0.01
    d.curves["P2x"].join = "G0"
    d.curves["P2x"].v[1] += 0.02
    pd = RM.pin(d, orig)
    dy = lambda b: (lambda q: q[:, 1] / q[:, 0])(b.derivative([0.0]))
    check("change: root LE 20 mm forward, the LE still meets the original at H",
          [pd.curve("LE")([1.0])[0, 1], *pd.curve("LE")([0.0])[0]], [-20.0, Z_OPT, x_le_true(Z_OPT)], 1e-12)
    check("change: G1 keeps the slopes at H (LE, P2z)",
          [dy(pd.curve("LE"))[0] - orig.slope("LE", Z_OPT)[0], dy(pd.curve("P2z"))[0] - orig.slope("P2z", Z_OPT)[0]],
          [0, 0], 1e-12)
    check("change: G0 keeps the value at H only (P2x)",
          [pd.curve("P2x")([0.0])[0, 1] - orig.value("P2x", Z_OPT)[0],
           abs(pd.curves["P2x"].v[1] - base.curves["P2x"].v[1])], [0, 0.02], 1e-12)
    spd = RM.modified_sections(pd, orig, 50, 200, R_TE)
    check("change: root chord 20 mm longer, TE radius still 0.75 mm, TE on the TE line",
          [spd[0].chord - secs[0].chord, spd[0].r_mm, spd[0].selig[0, 0] - spd[0].chord, spd[-1].x_le - secs[-1].x_le],
          [20.0, R_TE, 0.0, 0.0], 1e-9)
    check("change: P2 no longer level with P1, so no thickness format",
          [spd[25].params() is None, secs[25].params() is not None], [1, 1], 0)
    bad = base.copy()
    bad.curves["LE"].v[-1] = 400.0
    raises("an LE behind the TE raises", lambda: RM.modified_sections(bad, orig, 10))
    raises("H above z_top raises", lambda: RM.pin(RM.Design(Z_OPT + 5.0, base.curves), orig))
    bad = base.copy()
    bad.curves["P3z"].s = np.array([1.0, 0.2, 0.6, 0.0])
    raises("control heights rising towards the root raise", lambda: RM.pin(bad, orig))

    space = RM.DesignSpace(orig)
    check("H bounds: [h_min, h_max], by default a quarter of z_top and z_top; h_max below z_top",
          [*RM.DesignSpace(orig, h_min=30.0).bounds[0], *space.bounds[0], *RM.DesignSpace(orig, h_max=60.0).bounds[0]],
          [30.0, Z_OPT, Z_OPT / 4, Z_OPT, 15.0, 60.0], 0.0)
    raises("h_min above z_top raises", lambda: RM.DesignSpace(orig, h_min=Z_OPT + 1.0))
    raises("h_max above z_top raises", lambda: RM.DesignSpace(orig, h_min=20.0, h_max=Z_OPT + 1.0))
    x0 = space.to_vector(base)
    check("design vector: 29 entries for order 3, G1 (H, then 4 per curve: LE and P1x .. P3z)",
          [len(space), space.bounds.shape[0]], [29, 29], 0)
    check("design vector of the baseline: H first, LE offsets of its control points",
          [x0[0], *(x0[3:5] - (base.curves["LE"].v[2:] - x_le_true(base.curves["LE"].s[2:] * Z_OPT)))],
          [Z_OPT, 0, 0], 1e-10)
    rng = np.random.default_rng(7)
    x = space.bounds[:, 0] + rng.uniform(0.2, 0.8, len(space)) * np.ptp(space.bounds, axis=1)
    check("design vector -> design -> vector", space.to_vector(space.to_design(x)), x, 1e-10)
    check("design from a vector is pinned at its H", space.to_design(x).curve("P3z")([0.0])[0],
          [x[0], orig.value("P3z", x[0])[0]], 1e-12)
    sp = RM.DesignSpace(orig, bounds={"LE.dx3": (-40.0, 0.0), "P1z": (0.02, 0.1), "z": (0.01, 0.3)},
                        fixed={"P3x.v2": 0.31, "H": 70.0}, h_min=30.0)
    ib = lambda n: list(sp.all_bounds[sp.all_names.index(n)])
    check("bounds: a slot beats its curve, a curve its group; fixed slots leave the vector",
          [len(sp), *ib("LE.dx3"), *ib("P1z.v2"), *ib("P3z.v3"), *ib("P2x.v2"), "P3x.v2" in sp.names, "H" in sp.names],
          [27, -40.0, 0.0, 0.02, 0.1, 0.01, 0.3, 0.0, 1.0, 0, 0], 0.0)
    xs = sp.to_vector(base)
    ds = sp.to_design(xs)
    check("fixed slots take their values in to_design (H 70, P3x C2 0.31)", [ds.H, ds.curves["P3x"].v[2]],
          [70.0, 0.31], 1e-12)
    rec = sp.to_dict(ds)
    sp2, ds2 = RM.DesignSpace.from_dict(orig, rec)
    check("to_dict / from_dict: same variables, bounds and design",
          [sp2.names == sp.names, np.abs(sp2.bounds - sp.bounds).max(), np.abs(sp2.to_vector(ds2) - sp.to_vector(ds)).max()],
          [1, 0, 0], 1e-12)
    raises("fixing a slot the space does not have raises", lambda: RM.DesignSpace(orig, fixed={"LE.dx9": 0.0}))
    one = RM.section_at(base, orig, secs[17].y, 200, R_TE, index=18)
    check("section_at gives the section modified_sections builds there", np.abs(one.selig - secs[17].selig).max(), 0.0, 1e-12)
    check("design_problems: none for the baseline", len(RM.design_problems(base, orig, 50, R_TE)), 0, 0)
    bad = base.copy()
    bad.curves["LE"].v[-1] = 400.0
    bad2 = base.copy()
    bad2.curves["P1z"].v[-1] = -0.01
    bad3 = base.copy()
    bad3.curves["P2x"].v[-1] = 0.9
    bad3.curves["P3x"].v[-1] = -0.6
    pr, pr2, pr3 = (" ".join(RM.design_problems(b, orig, 50, R_TE)) for b in (bad, bad2, bad3))
    check("design_problems: an LE behind the TE, P1 below the chord, P3 far ahead of P2",
          ["LE is at or behind the TE" in pr, "crosses the chord line" in pr2, "fold the half-section back" in pr3],
          [1, 1, 1], 0)
    raises("modified_sections names the section that folds back", lambda: RM.modified_sections(bad3, orig, 50))
    yy = np.linspace(0.0, Z_OPT, 33)
    up = RM.refit_curve(base, orig, "P3z", 6)
    down = RM.refit_curve(base, orig, "P3z", 1)
    check("refit_curve: order 6 is the same curve, order 1 still pinned at H",
          [np.abs(up.curve("P3z").at(yy)[:, 1] - base.curve("P3z").at(yy)[:, 1]).max(), up.curves["P3z"].order,
           down.curve("P3z")([0.0])[0, 1] - orig.value("P3z", Z_OPT)[0]], [0, 6, 0], 1e-12)

    d6, o6, err = RM.change_degree(base, orig, 6)
    s6 = RM.modified_sections(d6, o6, 50, 200, R_TE)
    check("change_degree 4 -> 6 (two points added): the same sections, 11 curves",
          [max(np.abs(a.selig - b.selig).max() for a, b in zip(s6, secs)), err, len(d6.names), d6.degree],
          [0, 0, 11, 6], 1e-11)
    check("the original at degree 6 is the same shape: C0 at H of every curve on it",
          max(abs(d6.curve(c)([0.0])[0, 1] - o6.value(c, Z_OPT)[0]) for c in o6.names), 0.0, 1e-12)
    dm, om, errm = RM.change_degree(pd, orig, 5)
    sm = RM.modified_sections(dm, om, 50, 200, R_TE)
    check("change_degree of a changed design (G0 on P2x): the same sections, raising is exact (mm)",
          [max(np.abs(a.selig - b.selig).max() for a, b in zip(sm, spd)), errm], 0.0, 1e-11)
    d4, o4, _ = RM.change_degree(d6, o6, 4)
    check("and back to 4: the thickness format again", all(s.params() is not None for s in
                                                          RM.modified_sections(d4, o4, 50, 200, R_TE)), 1, 0)
    sp6 = RM.DesignSpace(o6)
    rec6 = sp6.to_dict(d6)
    sp6b, d6b = RM.DesignSpace.from_dict(orig, rec6)
    check("a degree-6 space: 45 slots, read back from its record through the degree-4 original",
          [len(sp6), sp6b.orig.degree, np.abs(sp6b.to_vector(d6b) - sp6.to_vector(d6)).max()], [45, 6, 0], 1e-12)
    raises("pin refuses a design of another degree", lambda: RM.pin(d6, orig))
    return orig, base, secs


def test_xcad_and_writers(p, orig, base, secs):
    print("\nXCAD file and writers (synthetic rudder)")
    loops = RM.read_xcad(p["xcad"])
    with tempfile.TemporaryDirectory() as tmp:
        again = os.path.join(tmp, "again.dat")
        RM.write_xcad_lines(again, [RM.xcad_lines(RM.xcad_to_working(pts, "m"), "m") for _, pts in loops])
        check("read_xcad -> working frame -> write: the same bytes",
              open(again, "rb").read() == open(p["xcad"], "rb").read(), 1, 0)
        lp = secs[-1].loop()
        check("loop: LE first and last, TE at index 99, at height y",
              [*(lp[0] - lp[-1]), lp[99, 0] - secs[-1].x_te, np.ptp(lp[:, 1]), lp[0, 0] - secs[-1].x_le],
              0.0, 1e-12)
        res = RM.write_case(tmp, "t", base, orig, p["xcad"], 50, 200, R_TE, plot=False, h_min=20.0)
        low = RM.baseline_design(orig, 15.0)
        raises("write_case refuses an H below h_min",
               lambda: RM.write_case(tmp, "low", low, orig, p["xcad"], 10, 200, R_TE, plot=False, h_min=20.0))
        raises("write_case refuses an H above h_max",
               lambda: RM.write_case(tmp, "high", base, orig, p["xcad"], 10, 200, R_TE, plot=False, h_max=60.0))
        k_top = RM.kept_loops(loops, 149.0, 5.0)
        check("kept_loops: horizontal loops from H + gap up, the tilted cap loops from H up",
              [len(k_top), all(np.ptp(loops[k][1][:, 2]) > 0 for k in k_top)], [2, 1], 0)
        out = RM.read_xcad(res["paths"]["xcad"])
        gap = Z_OPT / 49
        kept = [k for k, (_, pts) in enumerate(loops) if pts[:, 2].min() * 1e3 >= Z_OPT + gap - 1e-9]
        check("combined: 50 modified loops, then the loops from H + gap up",
              [len(out), res["xcad"]["kept"][0]], [50 + len(kept), kept[0] + 1], 0)
        check("combined: the kept loops are the original lines, cap loops too",
              all(out[50 + j][0] == loops[k][0] for j, k in enumerate(kept)), 1, 0)
        top = [ln for ln in loops if abs(ln[1][0, 2] * 1e3 - Z_OPT) < 1e-9][0]
        check("combined: the loop at H is the original one there (to print precision)",
              np.abs(out[49][1] - top[1]).max(), 0.0, 1.01e-8)
        raw = open(res["paths"]["xcad"], "rb").read()
        check("combined: CRLF line ends, #k markers, no negative zeros",
              [raw.count(b"\n") - raw.count(b"\r\n"), raw.count(b"#"), raw.count(b"-0.00000000 ")],
              [0, len(out), 0], 0)
        s2, prm = RF.read_thickness_params(res["paths"]["params"])
        check("thickness-format file: rows read back (7 decimals)",
              np.abs(prm - np.array([s.params() for s in secs])).max(), 0.0, 5.1e-8)
        tab = np.loadtxt(res["paths"]["table"])
        check("sections table: 50 rows x 17 columns, y, chord and P3z as built",
              [*tab.shape, np.abs(tab[:, 1] - [s.y for s in secs]).max(), np.abs(tab[:, 4] - [s.chord for s in secs]).max(),
               np.abs(tab[:, 16] - [s.ctrl[3, 1] for s in secs]).max()], [50, 17, 0, 0, 0], 1e-4)
        import json
        rec = json.load(open(res["paths"]["json"]))
        check("design.json records h_min, the TE radius and the degree",
              [rec["settings"]["h_min_mm"], rec["settings"]["te_radius_mm"], rec["settings"]["degree"]],
              [20.0, R_TE, 4], 0.0)
        back = RM.modified_sections(RM.design_from_record(rec["design"]), orig, 50, 200, R_TE)
        check("design.json: the design read back gives the same sections",
              max(np.abs(a.selig - b.selig).max() for a, b in zip(back, secs)), 0.0, 1e-12)
        old = {"H_mm": Z_OPT, "curves": {c: {"s": list(base.curves[n].s), "value": list(base.curves[n].v),
                                               "weights": [1.0] * 4, "join": "G1"}
                                           for c, n in (("LE", "LE"), ("T1", "P1z"), ("X1", "P2x"), ("X2", "P3x"),
                                                        ("T2", "P3z"))}}
        back = RM.modified_sections(RM.design_from_record(old), orig, 50, 200, R_TE)
        check("a record of the earlier layout (T1, X1, X2, T2) is read as its quartic",
              max(np.abs(a.selig - b.selig).max() for a, b in zip(back, secs)), 0.0, 1e-12)
        d5, o5, _ = RM.change_degree(base, orig, 5)
        d5.curves["P2z"].v[-1] += 0.005
        r5 = RM.write_case(tmp, "q", d5, o5, p["xcad"], 20, 200, R_TE, plot=True)
        t5 = np.loadtxt(r5["paths"]["table"])
        check("degree 5: no thickness format, the table has P1x .. P4z, the check plot is written",
              ["params" in r5["paths"], t5.shape[1], os.path.exists(r5["paths"]["png"])], [0, 19, 1], 0)
        mm = RM.write_modified_xcad(os.path.join(tmp, "mm.dat"), secs, p["xcad"], Z_OPT, units="mm")
        o_mm = RM.read_xcad(mm["path"])
        check("units mm: 1000 x the metre file", np.abs(o_mm[60][1] - out[60][1] * 1e3).max(), 0.0, 1e-5)
    table = RM.load_section_table(p["table"])
    check("load_section_table reads the columns back", [len(table), len(table["y_mm"])],
          [len(RM.SECTION_TABLE_COLUMNS), 50], 0)
    stack = RM.read_stack(p["stack"])
    x_le, xy = RM.original_section_at(stack, 40.0)
    check("read_stack root first; original_section_at on a station",
          [stack[0].y, stack[-1].y, x_le - x_le_true(40.0), np.abs(xy - true_section(40.0)[4]).max()],
          [0.0, 150.0, 0.0, 0.0], 5e-7)
    g = RM.distance_to_original(secs[20], stack)
    check("distance to the original: the baseline on it (to the stack's interpolation)", g.max(), 0.0, 5e-4)


def test_real():
    print("\nthe real rudder (outputs/ of the extraction)")
    paths = {k: os.path.join(HERE, v) for k, v in REAL.items()}
    missing = [v for k, v in REAL.items() if not os.path.exists(paths[k])]
    if missing:
        print("       skipped, not here: " + ", ".join(missing))
        return
    orig = RM.fit_original(paths["corner"], paths["table"], 80.0)
    check("fits to the original: LE (mm) and section coordinates (fractions)",
          [np.abs(orig.residuals["LE"]).max(), max(np.abs(orig.residuals[c]).max() for c in orig.names[1:])],
          [0, 0], 5e-6)
    base = RM.baseline_design(orig, 80.0)
    secs = RM.modified_sections(base, orig, 50, 200, 0.75)
    table = RM.load_section_table(paths["table"])
    prm = np.array([s.params() for s in secs])
    check("baseline: X1, X2, T1, T2 at the table's heights = the section fits",
          max(np.abs(prm[:, k] - table[c]).max() for k, c in enumerate(("X1", "X2", "T1", "T2"))), 0.0, 2e-6)
    stack = RM.read_stack(paths["stack"])
    g = max(RM.distance_to_original(s, stack).max() for s in secs)
    print(f"       replica error of the baseline: largest distance {g * 1e3:.1f} um")
    check("baseline: within 0.2 mm of the original sections everywhere", g < 0.2, 1, 0)
    d5, o5, _ = RM.change_degree(base, orig, 5)
    s5 = RM.modified_sections(d5, o5, 50, 200, 0.75)
    check("degree 5 (one section control point added): the same sections (mm)",
          max(np.abs(a.selig - b.selig).max() for a, b in zip(s5, secs)), 0.0, 1e-11)
    loops = RM.read_xcad(paths["xcad"])
    with tempfile.TemporaryDirectory() as tmp:
        again = os.path.join(tmp, "again.dat")
        RM.write_xcad_lines(again, [RM.xcad_lines(RM.xcad_to_working(pts, "m"), "m") for _, pts in loops])
        check("extraction's XCAD file: read, write again, the same bytes",
              open(again, "rb").read() == open(paths["xcad"], "rb").read(), 1, 0)
    flat = [s for s in stack if s.tilt_deg == 0.0]
    dev = 0.0
    for k, s in enumerate(flat):
        mine = (RM.xcad_loop(s.xy, s.x_le, s.y) @ RM.XCAD_FRAME.T) * 1e-3
        dev = max(dev, np.abs(mine - loops[k][1]).max())
    check("stack sections as XCAD loops = the extraction's loops (m)", dev, 0.0, 1.01e-8)
    full = os.path.join(HERE, "outputs/thickness_params_y0_y200_info.dat")
    if os.path.exists(full):
        of = RM.fit_original(paths["corner"], full)
        top_sec = max(s.y for s in stack if s.tilt_deg == 0.0)
        sf = RM.modified_sections(RM.baseline_design(of, of.z_top), of, 60, 200, 0.75)
        gf = max(RM.distance_to_original(s, stack).max() for s in sf)
        with tempfile.TemporaryDirectory() as tmp:
            xf = RM.write_modified_xcad(os.path.join(tmp, "x.dat"), sf, paths["xcad"], of.z_top)
        print(f"       whole stack: H = z_top = {of.z_top:.4f} mm, largest distance {gf * 1e3:.1f} um")
        check("whole-stack original: z_top the top horizontal section; H = z_top within 0.2 mm; tip and cap kept",
              [of.z_top - top_sec, gf < 0.2, len(xf["kept"])], [0, 1, 11], 1e-6)   # the stack header has 6 decimals
    else:
        print("       (outputs/thickness_params_y0_y200_info.dat not here: whole-stack check skipped)")
    rows = RM.fit_section_table(paths["stack"], 0.5, verbose=False)
    check("fit_section_table: the root section as in the table",
          [rows[0][c] - table[c][0] for c in ("X1", "X2", "T1", "T2", "r")], 0.0, 1e-6)


def main():
    print("rudder_modify self-check")
    test_bezier()
    test_fits()
    test_replica()
    test_bezier_section()
    with tempfile.TemporaryDirectory() as tmp:
        p = write_synthetic(tmp)
        orig, base, secs = test_original_and_designs(p)
        test_xcad_and_writers(p, orig, base, secs)
    test_real()
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
