"""
_hull_modify_test.py

Self-check for hull_modify.py: the CST curves (joins, ends, slope), the fit (a curve finds itself again;
SUBOFF from its equations), an elliptical body (an ellipsoid), the design space, the files, and with OCC
(pythonocc-core or cadquery-ocp) the CAD (SUBOFF's fit, an elliptical body, a flat end, a pointed tail).

    python _hull_modify_test.py
"""

import os
import sys
import tempfile

import numpy as np

import hull_extraction as HE
import hull_modify as HM
import suboff as S
import xcad_loft as XL

_FAILED = []


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<66s} max error {err:.3e}   (tol {tol:.0e})")


def ellipsoid_params(a=1.5, b=0.3, c=0.2, n=301):
    """An ellipsoid of semi-axes a (x), b (y), c (z) as a parameter file: nose and tail halves, no middle body."""
    t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    x = 2.0 * a * t
    f = np.sqrt(np.clip(1.0 - ((x - a) / a) ** 2, 0.0, None))
    return HM.HullParams(a, 0.0, a, 0.0, x, c * f, b * f, False)


def test_curves():
    print("\nCST curves")
    rng = np.random.default_rng(3)
    errs = []
    for kind, e in (("nose", 0.5), ("nose", 1.0), ("tail", 2.0), ("tail", 0.7)):
        c = HM.CSTCurve(kind, e, rng.uniform(0.5, 2.0, 4))
        join, end = (1.0, 0.0) if kind == "nose" else (0.0, 1.0)
        errs += [c.shape(join)[0] - 1.0, c.dshape(join)[0], c.shape(end)[0]]
    check("join r = r* with zero slope, the nose tip and tail end at 0 (normalized)", errs, 0.0, 1e-12)
    c = HM.CSTCurve("tail", 1.7, [1.2, 0.9, 1.4, 0.8])
    u, h = np.linspace(0.05, 0.95, 37), 1e-7
    fd = (c.shape(u + h) - c.shape(u - h)) / (2 * h)
    back = HM.CSTCurve.from_keys("tail", {k: c.get(k) for k in c.keys()}, c.order)
    check("slope against central differences; keys round trip", [np.abs(fd - c.dshape(u)).max(),
          np.abs(back.coeffs() - c.coeffs()).max()], 0.0, 1e-6)


def test_fit():
    print("\nfit")
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 400)))
    out = []
    for kind, e, free in (("nose", 0.47, [1.9, 1.4, 1.2, 1.25]), ("tail", 1.9, [2.1, 2.7, 3.3, 5.6])):
        c = HM.CSTCurve(kind, e, free)
        f, rms = HM.fit_cst(kind, psi, c.shape(psi), c.order)
        out += [f.exponent - e, np.abs(f.coeffs() - c.coeffs()).max() / 10.0, rms]
    check("a curve finds itself again (exponent, coefficients / 10, residual)", out, 0.0, 1e-5)
    p = HM.suboff_params()
    d = HM.fit_design(p)
    x = np.linspace(0.0, d.joins[3], 20001)
    xc = np.linspace(d.joins[3], d.joins[4], 2001)
    rep = HM.fit_report(p, d)
    check("SUBOFF, order 5: nose and tail against the equations (mm, within 0.15)",
          [max(0.0, 1000 * np.abs(d.radius("r", x) - S.radius(x)).max() - 0.15),
           1000 * np.abs(d.radius("r", xc) - S.radius(xc)).max(), max(0.0, 1000 * max(rep.values()) - 0.15)], 0.0, 1e-2)
    check("SUBOFF: N1 = 1/2.1 (the equations' exponent), N2 = 2 (zero slope at the cap)",
          [d.curves["nose_r"].exponent - 1 / 2.1, (d.curves["tail_r"].exponent - 2.0) / 25.0], 0.0, 2e-3)
    ex, hs = S.exact_hydrostatics(), HM.hydrostatics(d)
    check("SUBOFF fit: V, S (fractions), LCB (fraction of L) against the equations",
          [hs["volume"] / ex["volume"] - 1, hs["wetted_surface"] / ex["wetted_surface"] - 1,
           (hs["lcb"] - ex["lcb"]) / d.length], 0.0, 1e-5)
    d6 = d.copy()
    d6.curves["nose_r"] = HM.refit_curve(d, "nose_r", 7)
    d6.curves["tail_r"] = HM.refit_curve(d, "tail_r", 4)
    check("refit to another order keeps the shape (nose 7, tail 4; mm)",
          1000 * np.abs(d6.radius("r", x) - d.radius("r", x)).max(), 0.0, 0.5)
    return p, d


def test_ellipsoid():
    print("\nan ellipsoid (semi-axes 1.5, 0.3, 0.2 m): r and r' curves of their own, no middle body")
    a, b, c = 1.5, 0.3, 0.2
    p = ellipsoid_params(a, b, c)
    d = HM.fit_design(p, {k: 6 for k in ("nose_r", "tail_r", "nose_rp", "tail_rp")})
    rep = HM.fit_report(p, d)
    hs = HM.hydrostatics(d)
    check("r and r' within 0.1 mm of the ellipsoid; V, LCB (fractions)",
          [1000 * max(rep.values()) / 100.0, hs["volume"] / (4 / 3 * np.pi * a * b * c) - 1,
           (hs["lcb"] - a) / (2 * a)], 0.0, 1e-3)
    check("sections are ellipses: area pi r r', breadth 2 r', depth 2 r; Cp 2/3",
          [np.abs(hs["am"] / (np.pi * b * c) - 1), hs["breadth"] - 2 * b, hs["depth"] - 2 * c, hs["cp"] - 2 / 3],
          0.0, 1e-4)
    return d


def test_space(d):
    print("\nthe design space")
    sp = HM.HullSpace(d)
    x0 = sp.to_vector(d)
    d2 = sp.to_design(x0)
    x = np.linspace(0.0, d.length, 2001)
    inside = np.all((x0 >= sp.bounds[:, 0]) & (x0 <= sp.bounds[:, 1]))
    check("SUBOFF: 13 free slots (lengths, curves), vector round trip, the fit inside its bounds",
          [len(sp), np.abs(d2.radius("r", x) - d.radius("r", x)).max(), inside], [13, 0, 1], 0)
    sp2, d3 = HM.HullSpace.from_dict(sp.to_dict(d))
    check("to_dict then from_dict: the same slots, values, bounds and free flags",
          [sp2.names == sp.names, np.abs(sp2.bounds - sp.bounds).max(), np.abs(sp2.to_vector(d3) - x0).max(),
           sp2.fixed == sp.fixed], [1, 0, 0, 1], 0)
    e = test_ellipsoid.d
    se = HM.HullSpace(e, free=("middle", "nose_rp", "tail_rp", "rp"))
    check("elliptic: r' slots of its own; free by name (middle, r', its two curves)",
          [len(se), "rp" in se.names, "nose_r.N1" in se.names], [1 + 1 + 2 * 6, 1, 0], 0)


def test_files(p, d):
    print("\nfiles")
    with tempfile.TemporaryDirectory() as tmp:
        pa = HM.write_params(os.path.join(tmp, "suboff.dat"), p)
        q = HM.read_params(pa)
        e = test_ellipsoid.d
        pe = HM.write_params(os.path.join(tmp, "ell.dat"), HM.design_params(e))
        qe = HM.read_params(pe)
        check("parameter files read back: lengths, table (9 decimals), axisymmetric or not",
              [abs(q.middle - p.middle), np.abs(q.r - p.r).max(), q.axisymmetric, qe.axisymmetric,
               np.abs(qe.rp - HM.design_params(e).rp).max()], [0, 0, 1, 0, 0], 1e-8)
        files, hs = HM.write_case(tmp, "t", d, p, HM.HullSpace(d), cad=False)
        back = HM.read_design(files["design"])
        x = np.linspace(0.0, d.length, 2001)
        tab = np.loadtxt(files["stations"])
        pts = [ln for ln in open(files["sections"]) if ln.startswith("#")]
        sp, d2, rec = HM.load_hull_space(files["space"])
        check("write_case: design (exact), stations, sections, set-up, plot",
              [np.abs(back.radius("r", x) - d.radius("r", x)).max(), tab.shape[1], len(pts) > 50,
               np.abs(sp.to_vector(d2) - HM.HullSpace(d).to_vector(d)).max(), os.path.exists(files["plot"])],
              [0, 8, 1, 0, 1], 0)


def test_cad(d):
    print("\nthe CAD (OCC)")
    if not XL.occ_available():
        print("       skipped: no OCC binding (pythonocc-core or cadquery-ocp)")
        return
    out = []
    for name, des in (("suboff", d), ("ellipsoid", test_ellipsoid.d)):
        sol = HM.build_solid(des)
        v, s, c = HE.solid_properties(sol)
        hs = HM.hydrostatics(des)
        out.append((name, HE.is_valid(sol), XL.count_faces(sol), v / hs["volume"] - 1, (c[0] - hs["lcb"]) / des.length))
    check("solids valid, SUBOFF 4 faces, ellipsoid 2; V and LCB as the stations",
          [out[0][1], out[0][2], out[1][1], out[1][2], out[0][3] * 1e3, out[0][4] * 1e3, out[1][3] * 1e3,
           out[1][4] * 1e3], [1, 4, 1, 2, 0, 0, 0, 0], 5e-3)
    e = test_ellipsoid.d
    with tempfile.TemporaryDirectory() as tmp:
        path = XL.write_step(HM.build_solid(e), os.path.join(tmp, "e.step"))
        st = HE.stations_from_cad(HE.load_cad(path), 21)
    x = st.x[1:-1]
    check("cuts of the elliptical body: breadth 2 r', depth 2 r (m)",
          [np.abs(st.breadth[1:-1] / 2 - e.radius("rp", x)).max(), np.abs(st.depth[1:-1] / 2 - e.radius("r", x)).max()],
          0.0, 2e-5)
    flat = d.copy()
    flat.cap = 0.0
    pointed = d.copy()
    pointed.cap, pointed.re = 0.0, 0.0
    pointed.curves["tail_r"] = HM.CSTCurve("tail", 1.0, [1.2, 1.1, 1.0, 1.0])
    res = []
    for des in (flat, pointed):
        sol = HM.build_solid(des)
        res += [HE.is_valid(sol), HE.solid_properties(sol)[0] / HM.hydrostatics(des)["volume"] - 1]
    check("a flat tail end and a pointed tail: valid solids, V as the stations",
          [res[0], res[2], res[1] * 1e3, res[3] * 1e3, len(HM.design_problems(pointed))], [1, 1, 0, 0, 0], 5e-3)


def main():
    print("hull model self-check")
    test_curves()
    p, d = test_fit()
    test_ellipsoid.d = test_ellipsoid()
    test_space(d)
    test_files(p, d)
    test_cad(d)
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
