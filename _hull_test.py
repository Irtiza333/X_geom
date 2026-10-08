"""
_hull_test.py

Self-check for suboff.py and hull_extraction.py: the SUBOFF equations (ends,
joins, slope), its hydrostatics against the report, the hydrostatics from
stations against exact values (SUBOFF, an ellipsoid of revolution), the files,
and with OCC (pythonocc-core or cadquery-ocp) the solid of revolution, its STEP
and the stations cut from it.

    python _hull_test.py
"""

import os
import sys
import tempfile

import numpy as np

import hull_extraction as HE
import suboff as SB
import xcad_loft as XL

_FAILED = []


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<66s} max error {err:.3e}   (tol {tol:.0e})")


def test_equations():
    print("\nSUBOFF equations")
    check("length 14.291667 ft and diameter 5/3 ft (4.3561 m, 0.508 m)", [SB.LENGTH, SB.DIAMETER],
          [4.3561, 0.508], 1e-6)
    check("nose and tail on the axis, the middle body at Rmax (m)",
          [SB.radius(0.0), SB.radius(SB.LENGTH), SB.radius(2.0)], [0.0, 0.0, 0.254], 1e-12)
    e = 1e-9
    jumps = [abs(SB.radius(j - e) - SB.radius(j + e)) for j in SB.JOINS[1:4]]
    slopes = [SB.slope(j + s * e) for j in SB.JOINS[1:4] for s in (-1, 1)]
    check("the three joins: radius continuous (m), slope 0 on both sides", jumps + slopes, 0.0, 1e-6)
    xs, h = np.linspace(0.001, SB.LENGTH - 0.001, 2001), 1e-6
    fd = (SB.radius(xs + h) - SB.radius(xs - h)) / (2.0 * h)
    check("slope: the radius's derivative (central differences)", np.abs(fd - SB.slope(xs)).max(), 0.0, 1e-5)
    check("slope at the ends: +inf at the nose, -inf at the tail",
          [SB.slope(0.0) == np.inf, SB.slope(SB.LENGTH) == -np.inf], [1, 1], 0)


def test_hydrostatics():
    print("\nSUBOFF hydrostatics")
    ex = SB.exact_hydrostatics()
    check("the equations against the report (bare hull: 0.699 m^3, 5.988 m^2)",
          [ex["volume"], ex["wetted_surface"]], [0.699, 5.988], 5e-4)
    x, r = SB.stations(401)
    hs = HE.hydrostatics(HE.Stations.circles(x, r, length=SB.LENGTH))
    check(f"from {len(x)} stations: V, S (fractions), LCB (fraction of L) against the quadrature",
          [hs["volume"] / ex["volume"] - 1.0, hs["wetted_surface"] / ex["wetted_surface"] - 1.0,
           (hs["lcb"] - ex["lcb"]) / SB.LENGTH], 0.0, 2e-5)
    check("the parallel middle body from the stations alone: the joins, within 1 cm (m)",
          [hs["pmb_start"], hs["pmb_end"]], [SB.JOINS[1], SB.JOINS[2]], 1e-2)
    half, mid = 0.5 * hs["length"], 0.5 * hs["length"]
    vf, va = hs["cp_fore"] * hs["am"] * half, hs["cp_aft"] * hs["am"] * half
    moment = (va * hs["xc_aft"] - vf * hs["xc_fore"]) * half / (vf + va) - (hs["lcb"] - mid)
    check("Cm pi/4, Cb = Cp Cm, L/B 8.575; the halves add up to Cp and the LCB",
          [hs["cm"], hs["cb"] - hs["cp"] * hs["cm"], hs["l_over_b"], 0.5 * (hs["cp_fore"] + hs["cp_aft"]) - hs["cp"],
           moment], [np.pi / 4.0, 0.0, 8.575, 0.0, 0.0], 1e-6)


def test_ellipsoid():
    print("\nan ellipsoid of revolution (semi-axes 2 m and 0.25 m), from 801 stations")
    a, b = 2.0, 0.25
    x = a * (1.0 - np.cos(np.linspace(0.0, np.pi, 801)))
    r = b * np.sqrt(np.clip(1.0 - ((x - a) / a) ** 2, 0.0, None))
    hs = HE.hydrostatics(HE.Stations.circles(x, r))
    e = np.sqrt(1.0 - (b / a) ** 2)
    vol, area = 4.0 / 3.0 * np.pi * a * b * b, 2.0 * np.pi * b * b * (1.0 + a / (b * e) * np.arcsin(e))
    check("V, S (fractions), LCB at the middle (fraction of L)",
          [hs["volume"] / vol - 1.0, hs["wetted_surface"] / area - 1.0, (hs["lcb"] - a) / (2.0 * a)], 0.0, 1e-4)
    check("Cp 2/3; each half Cp 2/3, centroid 3/8 of L/2 from midship, no parallel middle body",
          [hs["cp"], hs["cp_fore"], hs["cp_aft"], hs["xc_fore"], hs["xc_aft"], hs["p_fore"], hs["p_aft"]],
          [2 / 3, 2 / 3, 2 / 3, 3 / 8, 3 / 8, 0.0, 0.0], 1e-4)


def test_files(tmp):
    print("\nfiles")
    hs, files, _ = HE.run_suboff(tmp, n=101, cad=False)
    tab = np.loadtxt(files[0])
    back = HE.read_hydrostatics(files[1])
    err = max(abs(back[k] - hs[k]) / max(1.0, abs(hs[k])) for k in back)
    check("stations (8 columns, x/L 0 to 1, A/Am to 1), hydrostatics read back, the plot",
          [tab.shape[1], tab[0, 1], tab[-1, 1], tab[:, 3].max(), len(back) == len(HE.HYDRO), err,
           os.path.exists(files[2])], [8, 0, 1, 1, 1, 0, 1], 1e-9)


def test_cad(tmp):
    print("\nthe CAD (OCC)")
    if not XL.occ_available():
        print("       skipped: no OCC binding (pythonocc-core or cadquery-ocp)")
        return
    ex = SB.exact_hydrostatics()
    solid = HE.revolve(*SB.profile())
    vol, area, c = HE.solid_properties(solid)
    check("the solid of revolution: valid, 4 faces; V, S (fractions), LCB (of L) as the quadrature",
          [HE.is_valid(solid), XL.count_faces(solid), vol / ex["volume"] - 1.0, area / ex["wetted_surface"] - 1.0,
           (c[0] - ex["lcb"]) / SB.LENGTH], [1, 4, 0, 0, 0], 1e-7)
    path = XL.write_step(solid, os.path.join(tmp, "suboff.step"))
    shape = HE.load_cad(path)
    st = HE.stations_from_cad(shape, 41)
    rr = SB.radius(st.x)
    rnd = max(np.ptp(np.hypot(*lp.T)) / np.mean(np.hypot(*lp.T)) for lp in st.loops)
    check("STEP read back (V); 41 stations cut from it: radius as the equations, round (fractions)",
          [HE.solid_properties(shape)[0] / vol - 1.0, np.abs(st.r[1:-1] / rr[1:-1] - 1.0).max(), rnd], 0.0, 2e-6)
    hs, _ = HE.run_cad(path, tmp, n=101, plot=False)
    check("the STEP's hydrostatics: V, S, LCB from the solid; Cp, the halves from 101 stations",
          [hs["volume"] / ex["volume"] - 1.0, hs["wetted_surface"] / ex["wetted_surface"] - 1.0,
           (hs["lcb"] - ex["lcb"]) / SB.LENGTH, hs["cp"] - 0.7919413, hs["cp_fore"] - 0.8687015,
           hs["cp_aft"] - 0.7151649], 0.0, 2e-4)


def main():
    print("hull self-check")
    test_equations()
    test_hydrostatics()
    test_ellipsoid()
    with tempfile.TemporaryDirectory() as tmp:
        test_files(tmp)
        test_cad(tmp)
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
