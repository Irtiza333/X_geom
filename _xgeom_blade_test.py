"""
_xgeom_blade_test.py

Self-check for the propeller blade in XGeom: the parameter file and its fitted
curves of 1 and 2 segments (blade_modify.py), the blade radius, hub radius and
hub height, the geometry through the MSc code (para.py, blade_surface_new.py),
the design tool's blade adapter (xgeom_blade.py) and, with --gui, the window;
with --cad the DRDC CAD (IGES and STEP, pythonocc-core, about two minutes).

    python _xgeom_blade_test.py                 file, curves, design space, geometry, adapter
    python _xgeom_blade_test.py --gui           also the window
    python _xgeom_blade_test.py --cad           also the CAD
    python _xgeom_blade_test.py --section-table PATH
                                                another section table than airfoil_data_fixed.csv

The geometry and adapter checks need the section table and are skipped without it.
"""

import json
import os
import sys
import tempfile

import numpy as np

import blade_modify as BM
import rudder_modify as RM

_FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))
ONE = {c: 1 for c in BM.CURVES}                      # every curve 1 segment


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<66s} max error {err:.3e}   (tol {tol:.0e})")


def msc_file(tmp, table=None):
    """The MSc blade's parameter file, with another section table if given."""
    p = BM.msc_params()
    if table:
        p.section_table = os.path.abspath(table)
    return BM.write_params(os.path.join(tmp, "msc.dat"), p)


def refused(fn, *args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1


def test_file(tmp):
    print("\nthe parameter file")
    import x_blade_new as XB
    path = msc_file(tmp)
    p = BM.read_params(path)
    p0 = BM.msc_params()
    check("written and read back: globals (hub height too) and the table (8 decimals)",
          [p.diameter - 1.4, p.blades - 5, p.root_r - XB.R_VALUES[0], p.hub_height - BM.HUB_HEIGHT,
           p.section_table == "airfoil_data_fixed.csv",
           max(np.abs(p.values[c] - p0.values[c]).max() for c in BM.CURVES), np.abs(p.r - p0.r).max()],
          [0, 0, 0, 0, 1, 0, 0], 5e-9)
    rf = np.linspace(XB.R_VALUES[0], 1.0, 5001)
    fns = {"pitch": XB.BASE_PITCH, "chord": XB.BASE_CHORD, "thickness": XB.BASE_MAX_THICKNESS,
           "camber": XB.BASE_MAX_CAMBER, "skew": XB.BASE_SKEW_ANGLE, "rake": XB.BASE_RAKE}
    check("the file's splines follow the MSc polynomials (fraction of each range)",
          [np.abs(p.spline(c)(rf) - fns[c](rf)).max() / np.ptp(fns[c](rf)) for c in BM.CURVES], 0.0, 1e-5)
    q = BM.read_params(path)
    q.root_r, q.hub_height = 0.2, None
    BM.write_params(os.path.join(tmp, "root02.dat"), q)
    q2 = BM.read_params(os.path.join(tmp, "root02.dat"))
    r, v = q2.rows()
    check("a root between two rows gets a row of its own (value from the spline); no hub height: None",
          [r[0], r[1] > 0.2, v["pitch"][0] - fns["pitch"](0.2), q2.hub_height is None,
           BM.fit_design(q2, segments=ONE)[0].hub_height], [0.2, 1, 0, 1, BM.HUB_HEIGHT], 1e-5)
    bad = os.path.join(tmp, "bad.dat")
    out = []
    for text in ("# diameter_m 1.4\n# blades 5\n0.2 1 2 3 4 5 6\n",
                 "# diameter_m 1.4\n# blades 5\n# section_table x.csv\n0.2 1 2 3\n0.3 1 2 3\n0.4 1 2 3\n0.5 1 2 3\n"):
        with open(bad, "w") as fh:
            fh.write(text)
        out.append(refused(BM.read_params, bad))
    check("a file without the section table, or with 4 columns, is refused", out, [1, 1], 0)
    q = BM.read_params(path)
    q.section_table, q.airfoil = "", "NACA 2416"
    pa = BM.write_params(os.path.join(tmp, "naca.dat"), q)
    qa = BM.read_params(pa)
    tab = np.loadtxt(qa.section_path(), delimiter=",", skiprows=1)
    pts, _ = BM.blade_points(BM.fit_design(qa, segments=ONE)[0], qa)
    out = []
    for head in ("# section_table airfoil_data_fixed.csv\n# airfoil naca2412\n", "# airfoil nosuchfoil\n"):
        with open(bad, "w") as fh:
            fh.write("# diameter_m 1.4\n# blades 5\n" + head + open(pa).read().split("# r/R", 1)[1].split("\n", 1)[1])
        out.append(refused(BM.read_params, bad))
    check("an airfoil by name (NACA 2416): read back, its section table (26 rows), the blade; both keys or an "
          "unknown airfoil refused", [qa.airfoil == "NACA 2416", qa.section_table == "", tab.shape[0],
                                      np.all(np.isfinite(pts))] + out, [1, 1, 26, 1, 1, 1], 0)


def test_curves(tmp):
    print("\nthe curves")
    u = np.array([0.0, 0.3, 0.55, 0.8, 1.0])
    v = np.array([1.2, 1.4, 0.9, 1.1, 0.6])
    d = BM.BladeDesign(0.7, 0.14, 1.0, 1.0, {c: BM.BladeCurve(u, v) for c in BM.CURVES})
    rr = np.linspace(0.2, 1.0, 41)
    cu, res = BM.fit_curve(rr, d.at("pitch", rr), 0.2, 1.0, 4)
    d2 = BM.BladeDesign(0.7, 0.14, 1.0, 1.0, {c: cu for c in BM.CURVES})
    rf = np.linspace(0.2, 1.0, 2001)
    check("1 segment: fit_curve finds a Bezier of its own order again",
          [d.r0 - 0.2, np.abs(res).max(), np.abs(d2.prop("pitch")(rf) - d.prop("pitch")(rf)).max()], 0.0, 1e-8)
    two = BM.TwoSegmentCurve(1.2, 0.3, 0.5, 0.45, 1.5, 0.6, 0.8, 0.7)
    d3 = BM.BladeDesign(0.7, 0.14, 1.0, 1.0, {c: two for c in BM.CURVES})
    rr = np.linspace(0.2, 1.0, 81)
    t2, res2 = BM.fit_two_segment(rr, d3.at("pitch", rr), 0.2, 1.0)
    d4 = BM.BladeDesign(0.7, 0.14, 1.0, 1.0, {c: t2 for c in BM.CURVES})
    check("2 segments: fit_two_segment finds a 2-segment curve again",
          [np.abs(res2).max(), np.abs(d4.prop("pitch")(rf) - d3.prop("pitch")(rf)).max()], 0.0, 1e-6)
    r4, h = two.radii(0.2, 1.0)[1], 1e-7
    p = two.ctrl(0.2, 1.0)
    check("2 segments: P2 = P3 and P5 = P6 at P4's value, zero slope at P4, ends at P1 and P7",
          [np.abs(p[1] - p[2]).max(), np.abs(p[4] - p[5]).max(), np.ptp(p[1:6, 1]),
           (d3.at("pitch", r4) - d3.at("pitch", r4 - h)) / h, (d3.at("pitch", r4 + h) - d3.at("pitch", r4)) / h,
           d3.at("pitch", 0.2) - 1.2, d3.at("pitch", 1.0) - 0.7], 0.0, 1e-6)
    p = BM.read_params(msc_file(tmp))
    for segs, tol in ((ONE, 0.01), (None, 0.08)):
        design, _ = BM.fit_design(p, segments=segs)
        rep = BM.fit_report(p, design)
        form = "1 segment, 6 control points" if segs else "2 segments"
        for c in BM.CURVES:
            print(f"       {c:9s} {form}: largest difference from the file {rep[c][0]:.2e} {BM.UNITS[c]} "
                  f"({100 * rep[c][1]:.2f} % of its range)")
        check(f"fit_design on the MSc file, {form}: each curve within {100 * tol:.0f} % of its range",
              [rep[c][1] for c in BM.CURVES], 0.0, tol)
    d1, _ = BM.fit_design(p, segments=ONE)
    rs = d1.r0 + rf * (d1.r1 - d1.r0)
    ref = d1.prop("chord")(rs)
    d7 = BM.refit_curve(d1, "chord", 7)
    check("refit_curve to 8 control points keeps the shape (c/D)",
          [d7.curves["chord"].order - 7, np.abs(d7.prop("chord")(rs) - ref).max()], 0.0, 5e-4)
    d2s = BM.refit_curve(d1, "chord", segments=2)
    check("refit_curve from 1 to 2 segments: close to the shape (fraction of the range)",
          [d2s.curves["chord"].segments - 2,
           np.abs(d2s.prop("chord")(rs) - ref).max() / np.ptp(ref)], 0.0, 0.02)
    design, _ = BM.fit_design(p)
    path = BM.write_params(os.path.join(tmp, "design.dat"), BM.design_params(design, p))
    q = BM.read_params(path)
    rr = np.linspace(design.r0, design.r1, 3001)
    check("the design (2 segments) written as a parameter file reproduces it (fraction of each range)",
          [np.abs(q.spline(c)(rr) - design.prop(c)(rr)).max() / np.ptp(design.prop(c)(rr)) for c in BM.CURVES],
          0.0, 2e-3)
    f = design.prop("skew")
    check("a property function: single values and arrays agree, linear beyond the ends",
          [np.abs(np.array([float(f(x)) for x in rr[::50]]) - f(rr[::50])).max(),
           float(f(1.01)) - (float(f(1.0)) + 0.01 * (float(f(1.0)) - float(f(1.0 - 1e-6))) / 1e-6)], 0.0, 1e-4)


def test_space(tmp):
    print("\nthe design space")
    p = BM.read_params(msc_file(tmp))
    d1, _ = BM.fit_design(p, {"pitch": 6}, ONE)
    d2, _ = BM.fit_design(p)
    for design, form, n_all, n_free in ((d1, "1 segment", 3 + 5 * 10 + 12, 12 + 10), (d2, "2 segments", 3 + 6 * 8, 16)):
        space = BM.BladeSpace(p, design, free=("pitch", "chord"))
        x0 = space.to_vector(design)
        dd = space.to_design(x0)
        check(f"{form}: slots (3 globals and the curves'), free pitch and chord",
              [len(space.all_names), len(space), all(n.split(".")[0] in ("pitch", "chord") for n in space.names)],
              [n_all, n_free, 1], 0)
        rr = np.linspace(design.r0, design.r1, 501)
        allv = space.values(design)
        inside = [space.all_bounds[i, 0] <= allv[n] <= space.all_bounds[i, 1] for i, n in enumerate(space.all_names)]
        check(f"{form}: design vector round trip; the fitted design inside its default bounds",
              [max(np.abs(dd.at(c, rr) - design.at(c, rr)).max() for c in BM.CURVES), dd.radius - design.radius,
               all(inside)], [0, 0, 1], 1e-14)
    space = BM.BladeSpace(p, d2, free=("pitch", "radius", "hub_height"))
    vals = space.values(d2)
    vals.update(radius=0.8, hub_height=1.2)
    d3 = space.design_of(vals)
    check("globals in the space: radius and hub height free, the hub radius held",
          [space.names[:2] == ["radius", "hub_height"], d3.radius, d3.hub_height, d3.hub_radius, d3.r0],
          [1, 0.8, 1.2, d2.hub_radius, d2.hub_radius / 0.8], 1e-12)
    rec = BM.space_record(space, d3, p)
    path = os.path.join(tmp, "space.json")
    with open(path, "w") as fh:
        json.dump(rec, fh)
    s3, d4, rec3 = BM.load_blade_space(path)
    check("space_record, then load_blade_space: the same variables, values, bounds, forms",
          [s3.names == space.names, np.abs(s3.to_vector(d4) - space.to_vector(d3)).max(),
           np.abs(s3.bounds - space.bounds).max(), rec3["geometry"] == "blade", s3.forms == space.forms],
          [1, 0, 0, 1, 1], 0)
    old = BM.BladeSpace(p, d1, free=("pitch",)).to_dict(d1)           # a set-up saved before the globals
    old.pop("forms")
    old["r0"] = d1.r0
    old["slots"] = [q for q in old["slots"] if q["name"] not in BM.GLOBAL_SLOTS]
    s5, d5 = BM.BladeSpace.from_dict(p, old)
    rr = np.linspace(d1.r0, d1.r1, 201)
    check("an older set-up (no globals, 1 segment): read, the globals held at the file's",
          [max(np.abs(d5.at(c, rr) - d1.at(c, rr)).max() for c in BM.CURVES), d5.radius - 0.7, d5.r0 - d1.r0,
           "radius" in s5.fixed, len(s5) - 12], [0, 0, 0, 1, 0], 1e-12)


def test_geometry(tmp, table):
    print("\nthe geometry (MSc code: para.py, BladeSurface)")
    if not os.path.exists(table):
        print(f"       skipped: no section table {table}")
        return False
    import x_blade_new as XB
    from para import para
    p = BM.read_params(msc_file(tmp, table))
    st = BM.r_stations(XB.R_VALUES[0])
    check(f"r_stations({XB.R_VALUES[0]:g}) are x_blade_new.R_VALUES",
          [st.shape == XB.R_VALUES.shape, np.abs(st - XB.R_VALUES).max() if st.shape == XB.R_VALUES.shape else 1],
          [1, 0], 0)
    props = tuple(p.spline(c) for c in ("camber", "pitch", "chord", "thickness", "skew", "rake"))
    pts = para(*props, st, 0, write_dat=False, d=p.diameter, airfoil_path=p.section_path())
    ref = para(XB.BASE_MAX_CAMBER, XB.BASE_PITCH, XB.BASE_CHORD, XB.BASE_MAX_THICKNESS, XB.BASE_SKEW_ANGLE,
               XB.BASE_RAKE, XB.R_VALUES, 0, write_dat=False, airfoil_path=p.section_path())
    check("para.py on the file's splines = para.py on the polynomials (m)", np.abs(pts - ref).max(), 0.0, 2e-5)
    from scipy.spatial import cKDTree
    from rot_axis import rot_axis
    p2 = ref @ rot_axis(np.array([1, 0, 0]), np.deg2rad(72))
    gap_xb = min(float(np.min(cKDTree(p2).query(ref, k=1)[0])), float(np.min(cKDTree(ref).query(p2, k=1)[0])))
    check("clearance = X_blade's check (m)", BM.clearance(ref, 5) - gap_xb, 0.0, 1e-15)
    design, _ = BM.fit_design(p)
    bs = BM.blade_surface(design, p)
    eta = np.linspace(bs.eta_min, 1.0, 40)
    xi = np.linspace(0.0, 0.5, 40)
    check("BladeSurface of the design: TE and tip closed (m)",
          [np.abs(bs.b(0.0, eta) - bs.b(1.0, eta)).max(), np.abs(bs.b(xi, 1.0) - bs.b(1.0 - xi, 1.0)).max()], 0.0, 1e-12)
    probs = BM.design_problems(design, p)
    bad = design.copy()
    bad.curves["chord"] = bad.curves["chord"].scaled(-1.0)
    short, wide = design.copy(), design.copy()
    short.hub_height = 0.2
    wide.hub_radius = 0.95 * design.radius
    check("problems: none for the fitted design; a negative chord, a large clearance limit, a short hub, a hub "
          "radius near the tip found",
          [len(probs), "chord not positive" in " ".join(BM.design_problems(bad, p)),
           "apart" in " ".join(BM.design_problems(design, p, clearance_mm=1e4)),
           "does not cover the blade root" in " ".join(BM.design_problems(short, p)),
           "hub radius" in " ".join(BM.design_problems(wide, p)),
           len(BM.design_problems(short, p, hub=False))], [0, 1, 1, 1, 1, 0], 0)
    big = design.copy()
    big.radius *= 1.1
    big.hub_radius *= 1.2
    pb, sb = BM.blade_points(big, p)
    rad = np.hypot(pb[:, 1], pb[:, 2]).reshape(-1, 53)
    frac = np.linspace(0.0, 1.0, 11)
    check("blade and hub radius: the root and tip sections at them (mm), the curves stretched over the span",
          [1000.0 * rad[0].mean() - 1200.0 * design.hub_radius, 1000.0 * rad[-1].mean() - 1100.0 * design.radius,
           max(np.abs(big.at(c, big.r0 + frac * (big.r1 - big.r0)) - design.at(c, design.r0 + frac * (
               design.r1 - design.r0))).max() for c in BM.CURVES)], 0.0, 1e-6)
    res = BM.write_case(tmp, "t", design, p, BM.BladeSpace(p, design, free=("pitch",)), cad=False)
    secs = [b for b in open(res["paths"]["xcad_m.dat"]).read().split("#")[1:]]
    pts_back = np.loadtxt(res["paths"]["xcad_m.dat"], comments="#")
    q = BM.read_params(res["paths"]["params.dat"])
    check(f"write_case: XCAD points ({BM.SECTIONS} sections of 53, as para.py), the parameter file, the JSON, "
          "the plot",
          [len(secs), pts_back.shape[0], q.blades, json.load(open(res["paths"]["design.json"]))["globals"]["blades"],
           os.path.exists(res["paths"]["check"])], [BM.SECTIONS, BM.SECTIONS * 53, 5, 5, 1], 0)
    res_b = BM.write_case(tmp, "big", big, p, cad=False, plot=False)
    qb = BM.read_params(res_b["paths"]["params.dat"])
    check("write_case of a bigger blade and hub: the parameter file has its D, root r/R, hub height",
          [qb.diameter - big.diameter, qb.root_r - big.r0, qb.hub_height - big.hub_height], 0.0, 1e-9)
    s = BM.section_stations(0.17, 57, 0.93, 14)
    s2 = BM.section_stations(0.2, 30, 0.9, 8)
    check("section_stations: 57 from the root to the tip, 14 from r/R 0.93 closer together, even below",
          [len(s), s[0], s[-1], s[43], np.ptp(np.diff(s[:44])), np.all(np.diff(s[43:]) > 0),
           np.all(np.diff(np.diff(s[43:])) < 0), len(s2), s2[0], s2[22], s2[-1]],
          [57, 0.17, 1.0, 0.93, 0.0, 1, 1, 30, 0.2, 0.9, 1.0], 1e-12)
    out = [refused(BM.section_stations, *args)
           for args in ((0.17, 20, 0.1, 5), (0.17, 20, 1.0, 5), (0.17, 10, 0.93, 10), (0.17, 10, 0.93, 1))]
    check("section_stations refuses a band outside the blade, too few sections", out, [1, 1, 1, 1], 0)
    res2 = BM.write_case(tmp, "t30", design, p, cad=False, settings={"sections": 30, "tip_sections": 8}, plot=False)
    secs2 = open(res2["paths"]["xcad_m.dat"]).read().split("#")[1:]
    rec2 = json.load(open(res2["paths"]["design.json"]))
    check("write_case with 30 sections (8 in the tip band): 30 in the file; the clearance as before",
          [len(secs2), res2["sections"], rec2["settings"]["sections"], res2["clearance_mm"] - res["clearance_mm"]],
          [30, 30, 30, 0], 0)
    return True


def test_adapter(tmp, table):
    print("\nthe tool's blade adapter (xgeom_blade.py)")
    import xgeom_blade as XBL
    path = msc_file(tmp, table)
    ad = XBL.BladeAdapter(params=path)
    check("starts with 2-segment curves, pitch and chord free: 16 of 51 variables",
          [ad.n_free()[0], ad.n_free()[1], all(ad.segments(c) == 2 for c in BM.CURVES)], [16, 51, 1], 0)
    v = ad.set("pitch.v4", 99.0)
    check("set clamps to the bounds", v, ad.bounds("pitch.v4")[1], 0)
    ad.set_bounds("chord.w56", 0.1, 1.5)
    w_lo = ad.bounds("chord.w56")[0]
    ad.set_free("skew.v7", True)
    no_order = refused(ad.set_order, "chord", 6)
    msg1 = ad.set_segments("chord", 1)
    msg2 = ad.set_order("chord", 6)
    msg3 = ad.set_segments("pitch", 1)                    # pitch was changed above: its present shape kept
    check("bounds, a free flag, order refused at 2 segments; chord (as fitted) to 1 segment, 7 control points "
          "(free), fitted to the file; pitch (changed) fitted to its shape",
          [w_lo, ad.is_free("skew.v7"), no_order, ad.segments("chord"), ad.order("chord"), ad.n_free("chord")[0],
           msg1.startswith("fitted to the parameter file"), msg2.startswith("fitted to the parameter file"),
           BM.fit_report(ad.params, ad.design)["chord"][1] < 0.01, msg3.startswith("fitted to the curve as it was")],
          [0.1, 1, 1, 1, 6, 12, 1, 1, 1, 1], 0)
    print(f"       {msg1}; {msg2}; pitch: {msg3}")
    pv = ad.preview(skeleton=True)
    check(f"preview: no problems, 6 sections, {BM.SECTIONS} sections in the skeleton",
          [len(pv["problems"]), np.shape(pv["sections"])[0], np.shape(pv["skeleton"])[0]], [0, 6, BM.SECTIONS], 0)
    ad.set("hub_radius", 0.13)
    ad.set("radius", 0.72)
    pv2 = ad.preview()
    ad.set_bounds("hub_height", 0.1, 2.0)
    ad.set("hub_height", 0.2)
    pv3 = ad.preview()
    ad.set("hub_height", 1.0)
    check("globals: hub radius 0.13 m, R 0.72 m (root r/R follows), a 0.2 m hub is a problem",
          [ad.design.r0, ad.row_info("hub_radius") == f"r/R {0.13 / 0.72:.3f}", len(pv2["problems"]),
           "does not cover" in " ".join(pv3["problems"])], [0.13 / 0.72, 1, 0, 1], 1e-12)
    ad.set_sections(40, 0.9, 10)
    no_sec = refused(ad.set_sections, 12, 0.9, 12)
    pv = ad.preview(skeleton=True)
    check("set_sections: 40 sections in the skeleton, a bad set refused",
          [np.shape(pv["skeleton"])[0], no_sec, ad.settings["tip_sections"]], [40, 1, 10], 0)
    sp = ad.save_space(os.path.join(tmp, "setup.json"))
    s2, d2, _ = BM.load_blade_space(sp)
    ad2 = XBL.BladeAdapter(params=path)
    ad2.load_space(sp)
    check("Save set-up: load_blade_space and the adapter read it back (forms, globals, sections)",
          [s2.names == ad.free_space().names, np.abs(s2.to_vector(d2) - ad.free_space().to_vector(ad.design)).max(),
           max(abs(ad2.values[k] - ad.values[k]) for k in ad.values), ad2.n_free() == ad.n_free(),
           ad2.segments("chord"), ad2.design.radius, ad2.settings["sections"] == 40],
          [1, 0, 0, 1, 1, 0.72, 1], 1e-14)
    saved = ad.save_params(os.path.join(tmp, "saved.dat"))
    ad3 = XBL.BladeAdapter(params=saved)
    for c in BM.CURVES:                                    # the same forms (a curve as fitted: from the file)
        ad3.set_segments(c, ad.segments(c))
    rr = np.linspace(ad.design.r0, ad.design.r1, 501)
    check("Save parameters, then Load: the globals, the curves fitted again to the saved file (fraction of range)",
          [ad3.design.radius - 0.72, ad3.design.hub_radius - 0.13]
          + [np.abs(ad3.design.prop(c)(rr) - ad.design.prop(c)(rr)).max() / np.ptp(ad.design.prop(c)(rr))
             for c in BM.CURVES], 0.0, 0.01)
    gap5 = ad.preview()["clearance_mm"]
    no_z = [refused(ad.set_blades, z) for z in (1, 8, 4.5)]
    ad.set_blades(7)
    pv7 = ad.preview()
    sp7 = ad.save_space(os.path.join(tmp, "setup7.json"))
    s7, _, rec7 = BM.load_blade_space(sp7)
    ad7 = XBL.BladeAdapter(params=path)
    ad7.load_space(sp7)
    p7 = BM.read_params(ad.save_params(os.path.join(tmp, "z7.dat")))
    check("blades Z: 1, 8 and 4.5 refused (2 to 7); 7: the blades closer, the set-up and the saved file keep 7, "
          "the file's 5 kept apart", no_z + [pv7["clearance_mm"] < gap5, rec7["globals"]["blades"], s7.params.blades,
                                             ad7.params.blades, ad7.file_blades, p7.blades],
          [1, 1, 1, 1, 7, 7, 7, 5, 7], 0)
    print(f"       blades {gap5:.1f} mm apart with 5, {pv7['clearance_mm']:.1f} mm with 7")
    ad.settings["clearance_mm"] = pv7["clearance_mm"] + 10.0       # closer than the clearance: interfering
    pvw = ad.preview()
    res = BM.write_case(tmp, "warn", ad.design, ad.params, cad=False, settings=ad._case_settings(), plot=False)
    rec = json.load(open(res["paths"]["design.json"]))
    strict = BM.design_problems(ad.design, ad.params, ad.settings["clearance_mm"])
    check("blades closer than the clearance: a warning, not a problem (the tool builds the CAD); design.json "
          "records it; design_problems (the optimiser's check) still rejects the design",
          [len(pvw["problems"]), len(pvw["warnings"]), "interfere" in pvw["warnings"][0], len(res["problems"]),
           len(rec["warnings"]), len(strict)], [0, 1, 1, 0, 1, 1], 0)
    ad.settings["clearance_mm"] = BM.CLEARANCE_MM


def test_gui(tmp, table):
    print("\nthe window (xgeom_tool.py, blade)")
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:
        print(f"       skipped: {exc}")
        return
    import xgeom_tool as XT
    try:
        app = XT.App(root, "blade", params=msc_file(tmp, table))
        root.update()
        app.curve.set("chord")
        app.on_curve()
        app.set_value("chord.v4", app.ad.get("chord.v4") * 1.1, None)
        rows2 = len(app.rows)
        app.nseg.set(1)
        app.on_segments()
        app.nctrl.set(8)
        app.on_order()
        app.update_views()
        app.book.select(1)
        root.update()
        app.update_views()
        root.update()
        n3 = sum(len(c.get_segments()) for c in app.view3.col.values())
        check("the window: a slider, chord to 1 segment of 8 control points, the propeller tab with the hub",
              [rows2, app.ad.segments("chord"), app.ad.order("chord"), len(app.rows), n3],
              [3 + 8, 1, 7, 3 + 14, BM.SECTIONS * 5 + 14], 0)
        ok_bg = app.status.cget("background")
        app.sec["sections"].set(30)
        app.sec["tip_sections"].set(6)
        app.on_sections()
        app.update_views()
        root.update()
        n30 = sum(len(c.get_segments()) for c in app.view3.col.values())
        app.sec["tip_sections"].set(40)                       # more than fit: refused, the boxes reset
        app.on_sections()
        err_bg = app.status.cget("background")
        kept = app.sec["tip_sections"].get()
        check("Sections: 30 sections (6 in the tip band) in the propeller tab, 40 in the band refused",
              [app.ad.settings["sections"], n30, kept == "6"], [30, 30 * 5 + 14, 1], 0)
        app.ad.set_bounds("hub_height", 0.1, 2.0)
        app.set_value("hub_height", 0.2, None)                # too short for the blade root
        app.update_views()
        prob = app.status.cget("text")
        check("the status bar: green ok, red for an error and for a problem (a short hub: no CAD)",
              [ok_bg == XT.STATUS["ok"][0], err_bg == XT.STATUS["error"][0], prob.startswith("problem (no CAD"),
               "hub height" in prob, app.status.cget("background") == XT.STATUS["error"][0],
               app.status.winfo_ismapped()], [1] * 6, 0)
        app.nblades.set(7)
        app.on_blades()
        app.book.select(1)
        root.update()
        app.update_views()
        root.update()
        n7 = sum(len(c.get_segments()) for c in app.view3.col.values())
        app.nblades.set(8)                                    # refused: the box shows the Z in use
        app.on_blades()
        check("blades Z 7 from the box: 7 blades in the propeller tab; 8 refused (the box shows 7)",
              [app.ad.params.blades, n7, app.nblades.get() == "7"], [7, 30 * 7 + 14, 1], 0)
        app.ad.set("hub_height", 1.0)                          # no problem left
        app.ad.settings["clearance_mm"] = 500.0               # every gap is closer than that
        app.update_views()
        warn = app.status.cget("text")
        check("a warning in orange (blades interfere), the build still allowed",
              [app.status.cget("background") == XT.STATUS["warn"][0], warn.startswith("warning (CAD still possible)"),
               len(app.ad.preview()["problems"])], [1, 1, 0], 0)
        app.ad.settings["clearance_mm"] = BM.CLEARANCE_MM
    finally:
        root.destroy()


def test_cad(tmp, table):
    print("\nthe CAD (tip_surfaces_new, X_CAD_new; about two minutes)")
    import importlib.util
    if importlib.util.find_spec("OCC") is None:
        print("       skipped: no pythonocc-core")
        return
    p = BM.read_params(msc_file(tmp, table))
    design, _ = BM.fit_design(p)
    design.hub_height = 0.8
    out = BM.write_cad(tmp, "cad", design, p)
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    rd = STEPControl_Reader()
    rd.ReadFile(out["cad_paths"]["step"])
    rd.TransferRoots()
    shape = rd.OneShape()
    n, ex = 0, TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        n += 1
        ex.Next()
    try:                                                   # the hub's length along x (mm)
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        box = Bnd_Box()
        brepbndlib.Add(shape, box)
        x0, _, _, x1, _, _ = box.Get()
    except ImportError:
        x0, x1 = -400.0, 400.0
    print(f"       grids {out['grid_seconds']:.0f} s, faces and IGES {out['cad_seconds']:.1f} s; x from {x0:.1f} "
          f"to {x1:.1f} mm")
    check("IGES and STEP (2-segment curves, hub 0.8 m): 5 blade faces, hub sector, 2 caps, valid; hub length",
          [os.path.exists(out["cad_paths"]["iges"]), n, BRepCheck_Analyzer(shape).IsValid(),
           not any(f.startswith("x_cad_") for f in os.listdir(tmp)), (x1 - x0) / 800.0], [1, 8, 1, 1, 1], 2e-3)


def main():
    print("blade self-check")
    table = os.path.join(HERE, "airfoil_data_fixed.csv")
    if "--section-table" in sys.argv:
        table = os.path.abspath(sys.argv[sys.argv.index("--section-table") + 1])
    assert RM.Bezier                                       # the curves are rudder_modify's Bezier
    with tempfile.TemporaryDirectory() as tmp:
        test_file(tmp)
        test_curves(tmp)
        test_space(tmp)
        if test_geometry(tmp, table):
            test_adapter(tmp, table)
            if "--gui" in sys.argv:
                test_gui(tmp, table)
            if "--cad" in sys.argv:
                test_cad(tmp, table)
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
