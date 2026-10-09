"""
_xgeom_wing_test.py

Self-check for the wing and the fin in XGeom: the parameter file and its fitted curves (wing_modify.py), the
sections (the airfoil, its Bezier changes, the pitch), the design space, the case files and the loft, the
design tool's adapters (xgeom_wing.py), the fins and the wing pair on the hull (xgeom_vehicle.py) and in the
vehicle's CAD (vehicle_cad.py); with --gui the window (xgeom_tool.py).

    python _xgeom_wing_test.py          model, adapters, placement; with OCC the loft and the vehicle's CAD
    python _xgeom_wing_test.py --gui    also the window
"""

import json
import os
import sys
import tempfile

import numpy as np

import wing_modify as WM
import xcad_loft as XL

_FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))
WING, FIN = os.path.join(HERE, "wing_params.dat"), os.path.join(HERE, "fin_params.dat")


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<70s} max error {err:.3e}   (tol {tol:.0e})")


def refused(fn, *args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1


def table_file(tmp, name, head, rows):
    """A parameter file from header lines and table rows."""
    path = os.path.join(tmp, name)
    with open(path, "w") as fh:
        fh.write("\n".join(["# XGeom wing parameters"] + [f"# {h}" for h in head]
                           + ["  ".join(f"{v:.8f}" for v in r) for r in rows]) + "\n")
    return path


def test_file(tmp):
    print("\nthe parameter file")
    w, f = WM.read_params(WING), WM.read_params(FIN)
    check("the example wing and fin: kind, span, airfoil, rows, no change columns, the root's sweep and rake 0",
          [w.kind == "wing", f.kind == "fin", w.span, f.span, w.airfoil == "naca2412", f.airfoil == "naca0015",
           len(w.eta), w.changes, w.values["sweep"][0], w.values["rake"][0]], [1, 1, 200, 220, 1, 1, 11, 0, 0, 0], 0)
    path = WM.write_params(os.path.join(tmp, "again.dat"), w)
    w2 = WM.read_params(path)
    check("written and read again: the same table", [np.abs(w2.values[c] - w.values[c]).max() for c in WM.CURVES]
          + [np.abs(w2.eta - w.eta).max(), w2.span - w.span], 0.0, 1e-8)
    e = np.linspace(0.0, 1.0, 5)
    rows = np.column_stack([e, 100 - 20 * e, 5 + 10 * e, 2 + 4 * e, 0 * e, 0.1 + 0 * e, 0.02 + 0 * e,
                            0.01 * e, -0.01 * e, 0.005 + 0 * e, 0 * e])
    p = WM.read_params(table_file(tmp, "shifted.dat", ["span_mm 150", "airfoil naca0012", "ref_xc 0.3"], rows))
    check("a root LE off the origin is taken from the root's row; 2 change points read from 4 more columns; "
          "a symmetric airfoil's camber line", [p.values["sweep"][-1], p.values["rake"][-1], p.changes, p.ref_xc,
                                                p.values["dt2"][-1], p.values["dc1"][0]], [10, 4, 2, 0.3, -0.01, 0.005], 1e-12)
    bad = [table_file(tmp, "b1.dat", ["airfoil naca0012"], rows[:, :7]),
           table_file(tmp, "b2.dat", ["span_mm 100", "airfoil naca0012"], rows[:, :8]),
           table_file(tmp, "b3.dat", ["span_mm 100", "airfoil naca0012"], rows[1:, :7]),
           table_file(tmp, "b4.dat", ["span_mm 100", "airfoil not_an_airfoil_9"], rows[:, :7]),
           table_file(tmp, "b5.dat", ["span_mm 100", "airfoil naca0012", "camber_xc 1.2"], rows[:, :7]),
           table_file(tmp, "b6.dat", ["span_mm -5", "airfoil naca0012"], rows[:, :7])]
    check("refused: no span, an odd change column, eta not from 0, an unknown airfoil, camber_xc 1.2, span -5",
          [refused(WM.read_params, b) for b in bad], [1] * len(bad), 0)


def test_section():
    print("\nthe sections (airfoil, changes, pitch)")
    p = WM.read_params(WING)
    T0, C0, info = WM.base_shape("naca2412")
    x = np.linspace(0.0, 1.0, 4001)
    check("NACA 2412: thickness and camber line over their maxima (1), the TE closed, both 0 at the LE, the "
          "maxima near 0.3 and 0.4 c (the database's file is coarse)",
          [T0(x).max(), C0(x).max(), T0(1.0), T0(0.0), C0(0.0), abs(info["x_t"] - 0.3) < 0.03,
           abs(info["x_c"] - 0.4) < 0.03], [1, 1, 0, 0, 0, 1, 1], 1e-6)
    _, C0s, info_s = WM.base_shape("naca0015", 0.3)
    check("NACA 0015, symmetric: f/c scales the NACA 4-digit camber line, its maximum (1) at camber_xc 0.3",
          [info_s["symmetric"], C0s(0.3), C0s(1.0), C0s(np.linspace(0, 1, 2001)).max()], [1, 1, 0, 1], 1e-6)
    d, _ = WM.fit_design(p)
    v = d.values([0.0, 0.5])
    T, C = WM.section_shape(v, p, x, d.changes)
    check("no changes: the largest thickness and camber are t/c and f/c", [T.max(axis=1) - v["thickness"],
                                                                            C.max(axis=1) - v["camber"]], 0.0, 1e-6)
    import airfoils
    xy, _ = airfoils.coordinates("naca2412")
    up, lo = airfoils.surfaces(xy)
    one = dict(v, thickness=np.array([info["t_max"]]), camber=np.array([info["c_max"]]))
    one = {k: q[:1] for k, q in one.items()}
    xs = up[5:-5, 0]
    T1, C1 = WM.section_shape(one, p, xs, d.changes)
    ramp = xs * info["te"]
    check("scaled to its own t/c and f/c the section is NACA 2412 (upper side, the TE ramp put back)",
          (C1 + 0.5 * T1)[0] + 0.5 * ramp - up[5:-5, 1], 0.0, 2e-5)
    d2 = d.copy()
    d2.curves["dt2"] = WM.BladeCurve(np.array([0.0, 0.5, 1.0]), np.array([0.01, 0.01, 0.01]))
    d2.curves["dc1"] = WM.BladeCurve(np.array([0.0, 0.5, 1.0]), np.array([-0.004, -0.004, -0.004]))
    Tc, Cc = WM.section_shape(d2.values([0.5]), p, x, d2.changes)
    B = WM.change_basis(x, 3)
    check("changes: dt2 0.01 adds 0.01 B(2, 4) to the thickness, dc1 -0.004 B(1, 4) to the camber line; none at "
          "the LE and the TE", [np.abs(Tc[0] - T[1] - 0.01 * B[:, 1]).max(), np.abs(Cc[0] - C[1] + 0.004 * B[:, 0]).max(),
                                Tc[0, 0], Tc[0, -1]], 0.0, 1e-12)
    d5 = WM.with_changes(d2, 5)
    T5, C5 = WM.section_shape(d5.values([0.5]), p, x, 5)
    d3 = WM.with_changes(d5, 3)
    T3, C3 = WM.section_shape(d3.values([0.5]), p, x, 3)
    check("5 change points instead of 3: the same sections (degree elevation); back to 3: the same again",
          [np.abs(T5 - Tc).max(), np.abs(C5 - Cc).max(), np.abs(T3 - Tc).max(), np.abs(C3 - Cc).max(),
           d5.changes, d3.changes], [0, 0, 0, 0, 5, 3], 1e-9)
    L = WM.section_loops(d, p, [0.0, 1.0])
    c, th, ref = v["chord"][0], np.radians(v["pitch"][0]), p.ref_xc
    m = (L.shape[1] + 1) // 2 - 1
    check("the root pitched 2 deg about its quarter chord: the LE up by c/4 sin 2 deg, the TE down by 3c/4 sin, "
          "the loop flat at y 0 and closed", [L[0, 0, 2], L[0, m, 2], np.ptp(L[0, :, 1]), np.abs(L[0, 0] - L[0, -1]).max(),
                                              L[0, 0, 0]],
          [ref * c * np.sin(th), -(1 - ref) * c * np.sin(th), 0, 0, ref * c * (1 - np.cos(th))], 2e-3)
    vt = d.values([1.0])
    check("the tip: at the span, its LE at (sweep, rake) moved by the pitch about the pitch axis",
          [L[1, 0, 1], L[1, 0, 0] - vt["sweep"][0] - ref * vt["chord"][0] * (1 - np.cos(np.radians(vt["pitch"][0]))),
           L[1, 0, 2] - vt["rake"][0] - ref * vt["chord"][0] * np.sin(np.radians(vt["pitch"][0]))], [200, 0, 0], 1e-9)


def test_curves():
    print("\nthe curves and the design space")
    p = WM.read_params(WING)
    d, res = WM.fit_design(p)
    rep = WM.fit_report(p, d)
    check("the example's distributions (polynomials) fitted by 4-point curves; the root's sweep and rake 0",
          [rep[c][0] for c in WM.CURVES] + [d.ctrl("sweep")[0, 1], d.ctrl("rake")[0, 1], d.changes,
                                            max(np.abs(d.curves[c].v).max() for c in WM.change_names(3))],
          [0] * 6 + [0, 0, 3, 0], 1e-7)
    q = WM.read_params(WING)
    q.values["chord"] = 120.0 * np.sqrt(1.0 - 0.75 * q.eta ** 2)
    q.values["sweep"] = 40.0 * q.eta ** 1.5
    dq, _ = WM.fit_design(q)
    d2, _ = WM.fit_design(q, segments={"chord": 2, "sweep": 2})
    rq, r2 = WM.fit_report(q, dq), WM.fit_report(q, d2)
    check("an elliptical chord and a 1.5-power sweep: 1 segment within 0.1 mm, 2 segments within 0.5 mm, the "
          "sweep's root pinned at 0", [rq["chord"][0] < 0.1, rq["sweep"][0] < 0.1, r2["chord"][0] < 0.5,
                                       r2["sweep"][0] < 0.5, dq.ctrl("sweep")[0, 1], d2.curves["sweep"].v1], [1, 1, 1, 1, 0, 0], 0)
    e = np.linspace(0.0, 1.0, 101)
    dr = WM.refit_curve(dq, "chord", order=6)
    check("the chord at 7 control points: its shape kept (mm)", dr.at("chord", e) - dq.at("chord", e), 0.0, 0.05)
    sp = WM.WingSpace(p, d, free=("chord", "span"))
    full = WM.WingSpace(p, d)
    x = sp.to_vector(d)
    d_back = sp.to_design(x)
    check("the space: span first, no slots for the pinned roots; free chord and span (8 + 1 + 0); the design "
          "back from its vector", [full.all_names[0] == "span", "sweep.v0" in full.all_names,
                                   "rake.v0" in full.all_names, len(sp), max(np.abs(d_back.at(c, e) - d.at(c, e)).max()
                                                                            for c in d.names)], [1, 0, 0, 7, 0], 1e-12)
    b = WM.default_bounds(p, d)
    check("default bounds: the chord and thickness above 0, the changes +-0.02, the span 100 to 300 mm",
          [b["chord"][0] >= 0, b["thickness"][0] >= 0, b["dt1"][0], b["dc3"][1], b["span"][0], b["span"][1],
           b["pitch"][0] < -1, b["pitch"][1] > 2], [1, 1, -0.02, 0.02, 100, 300, 1, 1], 1e-12)
    vals = full.values(d)
    vals["span"] = 300.0
    d300 = full.design_of(vals)
    L1, L2 = WM.section_loops(d, p, [0.5]), WM.section_loops(d300, p, [0.5])
    check("a span of 300 mm: the curves stretch over it, the sections the same at the same eta, 1.5 x as far out",
          [np.abs(L1[..., [0, 2]] - L2[..., [0, 2]]).max(), L2[0, 0, 1] / L1[0, 0, 1]], [0, 1.5], 1e-12)
    rec = sp.to_dict(d)
    sp2, d2b = WM.WingSpace.from_dict(p, json.loads(json.dumps(rec)))
    check("the space as plain data and back: names, bounds, free slots, the design",
          [sp2.names == sp.names, np.abs(sp2.bounds - sp.bounds).max(),
           max(np.abs(d2b.at(c, e) - d.at(c, e)).max() for c in d.names)], [1, 0, 0], 1e-12)
    bad = d.copy()
    bad.curves["chord"] = WM.BladeCurve(np.array([0.0, 0.5, 1.0]), np.array([100.0, 50.0, -10.0]))
    thin = d.copy()
    thin.curves["dt3"] = WM.BladeCurve(np.array([0.0, 0.5, 1.0]), np.array([-0.12, -0.12, -0.12]))
    flat = d.copy()
    flat.span = 0.0
    check("checks: the example fine; a chord below 0, sides that cross near the TE, a span of 0 refused",
          [len(WM.design_problems(d, p)), len(WM.design_problems(bad, p)), len(WM.design_problems(thin, p)),
           len(WM.design_problems(flat, p)), "cross" in WM.design_problems(thin, p)[0]], [0, 1, 1, 1, 1], 0)


def test_case(tmp):
    print("\nthe case files and the loft")
    p = WM.read_params(WING)
    d, _ = WM.fit_design(p)
    d.curves["dc2"] = WM.BladeCurve(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.004, 0.008]))
    sp = WM.WingSpace(p, d, free=("chord", "pitch", "dc2"))
    out = WM.write_case(tmp, "c1", d, p, sp, cad=False, settings={"sections": 31}, plot=True)
    paths = out["paths"]
    loops = XL.read_loops(paths["xcad_m.dat"], "m")
    z = np.array([lp[:, 2] for lp in loops])
    check("written: parameter file, XCAD file, sections table, design, set-up, plot; 31 loops of 121 points, each "
          "flat at its span station (XCAD Z = y)", [all(os.path.exists(q) for q in paths.values()), len(loops),
                                                    len(loops[0]), np.ptp(z, axis=1).max(), z[-1, 0], z[0, 0]],
          [1, 31, 121, 0, 200, 0], 1e-6)
    tab = np.loadtxt(paths["sections.dat"])
    check("the sections table: t_max/c is t/c (the thickness has no change), f_max/c above f/c (dc2 adds camber)",
          [np.abs(tab[:, 9] - tab[:, 7]).max(), (tab[1:, 11] > tab[1:, 8]).all()], [0, 1], 1e-5)
    p2 = WM.read_params(paths["params.dat"])
    d2, _ = WM.fit_design(p2)
    e = np.linspace(0.0, 1.0, 51)
    check("the design as a parameter file, read and fitted again: the same curves, its change columns",
          [p2.changes, max(np.abs(d2.at(c, e) - d.at(c, e)).max() for c in d.names)], [3, 0], 1e-5)
    space, d3, rec = WM.load_wing_space(paths["space"])
    check("load_wing_space: the design and the free slots", [rec["geometry"] == "wing", space.names == sp.names,
                                                           max(np.abs(d3.at(c, e) - d.at(c, e)).max() for c in d.names)],
          [1, 1, 0], 1e-12)
    if not XL.occ_available():
        print("       the loft skipped: no OCCT binding")
        return
    cad = WM.write_cad(tmp, "c1", paths["xcad_m.dat"])
    lr = cad["loft"]
    check("the loft: one valid solid, closed, 4 faces (the two sides, the flat root and tip), its volume the "
          "sections' (fraction)", [lr.valid, lr.free_edges, XL.count_faces(lr.shape),
                                   lr.volume / out["volume_mm3"] - 1.0, os.path.exists(cad["cad_paths"]["step"])],
          [1, 0, 4, 0, 1], 2e-3)


def test_adapter(tmp):
    print("\nthe adapters (xgeom_wing.py)")
    import xgeom_wing as XW
    ad, fin = XW.WingAdapter(), XW.FinAdapter()
    rows = ad.curve_rows("sweep")
    check("wing and fin from their own files; the sweep's root a pinned row; chord and pitch free",
          [ad.title.startswith("Wing"), fin.title.startswith("Fin"), rows[0].kind == "pinned",
           ad.is_free("chord.v1"), ad.is_free("pitch.v0"), ad.is_free("sweep.v1"), ad.out_dir.endswith("wing"),
           fin.out_dir.endswith("fin")], [1, 1, 1, 1, 1, 0, 1, 1], 0)
    ad.set("chord.v3", 60.0)
    v = ad.set("span", 1e6)
    check("a variable set (the tip chord 60 mm); the span clamped to its bounds", [ad.design.at("chord", 1.0),
                                                                                  v, ad.design.span], [60, 300, 300], 1e-9)
    check("bounds refused: low above high, a negative span, d outside [0, 1]",
          [refused(ad.set_bounds, "chord.v0", 5, 1), refused(ad.set_bounds, "span", -1, 10),
           refused(ad.set_bounds, "chord.d1", -0.1, 0.5)], [1, 1, 1], 0)
    e = np.linspace(0.0, 1.0, 101)
    before = ad.design.at("chord", e)
    msg = ad.set_order("chord", 5)
    check("6 control points for the changed chord: refitted to its shape", [ad.order("chord"), "as it was" in msg,
                                                                          np.abs(ad.design.at("chord", e) - before).max() < 0.05],
          [5, 1, 1], 0)
    msg = ad.set_segments("pitch", 2)
    check("pitch as 2 segments: fitted to the file (untouched), P1 .. P7", [ad.segments("pitch"),
                                                                           "parameter file" in msg,
                                                                           len(ad.curve_rows("pitch"))], [2, 1, 8], 0)
    ad.set("dt2.v1", 0.01)
    x = np.linspace(0.0, 1.0, 201)
    sec0 = WM.section_shape(ad.design.values(e), ad.params, x, 3)
    msg = ad.set_changes(5)
    sec1 = WM.section_shape(ad.design.values(e), ad.params, x, 5)
    check("5 change points: the sections kept, dt1 .. dt5 and dc1 .. dc5, held as before",
          [ad.design.changes, np.abs(sec1[0] - sec0[0]).max(), "dc5" in ad.names, ad.is_free("dt4.v0")],
          [5, 0, 1, 0], 1e-9)
    path = ad.save_space(os.path.join(tmp, "w_design_space.json"))
    ad2 = XW.WingAdapter()
    ad2.load_space(path)
    check("Save set-up, then load_space: values, bounds, free flags, 5 change points",
          [ad2.design.changes, max(abs(ad2.get(n) - ad.get(n)) for n in ad.space.all_names),
           all(ad2.is_free(n) == ad.is_free(n) for n in ad.space.all_names)], [5, 0, 1], 1e-12)
    pf = ad.save_params(os.path.join(tmp, "w_params.dat"))
    ad3 = XW.WingAdapter(pf)
    check("save_params and a new adapter from it (its curves 4 points again): the same wing, changes included",
          [ad3.design.changes, max(np.abs(ad3.design.at(c, e) - ad.design.at(c, e)).max() for c in ad.names)],
          [5, 0], 1e-3)
    pv = ad.preview(skeleton=True)
    check("preview: no problems, 5 view sections, the XCAD sections (41 of 121 points)",
          [len(pv["problems"]), len(pv["sections"]), pv["skeleton"].shape[0], pv["skeleton"].shape[1]],
          [0, 5, 41, 121], 0)
    ad.out_dir = os.path.join(tmp, "out")
    ad.settings["plot"] = False
    res = ad.build_task("b")()
    occ = XL.occ_available()
    check("the build: the case written" + ("; the loft's STEP, valid" if occ else "; no CAD without OCC (a message)"),
          [os.path.exists(os.path.join(tmp, "out", "b_xcad_m.dat")), bool(res["view"]) == occ, res["error"] != occ],
          [1, 1, 1], 0)


class FakeAdapter:
    """What wing_part needs of an adapter."""
    def __init__(self, path):
        self.params = WM.read_params(path)
        self.design = WM.fit_design(self.params)[0]


def test_vehicle():
    print("\nfins and wings on the hull (xgeom_vehicle.py)")
    import hull_modify as HM
    import xgeom_vehicle as XV
    hp = HM.read_params(os.path.join(HERE, "suboff_hull_params.dat"))
    hull = HM.fit_design(hp)
    parts = {"hull": {"design": hull, "lines": [], "problems": []}, "fin": XV.wing_part(FakeAdapter(FIN)),
             "wing": XV.wing_part(FakeAdapter(WING))}
    pl = XV.default_placement(parts)
    j = hull.joins
    check("defaults: one fin on top a tenth along the middle body, the wings level a quarter along it",
          [pl.fins, pl.fin_angle, pl.fin_x, pl.wings, pl.wing_angle, pl.wing_x],
          [1, 0, round(j[1] + 0.1 * (j[2] - j[1]), 3), 2, 90, round(j[1] + 0.25 * (j[2] - j[1]), 3)], 1e-12)
    pl.fins, pl.fin_angle, pl.wing_angle = 3, 30.0, 60.0
    segs, info = XV.assemble(parts, pl)
    chord = (parts["fin"]["root_x"][1] - parts["fin"]["root_x"][0]) / 1000.0
    out = []
    for k, a in enumerate(info["fin_frames"]):
        phi = np.radians(30.0 + 120.0 * k)
        x_le, x_te = info["fin_x_le"][k], info["fin_x_te"][k]
        r_le = XV.surface_distance(hull, np.array([x_le]), phi)[0]
        r_te = XV.surface_distance(hull, np.array([x_te]), phi)[0]
        out += [np.hypot(a[0][1], a[0][2]) - r_le, x_le - pl.fin_x, np.hypot(x_te - x_le, r_te - r_le) - chord,
                np.arctan2(a[0][1], a[0][2]) % (2 * np.pi) - phi]
    check("3 fins from 30 deg: each root LE on the hull at fin x in its own half-plane, the TE a root chord aft "
          "along the hull", out, 0.0, 1e-9)
    n = len(segs["wings"]) // 2
    a, ec, eh, et = info["wing_frames"][0]
    phi = np.radians(60.0)
    r_le = XV.surface_distance(hull, np.array([pl.wing_x]), phi)[0]
    check("the starboard wing at 60 deg on the hull (z up: et above the horizontal), the port one its mirror "
          "image", [np.hypot(a[1], a[2]) - r_le, a[1] > 0, et[2] > 0, eh @ [0, np.sin(phi), np.cos(phi)],
                    max(np.abs(p * [1, -1, 1] - q).max() for p, q in zip(segs["wings"][:n], segs["wings"][n:])),
                    len(segs["wing_roots"]) == 34],
          [0, 1, 1, 1, 0, 1], 1e-9)
    pl_t = XV.Placement(**{**pl.__dict__, "fin_x": float(j[3] - 0.4), "fins": 1, "fin_angle": 0.0})
    segs_t, info_t = XV.assemble(parts, pl_t)
    x_le, x_te = info_t["fin_x_le"][0], info_t["fin_x_te"][0]
    r = XV.surface_distance(hull, np.array([x_le, x_te]), 0.0)
    check("a fin on the tail: pitched to the hull's slope under its root (the TE on the hull, a chord aft)",
          [info_t["fin_pitch"][0] - np.degrees(np.arctan2(r[0] - r[1], x_te - x_le)), info_t["fin_pitch"][0] > 5],
          [0, 1], 1e-9)
    L = hull.length
    probs = [XV.placement_problem({"wing_angle": a_}, L) for a_ in (0.0, 180.0, 200.0)]
    probs += [XV.placement_problem({"wings": 1}, L), XV.placement_problem({"fins": 13}, L),
              XV.placement_problem({"fin_x": L + 0.1}, L), XV.placement_problem({"wing_scale": 0.0}, L)]
    none = XV.Placement(**{**pl.__dict__, "fins": 0, "wings": 0})
    segs0, info0 = XV.assemble(parts, none)
    check("refused: wings at 0, 180 or 200 deg, 1 wing, 13 fins, a fin off the hull, scale 0; 0 fins and 0 wings: "
          "none drawn, not built", [p_ is not None for p_ in probs]
          + [len(segs0["fins"]), len(segs0["wings"]), XV.active(parts, none) == ["hull"],
             XV.placement_problem(pl, L) is None], [1] * 7 + [0, 0, 1, 1], 0)


def test_vehicle_cad(tmp):
    print("\nthe vehicle's CAD with fins and wings (vehicle_cad.py)")
    if not XL.occ_available():
        print("       skipped: no OCCT binding")
        return
    import vehicle_cad as VC
    import xgeom_hull as XH
    import xgeom_vehicle as XV
    import xgeom_wing as XW
    ads = {"hull": XH.HullAdapter(), "fin": XW.FinAdapter(), "wing": XW.WingAdapter()}
    for k, ad in ads.items():
        ad.out_dir = os.path.join(tmp, k)
        ad.settings["plot"] = False
    parts = {k: XV.PARTS[k](ad) for k, ad in ads.items()}
    pl = XV.default_placement(parts)
    pl.fins, pl.wing_angle = 2, 70.0
    segs, info = XV.assemble(parts, pl)
    tasks = {k: ads[k].build_task("v") for k in XV.active(parts, pl)}
    res = VC.run_vehicle(os.path.join(tmp, "vehicle"), "v", tasks, parts, info, pl, lambda s: None)
    rec = res["record"]
    shape = XL.read_step(rec["files"]["vehicle"])
    solids = VC._shapes(XL.read_step(rec["files"]["parts"]), "SOLID")
    bs, bp = VC._bbox(solids[3]), VC._bbox(solids[4])
    check("hull, 2 fins and the wing pair built, placed and fused: one valid solid; the port wing the "
          "starboard's mirror image", [rec["parts"] == ["hull", "fin 1", "fin 2", "wing starboard", "wing port"],
                                       rec["solids"], rec["valid"], np.abs(bs[:, [0, 2]] - bp[:, [0, 2]]).max(),
                                       bs[0, 1] + bp[1, 1], bs[1, 1] + bp[0, 1]], [1, 1, 1, 0, 0, 0], 1e-3)
    lines = np.vstack([1000.0 * q for k in ("hull", "fins", "wings") for q in segs[k]])
    box = VC._bbox(shape)
    check("the vehicle's box (mm): the Vehicle tab's lines', the parts 0.2 mm off the hull on their roots",
          np.r_[box[0] - lines.min(axis=0), box[1] - lines.max(axis=0)], 0.0, 0.25)
    import hull_modify as HM
    v_hull = HM.hydrostatics(ads["hull"].design)["volume"] * 1e9
    v_parts = sum(VC.volume(s) for s in solids[1:])
    check("its volume: the hull's and the appendages' outside it (less than with their whole root extrusions)",
          [rec["volume_mm3"] > v_hull + 0.95 * v_parts - 0.1 * v_parts, rec["volume_mm3"] < v_hull + v_parts],
          [1, 1], 0)


def test_gui(tmp):
    print("\nthe window (xgeom_tool.py: the vehicle, the wing alone)")
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:                                   # no tkinter or no display
        print(f"       skipped: {exc}")
        return
    import time
    import xgeom_tool as XT

    def settle(n=5):
        for _ in range(n):
            root.update()
            time.sleep(0.03)
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        app = XT.App(root, "vehicle")
        settle()
        app.switch("fin")
        app.switch("wing")
        settle()
        names = [r.name for r in app.rows]
        check("vehicle mode: the fin and the wing as components; the wing's panel: span, its curves, build 'CAD wing'",
              [app.kinds == ["hull", "blade", "rudder", "fin", "wing"], "span" in names, "chord.v0" in names,
               app.build_btn.cget("text") == "CAD wing", "dc3" in app.picks], [1, 1, 1, 1, 1], 0)
        app.set_value("chord.v3", 60.0, None)
        app.wsec["changes"].set(4)
        app.on_wing_sections()
        settle()
        check("a slider (the tip chord 60 mm); 4 change points from the box: dt4, dc4 to pick, the panel rebuilt",
              [app.ad.design.at("chord", 1.0), app.ad.design.changes, "dt4" in app.picks, "dc4" in app.picks],
              [60, 4, 1, 1], 1e-9)
        app.curve.set("dc2")
        app.on_curve()
        app.update_views()
        settle()
        app.book.select(1)
        app.update_views()
        settle()
        app.book.select(XT.VEHICLE_TAB)
        app.update_views()
        settle()
        n = {k: len(c.get_segments()) for k, c in app.viewv.col.items()}
        check("the Vehicle tab: one fin, two wings with their root extrusions; the status ok",
              [n["fins"] > 10, n["wings"] == 2 * n["fins"], n["fin_roots"], n["wing_roots"],
               app.status.cget("background") == XT.STATUS["ok"][0]], [1, 1, 17, 34, 1], 0)
        app.vset["fins"].set(0)
        app.vset["wings"].set(0)
        app.on_vehicle()
        app.update_views()
        settle()
        n0 = {k: len(c.get_segments()) for k, c in app.viewv.col.items()}
        app.vset["wings"].set(2)
        app.vset["wing_angle"].set(190)
        app.on_vehicle()
        refused = app.status.cget("background") == XT.STATUS["error"][0] and app.vset["wing_angle"].get() == "90"
        check("0 fins and 0 wings: none drawn; the wing at 190 deg refused (red), the box shows 90",
              [n0["fins"], n0["wings"], refused], [1, 1, 1], 0)
        app.vset["wing_angle"].set(90)
        app.on_vehicle()
        app.book.select(0)
        settle()
        app.case.delete(0, "end")
        app.case.insert(0, "gui_test")
        app.ad.out_dir = os.path.join(tmp, "gui_wing")
        app.on_build("wing")
        t0 = time.time()
        while app._job is not None and time.time() - t0 < 120:
            settle(2)
        occ = XL.occ_available()
        ok = app.status.cget("background") == XT.STATUS["ok" if occ else "error"][0]
        check("CAD wing: the case and " + ("the loft's STEP" if occ else "no CAD without OCC (red)"),
              [os.path.exists(os.path.join(tmp, "gui_wing", "gui_test_xcad_m.dat")), ok, bool(app.step) == occ],
              [1, 1, 1], 0)
        settle()
        root.destroy()
        root = tk.Tk()
        app = XT.App(root, "fin")
        settle()
        check("the fin alone: no switcher, no Vehicle tab; its panel", [not hasattr(app, "comp"), app.book.index("end"),
                                                                       "span" in [r.name for r in app.rows]], [1, 2, 1], 0)
        settle()
        root.destroy()
    finally:
        os.chdir(cwd)


def main():
    print("wing and fin: self-check")
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            test_file(tmp)
            test_section()
            test_curves()
            test_case(tmp)
            test_adapter(tmp)
            test_vehicle()
            test_vehicle_cad(tmp)
            if "--gui" in sys.argv:
                test_gui(tmp)
    finally:
        os.chdir(cwd)
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
