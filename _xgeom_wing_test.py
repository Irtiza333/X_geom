"""
_xgeom_wing_test.py

Self-check for the wing and the fin in XGeom: the component file and its fitted curves (wing_modify.py), the
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
WING, FIN = os.path.join(HERE, "wing.txt"), os.path.join(HERE, "fin.txt")      # the samples (read_params: the table)


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
    """A wing's table file (a component file with no curve or var lines) from geometry lines and table rows: eta
    pitch_deg chord_mm t/c f/c sweep_mm rake_mm, then dt1 .. dtk dc1 .. dck (an odd number of them: one dt
    more)."""
    n = rows.shape[1] - len(WM.XF.COLUMNS["wing"])
    cols = list(WM.XF.COLUMNS["wing"]) + [f"dt{i}" for i in range(1, n - n // 2 + 1)] \
        + [f"dc{i}" for i in range(1, n // 2 + 1)]
    path = os.path.join(tmp, name)
    with open(path, "w") as fh:
        fh.write("\n".join(["# XGeom component", "# kind  wing"] + [f"# {h}" for h in head] + ["# " + "  ".join(cols)]
                           + ["  ".join(f"{v:.8f}" for v in r) for r in rows]) + "\n")
    return path


def test_file(tmp):
    print("\nthe component file's geometry and table (read_params, write_params)")
    w, f = WM.read_params(WING), WM.read_params(FIN)
    check("the example wing and fin: kind, span, airfoil, rows, 3 change points (zero columns), the root's sweep "
          "and rake 0", [w.kind == "wing", f.kind == "fin", w.span, f.span, w.airfoil == "naca2412",
                         f.airfoil == "naca0015", len(w.eta), w.changes,
                         max(np.abs(w.values[c]).max() for c in WM.change_names(3)), w.values["sweep"][0],
                         w.values["rake"][0]], [1, 1, 200, 220, 1, 1, 11, 3, 0, 0, 0], 0)
    path = WM.write_params(os.path.join(tmp, "again.txt"), w)
    w2 = WM.read_params(path)
    comp = WM.XF.read_component(path)
    check("written as a table file (no curve or var lines) and read again: the same table",
          [np.abs(w2.values[c] - w.values[c]).max() for c in w.names]
          + [np.abs(w2.eta - w.eta).max(), w2.span - w.span, len(comp.curves) + len(comp.vars)], 0.0, 1e-8)
    e = np.linspace(0.0, 1.0, 5)
    rows = np.column_stack([e, 0 * e, 100 - 20 * e, 0.1 + 0 * e, 0.02 + 0 * e, 5 + 10 * e, 2 + 4 * e,
                            0.01 * e, -0.01 * e, 0.005 + 0 * e, 0 * e])
    p = WM.read_params(table_file(tmp, "shifted.txt", ["span_mm 150", "section naca0012", "ref_xc 0.3"], rows))
    check("a root LE off the origin is taken from the root's row; 2 change points read from 4 more columns; "
          "a symmetric airfoil's camber line", [p.values["sweep"][-1], p.values["rake"][-1], p.changes, p.ref_xc,
                                                p.values["dt2"][-1], p.values["dc1"][0]], [10, 4, 2, 0.3, -0.01, 0.005], 1e-12)
    bad = [table_file(tmp, "b1.txt", ["section naca0012"], rows[:, :7]),
           table_file(tmp, "b2.txt", ["span_mm 100", "section naca0012"], rows[:, :8]),
           table_file(tmp, "b3.txt", ["span_mm 100", "section naca0012"], rows[1:, :7]),
           table_file(tmp, "b4.txt", ["span_mm 100", "section not_an_airfoil_9"], rows[:, :7]),
           table_file(tmp, "b5.txt", ["span_mm 100", "section naca0012", "camber_xc 1.2"], rows[:, :7]),
           table_file(tmp, "b6.txt", ["span_mm -5", "section naca0012"], rows[:, :7])]
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
    check("written: component file, XCAD file, sections table, design, plot; 31 loops of 121 points, each "
          "flat at its span station (XCAD Z = y)", [all(os.path.exists(q) for q in paths.values()), len(loops),
                                                    len(loops[0]), np.ptp(z, axis=1).max(), z[-1, 0], z[0, 0]],
          [1, 31, 121, 0, 200, 0], 1e-6)
    tab = np.loadtxt(paths["sections.dat"])
    check("the sections table: t_max/c is t/c (the thickness has no change), f_max/c above f/c (dc2 adds camber)",
          [np.abs(tab[:, 9] - tab[:, 7]).max(), (tab[1:, 11] > tab[1:, 8]).all()], [0, 1], 1e-5)
    p2 = WM.read_params(paths["component"])
    d2, _ = WM.fit_design(p2)
    e = np.linspace(0.0, 1.0, 51)
    check("the design's component file, its table read and fitted again: the same curves, its change columns",
          [p2.changes, max(np.abs(d2.at(c, e) - d.at(c, e)).max() for c in d.names)], [3, 0], 1e-5)
    ld = WM.load_component(paths["component"])
    check("load_component: the kind, the design and the free variables",
          [ld.kind == "wing", ld.space.names == sp.names, max(np.abs(ld.design.at(c, e) - d.at(c, e)).max()
                                                              for c in d.names)], [1, 1, 0], 1e-9)
    if not XL.occ_available():
        print("       the loft skipped: no OCCT binding")
        return
    cad = WM.write_cad(tmp, "c1", paths["xcad_m.dat"])
    lr = cad["loft"]
    check("the loft: one valid solid, closed, 6 faces (the four wrap patches, the flat root and top), its "
          "volume the sections' (fraction)", [lr.valid, lr.free_edges, XL.count_faces(lr.shape),
                                              lr.volume / out["volume_mm3"] - 1.0,
                                              os.path.exists(cad["cad_paths"]["step"])],
          [1, 0, 6, 0, 1], 2e-3)
    try:
        WM.write_cad(tmp, "c1h", paths["xcad_m.dat"], half=True)
        refused = 0
    except ValueError:
        refused = 1
    check("a half body of the cambered wing is refused", refused, 1, 0)
    pf = WM.read_params(FIN)
    df, _ = WM.fit_design(pf)
    outf = WM.write_case(tmp, "fh", df, pf, cad=False, settings={"sections": 21}, plot=False)
    lrf = WM.write_cad(tmp, "fh_full", outf["paths"]["xcad_m.dat"])["loft"]
    lrh = WM.write_cad(tmp, "fh_half", outf["paths"]["xcad_m.dat"], half=True)["loft"]
    check("the fin's half body: valid, closed, 6 faces (the wrap patches and the ends cut on the plane of "
          "symmetry, and the plane itself), half the volume",
          [lrh.valid, lrh.free_edges, lrh.half, XL.count_faces(lrh.shape),
           lrh.volume / lrf.volume], [1, 0, 1, 6, 0.5], 1e-5)


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
    path = ad.save_file(os.path.join(tmp, "w_saved.txt"))
    ad2 = XW.WingAdapter(path)
    check("Save set-up (a component file), then a new adapter from it: values, bounds, free flags, 5 change "
          "points (to the figures written)",
          [ad2.design.changes, max(abs(ad2.get(n) - ad.get(n)) for n in ad.space.all_names),
           max(abs(ad2.bounds(n)[i] - ad.bounds(n)[i]) for n in ad.space.all_names for i in (0, 1)),
           all(ad2.is_free(n) == ad.is_free(n) for n in ad.space.all_names)], [5, 0, 0, 1], 1e-9)
    pf = WM.write_params(os.path.join(tmp, "w_table.txt"), WM.design_params(ad.design, ad.params))
    ad3 = XW.WingAdapter(pf)
    check("the design as a table file and a new adapter from it (its curves 4 points again): the same wing, changes "
          "included",
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


def planar_rudder_file(tmp):
    """fin.txt written again as a planar rudder's component file (kind rudder)."""
    ld = WM.load_component(os.path.join(HERE, "fin.txt"))
    return WM.write_component(os.path.join(tmp, "planar_rudder.txt"), ld.params, ld.design, ld.space,
                              {"sections": 41, "half": False}, "planar_rudder", ["a planar rudder for the checks"],
                              "rudder")


def planar_copy(tmp, sub):
    """outputs/wind_tunnel_rudder.txt (the planar set-up of 18 Nov 2026) and its section file copied into
    tmp/sub, so what is made beside it stays in tmp."""
    import shutil
    folder = os.path.join(tmp, sub)
    os.makedirs(folder, exist_ok=True)
    for name in ("wind_tunnel_rudder.txt", "wind_tunnel_rudder_section.dat"):
        shutil.copy(os.path.join(HERE, "outputs", name), folder)
    return os.path.join(folder, "wind_tunnel_rudder.txt")


def test_planar_original(tmp):
    print("\na planar rudder as the original modified below H (xgeom_from_cad.rudder_from_planar)")
    if not os.path.exists(os.path.join(HERE, "outputs", "wind_tunnel_rudder.txt")):
        print("       skipped: outputs/wind_tunnel_rudder.txt is not here")
        return
    import rudder_modify as RM
    import xgeom_from_cad as FC
    path = planar_copy(tmp, "orig")
    res = FC.rudder_from_planar(path, sections=21, progress=lambda text: None)
    ad, ld = res["adapter"], WM.load_component(path)
    print(f"       21 sections fitted in {res['seconds']:.1f} s; TE radius {ad.orig.te_radius:.4g} mm")
    out = ad.build_xcad("base", plot=False, cad=False)
    loops = [RM.xcad_to_working(pts, "m") for _, pts in RM.read_xcad(out["case"]["paths"]["xcad"])]
    nm = out["case"]["xcad"]["n_modified"]
    ys = np.array([lp[0, 1] for lp in loops[:nm]])
    ref = WM.section_loops(ld.design, ld.params, ys / ld.design.span, FC.PLANAR_POINTS)
    dev = max(float(np.abs(a - b).max()) for a, b in zip(loops[:nm], ref))
    z_top = ld.design.span * 20.0 / 20.5                       # the top of 21 sections under the tilted tip
    again = FC.rudder_from_planar(path, sections=21, progress=lambda text: None)
    check("its sections fitted as the original (the model's own rudder adapter, the files beside the planar one), "
          "the top section half a step under the tilted tip, its TE radius and 121 points a section in the "
          "settings, H from 0.1 to 0.4 of the top and at 0.4, the design the rudder itself: the written sections "
          "are the planar ones (mm); a second time it is read back, the same",
          [getattr(ad, "role", None) is None, os.path.basename(res["folder"]) == "wind_tunnel_rudder_modified_below_H",
           ad.orig.z_top - z_top, ad.settings["te_radius_mm"] - ad.orig.te_radius, ad.settings["num_pts"],
           ad.settings["h_min"] / z_top, ad.design.H / z_top, len(ad.preview()["problems"]), dev < 1e-4,
           again["cached"], max(abs(again["adapter"].values[k] - ad.values[k]) for k in ad.values)],
          [1, 1, 0, 0, 2 * FC.PLANAR_POINTS - 1, 0.1, 0.4, 0, 1, 1, 0], 1e-9)
    with open(path, "a") as fh:
        fh.write("\n")                                          # the file changed: made again
    third = FC.rudder_from_planar(path, sections=21, progress=lambda text: None)
    check("a changed file is made again", third["cached"], 0, 0)


def test_planar_rudder(tmp):
    print("\nthe planar rudder (xgeom_wing.PlanarRudderAdapter, xgeom_from_cad.wing_setup)")
    import xgeom_files as XF
    import xgeom_from_cad as FC
    import xgeom_vehicle as XV
    import xgeom_wing as XW
    path = planar_rudder_file(tmp)
    ad = XW.PlanarRudderAdapter(params=path)
    saved = ad.save_file(os.path.join(tmp, "planar_rudder_saved.txt"))
    ad2 = XW.PlanarRudderAdapter(params=saved)
    part = XV.rudder_part(ad)
    eta = np.linspace(0.0, 1.0, 41)
    v = ad.design.values(eta)
    th = np.deg2rad(v["pitch"])
    x_te = v["sweep"] + v["chord"] * (ad.params.ref_xc + (1.0 - ad.params.ref_xc) * np.cos(th))
    check("a rudder on the planar model: from a component file of kind rudder (none without one), Save set-up "
          "writes kind rudder and loads back, the vehicle's part the wing's with its TE line's slope",
          [ad.title.startswith("Rudder"), refused(XW.PlanarRudderAdapter), XF.component_kind(saved) == "rudder",
           max(abs(ad2.values[n] - ad.values[n]) for n in ad.values), len(part["loops"]) > 2,
           part["te_slope"] - np.polyfit(eta * ad.design.span, x_te, 1)[0], ad.out_dir.endswith("rudder_planar")],
          [1, 1, 1, 0, 1, 0, 1], 1e-9)
    params = WM.read_params(path)
    forms = FC.best_wing_forms(params)
    setup, forms2 = FC.wing_setup(params, os.path.join(tmp, "set_up.txt"), "rudder", "set_up")
    ld = WM.load_component(setup)
    space = ld.space
    inside = all(lo - 1e-9 <= ld.space.values(ld.design)[n] <= hi + 1e-9
                 for n, (lo, hi) in zip(space.all_names, space.all_bounds))
    free = {n.split(".")[0] for n in space.names}
    check("wing_setup: every curve in its best form (the fin's chord straight: 2 points, its zero rake 2), kind "
          "rudder, the variables of xgeom_wing.FREE free, every value inside its bounds",
          [forms == forms2, forms["chord"][0] == (1, 1), forms["rake"][0] == (1, 1), ld.kind == "rudder",
           free <= set(XW.FREE), inside, all(ld.design.curves[c].segments == f[0][0] for c, f in forms.items())],
          [1, 1, 1, 1, 1, 1, 1], 0)


def test_vehicle():
    print("\nfins and wings on the hull (xgeom_vehicle.py)")
    import hull_modify as HM
    import xgeom_vehicle as XV
    hp = HM.suboff_params()                              # SUBOFF's stations from its equations
    hull = HM.fit_design(hp)
    parts = {"hull": {"design": hull, "lines": [], "problems": []}, "fin": XV.wing_part(FakeAdapter(FIN)),
             "wing": XV.wing_part(FakeAdapter(WING))}
    pl = XV.default_placement(parts)
    j, L = hull.joins, hull.length
    check("defaults: one fin on top a tenth along the middle body, the wings level a quarter along it (x/L)",
          [pl.fins, pl.fin_angle, pl.fin_x, pl.wings, pl.wing_angle, pl.wing_x],
          [1, 0, round((j[1] + 0.1 * (j[2] - j[1])) / L, 5), 2, 90, round((j[1] + 0.25 * (j[2] - j[1])) / L, 5)],
          1e-12)
    pl.fins, pl.fin_angle, pl.wing_angle = 3, 30.0, 60.0
    segs, info = XV.assemble(parts, pl)
    chord = (parts["fin"]["root_x"][1] - parts["fin"]["root_x"][0]) / 1000.0
    out = []
    for k, a in enumerate(info["fin_frames"]):
        phi = np.radians(30.0 + 120.0 * k)
        x_le, x_te = info["fin_x_le"][k], info["fin_x_te"][k]
        r_le = XV.surface_distance(hull, np.array([x_le]), phi)[0]
        r_te = XV.surface_distance(hull, np.array([x_te]), phi)[0]
        out += [np.hypot(a[0][1], a[0][2]) - r_le, x_le - pl.fin_x * L, np.hypot(x_te - x_le, r_te - r_le) - chord,
                np.arctan2(a[0][1], a[0][2]) % (2 * np.pi) - phi]
    check("3 fins from 30 deg: each root LE on the hull at fin x in its own half-plane, the TE a root chord aft "
          "along the hull", out, 0.0, 1e-9)
    n = len(segs["wings"]) // 2
    a, ec, eh, et = info["wing_frames"][0]
    phi = np.radians(60.0)
    r_le = XV.surface_distance(hull, np.array([pl.wing_x * L]), phi)[0]
    check("the starboard wing at 60 deg on the hull (z up: et above the horizontal), the port one its mirror "
          "image", [np.hypot(a[1], a[2]) - r_le, a[1] > 0, et[2] > 0, eh @ [0, np.sin(phi), np.cos(phi)],
                    max(np.abs(p * [1, -1, 1] - q).max() for p, q in zip(segs["wings"][:n], segs["wings"][n:])),
                    len(segs["wing_roots"]) == 34],
          [0, 1, 1, 1, 0, 1], 1e-9)
    pl_t = XV.Placement(**{**pl.__dict__, "fin_x": float(j[3] - 0.4) / L, "fins": 1, "fin_angle": 0.0})
    segs_t, info_t = XV.assemble(parts, pl_t)
    x_le, x_te = info_t["fin_x_le"][0], info_t["fin_x_te"][0]
    r = XV.surface_distance(hull, np.array([x_le, x_te]), 0.0)
    check("a fin on the tail: pitched to the hull's slope under its root (the TE on the hull, a chord aft)",
          [info_t["fin_pitch"][0] - np.degrees(np.arctan2(r[0] - r[1], x_te - x_le)), info_t["fin_pitch"][0] > 5],
          [0, 1], 1e-9)
    probs = [XV.placement_problem({"wing_angle": a_}, L) for a_ in (0.0, 180.0, 200.0)]
    probs += [XV.placement_problem({"wings": 1}, L), XV.placement_problem({"fins": 13}, L),
              XV.placement_problem({"fin_x": 1.1}, L), XV.placement_problem({"wing_scale": 0.0}, L)]
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
        import modify_control as MC
        rudder_h = all(os.path.exists(os.path.join(HERE, p)) for p in (MC.CORNER_POINTS, MC.SECTION_TABLE,
                                                                       MC.XCAD_ORIGINAL, MC.STACK))
        if rudder_h and os.path.exists(os.path.join(HERE, "outputs", "wind_tunnel_rudder.txt")):
            import xgeom_from_cad as FC
            import xgeom_rudder as XR
            planar = planar_copy(tmp, "gui")
            keep_n, FC.PLANAR_SECTIONS = FC.PLANAR_SECTIONS, 21           # quicker fits for the check
            ask = XT.filedialog.askopenfilename
            try:
                app = XT.App(root, "vehicle", params=[planar])
                settle()
                app.switch("rudder")
                settle()
                made = os.path.join(os.path.dirname(planar), "wind_tunnel_rudder_modified_below_H")
                first = (type(app.ad).__name__, app.build_btn.cget("text"), "H" in [r.name for r in app.rows],
                         hasattr(app, "tangent_all"), os.path.dirname(os.path.abspath(app.ad.files["stack"])) == made)
                setup = XR.RudderAdapter(out_dir=tmp).save_space(os.path.join(tmp, "rh_design_space.json"))
                XT.filedialog.askopenfilename = lambda **kw: setup
                app.on_load_params()                               # a set-up of the model modified below H
                settle()
                second = (type(app.ad).__name__, app.ad.setup_file == setup, "H" in [r.name for r in app.rows])
                XT.filedialog.askopenfilename = lambda **kw: planar
                app.on_load_space()                                # the Set-up box: a planar rudder's file
                t0 = time.time()
                while app._job is not None and time.time() - t0 < 300.0:
                    settle(2)
                settle()
                third = (type(app.ad).__name__,
                         app.ad.setup_file == os.path.join(made, "wind_tunnel_rudder_original_design_space.json"),
                         "H" in [r.name for r in app.rows], hasattr(app, "tangent_all"),
                         "as the original modified below H" in app.status.cget("text"))
                app.book.select(XT.VEHICLE_TAB)
                app.update_views()
                settle()
            finally:
                XT.filedialog.askopenfilename = ask
                FC.PLANAR_SECTIONS = keep_n
            check("the Rudder slot: a planar rudder's file (the wind-tunnel rudder's) becomes the original of the "
                  "model modified below H, its panel as without it (H, tangent at H, 'XCAD rudder'), the original "
                  "made beside the file; a set-up of that model loads as before; the planar file through the Set-up "
                  "box's Load ... the same (read back, off the GUI thread); the Vehicle tab places it",
                  [first == ("RudderAdapter", "XCAD rudder", True, True, True), second == ("RudderAdapter", True, True),
                   third == ("RudderAdapter", True, True, True, True),
                   app.status.cget("background") != XT.STATUS["error"][0]], [1, 1, 1, 1], 0)
            app.root.update()
            for w in root.winfo_children():
                w.destroy()
        else:
            print("       the Rudder slot's models: skipped, the extraction's outputs or the wind-tunnel rudder's "
                  "planar set-up are not here")
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


def test_cap(tmp):
    print("\nthe tip cap of a planar surface (wing_modify's cap keys, tip_cap.py, the loft's tip patch)")
    import tip_cap as TC
    import xgeom_files as XF
    import xgeom_vehicle as XV
    import xgeom_wing as XW
    ld = WM.load_component(os.path.join(HERE, "fin.txt"))
    p, d = ld.params, ld.design
    cap = TC.ScaledCap(0.25, 0.27, -0.005, -0.02, 2.44, 2.32)                # BB2's stern plane's, about
    p.cap = cap
    path = WM.write_component(os.path.join(tmp, "capped.txt"), p, d, ld.space, {"sections": 21, "cap_loops": 12},
                              "capped", ["a fin with a scaled tip cap"], "rudder")
    comp = XF.read_component(path)
    back = WM.load_component(path)
    keys = XF.CAP_KEYS["scaled"]
    check("the cap keys written and read back: cap scaled and its six numbers; the cap_loops setting",
          [comp.geometry.get("cap") == "scaled", all(k in comp.geometry for k in keys),
           max(abs(a - b) for a, b in zip(back.params.cap.as_dict().values(), cap.as_dict().values())),
           back.settings.get("cap_loops")], [1, 1, 0, 12], 1e-9)
    rad = TC.CapParams(0.6, 3.0, -0.1, 0.1, 60.0, 0.7, 3.3)
    p.cap = rad
    path_r = WM.write_component(os.path.join(tmp, "capped_r.txt"), p, d, ld.space, {"sections": 21}, "capped_r", [], "fin")
    back_r = WM.load_component(path_r).params.cap
    flat = WM.cap_from_geometry({"cap": "radius", "cap_h_le_mm": "1", "cap_h_te_mm": "2"})
    check("the radius family (the wind-tunnel rudder's) round trips too; no crown radius: a flat face",
          [isinstance(back_r, TC.CapParams), max(abs(a - b) for a, b in zip(back_r.as_dict().values(), rad.as_dict().values())
                                                 if a is not None), flat.crown_radius == TC.INF, flat.edge_radius], [1, 0, 1, 0], 1e-9)
    bad = [{"cap_k_le": "0.25"},                                              # keys without a cap line
           {"cap": "scaled", "cap_k_le": "0.25"},                             # k_te missing
           {"cap": "scaled", "cap_k_le": "0.25", "cap_k_te": "0.25", "cap_h_le_mm": "1"},   # the other family's key
           {"cap": "scaled", "cap_k_le": "0.25", "cap_k_te": "x"},            # not a number
           {"cap": "scaled", "cap_k_le": "0.25", "cap_k_te": "0.25", "cap_p": "1"},         # p above 1
           {"cap": "round"}]                                                  # no such family
    check("bad cap lines refused: keys without 'cap', a missing k, a key of the other family, a word for a number, "
          "p not above 1, an unknown family; 'cap none' is a flat tip",
          [refused(WM.cap_from_geometry, g) for g in bad] + [WM.cap_from_geometry({"cap": "none"}) is None], [1] * 7, 0)
    p.cap = cap
    base = WM.cap_base(d, p)
    tip = WM.section_loops(d, p, [1.0])[0]
    cl = WM.cap_loops(d, p, WM.POINTS, 5)
    lv = WM.cap_levels(5)
    check("cap_base: the tip section's LE, TE and sides as the loops have them; cap_loops: 5 loops of the sections' "
          "point count, each through the tip's LE and TE, rising to CAP_TOP of the cap's height, the levels closer "
          "near the crest",
          [np.abs(base.to_3d(0.0, 0.0, 0.0) - tip[0]).max(), np.abs(base.to_3d(base.chord, 0.0, 0.0) - tip[WM.POINTS - 1]).max(),
           np.abs(base.to_3d(base.x_up, base.y_up, 0.0) - tip[:WM.POINTS]).max(), cl.shape[0], cl.shape[1],
           np.abs(cl[:, 0] - tip[0]).max(), np.abs(cl[:, WM.POINTS - 1] - tip[WM.POINTS - 1]).max(),
           (cl[-1, :, 1] - d.span).max() / float(TC.cap_top(base.x_up, cap, base).max()), lv[-1], (np.diff(lv) > 0).all(),
           np.diff(lv)[-1] < np.diff(lv)[0]],
          [0, 0, 0, 5, 2 * WM.POINTS - 1, 0, 0, WM.CAP_TOP, WM.CAP_TOP, 1, 1], 1e-9)
    out = WM.write_case(tmp, "cap1", d, p, ld.space, cad=False, settings={"sections": 21, "cap_loops": 12}, plot=False)
    loops = XL.read_loops(out["paths"]["xcad_m.dat"], "m")
    check("write_case: 21 sections then 12 cap loops in the XCAD file (the setting), the cap in the design record "
          "and the sections table's note",
          [out["cap_loops"], len(loops), np.ptp(loops[20][:, 2]), np.ptp(loops[21][:, 2]) > 0.0,
           json.load(open(out["paths"]["design.json"]))["globals"]["cap"]["cap"] == "scaled",
           "tip cap" in open(out["paths"]["sections.dat"]).read()], [12, 33, 0, 1, 1, 1], 1e-9)
    ad = XW.PlanarRudderAdapter(params=path)
    pv = ad.preview(skeleton=True)
    part = XV.rudder_part(ad)
    check("the adapter: the cap in its 3D view and info line, its CASE settings; the vehicle's part has the cap loops too",
          [pv["cap"] is not None and len(pv["cap"]) == 12, "tip cap" in ad.sections_info(), ad._case_settings()["cap_loops"],
           len(part["loops"]) - XV.LOOPS], [1, 1, 12, 6], 0)
    if not XL.occ_available():
        print("       the loft skipped: no OCCT binding")
        return
    out = WM.write_case(tmp, "cap2", d, p, ld.space, cad=True, settings={"sections": 21, "cap_loops": 40}, plot=False)
    lr = out["loft"]
    check("the loft of the capped fin: one valid closed solid of 6 faces (the tip patch over the cap, no flat top), "
          "the tip patch within 0.05 mm of the cap loops (a 220 mm chord), its lines at 3 deg or more (no fold), its "
          "volume above the sections' (the cap)",
          [lr.valid, lr.free_edges, XL.count_faces(lr.shape), lr.tip_error < 0.05, lr.tip_angle > 3.0,
           lr.volume > out["volume_mm3"]], [1, 0, 6, 1, 1, 1], 0)
    p.cap = rad
    out_r = WM.write_case(tmp, "cap3", d, p, ld.space, cad=True, settings={"sections": 21, "cap_loops": 40}, plot=False)
    lr_r = out_r["loft"]
    check("the same with a radius cap: valid, closed, 6 faces, no fold", [lr_r.valid, lr_r.free_edges, XL.count_faces(lr_r.shape),
                                                                           lr_r.tip_angle > 3.0], [1, 0, 6, 1], 0)
    p.cap = None


def test_tilt(tmp):
    print("\nthe tilted tip (tip_tilt_deg) and a planar rudder from an extraction folder")
    import tip_cap as TC
    import xgeom_files as XF
    ld = WM.load_component(os.path.join(HERE, "fin.txt"))         # straight taper: chord 300 to 220, sweep 0 to 50
    p, d = ld.params, ld.design
    p.tip_tilt, n = 4.0, WM.POINTS
    loop, le, ct = WM.tilted_tip(d, p)
    tan = np.tan(np.radians(4.0))
    # the TE line x = 300 - 30 y / 220 (sweep 50 y / 220 plus the chord 300 - 80 y / 220), the plane y = 220 + (x - 50) tan
    x_te = (300.0 - 30.0 * 220.0 / 220.0 + 30.0 * 50.0 * tan / 220.0) / (1.0 + 30.0 * tan / 220.0)
    eta = loop[:n, 1] / d.span
    v = WM.values_beyond(d, eta)
    xl = (loop[:n, 0] - v["sweep"]) / v["chord"]
    T, _ = WM.section_shape(v, p, xl, d.changes)
    zt = 0.5 * v["chord"] * np.diag(T)
    check("tilted_tip on a straight taper: the LE at the tip's LE, the TE on the TE line where the plane meets it, "
          "every point on the plane, its half-thickness the body's (the distributions carried on past the tip)",
          [np.abs(le - [50.0, 220.0, 0.0]).max(), loop[n - 1, 0] - x_te, np.abs(loop[:, 1] - 220.0 - (loop[:, 0] - 50.0) * tan).max(),
           np.abs(loop[:n, 2] - zt).max(), np.abs(loop[n - 1:, 2][::-1] + zt).max(), ct - (x_te - 50.0) / np.cos(np.radians(4.0))],
          [0, 0, 0, 0, 0, 0], 1e-6)
    st = WM.section_stations(9, tilted=True)
    check("tilted: the horizontal sections from the root to half a step under the tip's LE (k / (n - 1/2))",
          [st[0], st[-1], np.ptp(np.diff(st))], [0.0, 1.0 - 0.5 / 8.5, 0.0], 1e-12)
    p.cap = TC.CapParams(0.6, 3.0, -0.1, 0.1, 60.0, 0.7, 3.3)
    path = WM.write_component(os.path.join(tmp, "tilted.txt"), p, d, ld.space, {"sections": 21, "cap_loops": 10},
                              "tilted", [], "rudder")
    back = WM.load_component(path)
    base = WM.cap_base(d, p)
    cl = WM.cap_loops(d, p, n, 3)
    check("the key written and read back; cap_base on the tilted section (its frame turned by the tilt); the cap "
          "loops pinched to the tilted section's LE and TE",
          [back.params.tip_tilt, XF.read_component(path).geometry["tip_tilt_deg"] == "4",
           np.abs(base.n - [-np.sin(np.radians(4.0)), np.cos(np.radians(4.0)), 0.0]).max(), np.abs(base.pivot - le).max(),
           np.abs(cl[:, 0] - loop[0]).max(), np.abs(cl[:, n - 1] - loop[n - 1]).max()], [4.0, 1, 0, 0, 0, 0], 1e-9)
    bad = [{"tip_tilt_deg": "-1"}, {"tip_tilt_deg": "50"}, {"tip_tilt_deg": "a"}]
    comp = XF.read_component(path)
    refusals = []
    for g in bad:
        c2 = XF.read_component(path)
        c2.geometry.update(g)
        q = XF.write_component(os.path.join(tmp, "bad_tilt.txt"), c2)
        refusals.append(refused(WM.load_component, q))
    check("a tilt below 0, at or above TILT_MAX, or not a number is refused", refusals, [1, 1, 1], 0)
    out = WM.write_case(tmp, "tilt1", d, p, ld.space, cad=False, settings={"sections": 21, "cap_loops": 10}, plot=False)
    loops = XL.read_loops(out["paths"]["xcad_m.dat"], "m")
    check("write_case: 21 horizontal sections, the tilted section, 10 cap loops", [len(loops), np.ptp(loops[20][:, 2]) < 1e-9,
                                                                                   np.ptp(loops[21][:, 2]) > 1.0], [32, 1, 1], 0)
    p.cap, p.tip_tilt = None, 0.0
    folder = os.path.join(HERE, "outputs")
    need = ("corner_points.dat", "parameters.json")
    if all(os.path.exists(os.path.join(folder, f)) for f in need) and any(
            f.startswith("sections_stack_y0_y200") for f in os.listdir(folder)):
        import rudder_modify as RM
        import xgeom_from_cad as FC
        setup, report, forms = FC.planar_rudder_setup(folder, os.path.join(tmp, "wt_planar.txt"), "wt_planar")
        lw = WM.load_component(setup)
        loop_w, le_w, ct_w = WM.tilted_tip(lw.design, lw.params)
        stack = RM.read_stack(FC.rudder_files_partial(folder, "sections_stack_*_raw_mm.dat"))
        flat = [s for s in stack if s.tilt_deg == 0.0]
        worst = 0.0
        for s in flat[::20]:
            e = s.y / lw.design.span
            sec = WM.section_loops(lw.design, lw.params, [e], 101)[0]
            q = np.column_stack((s.xy[:, 0] + s.x_le, s.xy[:, 1]))
            worst = max(worst, float(RM._to_polyline(q, sec[:, [0, 2]]).max()))
        check("the wind-tunnel rudder from its extraction folder: kind rudder, the tilt and the radius cap of "
              "parameters.json, the tilted tip's TE top at (150.684, 208.543) and chord 136.950 as extracted, "
              "every 20th extracted section within 0.15 mm of the set-up's (the scaled TE round)",
              [lw.kind == "rudder", abs(lw.params.tip_tilt - 3.576626) < 1e-5, isinstance(lw.params.cap, TC.CapParams),
               abs(loop_w[n - 1, 0] - 150.683680) < 1e-4, abs(loop_w[n - 1, 1] - 208.543428) < 1e-4, abs(ct_w - 136.950424) < 1e-4,
               worst < 0.15], [1] * 7, 0)
        print(f"       extracted sections from the set-up's: {worst:.4f} mm at most")
    else:
        print("       the wind-tunnel rudder's part skipped: no extraction outputs in outputs/")
    if not XL.occ_available():
        print("       the loft skipped: no OCCT binding")
        return
    p.tip_tilt, p.cap = 4.0, TC.CapParams(0.6, 3.0, -0.1, 0.1, 60.0, 0.7, 3.3)
    lr = WM.write_case(tmp, "tilt2", d, p, ld.space, cad=True, settings={"sections": 21, "cap_loops": 40}, plot=False)["loft"]
    check("the loft of the tilted, capped fin: one valid closed solid of 6 faces, no fold",
          [lr.valid, lr.free_edges, XL.count_faces(lr.shape), lr.tip_angle > 3.0], [1, 0, 6, 1], 0)
    p.tip_tilt, p.cap = 0.0, None


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
            test_planar_rudder(tmp)
            test_cap(tmp)
            test_tilt(tmp)
            test_planar_original(tmp)
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
