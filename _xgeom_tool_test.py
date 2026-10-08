"""
_xgeom_tool_test.py

Self-check for the design tool: the CAD loft (xcad_loft.py), the rudder
adapter (xgeom_rudder.py) and, with --gui, the window itself (xgeom_tool.py).

    python _xgeom_tool_test.py          loft and adapter
    python _xgeom_tool_test.py --gui    also open the tool for a moment

The loft checks need pythonocc-core (or cadquery-ocp) and are skipped
without it; the adapter needs the extraction's outputs in outputs/ (the X_geom
repo) and is skipped without them.
"""

import os
import sys
import tempfile

import numpy as np

import replica_funcs as RF
import xcad_loft as XL

_FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<62s} max error {err:.3e}   (tol {tol:.0e})")


def wing_loops(n=40, tilt=True):
    """A tapered wing in the XCAD frame (x chordwise, y thickness, z height),
    mm: n horizontal loops from z = 0 to 100, then a tilted loop and a cap
    loop, each 200 points starting at the LE; and the section areas."""
    loops, areas, zs = [], [], np.linspace(0.0, 100.0, n)
    for z in zs:
        c = 150.0 - 0.4 * z
        xy = RF.rounded_section(0.2177, 0.30, 0.0734, 0.1235, 0.004, chord=c, chord_type="actual", num_pts=200)
        i_le = 99
        loop2 = np.vstack([xy[:i_le + 1][::-1], xy[i_le:][::-1][1:]])            # LE, +y side, TE, back
        loops.append(np.column_stack((0.05 * z + loop2[:, 0], -loop2[:, 1], np.full(200, z))))
        x, y = xy[:, 0], xy[:, 1]
        areas.append(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    if tilt:
        top = loops[-1].copy()
        top[:, 2] += 0.5 + 0.04 * (top[:, 0] - top[0, 0])                        # tilted, rising to the TE
        cap = top.copy()
        cap[:, 1] *= 0.3                                                          # a thin cap loop above it
        cap[:, 2] += 1.0 + 0.01 * (cap[:, 0] - cap[0, 0])
        loops += [top, cap]
    return loops, np.array(areas), zs


def test_loft():
    print("\nCAD loft (xcad_loft.py)")
    loops, areas, zs = wing_loops(tilt=False)
    fit = XL._SectionFitter(200, 40)
    err = max(fit.wire(lp)[1] for lp in loops) if XL.occ_available() else None
    a, _ = XL._basis(100, 40)
    pts = np.column_stack((np.linspace(0, 1, 100)**3, np.linspace(0, 1, 100)**2, np.linspace(0, 1, 100)))
    check("section fit: a cubic is reproduced, ends held", np.abs(a @ XL._fit_poles(pts, a) - pts).max(), 0.0, 1e-12)
    if not XL.occ_available():
        print("       skipped: no pythonocc-core or cadquery-ocp here")
        return
    check("section curves within 20 um of the loops' points", err < 0.02, 1, 0)
    res = XL.loft_loops(loops)
    vol_ref = float(np.sum(0.5 * (areas[1:] + areas[:-1]) * np.diff(zs)))
    print(f"       {res.n_smooth} loops smooth, {res.seconds:.1f} s, volume {res.volume:.1f} mm^3 "
          f"(trapezoid rule {vol_ref:.1f})")
    check("horizontal loops: a valid closed solid", [res.valid, res.free_edges, res.n_ruled], [1, 0, 0], 0)
    check("its volume matches the sections' (relative)", abs(res.volume / vol_ref - 1.0), 0.0, 5e-4)
    loops2, _, _ = wing_loops(tilt=True)
    res2 = XL.loft_loops(loops2)
    check("with a tilted loop and a cap loop: valid, closed, 8 faces",
          [res2.valid, res2.free_edges, res2.n_ruled, res2.volume > res.volume, XL.count_faces(res2.shape)],
          [1, 0, 2, 1, 8], 0)
    v, t = XL.tessellate(res2.shape, 0.5, 0.5)
    check("tessellation: triangles index the vertices", [len(t) > 500, t.min() >= 0, t.max() < len(v)], [1, 1, 1], 0)
    with tempfile.TemporaryDirectory() as tmp:
        step = XL.write_step(res2.shape, os.path.join(tmp, "wing.step"))
        back = XL.read_step(step)
        check("STEP written and read back: the same volume", XL.mass_properties(back)[0], res2.volume, 1e-5 * res2.volume)


def test_adapter():
    print("\nrudder adapter (xgeom_rudder.py)")
    need = ["outputs/corner_points.dat", "outputs/thickness_params_H80_info.dat",
            "outputs/sections_xcad_with_cap_y0_y200_m.dat", "outputs/sections_stack_y0_y200_raw_mm.dat"]
    missing = [p for p in need if not os.path.exists(os.path.join(HERE, p))]
    if missing:
        print("       skipped, not here: " + ", ".join(missing))
        return
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        import rudder_modify as RM
        import xgeom_rudder as XR
        ad = XR.RudderAdapter()
        check("starts from the original below H (no problems), P1x held",
              [len(ad.preview()["problems"]), ad.n_free("P1x")[0], ad.n_free("P2x")[0]], [0, 0, 4], 0)
        v = ad.set("LE.dx3", -500.0)
        check("a value is clamped into its bounds", v, ad.bounds("LE.dx3")[0], 0.0)
        ad.set("LE.dx3", -20.0)
        check("the LE root moves 20 mm forward", ad.design.curve("LE")([1.0])[0, 1], -20.0, 1e-9)
        ad.set_order("P2z", 5)
        rows = [r.name for r in ad.curve_rows("P2z") if r.kind == "var"]
        check("6 control points: d1 .. d4 and the values of C2 .. C5 (C1 on the tangent)",
              [len(rows), rows[0] == "P2z.d1", rows[-1] == "P2z.v5"], [8, 1, 1], 0)
        ad.set_tangent("P2z", False)
        check("off the tangent frees the value of C1", "P2z.v1" in [r.name for r in ad.curve_rows("P2z") if r.kind == "var"],
              1, 0)
        ad.set("P2z.v5", 0.085)
        ad.set_bounds("H", 40.0, 150.0)
        ad.set("H", 150.0)
        sk = ad.skeleton()
        tip = [lp for lp in sk["unchanged"] if np.ptp(lp[:, 1]) > 1e-9]
        check("H bounds 40 .. 150: H = 150 builds; the 3D view keeps the tip section and the 10 cap loops",
              [len(ad.preview()["problems"]), ad.get("H"), len(tip)], [0, 150.0, 11], 1e-9)
        try:
            ad.set_bounds("H", 40.0, 250.0)
            refused = False
        except ValueError:
            refused = True
        check("an H bound above the top horizontal section is refused", [refused, ad.bounds("H")[1]], [1, 150.0], 0)
        check("3D view: 49 modified sections + the cut at H, 70 .. 200 loops in all",
              [len(sk["modified"]), len(sk["cut"]), 70 <= sum(len(v) for v in sk.values()) <= 200], [49, 1, 1], 0)
        secs_before = RM.modified_sections(ad.design, ad.orig, 30)
        err = ad.set_degree(5)
        secs_after = RM.modified_sections(ad.design, ad.orig, 30)
        check("a section control point added: P4 x and z curves, the sections as before (mm)",
              [len(ad.names), "P4z" in ad.names, max(np.abs(a.selig - b.selig).max() for a, b in zip(secs_before,
                                                                                                  secs_after))],
              [9, 1, 0], 1e-9)
        print(f"       degree 4 -> 5 carried the design over, largest refit change {err:.1e}")
        ad.set_free("P3x.v2", False)
        n_free, n_all = ad.n_free()
        with tempfile.TemporaryDirectory() as tmp:
            ad.out_dir = tmp
            res = ad.build_xcad("t", plot=False, cad=XL.occ_available())
            check("XCAD: the case files are written (no thickness format: P2z moved off P1z)",
                  [os.path.exists(p) for p in res["case"]["paths"].values()] + ["params" in res["case"]["paths"]],
                  [1] * len(res["case"]["paths"]) + [0], 0)
            if XL.occ_available():
                lr = res["loft"]
                print(f"       solid: volume {lr.volume:.1f} mm^3, {lr.seconds:.1f} s")
                check("XCAD: a valid closed solid and its STEP file",
                      [lr.valid, lr.free_edges, os.path.exists(res["step"])], [1, 0, 1], 0)
            before = dict(ad.values)
            ad2 = XR.RudderAdapter()
            ad2.load_space(res["case"]["paths"]["space"])
            check("the design space JSON read back: degree, values, free flags, H bounds",
                  [ad2.degree, max(abs(ad2.values[k] - before[k]) for k in before), ad2.is_free("P3x.v2"),
                   ad2.settings["h_min"], ad2.settings["h_max"], ad2.n_free()[0] == n_free], [5, 0, 0, 40.0, 150.0, 1],
                  1e-12)
            setup = ad.save_space(os.path.join(tmp, "setup.json"))
            space3, design3, _ = RM.load_design_space(setup)
            check("Save set-up, then rudder_modify.load_design_space: the same free variables, values, bounds",
                  [space3.names == ad.free_space().names,
                   np.abs(space3.to_vector(design3) - ad.free_space().to_vector(ad.design)).max(),
                   np.abs(space3.bounds - ad.free_space().bounds).max()], [1, 0, 0], 1e-12)
            space = ad2.free_space()
            check("the optimiser's space leaves the held slots out", ["P3x.v2" in space.names, len(space)],
                  [0, n_free], 0)
        check("every curve stays pinned to the original at H",
              max(abs(ad.design.curve(c)([0.0])[0, 1] - ad.orig.value(c, ad.design.H)[0]) for c in ad.names), 0.0,
              1e-12)
    finally:
        os.chdir(cwd)


def test_span():
    print("\nthe full height (xgeom_rudder.set_span)")
    need = ["outputs/corner_points.dat", "outputs/thickness_params_H80_info.dat",
            "outputs/sections_xcad_with_cap_y0_y200_m.dat", "outputs/sections_stack_y0_y200_raw_mm.dat"]
    if any(not os.path.exists(os.path.join(HERE, p)) for p in need):
        print("       skipped, the extraction's outputs are not here")
        return
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        import rudder_modify as RM
        import xgeom_rudder as XR
        ad = XR.RudderAdapter()
        ad.set("LE.dx3", -10.0)
        span0, h0, (lo0, hi0), vals0 = ad.span, ad.design.H, ad.bounds("H"), dict(ad.values)
        secs0 = RM.modified_sections(ad.design, ad.orig, 20)
        r = ad.set_span(1.5 * span0)
        secs1 = RM.modified_sections(ad.design, ad.orig, 20)
        top = max(lp[:, 1].max() for v in ad.skeleton().values() for lp in v)
        check("full height x1.5: H and its bounds x1.5, the other variables kept, the sections x1.5 as high, "
              "the 3D loops to the new top",
              [r, ad.design.H / h0, ad.bounds("H")[0] / lo0, ad.bounds("H")[1] / hi0,
               max(abs(ad.values[n] - vals0[n]) for n in vals0 if n != "H"),
               max(abs(b.y - 1.5 * a.y) + np.abs(a.selig - b.selig).max() for a, b in zip(secs0, secs1)),
               top / (1.5 * span0), len(ad.preview()["problems"])], [1.5, 1.5, 1.5, 1.5, 0, 0, 1, 0], 1e-9)
        try:
            ad.set_span(0.1 * span0)
            refused = False
        except ValueError:
            refused = True
        with tempfile.TemporaryDirectory() as tmp:
            setup = ad.save_space(os.path.join(tmp, "s_design_space.json"))
            space, design, _ = RM.load_design_space(setup)
            ad2 = XR.RudderAdapter()
            ad2.load_space(setup)
            check("a full height of a tenth refused; the set-up keeps it: load_design_space and load_space",
                  [refused, space.orig.scale, design.H / h0, ad2.span / span0,
                   np.abs(space.to_vector(design) - ad.free_space().to_vector(ad.design)).max(),
                   max(abs(ad2.values[n] - ad.values[n]) for n in ad.values)], [1, 1.5, 1.5, 1.5, 0, 0], 1e-9)
        ad.set_span(span0)
        check("back to the extracted height: H and the stretch as before", [ad.design.H - h0, ad.orig.scale],
              [0, 1], 1e-9)
    finally:
        os.chdir(cwd)


def test_gui():
    print("\nthe window (xgeom_tool.py)")
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:                                   # no tkinter or no display
        print(f"       skipped: {exc}")
        return
    import xgeom_tool as XT
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        app = XT.App(root, "rudder")
        root.update()
        app.set_value("H", 60.0, None)
        app.update_views()
        root.update()
        app.curve.set("P3z")
        app.on_curve()
        app.npts.set(6)
        app.on_points()
        app.book.select(1)
        root.update()
        app.update_views()
        root.update()
        n3 = sum(len(c.get_segments()) for c in app.view3.col.values())
        check("the window follows a slider, a new section point, the 3D sections",
              [app.ad.get("H"), app.ad.degree, len(app.rows) > 3, 70 <= n3 <= 200], [60.0, 5, 1, 1], 1e-12)
        span0 = app.ad.span
        app.span_box.set("300")
        app.on_span()
        app.update_views()
        root.update()
        tops = max(s[:, 2].max() for c in app.view3.col.values() for s in c._segments3d)
        app.span_box.set("10")                                     # too low: refused, the box shows 300
        app.on_span()
        check("the full height box: 300 mm stretches the rudder (H 60 -> 60 x 300 / the extracted), 10 refused",
              [app.ad.span, app.ad.get("H") / (60.0 * 300.0 / span0), tops > 290.0, app.span_box.get() == "300.00",
               app.status.cget("background") == XT.STATUS["error"][0]], [300.0, 1, 1, 1, 1], 1e-9)
    finally:
        root.destroy()
        os.chdir(cwd)


def main():
    print("design tool self-check")
    test_loft()
    test_adapter()
    test_span()
    if "--gui" in sys.argv:
        test_gui()
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
