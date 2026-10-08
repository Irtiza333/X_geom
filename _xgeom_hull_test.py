"""
_xgeom_hull_test.py

Self-check for the hull in the design tool (xgeom_hull.py) and the vehicle (xgeom_vehicle.py): the adapter's
variables, orders, elliptical sections, set-up and parameter files and build; the placement of the propeller
and the rudder on the hull; with --gui the window in vehicle mode and the hull alone (xgeom_tool.py).

    python _xgeom_hull_test.py          adapter and placement
    python _xgeom_hull_test.py --gui    also open the tool for a moment
"""

import os
import sys
import tempfile

import numpy as np

import hull_modify as HM
import xcad_loft as XL
import xgeom_hull as XH
import xgeom_vehicle as XV

_FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))


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
    except ValueError:
        return True
    return False


def test_adapter(tmp):
    print("\nthe hull adapter (xgeom_hull.py)")
    ad = XH.HullAdapter()
    d0 = ad.design.copy()
    x = np.linspace(0.0, d0.length, 2001)
    check("SUBOFF: nose r and tail r, 13 of 16 variables free (lengths and curves), no problems",
          [ad.names == ["nose_r", "tail_r"], ad.n_free()[0], ad.n_free()[1], len(ad.preview()["problems"])],
          [1, 13, 16, 0], 0)
    v = ad.set("middle", 10.0)
    check("a value is clamped into its bounds; the length follows", [v, ad.design.length - d0.length],
          [ad.bounds("middle")[1], ad.bounds("middle")[1] - d0.middle], 1e-12)
    ad.set("middle", d0.middle)
    msg = ad.set_order("nose_r", 7)
    rows = [r.name for r in ad.curve_rows("nose_r") if r.kind == "var"]
    check("8 control points as fitted to the file: N1 and b0 .. b5, the file within 0.05 mm",
          [len(rows), rows[0] == "nose_r.N1", rows[-1] == "nose_r.b5", "parameter file" in msg,
           HM.fit_report(ad.params, ad.design)["nose_r"] * 1000 < 0.05], [7, 1, 1, 1, 1], 0)
    ad.set("tail_r.t3", ad.get("tail_r.t3") + 0.05)
    before = ad.design.radius("r", x)
    msg = ad.set_order("tail_r", 6)
    check("a changed curve keeps its shape at another order (mm)",
          [1000 * np.abs(ad.design.radius("r", x) - before).max(), "as it was" in msg], [0, 1], 0.3)
    check("bounds: low above high, a negative middle body and a zero exponent are refused",
          [refused(ad.set_bounds, "nose", 1.2, 1.1), refused(ad.set_bounds, "middle", -1.0, 3.0),
           refused(ad.set_bounds, "nose_r.N1", 0.0, 1.0)], [1, 1, 1], 0)
    ad.set_bounds("r", 0.2, 0.3)
    ad.set_free("r", True)
    ad.set("r", 0.27)
    check("a radius freed and moved: r* 0.27, the depth in 'where'", [ad.design.r, ad.row_info("r") == "2r 0.540"],
          [0.27, 1], 1e-12)
    ad.set("r", d0.r)
    shape = ad.design.radius("r", x)
    n_slots = ad.n_free()[1]
    ad.set_elliptic(True)
    check("elliptical sections: nose r' and tail r' as copies (r' = r everywhere), r'* and r'e held",
          [len(ad.names), np.abs(ad.design.radius("rp", x) - shape).max(), ad.n_free()[1] - n_slots,
           ad.is_free("rp"), ad.is_free("tail_rp.t6")], [4, 0, 2 + 7 + 6, 0, 1], 1e-12)
    ad.set("rp", 0.2)
    hs = HM.hydrostatics(ad.design)
    check("r'* 0.2: breadth 0.4 m, depth 0.508 m", [hs["breadth"], hs["depth"]], [0.4, 0.508], 1e-9)
    path = ad.save_params(os.path.join(tmp, "flat.dat"))
    ad2 = XH.HullAdapter(path)
    check("the elliptical design as a parameter file, loaded again (order 5): elliptic, within 0.3 mm",
          [ad2.elliptic, 1000 * max(HM.fit_report(ad2.params, ad2.design).values())], [1, 0], 0.3)
    ad.set_free("tail_rp.t2", False)
    setup = ad.save_space(os.path.join(tmp, "s_design_space.json"))
    ad3 = XH.HullAdapter()
    ad3.load_space(setup)
    sp, des, _ = HM.load_hull_space(setup)
    check("Save set-up, then load_space and load_hull_space: values, bounds, free flags, vector",
          [max(abs(ad3.values[k] - ad.values[k]) for k in ad.values),
           max(abs(ad3.bounds(k)[i] - ad.bounds(k)[i]) for k in ad.values for i in (0, 1)),
           ad3.is_free("tail_rp.t2"), sp.names == ad.free_space().names,
           np.abs(sp.to_vector(des) - ad.free_space().to_vector(ad.design)).max()], [0, 0, 0, 1, 0], 1e-12)
    ad.set_elliptic(False)
    check("circular sections again: 2 curves, r' = r",
          [len(ad.names), ad.design.rp is None, np.abs(ad.design.radius("rp", x) - ad.design.radius("r", x)).max()],
          [2, 1, 0], 0)
    ad.out_dir = tmp
    ad.settings["plot"] = False
    res = ad.build_task("b")()
    files = res["files"]
    check("build: the case files written; with OCC the STEP (its V as the stations', 1e-3)",
          [all(os.path.exists(p) for p in files.values()), ("step" in files) == XL.occ_available(),
           res["error"] == (not XL.occ_available())], [1, 1, 1], 0)
    if XL.occ_available():
        v_st = HM.hydrostatics(ad.design)["volume"]
        check("build: the solid's volume against the stations' (fraction)", res["hydrostatics"]["volume"] / v_st - 1,
              0.0, 1e-3)
    return XH.HullAdapter()


class FakeRudder:
    """A rudder part: a 100 mm by 200 mm plate (x chordwise, y the height), loops at y 0 and 200."""
    @staticmethod
    def part():
        t = np.linspace(0.0, 2.0 * np.pi, 41)
        loops = [np.column_stack((50.0 - 50.0 * np.cos(t), np.full_like(t, y), 6.0 * np.sin(t))) for y in (0.0, 200.0)]
        return {"loops": loops, "root_x": (0.0, 100.0), "problems": []}


def test_vehicle(hull):
    print("\nthe vehicle (xgeom_vehicle.py)")
    d = hull.design
    for phi, want in ((0.0, 0.2), (90.0, 0.3), (45.0, 0.2 * 0.3 / np.hypot(0.2, 0.3) * np.sqrt(2.0))):
        e = d.copy()
        e.curves["nose_rp"], e.curves["tail_rp"] = e.curves["nose_r"].copy(), e.curves["tail_r"].copy()
        e.r, e.rp, e.rpe = 0.2, 0.3, e.re
        got = XV.surface_distance(e, np.array([2.0]), np.deg2rad(phi))[0]
        check(f"the distance to an elliptical section (0.2 high, 0.3 wide) at {phi:g} deg", got, want, 1e-12)
    blade = {"blades": 3, "diameter": 2.0, "hub": (0.4, 0.6), "problems": [],
             "sections": [np.array([[-0.1, 0.0, 0.5], [0.1, 0.05, 0.9], [0.0, 0.0, 1.0]])]}
    parts = {"hull": XV.hull_part(hull), "blade": blade, "rudder": FakeRudder.part()}
    pl = XV.default_placement(parts)
    check("defaults: propeller D = r* (half the depth), the rudder's root TE a quarter D clear of the blades, "
          "4 rudders from 180 deg", [pl.prop_d, pl.rudder_dx, pl.rudder_angle, pl.rudders],
          [0.254, round(min(-d.cap, -0.1 * 0.254 / 2.0 - 0.25 * 0.254), 3), 180.0, 4], 1e-12)
    pl.rudders, pl.rudder_angle = 6, 30.0
    segs, info = XV.assemble(parts, pl)
    tops = segs["rudders"][1::2]                                    # the loop at y = 200 of each rudder
    angles = [np.degrees(np.arctan2(t[:, 1].mean(), t[:, 2].mean())) % 360 for t in tops]
    check("6 rudders, the first at 30 deg: 360/6 apart, each root on the hull",
          [len(tops)] + [a - (30.0 + 60.0 * k) for k, a in enumerate(angles)] + [len(info["rudder_root"])],
          [6] + [0.0] * 6 + [6], 1e-9)
    pl.rudders, pl.rudder_angle, pl.rudder_scale = 4, 45.0, 0.5
    segs, info = XV.assemble(parts, pl)
    L, s = d.length, 0.254 / 2.0
    tip = segs["blades"][1][2]                                      # the second blade's tip, turned by 120 deg
    check("propeller: scaled to D, on the axis at the plane, turned by 360/Z; hub radius scaled",
          [tip[0], np.hypot(tip[1], tip[2]), np.arctan2(tip[2], tip[1]) % (2 * np.pi),
           np.abs(segs["hub"][0][:, 1:]).max()],
          [L + pl.prop_dx, s * 1.0, np.deg2rad(90.0 + 120.0), s * 0.4], 1e-12)
    te, chord = L + pl.rudder_dx, 0.05                              # the plate at scale 0.5: chord 50 mm
    overlap = XV.ROOT_OVERLAP * chord
    errs, angles, clear, reach = [], [], [], []
    for k in range(4):
        root, top = segs["rudders"][2 * k], segs["rudders"][2 * k + 1]
        inner = segs["roots"][17 * k]                               # the root section's inner copy
        phi = np.deg2rad(45.0 + 90.0 * k)
        er, et = np.array([0.0, np.sin(phi), np.cos(phi)]), np.array([0.0, -np.cos(phi), np.sin(phi)])
        rho = info["rudder_root"][k]
        f = np.linspace(-1.0, 1.0, 41)                              # across the thickness, finer than the placement
        dense = np.vstack([root - np.outer(root @ et, et) + np.outer((root @ et) * fi, et) for fi in f])
        clear.append((np.hypot(dense[:, 1], dense[:, 2]) - d.radius("r", dense[:, 0])).min())
        w, x = inner @ et, inner[:, 0]
        reach.append((np.sqrt(np.clip(d.radius("r", x) ** 2 - w ** 2, 0.0, None)) - inner @ er).min() - overlap)
        errs += [rho - d.radius("r", np.array([te - chord]))[0], root[:, 0].max() - te, np.abs(root @ et).max() - 0.003,
                 np.abs(root @ er - rho).max(), np.abs(inner @ er - (rho - info["rudder_depth"][k])).max()]
        angles.append(np.degrees(np.arctan2(top[:, 1].mean(), top[:, 2].mean())) % 360)
    check("4 rudders from 45 deg, tangent: the root touches the hull at the LE (the highest point under it), TE at "
          "x, thickness; the inner copy at the extrusion depth", errs + [a - (45.0 + 90.0 * k)
                                                                          for k, a in enumerate(angles)], 0.0, 1e-9)
    check("no point of the root inside the hull (41 points across the thickness), one touching; the extrusion "
          "reaches at least the overlap into the hull everywhere, just that at its deepest gap",
          [min(clear), min(reach)], [0.0, 0.0], 1e-9)
    top = segs["rudders"][1]
    check("the rudder's height points away from the axis: the top loop 100 mm (scale 0.5) further out",
          np.abs(top @ np.array([0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]) - info["rudder_root"][0] - 0.1).max(),
          0.0, 1e-12)
    e = d.copy()                                                    # an elliptical middle body: 0.4 high, 0.6 wide
    e.curves["nose_rp"], e.curves["tail_rp"] = e.curves["nose_r"].copy(), e.curves["tail_r"].copy()
    e.r, e.rp, e.rpe = 0.2, 0.3, e.re
    hp = {"design": e, "lines": [np.zeros((2, 3))], "problems": []}
    pe = XV.Placement(rudder_dx=2.05 - e.length, rudders=4, rudder_angle=45.0, rudder_scale=0.5)
    segs, info = XV.assemble({"hull": hp, "rudder": FakeRudder.part()}, pe)
    q_min = []
    for k in range(4):
        root = segs["rudders"][2 * k]
        phi = np.deg2rad(45.0 + 90.0 * k)
        et = np.array([0.0, -np.cos(phi), np.sin(phi)])
        dense = np.vstack([root - np.outer(root @ et, et) + np.outer((root @ et) * fi, et)
                           for fi in np.linspace(-1.0, 1.0, 41)])
        q_min.append(((dense[:, 1] / 0.3) ** 2 + (dense[:, 2] / 0.2) ** 2).min())
    check("an elliptical body at 45 deg (the ellipse's own equation): every root point on or outside it, one on it",
          q_min, 1.0, 1e-9)


def test_parts(hull):
    print("\nthe propeller and the rudder as the Vehicle tab gets them (their adapters)")
    need = ["msc_blade_params.dat", "outputs/corner_points.dat", "outputs/thickness_params_H80_info.dat",
            "outputs/sections_xcad_with_cap_y0_y200_m.dat", "outputs/sections_stack_y0_y200_raw_mm.dat"]
    missing = [p for p in need if not os.path.exists(os.path.join(HERE, p))]
    if missing:
        print("       skipped, not here: " + ", ".join(missing))
        return
    import xgeom_blade as XB
    import xgeom_rudder as XR
    blade, rudder = XB.BladeAdapter(), XR.RudderAdapter()
    bp, rp = XV.blade_part(blade), XV.rudder_part(rudder)
    st = blade.stations()
    n_sec = len(np.unique(np.r_[st[::XV.BLADE_EVERY], st[-1]]))
    check("the blade: every 4th XCAD section and the last, 53 points; the rudder: 30 loops, root LE at x 0 (mm)",
          [len(bp["sections"]), bp["sections"][0].shape[0], len(bp["problems"]), len(rp["loops"]),
           rp["root_x"][0], len(rp["problems"])], [n_sec, 53, 0, XV.LOOPS, 0, 0], 1e-6)
    parts = {"hull": XV.hull_part(hull), "blade": bp, "rudder": rp}
    pl = XV.default_placement(parts)
    segs, info = XV.assemble(parts, pl)
    L = hull.design.length
    front = min(float(p[:, 0].min()) for p in segs["blades"]) - L
    le, te = info["rudder_le"], info["rudder_te"]
    r_le, r_te = hull.design.radius("r", np.array([le, te]))
    check("the default placement: the root TE a quarter D ahead of the blades; tangent at the root LE (the largest "
          "radius under it); extruded by the gap to the TE's radius and 2 % of the chord (m, 0.05 mm)",
          [pl.rudder_dx, info["rudder_root"][0], info["rudder_depth"][0], info["gap"] > 0],
          [round(min(-hull.design.cap, front - 0.25 * pl.prop_d), 3), r_le, r_le - r_te + XV.ROOT_OVERLAP * (te - le), 1],
          5e-5)
    print(f"       the wind-tunnel rudder on SUBOFF: root {1000 * r_le:.1f} mm from the axis, extruded "
          f"{1000 * info['rudder_depth'][0]:.1f} mm inwards")


def test_gui():
    print("\nthe window (xgeom_tool.py: the vehicle, the hull alone)")
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
        app = XT.App(root, "vehicle")
        root.update()
        app.set_value("middle", 2.0, None)
        app.curve.set("tail_r")
        app.on_curve()
        app.nctrl.set(8)
        app.on_order()
        n_hull = len(app.rows)
        app.switch("rudder")
        app.switch("blade")
        app.switch("hull")
        check("switching: the hull keeps its state (middle 2 m, tail r of 8 points, picked) and its panel",
              [app.ad.get("middle"), app.ad.order("tail_r"), app.curve.get() == "tail_r", len(app.rows) == n_hull],
              [2.0, 7, 1, 1], 1e-12)
        app.book.select(XT.VEHICLE_TAB)
        root.update()
        app.update_views()
        root.update()
        n = {k: len(c.get_segments()) for k, c in app.viewv.col.items()}
        per = n["rudders"] // 4
        check("the Vehicle tab: hull, blades and hub, 4 rudders with their root extrusions; the status ok",
              [n["hull"] > 40, n["blades"] > 20, n["hub"], n["rudders"] == 4 * per, 10 <= per <= XV.LOOPS,
               n["roots"], app.status.cget("background") == XT.STATUS["ok"][0]], [1, 1, 14, 1, 1, 4 * 17, 1], 0)
        reach0 = max(np.hypot(q[:, 1], q[:, 2]).max() for q in app.viewv.col["rudders"]._segments3d)
        app.switch("rudder")
        app.span_box.set(f"{1.5 * app.ad.span:.2f}")
        app.on_span()
        app.update_views()
        root.update()
        reach1 = max(np.hypot(q[:, 1], q[:, 2]).max() for q in app.viewv.col["rudders"]._segments3d)
        span1 = app.ad.span
        app.span_box.set(f"{span1 / 1.5:.4f}")
        app.on_span()
        app.switch("hull")
        check("the rudder's full height x1.5 in its panel: the vehicle's rudders reach 1.5 x as far from the root",
              (reach1 - info0_root(app)) / (reach0 - info0_root(app)), 1.5, 2e-3)
        app.vset["rudders"].set(6)
        app.vset["rudder_angle"].set(90)
        app.vshow.set("stern")
        app.on_vehicle()
        app.update_views()
        root.update()
        n6 = len(app.viewv.col["rudders"].get_segments())
        app.vset["rudders"].set(13)                                 # more than MAX_RUDDERS: refused
        app.on_vehicle()
        refused13 = app.status.cget("background") == XT.STATUS["error"][0] and app.vset["rudders"].get() == "6"
        check("placement: 6 rudders from 90 deg, the stern; 13 refused (the box shows 6)",
              [app.placement.rudders, app.placement.rudder_angle, app.placement.view == "stern", n6 == 6 * per,
               refused13], [6, 90.0, 1, 1, 1], 0)
        app.switch("blade")
        n_sec = len(app._vehicle_part("blade")["sections"])
        app.nblades.set(7)
        app.on_blades()
        app.update_views()
        root.update()
        n7 = len(app.viewv.col["blades"].get_segments())
        app.switch("hull")
        check("blades Z 7 in the propeller panel: the vehicle draws 7 blades; Z kept when the hull is shown",
              [app.adapters["blade"].params.blades, n7, app.adapters["blade"].file_blades], [7, 7 * n_sec, 5], 0)
        app.vset["prop_d"].set(-1)
        app.on_vehicle()
        check("a negative propeller D is refused (red), the box shows the D in use",
              [app.status.cget("background") == XT.STATUS["error"][0], float(app.vset["prop_d"].get())],
              [1, app.placement.prop_d], 1e-12)
        app.elliptic.set(True)
        app.on_elliptic()
        app.set_value("rp", 0.2, None)
        app.update_views()
        check("elliptical hull in the vehicle: 4 curves, the vehicle drawn again", [len(app.ad.names),
              app.viewv.ax.get_ylim()[1] > 0.0], [4, 1], 0)
        settle(root)
        root.destroy()
        root = tk.Tk()
        app = XT.App(root, "hull")
        root.update()
        app.book.select(1)
        root.update()
        app.update_views()
        root.update()
        n3 = sum(len(c.get_segments()) for c in app.view3.col.values())
        check("the hull alone: no switcher, no Vehicle tab; its 3D sections", [app.book.index("end"), n3 > 60],
              [2, 1], 0)
    finally:
        settle(root)
        root.destroy()
        os.chdir(cwd)


def info0_root(app):
    """The first rudder's root distance from the axis in the Vehicle tab (m)."""
    parts = {k: app._vehicle_part(k) for k in app.kinds if k in app.adapters}
    return XV.assemble(parts, app.placement)[1]["rudder_root"][0]


def settle(root):
    """Let the window's pending updates run before it is closed."""
    import time
    for _ in range(5):
        root.update()
        time.sleep(0.03)


def main():
    print("hull in the tool, the vehicle: self-check")
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            hull = test_adapter(tmp)
            test_vehicle(hull)
            test_parts(hull)
    finally:
        os.chdir(cwd)
    if "--gui" in sys.argv:
        test_gui()
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
