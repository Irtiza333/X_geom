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
        return {"loops": loops, "root": loops[0], "root_x": (0.0, 100.0), "problems": []}


def frame(phi):
    """ex, er (away from the axis at phi) and et (the rudder's thickness) of a rudder at phi (rad)."""
    return (np.array([1.0, 0.0, 0.0]), np.array([0.0, np.sin(phi), np.cos(phi)]),
            np.array([0.0, -np.cos(phi), np.sin(phi)]))


def gaps_along(pts, eh, outside, reach=0.05):
    """Independent of xgeom_vehicle: how far each point moves along -eh before it is inside the hull
    (outside(p) True outside it), 0 for a point already inside; bisection over [0, reach]."""
    out = []
    for p in pts:
        if not outside(p):
            out.append(0.0)
            continue
        lo, hi = 0.0, reach
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if outside(p - mid * eh) else (lo, mid)
        out.append(0.5 * (lo + hi))
    return np.array(out)


def check_rudders(name, segs, info, angles, te, chord, height, radii):
    """Each rudder (the FakeRudder plate): its root's LE and TE on the hull in its half-plane (radii(x, phi): the
    hull's r and r' there), a chord apart, the TE at x; its height along the normal to the root chord in that
    half-plane (the pitch, the root chord's angle to the axis); the inner copy at the largest gap under the
    root (found here by bisection) plus the overlap, wholly inside the hull."""
    errs, inside = [], []
    for k, phi in enumerate(angles):
        root, top, inner = segs["rudders"][2 * k], segs["rudders"][2 * k + 1], segs["roots"][17 * k]
        ex, er, et = frame(phi)
        le, tp = root[0], root[20]                                  # the plate's LE (t = 0) and TE (t = pi)

        def q(p):                                                   # > 1 outside the hull's section at p's x
            r, rp = radii(p[0])
            return (p[1] / rp) ** 2 + (p[2] / r) ** 2

        rho_le, rho_te = le @ er, tp @ er
        ec = (tp - le) / chord
        eh = -(ec @ er) * ex + (ec @ ex) * er
        gap = gaps_along(root, eh, lambda p: q(p) > 1.0).max()
        depth = gap + XV.ROOT_OVERLAP * chord
        errs += [q(le) - 1.0, q(tp) - 1.0, le @ et, tp @ et, tp[0] - te, np.linalg.norm(tp - le) - chord,
                 np.abs(top - root - height * eh).max(),
                 info["rudder_pitch"][k] - np.degrees(np.arctan2(rho_le - rho_te, tp[0] - le[0])),
                 info["rudder_gap"][k] - gap, np.abs(inner - (root - depth * eh)).max()]
        inside.append(max(q(p) for p in inner) < 1.0)
    check(name, errs + [all(inside) - 1.0], 0.0, 1e-9)
    return [info["rudder_pitch"][k] for k in range(len(angles))]


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
    check("6 rudders, the first at 30 deg: 360/6 apart, each on the hull",
          [len(tops)] + [a - (30.0 + 60.0 * k) for k, a in enumerate(angles)] + [len(info["rudder_pitch"])],
          [6] + [0.0] * 6 + [6], 1e-9)
    pl.rudders, pl.rudder_angle, pl.rudder_scale = 4, 45.0, 0.5
    segs, info = XV.assemble(parts, pl)
    L, s = d.length, 0.254 / 2.0
    tip = segs["blades"][1][2]                                      # the second blade's tip, turned by 120 deg
    check("propeller: scaled to D, on the axis at the plane, turned by 360/Z; hub radius scaled",
          [tip[0], np.hypot(tip[1], tip[2]), np.arctan2(tip[2], tip[1]) % (2 * np.pi),
           np.abs(segs["hub"][0][:, 1:]).max()],
          [L + pl.prop_dx, s * 1.0, np.deg2rad(90.0 + 120.0), s * 0.4], 1e-12)

    def round_hull(x):
        r = d.radius("r", np.array([x]))[0]
        return r, r

    phis = [np.deg2rad(45.0 + 90.0 * k) for k in range(4)]          # the plate at scale 0.5: chord 50, 100 mm high
    p0 = check_rudders("4 rudders from 45 deg (TE at the start of the cap): root LE and TE on the hull, a chord "
                       "apart; height square to the root chord; extruded by the largest gap + overlap, inside",
                       segs, info, phis, L + pl.rudder_dx, 0.05, 0.1, round_hull)
    tops = segs["rudders"][1::2]
    check("each in its own half-plane (45 + 90 k deg), the four pitched alike on the round hull",
          [np.degrees(np.arctan2(t[:, 1].mean(), t[:, 2].mean())) % 360 - (45.0 + 90.0 * k) for k, t in enumerate(tops)]
          + [p - p0[0] for p in p0], 0.0, 1e-9)
    pl.rudder_dx = -0.6
    segs, info = XV.assemble(parts, pl)
    p1 = check_rudders("moved forward along the tail (TE at L - 0.6 m): on the hull again, pitched to the slope there",
                       segs, info, phis, L - 0.6, 0.05, 0.1, round_hull)
    pl.rudder_dx = 2.05 - L
    segs, info = XV.assemble(parts, pl)
    p2 = check_rudders("on the parallel middle body: on the hull, its sides 3 mm out over the curve",
                       segs, info, phis, 2.05, 0.05, 0.1, round_hull)
    check("the pitch follows the hull: changed along the tail, 0 on the middle body, where "
          "the gap is the thickness's over the curve, r* - sqrt(r*^2 - 3 mm^2)",
          [abs(p1[0] - p0[0]) > 1.0, p2[0], info["rudder_gap"][0] - (d.r - np.sqrt(d.r ** 2 - 0.003 ** 2))],
          [1, 0, 0], 1e-9)
    print(f"       pitch {p0[0]:.2f} deg (TE at the start of the cap), {p1[0]:.2f} deg (L - 0.6 m), "
          f"{p2[0]:.2f} deg (middle body)")
    e = d.copy()                                                    # an elliptical middle body: 0.4 high, 0.6 wide
    e.curves["nose_rp"], e.curves["tail_rp"] = e.curves["nose_r"].copy(), e.curves["tail_r"].copy()
    e.r, e.rp, e.rpe = 0.2, 0.3, e.re
    hp = {"design": e, "lines": [np.zeros((2, 3))], "problems": []}

    def ellipse(x):
        return e.radius("r", np.array([x]))[0], e.radius("rp", np.array([x]))[0]

    pe = XV.Placement(rudder_dx=2.05 - e.length, rudders=4, rudder_angle=45.0, rudder_scale=0.5)
    segs, info = XV.assemble({"hull": hp, "rudder": FakeRudder.part()}, pe)
    pm = check_rudders("an elliptical body (0.4 high, 0.6 wide), its middle, 4 rudders from 45 deg: on it (by the "
                       "ellipse's own equation)", segs, info, phis, 2.05, 0.05, 0.1, ellipse)
    pe.rudder_dx, pe.rudder_angle = -0.3, 30.0
    segs, info = XV.assemble({"hull": hp, "rudder": FakeRudder.part()}, pe)
    pt = check_rudders("its tail (TE at L - 0.3 m), 4 rudders from 30 deg: each on it in its own half-plane",
                       segs, info, [np.deg2rad(30.0 + 90.0 * k) for k in range(4)], e.length - 0.3, 0.05, 0.1, ellipse)
    check("the elliptical body: pitch 0 on its middle; on its tail each rudder pitched to the slope under it "
          "(30 and 120 deg differ, 30 and 210 deg alike)", [max(abs(p) for p in pm), abs(pt[0] - pt[1]) > 0.1,
                                                         pt[0] - pt[2], pt[1] - pt[3]], [0, 1, 0, 0], 1e-9)


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
    d = hull.design
    a, ec, eh, et = info["rudder_frames"][0]
    root, sc = rp["root"], pl.rudder_scale / 1000.0
    xa, y0 = rp["root_x"][0], float(root[:, 1].mean())
    whole = a + np.outer(sc * (root[:, 0] - xa), ec) + np.outer(sc * (root[:, 1] - y0), eh) + np.outer(sc * root[:, 2], et)
    chord = sc * (rp["root_x"][1] - rp["root_x"][0])
    x_le, x_te = info["rudder_x_le"][0], info["rudder_x_te"]
    r_le, r_te = d.radius("r", np.array([x_le, x_te]))
    gap = gaps_along(whole, eh, lambda p: np.hypot(p[1], p[2]) > d.radius("r", np.array([p[0]]))[0]).max()
    inner = segs["roots"][0]
    inside = bool((np.hypot(inner[:, 1], inner[:, 2]) < d.radius("r", inner[:, 0])).all())
    tip = a + chord * ec
    check("the default placement: the root TE a quarter D ahead of the blades; the wind-tunnel rudder's root LE and "
          "TE on SUBOFF, a root chord apart, pitched to the line between them; extruded by the largest gap (found "
          "here) + 2 % of the chord, inside the hull; clear of the blades",
          [pl.rudder_dx, np.hypot(a[1], a[2]) - r_le, np.hypot(tip[1], tip[2]) - r_te, tip[0] - x_te,
           info["rudder_pitch"][0] - np.degrees(np.arctan2(r_le - r_te, x_te - x_le)), info["rudder_gap"][0] - gap,
           info["rudder_depth"][0] - gap - XV.ROOT_OVERLAP * chord, inside, info["gap"] > 0],
          [round(min(-d.cap, front - 0.25 * pl.prop_d), 3), 0, 0, 0, 0, 0, 0, 1, 1], 1e-9)
    print(f"       the wind-tunnel rudder on SUBOFF: root LE at x {x_le:.4f} m, {1000 * r_le:.1f} mm from the axis; "
          f"TE at x {x_te:.4f} m, {1000 * r_te:.1f} mm; pitched {info['rudder_pitch'][0]:.2f} deg; extruded "
          f"{1000 * info['rudder_depth'][0]:.1f} mm (largest gap {1000 * gap:.1f} mm); {1000 * info['gap']:.0f} mm "
          f"clear of the blades")


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
        h0 = rudder_height(app, per)
        app.switch("rudder")
        span0 = app.ad.span
        app.span_box.set(f"{1.5 * span0:.2f}")
        app.on_span()
        app.update_views()
        root.update()
        h1 = rudder_height(app, per)
        span1 = app.ad.span
        app.span_box.set(f"{span0:.5f}")
        app.on_span()
        app.switch("hull")
        check("the rudder's full height x1.5 in its panel: the vehicle's rudders 1.5 x as high along their pitched "
              "height", [h1 / h0 - span1 / span0, abs(span1 / span0 - 1.5) < 1e-4], [0, 1], 1e-9)
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
        dx = app.placement.rudder_dx
        app.vset["rudder_dx"].set(0.05)                             # the root TE behind the hull's end: refused
        app.on_vehicle()
        check("a root TE off the hull (x - L above 0) is refused (red), the box shows the x in use",
              [app.status.cget("background") == XT.STATUS["error"][0], app.placement.rudder_dx - dx,
               float(app.vset["rudder_dx"].get()) - dx], [1, 0, 0], 1e-12)
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


def rudder_height(app, per):
    """The first rudder's height in the Vehicle tab (m): how far its drawn loops (the first `per` of the
    collection) reach from its root LE along its pitched height (its frame from xgeom_vehicle.assemble)."""
    parts = {k: app._vehicle_part(k) for k in app.kinds if k in app.adapters}
    a, ec, eh, et = XV.assemble(parts, app.placement)[1]["rudder_frames"][0]
    return max(float(((np.asarray(q) - a) @ eh).max()) for q in app.viewv.col["rudders"]._segments3d[:per])


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
