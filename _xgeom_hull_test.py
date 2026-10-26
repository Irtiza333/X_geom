"""
_xgeom_hull_test.py

Self-check for the hull in the design tool (xgeom_hull.py) and the vehicle (xgeom_vehicle.py): the adapter's
variables, orders, elliptical sections, set-up and parameter files and build; the placement of the propeller
and the rudder on the hull; with --gui the window in vehicle mode (Save vehicle CAD with the vehicle's CAD
left out) and the hull alone (xgeom_tool.py).

    python _xgeom_hull_test.py          adapter and placement; with OCC the vehicle's CAD (hull and rudders)
    python _xgeom_hull_test.py --gui    also open the tool for a moment
    python _xgeom_hull_test.py --vehicle-cad
                                        also the whole vehicle's CAD with the propeller (pythonocc-core, a few
                                        minutes)
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
    check("r'* 0.2: breadth 0.4 m, depth 0.508 m (the curves may rise above r* by 0.1 um)",
          [hs["breadth"], hs["depth"]], [0.4, 0.508], 1e-7)
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
    """A rudder part: a 100 mm by 200 mm plate (x chordwise, y the height), loops at y 0 and 200; sloped: its
    top loop from x 14 to 71.6 (the TE line's dx/dy -0.142, the wind-tunnel rudder's)."""
    @staticmethod
    def part(sloped=False):
        t = np.linspace(0.0, 2.0 * np.pi, 41)
        lo, hi = (14.0, 71.6) if sloped else (0.0, 100.0)
        loops = [np.column_stack((a + 0.5 * (b - a) * (1.0 - np.cos(t)), np.full_like(t, y), 6.0 * np.sin(t)))
                 for a, b, y in ((0.0, 100.0, 0.0), (lo, hi, 200.0))]
        return {"loops": loops, "root": loops[0], "root_x": (0.0, 100.0), "problems": []}


def frame(phi):
    """ex, er (away from the axis at phi) and et (the rudder's thickness) of a rudder at phi (rad)."""
    return (np.array([1.0, 0.0, 0.0]), np.array([0.0, np.sin(phi), np.cos(phi)]),
            np.array([0.0, -np.cos(phi), np.sin(phi)]))


def gaps_along(pts, eh, outside, reach=0.05):
    """Independent of xgeom_vehicle: how far each point moves along -eh before it is inside the hull
    (outside(p) True outside it), 0 for a point already inside; bisection over [0, reach] (a number, or one
    per point)."""
    out = []
    for p, top in zip(pts, np.broadcast_to(np.asarray(reach, dtype=float), (len(pts),))):
        if not outside(p):
            out.append(0.0)
            continue
        lo, hi = 0.0, float(top)
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


def check_raked(name, segs, info, angles, x_hull, rake, slope, chord, radii):
    """Each rudder placed with a set TE rake (a FakeRudder plate): its TE line (the root's TE to the top's) at
    the rake from square to the axis in its half-plane, positive with the top aft, through the hull's profile at
    x_hull (radii(x): the hull's r and r' there), the root's TE on it above the hull; the root chord at
    rake - atan(slope) to the axis; resting on the hull: the smallest gap under the root along the TE line 0
    (found here by bisection); the inner copy along the TE line at the largest gap plus the overlap, wholly
    inside the hull."""
    errs, inside = [], []
    for k, phi in enumerate(angles):
        root, top, inner = segs["rudders"][2 * k], segs["rudders"][2 * k + 1], segs["roots"][17 * k]
        ex, er, et = frame(phi)
        le, te = root[0], root[20]                                  # the plate's LE (t = 0) and TE (t = pi)
        u = (top[20] - te) / np.linalg.norm(top[20] - te)
        r, rp = radii(x_hull)
        on = x_hull * ex + r * rp / np.hypot(r * np.sin(phi), rp * np.cos(phi)) * er
        lam = (te - on) @ u

        def q(p):                                                   # > 1 outside the hull's section at p's x
            r_, rp_ = radii(p[0])
            return (p[1] / rp_) ** 2 + (p[2] / r_) ** 2

        g = gaps_along(root, u, lambda p: q(p) > 1.0, reach=(root @ er) / (u @ er))     # down to the axis
        depth = g.max() + XV.ROOT_OVERLAP * chord
        pitch = np.degrees(np.arctan2(le @ er - te @ er, te[0] - le[0]))
        errs += [np.linalg.norm(u - np.sin(np.deg2rad(rake)) * ex - np.cos(np.deg2rad(rake)) * er),
                 np.linalg.norm(te - on - lam * u), min(lam, 0.0), le @ et, te @ et,
                 pitch - rake + np.degrees(np.arctan(slope)), info["rudder_pitch"][k] - pitch,
                 info["rudder_rake"][k] - rake, info["rudder_slope"][k] - slope, g.min(),
                 info["rudder_gap"][k] - g.max(), np.abs(inner - (root - depth * u)).max()]
        inside.append(max(q(p) for p in inner) < 1.0)
    check(name, errs + [all(inside) - 1.0], 0.0, 1e-9)
    return [info["rudder_pitch"][k] for k in range(len(angles))]


def test_rake(hull):
    print("\nthe rudders' TE rake set (xgeom_vehicle.py)")
    d = hull.design
    L = d.length

    def round_hull(x):
        r = d.radius("r", np.array([x]))[0]
        return r, r

    parts = {"hull": XV.hull_part(hull), "rudder": FakeRudder.part()}
    pl = XV.Placement(rudder_dx=-d.cap, rudders=4, rudder_angle=45.0, rudder_scale=0.5, rudder_rake=0.0)
    phis = [np.deg2rad(45.0 + 90.0 * k) for k in range(4)]
    segs, info = XV.assemble(parts, pl)
    p0 = check_raked("TE rake 0 (TE at the start of the cap): the TE square to the axis through the hull there, the "
                     "plate's root chord along the axis, resting on the hull, extruded along the TE line, inside",
                     segs, info, phis, L - d.cap, 0.0, 0.0, 0.05, round_hull)
    pl.rudder_rake, pl.rudder_dx = 15.0, -0.6
    segs, info = XV.assemble(parts, pl)
    p1 = check_raked("TE rake 15 deg (top aft), TE line at L - 0.6 m: as set, the root chord at 15 deg, resting",
                     segs, info, phis, L - 0.6, 15.0, 0.0, 0.05, round_hull)
    sloped = {"hull": parts["hull"], "rudder": FakeRudder.part(sloped=True)}
    pl.rudder_rake = 0.0
    segs, info = XV.assemble(sloped, pl)
    b = (71.6 - 100.0) / 200.0
    p2 = check_raked("a TE sloped in the rudder (dx/dy -0.142), rake 0: the TE square to the axis, the root chord at "
                     "8.08 deg (atan 0.142), resting", segs, info, phis, L - 0.6, 0.0, b, 0.05, round_hull)
    pl.rudder_rake = None
    segs_h, info_h = XV.assemble(sloped, pl)
    check("pitched to the hull (no rake set): the TE's rake reported is the pitch plus atan(dx/dy); the plate's "
          "pitch the hull's", [info_h["rudder_rake"][0] - info_h["rudder_pitch"][0] - np.degrees(np.arctan(b)),
                               p0[0], p1[0] - 15.0, p2[0] - np.degrees(np.arctan(0.142))], 0.0, 1e-9)
    e = d.copy()                                                    # an elliptical middle body: 0.4 high, 0.6 wide
    e.curves["nose_rp"], e.curves["tail_rp"] = e.curves["nose_r"].copy(), e.curves["tail_r"].copy()
    e.r, e.rp, e.rpe = 0.2, 0.3, e.re

    def ellipse(x):
        return e.radius("r", np.array([x]))[0], e.radius("rp", np.array([x]))[0]

    pe = XV.Placement(rudder_dx=-0.3, rudders=4, rudder_angle=30.0, rudder_scale=0.5, rudder_rake=-10.0)
    hp = {"design": e, "lines": [np.zeros((2, 3))], "problems": []}
    segs, info = XV.assemble({"hull": hp, "rudder": FakeRudder.part(sloped=True)}, pe)
    check_raked("an elliptical body's tail, 4 rudders from 30 deg, TE rake -10 deg (top forward): each in its own "
                "half-plane, resting on the ellipse", segs, info, [np.deg2rad(30.0 + 90.0 * k) for k in range(4)],
                e.length - 0.3, -10.0, b, 0.05, ellipse)
    sub = FakeRudder.part()
    check("refused: a rake with the LE placed (fins), a rake beyond 60 deg; a rake within it or none is fine",
          [refused(XV.place_rudder, d, sub["loops"], 5e-4, 1.0, 0.0, None, True, 5.0),
           XV.placement_problem({"rudder_rake": 61.0}, L) is not None,
           XV.placement_problem({"rudder_rake": -60.0}, L) is None, XV.placement_problem({"rudder_rake": None}, L) is None],
          [1, 1, 1, 1], 0)


def test_parts(hull):
    print("\nthe propeller and the rudder as the Vehicle tab gets them (their adapters)")
    need = ["msc_blade.txt", "outputs/corner_points.dat", "outputs/thickness_params_H80_info.dat",
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
    pl.rudder_rake = 0.0
    segs, info = XV.assemble(parts, pl)
    a, ec, eh, et = info["rudder_frames"][0]
    whole = a + np.outer(sc * (root[:, 0] - xa), ec) + np.outer(sc * (root[:, 1] - y0), eh) + np.outer(sc * root[:, 2], et)
    te = a + chord * ec
    u = np.array([0.0, 0.0, 0.0])
    u[1:] = te[1:] / np.hypot(te[1], te[2])                          # radial: the TE square to the axis
    outside = lambda p: np.hypot(p[1], p[2]) > d.radius("r", np.array([p[0]]))[0]
    g = gaps_along(whole, u, outside, reach=(whole @ u) / (u @ u))                  # down to the axis
    inner = segs["roots"][0]
    loops_te = np.array([[float(q[:, 1].mean()), float(q[:, 0].max())] for q in rp["loops"][:-3]])
    check("the wind-tunnel rudder with its TE line square to the axis (rake 0) at the same x: its TE line's dx/dy "
          "from the TE rows (the loops' TEs within 0.1 mm of that line); the root chord at atan of it; the TE "
          "radial through the hull at x; resting on SUBOFF; extruded along the TE, inside",
          [abs(rp["te_slope"] + 0.141956) < 1e-5,
           np.abs(loops_te[:, 1] - rp["root_x"][1] - rp["te_slope"] * loops_te[:, 0]).max() < 0.1,
           info["rudder_pitch"][0] - np.degrees(-np.arctan(rp["te_slope"])), te[0] - x_te,
           info["rudder_slope"][0] - rp["te_slope"], g.min(),
           bool((np.hypot(inner[:, 1], inner[:, 2]) < d.radius("r", inner[:, 0])).all())], [1, 1, 0, 0, 0, 0, 1], 1e-9)
    print(f"       ... with rake 0: root chord at {info['rudder_pitch'][0]:.2f} deg; root LE {1000 * info['rudder_rho_le'][0]:.1f} "
          f"mm, TE {1000 * info['rudder_rho_te'][0]:.1f} mm from the axis; extruded {1000 * info['rudder_depth'][0]:.1f} mm "
          f"along the TE line")


def synthetic_blade_cad(r_hub=20.0, r_mid=50.0, r_tip=80.0, n=41):
    """A propeller "CAD" as blade_modify.write_cad gives it (mm): a compound of the blade's shell, open at its
    root ring on the hub cylinder (radius r_hub about x), and the hub's shell."""
    import vehicle_cad as VC
    t = np.linspace(0.0, 2.0 * np.pi, n)

    def loop(chord, thick, r, on_cylinder):
        s, w = -0.5 * chord * np.cos(t), 0.5 * thick * np.sin(t)          # LE first, TE at t = pi
        if on_cylinder:
            return np.column_stack((s, r * np.sin(w / r), r * np.cos(w / r)))
        return np.column_stack((s, w, np.full_like(s, r)))
    loops = [loop(40.0, 8.0, r_hub, True), loop(30.0, 5.0, r_mid, False), loop(16.0, 2.0, r_tip, False)]
    fitter = XL._SectionFitter(n, 12)
    wires = [fitter.wire(lp)[0] for lp in loops]
    faces = VC._shapes(XL._thru(wires, False, False), "FACE") + [XL._planar_face(wires[-1])]
    Sewing = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_Sewing")
    sew = Sewing(1e-3)
    for f in faces:
        sew.Add(f)
    sew.Perform()
    gp_Pnt, gp_Dir, gp_Ax2 = XL._occ("gp", "gp_Pnt", "gp_Dir", "gp_Ax2")
    MakeCylinder = XL._occ("BRepPrimAPI", "BRepPrimAPI_MakeCylinder")
    hub = MakeCylinder(gp_Ax2(gp_Pnt(-30.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), r_hub, 60.0).Shape()
    return VC.compound([VC._shapes(sew.SewedShape(), "SHELL")[0], VC._shapes(hub, "SHELL")[0]])


def test_vehicle_cad(hull, full=False):
    print("\nthe vehicle's CAD (vehicle_cad.py)")
    if not XL.occ_available():
        print("       skipped: no pythonocc-core or cadquery-ocp here")
        return
    import vehicle_cad as VC
    r_hub, h_hub = 20.0, 60.0
    blade = VC.blade_solid(synthetic_blade_cad(r_hub), r_hub)
    n_b, ok_b, v_b = VC.solid_report(blade)
    prop = VC.propeller_solid(blade, 3, r_hub, h_hub)
    n_p, ok_p, v_p = VC.solid_report(prop)
    v_hub = np.pi * r_hub ** 2 * h_hub
    check("a blade shell open at its root ring on the hub cylinder, closed by the cylinder inside the ring: one "
          "valid solid; 3 of them and the hub fused: one valid solid of their volumes (the root on the hub)",
          [n_b, ok_b, v_b > 0, n_p, ok_p, v_p / (v_hub + 3.0 * v_b) - 1.0], [1, 1, 1, 1, 1, 0], 1e-5)
    t = np.linspace(0.0, 2.0 * np.pi, 41)
    wing = XL.loft_loops([np.column_stack((50.0 - 50.0 * np.cos(t), -6.0 * np.sin(t), np.full_like(t, z)))
                          for z in (0.0, 60.0, 120.0)], n_poles=12).shape    # a plate-like wing, flat root at z 0
    root_area = XL.mass_properties(VC._root_face(wing))[1]
    ext = VC.extended_rudder(wing, 7.0)
    n_e, ok_e, v_e = VC.solid_report(ext)
    check("a rudder's root extrusion sewn on (no boolean): one valid solid of the rudder's volume and the root "
          "face's area times the depth", [n_e, ok_e, v_e / (VC.volume(wing) + 7.0 * root_area) - 1.0, VC._bbox(ext)[0, 2]],
          [1, 1, 0, -7.0], 1e-6)
    ext2 = VC.extended_rudder(wing, 7.0, direction=(0.3, 0.0, 1.0))
    n_o, ok_o, v_o = VC.solid_report(ext2)
    h = 7.0 / np.hypot(1.0, 0.3)
    check("... along a slanted line (dx/dz 0.3, a TE line's): one valid solid, the root face's area times the "
          "height the 7 mm along the line reach; its end 7 mm back along the line",
          [n_o, ok_o, v_o / (VC.volume(wing) + h * root_area) - 1.0, VC._bbox(ext2)[0, 2] + h,
           VC._bbox(ext2)[0, 0] - VC._bbox(wing)[0, 0] + 0.3 * h], [1, 1, 0, 0, 0], 1e-6)
    need = ["outputs/corner_points.dat", "outputs/thickness_params_y0_y200_info.dat",
            "outputs/sections_xcad_with_cap_y0_y200_m.dat", "outputs/sections_stack_y0_y200_raw_mm.dat"]
    missing = [p for p in need if not os.path.exists(os.path.join(HERE, p))]
    if missing:
        print("       vehicle skipped, not here: " + ", ".join(missing))
        return
    import xgeom_rudder as XR
    rudder = XR.RudderAdapter()
    kinds = ("hull", "rudder", "blade") if full else ("hull", "rudder")
    ads = {"hull": hull, "rudder": rudder}
    if full:
        import xgeom_blade as XB
        ads["blade"] = XB.BladeAdapter()
    with tempfile.TemporaryDirectory() as tmp:
        for k in kinds:
            ads[k].out_dir = os.path.join(tmp, k)
            ads[k].settings["plot"] = False
        parts = {k: XV.PARTS[k](ads[k]) for k in kinds}
        pl = XV.default_placement(parts)
        segs, info = XV.assemble(parts, pl)
        tasks = {k: ads[k].build_task("v") for k in kinds}
        said = []
        res = VC.run_vehicle(os.path.join(tmp, "vehicle"), "v", tasks, parts, info, pl, said.append)
        rec = res["record"]
        names = ["hull", "rudder 1", "rudder 2", "rudder 3", "rudder 4"] + (["propeller"] if full else [])
        files = [os.path.join(tmp, "vehicle", f"v_vehicle{x}") for x in (".step", "_parts.step", ".json",
                                                                          "_setup.json")]
        shape = XL.read_step(rec["files"]["vehicle"])
        box = VC._bbox(shape)
        lines = np.vstack([1000.0 * q for k in ("hull", "rudders") for q in segs[k]])
        lo, hi = lines.min(axis=0), lines.max(axis=0)
        rudder_v = 1e3 * rec["rudder"]["scale"] ** 3 * 583.67                     # mm^3, the loft ~ 583.67 cm^3
        check(f"the {', '.join(kinds)} built (each in its own folder), placed, fused in a process of its own: "
              f"the files, the parts as the Vehicle tab has them, one valid solid, the rudders' pitch",
              [all(os.path.exists(f) for f in files), rec["parts"] == names, rec["solids"], rec["valid"],
               said[-1].startswith("the fuse"), rec["rudder"]["pitch"][0] - info["rudder_pitch"][0]],
              [1, 1, 1, 1, 1, 0], 1e-9)
        check("the vehicle's box (mm): " + ("around the Vehicle tab's hull and rudder lines (the propeller reaches "
                                            "further)" if full else "the Vehicle tab's hull and rudder lines'"),
              np.r_[box[0] - lo, box[1] - hi] if not full else np.r_[np.maximum(box[0] - lo, 0.0),
                                                                     np.maximum(hi - box[1], 0.0)], 0.0, 0.5)
        v_hull = HM.hydrostatics(hull.design)["volume"] * 1e9
        print(f"       vehicle V {rec['volume_mm3'] / 1e9:.6f} m^3 (hull {v_hull / 1e9:.6f}), "
              f"{rec['fuse_seconds']:.1f} s to fuse")
        check("the vehicle's volume: the hull's and the rudders' (outside the hull), and less than with their "
              "whole root extrusions", [rec["volume_mm3"] > v_hull + 3.9 * rudder_v,
                                       rec["volume_mm3"] < v_hull + 4 * rudder_v * 1.2 + (full and 1e6 or 0)], [1, 1], 0)
        pl.rudder_rake = 0.0                                          # the TE square to the axis
        segs, info = XV.assemble(parts, pl)
        tasks = {k: ads[k].build_task("r") for k in kinds}
        rec = VC.run_vehicle(os.path.join(tmp, "vehicle"), "r", tasks, parts, info, pl, said.append)["record"]
        print(f"       vehicle with the rudders' TE square to the axis: V {rec['volume_mm3'] / 1e9:.6f} m^3, "
              f"{rec['fuse_seconds']:.1f} s to fuse")
        check("TE rake 0: the set-up keeps the rake and the TE line's slope; one valid solid, the hull's volume and "
              "the rudders' outside it", [rec["solids"], rec["valid"], rec["rudder"]["rake"][0],
                                          rec["rudder"]["slope"][0] - parts["rudder"]["te_slope"],
                                          rec["volume_mm3"] > v_hull + 3.9 * rudder_v,
                                          rec["volume_mm3"] < v_hull + 4 * rudder_v * 1.5 + (full and 1e6 or 0)],
              [1, 1, 0, 0, 1, 1], 1e-9)


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
        import time
        XT.LOG = os.path.join(tempfile.mkdtemp(), "log.txt")

        def wait():
            t0 = time.time()
            while app._job is not None and time.time() - t0 < 300:
                root.update()
                time.sleep(0.05)
        calls, labels, said = [], {}, {}
        for kind in ("rudder", "hull", "blade"):                    # each panel's button builds its own component
            app.switch(kind)
            ad = app.adapters[kind]
            real = ad.build_task
            ad.build_task = lambda case, k=kind: (lambda: calls.append((k, case)) or
                                                  {"report": [], "message": f"{k} built", "error": False, "view": None})
            labels[kind] = app.build_btn.cget("text")
            app.build_btn.invoke()
            wait()
            said[kind] = app.status.cget("text")
            ad.build_task = real
        check("each component's build button names it and builds that component (hull, propeller, rudder)",
              [labels == {"rudder": "XCAD rudder", "hull": "CAD hull", "blade": "CAD propeller"},
               calls == [("rudder", "gui"), ("hull", "gui"), ("blade", "gui")],
               all(said[k].startswith(f"{labels[k]} done") and f"{k} built" in said[k] for k in said)], [1, 1, 1], 0)
        stubs = {k: app.adapters[k].build_task for k in app.adapters}
        for k in app.adapters:
            app.adapters[k].build_task = lambda case, k=k: (lambda: {"report": [], "message": f"{k}: no CAD here",
                                                                     "error": True, "view": None})
        app.vbuild_btn.invoke()
        busy = (str(app.vbuild_btn.cget("state")), str(app.build_btn.cget("state")))
        wait()
        for k, f in stubs.items():
            app.adapters[k].build_task = f
        log = open(XT.LOG).read()
        check("Vehicle CAD: both buttons off while it runs; a component without its CAD stops it (red), the "
              "buttons back, the error in the log", [busy == ("disabled", "disabled"),
                                                     app.status.cget("background") == XT.STATUS["error"][0],
                                                     "Vehicle CAD failed" in app.status.cget("text"),
                                                     str(app.vbuild_btn.cget("state")) == "normal",
                                                     app.vbuild_btn.cget("text") == "Vehicle CAD",
                                                     "Traceback" in log and "the hull: hull: no CAD here" in log],
              [1, 1, 1, 1, 1, 1], 0)
        if XL.occ_available():
            save_vehicle(app, wait)
        else:
            print("       Save vehicle CAD skipped: no pythonocc-core or cadquery-ocp here")
        app.switch("hull")
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
        app.vrake.set(True)
        app.vset["rudder_rake"].set(0)
        app.on_vehicle()
        app.update_views()
        root.update()
        rake0, view0 = app.placement.rudder_rake, app._vinfo["rudder_rake"][0]
        app.vset["rudder_rake"].set(70)
        app.on_vehicle()
        refused70 = app.status.cget("background") == XT.STATUS["error"][0] and float(app.vset["rudder_rake"].get()) == 0
        app.vrake.set(False)
        app.on_vehicle()
        app.update_views()
        root.update()
        check("TE rake ticked, 0: the rudders' TE square to the axis; 70 refused (the box shows 0); unticked: pitched "
              "to the hull again, the box greyed with the TE's rake the hull's slope gives",
              [rake0, view0, refused70, app.placement.rudder_rake is None,
               str(app.vset["rudder_rake"].cget("state")) == "disabled",
               float(app.vset["rudder_rake"].get()) - round(app._vinfo["rudder_rake"][0], 2)], [0, 0, 1, 1, 1, 0], 1e-9)
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


def save_vehicle(app, wait):
    """Save vehicle CAD with vehicle_cad.run_vehicle left out (a box for the vehicle and each component):
    built first, then saved again without a build (IGES), a component's CAD from that build saved from its
    panel, and built again after the placement changes."""
    import xgeom_tool as XT
    import vehicle_cad as VC
    tmp = tempfile.mkdtemp()
    box = XL._occ("BRepPrimAPI", "BRepPrimAPI_MakeBox")(10.0, 20.0, 30.0).Shape()
    runs = []

    def fake_run(out_dir, case, tasks, parts, info, placement, progress=print, separate=True):
        runs.append(sorted(tasks))
        n = len(runs)
        steps = {k: XL.write_step(box, os.path.join(tmp, f"{k}{n}.step")) for k in tasks}
        return {"report": [], "message": "vehicle built", "error": False, "warn": False,
                "view": XL.write_step(box, os.path.join(tmp, f"{case}_vehicle{n}.step")), "record": {"steps": steps}}
    real_run, ask = VC.run_vehicle, XT.filedialog.asksaveasfilename
    picks = iter(os.path.join(tmp, n) for n in ("v1.step", "v2.igs", "hull.stp", "v3.step"))
    VC.run_vehicle, XT.filedialog.asksaveasfilename = fake_run, (lambda **kw: next(picks))
    said = []
    try:
        for kind in ("vehicle", "vehicle", "hull", "vehicle"):
            if len(said) == 3:
                app.vset["rudders"].set(3)                          # a new placement: the vehicle again
                app.on_vehicle()
            app.on_save_cad(kind)
            wait()
            said.append(app.status.cget("text"))
        app.vset["rudders"].set(4)
        app.on_vehicle()
    finally:
        VC.run_vehicle, XT.filedialog.asksaveasfilename = real_run, ask
    same = [open(os.path.join(tmp, a), "rb").read() == open(os.path.join(tmp, b), "rb").read()
            for a, b in (("v1.step", "gui_vehicle1.step"), ("hull.stp", "hull1.step"), ("v3.step", "gui_vehicle2.step"))]
    check("Save vehicle CAD: built first (all components) and saved, then IGES without a build; the hull's CAD "
          "from that build saved from its panel; a new placement builds again",
          [len(runs), runs[0] == sorted(XV.active({k: 0 for k in app.adapters}, app.placement)),
           said[0].startswith("Vehicle CAD done") and "saved" in said[0], said[1].startswith("Save CAD done"),
           XL.count_faces(XL.read_iges(os.path.join(tmp, "v2.igs"))), said[2].startswith("Save CAD done"),
           said[3].startswith("Vehicle CAD done")] + same, [2, 1, 1, 1, 6, 1, 1, 1, 1, 1], 0)


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
            test_rake(hull)
            test_parts(hull)
            test_vehicle_cad(XH.HullAdapter(), full="--vehicle-cad" in sys.argv)
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
