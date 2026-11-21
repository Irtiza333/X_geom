"""
_xgeom_tool_test.py

Self-check for the design tool: the CAD loft (xcad_loft.py: one loft, and
two joined at a section, loft_split) and Save CAD's files, the rudder adapter
(xgeom_rudder.py: its tangent switches, and the two lofts of a design not
tangent at H), temp_rudder_cad.py (sections of any number of points on the
original's layout, its two lofts) and, with --gui, the window itself
(xgeom_tool.py, Save CAD included).

    python _xgeom_tool_test.py          loft and adapter
    python _xgeom_tool_test.py --gui    also open the tool for a moment

The loft checks need pythonocc-core (or cadquery-ocp) and are skipped
without it; the adapter needs the extraction's outputs in outputs/ (the X_geom
repo) and is skipped without them.
"""

import copy
import json
import os
import sys
import tempfile
import time

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


CAP_RINGS = (0.15, 0.3, 0.5, 0.7, 0.85, 1.0)               # the round cap's loops: shares of a quarter turn


def cad_edge(shape, a, b, tol=1.0e-3):
    """The length (mm) of the solid's edge from point a to point b (either way), as OCCT measures it on the
    CAD (BRepGProp on the edge); None when no edge runs between them."""
    import importlib
    explorer, edge_kind = XL._occ("TopExp", "TopExp_Explorer"), XL._occ("TopAbs", "TopAbs_EDGE")
    as_edge = XL._topods("Edge")
    mod = importlib.import_module(f"{XL._PKG}.TopExp")
    topexp = getattr(mod, "topexp", None) or getattr(mod, "TopExp")
    first, last = XL._static(topexp, "FirstVertex"), XL._static(topexp, "LastVertex")
    pnt = XL._static(XL._occ("BRep", "BRep_Tool"), "Pnt")
    mod = importlib.import_module(f"{XL._PKG}.BRepGProp")
    linear = XL._static(getattr(mod, "brepgprop", None) or getattr(mod, "BRepGProp"), "LinearProperties")
    props_of = XL._occ("GProp", "GProp_GProps")
    ex = explorer(shape, edge_kind)
    while ex.More():
        e = as_edge(ex.Current())
        ends = [np.array([p.X(), p.Y(), p.Z()]) for p in (pnt(first(e)), pnt(last(e)))]
        if min(max(np.linalg.norm(ends[0] - p), np.linalg.norm(ends[1] - q)) for p, q in ((a, b), (b, a))) < tol:
            props = props_of()
            linear(e, props)
            return float(props.Mass())
        ex.Next()
    return None


def wing_loops(n=40, tilt=True, cap="round"):
    """A tapered wing in the XCAD frame (x chordwise, y thickness, z height),
    mm: n horizontal loops from z = 0 to 100, then (tilt) a tilted loop and a
    cap over it, each 200 points starting at the LE; and the section areas.
    cap "round": six loops of a cap with half an ellipse as its cross
    profile, 2 % of the chord tall at mid-chord and falling to nothing at
    the two ends, where every loop runs into the tilted loop's LE and TE
    points (as the extraction's cap loops do); the last is the crest line.
    cap "lid": one thin loop above the tilted one, its LE and TE points over
    the tilted loop's, not in them."""
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
        loops.append(top)
        if cap == "lid":
            lid = top.copy()
            lid[:, 1] *= 0.3                                                      # a thin loop above it
            lid[:, 2] += 1.0 + 0.01 * (lid[:, 0] - lid[0, 0])
            loops.append(lid)
        else:
            s = (top[:, 0] - top[0, 0]) / (top[99, 0] - top[0, 0])
            height = 0.02 * (top[99, 0] - top[0, 0]) * np.sqrt(np.clip(4.0 * s * (1.0 - s), 0.0, None))
            for f in CAP_RINGS:
                ring = top.copy()
                ring[:, 1] *= np.cos(0.5 * np.pi * f) if f < 1.0 else 0.0
                ring[:, 2] += height * np.sin(0.5 * np.pi * f)
                loops.append(ring)
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
    check("with a tilted loop and a round cap of 6 loops over it: valid, closed; the five patches and the "
          "root, 6 faces", [res2.valid, res2.free_edges, res2.n_ruled, res2.volume > res.volume,
                            XL.count_faces(res2.shape)], [1, 0, 1 + len(CAP_RINGS), 1, 6], 0)
    res2h = XL.loft_loops(loops2, half=True)
    check("its half body: valid, closed, 6 faces, half the volume",
          [res2h.valid, res2h.free_edges, XL.count_faces(res2h.shape), res2h.volume / res2.volume], [1, 0, 6, 0.5],
          1e-4)
    r_te = 0.004 * 150.0                                   # the TE's round at the root: radius 0.6 mm
    arc = r_te * (0.5 * np.pi - np.radians(10.0))          # from the TE to about where it meets the flank
    cor = res2.corners
    cad = [cad_edge(res2.shape, cor["ur_a"], cor["ur_b"]), cad_edge(res2.shape, cor["ul_a"], cor["ul_b"])]
    print(f"       the tip patch's LE and TE edges {res2.tip_edges[0]:.4f} and {res2.tip_edges[1]:.4f} mm (the CAD's "
          f"{cad[0]:.4f} and {cad[1]:.4f}); the strips {res2.strips[0]:.3f} and {res2.strips[1]:.3f} mm, "
          f"{len(res2.strip_knots) - 4} poles across")
    check("the LE strip as wide as the LE's rounding (a few percent of the chord); the TE strip wider than the "
          "TE's round (its arc to the flank), until the tip patch's TE edge is as long as its LE edge (to "
          "EDGE_RTOL; the CAD's two edges measured by OCCT the same); the note says so",
          [res2.te_round, res2.strips[1] > 2.0 * arc, 0.03 * 150.0 < res2.strips[0] < 0.12 * 150.0,
           abs(res2.tip_edges[1] / res2.tip_edges[0] - 1.0) < XL.EDGE_RTOL, abs(cad[1] / cad[0] - 1.0) < 2e-3,
           abs(cad[0] / res2.tip_edges[0] - 1.0) < 1e-3, "as long as its LE edge" in res2.notes[0]],
          [0, 1, 1, 1, 1, 1, 1], 0)
    check("no cap (a flat top): its TE edge as long as its LE edge too, so the TE strip as wide as the LE strip",
          [abs(res.tip_edges[1] / res.tip_edges[0] - 1.0) < XL.EDGE_RTOL,
           abs(res.strips[1] / res.strips[0] - 1.0) < 0.01, "flat top's TE edge" in res.notes[0]], [1, 1, 1], 0)
    levels = [XL._SmoothLoop(lp, 99) for lp in loops2[:res2.n_smooth]]
    plain = XL._refined_knots(40, "strip")
    ring0 = XL._strip_knots(levels, res2.strips[0], res2.strips[1], plain, cap=len(plain) - 4)[1]
    k_ref, ring1 = XL._strip_knots(levels, res2.strips[0], res2.strips[1], plain)
    print(f"       the strips' rows on the plain knots within {ring0:.3f} deg of the sections' curves, on the "
          f"refined ones ({len(k_ref) - 4} poles) {ring1:.3f} deg")
    check("the TE's round inside the wider TE strip: fitted across on the plain knots, the rows turn off the "
          "sections' curves by more than STRIP_ANGLE about the round's ends; on the knots refined there (as the "
          "loft's are, mirrored about the middle) within it",
          [ring0 > XL.STRIP_ANGLE, ring1 <= XL.STRIP_ANGLE, len(k_ref) > len(plain),
           len(res2.strip_knots) > len(plain), np.abs(k_ref[::-1] - (1.0 - k_ref)).max() < 1e-12], [1, 1, 1, 1, 1], 0)
    k2 = res2.n_smooth                                     # the rim: the tilted loop
    top_le, top_te, cor = loops2[k2][0], loops2[k2][99], res2.corners
    run = [0.5 * (abs(cor[a][0] - top[0]) + abs(cor[b][0] - top[0]))
           for a, b, top in (("ur_a", "ur_b", top_le), ("ul_a", "ul_b", top_te))]
    check("the tip patch's corners sit on the strip lines under the tip: under the tops of the LE and TE lines "
          f"by TIP_DROP ({XL.TIP_DROP:g}) times their distance from them along the chord (the diagonals' slope), "
          "the same on both sides",
          [res2.drops[0] / run[0], res2.drops[1] / run[1], top_le[2] - cor["ur_a"][2] - res2.drops[0],
           cor["ur_a"][2] - cor["ur_b"][2], cor["ul_a"][2] - cor["ul_b"][2]], [XL.TIP_DROP, XL.TIP_DROP, 0, 0, 0],
          1e-6)
    pair = XL.loft_loops(loops2, tip_drop=(2.0, 1.5))
    run = [0.5 * (abs(pair.corners[a][0] - top[0]) + abs(pair.corners[b][0] - top[0]))
           for a, b, top in (("ur_a", "ur_b", top_le), ("ul_a", "ul_b", top_te))]
    check("tip_drop sets the slope, a pair one for each end: valid, 6 faces",
          [pair.valid, XL.count_faces(pair.shape), pair.drops[0] / run[0], pair.drops[1] / run[1]],
          [1, 6, 2.0, 1.5], 1e-6)
    flat = XL.loft_loops(loops2, tip_drop=(1.0, 0.5))
    run = [0.5 * (abs(flat.corners[a][0] - top[0]) + abs(flat.corners[b][0] - top[0]))
           for a, b, top in (("ur_a", "ur_b", top_le), ("ul_a", "ul_b", top_te))]
    note = next(t for t in flat.notes if t.startswith("the tip patch within"))
    check("a slope that leaves the normal turning across the tip patch's edges by more than EDGE_TURN (0.5 at "
          "the TE of this wing, 3.1 deg) is built again at TIP_DROP_SAFE (1.5) and the smoother kept, the note "
          "says so; at the default the turn is under EDGE_TURN",
          [flat.valid, XL.count_faces(flat.shape), flat.drops[0] / run[0], flat.drops[1] / run[1],
           flat.tip_turn <= XL.EDGE_TURN, "built again at 1.5/1.5" in note, res2.tip_turn <= XL.EDGE_TURN],
          [1, 6, XL.TIP_DROP_SAFE, XL.TIP_DROP_SAFE, 1, 1, 1], 1e-6)
    check("the tip patch lies within a ten-thousandth of the chord of the tip; no more than 150 poles each way",
          [res2.tip_error < 1.5e-4 * 150.0, max(res2.tip_poles) <= 150, min(res2.tip_poles) > 20], [1, 1, 1], 0)
    k4 = np.r_[[0.0] * 4, [0.5], [1.0] * 4]                # a 5 x 5 cubic patch: flat and square, sheared
    gu, gv = np.meshgrid(XL._greville(k4), XL._greville(k4))               # by 60 deg, and doubled back on
    square = np.dstack((gu, gv, 0.0 * gu))                                 # itself along u
    shear = np.dstack((gu + gv * np.cos(np.radians(60.0)), gv * np.sin(np.radians(60.0)), 0.0 * gu))
    fold = np.dstack((gu * (1.0 - gu) * 4.0, gv, 0.0 * gu))
    check("the angle between a patch's two families of lines (_lines_angle): 90 deg on a square patch, 60 on a "
          "sheared one, 0 where the lines cross (a fold); the tip patch does not fold, and the note says so",
          [XL._lines_angle(square, k4, k4), XL._lines_angle(shear, k4, k4), XL._lines_angle(fold, k4, k4),
           res2.tip_angle > 2.0, flat.tip_angle > 2.0, "no fold" in res2.notes[-1]], [90, 60, 0, 1, 1, 1], 1e-4)
    lid = XL.loft_loops(wing_loops(cap="lid")[0])
    check("a cap loop that ends over the tops of the LE and TE lines, not in them (a blade over each end): "
          "still a valid closed solid of 6 faces, and the result and its note say that the tip patch folds",
          [lid.valid, lid.free_edges, XL.count_faces(lid.shape), lid.tip_angle, "folds" in lid.notes[-1]],
          [1, 0, 6, 0, 1], 0)
    from scipy.interpolate import CubicSpline
    r_arc, s_pt = 0.3, 0.05                                # a wrap over a rounded cap: a wall 1 tall, a round
    th = np.linspace(0.0, 0.5 * np.pi, int(round(0.5 * np.pi * r_arc / s_pt)) + 1)[1:]   # of radius 0.3, a
    left = np.vstack((np.column_stack((0 * np.arange(0, 1, s_pt), -1.0 + 0 * np.arange(0, 1, s_pt), np.arange(0, 1, s_pt))),
                      np.column_stack((0 * th, -1.0 + r_arc * (1.0 - np.cos(th)), 1.0 + r_arc * np.sin(th)))))
    flat = np.column_stack((0 * np.arange(-1.0 + r_arc + s_pt, 1.0 - r_arc, s_pt), np.arange(-1.0 + r_arc + s_pt, 1.0 - r_arc, s_pt), 1.0 + r_arc + 0 * np.arange(-1.0 + r_arc + s_pt, 1.0 - r_arc, s_pt)))
    right = left[::-1] * np.array([1.0, -1.0, 1.0])
    q = np.vstack((left, flat, right))                     # flat top, the round and a wall down again
    s = XL._loop_arc(q)
    ends = ((1, np.array([0.0, 0.0, 1.0])), (1, np.array([0.0, 0.0, -1.0])))
    wrap = CubicSpline(s, q, axis=0, bc_type=ends)
    jump, top = XL._curvature_jumps(wrap, s)
    th2 = np.linspace(np.pi, 0.0, 61)                       # a half circle: no jump, its curvature 1
    q2 = np.column_stack((0 * th2, np.cos(th2), 1.0 + np.sin(th2)))
    s2 = XL._loop_arc(q2)
    dome = CubicSpline(s2, q2, axis=0, bc_type=((1, np.array([0.0, 0.0, 1.0])), (1, np.array([0.0, 0.0, -1.0]))))
    jump2, top2 = XL._curvature_jumps(dome, s2)
    check("_curvature_jumps: a wrap up a wall, round a 0.3 radius and over a flat top jumps (its largest "
          "curvature 1 / 0.3, plus the cubic's overshoot at the jump, under 25 %); a half circle does not "
          "(curvature 1)", [jump, 1.0 < top * r_arc < 1.25, jump2, top2], [1, 1, 0, 1], 0.05)
    length = XL.CAP_BLEND / top
    soft = XL._blended(wrap, length, ends)
    fine = np.linspace(0.0, s[-1], 2001)
    moved = np.linalg.norm(soft(fine) - wrap(fine), axis=1)
    mid = np.abs(fine - 0.5 * s[-1]) < 0.3                  # the middle of the flat top, far from the round

    def ramp(curve):
        """The loops' steps the curvature takes to rise from a tenth to nine tenths of its top, up the first
        round (the cubic through the points)."""
        d1, d2 = curve(s, 1), curve(s, 2)
        kap = np.linalg.norm(np.cross(d1, d2), axis=1) / np.linalg.norm(d1, axis=1) ** 3
        kap = kap[:len(s) // 2]
        return int(np.argmax(kap > 0.9 * kap.max()) - np.argmax(kap > 0.1 * kap.max()))

    check("_blended: the ends and their directions held, the middle of the flat top left alone, the round "
          "moved in by about its curvature x length^2 / 12 (two passes), and the curvature now climbs the "
          "round over two or more of the loops' steps, not one",
          [np.linalg.norm(soft(0.0) - q[0]), np.linalg.norm(soft(s[-1]) - q[-1]),
           np.linalg.norm(soft(0.0, 1) / np.linalg.norm(soft(0.0, 1)) - ends[0][1]), moved[mid].max() < 1e-7,
           0.6 < moved.max() / (top * length ** 2 / 12.0) < 1.4, ramp(wrap), ramp(soft) >= 2],
          [0, 0, 0, 1, 1, 1, 1], 1e-9)
    check("the test wing's cap and the note agree on whether it was blended; a blend moves it under 0.05 mm",
          [("blended" in res2.notes[-1]) == (res2.tip_blend[1] > 0.0), res2.tip_blend[1] < 0.05], [1, 1], 0)
    one = XL.loft_loops(loops2, strip_frac=0.07)
    check("one fraction: the LE strip 0.07 of the root chord, the TE strip as wide as the tip patch's edges "
          "need, the note says so",
          [one.strips[0] - 0.07 * 150.0, one.te_round, one.strips[1] > 2.0 * arc,
           abs(one.tip_edges[1] / one.tip_edges[0] - 1.0) < XL.EDGE_RTOL, "as long as its LE edge" in one.notes[0]],
          [0, 0, 1, 1, 1], 1e-9)
    wide = XL.loft_loops(loops2, strip_frac=0.12)
    check("a wider LE strip: still valid, 6 faces; the TE strip widens with it, its edge as long",
          [wide.valid, XL.count_faces(wide.shape), wide.strips[0] - 0.12 * 150.0, wide.strips[1] > one.strips[1],
           abs(wide.tip_edges[1] / wide.tip_edges[0] - 1.0) < XL.EDGE_RTOL], [1, 6, 0, 1, 1], 1e-9)
    pair = XL.loft_loops(loops2, strip_frac=(0.07, 0.03))
    check("a strip width each for the LE and the TE is used as given (the TE edge then shorter): valid, 6 "
          "faces; no step found in the horizontal loops",
          [pair.valid, XL.count_faces(pair.shape), len(res.steps), pair.strips[1] - 0.03 * 150.0, pair.te_round,
           pair.tip_edges[1] < 0.9 * pair.tip_edges[0]], [1, 6, 0, 0, 0, 1], 1e-9)
    print(f"       with that 4.5 mm TE strip round a 0.6 mm TE: {pair.notes[-1]}")
    lo = XL._SmoothLoop(loops2[5], 99)
    st = np.array([0.0, 0.3 * lo.S, lo.s_te, 0.8 * lo.S, lo.S, lo.S + 2.0])
    poly = np.column_stack([np.interp(np.mod(st, lo.S), lo.s, lo.dense[:, c]) for c in range(3)])
    kap = lo.curvature([lo.s_te, lo.s_te - 0.3, 0.5 * lo.s_te])
    check("rows are sampled from the loop's smooth curve: on its dense polyline to its sag, the LE and the TE "
          "exactly; the TE round's curvature 1 / 0.6 mm, the flank's nearly none",
          [np.abs(lo.at(st) - poly).max() < 2e-3, np.abs(lo.at([0.0, lo.s_te]) - [loops2[5][0], loops2[5][99]]).max(),
           abs(kap[0] * 0.6 - 1.0) < 0.25, abs(kap[1] * 0.6 - 1.0) < 0.25, kap[2] < 0.05], [1, 0, 1, 1, 1], 1e-12)
    stepped = [lp.copy() for lp in loops]
    for lp in stepped[:15]:
        lp[:, 1] *= 1.01                                   # the loops below z = 36 a hundredth thicker: a step
    jump = 0.01 * float(np.abs(loops[14][:, 1]).max())
    low = XL.loft_loops(stepped, ease="below")
    both = XL.loft_loops(stepped)
    print(f"       a step of {jump:.3f} mm in the loops: found {low.steps[0][1]:.3f} mm above loop {low.steps[0][0]}, "
          f"eased over {low.steps[0][2]:.1f} mm below; from both sides over {both.steps[0][2]:.1f} and "
          f"{both.steps[0][3]:.1f} mm")
    check("a step between loops is found and eased out of the loops below it; the solid valid, 6 faces",
          [low.valid, XL.count_faces(low.shape), len(low.steps), low.steps[0][0], low.steps[0][2] > 5.0,
           low.steps[0][3], abs(low.steps[0][1] / jump - 1.0) < 0.5], [1, 6, 1, 15, 1, 0, 1], 0)
    check("eased from both sides by default: a band below and one above, the loops within the step of the patches",
          [both.valid, len(both.steps), both.steps[0][2] > 5.0, both.steps[0][3] > 5.0, both.v_error < jump],
          [1, 1, 1, 1, 1], 0)
    try:
        XL.loft_loops(loops2, strip_frac=0.5)
        overlapped = 0
    except ValueError:
        overlapped = 1
    check("strips too wide to fit are refused", overlapped, 1, 0)
    bent = [lp.copy() for lp in loops2]
    for lp in bent:
        lp[:, 1] -= 0.4 * np.sin(np.pi * np.clip(lp[:, 0] / 150.0, 0, 1))      # cambered: not symmetric
    try:
        XL.loft_loops(bent, half=True)
        refused_half = 0
    except ValueError:
        refused_half = 1
    check("a cambered set of loops is refused a half body", refused_half, 1, 0)
    j = 20                                                 # the loops below loop 20 swept back: a kink there
    kinked = [lp + np.column_stack((np.where(lp[:, 2] < zs[j], 0.3 * (zs[j] - lp[:, 2]), 0.0),
                                    np.zeros(len(lp)), np.zeros(len(lp)))) for lp in loops2]
    one_k, two = XL.loft_loops(kinked), XL.loft_split(kinked, j)
    two_h = XL.loft_split(kinked, j, half=True)
    print(f"       a kink in the LE line at loop {j + 1}: one loft {one_k.v_error:.3g} mm off the loops, two "
          f"lofts {two.v_error:.2g} mm (the tip patch {two.tip_error:.2g} mm of that), {two.seconds:.1f} s")
    check("a kink at a section (the part below swept back): two lofts joined there (loft_split) follow it, one "
          "loft smooths it over; one valid solid, the seams' strips the whole stack's (the TE strip as the upper "
          "loft's tip patch needs, the one loft's to EDGE_RTOL), nine patches and the root; its half body 9 "
          "faces, half the volume",
          [two.valid, two.free_edges, two.faces, two.joint, two.v_error < 0.005, one_k.v_error > 0.02,
           two.strips[0] - one_k.strips[0], abs(two.strips[1] / one_k.strips[1] - 1.0) < XL.EDGE_RTOL,
           abs(two.tip_edges[1] / two.tip_edges[0] - 1.0) < XL.EDGE_RTOL, abs(two.volume / one_k.volume - 1.0),
           two_h.valid, two_h.faces, two_h.volume / two.volume],
          [1, 0, 10, j + 1, 1, 1, 0, 1, 1, 0, 1, 9, 0.5], 1e-4)
    refused = []
    for bad in ((kinked, 0), (kinked, len(kinked) - 1), (kinked, len(kinked) - 3)):
        try:
            XL.loft_split(*bad)
            refused.append(0)
        except ValueError:
            refused.append(1)
    check("loft_split refuses a joint at the first or the last loop, or among the cap's loops", refused, [1, 1, 1], 0)
    v, t = XL.tessellate(res2.shape, 0.5, 0.5)
    check("tessellation: triangles index the vertices", [len(t) > 500, t.min() >= 0, t.max() < len(v)], [1, 1, 1], 0)
    with tempfile.TemporaryDirectory() as tmp:
        step = XL.write_step(res2.shape, os.path.join(tmp, "wing.step"))
        back = XL.read_step(step)
        check("STEP written and read back: the same volume", XL.mass_properties(back)[0], res2.volume, 1e-5 * res2.volume)
        igs, stp, own = (os.path.join(tmp, n) for n in ("saved.igs", "saved.stp", "own.iges"))
        how = [XL.save_cad(step, igs), XL.save_cad(step, stp), XL.save_cad(step, own, iges=igs)]
        back_i = XL.read_iges(igs)
        try:
            XL.save_cad(step, os.path.join(tmp, "saved.txt"))
            txt = 0
        except ValueError:
            txt = 1
        check("save_cad: IGES converted from the STEP (its 6 faces, their area), STEP copied, a build's own IGES "
              "copied, .txt refused", [how[0].startswith("IGES converted"), XL.count_faces(back_i),
                                       XL.mass_properties(back_i)[1] / XL.mass_properties(back)[1] - 1.0,
                                       open(stp, "rb").read() == open(step, "rb").read(),
                                       open(own, "rb").read() == open(igs, "rb").read(), txt],
              [1, 6, 0, 1, 1, 1], 1e-6)


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

        def off_original(lp):                                # a loop of the 3D view against the original there
            x_le, xy = RM.original_section_at(ad.stack, lp[0, 1])
            return np.abs(lp - RM.xcad_loop(xy, x_le, lp[0, 1])).max()

        sk0 = ad.skeleton()
        model = RM.modified_sections(ad.design, ad.orig, 50)
        check("as it starts the sections drawn (and written) are the extracted ones, not the model's fit of "
              "them, 0.1 to 0.2 mm away (mm)",
              [max(off_original(lp) for lp in sk0["modified"] + sk0["cut"]) < 1e-4,
               0.1 < max(off_original(sec.loop()) for sec in model) < 0.2,
               max(off_original(sec.loop()) for sec in ad.preview()["sections"]) < 1e-4], [1, 1, 1], 0)
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
        secs0 = RM.modified_sections(ad.design, ad.orig, 30)
        ad.set_tangent_all(False)
        names_off, state_off = set(ad.values), ad.tangent_state()
        secs1 = RM.modified_sections(ad.design, ad.orig, 30)
        ad.set_tangent("LE", True)
        state_mixed = ad.tangent_state()
        ad.set_tangent_all(True)
        H0 = ad.design.H
        on_tangent = max(abs(ad.design.curves[c].v[1] - RM._pins(ad.orig, c, ad.design.curves[c].s, H0, "G1")[1])
                         for c in ad.names)
        check("tangent at H for every curve (set_tangent_all): off frees every C1 where it was (the sections as "
              "before), one curve back on makes it mixed, on puts every C1 back on the original's tangent",
              [state_off == "none", all(f"{c}.{'dx' if c == 'LE' else 'v'}1" in names_off for c in ad.names),
               max(np.abs(a.selig - b.selig).max() for a, b in zip(secs0, secs1)), state_mixed == "mixed",
               ad.tangent_state() == "all", on_tangent], [1, 1, 0, 1, 1, 0], 1e-12)
        ad.set("P2z.v5", 0.085)
        ad.set_bounds("H", 40.0, 150.0)
        ad.set("H", 150.0)
        sk = ad.skeleton()
        tip = [lp for lp in sk["unchanged"] if np.ptp(lp[:, 1]) > 1e-9]
        n_top = sum(1 for lp in ad.loops0 if np.ptp(lp[:, 1]) > 1e-9)
        check("H bounds 40 .. 150: H = 150 builds; the 3D view keeps the tip section and every cap loop",
              [len(ad.preview()["problems"]), ad.get("H"), len(tip), n_top > 1], [0, 150.0, n_top, 1], 1e-9)
        check("the changed design still meets the extracted section at H (no step), its root 20 mm forward (mm)",
              [off_original(sk["cut"][0]) < 1e-4, sk["modified"][0][0, 0] - sk0["modified"][0][0, 0]], [1, -20.0],
              1e-9)
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
                print(f"       solid: volume {lr.volume:.1f} mm^3, {lr.seconds:.1f} s, {lr.faces} faces")
                print(f"       the strips {lr.strips[0]:.3f} and {lr.strips[1]:.3f} mm, {len(lr.strip_knots) - 4} "
                      f"poles across; the tip patch's edges {lr.tip_edges[0]:.3f} and {lr.tip_edges[1]:.3f} mm")
                check("XCAD: a valid closed solid and its STEP file; the rudder's 6 patches (the LE and TE "
                      "strips, the central panels, the tip, the root); no step at H for the loft to ease; "
                      "the TE strip wider than the 0.75 mm round (8 to 14 mm to each side), as the tip "
                      "patch's TE edge needs to be as long as its LE edge; the tip patch's corners under the tip "
                      "on the strip lines; the cap's 0.7 mm edge rounding blended over 0.2 to 0.4 mm, which "
                      "moved the cap by 0.005 to 0.03 mm",
                      [lr.valid, lr.free_edges, os.path.exists(res["step"]), lr.faces, len(lr.steps), lr.te_round,
                       8.0 < lr.strips[1] < 14.0, abs(lr.tip_edges[1] / lr.tip_edges[0] - 1.0) < XL.EDGE_RTOL,
                       lr.drops[0] > 0.0, lr.drops[1] > 0.0, lr.tip_angle > 5.0,
                       0.2 < lr.tip_blend[0] < 0.4, 0.005 < lr.tip_blend[1] < 0.03],
                      [1, 0, 1, 6, 0, 0, 1, 1, 1, 1, 1, 1, 1], 0)
                hres = ad.build_xcad("th", plot=False, cad=True, half=True)
                hlr = hres["loft"]
                check("XCAD half body: valid, 6 faces, half the volume",
                      [hlr.valid, hlr.half, hlr.faces, hlr.volume / lr.volume], [1, 1, 6, 0.5], 2e-3)
                keep = (ad.design.copy(), dict(ad.join))
                ad.set_tangent_all(False)                     # every curve free at H; two of them turn there
                ad.set("P2z.v1", ad.get("P2z.v1") + 0.01)
                ad.set("LE.dx1", ad.get("LE.dx1") - 5.0)
                out2 = ad.build_task("tk")()
                klr = out2["loft"]
                print(f"       not tangent at H: two lofts, {klr.faces} faces, {klr.seconds:.1f} s, the walls within "
                      f"{max(float(t.split('the rows within ')[1].split(' mm')[0]) for t in klr.notes if 'the rows within' in t):.2g} "
                      f"mm of the loops' rows")
                check("XCAD with the curves not tangent at H (C1 of the LE and of P2z moved off the tangent): two "
                      "lofts joined at the H section (loop 50): one valid closed solid, nine patches and the root, "
                      "no step taken out of the loops, the patches on them; the status line says so",
                      [klr.valid, klr.free_edges, klr.faces, klr.joint, len(klr.steps), klr.v_error < 0.01,
                       "not tangent at H" in out2["message"], out2["error"]], [1, 0, 10, 50, 0, 1, 1, 0], 0)
                hk = ad.build_xcad("tkh", plot=False, cad=True, half=True)["loft"]
                check("... its half body: valid, 9 faces, half the volume",
                      [hk.valid, hk.half, hk.faces, hk.volume / klr.volume], [1, 1, 9, 0.5], 2e-3)
                ad.set_bounds("H", 40.0, ad.settings["z_top"])
                ad.set("H", 196.0)                            # above the tip patch's corners (about 190 mm)
                try:
                    ad.build_xcad("tt", plot=False, cad=True)
                    near = ""
                except ValueError as exc:
                    near = str(exc)
                check("... with H among the tip patch's rows (196 mm): refused, saying why and what to do",
                      ["too close to the tip" in near, "tick tangent at H" in near], [1, 1], 0)
                ad.design, ad.join = keep
                ad._rebuild()
                ad.set_bounds("H", 40.0, 150.0)
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
            with open(setup, "rb") as fh:
                first = fh.read()
            time.sleep(1.1)                                   # a stamp to the second would differ (the key is checked too)
            again = ad.save_space(os.path.join(tmp, "setup_again.json"))
            with open(again, "rb") as fh:
                second = fh.read()
            check("an unchanged set-up saves to the same bytes (no time stamp in it)",
                  [first == second, b'"written"' in first], [1, 0], 0)
            space = ad2.free_space()
            check("the optimiser's space leaves the held slots out", ["P3x.v2" in space.names, len(space)],
                  [0, n_free], 0)
        check("every curve stays pinned to the original at H",
              max(abs(ad.design.curve(c)([0.0])[0, 1] - ad.orig.value(c, ad.design.H)[0]) for c in ad.names), 0.0,
              1e-12)
        check("strip_frac from modify_control: None (the strips as wide as their edges' rounding, the default), "
              "one fraction, or a pair (LE, TE) kept as a pair; tip_drop beside it",
              [XR._strip_frac(None) is None, XR._strip_frac(0.07), *XR._strip_frac((0.07, 0.05)),
               ad.settings["strip_frac"] is None, ad.settings["tip_drop"]], [1, 0.07, 0.07, 0.05, 1, XL.TIP_DROP], 0)
        thin = XR.RudderAdapter()
        thin.set("P1z.v3", 0.002)                            # a nose far thinner than the model's fit is off
        pv = thin.preview(skeleton=True)
        check("a section too thin to write as extracted + change is a problem (named), not a broken loop",
              [len(pv["problems"]), "too thin to write" in pv["problems"][0], len(pv["skeleton"]["modified"])],
              [1, 1, 0], 0)
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
            ad3 = XR.RudderAdapter()                               # an adapter that has moved on, every way
            ad3.set_degree(6)
            ad3.set_span(0.8 * span0)
            ad3.set_order("LE", 5)
            ad3.set("H", 0.9 * ad3.get("H"))
            ad3.set_bounds("LE.dx2", -3.0, 4.0)
            ad3.set_free("LE.dx3", False)
            ad3.settings.update(n_sections=7, half=True, te_radius_mm=0.5)
            ad3.load_space(setup)
            check("load_space at any time (another degree, height, order, value, bounds, free flag, settings): "
                  "the adapter as the one that saved the set-up",
                  [ad3.degree - ad.degree, ad3.span / ad.span, ad3.orders == ad.orders, ad3.join == ad.join,
                   sorted(ad3.values) == sorted(ad.values), max(abs(ad3.values[n] - ad.values[n]) for n in ad.values),
                   ad3.meta == ad.meta, all(ad3.settings[k] == ad.settings[k] for k in ad.settings),
                   ad3.setup_file == os.path.abspath(setup), len(ad3.preview()["problems"])],
                  [0, 1, 1, 1, 1, 0, 1, 1, 1, 0], 1e-9)
            ad3.set_degree(5)
            ad3.set_span(0.9 * span0)
            ad3.set_bounds("H", 50.0, 120.0)
            was = (ad3.orig, ad3.design, ad3.space, ad3.stack, ad3.loops0, ad3.settings, ad3.setup_file)
            vals, meta, sets = dict(ad3.values), copy.deepcopy(ad3.meta), dict(ad3.settings)
            with open(setup) as fh:
                rec = json.load(fh)
            rec["settings"].update(n_sections=99, height_scale=1.2, half=True)
            del rec["space"]["orders"]                             # read as far as the space, then it fails
            said = []
            for name, bad in (("bad_design_space.json", json.dumps(rec)), ("wing.json", '{"geometry": "wing"}'),
                              ("list.json", "[1, 2]"), ("notes.txt", "H = 60 mm, LE swept back")):
                with open(os.path.join(tmp, name), "w") as fh:
                    fh.write(bad)
                try:
                    ad3.load_space(os.path.join(tmp, name))
                    said.append("read")
                except ValueError as exc:
                    said.append(str(exc))
            check("a file that cannot be taken over (a set-up broken part way, another geometry's, no set-up, no "
                  "JSON) raises a ValueError that names it and says why, and leaves the adapter as it was",
                  [said[0].startswith("bad_design_space.json could not be taken as a rudder set-up (KeyError"),
                   said[1].startswith("wing.json is not a rudder set-up"),
                   said[2].startswith("list.json is not a rudder set-up"),
                   said[3].startswith("notes.txt is not a rudder set-up (not a JSON file"),
                   all(a is b for a, b in zip(was, (ad3.orig, ad3.design, ad3.space, ad3.stack, ad3.loops0,
                                                    ad3.settings, ad3.setup_file))),
                   ad3.values == vals, ad3.meta == meta, ad3.settings == sets, ad3.span / (0.9 * span0)],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1], 1e-12)
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
        app.tangent_all.set(False)
        app.on_tangent_all()
        root.update()
        off = [app.ad.tangent_state(), app.tangent.get(), app.tangent_all.get(),
               "P3z.v1" in [r.name for r in app.rows]]
        app.tangent.set(True)                                      # the picked curve's own switch back on
        app.on_tangent()
        root.update()
        mixed = [app.ad.tangent_state(), app.tangent_all.get()]
        app.tangent_all.set(True)
        app.on_tangent_all()
        root.update()
        check("the Region box's tangent at H: off frees every curve's C1 (the picked curve's switch and rows show "
              "it), that curve's own switch back on leaves the box unticked (mixed), on ties every curve again",
              [off == ["none", False, False, True], mixed == ["mixed", False], app.ad.tangent_state() == "all",
               app.tangent.get(), app.tangent_all.get()], [1, 1, 1, 1, 1], 0)
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
        ask_open, out_dir = XT.filedialog.askopenfilename, app.ad.out_dir
        with tempfile.TemporaryDirectory() as tmp:
            try:
                app.ad.out_dir = tmp
                none = app.setup_label.cget("text")
                app.case.delete(0, "end")
                app.case.insert(0, "kept")
                app.on_save()
                root.update()
                setup = os.path.join(tmp, "kept_design_space.json")
                named = app.setup_label.cget("text")
                vals, meta, span = dict(app.ad.values), copy.deepcopy(app.ad.meta), app.ad.span
                app.npts.set(5)                                    # the design moved on: degree, height, H, half
                app.on_points()
                app.span_box.set("250")
                app.on_span()
                app.set_value("H", 0.8 * app.ad.get("H"), None)
                app.vhalf.set(True)
                app.on_half()
                app.case.delete(0, "end")
                app.case.insert(0, "other")
                moved = [app.ad.degree, app.ad.span, app.ad.settings["half"]]
                asked = []
                XT.filedialog.askopenfilename = lambda **kw: asked.append(kw) or setup
                app.on_load_space()
                app.update_views()                                 # the update it scheduled, now
                loaded = app.status.cget("text")
                root.update()
                check("Save set-up, a changed design, then Load ... in the Set-up box: the design, the panel and "
                      "the views as saved (the case name left alone); the dialog opens on the set-up's folder; "
                      "the status line says what was loaded",
                      [os.path.exists(setup), none.startswith("no set-up file loaded"),
                       named.endswith("kept_design_space.json"), moved == [4, 250.0, True], app.ad.degree,
                       app.ad.span / span,
                       max(abs(app.ad.values[n] - vals[n]) for n in vals), app.ad.meta == meta,
                       app.npts.get() == "6", app.span_box.get() == f"{span:.2f}", app.vhalf.get(),
                       app.case.get() == "other", app.setup_label.cget("text") == named,
                       asked[0]["initialdir"] == tmp, app.status.cget("background") == XT.STATUS["ok"][0],
                       loaded == f"set-up loaded: {setup}"],
                      [1, 1, 1, 1, 5, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1], 1e-9)
                app.set_value("H", 0.9 * app.ad.get("H"), None)
                design, bad = app.ad.design, os.path.join(tmp, "other_design.json")
                with open(bad, "w") as fh:
                    json.dump({"H": 60.0, "curves": {}}, fh)       # a design JSON, not a set-up
                XT.filedialog.askopenfilename = lambda **kw: bad
                app.on_load_space()
                root.update()
                text = app.status.cget("text")
                XT.filedialog.askopenfilename = lambda **kw: ""    # the dialog cancelled
                app.on_load_space()
                check("Load ... of a file that is no set-up: said in red, the design as it was; a cancelled dialog "
                      "changes nothing",
                      [app.ad.design is design, "is not a rudder set-up" in text,
                       app.status.cget("background") == XT.STATUS["error"][0],
                       app.setup_label.cget("text") == named], [1, 1, 1, 1], 0)
            finally:
                XT.filedialog.askopenfilename, app.ad.out_dir = ask_open, out_dir
        import xgeom_from_cad as FC
        import xgeom_rudder as XR
        with tempfile.TemporaryDirectory() as tmp:
            ad0 = XR.RudderAdapter(out_dir=tmp)
            ad0.set("LE.dx3", -7.0)
            fake = ad0.save_space(os.path.join(tmp, "fake_design_space.json"))
            real, out_dir = FC.from_cad, app.ad.out_dir
            FC.from_cad = lambda kind, path, out, case, progress: {"kind": kind, "setup": fake, "file": tmp,
                                                                    "forms": {"LE": (2, 0.0)}, "report": ["LE: order 2"],
                                                                    "seconds": 0.0}
            XT.filedialog.askopenfilename = lambda **kw: os.path.join(tmp, "some.step")
            try:
                app.on_from_cad()
                wait_job(app, root)
                root.update()
                check("From CAD ... (the extraction stubbed): the job runs off the GUI thread, its set-up is loaded "
                      "into the panel when it ends (the value it holds, the Set-up box naming it)",
                      [app.ad.get("LE.dx3"), app.ad.setup_file == fake, app.setup_label.cget("text").endswith(
                          "fake_design_space.json"), app.status.cget("background") == XT.STATUS["ok"][0]],
                      [-7.0, 1, 1, 1], 1e-12)
            finally:
                FC.from_cad, app.ad.out_dir = real, out_dir
                XT.filedialog.askopenfilename = ask_open
        if not XL.occ_available():
            print("       Save CAD skipped: no pythonocc-core or cadquery-ocp here")
            return
        builds, ask, log = [], XT.filedialog.asksaveasfilename, XT.LOG
        real = app.ad.build_task
        app.ad.build_task = lambda case, **kw: builds.append(case) or real(case, **kw)
        with tempfile.TemporaryDirectory() as tmp:
            app.ad.out_dir = tmp                                   # the build's case files go here
            XT.LOG = os.path.join(tmp, "log.txt")
            picks = iter(os.path.join(tmp, n) for n in ("saved.step", "saved.igs", "changed"))
            XT.filedialog.asksaveasfilename = lambda **kw: next(picks)
            try:
                said = []
                for change in (False, False, True):
                    if change:
                        app.set_value("H", 0.9 * app.ad.get("H"), None)
                    app.save_btn.invoke()
                    busy = str(app.save_btn.cget("state"))
                    wait_job(app, root)
                    said.append(app.status.cget("text"))
                built = app.built["rudder"]["step"]
                faces = XL.count_faces(XL.read_iges(os.path.join(tmp, "saved.igs")))
                check("Save CAD: built first and saved (the build's STEP), then as IGES without a new build; a "
                      "change builds again (no extension: .step); the button off while it runs",
                      [len(builds), said[0].startswith("XCAD done") and "saved" in said[0],
                       said[1].startswith("Save CAD done"), faces, os.path.exists(os.path.join(tmp, "changed.step")),
                       open(os.path.join(tmp, "changed.step"), "rb").read() == open(built, "rb").read(),
                       busy == "disabled", app.status.cget("background") == XT.STATUS["ok"][0]],
                      [2, 1, 1, 6, 1, 1, 1, 1], 0)
            finally:
                XT.filedialog.asksaveasfilename, XT.LOG = ask, log
                app.ad.build_task = real
    finally:
        root.destroy()
        os.chdir(cwd)


def test_from_cad():
    print("\nset-ups from other files (xgeom_from_cad.py, the rudder adapter on an extraction folder)")
    import modify_control as MC
    import xgeom_from_cad as FC
    check("case names from CAD file names", [FC.case_name("a b/Wind Tunnel-Rudder (2).stp") == "Wind_Tunnel_Rudder_2",
                                            FC.case_name("x/(-).stp") == "cad"], [1, 1], 0)
    got = FC.pick([(3, "a", 0.05), (4, "b", 0.0101), (5, "c", 0.0100), (7, "d", 0.0094), (8, "e", 0.002)])
    check("pick: the least error, a costlier form only when FIT_TIE better (c and d not over b, e over all)",
          [got[0] == "e", got[1], FC.pick([(3, "a", 0.05), (5, "c", 0.0495)])[0] == "a"], [1, 0.002, 1], 0)
    need = [MC.CORNER_POINTS, MC.SECTION_TABLE, MC.XCAD_ORIGINAL, MC.STACK]
    missing = [p for p in need if not os.path.exists(os.path.join(HERE, p))]
    if missing:
        print("       the rudder adapter on an extraction folder skipped, not here: " + ", ".join(missing))
        return
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        import shutil
        import rudder_modify as RM
        import xgeom_rudder as XR
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "other_extraction")           # the repo's extraction, copied
            os.makedirs(folder)
            for p in need:
                shutil.copy(os.path.join(HERE, p), folder)
            files = FC.rudder_files(folder)
            orders = FC.best_rudder_orders(files["corner_points"], files["section_table"])
            ad = XR.RudderAdapter(files=files, fit_orders={c: v[0] for c, v in orders.items()}, out_dir=tmp)
            ad.set("LE.dx3", -12.0)
            setup = ad.save_space(os.path.join(tmp, "other_design_space.json"))
            ad2 = XR.RudderAdapter()                                 # modify_control's files, then the set-up
            ad2.load_space(setup)
            rr = np.linspace(0.0, ad.get("H"), 101)
            base = XR.RudderAdapter(files=files, out_dir=tmp)       # modify_control's orders (3) on the same files
            d_orig = max(float(np.abs(ad.orig.value(c, rr) - base.orig.value(c, rr)).max()) for c in ad.orig.names)
            check("the rudder adapter on an extraction folder: the files found, the laws at the orders that fit "
                  "best (within 1e-4 of the cubic fits: this rudder's laws are that simple), a set-up from them "
                  "loaded into the default adapter switches its files, the orders and the values",
                  [all(os.path.isabs(v) for v in files.values()), all(2 <= v[0] <= FC.MAX_LAW_ORDER for v in orders.values()),
                   d_orig < 1e-4, ad2.files == ad.files, {c: ad2.orig.curves[c].order for c in ad.orig.names} ==
                   {c: ad.orig.curves[c].order for c in ad.orig.names}, ad2.get("LE.dx3"),
                   max(abs(ad2.values[n] - ad.values[n]) for n in ad.values), len(ad2.preview()["problems"])],
                  [1, 1, 1, 1, 1, -12.0, 0, 0], 1e-9)
            sp, d, rec = RM.load_design_space(setup)
            check("rudder_modify.load_design_space reads it too (the files it names)", [d.H, len(rec["files"])],
                  [ad.get("H"), 4], 1e-12)
    finally:
        os.chdir(cwd)


def test_temp_cad():
    print("\ntemp_rudder_cad.py: modified sections of any number of points, two lofts")
    original = os.path.join(HERE, "outputs", "sections_xcad_with_cap_y0_y200_m.dat")
    if not os.path.exists(original):
        print("       skipped: no outputs/sections_xcad_with_cap_y0_y200_m.dat here")
        return
    import temp_rudder_cad as TC
    orig = TC.read_sections(original)
    flats = [s for s in orig if TC.is_flat(s)]
    below = [s for s in flats if s[0, 2] <= 80.0]
    layout = TC.Layout(flats, len(flats[0]))

    def spaced(s, n):                                    # the section's smooth curve at n points, cosine arcs
        sl = XL._SmoothLoop(s, (len(s) + 1) // 2 - 1)
        u = lambda k: 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, k)))
        q = np.vstack((sl.at(u((n + 1) // 2) * sl.s_te),
                       sl.at(sl.s_te + u(n - (n + 1) // 2 + 1) * (sl.S - sl.s_te))[1:]))
        q[:, 2] = s[0, 2]
        return q

    s0 = below[40]
    q = spaced(s0, 121)
    on, how = layout.onto(q)
    back, how0 = layout.onto(s0)
    rev, how_r = layout.onto(np.vstack((q[:1], q[1:-1][::-1], q[:1])))
    opened, _ = layout.onto(q[:-1])
    check("a section of 121 points goes onto the original's layout at its height (200 points, the LE first, the "
          "TE at index 99): within 0.03 mm of the original's points there; the original's own section comes back "
          "unchanged; the other side first is turned round, an open loop closed, to the same points",
          [len(on), how == "resampled", np.abs(on[0] - s0[0]).max(), np.abs(on[99] - s0[99]).max() < 1e-6,
           np.linalg.norm(on - s0, axis=1).max() < 0.03, how0 == "kept", np.abs(back - s0).max(),
           how_r == "turned round", np.abs(rev - on).max(), np.abs(opened - on).max()],
          [200, 1, 0, 1, 1, 1, 0, 1, 0, 0], 1e-9)
    if not XL.occ_available():
        print("       the two lofts skipped: no pythonocc-core or cadquery-ocp here")
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mixed_m.dat")
        with open(path, "w") as fh:            # 121 points, no '#' lines, every third reversed, every other open
            for k, s in enumerate(below):
                lp = spaced(s, 121)
                if k % 3 == 0:
                    lp = np.vstack((lp[:1], lp[1:-1][::-1], lp[:1]))
                lp = lp[:-1] if k % 2 else lp
                fh.write("\n".join(f"{x / 1000:.8f} {y / 1000:.8f} {z / 1000:.8f}" for x, y, z in lp) + "\n")
        t0 = time.time()
        res, line = TC.build(path, original)
        print(f"       {line}; {res.faces} faces, volume {res.volume:.1f} mm^3, {time.time() - t0:.0f} s")
        check("two lofts joined at the top modified section (loop 80): one valid closed solid, nine patches and "
              "the root; no step at H (a change of layout there read as a 3.6 mm step before 20 Nov 2026), the "
              "patches on the loops",
              [res.valid, res.free_edges, res.faces, res.joint, len(res.steps), res.v_error < 0.01],
              [1, 0, 10, 80, 0, 1], 0)


def wait_job(app, root, limit=600.0):
    """Let the window run until its build (or save) is done."""
    t0 = time.time()
    while app._job is not None and time.time() - t0 < limit:
        root.update()
        time.sleep(0.05)


def main():
    print("design tool self-check")
    test_loft()
    test_adapter()
    test_span()
    test_from_cad()
    test_temp_cad()
    if "--gui" in sys.argv:
        test_gui()
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
