"""
bb2_sternplane_loft.py

BB2's stern plane rebuilt with xcad_loft's mesh patch topology, the blade's five-patch decomposition (user,
6 Nov 2026), and the set-up files of bb2_sternplanes.py written again with it; the hull and the propeller
stay as bb2_build.py made them.

The stern plane of hull.iges (bb2_sternplane.step) is one imported B-rep whose patches do not follow the mesh
blocking. It is sectioned here and lofted again: cuts normal to its span (the 45 deg direction, planes at
constant s = (y + z) / sqrt(2)) from S0, about 130 mm under the hull at the trailing edge, up to the foot of
the tip cap at s = 6811.5 mm, close together over the last 200 mm, where the TE-tip corner's rounding
sweeps the TE forward; above it the cap is resampled into rings (cuts normal to x at the chord
stations give the cap's cross profile from the base over the crest; a ring holds the points at one fraction
of each profile's arc, symmetrized onto +-w, runs through the cap's two base corners, and is resampled by
arc into the flat sections' layout, which keeps the loft's columns coherent at the corners). Each loop is laid out as
xcad_loft expects (201 points, the LE first, the w < 0 side to the TE and back) in the frame (x, w, s), w
along (0, 1, -1)/sqrt(2) so the triad is right-handed; loft_loops builds the solid there, and it is turned
to the STP frame after: the LE and TE strips wrapping their edges (strip_frac of the root chord wide), the
central pressure and suction panels, the tip patch over the top TIP_FRAC of the span, and the flat root; six
faces. Below s = 2920 the plane's buried root is chopped by its sloping root face, so the
sections there are synthesized instead: the straight taper continued, a linear fit per loop point over the
full sections at s = 2920 to 3720 (which lie within 0.1 mm of it). The chop sits 150 mm or more inside the
hull, so wherever the synthetic root differs from the solid the difference is buried; against the true
sections near the TE, exposed down to s = 1975, it stays within about 0.25 mm. The rebuilt plane's part
outside the hull (cut by bb2_hull_hub.step) is compared with the original's: the volumes, and cuts over the
exposed span and across the cap (the largest distance per cut printed).

The set-up files are then written as bb2_sternplanes.py writes them, with the lofted plane in place of the
imported one (the same names, so they replace the 22 Oct files): bb2_vehicle_rudders and
bb2_vehicle6_rudders (.step/.iges), bb2_hull_hub_rudder_quadrant.iges, bb2_hull_rudder_quadrant
(.step/.iges) and bb2_quadrant_sector (.step/.iges); and the lofted plane itself as
bb2_sternplane_loft.step/.iges (bb2_sternplane.step stays the extracted original).

    python bb2_sternplane_loft.py [out_dir]        (reads bb2_build.py's outputs; about 17 min)
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # the X_geom folder
sys.path.insert(0, HERE)

import bb2_build as B                                      # noqa: E402
import bb2_sections as S                                   # noqa: E402
import bb2_sternplanes as SP                               # noqa: E402
import vehicle_cad as VC                                   # noqa: E402
import xcad_loft as XL                                     # noqa: E402

E_SPAN = SP.E_SPAN                                         # the span direction (45 deg)
E_THICK = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)        # thickness, w ((x, w, s) right-handed)
S0 = 1850.0                                                # the loft's root, mm along the span (inside the hull)
S_FULL = 2920.0                                            # the chop ends at 2918: full sections above
FIT_S = np.arange(2920.0, 3721.0, 80.0)                    # the root taper's fit runs over sections here
S_CAP = 6811.5                                             # the cap's foot: the side faces end at 6812
N_FLATS = 112                                              # sections from S0 up to the dense band
N_TIP = 40                                                 # sections over the band (the last BAND_MM), 5 mm
                                                           # apart: the TE-tip corner's rounding (s 6787 on)
                                                           # needs them in the smooth interim the tip patch
                                                           # resamples; they stop 2.5 mm under the cap's
                                                           # foot, the first ring being the next row
N_PTS = 201                                                # points per loop (the LE first, TE at index 100)
TIP_FRAC = 0.03                                            # the tip patch owns the top 3 % of the span: the
                                                           # cap is 1 % of it, and a taller tip patch makes
                                                           # the wrap family too uneven to fit cleanly
N_POLES = 60                                               # poles of a section curve: 40 leaves the fit at
                                                           # the TE's blunt roll near 0.65 mm and lets the
                                                           # seam fold there; 60 holds it near 0.3 mm, valid
N_CHORD = 99                                               # interior chord stations of a cap ring
RING_F = (0.03, 0.07, 0.12, 0.19, 0.28, 0.38, 0.50, 0.62, 0.74, 0.85, 0.93, 0.98)   # arc fractions, base to crest
BAND_MM = 200.0                                            # the dense station band under the cap's foot
CHECK_S = (2100.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 6600.0, 6800.0)
CHECK_X = (5400.0, 5600.0, 6000.0, 7000.0, 8000.0, 8050.0, 8075.0)


def plane_cut(shape, origin, normal, n=400):
    """The cut of a shape by a plane, chained into one ordered polyline (bb2_sections._chain)."""
    gp_Pln, gp_Pnt, gp_Dir = XL._occ("gp", "gp_Pln", "gp_Pnt", "gp_Dir")
    Section = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Section")
    algo = Section(shape, gp_Pln(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, normal))))
    algo.Build()
    edges = S._explore(algo.Shape(), "EDGE")
    if not edges:
        return None
    return S._chain([S._polyline(e, n) for e in edges])


def span_cut(shape, s, n=400):
    return plane_cut(shape, s * E_SPAN, E_SPAN, n)


def _arc(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _resample(p, stations):
    """Points of a polyline at fractions of its arc."""
    s = _arc(p)
    out = np.empty((len(stations), p.shape[1]))
    for c in range(p.shape[1]):
        out[:, c] = np.interp(np.asarray(stations) * s[-1], s, p[:, c])
    return out


def cosine(n):
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))


def _on_symmetry(p, i):
    """p[i], an x extreme of the loop, moved onto the symmetry line w = 0 (where the extreme lies, the
    sections being symmetric): the crossing of the segment next to it, by linear interpolation."""
    n = len(p)
    for j in range(i - 2, i + 2):
        w1, w2 = p[j % n, 1], p[(j + 1) % n, 1]
        if w1 * w2 < 0.0:
            p[i] = p[j % n] + w1 / (w1 - w2) * (p[(j + 1) % n] - p[j % n])
            return
    p[i, 1] = 0.0


def flat_loop(plane, s):
    """One section loop in the loft's layout: (N_PTS, 3) in (x, w, s), the LE (largest x) first, the w < 0
    side to the TE (smallest x), back along the other side; the LE and the TE exactly on w = 0."""
    p3 = span_cut(plane, s, 500)
    p = np.column_stack((p3[:, 0], p3 @ E_THICK))
    if np.linalg.norm(p[0] - p[-1]) > 1.0:
        raise RuntimeError(f"the cut at s = {s:.1f} mm did not close")
    p = p[:-1]                                             # closed: drop the duplicate
    i_le, i_te = int(np.argmax(p[:, 0])), int(np.argmin(p[:, 0]))
    p = np.roll(p, -i_le, axis=0)
    i_te = (i_te - i_le) % len(p)
    _on_symmetry(p, 0)
    _on_symmetry(p, i_te)
    a, b = p[:i_te + 1], np.vstack((p[i_te:], p[:1]))      # LE to TE, TE back to LE
    if a[:, 1].mean() > b[:, 1].mean():                    # the w < 0 side first
        a, b = b[::-1], a[::-1]
    u = cosine(101)
    side1, side2 = _resample(a, u), _resample(b, u)
    loop2 = np.vstack((side1, side2[1:]))
    return np.column_stack((loop2[:, 0], loop2[:, 1], np.full(len(loop2), s)))


def _smooth_ends(v, k=12, w=5):
    """v with its first and last k entries smoothed by a w-point moving average: the rings' chord stations
    cluster at the corners, where the corner blends kink the profiles; the kink is ironed into a ramp there
    (the middle stays as it is)."""
    out = v.copy()
    pad = np.r_[np.repeat(v[0], w // 2), v, np.repeat(v[-1], w // 2)]
    sm = np.convolve(pad, np.ones(w) / w, mode="valid")
    out[:k], out[-k:] = sm[:k], sm[-k:]
    return out


def cap_rings(plane, corners):
    """The cap resampled into rings: cuts normal to x at the chord stations give the cross profile from the
    base (s = S_CAP) over the crest; ring f holds the points at the fraction f of each half profile's arc
    (base to crest), symmetrized onto +-w, and runs through the two base corners. Returns the rings, base to
    crest, in the loft's layout and frame (x, w, s)."""
    x_te, x_le = corners
    xs = x_te + (x_le - x_te) * cosine(N_CHORD + 2)[1:-1][::-1]        # LE (largest x) first
    halves = []                                            # per station: (w < 0 half, w > 0 half), base to crest
    for x0 in xs:
        p3 = plane_cut(plane, (x0, 0.0, 0.0), (1.0, 0.0, 0.0), 400)
        q = np.column_stack((p3 @ E_THICK, p3 @ E_SPAN))
        q = q[q[:, 1] > S_CAP - 0.6]
        o = np.argsort(q[:, 0])
        q = q[o]
        i_top = int(np.argmax(q[:, 1]))
        lo, hi = q[:i_top + 1], q[i_top:][::-1]            # each from its base end to the crest
        halves.append((lo, hi))
    rings = []
    for f in RING_F:
        p1 = np.array([_resample(lo, [f])[0] for lo, hi in halves])    # (w, s), the w < 0 side, LE to TE
        p2 = np.array([_resample(hi, [f])[0] for lo, hi in halves])
        a = 0.5 * (p2[:, 0] - p1[:, 0])                    # the cap is symmetric: the halves averaged onto
        sm = 0.5 * (p1[:, 1] + p2[:, 1])                   # +-w, or the seams' sides drift apart
        a, sm = _smooth_ends(a), _smooth_ends(sm)          # the corner blends' kinks ironed out
        le = np.array([x_le, 0.0, S_CAP])
        te = np.array([x_te, 0.0, S_CAP])
        side1 = np.vstack((le, np.column_stack((xs, -a, sm)), te))     # the w < 0 side, LE to TE
        side2 = np.vstack((te, np.column_stack((xs, a, sm))[::-1], le))
        u = cosine(101)                                    # resampled by arc as flat_loop resamples, so that
        r1, r2 = _resample(side1, u), _resample(side2, u)  # an index means the same arc fraction in every
        rings.append(np.vstack((r1, r2[1:])))              # row, or the loft's columns kink at the corners
    return rings


def root_fit(plane):
    """The buried root's sections (the chop, below S_FULL) as the straight taper continued: a linear fit in
    s, per loop point, over the full sections at FIT_S; returns (a, b), the loop at s being a + b s."""
    loops = np.array([flat_loop(plane, s) for s in FIT_S])
    des = np.column_stack((np.ones(len(FIT_S)), FIT_S))
    coef, *_ = np.linalg.lstsq(des, loops.reshape(len(FIT_S), -1), rcond=None)
    resid = np.abs((des @ coef).reshape(loops.shape) - loops).max()
    print(f"the root taper's fit over s {FIT_S[0]:.0f} to {FIT_S[-1]:.0f} mm: within {resid:.3f} mm")
    return coef[0].reshape(-1, 3), coef[1].reshape(-1, 3)


def synth_loop(a, b, s):
    lp = a + b * s
    lp[:, 2] = s
    return lp


def cut_polylines(shape, origin, normal, n=400):
    """The cut of a shape by a plane, one polyline per section edge (a boolean part's cut can be several
    separate loops, so they are not chained)."""
    gp_Pln, gp_Pnt, gp_Dir = XL._occ("gp", "gp_Pln", "gp_Pnt", "gp_Dir")
    Section = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Section")
    algo = Section(shape, gp_Pln(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, normal))))
    algo.Build()
    return [S._polyline(e, n) for e in S._explore(algo.Shape(), "EDGE")]


def nearest_distance(pts, curves):
    """Largest distance from 2D points to a set of polylines, mm (nearest vertices, refined on their
    segments)."""
    from scipy.spatial import cKDTree
    best = np.full(len(pts), np.inf)
    for curve in curves:
        if len(curve) < 2:
            continue
        i = cKDTree(curve).query(pts)[1]
        for di in (-1, 0):
            j = np.clip(i + di, 0, len(curve) - 2)
            a, b = curve[j], curve[j + 1]
            ab = b - a
            t = np.einsum("ij,ij->i", pts - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1.0e-12)
            foot = a + np.clip(t, 0.0, 1.0)[:, None] * ab
            best = np.minimum(best, np.linalg.norm(pts - foot, axis=1))
    return float(best.max())


def check_against(plane, loft):
    """The lofted plane against the original at cuts over the span and across the cap: the largest distance
    per cut (both ways round), printed; returns the largest of all. Below the chop's end only the airfoil
    clear of the chop's trace is compared (the loft is synthetic there, and the strip within 30 mm of the
    trace carries the chop's corner blend); the hull reaches neither."""
    worst = 0.0
    for s in CHECK_S:
        a = cut_polylines(plane, s * E_SPAN, E_SPAN, 1200)
        b = cut_polylines(loft, s * E_SPAN, E_SPAN, 1200)
        qa = [np.column_stack((p[:, 0], p @ E_THICK)) for p in a]
        qb = [np.column_stack((p[:, 0], p @ E_THICK)) for p in b]
        pa, pb = np.vstack(qa), np.vstack(qb)
        note = ""
        if s < S_FULL:                                     # the original is chopped at this height
            bound = pa[:, 0].max() - 30.0
            pa, pb = pa[pa[:, 0] < bound], pb[pb[:, 0] < bound]
            note = " (the airfoil clear of the chop)"
        d = max(nearest_distance(pa, qb), nearest_distance(pb, qa))
        worst = max(worst, d)
        print(f"  cut s = {s:6.0f} mm: largest distance {d:6.2f} mm{note}")
    for x0 in CHECK_X:
        a = cut_polylines(plane, (x0, 0.0, 0.0), (1.0, 0.0, 0.0), 800)
        b = cut_polylines(loft, (x0, 0.0, 0.0), (1.0, 0.0, 0.0), 800)
        qa = [np.column_stack((p @ E_THICK, p @ E_SPAN)) for p in a]
        qb = [np.column_stack((p @ E_THICK, p @ E_SPAN)) for p in b]
        pa = np.vstack(qa)
        pb = np.vstack(qb)
        pa = pa[pa[:, 1] > S_CAP - 400.0]                  # points of the tip region; whole curves
        pb = pb[pb[:, 1] > S_CAP - 400.0]
        d = max(nearest_distance(pa, qb), nearest_distance(pb, qa))
        worst = max(worst, d)
        print(f"  cap cut x = {x0:6.0f} mm: largest distance {d:6.2f} mm (the tip region)")
    return worst


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "outputs")
    t0 = time.time()
    plane = S._explore(S.read_cad(os.path.join(out, "bb2_sternplane.step")), "SOLID")[0]
    hull_hub = S._explore(S.read_cad(os.path.join(out, "bb2_hull_hub.step")), "SOLID")[0]

    # the loops (sections from S0 to the cap's foot, the chopped root synthesized, the cap as rings),
    # lofted in the frame (x, w, s)
    base = flat_loop(plane, S_CAP)
    corners = float(base[:, 0].min()), float(base[:, 0].max())
    a, b = root_fit(plane)
    stations = np.r_[np.linspace(S0, S_CAP - BAND_MM, N_FLATS, endpoint=False),
                     np.linspace(S_CAP - BAND_MM, S_CAP - 2.5, N_TIP)]
    loops = [synth_loop(a, b, s) if s < S_FULL else flat_loop(plane, s) for s in stations]
    loops += cap_rings(plane, corners)
    res = XL.loft_loops(loops, n_poles=N_POLES, v_poles=30, tip_frac=TIP_FRAC)
    if res.shape is None or not res.valid or res.free_edges:
        raise RuntimeError(f"the loft failed: {res}")
    m = np.column_stack((np.array([1.0, 0.0, 0.0]), E_THICK, E_SPAN))      # x -> m x: to the STP frame
    loft = VC.transformed(res.shape, m, np.zeros(3))
    n_s, ok, vol = VC.solid_report(loft)
    if n_s != 1 or not ok:
        raise RuntimeError(f"the turned loft: {n_s} solids, valid {ok}")
    print(f"lofted in {time.time() - t0:.0f} s: {res.faces} faces, fit {res.fit_error:.2f} mm, "
          f"v-fit {res.v_error:.3f} mm, {vol:.6e} mm^3 (with the synthetic root; the chopped "
          f"original {VC.volume(plane):.6e})")
    print("against the original:")
    worst = check_against(plane, loft)
    print(f"  largest over the cuts: {worst:.2f} mm")
    XL.write_step(loft, os.path.join(out, "bb2_sternplane_loft.step"))
    XL.write_iges(loft, os.path.join(out, "bb2_sternplane_loft.iges"))

    # the four stern planes outside the hull, and the exposed part's volume against the original's
    exposed = [SP.boolean("Cut", VC.transformed(loft, VC.turn_x(0.5 * np.pi * k), np.zeros(3)), hull_hub)
               for k in range(4)]
    vols = [VC.volume(e) for e in exposed]
    old = VC.volume(SP.boolean("Cut", plane, hull_hub))
    if max(vols) > 1.001 * min(vols) or max(vols) > 0.999 * vol:
        raise RuntimeError(f"the stern planes' cuts by the hull differ (volumes {vols})")
    print(f"outside the hull: {vols[0]:.6e} mm^3 each (the four within {max(vols) / min(vols) - 1:.1e}); "
          f"the original's part outside {old:.6e} mm^3 ({vols[0] / old - 1:+.2e})")

    # the set-up files, as bb2_sternplanes.py writes them, with the lofted plane
    for name in ("vehicle", "vehicle6"):
        vehicle = SP.one_solid(VC.fuse([S.read_cad(os.path.join(out, f"bb2_{name}.step"))] + exposed))
        print(f"{name} with the four lofted stern planes: solids, valid, volume mm^3 {VC.solid_report(vehicle)}")
        XL.write_step(vehicle, os.path.join(out, f"bb2_{name}_rudders.step"))
        XL.write_iges(vehicle, os.path.join(out, f"bb2_{name}_rudders.iges"))
    q = B.quadrant(SP.one_solid(VC.fuse([hull_hub, exposed[0]])))
    print(f"hull, hub and stern plane in the quadrant y >= 0, z >= 0: {VC.solid_report(q)}")
    B.write_iges_solids(q, os.path.join(out, "bb2_hull_hub_rudder_quadrant.iges"))

    # the propeller periodic: the hull without the hub, the stern plane, and the blade sector
    hull = S._explore(S.read_cad(os.path.join(out, "bb2_hull.step")), "SOLID")[0]
    q = B.quadrant(SP.one_solid(VC.fuse([hull, exposed[0]])))
    n_s, ok, vq = VC.solid_report(q)
    if n_s != 1 or not ok:
        raise RuntimeError(f"the hull and stern plane's quadrant: {n_s} solids, valid {ok}")
    XL.write_step(q, os.path.join(out, "bb2_hull_rudder_quadrant.step"))
    B.write_iges_solids(q, os.path.join(out, "bb2_hull_rudder_quadrant.iges"))
    sector = S._explore(S.read_cad(os.path.join(out, "bb2_blade_sector.step")), "SOLID")[0]
    ends = VC._bbox(q)[0, 0], VC._bbox(sector)[1, 0]
    if abs(ends[0] - ends[1]) > 0.01:
        raise RuntimeError(f"the hull ends at x {ends[0]:.3f} mm, the hub strip at {ends[1]:.3f}")
    lo, hi = SP.end_face_span(sector, ends[1])
    turn = SP.SECTOR_CENTRE_DEG - 0.5 * (lo + hi)
    sector = VC.transformed(sector, VC.turn_x(np.radians(turn)), np.zeros(3))
    lo2, hi2 = SP.end_face_span(sector, ends[1])
    print(f"blade sector: its strip's face at the hull's end from {lo:.2f} to {hi:.2f} deg, "
          f"turned by {turn:.2f} deg to {lo2:.2f} to {hi2:.2f} deg")
    both = VC.compound([q, sector])
    XL.write_step(both, os.path.join(out, "bb2_quadrant_sector.step"))
    B.write_iges_solids(both, os.path.join(out, "bb2_quadrant_sector.iges"))
    print(f"hull without the hub and stern plane in the quadrant: valid, {vq:.6e} mm^3; with the blade "
          f"sector: {VC.solid_report(both)[:2]}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
