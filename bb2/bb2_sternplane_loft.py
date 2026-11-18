"""
bb2_sternplane_loft.py

BB2's stern plane rebuilt with xcad_loft's patches in the blade's layout (five and the flat root; user's
sketch, 17 Nov 2026: the tip patch's corners sit on the strip lines under the tip, the strips' top edges run
up to the tops of the LE and TE lines, and the tip patch covers the cap and the sides above those edges), and
the set-up files of bb2_sternplanes.py written again with it; the hull and the propeller stay as
bb2_build.py made them. The plane keeps its own tip: the cap is not flattened, the patches take it as it is.

The stern plane of hull.iges (bb2_sternplane.step) is one imported B-rep whose patches do not follow the mesh
blocking. It is sectioned here and lofted again: cuts normal to its span (the 45 deg direction, planes at
constant s = (y + z) / sqrt(2)) from S0, about 130 mm under the hull at the trailing edge, up to the foot of
the tip cap: 5 mm apart over the last 200 mm, 1 mm apart over the last 30, where the TE-tip corner's rounding
sweeps the TE forward, and the last one at s = 6811.99, right under the end of the side faces at 6812. That
one is the rim, where the straight wall ends and the cap's rounding begins. Above it the cap is resampled
into rings: cuts normal to x at chord stations (1.2 mm apart near the two ends, where the cap rolls over the
LE and TE lines within a few mm) give the cap's cross profile from its base, exactly on the rim's height, to
its crest on w = 0; a ring holds the points at one fraction of each profile's arc (0.01 to 1: the last ring
is the crest line), symmetrized onto +-w, runs through the rim's LE and TE points, and is resampled by arc
along the spline through its stations into the flat sections' layout, which keeps the loft's columns
coherent at the corners.

Every loop point lies on the original surface. The section algorithm returns a cut as a polygon through its
own walking points, 2 to 3 mm apart: up to 0.011 mm inside the nose, which a loft through the loops shows as
a ripple. So the cuts are taken with the algorithm's approximation on (curves within 0.001 mm of the
surface), and each resampled point is then dropped onto the surface itself: its closest point (Foot), put
back into the cut's plane for a section.

Each loop is laid out as xcad_loft expects (801 points, the LE first, the w < 0 side to the TE and back; the
TE is a 15 mm round, and where it meets the sides the loop's points have to be about 2 mm apart for the
smooth curve through them to hold the shape) in the frame (x, w, s), w along (0, 1, -1)/sqrt(2) so the triad
is right-handed; loft_loops builds the solid there, and it is turned to the STP frame after: the LE and TE
strips wrapping their edges, as wide as loft_loops finds the edges' rounding (about 190 mm to each side of
the LE, 1.5 times the nose's rounding; about 22 mm at the TE, the 15 mm round itself, so its seams lie where
the round meets the flanks), the central pressure and suction panels, the tip patch, and the flat root; six
faces. The strips' top edges rise at loft_loops' slope of 1.5 in the side view, which puts the tip patch's
corners about 146 mm under the top of the LE line and 7 mm under the top of the TE line. Below s = 2920 the
plane's buried root is chopped by its sloping root face, so the sections there are synthesized instead: the
straight taper continued, a linear fit per loop point over the full sections at s = 2920 to 3720 (which lie
within 0.1 mm of it). The chop sits 150 mm or more inside the hull, so wherever the synthetic root differs
from the solid the difference is buried; against the true sections near the TE, exposed down to s = 1975, it
stays within about 0.25 mm. The rebuilt plane's part outside the hull (cut by bb2_hull_hub.step) is compared
with the original's: the volumes, cuts over the exposed span and across the cap, the tip's two ends
included (the largest distance per cut printed), and the tip patch point by point (its distance from the
original).

The set-up files are then written as bb2_sternplanes.py writes them, with the lofted plane in place of the
imported one (the same names, so they replace the 22 Oct files): bb2_vehicle_rudders and
bb2_vehicle6_rudders (.step/.iges), bb2_hull_hub_rudder_quadrant.iges, bb2_hull_rudder_quadrant
(.step/.iges) and bb2_quadrant_sector (.step/.iges); and the lofted plane itself as
bb2_sternplane_loft.step/.iges (bb2_sternplane.step stays the extracted original).

    python bb2_sternplane_loft.py [out_dir]        (reads bb2_build.py's outputs; about 24 min, the
                                                    sectioning 19 of them)
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
FRAME = np.column_stack((np.array([1.0, 0.0, 0.0]), E_THICK, E_SPAN))     # (x, w, s) -> the STP frame: m x
S0 = 1850.0                                                # the loft's root, mm along the span (inside the hull)
S_FULL = 2920.0                                            # the chop ends at 2918: full sections above
FIT_S = np.arange(2920.0, 3721.0, 80.0)                    # the root taper's fit runs over sections here
S_CAP = 6811.99                                            # the rim, the last section: the side faces end
                                                           # at 6812, the wall is straight up to there
S_BAND = 6811.5                                            # the sections under the rim are counted down
N_FLATS = 112                                              # from here: N_FLATS from S0 up to the band,
BAND_MM = 200.0                                            # the band: sections 5 mm apart, and
FINE_MM = 30.0                                             # 1 mm apart over its last FINE_MM: the TE-tip
                                                           # corner's rounding (s 6787 on) turns the TE line
                                                           # forward within 25 mm, and the patches' rows of
                                                           # poles are as close as the sections allow
N_PTS = 801                                                # points per loop (the LE first, the TE in the
                                                           # middle): about 2 mm apart where the TE's 15 mm
                                                           # round meets the sides (201 left 0.15 mm there)
N_CUT = 4000                                               # points per edge of a span cut's polyline
N_POLES = 60                                               # poles along a section between the strips, before
                                                           # xcad_loft's own added knots
V_POLES = 40                                               # rows of poles along the span
RING_F1, RING_TOP = 0.01, 1.0                              # arc fractions of the first ring (just above
N_RINGS, RING_POWER = 30, 1.5                              # the rim) and of the last (the crest line); the
                                                           # rings between crowd towards the first (the
                                                           # profile turns from the wall to 25 deg within
                                                           # 0.2)
FOOT_MM = 0.1                                              # a closest point further off than this is on
                                                           # another branch of the surface: not taken
RING_H0, RING_N0 = 1.2, 12                                 # chord stations of the rings: RING_H0 mm apart
RING_GROW, RING_HMAX = 1.15, 50.0                          # for the first RING_N0 from each end, then
                                                           # growing by RING_GROW a step up to RING_HMAX
CHECK_S = (2100.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 6600.0, 6790.0, 6800.0, 6808.0)
CHECK_X = (5335.0, 5345.0, 5360.0, 5400.0, 5600.0, 6000.0, 7000.0, 8000.0, 8050.0, 8070.0, 8078.0)


def plane_cut(shape, origin, normal, n=400):
    """The cut of a shape by a plane, chained into one ordered polyline (bb2_sections._chain). The section's
    approximation is on: its curves lie within 0.001 mm of the surface (without it a cut is the polygon
    through the algorithm's walking points, up to 0.011 mm inside the plane's nose)."""
    gp_Pln, gp_Pnt, gp_Dir = XL._occ("gp", "gp_Pln", "gp_Pnt", "gp_Dir")
    Section = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Section")
    algo = Section(shape, gp_Pln(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, normal))), False)
    algo.Approximation(True)
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
    side to the TE (smallest x), back along the other side (each side by arc, crowded towards both ends);
    the LE and the TE exactly on w = 0."""
    p3 = span_cut(plane, s, N_CUT)
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
    u = cosine((N_PTS + 1) // 2)
    side1, side2 = _resample(a, u), _resample(b, u)
    loop2 = np.vstack((side1, side2[1:]))
    return np.column_stack((loop2[:, 0], loop2[:, 1], np.full(len(loop2), s)))


def span_stations():
    """The sections' stations along the span, root to tip: N_FLATS up to the band under the cap's foot, 5 mm
    apart over the band, 1 mm apart over its last FINE_MM, two more half a mm apart, and the rim."""
    return np.r_[np.linspace(S0, S_BAND - BAND_MM, N_FLATS, endpoint=False),
                 S_BAND - np.arange(BAND_MM, FINE_MM, -5.0), S_BAND - np.arange(FINE_MM, 0.5, -1.0),
                 S_BAND - 0.5, S_BAND, S_CAP]


def _knots(bs):
    """The distinct knots of a B-spline surface, (u, v)."""
    return (np.unique([bs.UKnot(i) for i in range(1, bs.NbUKnots() + 1)]),
            np.unique([bs.VKnot(i) for i in range(1, bs.NbVKnots() + 1)]))


class Foot:
    """Closest points of a shape: every face's surface is seeded once (a grid of its parameters, and for a
    B-spline surface every knot and the middle of every span as well, since a face with hundreds of poles
    one way has spans a plain grid steps over); a query is finished by Newton on the surfaces of the faces
    whose seeds lie nearest, and only a foot inside its face counts. A foot further off than FOOT_MM is on
    another branch (the first try from the next seeds of the face, then the point stays as it is)."""

    def __init__(self, shape, grid=200):
        from scipy.spatial import cKDTree
        BRep_Tool = XL._occ("BRep", "BRep_Tool")
        BRepAdaptor_Surface = XL._occ("BRepAdaptor", "BRepAdaptor_Surface")
        GeomAbs_BSplineSurface = XL._occ("GeomAbs", "GeomAbs_BSplineSurface")
        ShapeAnalysis_Surface = XL._occ("ShapeAnalysis", "ShapeAnalysis_Surface")
        FClass2d = XL._occ("BRepTopAdaptor", "BRepTopAdaptor_FClass2d")
        self.out = XL._occ("TopAbs", "TopAbs_OUT")
        self.gp_Pnt, self.gp_Pnt2d = XL._occ("gp", "gp_Pnt", "gp_Pnt2d")
        self.sas, self.cls, self.trees, self.uvs = [], [], [], []
        p = self.gp_Pnt()
        for face in XL._faces_of(shape):
            surf = XL._static(BRep_Tool, "Surface")(face)
            ad = BRepAdaptor_Surface(face, True)
            u0, u1, v0, v1 = (ad.FirstUParameter(), ad.LastUParameter(), ad.FirstVParameter(),
                              ad.LastVParameter())
            us, vs = np.linspace(u0, u1, int(grid)), np.linspace(v0, v1, int(grid))
            if ad.GetType() == GeomAbs_BSplineSurface:
                ku, kv = _knots(ad.BSpline())
                us, vs = self._with_spans(us, ku, u0, u1), self._with_spans(vs, kv, v0, v1)
            pts = np.empty((len(vs), len(us), 3))
            for j, v in enumerate(vs):
                for i, u in enumerate(us):
                    surf.D0(float(u), float(v), p)
                    pts[j, i] = (p.X(), p.Y(), p.Z())
            uu, vv = np.meshgrid(us, vs)
            self.trees.append(cKDTree(pts.reshape(-1, 3)))
            self.uvs.append(np.column_stack((uu.ravel(), vv.ravel())))
            self.sas.append(ShapeAnalysis_Surface(surf))
            self.cls.append(FClass2d(face, 1.0e-7))

    @staticmethod
    def _with_spans(grid, knots, lo, hi):
        """The grid with the knots and the spans' middles inside lo .. hi."""
        more = np.r_[knots, 0.5 * (knots[:-1] + knots[1:])]
        return np.unique(np.r_[grid, more[(more >= lo) & (more <= hi)]])

    def __call__(self, pts, faces=3):
        """The shape's closest points to pts (n, 3) and their distances (nan, the point as given, where no
        foot lies within FOOT_MM)."""
        pts = np.asarray(pts, dtype=float)
        seeds = [t.query(pts) for t in self.trees]
        ds = np.column_stack([q[0] for q in seeds])
        js = np.column_stack([q[1] for q in seeds])
        order = np.argsort(ds, axis=1)[:, :faces]
        out, dist = pts.copy(), np.full(len(pts), np.inf)
        for i, q in enumerate(pts):
            gq = self.gp_Pnt(float(q[0]), float(q[1]), float(q[2]))
            for f in order[i]:
                starts = [js[i, f]]
                for attempt in range(2):
                    for j in starts:
                        u0, v0 = self.uvs[f][j]
                        uv = self.sas[f].NextValueOfUV(self.gp_Pnt2d(float(u0), float(v0)), gq, 1.0e-10)
                        if self.cls[f].Perform(uv) == self.out:
                            continue
                        p = self.sas[f].Value(uv)
                        d = float(np.hypot(np.hypot(p.X() - q[0], p.Y() - q[1]), p.Z() - q[2]))
                        if d < dist[i]:
                            dist[i], out[i] = d, (p.X(), p.Y(), p.Z())
                    if dist[i] <= FOOT_MM or attempt:
                        break
                    starts = self.trees[f].query(q, k=6)[1][1:]    # the next seeds of this face
        far = dist > FOOT_MM
        out[far], dist[far] = pts[far], np.nan
        return out, dist


def on_surface(foot, loop, flat=True):
    """A loop (x, w, s) with its points on the original surface (foot: its Foot, the STP frame): each point's
    closest point; for a section (flat) put back into the cut's plane, s as it was, and the foot taken once
    more from there; the points on w = 0 stay on it. Returns (the loop, the largest move, mm)."""
    q = loop.copy()
    sym = np.abs(loop[:, 1]) < 1.0e-9
    worst, missed = 0.0, 0
    for _ in range(2 if flat else 1):
        p, d = foot(q @ FRAME.T)
        ok = np.isfinite(d)
        worst = max(worst, float(d[ok].max()))
        missed = max(missed, int((~ok).sum()))
        new = p @ FRAME
        if flat:
            new[:, 2] = loop[:, 2]
        new[sym, 1] = 0.0
        q[ok] = new[ok]
    if missed:
        print(f"  {missed} points of the loop at s {loop[0, 2]:.2f} found no foot nearby and stay as cut")
    return q, worst


def chord_stations(x_te, x_le):
    """The rings' chord stations, LE first: RING_H0 apart for the first RING_N0 from each end (the cap rolls
    over the LE and TE lines within a few mm there), the step then growing by RING_GROW up to RING_HMAX; the
    two halves meet on mid-chord."""
    half = 0.5 * (x_le - x_te)
    d, h, x = [], RING_H0, 0.0
    while x < half:
        x += h
        d.append(x)
        if len(d) >= RING_N0:
            h = min(h * RING_GROW, RING_HMAX)
    d = np.array(d) * half / d[-1]                         # the last one on mid-chord
    return np.r_[x_le - d, (x_te + d[:-1])[::-1]]


def cap_halves(q):
    """The cap's cross profile out of a closed cut q (w, s), the duplicate end dropped: its two halves, each
    from the base (the cut's crossing of s = S_CAP) to the crest (its crossing of w = 0), the w < 0 half
    first."""
    n = len(q)
    top = int(np.argmax(q[:, 1]))
    runs = []
    for step in (1, -1):                                   # from the top down each side to the foot
        pts, i = [q[top]], top
        for _ in range(n):
            j = (i + step) % n
            if q[j, 1] < S_CAP:
                t = (q[i, 1] - S_CAP) / (q[i, 1] - q[j, 1])
                pts.append(q[i] + t * (q[j] - q[i]))
                break
            pts.append(q[j])
            i = j
        else:
            raise RuntimeError("a cut across the cap does not come down to its foot")
        runs.append(np.array(pts))
    prof = np.vstack((runs[0][::-1], runs[1][1:]))         # base, over the top, base
    if prof[0, 0] > prof[-1, 0]:
        prof = prof[::-1]                                  # from the w < 0 base
    k = int(np.where((prof[:-1, 0] <= 0.0) & (prof[1:, 0] > 0.0))[0][0])
    t = -prof[k, 0] / (prof[k + 1, 0] - prof[k, 0])
    crest = prof[k] + t * (prof[k + 1] - prof[k])
    return np.vstack((prof[:k + 1], crest)), np.vstack((prof[k + 1:][::-1], crest))


def cap_rings(plane, corners):
    """The cap resampled into rings: cuts normal to x at the chord stations give the cross profile from the
    base (s = S_CAP) to the crest (w = 0); ring f holds the points at the fraction f of each half profile's
    arc, symmetrized onto +-w, and runs through the two base corners. Each side of a ring is the cubic
    spline through its stations (the two sides leave a base corner in a V, the more upright the higher the
    ring), resampled by arc as the flat sections are. Returns the rings, base to crest, in the loft's layout
    and frame (x, w, s)."""
    from scipy.interpolate import CubicSpline
    x_te, x_le = corners
    xs = chord_stations(x_te, x_le)
    halves = []                                            # per station: (w < 0 half, w > 0 half), base to crest
    for x0 in xs:
        p3 = plane_cut(plane, (x0, 0.0, 0.0), (1.0, 0.0, 0.0), 400)
        q = np.column_stack((p3 @ E_THICK, p3 @ E_SPAN))
        if np.linalg.norm(q[0] - q[-1]) < 1.0e-6:
            q = q[:-1]
        halves.append(cap_halves(q))
    le = np.array([x_le, 0.0, S_CAP])
    te = np.array([x_te, 0.0, S_CAP])
    u = cosine((N_PTS + 1) // 2)                           # by arc as flat_loop resamples, so that an index
    mirror = np.array([1.0, -1.0, 1.0])                    # means the same arc fraction in every row
    fracs = RING_F1 + (RING_TOP - RING_F1) * (np.arange(N_RINGS) / (N_RINGS - 1.0)) ** RING_POWER
    rings = []
    for f in fracs:
        p1 = np.array([_resample(lo, [f])[0] for lo, hi in halves])    # (w, s), the w < 0 side, LE to TE
        p2 = np.array([_resample(hi, [f])[0] for lo, hi in halves])
        a = 0.5 * (p2[:, 0] - p1[:, 0])                    # the cap is symmetric: the halves averaged onto
        sm = 0.5 * (p1[:, 1] + p2[:, 1])                   # +-w, or the seams' sides drift apart
        side = np.vstack((le, np.column_stack((xs, -a, sm)), te))      # the w < 0 side, LE to TE
        t = _arc(side)
        cs = CubicSpline(t, side, axis=0)
        tt = np.linspace(0.0, t[-1], 20001)
        sa = _arc(cs(tt))
        r1 = cs(np.interp(u * sa[-1], sa, tt))
        r1[0], r1[-1] = le, te
        r1[:, 1] = -np.abs(r1[:, 1])
        r1[0, 1] = r1[-1, 1] = 0.0
        rings.append(np.vstack((r1, (r1 * mirror)[::-1][1:])))
    return rings


def root_fit(plane, foot):
    """The buried root's sections (the chop, below S_FULL) as the straight taper continued: a linear fit in
    s, per loop point, over the full sections at FIT_S; returns (a, b), the loop at s being a + b s."""
    loops = np.array([on_surface(foot, flat_loop(plane, s))[0] for s in FIT_S])
    des = np.column_stack((np.ones(len(FIT_S)), FIT_S))
    coef, *_ = np.linalg.lstsq(des, loops.reshape(len(FIT_S), -1), rcond=None)
    resid = np.abs((des @ coef).reshape(loops.shape) - loops).max()
    print(f"the root taper's fit over s {FIT_S[0]:.0f} to {FIT_S[-1]:.0f} mm: within {resid:.3f} mm")
    return coef[0].reshape(-1, 3), coef[1].reshape(-1, 3)


def synth_loop(a, b, s):
    lp = a + b * s
    lp[:, 2] = s
    return lp


def cut_curves(shape, origin, normal, h=0.25):
    """The cut of a shape by a plane as polylines, one per section edge (a boolean part's cut can be several
    separate loops, so they are not chained), their points about h mm apart: a coarse polyline cuts the
    corner of a tightly curved stretch, which would read as a distance. The section algorithm now and then
    drops an edge; a cut that comes back open is taken again a hundredth of a mm further on. Returns (the
    polylines, whether the cut is closed)."""
    gp_Pln, gp_Pnt, gp_Dir = XL._occ("gp", "gp_Pln", "gp_Pnt", "gp_Dir")
    Section = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Section")
    origin, normal = np.asarray(origin, dtype=float), np.asarray(normal, dtype=float)
    curves = []
    for k in range(4):
        algo = Section(shape, gp_Pln(gp_Pnt(*map(float, origin + 0.01 * k * normal)), gp_Dir(*map(float, normal))))
        algo.Build()
        curves = []
        for e in S._explore(algo.Shape(), "EDGE"):
            n = int(min(max(_arc(S._polyline(e, 60))[-1] / h, 60.0), 40000.0))
            curves.append(S._polyline(e, n))
        ends = np.array([c[0] for c in curves] + [c[-1] for c in curves])
        if len(curves) and all(np.count_nonzero(np.linalg.norm(ends - e, axis=1) < 1.0e-3) >= 2 for e in ends):
            return curves, True
    return curves, False


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
    trace carries the chop's corner blend); the hull reaches neither. A cut that stayed open (an edge
    missing) is marked: its distance is not to be trusted."""
    worst = 0.0
    for s in CHECK_S:
        (a, ok_a), (b, ok_b) = cut_curves(plane, s * E_SPAN, E_SPAN), cut_curves(loft, s * E_SPAN, E_SPAN)
        qa = [np.column_stack((p[:, 0], p @ E_THICK)) for p in a]
        qb = [np.column_stack((p[:, 0], p @ E_THICK)) for p in b]
        pa, pb = np.vstack(qa), np.vstack(qb)
        note = ""
        if s < S_FULL:                                     # the original is chopped at this height
            bound = pa[:, 0].max() - 30.0
            pa, pb = pa[pa[:, 0] < bound], pb[pb[:, 0] < bound]
            note = " (the airfoil clear of the chop)"
        if not (ok_a and ok_b):
            note += " (the cut came back open)"
        d = max(nearest_distance(pa, qb), nearest_distance(pb, qa))
        worst = max(worst, d)
        print(f"  cut s = {s:6.0f} mm: largest distance {d:6.3f} mm{note}")
    for x0 in CHECK_X:
        (a, ok_a), (b, ok_b) = (cut_curves(plane, (x0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                                cut_curves(loft, (x0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        qa = [np.column_stack((p @ E_THICK, p @ E_SPAN)) for p in a]
        qb = [np.column_stack((p @ E_THICK, p @ E_SPAN)) for p in b]
        pa = np.vstack(qa)
        pb = np.vstack(qb)
        pa = pa[pa[:, 1] > S_CAP - 400.0]                  # points of the tip region; whole curves
        pb = pb[pb[:, 1] > S_CAP - 400.0]
        d = max(nearest_distance(pa, qb), nearest_distance(pb, qa))
        worst = max(worst, d)
        print(f"  cap cut x = {x0:6.0f} mm: largest distance {d:6.3f} mm (the tip region)"
              + ("" if ok_a and ok_b else " (the cut came back open)"))
    return worst


def patch_points(face):
    """Points of a face's B-spline surface, one in the middle of every knot span (the tip patch's spans
    crowd where the tip turns; a plain grid of its parameters would pass most of them by): (n, 3), or None
    for a face that is not a B-spline (the flat root)."""
    BRepAdaptor_Surface = XL._occ("BRepAdaptor", "BRepAdaptor_Surface")
    GeomAbs_BSplineSurface = XL._occ("GeomAbs", "GeomAbs_BSplineSurface")
    gp_Pnt = XL._occ("gp", "gp_Pnt")
    ad = BRepAdaptor_Surface(face, True)
    if ad.GetType() != GeomAbs_BSplineSurface:
        return None
    bs = ad.BSpline()
    ku, kv = _knots(bs)
    us, vs = 0.5 * (ku[:-1] + ku[1:]), 0.5 * (kv[:-1] + kv[1:])
    p = gp_Pnt()
    out = np.empty((len(vs), len(us), 3))
    for j, v in enumerate(vs):
        for i, u in enumerate(us):
            bs.D0(float(u), float(v), p)
            out[j, i] = (p.X(), p.Y(), p.Z())
    return out.reshape(-1, 3)


def check_patches(foot, shape):
    """The loft's patches against the original point by point: per patch the largest and the rms distance of
    its points (one in every knot span) from the original, mm. shape: the loft in its own frame (x, w, s);
    only points above S_FULL count (the root below is synthetic). Returns the largest distance."""
    grids = [g for g in (patch_points(f) for f in XL._faces_of(shape)) if g is not None]
    if len(grids) != 5:
        raise RuntimeError(f"the loft has {len(grids)} B-spline faces, expected 5")
    tip = int(np.argmax([g[:, 2].mean() for g in grids]))
    rest = [k for k in range(5) if k != tip]
    le = max(rest, key=lambda k: grids[k][:, 0].mean())
    te = min(rest, key=lambda k: grids[k][:, 0].mean())
    names = {tip: "the tip patch", le: "the LE strip", te: "the TE strip"}
    for k in rest:
        if k not in names:
            names[k] = f"the central panel, w {'<' if grids[k][:, 1].mean() < 0.0 else '>'} 0"
    worst = 0.0
    for k in (le, *[j for j in rest if j not in (le, te)], te, tip):
        g = grids[k][grids[k][:, 2] >= S_FULL]
        d = foot(g @ FRAME.T)[1]
        ok = np.isfinite(d)
        worst = max(worst, float(d[ok].max()))
        print(f"  {names[k]}: {len(g)} points, within {d[ok].max():.4f} mm of the original "
              f"(rms {np.sqrt((d[ok] ** 2).mean()):.4f})"
              + ("" if ok.all() else f"; {int((~ok).sum())} points more than {FOOT_MM} mm off"))
    return worst


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "outputs")
    t0 = time.time()
    plane = S._explore(S.read_cad(os.path.join(out, "bb2_sternplane.step")), "SOLID")[0]
    hull_hub = S._explore(S.read_cad(os.path.join(out, "bb2_hull_hub.step")), "SOLID")[0]

    # the loops (sections from S0 to the rim, the chopped root synthesized, the cap as rings), every point
    # on the original surface, lofted in the frame (x, w, s)
    foot = Foot(plane)
    base, moved = on_surface(foot, flat_loop(plane, S_CAP))
    corners = float(base[:, 0].min()), float(base[:, 0].max())
    a, b = root_fit(plane, foot)
    loops = []
    for s in span_stations()[:-1]:
        if s < S_FULL:
            loops.append(synth_loop(a, b, s))
            continue
        lp, d = on_surface(foot, flat_loop(plane, s))
        moved = max(moved, d)
        loops.append(lp)
    loops.append(base)
    n_flat, lifted = len(loops), 0.0
    i_te = (N_PTS + 1) // 2 - 1
    for ring in cap_rings(plane, corners):
        q, d = on_surface(foot, ring, flat=False)
        q[0], q[i_te], q[-1] = ring[0], ring[i_te], ring[-1]               # the rim's LE and TE points
        lifted = max(lifted, d)
        loops.append(q)
    print(f"{n_flat} sections and {len(loops) - n_flat} rings in {time.time() - t0:.0f} s, their points put on "
          f"the original surface (the sections' by up to {1000.0 * moved:.1f} um, the rings' by up to "
          f"{1000.0 * lifted:.1f} um)")
    res = XL.loft_loops(loops, n_poles=N_POLES, v_poles=V_POLES)
    if res.shape is None or not res.valid or res.free_edges:
        raise RuntimeError(f"the loft failed: {res}")
    loft = VC.transformed(res.shape, FRAME, np.zeros(3))
    n_s, ok, vol = VC.solid_report(loft)
    if n_s != 1 or not ok:
        raise RuntimeError(f"the turned loft: {n_s} solids, valid {ok}")
    print(f"lofted in {time.time() - t0:.0f} s: {res.faces} faces, the sections' curves within "
          f"{res.fit_error:.4f} mm, the loops and the cap within {res.v_error:.4f} mm of the patches, "
          f"{vol:.6e} mm^3 (with the synthetic root; the chopped original {VC.volume(plane):.6e})")
    for note in res.notes:
        print(f"  {note}")
    print("against the original, cut by cut:")
    worst = check_against(plane, loft)
    print(f"  largest over the cuts: {worst:.3f} mm")
    print("against the original, patch by patch:")
    worst = check_patches(foot, res.shape)
    print(f"  largest over the patches: {worst:.4f} mm")
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
