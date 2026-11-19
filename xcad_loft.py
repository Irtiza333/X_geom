"""
xcad_loft.py

The CAD solid through the section loops of an XCAD point file, or through
loops in memory, for any geometry whose loops start at the LE, run along one
side to the TE at index (n + 1) // 2 - 1 and come back to the LE (the layout
of Rudder_geom_extraction.write_xcad and rudder_modify.write_modified_xcad).

How the solid is built
----------------------
Five patches and the flat root, six faces (the half body too): a
leading-edge strip and a trailing-edge strip wrapping their edges, the two
central panels between them, and the tip patch, laid out as the blade's five
surfaces are (DRDC TM 2013-178; user's sketch, 17 Nov 2026: the layout before
ran the four patches along the span up to the tip's edge and gave the tip
patch the cap alone).

- The tip patch's four corners sit on the strip lines (the seams between the
  strips and the central panels) under the tip. A strip's top edge is the cut
  of the body by the plane through its two corners and the top of its edge
  line, where the cap begins: a straight diagonal in the side view, from the
  strip line's top up to the top of the LE or TE. A central panel's top edge
  is the straight line between its two corners. The tip patch covers the
  sides above those edges and the cap. Each pair of corners sits under the
  top of its edge line by tip_drop times its distance from that top along
  the chord (the diagonals' slope; 1.5 gave the smoothest patches on BB2's
  stern plane among 0.5 to 2).
- The strips are as wide as their edges' rounding (strip_frac None). The LE
  strip reaches 1.5 times the arc from the LE to where the section's
  curvature has fallen to a quarter of its peak. The TE strip reaches the
  arc from the TE to where the curvature has fallen to a tenth of its peak:
  the strip is the TE's round, its seams lie where the round meets the
  flanks, and no patch has that jump of curvature inside it (a jump inside a
  patch leaves a ripple along it). The widths are the sections' median.
- The walls. The horizontal loops are fitted over the height once, point
  index by point index, as the patches' rows of poles are fitted along the
  span (_wall_loops). The rows, the lines up the body and the tip patch's
  sides all come off that one smooth surface; scatter from loop to loop (a
  section algorithm's steps, a crease of the CAD the loops were cut from)
  is taken out there.
- The rows. Every horizontal loop is a row of the four patches. In a band
  under the top edges the rows are turned onto them, each point moving up
  its own line through the loops (_tip_rows), so the rows turn without a
  jump while the surface stays the one the loops describe.
- Each row is sampled from the smooth closed curve through its points (a
  cubic spline, not their polygon) and split into four arcs at shared arc
  stations. Each arc is fitted with held ends, so neighbouring patches share
  their seam poles row by row, and each family is fitted along the span on
  common parameters and knots: identical boundary data, parameters and
  knots make the shared boundary curves identical, so the faces sew
  watertight.
- The rim is where the walls end and the cap begins: the first loop that is
  not horizontal (a tilted tip section), or the last horizontal loop when
  the loops above it run through its LE and TE points (a cap's rings on a
  horizontal foot). The tip turns there from the straight wall into the
  cap's rounding: its curvature jumps.
- The tip patch. Per chord station one wrap runs from the A side's top edge
  up the wall to the rim, over the cap and down the B side: under the rim
  the wall's own line, over the cap the cubic spline through the cap loops'
  points, leaving both rims in the wall's tangent plane (no crease on the
  rim). The wraps' grid, with the bicubic through it block by block (the
  walls and the cap meet on the rims), is the surface the patch is fitted
  to (_TipTruth). Its mesh is mapped onto the patch's unit square with the
  four top edges on the square's sides (mean value coordinates), and every
  line u = constant of the map is slid along itself until the rim lies on
  two lines v = constant, knots of the patch taken twice (_rim_lines): the
  patch is C1 there and free to change its curvature, as the tip does. A
  smooth closed rim cannot lie on a patch's lines all round; it leaves them
  within a short stretch of the tops of the LE and TE lines, where the
  knots are closest. The patch is the least-squares tensor fit on those
  knots, refined where it misses the tip (_tip_fit), with its four edges
  the neighbours' top rows (the pole rows are shared verbatim) and its
  first rows of poles inside the edges turned into the neighbours' tangent
  planes: the normal does not turn across the tip patch's edges.
- The tops of the LE and TE lines. Every cap loop meets the rim there (a
  pinch), the strips' top rows run through them, and all the cap's lines
  leave a strip's top row within RIM_SHARE of its width beside them: the
  patch is at its tightest there, and its lines doubled back within a tenth
  of a mm of the wind-tunnel rudder's two tops until four things were seen
  to. The wraps' stations close in on the two tops (APEX_STEPS), so the map
  has wraps where the lines fan out. Between a pinch and the loops' first
  points beside it, about which the loops say nothing, the wraps are the
  first whole wrap scaled down into the pinch (_into_pinch; each loop's own
  cubic lets the upper loops cross the lower ones there), and the surface
  the patch is fitted to runs smoothly into the pinch itself. The mesh's
  boundary takes its parameters from the edges as fitted (_on_edges), not
  from the sections' arcs, which place a point some um further along. And
  no line leaves an edge at less than EDGE_ANGLE (_into_plane). The result's
  tip_angle is the smallest angle between the patch's two families of
  lines, 0 when they cross somewhere (a fold; the notes say so): 14 deg on
  BB2's stern plane, 15 on the wind-tunnel rudder.
- A rounding along the tip's edge. The wind-tunnel rudder's cap runs from
  the rim up a straight wall (0.4 to 1.9 mm, longer towards the TE), round
  its 0.7 mm edge radius and over the crown: two jumps of curvature on each
  side, neither on a line of the patch (the rim is, and it is not a jump).
  A patch fitted through them rings, 1 deg either side of the round in a
  band 1 mm wide. Where most wraps jump in curvature between their ends,
  every wrap's cap part is blended: averaged over CAP_BLEND (0.5) of the
  cap's tightest radius (0.29 mm on that rudder), twice over, so the
  curvature runs up and down without a kink (_curvature_jumps, _blended).
  The round moves in by about CAP_BLEND^2 / 12 of that radius, 0.013 mm
  there (its cap model itself lies 0.1 mm from the CAD), and the ringing is
  gone: the patch's normals within 0.4 deg of its data over 99 % of the
  band (0.9 before), its curvature across the round one smooth hump. The
  loops sample that round's end at 0.14 to 0.44 mm, no finer than the
  blend. A cap without jumps (BB2's, a dome) is left as it is; the note and
  the result's tip_blend say when a cap was blended, over what stretch and
  by how much.
- A step from one horizontal loop to the next (sections from another
  source under the extracted ones: a rudder's Bezier model alone lies a
  tenth of a mm off its extracted loops, which is why rudder_modify writes
  the extracted sections plus the model's change since 12 Nov 2026; a TE
  radius other than the original's still steps) cannot be followed without
  a ripple round the body. It is found where loops miss a fit of their
  points along the span, and taken out of the loops over a band below it
  and one above, tall enough for the surface to tilt by 0.1 deg at most
  (_ease_flats; ease names the side that gives way, both by default), so
  the patches cross it smoothly. The step and the bands are reported in
  the notes and in the result's steps; the distances reported are those of
  the loops as given. A step right under a tilted tip section (sections
  that step up to the top one) has no loops above it to ease into and
  stays a blend.
- Two lofts (loft_split; 19 Nov 2026, the user: a rudder modified below H
  whose curves leave the original there without its slope). One loft fits
  its walls and patches over the whole height, so a turn of the surface at
  one section (a crease along it) is smoothed over the sections about it,
  taken for steps and rippled about. loft_split lofts the loops up to that
  section and those from it on, each on its own, both through it exactly
  and with the same strips (the whole stack's), so they end on the same four
  curves there and the seams run on across it; the four patches below, the
  four and the tip patch above and the flat root are sewn into one solid,
  the two flat faces on the joint left out: nine patches and the root. The
  wind-tunnel rudder with its curves free at H, C1 off the tangent: one loft
  0.19 mm off the loops about H, with two steps it took the turn for; two
  lofts 0.0001 mm.
- half=True cuts the sewn full body on the plane of symmetry and keeps the
  LE-to-TE side: the same surfaces, trimmed, closed by one planar face. The
  loops must be symmetric about the plane within sym_tol.
- A wing or fin whose loops are all horizontal gets no tip patch: the four
  wrap patches run the whole span and the flat top is closed by a planar
  face, six faces again.
- The cap is only as fine as its loops: a rounded tip edge needs loops close
  together across it (Rudder_geom_extraction writes 40 cap loops for that).
  The loops' points have to lie on the body: a section algorithm's cut is a
  polygon through its own steps, up to 0.01 mm inside a nose (BB2's loops
  are put back on the CAD point by point, bb2_sternplane_loft.py). The cap's
  loops have to run into the rim's LE and TE points, as both of those do: a
  loop that ends over such a point leaves a blade above it, which the tip
  patch can only follow folded. A cap whose ends are far blunter than these
  two (a test wing's, rising as the 0.3rd power of the distance from its
  ends) still folds the patch; the result says so.

Coordinates: the XCAD frame (x chordwise, y thickness, z the height) in mm;
point files in metres are scaled by 1000, as X_CAD.py does.

    python xcad_loft.py outputs/modified/baseline_sections_xcad_m.dat   (writes the .step next to it)
    python xcad_loft.py <xcad file> --view        and shows it in the pythonocc viewer
    python xcad_loft.py --view-step <file.step>   shows a STEP file

save_cad writes a built STEP file to a file of the user's choice, as STEP or
IGES (the design tool's Save CAD).

Needs pythonocc-core (OCC.Core, conda-forge) or cadquery-ocp (OCP).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

if os.name == "nt" and hasattr(os, "add_dll_directory"):          # conda's OCCT DLLs (as X_CAD.py does)
    _lib = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(_lib):
        os.add_dll_directory(_lib)

_PKG = None
for _cand in ("OCC.Core", "OCP"):
    try:
        importlib.import_module(_cand + ".gp")
        if not hasattr(importlib.import_module(_cand + ".TColgp"), "TColgp_Array1OfPnt"):
            continue                                            # OCCT 8 bindings: arrays not exposed this way
        _PKG = _cand
        break
    except ImportError:
        continue


def occ_available():
    """True when pythonocc-core or cadquery-ocp can be imported."""
    return _PKG is not None


def _occ(module, *names):
    if _PKG is None:
        raise ImportError("no OCCT binding: install pythonocc-core (conda install -c conda-forge "
                          "pythonocc-core) or cadquery-ocp")
    mod = importlib.import_module(f"{_PKG}.{module}")
    objs = tuple(getattr(mod, n) for n in names)
    return objs[0] if len(objs) == 1 else objs


def _static(owner, name):
    """OCP spells static methods Name_s, pythonocc Name."""
    for cand in (name, name + "_s"):
        if hasattr(owner, cand):
            return getattr(owner, cand)
    raise AttributeError(f"{owner!r} has no static method {name!r}")


def _topods(kind):
    mod = importlib.import_module(f"{_PKG}.TopoDS")
    return _static(getattr(mod, "topods", None) or getattr(mod, "TopoDS"), kind)


XCAD_UNITS = {"m": 1.0e-3, "mm": 1.0}


def read_loops(path, units="m"):
    """The loops of an XCAD point file as arrays in mm, XCAD frame."""
    loops, cur = [], None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#"):
                cur = []
                loops.append(cur)
            elif s:
                cur.append([float(v) for v in s.split()])
    return [np.array(lp) / XCAD_UNITS[units] for lp in loops]


# --------------------------------------------------------------------------
# Section curves on common knots
# --------------------------------------------------------------------------

def _basis_at(u, n_poles, degree=3, spaced=False):
    """Least-squares B-spline basis at the parameters u (rising, 0 to 1):
    (A (len(u), n_poles), knots). Interior knots uniform, or placed from the
    parameters (spaced=True, Piegl and Tiller's averaging), so that every
    pole keeps samples in its support when the parameters bunch up."""
    from scipy.interpolate import BSpline
    n_poles = max(int(n_poles), degree + 1)
    u = np.asarray(u, dtype=float)
    if spaced:
        uu = np.interp(np.linspace(0.0, len(u) - 1.0, n_poles), np.arange(len(u)), u)
        inner = [uu[j:j + degree].mean() for j in range(1, n_poles - degree)]     # de Boor averaging
        knots = np.r_[[0.0] * (degree + 1), inner, [1.0] * (degree + 1)]
    else:
        knots = np.r_[[0.0] * degree, np.linspace(0.0, 1.0, n_poles - degree + 1), [1.0] * degree]
    try:
        a = BSpline.design_matrix(u, knots, degree).toarray()
    except AttributeError:                                         # scipy < 1.8
        a = np.column_stack([BSpline(knots, np.eye(n_poles)[j], degree)(u) for j in range(n_poles)])
    return a, knots


def _basis(m, n_poles, degree=3):
    """_basis_at with the parameter evenly spaced by point index."""
    return _basis_at(np.linspace(0.0, 1.0, m), n_poles, degree)


def _fit_poles(pts, a):
    """Poles of the side through pts, the two end points held."""
    poles = np.empty((a.shape[1], pts.shape[1]))
    poles[0], poles[-1] = pts[0], pts[-1]
    rhs = pts - np.outer(a[:, 0], poles[0]) - np.outer(a[:, -1], poles[-1])
    poles[1:-1] = np.linalg.lstsq(a[:, 1:-1], rhs, rcond=None)[0]
    return poles


def _bspline_edge(poles, knots, degree):
    gp_Pnt = _occ("gp", "gp_Pnt")
    TColgp_Array1OfPnt = _occ("TColgp", "TColgp_Array1OfPnt")
    TColStd_Array1OfReal, TColStd_Array1OfInteger = _occ("TColStd", "TColStd_Array1OfReal",
                                                         "TColStd_Array1OfInteger")
    Geom_BSplineCurve = _occ("Geom", "Geom_BSplineCurve")
    BRepBuilderAPI_MakeEdge = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeEdge")
    arr = TColgp_Array1OfPnt(1, len(poles))
    for i, p in enumerate(poles):
        arr.SetValue(i + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    uk, mult = np.unique(knots, return_counts=True)
    k_arr, m_arr = TColStd_Array1OfReal(1, len(uk)), TColStd_Array1OfInteger(1, len(uk))
    for i, (kv, mv) in enumerate(zip(uk, mult)):
        k_arr.SetValue(i + 1, float(kv))
        m_arr.SetValue(i + 1, int(mv))
    return BRepBuilderAPI_MakeEdge(Geom_BSplineCurve(arr, k_arr, m_arr, degree)).Edge()


class _SectionFitter:
    """Wires for loops of one point layout, both sides on common knots."""

    def __init__(self, n_points, n_poles, degree=3):
        self.i_te = (n_points + 1) // 2 - 1
        self.degree = degree
        self.a_up, self.k_up = _basis(self.i_te + 1, n_poles, degree)
        self.a_lo, self.k_lo = _basis(n_points - self.i_te, n_poles, degree)

    def sides(self, loop):
        """Poles of the two sides (LE -> TE, then TE -> LE) and the largest
        distance of the loop's points from them, mm."""
        up, lo = loop[:self.i_te + 1], loop[self.i_te:]
        p_up, p_lo = _fit_poles(up, self.a_up), _fit_poles(lo, self.a_lo)
        err = max(np.abs(self.a_up @ p_up - up).max(), np.abs(self.a_lo @ p_lo - lo).max())
        return p_up, p_lo, float(err)

    def wire(self, loop):
        """(wire, largest distance of the loop's points from it, mm)."""
        BRepBuilderAPI_MakeWire = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeWire")
        p_up, p_lo, err = self.sides(loop)
        w = BRepBuilderAPI_MakeWire()
        w.Add(_bspline_edge(p_up, self.k_up, self.degree))
        w.Add(_bspline_edge(p_lo, self.k_lo, self.degree))
        return w.Wire(), err

    def closing_face(self, loop):
        """The face across a loop that is not flat: the ruled surface between
        its two sides, straight lines from one side to the other (the knots
        are symmetric, so the lower side reversed is the same curve)."""
        BRepBuilderAPI_MakeWire = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeWire")
        p_up, p_lo, _ = self.sides(loop)
        e_up = _bspline_edge(p_up, self.k_up, self.degree)
        e_lo = _bspline_edge(p_lo[::-1], self.k_lo, self.degree)
        return _thru([BRepBuilderAPI_MakeWire(e_up).Wire(), BRepBuilderAPI_MakeWire(e_lo).Wire()], False, True)


# --------------------------------------------------------------------------
# The loft
# --------------------------------------------------------------------------

@dataclass
class LoftResult:
    shape: object                  # the solid (TopoDS_Shape), None when it failed
    valid: bool                    # BRepCheck_Analyzer
    volume: float                  # mm^3
    area: float                    # mm^2
    free_edges: int                # edges the sewing left open (0 for a closed solid)
    fit_error: float               # largest distance of the rows' points from their four arcs' curves, mm
    n_smooth: int                  # leading loops at one height each (the horizontal sections)
    n_ruled: int                   # loops after them (the tilted tip section and the cap loops)
    seconds: float
    notes: list = field(default_factory=list)
    faces: int = 0                 # faces of the solid
    v_error: float = 0.0           # largest distance of the loops from the four patches along the span and
                                   # of the cap from the tip patch, mm (along the patches' normals)
    half: bool = False             # the half body on the LE-to-TE side
    steps: list = field(default_factory=list)      # steps between horizontal loops eased out of them: (the
                                   # loop under the step counted from 1, the step mm, how far it is eased
                                   # below and above mm, the shares of it taken below and above)
    strips: tuple = (0.0, 0.0)     # how far the LE strip and the TE strip reach to each side of their edge, mm
    te_round: bool = False         # the TE strip was set by the TE's rounding
    drops: tuple = (0.0, 0.0)      # how far the tip patch's corners sit under the tops of the LE and TE lines, mm
    tip_error: float = 0.0         # largest distance of the tip from the tip patch, mm
    tip_poles: tuple = (0, 0)      # the tip patch's poles along the chord and across
    tip_angle: float = 90.0        # the smallest angle between the tip patch's two families of lines, deg; 0
                                   # when they cross somewhere (the patch folds on itself)
    corners: dict = None           # the tip patch's four corners (ur_a, ur_b at the LE strip; ul_a, ul_b)
    tip_blend: tuple = (0.0, 0.0)  # (the stretch the cap's wraps were blended over, how far that moved the
                                   # cap), mm; (0, 0) when the cap has no rounding to blend (CAP_BLEND)
    joint: int = 0                 # loft_split: the loop its two lofts meet at, counted from 1; 0: one loft


def _thru(wires, solid, ruled, max_degree=None):
    BRepOffsetAPI_ThruSections = _occ("BRepOffsetAPI", "BRepOffsetAPI_ThruSections")
    ts = BRepOffsetAPI_ThruSections(solid, ruled, 1.0e-6)
    ts.CheckCompatibility(False)
    if max_degree is not None:
        ts.SetMaxDegree(int(max_degree))
    for w in wires:
        ts.AddWire(w)
    ts.Build()
    if not ts.IsDone():
        raise RuntimeError("BRepOffsetAPI_ThruSections failed")
    return ts.Shape()


def _planar_face(wire):
    BRepBuilderAPI_MakeFace = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace")
    mk = BRepBuilderAPI_MakeFace(wire, True)
    if not mk.IsDone():
        raise RuntimeError("a flat end loop did not give a planar face")
    return mk.Face()


def _faces_of(shape):
    """The faces of a shape."""
    TopExp_Explorer = _occ("TopExp", "TopExp_Explorer")
    TopAbs_FACE = _occ("TopAbs", "TopAbs_FACE")
    as_face = _topods("Face")
    out, ex = [], TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        out.append(as_face(ex.Current()))
        ex.Next()
    return out


def count_faces(shape):
    """Number of faces of a shape."""
    return len(_faces_of(shape))


def mass_properties(shape, eps=1.0e-7):
    """(volume mm^3, area mm^2) of a shape, by adaptive integration to the
    relative error eps (the default Gauss rule is 0.4 % off on these
    many-span B-spline faces)."""
    GProp_GProps = _occ("GProp", "GProp_GProps")
    mod = importlib.import_module(f"{_PKG}.BRepGProp")
    owner = getattr(mod, "brepgprop", None) or getattr(mod, "BRepGProp")
    out = []
    for kind in ("VolumeProperties", "SurfaceProperties"):
        props = GProp_GProps()
        fn = _static(owner, kind)
        try:
            fn(shape, props, float(eps))
        except TypeError:                                   # a binding without the eps overload
            fn(shape, props)
        out.append(props.Mass())
    return abs(out[0]), out[1]


def _unique_mults(knots):
    uk, mult = np.unique(np.asarray(knots, dtype=float), return_counts=True)
    return uk, mult


def _real_array(vals):
    TColStd_Array1OfReal = _occ("TColStd", "TColStd_Array1OfReal")
    a = TColStd_Array1OfReal(1, len(vals))
    for i, v in enumerate(vals, start=1):
        a.SetValue(i, float(v))
    return a


def _int_array(vals):
    TColStd_Array1OfInteger = _occ("TColStd", "TColStd_Array1OfInteger")
    a = TColStd_Array1OfInteger(1, len(vals))
    for i, v in enumerate(vals, start=1):
        a.SetValue(i, int(v))
    return a


EASE_DEG = 0.1                             # a step between loops is eased out until it tilts the surface by
                                           # no more than this, deg (reflection lines stay straight)


def _span_solve(fams, params, n_v, deg, tol, passes=4):
    """The least-squares fit of the families' rows along the span on common knots, the end rows held:
    (grids, knots, each row's largest distance from its grid). It starts with n_v rows of poles; where a row
    lies further than tol from its grid, the knot spans about it are halved and the fit repeated (a span
    only while both halves keep two rows); where rows still miss after the passes by more than 0.3 of their
    first miss, the knots added about them are taken out again."""
    m = len(params)
    _, base = _basis_at(params, n_v, deg, spaced=True)

    def fit(knots):
        a = _design(params, knots, deg)
        grids, res = {}, np.zeros(m)
        for name, rows in fams.items():
            flat = rows.reshape(m, -1)
            g = _fit_poles(flat, a)
            grids[name] = g.reshape(a.shape[1], rows.shape[1], 3)
            res = np.maximum(res, np.abs(a @ g - flat).max(axis=1))
        return grids, res

    knots = base
    grids, res = fit(knots)
    res0 = res
    if deg == 3:
        for _ in range(passes):
            bad = np.where(res > tol)[0]
            if not len(bad):
                break
            uk = np.unique(knots)
            new = []
            for j in np.unique(np.clip(np.searchsorted(uk, params[bad], "right") - 1, 0, len(uk) - 2)):
                mid = 0.5 * (uk[j] + uk[j + 1])
                if (np.count_nonzero((params > uk[j]) & (params < mid)) >= 2
                        and np.count_nonzero((params > mid) & (params < uk[j + 1])) >= 2):
                    new.append(mid)
            if not new:
                break
            knots = np.sort(np.r_[knots, new])
            grids, res = fit(knots)
        bad = np.where(res > tol)[0]
        if len(bad) and len(knots) > len(base):            # steps the knots cannot follow: back to the blend
            ub = np.unique(base)
            reach = 1.5 * np.diff(ub)[np.clip(np.searchsorted(ub, params[bad], "right") - 1, 0, len(ub) - 2)]
            added = np.setdiff1d(knots, base)
            near = np.zeros(len(added), dtype=bool)
            for pj, rj in zip(params[bad], reach):
                about = np.abs(params - pj) <= rj
                if res[about].max() > 0.3 * res0[about].max():         # the knots there did not pay
                    near |= np.abs(added - pj) <= rj
            if near.any():
                knots = np.sort(np.r_[base, added[~near]])
                grids, res = fit(knots)
    return grids, knots, res


def _span_fit(fams, v_poles, params, tol, degree=3, passes=4):
    """The pole grids of several families of pole rows fitted along the span by least squares on one set of
    parameters and knots (so curves the families share come out identical), the end rows held exactly:
    ({name: grid (n_v, n_poles, 3)}, v knots, v degree, the largest distance of a row from its grid, mm).
    The fit starts with v_poles rows of poles, which smooths the scatter of extracted sections out, and adds
    knots where a sharp turn of the planform needs them (_span_solve). A step from one loop to the next
    cannot be followed that way (more knots only tighten the ripple about it): loft_loops takes it out of
    the loops beforehand (_ease_flats)."""
    fams = {name: np.asarray(rows, dtype=float) for name, rows in fams.items()}
    params = np.asarray(params, dtype=float)
    m = len(params)
    n_v = int(min(max(v_poles, 2), m))
    if n_v == 2:
        knots = np.array([0.0, 0.0, 1.0, 1.0])
        grids, err = {}, 0.0
        for name, rows in fams.items():
            grids[name] = rows[[0, -1]]
            pred = rows[0][None] + params[:, None, None] * (rows[-1] - rows[0])[None]
            err = max(err, float(np.abs(pred - rows).max()))
        return grids, knots, 1, err
    deg = int(min(degree, n_v - 1))
    grids, knots, res = _span_solve(fams, params, n_v, deg, tol, passes)
    return grids, knots, deg, float(res.max())


def _ease_flats(flats, z_axis, v_poles, tol, side="both", ease_deg=EASE_DEG):
    """The horizontal loops with the steps between them eased out: (loops, [(the loop under the step counted
    from 1, the step mm, how far it is eased below and above mm, the shares of it taken below and above)]).

    A step from one loop to the next (sections from another source under the extracted ones, as a rudder's
    Bezier model alone, a tenth of a mm off its extracted loops) cannot be followed by a smooth surface;
    fitted as it is, it leaves a ripple round the body. The loops' points are fitted along the span as the
    patches' rows will be; where loops still miss that fit by more than tol, the pair of neighbouring loops
    with the largest jump (what the loops' own trend on both sides does not explain) marks a step. The jump
    is then taken out of the loops, fading out smoothly over a band below the step and one above, tall
    enough for the surface to tilt by ease_deg at most (each at most half the span, and never as far as the
    end loops). side "both" shares the jump so that both bands tilt alike, "below" takes all of it out of
    the loops under the step (those above stay as they are), "above" out of those over it. The loops stay
    horizontal: a jump has no part along the height."""
    flats = [np.asarray(lp, dtype=float) for lp in flats]
    m = len(flats)
    steps = []
    if m < 8:
        return flats, steps
    rows = np.array(flats)
    z = rows[:, 0, z_axis]
    length = float(z[-1] - z[0])
    if length <= 0.0 or np.any(np.diff(z) <= 0.0):
        return flats, steps
    params = (z - z[0]) / length
    n_v = int(min(max(v_poles, 4), m))
    _, _, res = _span_solve({"loops": rows}, params, n_v, 3, tol)
    bad = np.where(res > tol)[0]
    if not len(bad):
        return flats, steps
    out = rows.copy()
    for group in np.split(bad, np.where(np.diff(bad) > 2)[0] + 1):
        best, at, jump = 0.0, None, None
        for j in range(max(int(group[0]) - 2, 1), min(int(group[-1]) + 2, m - 2) + 1):
            dp, gap = params[j + 1] - params[j], rows[j + 1] - rows[j]
            # the jump: what the loops' trend does not explain, by the trend under the pair and by the one
            # over it (the last loop has none over it). A pair next to a step shows a jump by one of the
            # two only, the step itself by both: the smaller one counts
            ds = [gap - (rows[j] - rows[j - 1]) / (params[j] - params[j - 1]) * dp]
            if j + 2 < m:
                ds.append(gap - (rows[j + 2] - rows[j + 1]) / (params[j + 2] - params[j + 1]) * dp)
            sizes = [float(np.linalg.norm(d, axis=1).max()) for d in ds]
            if min(sizes) > best:
                best, at, jump = min(sizes), j, sum(ds) / len(ds)
        if at is None or best <= 2.0 * tol:
            continue
        mid = 0.5 * (params[at] + params[at + 1])
        reach = 1.875 * best / np.tan(np.radians(ease_deg)) / length       # the two bands together
        lo_max = max(min(0.5, mid - params[1]), 0.0)
        hi_max = max(min(0.5, params[-2] - mid), 0.0)      # nothing over a step under the last loop
        if side == "below" or (side == "both" and hi_max == 0.0):
            lo, hi, share = min(reach, lo_max), 0.0, 1.0
        elif side == "above" or lo_max == 0.0:
            lo, hi, share = 0.0, min(reach, hi_max), 0.0
        else:                                              # both bands with the same tilt
            scale = min(1.0, reach / (lo_max + hi_max))
            lo, hi = scale * lo_max, scale * hi_max
            share = lo / (lo + hi)
        if lo + hi <= 0.0:
            continue
        below = params <= params[at]
        tau = np.where(below, (params - (mid - lo)) / max(lo, 1e-300), ((mid + hi) - params) / max(hi, 1e-300))
        tau = np.clip(tau, 0.0, 1.0)
        w = tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2) * np.where(below, share, share - 1.0)
        w[0] = w[-1] = 0.0
        out += w[:, None, None] * jump[None]
        steps.append((int(at) + 1, float(best), float(lo * length), float(hi * length), float(share),
                      float(1.0 - share)))
    return [lp for lp in out], steps


def _wall_loops(flats, z_axis, v_poles, tol):
    """The horizontal loops as their fit along the span gives them: (loops, the largest distance of a loop's
    point from it, mm).

    Every point index of the loops is a line up the body. Those lines are fitted over the height as the
    patches' rows of poles are fitted along the span (_span_solve: v_poles rows of poles to start with, knots
    added where a loop lies further than tol from the fit; the first and the last loop held), and the loops
    are read back off the fit. Scatter from loop to loop (a section algorithm's own steps, a crease of the
    CAD the loops were cut from) is taken out of them here, once: the patches' rows, the lines up the body
    that the rows are turned along and the tip patch's sides then all come off one smooth surface. The loops
    stay horizontal."""
    rows = np.array([np.asarray(lp, dtype=float) for lp in flats])
    m = len(rows)
    z = rows[:, 0, z_axis]
    if m < 8 or np.any(np.diff(z) <= 0.0):
        return [lp for lp in rows], 0.0
    params = (z - z[0]) / (z[-1] - z[0])
    n_v = int(min(max(v_poles, 4), m))
    grids, knots, res = _span_solve({"loops": rows}, params, n_v, 3, tol)
    fit = (_design(params, knots, 3) @ grids["loops"].reshape(len(knots) - 4, -1)).reshape(rows.shape)
    fit[0], fit[-1] = rows[0], rows[-1]
    fit[:, :, z_axis] = z[:, None]
    return [lp for lp in fit], float(np.linalg.norm(fit - rows, axis=2).max())


def _surface_face(grid, uknots, vknots, udeg=3, vdeg=3):
    """The face of the bicubic B-spline surface with the pole grid (n_v rows
    of n_u poles), u along the section."""
    gp_Pnt = _occ("gp", "gp_Pnt")
    TColgp_Array2OfPnt = _occ("TColgp", "TColgp_Array2OfPnt")
    Geom_BSplineSurface = _occ("Geom", "Geom_BSplineSurface")
    BRepBuilderAPI_MakeFace = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace")
    n_v, n_u = grid.shape[0], grid.shape[1]
    arr = TColgp_Array2OfPnt(1, n_u, 1, n_v)
    for i in range(n_u):
        for j in range(n_v):
            p = grid[j, i]
            arr.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    uk, um = _unique_mults(uknots)
    vk, vm = _unique_mults(vknots)
    surf = Geom_BSplineSurface(arr, _real_array(uk), _real_array(vk), _int_array(um), _int_array(vm),
                               int(udeg), int(vdeg))
    return BRepBuilderAPI_MakeFace(surf, 1.0e-6).Face()


def _line_edge(a, b):
    gp_Pnt = _occ("gp", "gp_Pnt")
    BRepBuilderAPI_MakeEdge = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeEdge")
    return BRepBuilderAPI_MakeEdge(gp_Pnt(*[float(v) for v in a]), gp_Pnt(*[float(v) for v in b])).Edge()


def _wire_face(edges):
    """The planar face bounded by the edges (chained into one wire), its
    orientation fixed (a wire chained from edges of other faces can come out
    unorientable otherwise)."""
    BRepBuilderAPI_MakeWire = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeWire")
    ShapeFix_Face = _occ("ShapeFix", "ShapeFix_Face")
    w = BRepBuilderAPI_MakeWire()
    for e in edges:
        w.Add(e)
    fix = ShapeFix_Face(_planar_face(w.Wire()))
    fix.FixOrientation()
    fix.Perform()
    return fix.Face()




# --------------------------------------------------------------------------
# The patches: four along the span, the tip patch over the cap
# --------------------------------------------------------------------------

N_DENSE = 1501                             # points per side of a row's smooth polyline
ROUND_DROP = 0.25                          # an edge's rounding ends where the curvature has fallen to this
                                           # share of its peak round the edge (_edge_rounding)
LE_MARGIN = 1.5                            # strip_frac None: the LE strip reaches this many times the LE's
                                           # rounding (to a quarter of the peak curvature) to each side of the LE
LE_WINDOW = 0.25                           # ... the LE's rounding looked for within this share of a side's arc
ROUND_SMALL = 0.5                          # one strip_frac given: the TE strip is narrowed to the TE's round
                                           # when that is under this share of the strip
TE_DROP = 0.1                              # the TE strip (as wide as the TE's round) reaches TE_MARGIN times the
TE_MARGIN = 1.0                            # arc from the TE to where the curvature has fallen to TE_DROP of its peak: a
                                           # round's end, where it meets the flank, lies just inside the strip's
                                           # edge (the strip is the round, the central panel starts on the flank;
                                           # BB2's 15 mm round inside a strip twice as wide left a ripple of 0.2
                                           # deg along its end, the whole span)
TIP_DROP = 1.5                             # the slope of the strips' top edges in the side view: the tip
                                           # patch's corners (the tops of the strip lines) sit under the tops
                                           # of the LE and TE lines by this times their distance from them
                                           # along the chord (loft_loops). On BB2's stern plane 0.5 to 2 all
                                           # gave valid patches without folds; the flatter the top edges, the
                                           # longer they run beside the rim and the less smooth the tip patch
                                           # under the LE's top (1.5 and 2 the smoothest)
FAN_FACTOR = 4.0                           # the rows turn onto the patches' top edges over a band this many
                                           # times as tall as those edges' spread in height
CAP_STATIONS = 401                         # stations along the chord of the tip's grid
CAP_ACROSS = 81                            # points from the rim to the middle of the cap on each wrap
BAND_ACROSS = 33                           # ... and from the patch's edge up the side to the rim
RIM_SHARE = 0.088                          # the cap's lines start from this share of a strip's top row to
                                           # each side of its edge line: the rim lies on the tip patch's
                                           # lines v = 0.5 -+ half of it (_rim_lines). A larger share (0.15
                                           # to 0.66 tried on BB2) leaves the rim crossing the patch's lines
                                           # over a longer stretch and takes two to seven times the poles
EDGE_ANGLE = 15.0                          # no line of the tip patch leaves one of its edges at less than this
                                           # (deg; _into_plane)
CAP_BLEND = 0.5                            # where the cap's wraps jump in curvature between their ends (an
                                           # edge rounding's two ends: a wall, the round, the crown) every
                                           # wrap is averaged over this share of the cap's tightest radius
                                           # (the median over the wraps of their own), twice over, before the
                                           # patch is fitted (_blended). Jumps that lie on no line of the
                                           # patch make it ring (the wind-tunnel rudder's 0.7 mm round: 1 deg
                                           # either side of it); blended, the round moves in by about
                                           # CAP_BLEND^2 / 12 of that radius (0.013 mm there). 0 turns it off
BLEND_SHARE = 0.08                         # ... and over at most this share of a wrap's own length (the
                                           # short wraps about the tops of the edge lines)
APEX_STEPS = 8                            # the tip's wraps close in on the top of each edge line: stations at
                                           # RIM_SHARE of the strip's width from it (where the cap's lines
                                           # leave the strip's top row) and at 3/4, 1/2, 3/8 .. of that, halved
                                           # this many times, beside the loops' own. The map needs wraps
                                           # there: loops with points 0.2 mm apart round a 0.75 mm TE gave it
                                           # two, and the tip patch's lines doubled back along that edge. The
                                           # wind-tunnel rudder is clean from 2 halvings on, a test wing with
                                           # a TE strip seven times its TE's round from 6
TIP_TOL = 3.0e-6                           # the tip patch gets knots where it misses the tip by more than this
TIP_PASSES = 6                             # share of the tip's chord, in at most this many rounds, up to
TIP_POLES = 150                            # this many poles each way (the spans that miss most first)
SPAN_TOL = 3.0e-5                          # the span fit adds knots where a pole row misses its grid by this
                                           # fraction of the span (_span_fit)
NAMES = ("le", "cen_a", "te", "cen_b")     # the four patches along the span


def _design(u, knots, degree=3, deriv=0):
    """The B-spline basis functions on knots at the parameters u (len(u), poles), or their derivative."""
    from scipy.interpolate import BSpline
    u = np.asarray(u, dtype=float)
    n = len(knots) - degree - 1
    if deriv:
        return BSpline(knots, np.eye(n), degree).derivative(deriv)(u)
    try:
        return BSpline.design_matrix(u, knots, degree).toarray()
    except AttributeError:                                         # scipy < 1.8
        return np.column_stack([BSpline(knots, np.eye(n)[j], degree)(u) for j in range(n)])


def _refined_knots(n_poles, kind, degree=3):
    """Clamped knots on 0 .. 1 for the wrap patches: n_poles - degree even spans, with more knots where the
    tip patch needs them. kind 'central' (along a section between the strips; the tip patch's chordwise
    direction): the first and the last span cut into 12, halved further towards the end, the next span
    halved (the rounding of the cap's two ends hugs those edges). kind 'strip' (around the LE or the TE; the
    tip patch's direction across): the first and last six spans halved (the wall and the edge rounding along
    the sides), the two spans to each side of the middle cut into 4 (a small TE radius)."""
    base = np.linspace(0.0, 1.0, max(int(n_poles), degree + 1) - degree + 1)
    d = base[1]
    if kind == "central":
        extra = [d * j / 12.0 for j in range(1, 12)] + [d / 16.0, d / 32.0, 1.5 * d]
        inner = np.r_[base[1:-1], extra, 1.0 - np.array(extra)]
    else:
        extra = [d * (i + 0.5) for i in range(6)]
        j = np.where((base[:-1] < 0.5 + 2 * d - 1e-12) & (base[1:] > 0.5 - 2 * d + 1e-12))[0]
        mid = (base[j][:, None] + np.array([0.25, 0.5, 0.75])[None, :] * d).ravel()
        inner = np.r_[base[1:-1], extra, 1.0 - np.array(extra), mid]
    inner = np.unique(np.round(inner[(inner > 1e-9) & (inner < 1.0 - 1e-9)], 13))
    return np.r_[[0.0] * (degree + 1), inner, [1.0] * (degree + 1)]


def _knot_stations(knots, per=4):
    """Fit stations for a knot vector: per stations in every span (every pole keeps data in its support), the
    two ends and the middle."""
    uk = np.unique(knots)
    t = (np.arange(per) + 0.5) / per
    s = (uk[:-1, None] + t[None, :] * np.diff(uk)[:, None]).ravel()
    return np.unique(np.r_[0.0, s, 0.5, 1.0])


def _loop_arc(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def _at_arc(p, s, stations):
    """Points of the polyline p (arc s) at the arc stations."""
    out = np.empty((len(np.atleast_1d(stations)), p.shape[1]))
    for c in range(p.shape[1]):
        out[:, c] = np.interp(stations, s, p[:, c])
    return out


class _SmoothLoop:
    """A loop as its smooth curve, in the loops' layout (LE first, the A side to the TE, the B side back to
    the LE): the cubic spline through its points over their chord length, periodic unless the TE is a corner
    (a closed sharp TE: one open curve from the TE round the LE back to the TE).

    dense   the curve as a polyline of 2 n - 1 points (the LE at 0 and at the end, the TE at it = n - 1)
    s       the polyline's arc at its points; S its length, s_te the arc at the TE
    at()    points of the curve itself at arcs measured along that polyline: rows are sampled from the curve,
            not from the polyline, whose corners a strip a few mm wide would otherwise take up
    """

    def __init__(self, lp, i_te, n=N_DENSE):
        from scipy.interpolate import CubicSpline, PchipInterpolator
        lp = np.asarray(lp, dtype=float)
        if np.linalg.norm(lp[0] - lp[-1]) > 1.0e-9:
            lp = np.vstack((lp, lp[:1]))
        m = len(lp)
        a, b = lp[i_te] - lp[i_te - 1], lp[i_te + 1] - lp[i_te]
        cosang = float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-300)
        seg = np.maximum(np.linalg.norm(np.diff(lp, axis=0), axis=1), 1e-12)
        self.corner = bool(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))) > 100.0)
        if self.corner:
            q = np.vstack((lp[i_te:], lp[1:i_te + 1]))             # TE -> B side -> LE -> A side -> TE
            s = np.r_[0.0, np.cumsum(np.r_[seg[i_te:], seg[:i_te]])]
            cs = CubicSpline(s, q, axis=0)
            s_le = s[m - 1 - i_te]
            t_b = np.linspace(0.0, s_le, n)                         # TE -> LE
            t_a = np.linspace(s_le, s[-1], n)                       # LE -> TE
        else:
            s = np.r_[0.0, np.cumsum(seg)]
            q = lp.copy()
            q[-1] = q[0]
            cs = CubicSpline(s, q, axis=0, bc_type="periodic")
            t_a = np.linspace(0.0, s[i_te], n)
            t_b = np.linspace(s[i_te], s[-1], n)
        out = np.vstack((cs(t_a), cs(t_b)[1:]))
        out[0], out[n - 1], out[-1] = lp[0], lp[i_te], lp[0]
        self.dense, self.it = out, n - 1
        self.s = _loop_arc(out)
        self.S, self.s_te = float(self.s[-1]), float(self.s[n - 1])
        self._cs = cs
        self._t_of = (PchipInterpolator(self.s[:n], t_a), PchipInterpolator(self.s[n - 1:], t_b))
        self._le, self._te = lp[0].copy(), lp[i_te].copy()
        par_a, par_b = (s[m - 1 - i_te:], s[:m - i_te]) if self.corner else (s[:i_te + 1], s[i_te:])
        self.pt_arc = np.r_[np.interp(par_a, t_a, self.s[:n]),          # the arc at the loop's own points
                            np.interp(par_b, t_b, self.s[n - 1:])[1:]]  # (0 at the LE, S at its return)

    def _t(self, stations):
        st = np.mod(np.atleast_1d(np.asarray(stations, dtype=float)), self.S)
        on_a = st <= self.s_te
        t = np.empty(len(st))
        t[on_a], t[~on_a] = self._t_of[0](st[on_a]), self._t_of[1](st[~on_a])
        return st, t

    def at(self, stations):
        """Points of the curve at the arcs `stations` (any real: the loop is closed)."""
        st, t = self._t(stations)
        out = self._cs(t)
        out[st <= 1.0e-12 * self.S] = self._le
        out[np.abs(st - self.s_te) <= 1.0e-12 * self.S] = self._te
        return out

    def curvature(self, stations):
        """The curve's curvature (1 / mm) at the arcs `stations`."""
        _, t = self._t(stations)
        d1, d2 = self._cs(t, 1), self._cs(t, 2)
        return np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1), 1e-300) ** 3


def _edge_rounding(loop, apex, window, share=ROUND_DROP):
    """The arc (mm) an edge's rounding reaches to each side of the edge at arc `apex` of a _SmoothLoop: the
    farthest point within `window` of the edge where the curvature is still `share` of its peak there (a
    rounded TE: to where the round meets the flank; a blunt base with rounded corners: past the corners; an
    LE: about 1.5 nose radii). The curvature is read at a scale that resolves the rounding (_bend, on steps
    of a hundredth of the window, halved until the rounding is 10 steps long), not finer.
    0 for a corner (a sharp TE has no rounding to wrap)."""
    if loop.corner and abs(apex - loop.s_te) < 1.0e-9 * loop.S:
        return 0.0
    step, out = float(window) / 100.0, 0.0
    for _ in range(12):
        m = int(round(float(window) / step))
        d = step * np.arange(m + 1)
        kap = np.maximum(_bend(loop, apex + d, step), _bend(loop, apex - d, step))
        drop = share * kap.max()
        j = int(np.where(kap >= drop)[0][-1])
        out = float(d[-1]) if j + 1 >= len(d) else \
            float(d[j] + (d[j + 1] - d[j]) * (kap[j] - drop) / max(kap[j] - kap[j + 1], 1e-300))   # the crossing
        if out >= 10.0 * step or step <= 1.0e-6 * loop.S:
            break                                              # the rounding is resolved at this scale
        step *= 0.5
    return out


def _bend(loop, arcs, step):
    """The curvature (1 / mm) of a _SmoothLoop at the arcs, as its points one step to each side of them give
    it (the circle through the three): the curvature at that scale, clear of the scatter of points packed
    closer than they are exact (801 points a loop put the spline's own curvature a hundredfold off within a
    few mm of BB2's LE)."""
    arcs = np.atleast_1d(np.asarray(arcs, dtype=float))
    a, b, c = loop.at(arcs - step), loop.at(arcs), loop.at(arcs + step)
    ab, bc, ac = (np.linalg.norm(q - p, axis=1) for p, q in ((a, b), (b, c), (a, c)))
    return 2.0 * np.linalg.norm(np.cross(b - a, c - a), axis=1) / np.maximum(ab * bc * ac, 1e-300)


def _row_arcs(loop, w_le, w_te, u_strip, u_cen):
    """One loop (a _SmoothLoop) split into the four wrap families' sample rows: (le strip, central A, te
    strip, central B), the strips wrapping their apexes from the A side to the B side, the centrals running
    from the LE split to the TE split of their side; the four split points are shared exactly, the apexes
    sit exactly at the strips' middle sample. Every row's parameter is linear in the arc along the loop."""
    S, s_te = loop.S, loop.s_te
    lim = 0.45 * min(s_te, S - s_te)
    if not (w_le < lim and w_te < lim):
        raise ValueError(f"the strips ({w_le:.6g} and {w_te:.6g} mm about the LE and the TE) overlap on the "
                         f"section at height {loop.dense[0, 2]:.6g}: its sides are {s_te:.6g} and "
                         f"{S - s_te:.6g} mm long (lower strip_frac)")
    p_le_a, p_te_a, p_te_b, p_le_b = loop.at([w_le, s_te - w_te, s_te + w_te, S - w_le])
    le = loop.at((S + w_le) - u_strip * (2.0 * w_le))
    te = loop.at((s_te - w_te) + u_strip * (2.0 * w_te))
    cen_a = loop.at(w_le + u_cen * (s_te - w_te - w_le))
    cen_b = loop.at((S - w_le) - u_cen * (S - w_le - s_te - w_te))
    mid = int(np.argmin(np.abs(u_strip - 0.5)))
    le[mid], te[mid] = loop.dense[0], loop.dense[loop.it]  # the apexes exactly
    le[0], le[-1] = p_le_a, p_le_b
    te[0], te[-1] = p_te_a, p_te_b
    cen_a[0], cen_a[-1] = p_le_a, p_te_a
    cen_b[0], cen_b[-1] = p_le_b, p_te_b
    return le, cen_a, te, cen_b


class _Track:
    """A line up the body through one point of every level (the horizontal loops, then the rim, whose points
    sit at heights of their own): the cubic spline through those points over their height. Loops that do not
    lie under the rim's point are left out of it."""

    def __init__(self, pts, z_axis):
        from scipy.interpolate import CubicSpline
        pts = np.asarray(pts, dtype=float)
        z = pts[:, z_axis]
        self.keep = np.r_[z[:-1] < z[-1] - 1.0e-9 * max(abs(float(z[-1])), 1.0), True]
        self.pts, self.z = pts[self.keep], z[self.keep]
        if len(self.z) < 2:
            raise ValueError("no horizontal loop lies under the tip's rim")
        self._cs = CubicSpline(self.z, self.pts, axis=0)

    def at(self, z):
        """Points of the track at the heights z (held within its levels)."""
        return self._cs(np.clip(z, self.z[0], self.z[-1]))

    def top_slope(self):
        """The track's direction at its top (its derivative over the height there)."""
        return self._cs(self.z[-1], 1)

    def meets(self, normal, point, slack=0.0):
        """The height at which the track crosses the plane through `point` with the normal `normal` (its
        highest crossing), or None when it does not cross it within its levels. A track that stays on one
        side of the plane with its top within slack (mm) of it meets it there (the rim's smooth curve passes
        the top of an edge line a hair off the plane through that top)."""
        from scipy.interpolate import CubicSpline
        from scipy.optimize import brentq
        g = (self.pts - point) @ normal
        if abs(g[-1]) <= 1.0e-12 * max(float(np.abs(g).max()), 1e-300):
            return float(self.z[-1])
        j = np.where(g[:-1] * g[1:] <= 0.0)[0]
        if not len(j):                                             # no crossing: the top itself when it is
            return float(self.z[-1]) if abs(g[-1]) <= float(slack) else None   # that close under the plane
        j = int(j[-1])
        if g[j] == 0.0:
            return float(self.z[j])
        return float(brentq(CubicSpline(self.z, g), self.z[j], self.z[j + 1], xtol=1.0e-12, rtol=1.0e-14))


def _plane(p, q, r=None, along=None):
    """(unit normal, a point) of the plane through p, q and r, or through p and q along a direction."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    nrm = np.cross(q - p, (np.asarray(r, dtype=float) - p) if r is not None else np.asarray(along, dtype=float))
    ln = float(np.linalg.norm(nrm))
    if ln <= 1.0e-12 * max(float(np.linalg.norm(q - p)), 1e-300):
        raise ValueError("the tip patch's corners and the edge's top do not span a plane")
    return nrm / ln, p


def _tip_rows(grid, zf, z_axis, thick, mid, slopes, slack=0.0, factor=FAN_FACTOR):
    """The rows of the four patches along the span under a tip, ending on the tip patch's edges.

    grid    {name: (levels, columns, 3)}: the four patches' sample rows on every horizontal loop and, last, on
            the rim (_row_arcs). A column is one arc station of its patch, so its points run up the body: a
            track (_Track)
    zf      the horizontal loops' heights
    mid     the strips' apex column (the LE or TE line)
    slopes  the slope of the LE strip's and of the TE strip's top edge in the side view (rise over run along
            the chord)

    The tip patch's four corners are points of the strip lines (the strips' end columns, shared with the
    central panels): each pair as far under the top of its edge line as slope times its distance from that
    top along the chord. A strip's top row is the cut of the body by the plane through its two corners and the
    top of its edge line (the blade's wrap cut, DRDC TM 2013-178: a straight diagonal in the side view, through
    the edge's top, where the cap begins); a central panel's top row is the cut by the plane through its two
    corners along the thickness (the straight line between the strip lines' tops). Every column ends where its
    track crosses its patch's plane.
    Under those edges the rows are the horizontal loops, turned onto the edges over a band `factor` times as
    tall as the edges' spread in height: at a level in the band every column's point moves up its track by a
    share of the edge's height there, growing from 0 at the band's foot to all of it at the top, with no jump
    in speed at either end. The points stay on the surface the loops describe.
    Returns ({name: [rows]}, info): info has the corners, the planes, the tops of the edge lines, the band."""
    from scipy.optimize import brentq
    names = NAMES
    le, ca, te, cb = (grid[nm] for nm in names)
    x_le, x_te = le[-1, mid].copy(), te[-1, mid].copy()
    chord = 3 - z_axis - thick
    room = ValueError("the tip patch's corners do not leave room for the patches along the span: lower "
                      "tip_drop or the strips' widths")

    def corner(top, cols, slope):
        """The height of a strip's two corners: as far under the top of its edge line as slope times their
        (mean) distance from it along the chord."""
        trks = [_Track(col, z_axis) for col in cols]

        def gap(z):
            return (top[z_axis] - z) - slope * float(np.mean([abs(t.at(z)[chord] - top[chord]) for t in trks]))

        lo = max(float(t.z[0]) for t in trks)
        if gap(lo) <= 0.0:
            raise room
        return float(brentq(gap, lo, float(top[z_axis]), xtol=1.0e-10))

    z_ur = corner(x_le, (ca[:, 0], cb[:, 0]), slopes[0])
    z_ul = corner(x_te, (ca[:, -1], cb[:, -1]), slopes[1])
    drops = (float(x_le[z_axis] - z_ur), float(x_te[z_axis] - z_ul))
    z_ref = min(z_ur, z_ul)
    spread = max(float(x_le[z_axis]), float(x_te[z_axis])) - z_ref
    if z_ref <= zf[0] + 0.05 * (max(float(x_le[z_axis]), float(x_te[z_axis])) - zf[0]) or len(zf) < 6:
        raise room
    band = min(factor * spread, 0.5 * (z_ref - zf[0]))
    z_b = z_ref - band
    j0 = int(np.searchsorted(zf, z_b + 1.0e-9))                  # the band's first loop
    js = max(j0 - 4, 0)
    tracks = {nm: [_Track(grid[nm][js:, c], z_axis) for c in range(grid[nm].shape[1])] for nm in names}
    ur_a, ur_b = tracks["cen_a"][0].at(z_ur), tracks["cen_b"][0].at(z_ur)
    ul_a, ul_b = tracks["cen_a"][-1].at(z_ul), tracks["cen_b"][-1].at(z_ul)
    e_t = np.zeros(3)
    e_t[thick] = 1.0
    planes = {"le": _plane(x_le, ur_a, ur_b), "te": _plane(x_te, ul_a, ul_b),
              "cen_a": _plane(ur_a, ul_a, along=e_t), "cen_b": _plane(ur_b, ul_b, along=e_t)}
    tops = {}
    for nm in names:
        zt = [trk.meets(*planes[nm], slack) for trk in tracks[nm]]
        if any(z is None for z in zt):
            raise ValueError("a patch's top edge leaves the body between the loops: lower tip_drop")
        tops[nm] = np.array(zt)
    tops["le"][[0, -1]] = tops["cen_a"][0] = tops["cen_b"][0] = z_ur          # the corners, shared exactly
    tops["te"][[0, -1]] = tops["cen_a"][-1] = tops["cen_b"][-1] = z_ul

    def ramp(level):
        tau = np.clip((level - z_b) / band, 0.0, 1.0)
        return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)

    # the rows' levels: the loops' own heights under the lowest corner and, for the loops above it, the
    # levels whose rows reach those loops' heights on the track that climbs most (an edge line), so the rows
    # are as close together up there as the loops are (a tip corner's rounding lives in the last of them)
    lv = zf[j0:]
    lv = lv[lv < z_ref]
    climb = max(float((tops[nm] - z_ref).max()) for nm in names)
    dense = np.linspace(z_b, z_ref, 4001)
    reach = dense + ramp(dense) * climb
    over = zf[(zf > z_ref) & (zf < z_ref + climb)]
    lv = np.sort(np.r_[lv, np.interp(over, reach, dense)])
    gap = 1.0e-9 * max(abs(z_ref), 1.0)
    keep = np.r_[True, np.diff(lv) > gap] if len(lv) else np.zeros(0, dtype=bool)
    lv = lv[keep]
    lv = lv[lv < z_ref - gap]
    if len(lv) >= 2 and z_ref - lv[-1] < 0.25 * (lv[-1] - lv[-2]):
        lv = lv[:-1]                                             # a level right under the top row
    levels = np.r_[lv, z_ref]
    g = ramp(levels)
    g[-1] = 1.0
    rows = {}
    for nm in names:
        new = np.empty((len(levels), grid[nm].shape[1], 3))
        for c, trk in enumerate(tracks[nm]):
            new[:, c] = trk.at(levels + g * (tops[nm][c] - z_ref))
        rows[nm] = new
    rows["le"][-1, mid], rows["te"][-1, mid] = x_le, x_te        # the tops of the edge lines exactly
    for a, ia, b, ib in (("le", 0, "cen_a", 0), ("le", -1, "cen_b", 0), ("te", 0, "cen_a", -1),
                         ("te", -1, "cen_b", -1)):
        rows[a][:, ia] = rows[b][:, ib]                          # the strip lines: one point for both patches
    out = {nm: [grid[nm][j] for j in range(j0)] + list(rows[nm]) for nm in names}
    info = {"x_le": x_le, "x_te": x_te, "planes": planes, "z_ur": z_ur, "z_ul": z_ul, "band": float(band),
            "js": js, "slack": float(slack), "drops": drops, "chord": chord,
            "corners": {"ur_a": rows["cen_a"][-1, 0].copy(), "ur_b": rows["cen_b"][-1, 0].copy(),
                        "ul_a": rows["cen_a"][-1, -1].copy(), "ul_b": rows["cen_b"][-1, -1].copy()}}
    return out, info


class _TipTruth:
    """The tip above the four patches' top rows as one smooth surface: the grid of its wraps' points
    (stations along the chord x points across) with the bicubic through it, block by block: the side under
    the rim on each side (cut at the stations of the patch's corners, where its lower edge turns from a
    strip's top row to a central panel's) and the cap between the two rims. The blocks meet on the rims and
    are smooth inside, as the tip is. The first and the last station are the tops of the LE and TE lines,
    where the wraps have shrunk to a point: the surface runs smoothly into them."""

    def __init__(self, xi, grid, j_ra, j_rb, cuts):
        from scipy.interpolate import RectBivariateSpline
        self.xi, self.grid = np.asarray(xi, dtype=float), np.asarray(grid, dtype=float)
        n_st, n_ac = self.grid.shape[:2]
        jj = np.arange(n_ac, dtype=float)
        self.blocks = []
        spans = [(j_ra, j_rb, [0, n_st - 1])]
        for (j0, j1), cut in (((0, j_ra), cuts[0]), ((j_rb, n_ac - 1), cuts[1])):
            spans.append((j0, j1, sorted({0, n_st - 1} | {int(c) for c in cut})))
        for j0, j1, ends in spans:
            for i0, i1 in zip(ends[:-1], ends[1:]):
                if i1 <= i0 or j1 <= j0:
                    continue
                kx, ky = min(3, i1 - i0), min(3, j1 - j0)
                spl = [RectBivariateSpline(self.xi[i0:i1 + 1], jj[j0:j1 + 1], self.grid[i0:i1 + 1, j0:j1 + 1, c],
                                           kx=kx, ky=ky, s=0) for c in range(3)]
                self.blocks.append((self.xi[i0], self.xi[i1], float(j0), float(j1), spl))

    def at(self, x, j):
        """The surface's points at stations x (the chord parameter) and places j across (the grid's index)."""
        x, j = np.asarray(x, dtype=float), np.asarray(j, dtype=float)
        out = np.full((len(x), 3), np.nan)
        todo = np.ones(len(x), dtype=bool)
        tol = 1.0e-9
        for x0, x1, j0, j1, spl in self.blocks:
            m = todo & (x >= x0 - tol) & (x <= x1 + tol) & (j >= j0 - tol) & (j <= j1 + tol)
            if m.any():
                xx, yy = np.clip(x[m], x0, x1), np.clip(j[m], j0, j1)
                out[m] = np.column_stack([s.ev(xx, yy) for s in spl])
                todo[m] = False
        return out


def _into_pinch(c_a, c_b, xi, x1, x2, w_a, w_b, pinched):
    """The cap's wraps between a pinch point (the top of an edge line, where the cap's loops meet the rim; the
    end of the stations nearer x1) and the loops' first points beside it, set in place.

    c_a, c_b  the rim's and the cap loops' points at the stations xi on the two sides, (loops, stations, 3)
    x1, x2    the station of the loops' first points beside the pinch, and that of their next
    w_a, w_b  the loops' points at x1, (loops, 3): the first whole wrap
    pinched   the loops that run through the rim's point at this end (the rim itself among them); a loop
              that ends elsewhere keeps its own cubic

    The loops say nothing about the cap between the pinch and x1, and a cubic through each loop's own points
    lets the upper loops cross the lower ones on their way into the pinch: the cap folds (loops with points
    0.5 mm apart: within 0.3 mm of the wind-tunnel rudder's LE top). So the wraps there are the first whole
    wrap scaled down into the pinch: each of its points keeps its place along the chord between the rim's
    two points, and its offset from that chord shrinks with the station's distance from the pinch (the cone
    over the first wrap, set on the rim as its own smooth curve runs). From x1 to x2 the wraps go over to the
    loops' own cubics, without a kink at either."""
    pinch = 0.0 if x1 < 0.5 else 1.0
    chord = w_b[0] - w_a[0]
    tau = [((w - w_a[0]) @ chord) / float(chord @ chord) for w in (w_a, w_b)]
    off = [w - (w_a[0] + np.outer(t, chord)) for w, t in zip((w_a, w_b), tau)]
    far = abs(x2 - pinch) / abs(x1 - pinch)
    for i in np.where(np.abs(xi - pinch) < abs(x2 - pinch))[0]:
        f = abs(xi[i] - pinch) / abs(x1 - pinch)
        ramp = min(max((f - 1.0) / (far - 1.0), 0.0), 1.0)
        share = ramp * ramp * (3.0 - 2.0 * ramp)               # of the loops' own cubics
        foot, span = c_a[0, i].copy(), c_b[0, i] - c_a[0, i]
        for c, t, n in ((c_a, tau[0], off[0]), (c_b, tau[1], off[1])):
            cone = foot + np.outer(t, span) + f * n
            c[pinched, i] = (1.0 - share) * cone[pinched] + share * c[pinched, i]


def _curvature_jumps(curve, s, share=0.5, turn=0.5):
    """(Whether a wrap's curvature jumps somewhere between its ends, its largest curvature): a jump when
    from one of the loops' points to the next the curvature changes by more than `share` of the
    curvature of its tightest stretch (the 95th percentile) within less than `turn` radians of turning.
    curve: the cubic through the wrap's points, at the arcs s. The two points at each end are left out of
    the jumps: the ends carry the turn into the side's tangent plane."""
    if len(s) < 9:
        return False, 0.0
    fine = np.linspace(s[0], s[-1], 1601)
    d1, d2 = curve(fine, 1), curve(fine, 2)
    kap = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1), 1e-300) ** 3
    inner = (fine >= s[2]) & (fine <= s[-3])
    top = float(np.percentile(kap[inner], 95.0))
    at = np.interp(s[2:-2], fine, kap)
    step = np.abs(np.diff(at)) > share * top
    return bool((step & (np.diff(s[2:-2]) * top < turn)).any()), float(kap.max())


def _blended(curve, length, ends, passes=2, n=4001):
    """A wrap with every jump of its curvature turned into a ramp: each point becomes the mean of the
    curve over the stretch `length` of its arc about it, `passes` times over (the curvature then runs from
    the one level to the other without a kink). The stretch shrinks to nothing towards the wrap's two
    ends, which stay where they are, with their directions (ends: the cubic's end conditions). A convex
    stretch moves in by about its curvature times length^2 / 24 per pass."""
    from scipy.interpolate import CubicSpline
    s = np.linspace(float(curve.x[0]), float(curve.x[-1]), int(n))
    half = np.minimum(0.5 * length, np.minimum(s - s[0], s[-1] - s))
    inner = half > 0.0
    for _ in range(int(passes)):
        area, pts = curve.antiderivative(), curve(s)
        pts[inner] = (area(s[inner] + half[inner]) - area(s[inner] - half[inner])) / (2.0 * half[inner, None])
        curve = CubicSpline(s, pts, axis=0, bc_type=ends)
    return curve


def _tip_mesh(flats, top, arcs, i_te, z_axis, info, widths, n_xi=CAP_STATIONS, m=CAP_ACROSS, m_band=BAND_ACROSS):
    """The tip above the four patches' top rows: its wraps' points as a grid and a triangle mesh, the
    boundary's place on the tip patch's unit square, and the smooth surface through the grid.

    flats   the horizontal loops from under the band (info's js) up to the rim (the rim not among them)
    top     the rim and the loops above it (the cap), nested and pinched to the rim's LE and TE points
    arcs    per level (flats, then the rim): (the arcs at the loop's own points, the arc at its TE)
    info    _tip_rows': the planes of the top rows, the corners' heights
    widths  the strips' reach, mm

    Per chord station (the loops' own index fraction; the four corners' stations among them, and stations
    closing in on the tops of the two edge lines, APEX_STEPS) one wrap runs from the A side's top row up the
    side to the rim, over the cap, and down the B side to its top row. Under the rim it is the station's
    track itself (the side as the loops give it). Over the cap it is the cubic spline through the rim's
    point, the points of the loops above and the rim's point on the other side, leaving both rims in the
    side's tangent plane there: the tip is smooth across the rim (it turns there from the straight side into
    the cap's rounding, without a crease).
    Between the top of an edge line, where every cap loop meets the rim, and the loops' first points beside
    it, the wraps are the first whole wrap scaled down into the pinch (_into_pinch).
    The boundary vertices are the wraps' two ends and the tops of the LE and TE lines; their parameters are
    those of the top rows they lie on: a point of a strip's top row at v from 0 (the A side's corner) through
    0.5 (the edge line) to 1 on u = 0 or 1, a point of a central panel's at u from 0 to 1 on v = 0 or 1,
    linear in the arc along the section at the point's height, as the rows' own parameters are.
    Where most wraps jump in curvature between their ends (an edge rounding's ends), every wrap's cap part
    is blended over CAP_BLEND of its tightest radius (_blended).
    Returns (vertices, triangles, boundary vertices' indices, their (u, v), the grid of the wraps' vertex
    indices (stations, across), (the grid's column on the A side's rim, on the B side's), the _TipTruth,
    per vertex its (station, place across) on it (the two extra vertices, the edge lines' tops, are the
    surface's two ends, with no place across of their own), and (the stretch the cap's wraps were blended
    over, how far that moved the cap), mm, both 0 when the cap was left as it is)."""
    from scipy.interpolate import CubicSpline, PchipInterpolator
    from scipy.optimize import brentq
    flats = [np.asarray(lp, dtype=float) for lp in flats]
    top = [np.asarray(lp, dtype=float) for lp in top]
    planes, (w_le, w_te) = info["planes"], widths
    n_pts = len(top[0])
    t_a, t_b = np.linspace(0.0, 1.0, i_te + 1), np.linspace(0.0, 1.0, n_pts - i_te)
    zf = np.array([lp[0, z_axis] for lp in flats])

    def tables(z):
        """The arcs from the LE at the loops' points on the section at height z: (A side, B side)."""
        j = int(np.clip(np.searchsorted(zf, z) - 1, 0, len(zf) - 2))
        t = float(np.clip((z - zf[j]) / (zf[j + 1] - zf[j]), 0.0, 1.0))
        out = []
        for pa, _ in (arcs[j], arcs[j + 1]):
            out.append((pa[:i_te + 1], (pa[-1] - pa[i_te:])[::-1]))
        return ((1.0 - t) * out[0][0] + t * out[1][0], (1.0 - t) * out[0][1] + t * out[1][1])

    levels = flats + [top[0]]
    line_a = [CubicSpline(t_a, lp[:i_te + 1], axis=0) for lp in levels]        # the levels' two sides over
    line_b = [CubicSpline(t_b, lp[i_te:][::-1], axis=0) for lp in levels]      # the loops' index

    def station(point, side, guess):
        """The station whose line up the side passes a corner of the tip patch: at the corner's height, as far
        along the chord as the corner (guess: the station the arcs of the loops under it give)."""
        def miss(x):
            trk = _Track(np.array([c(x) for c in (line_b if side else line_a)]), z_axis)
            return float((trk.at(point[z_axis]) - point)[info["chord"]])

        for d in (0.01, 0.04, 0.2):
            lo, hi = max(guess - d, 1.0e-6), min(guess + d, 1.0 - 1.0e-6)
            if miss(lo) * miss(hi) <= 0.0:
                return float(brentq(miss, lo, hi, xtol=1.0e-13))
        return guess

    # the stations: even in the loops' index, the four corners', and those closing in on the two ends
    corner = []                                                # (station, side 0 A or 1 B, (u, v))
    for z, at_le, names in ((info["z_ur"], True, ("ur_a", "ur_b")), (info["z_ul"], False, ("ul_a", "ul_b"))):
        sa, sb = tables(z)
        for side, (s, t) in enumerate(((sa, t_a), (sb, t_b))):
            guess = float(PchipInterpolator(s, t)(w_le if at_le else s[-1] - w_te))
            corner.append((station(info["corners"][names[side]], side, guess), side,
                           (0.0 if at_le else 1.0, float(side))))
    base = np.linspace(0.0, 1.0, int(n_xi))[1:-1]
    cs = np.array([c[0] for c in corner])
    apart = 0.1 / (int(n_xi) - 1)                              # a tenth of the even stations' spacing
    for j in range(len(cs)):                                   # the two sides' stations of one corner: one
        near = np.abs(cs - cs[j]) <= apart                     # station when they all but agree (a body
        cs[near] = cs[near].mean()                             # symmetric but for its loops' scatter)
    corner = [(float(c), side, uv) for c, (_, side, uv) in zip(cs, corner)]
    first = {side: st for st, side, uv in corner if uv[0] == 0.0}          # each side's station at its LE
    last = {side: st for st, side, uv in corner if uv[0] == 1.0}           # corner and at its TE corner
    xi = np.sort(np.r_[base[np.abs(base[:, None] - cs[None, :]).min(axis=1) > apart], np.unique(cs)])
    reach = RIM_SHARE * np.outer(0.5 ** np.arange(int(APEX_STEPS)), (1.0, 0.75)).ravel()
    s_rim = arcs[-1][0][:i_te + 1]                             # the rim's arcs from the LE, the A side
    close = PchipInterpolator(s_rim, t_a)(np.r_[w_le * reach, s_rim[-1] - w_te * reach])
    close = close[(close > 0.0) & (close < 1.0)]               # ... closing in on the tops of the edge lines
    end = np.minimum(close, 1.0 - close)                       # (one that all but falls on a station is left out)
    xi = np.sort(np.r_[xi, close[np.abs(close[:, None] - xi[None, :]).min(axis=1) > 0.01 * end]])

    p_a = np.array([c(xi) for c in line_a])                    # (levels, stations, 3)
    p_b = np.array([c(xi) for c in line_b])
    s_a = np.array([CubicSpline(t_a, pa[:i_te + 1])(xi) for pa, _ in arcs])              # arcs from the LE
    s_b = np.array([CubicSpline(t_b, (pa[-1] - pa[i_te:])[::-1])(xi) for pa, _ in arcs])
    l_a = np.array([s_te for _, s_te in arcs])
    l_b = np.array([pa[-1] - s_te for pa, s_te in arcs])
    ring_a = [CubicSpline(t_a, lp[:i_te + 1], axis=0) for lp in top]           # the rim and the cap's loops:
    ring_b = [CubicSpline(t_b, lp[i_te:][::-1], axis=0) for lp in top]         # their two sides over the index
    c_a = np.array([c(xi) for c in ring_a])                                    # (loops, stations, 3)
    c_b = np.array([c(xi) for c in ring_b])
    size = max(float(np.ptp(top[0], axis=0).max()), 1.0e-12)
    for at, x1, x2 in ((0, max(t_a[1], t_b[1]), max(t_a[2], t_b[2])),
                       (i_te, min(t_a[-2], t_b[-2]), min(t_a[-3], t_b[-3]))):
        pinched = np.array([float(np.linalg.norm(lp[at] - top[0][at])) <= 1.0e-6 * size for lp in top])
        _into_pinch(c_a, c_b, xi, x1, x2, np.array([c(x1) for c in ring_a]), np.array([c(x1) for c in ring_b]),
                    pinched)
    r_a = CubicSpline(t_a, top[0][:i_te + 1], axis=0)(xi, 1)                             # along the rim
    r_b = CubicSpline(t_b, top[0][i_te:][::-1], axis=0)(xi, 1)
    tb = np.arange(int(m_band)) / float(m_band)
    fb = 1.0 - (1.0 - tb) * (0.35 + 0.65 * (1.0 - tb))         # the top row to the rim, finer at the rim
    tc = np.linspace(0.0, 1.0, 2 * int(m) - 1)
    fc = tc - (0.65 / (2.0 * np.pi)) * np.sin(2.0 * np.pi * tc)   # rim to rim, finer at both

    def place(a, length, side, zone):
        """(u, v) of a top row's point at the arc a from the LE on a section side of that length: on the LE
        strip's top row (zone -1: its station lies before the side's LE corner), on the central panel's (0)
        or on the TE strip's (1). The station's place among the corners decides, not the arc: next to a
        corner the two can differ by a hair, and the boundary would double back on itself there."""
        sign = 2.0 * side - 1.0
        if zone < 0:
            return 0.0, 0.5 + sign * 0.5 * min(max(a, 0.0) / w_le, 1.0)
        if zone > 0:
            return 1.0, 0.5 + sign * 0.5 * min(max(length - a, 0.0) / w_te, 1.0)
        return min(max((a - w_le) / (length - w_le - w_te), 0.0), 1.0), side

    def in_plane(d, nrm):
        """The direction d turned into the plane with the normal nrm (unit vectors)."""
        d = d - (d @ nrm) * nrm
        ln = float(np.linalg.norm(d))
        return d / ln if ln > 1.0e-12 else None

    wraps, feet = [], []
    for i in range(len(xi)):
        band, uv_i, walls = [], [], []
        for side, (pts, rim, sarc, larc, names) in enumerate(
                ((p_a[:, i], r_a[i], s_a[:, i], l_a, ("le", "cen_a", "te")),
                 (p_b[:, i], r_b[i], s_b[:, i], l_b, ("le", "cen_b", "te")))):
            trk = _Track(pts, z_axis)
            zs = [z for z in (trk.meets(*planes[nm], info["slack"]) for nm in names) if z is not None]
            z0 = max(zs) if zs else float(trk.z[-1])           # the highest: the patches' top row here
            band.append(trk.at(z0 + fb * (trk.z[-1] - z0)))
            uv_i.append(place(float(np.interp(z0, trk.z, sarc[trk.keep])),
                              float(np.interp(z0, trk.z, larc[trk.keep])), float(side),
                              -1 if xi[i] < first[side] else 1 if xi[i] > last[side] else 0))
            nrm = np.cross(trk.top_slope(), rim)               # the side's normal on the rim
            ln = float(np.linalg.norm(nrm))
            walls.append(nrm / ln if ln > 1.0e-12 * size else None)
        for st, side, uv_c in corner:
            if abs(xi[i] - st) <= 1.0e-12:
                uv_i[side] = uv_c                              # a corner of the patch exactly
        q = np.vstack((c_a[:, i], c_b[::-1, i]))               # rim, the loops above, and down to the rim
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        q = q[np.r_[True, seg > 1.0e-9 * size]]
        s = _loop_arc(q)
        if len(q) < 3 or s[-1] <= 1.0e-7 * size:
            wraps.append(None)                                  # pinched to a corner
            continue
        free = CubicSpline(s, q, axis=0) if len(q) >= 4 else CubicSpline(s, q, axis=0, bc_type="natural")
        ends = []
        for at, nrm in ((0.0, walls[0]), (s[-1], walls[1])):
            d = free(at, 1)
            d = d / max(float(np.linalg.norm(d)), 1e-300)
            ends.append((1, in_plane(d, nrm) if nrm is not None and in_plane(d, nrm) is not None else d))
        over = CubicSpline(s, q, axis=0, bc_type=tuple(ends))
        wraps.append((band[0], over, tuple(ends), band[1][::-1]) + _curvature_jumps(over, s))
        feet.append(uv_i)
    good = [i for i, wr in enumerate(wraps) if wr is not None]
    if len(good) < 8:
        raise ValueError("the loops above the rim do not describe a cap")
    # the cap's rounding: where most wraps jump in curvature between their ends, every wrap is blended
    # over CAP_BLEND of the cap's tightest radius (the median over the wraps of their own: one stretch
    # along the chord, which the patch can follow), at most BLEND_SHARE of the wrap's own length (the
    # short wraps about the tops of the edge lines); a cap without jumps, BB2's, is left as it is
    blend = CAP_BLEND > 0.0 and 2 * sum(wraps[i][4] for i in good) > len(good)
    tight = float(np.median([wraps[i][5] for i in good]))
    grid, moved = [], 0.0
    stretch = CAP_BLEND / tight if blend and tight > 0.0 else 0.0
    for i in good:
        below, over, ends, above, _, _ = wraps[i]
        length = float(over.x[-1])
        plain = over(fc * length)
        if stretch > 0.0:
            over = _blended(over, min(stretch, BLEND_SHARE * length), ends)
        cap = over(fc * length)
        cap[0], cap[-1] = plain[0], plain[-1]
        moved = max(moved, float(np.linalg.norm(cap - plain, axis=1).max()))
        grid.append(np.vstack((below, cap, above)))
    g = np.array(grid)
    xg = xi[good]
    feet = np.array(feet)                                      # (stations kept, 2 sides, 2)
    n_st, n_ac = g.shape[:2]
    j_ra, j_rb = int(m_band), int(m_band) + 2 * int(m) - 2
    cuts = [[int(np.argmin(np.abs(xg - st))) + 1 for st, sd, _ in corner
             if sd == side and np.abs(xg - st).min() <= 1.0e-12] for side in (0, 1)]
    ends = [np.broadcast_to(info[key], (1, n_ac, 3)) for key in ("x_le", "x_te")]
    truth = _TipTruth(np.r_[0.0, xg, 1.0], np.concatenate((ends[0], g, ends[1])), j_ra, j_rb, cuts)
    verts = np.vstack((g.reshape(-1, 3), info["x_le"][None], info["x_te"][None]))
    c_le, c_te = n_st * n_ac, n_st * n_ac + 1
    idx = np.arange(n_st * n_ac).reshape(n_st, n_ac)
    on = np.full((len(verts), 2), np.nan)
    on[:c_le, 0] = np.repeat(xg, n_ac)
    on[:c_le, 1] = np.tile(np.arange(n_ac, dtype=float), n_st)
    on[c_le, 0], on[c_te, 0] = 0.0, 1.0                        # the pinches: the ends of the stations, at
                                                               # every place across at once
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tris = np.vstack((np.column_stack((a.ravel(), b.ravel(), c.ravel())),
                      np.column_stack((a.ravel(), c.ravel(), d.ravel())),
                      np.column_stack((np.full(n_ac - 1, c_le), idx[0, :-1], idx[0, 1:])),
                      np.column_stack((np.full(n_ac - 1, c_te), idx[-1, 1:], idx[-1, :-1]))))
    b_idx = np.r_[c_le, c_te, idx[:, 0], idx[:, -1]]
    b_uv = np.clip(np.vstack(([[0.0, 0.5], [1.0, 0.5]], feet[:, 0], feet[:, 1])), 0.0, 1.0)
    return verts, tris, b_idx, b_uv, idx, (j_ra, j_rb), truth, on, (stretch, moved)


def _on_edges(pts, uv, row0, row1, col0, col1, k_u, k_v):
    """The parameters of the tip mesh's boundary vertices as the tip patch's four edges run: (n, 2).

    pts, uv     the boundary vertices and their parameters as the sections' arcs give them (_tip_mesh)
    row0, row1  the poles of the edges v = 0 and v = 1 (the central panels' top rows, knots k_u)
    col0, col1  those of u = 0 and u = 1 (the strips' top rows, knots k_v)

    The edges are the neighbours' top rows as fitted, and a fitted row does not run through its points at
    exactly the arcs they were sampled at: a vertex can sit several um along its edge from where its
    parameter says (12 um on a wing with a wide TE strip). The patch then has to shear by that much within
    its first knot span off the edge, and its lines double back there. So every vertex takes the parameter
    of the edge's point nearest to it; the corners and the tops of the edge lines stay."""
    out = uv.copy()
    for fixed, value, poles, knots in ((1, 0.0, row0, k_u), (1, 1.0, row1, k_u), (0, 0.0, col0, k_v),
                                       (0, 1.0, col1, k_v)):
        free = 1 - fixed
        sel = np.where((uv[:, fixed] == value) & (uv[:, free] > 0.0) & (uv[:, free] < 1.0)
                       & ((fixed == 1) | (uv[:, free] != 0.5)))[0]
        if not len(sel):
            continue
        t = uv[sel, free].copy()
        for _ in range(6):                                     # Newton on (C(t) - X) . C'(t) = 0
            gap = _design(t, knots) @ poles - pts[sel]
            d1, d2 = _design(t, knots, deriv=1) @ poles, _design(t, knots, deriv=2) @ poles
            slope = np.einsum("ic,ic->i", d1, d1) + np.einsum("ic,ic->i", gap, d2)
            t = np.clip(t - np.einsum("ic,ic->i", gap, d1) / np.where(np.abs(slope) > 1e-300, slope, 1.0), 0.0, 1.0)
        order = np.argsort(uv[sel, free], kind="stable")
        t[order] = np.maximum.accumulate(t[order])             # in the order they had along the edge
        out[sel, free] = t
    return out


def _mean_value_map(verts, tris, bnd_idx, bnd_uv):
    """Parameters (n, 2) of a mesh's vertices with the boundary's held: every inner vertex is the mean value
    (Floater's coordinates, which follow the surface's angles and lengths) of its neighbours, so the map is
    smooth, one to one onto the square, and turns the rim's four smooth corners the way a tensor patch can
    follow."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    n = len(verts)
    p = verts[tris]
    floor = 1.0e-10 * max(float(np.ptp(verts, axis=0).max()), 1e-300)     # the wraps pinch to nothing at the
    rows, cols, vals = [], [], []                                          # tops of the LE and TE lines
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        e1, e2 = p[:, b] - p[:, a], p[:, c] - p[:, a]
        l1 = np.maximum(np.linalg.norm(e1, axis=1), floor)
        l2 = np.maximum(np.linalg.norm(e2, axis=1), floor)
        half = 0.5 * np.arccos(np.clip(np.einsum("ij,ij->i", e1, e2) / (l1 * l2), -1.0, 1.0))
        th = np.tan(np.clip(half, 1e-9, 0.5 * np.pi - 1e-9))
        for other, ln in ((b, l1), (c, l2)):
            rows.append(tris[:, a])
            cols.append(tris[:, other])
            vals.append(th / ln)
    wgt = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    lap = (sp.diags(np.asarray(wgt.sum(axis=1)).ravel()) - wgt).tocsr()
    inner = np.ones(n, dtype=bool)
    inner[bnd_idx] = False
    ii = np.where(inner)[0]
    uv = np.zeros((n, 2))
    uv[bnd_idx] = bnd_uv
    rhs = -(lap[ii][:, bnd_idx] @ bnd_uv)
    lu = splu(lap[ii][:, ii].tocsc())
    uv[ii] = np.column_stack((lu.solve(np.ascontiguousarray(rhs[:, 0])), lu.solve(np.ascontiguousarray(rhs[:, 1]))))
    return uv


def _rim_lines(uv, idx, cols, knots, share=None):
    """The tip's map with the rim laid on two lines of the square: (uv, v of the A side's rim, v of the B
    side's).

    The tip turns on its rim from the straight side into the cap's rounding: its curvature jumps there. A
    smooth patch that such a line crosses at a slant rings beside it (ripples on the flat side); one with
    the line on a double knot does not, it is free to change its curvature there. A patch's lines cannot
    follow the whole rim: the rim is one smooth closed line and touches the patch's edges only at the tops
    of the LE and TE lines. So the rim is laid on the two lines v = v_a and v = v_b, the knots of the patch
    across nearest 0.5 -+ share / 2, from close to one end of the patch to close to the other: the cap's
    lines all start from the stretch `share` of a strip's top row to each side of its edge line, and the
    rim crosses the patch's lines only there, within that stretch of the edge lines' tops, where the knots
    are closest. For that every line u = constant of the map is slid along itself: the side under the rim
    is stretched evenly onto 0 .. v_a (and v_b .. 1), the cap between the rims takes v_a .. v_b, running on
    from the sides with the same speed (the patch stays C1 across the rim: the map v_a .. v_b onto the
    cap's part of the line is a cubic with the sides' slopes at its ends). Where the rim's own place is
    past v_a (at the patch's two ends, where it runs into the tops of the edge lines at v = 0.5), nothing
    is changed; the two join smoothly."""
    from scipy.interpolate import PchipInterpolator
    j_a, j_b = cols
    u, vn = uv[:, 0], uv[:, 1]
    share = RIM_SHARE if share is None else float(share)
    inner = np.unique(knots)[1:-1]
    v_a = float(inner[np.argmin(np.abs(inner - (0.5 - 0.5 * share)))])
    v_b = float(inner[np.argmin(np.abs(inner - (0.5 + 0.5 * share)))])
    soft = 0.25 * min(0.5 - v_a, v_b - 0.5)

    def own(r):
        """The rim's own place over u: v as the smooth function of u through its vertices."""
        x, inv = np.unique(np.round(r[:, 0], 12), return_inverse=True)
        y = np.bincount(inv, weights=r[:, 1]) / np.bincount(inv)
        return PchipInterpolator(x, y, extrapolate=False)(np.clip(u, x[0], x[-1]))

    def larger(g, c):
        """max(g, c), rounded off within soft of where they meet."""
        x = g - c
        return np.where(x <= -soft, c, np.where(x >= soft, g, c + (x + soft) ** 2 / (4.0 * soft)))

    g_a = np.clip(own(uv[idx[:, j_a]]), 1.0e-9, 0.5)
    g_b = np.clip(own(uv[idx[:, j_b]]), 0.5, 1.0 - 1.0e-9)
    t_a, t_b = larger(g_a, v_a), 1.0 - larger(1.0 - g_b, 1.0 - v_b)
    out = vn.copy()
    low, high = vn <= g_a, vn >= g_b
    out[low] = (vn * t_a / g_a)[low]
    out[high] = (1.0 - (1.0 - vn) * (1.0 - t_b) / (1.0 - g_b))[high]
    mid = ~(low | high)
    if mid.any():                                              # the cap: v_n = G(v), G the cubic on t_a .. t_b
        ga, gb, ta, tb, y = g_a[mid], g_b[mid], t_a[mid], t_b[mid], vn[mid]
        h = np.maximum(tb - ta, 1e-300)
        ma, mb = h * ga / ta, h * (1.0 - gb) / (1.0 - tb)

        def cubic(s):
            return ((1.0 + 2.0 * s) * (1.0 - s) ** 2 * ga + s * (1.0 - s) ** 2 * ma
                    + s * s * (3.0 - 2.0 * s) * gb + s * s * (s - 1.0) * mb)

        lo, hi = np.zeros(len(y)), np.ones(len(y))
        for _ in range(60):
            s = 0.5 * (lo + hi)
            up = cubic(s) < y
            lo, hi = np.where(up, s, lo), np.where(up, hi, s)
        out[mid] = ta + 0.5 * (lo + hi) * h
    for j, val in ((j_a, v_a), (j_b, v_b)):                    # the rim itself exactly, where it lies on its line
        ii = idx[:, j]
        t = (t_a if j == j_a else t_b)[ii]
        out[ii[t == val]] = val
    return np.column_stack((u, out)), v_a, v_b


def _locate(uv, tris, un, vn):
    """Where the parameter nodes un x vn lie on the mesh as its map lays it on the square: (the triangle of
    each node, (len(vn), len(un)), -1 for a node no triangle holds; its three weights in it)."""
    q = uv[tris]
    lo, hi = q.min(axis=1), q.max(axis=1)
    i0, i1 = np.searchsorted(un, lo[:, 0], "left"), np.searchsorted(un, hi[:, 0], "right")
    j0, j1 = np.searchsorted(vn, lo[:, 1], "left"), np.searchsorted(vn, hi[:, 1], "right")
    det = ((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
           - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1]))
    ni, nj = i1 - i0, j1 - j0
    cnt = np.where(np.abs(det) > 1e-300, ni * nj, 0)
    sel = np.where(cnt > 0)[0]
    t = np.repeat(sel, cnt[sel])
    off = np.arange(len(t)) - np.repeat(np.cumsum(cnt[sel]) - cnt[sel], cnt[sel])
    ii, jj = i0[t] + off // nj[t], j0[t] + off % nj[t]
    px, py = un[ii] - q[t, 0, 0], vn[jj] - q[t, 0, 1]
    l1 = (px * (q[t, 2, 1] - q[t, 0, 1]) - (q[t, 2, 0] - q[t, 0, 0]) * py) / det[t]
    l2 = ((q[t, 1, 0] - q[t, 0, 0]) * py - px * (q[t, 1, 1] - q[t, 0, 1])) / det[t]
    l0 = 1.0 - l1 - l2
    score = np.minimum(np.minimum(l0, l1), l2)
    ok = np.where(score > -1.0e-9)[0]
    ok = ok[np.argsort(score[ok])]                             # the best fit last: it is the one kept
    tri = np.full((len(vn), len(un)), -1, dtype=int)
    wgt = np.zeros((len(vn), len(un), 3))
    tri[jj[ok], ii[ok]] = t[ok]
    wgt[jj[ok], ii[ok]] = np.column_stack((l0[ok], l1[ok], l2[ok]))
    return tri, wgt


def _tip_data(uv, tris, verts, truth, on, un, vn):
    """The tip's points at the parameter nodes un x vn, (len(vn), len(un), 3): each node's place on the mesh
    (_locate) read on the smooth surface through the grid (truth, a _TipTruth; `on` the vertices' places on
    it), so the points lie on the tip itself, not on the mesh's facets (flat facets between the last wrap
    and the top of an edge line left a kink on that wrap, which the patch followed with its lines doubled
    back). In a triangle at the top of an edge line, which lies at every place across at once, that vertex
    takes the place across of the triangle's other two. A node missed by rounding takes the nearest
    vertex."""
    tri, wgt = _locate(uv, tris, un, vn)
    held = tri >= 0
    corners = tris[np.where(held, tri, 0)]                     # (v, u, 3)
    out = np.einsum("ijk,ijkc->ijc", wgt, verts[corners])
    if truth is not None:
        at = on[corners].copy()                                # (v, u, 3, 2)
        apex = np.isnan(at[:, :, :, 1])
        others = np.where(apex, 0.0, at[:, :, :, 1]).sum(axis=2) / np.maximum((~apex).sum(axis=2), 1)
        at[:, :, :, 1] = np.where(apex, others[:, :, None], at[:, :, :, 1])
        place = np.einsum("ijk,ijkc->ijc", wgt, at)
        smooth = held & (~apex).any(axis=2)
        out[smooth] = truth.at(place[smooth][:, 0], place[smooth][:, 1])
    if not held.all():
        from scipy.spatial import cKDTree
        uu, vv = np.meshgrid(un, vn)
        out[~held] = verts[cKDTree(uv).query(np.column_stack((uu[~held], vv[~held])))[1]]
    return out


def _greville(knots, degree=3):
    n = len(knots) - degree - 1
    return np.array([knots[j + 1:j + degree + 1].mean() for j in range(n)])


def _tensor(a_v, grid, a_u):
    """a_v @ grid @ a_u.T for each coordinate: the surface of a pole grid at the bases' stations, or the
    least-squares poles when the two matrices are pseudo-inverses."""
    return np.stack([a_v @ grid[:, :, c] @ a_u.T for c in range(grid.shape[2])], axis=2)


def _gaps(data, grid, a_v, a_u, d_v, d_u):
    """The distances of data (len(v), len(u), 3) from the patch of a pole grid, measured at the data's own
    parameters along the patch's normal there (a point that only slides along the patch does not count):
    a_v, a_u the bases at the parameters, d_v, d_u their derivatives. 0 where the patch has no normal."""
    res = data - _tensor(a_v, grid, a_u)
    nrm = np.cross(_tensor(a_v, grid, d_u), _tensor(d_v, grid, a_u))
    ln = np.linalg.norm(nrm, axis=2)
    okay = ln > 1.0e-9 * max(float(ln.max()), 1e-300)
    out = np.zeros(ln.shape)
    out[okay] = np.abs(np.einsum("ijc,ijc->ij", res, nrm)[okay] / ln[okay])
    return out


def _normal_gap(data, grid, a_v, a_u, d_v, d_u):
    """The largest of _gaps."""
    return float(_gaps(data, grid, a_v, a_u, d_v, d_u).max())


def _tensor_fit(data, un, vn, k_u, k_v, row0, row1, col0, col1):
    """The pole grid (v poles, u poles, 3) of the tensor B-spline fitted to data (len(vn), len(un), 3) by least
    squares with its four boundary pole rows held: row0 at v = 0, row1 at v = 1, col0 at u = 0, col1 at
    u = 1. The inner poles start from the boundary's blend, so a pole the data leaves free stays in place.
    Returns (poles, the distances of the data from the fitted patch along its normal, mm)."""
    a_u, a_v = _design(un, k_u), _design(vn, k_v)
    gu, gv = _greville(k_u)[None, :, None], _greville(k_v)[:, None, None]
    c00, c10, c01, c11 = row0[0], row0[-1], row1[0], row1[-1]
    poles = ((1.0 - gv) * row0[None] + gv * row1[None] + (1.0 - gu) * col0[:, None] + gu * col1[:, None]
             - ((1.0 - gu) * (1.0 - gv) * c00 + gu * (1.0 - gv) * c10 + (1.0 - gu) * gv * c01 + gu * gv * c11))
    poles[0], poles[-1], poles[:, 0], poles[:, -1] = row0, row1, col0, col1
    res = data - _tensor(a_v, poles, a_u)
    poles[1:-1, 1:-1] += _tensor(np.linalg.pinv(a_v[:, 1:-1]), res, np.linalg.pinv(a_u[:, 1:-1]))
    return poles, _gaps(data, poles, a_v, a_u, _design(vn, k_v, deriv=1), _design(un, k_u, deriv=1))


def _refined(poles, knots, new_knots, degree=3):
    """The poles of the same curve on finer knots (the old ones among them): the curve is unchanged."""
    if len(new_knots) == len(knots):
        return poles
    st = _knot_stations(new_knots, per=5)
    out = np.linalg.lstsq(_design(st, new_knots, degree), _design(st, knots, degree) @ poles, rcond=None)[0]
    out[0], out[-1] = poles[0], poles[-1]
    return out


def _into_plane(first, edge, k_tip, nb_inner, nb_edge, k_nb):
    """The tip patch's first row of poles inside one of its edges, turned so that the patch leaves the edge in
    its neighbour's tangent plane: the two patches then share their tangent plane all along the edge (no
    turn of the normal across it).

    first, edge       the tip patch's first row of poles inside the edge and its row on the edge (on k_tip)
    nb_inner, nb_edge the neighbour's last row of poles but one and its last (the same edge), on k_nb

    The tip patch's step off the edge (first - edge, as a curve along the edge) loses its part along the
    neighbour's normal there; the two end poles, which lie on the tip patch's other edges, stay. Where the
    step then lies flatter along the edge than EDGE_ANGLE, it is turned up to that angle within the tangent
    plane, its length kept: beside the top of an edge line the cap's lines leave the strip's top row nearly
    along it, and the fit's last um would decide on which side (a patch folded there on test wings)."""
    st = _knot_stations(k_tip)
    a_t = _design(st, k_tip)
    along = _design(st, k_nb, deriv=1) @ nb_edge               # the edge's direction
    down = _design(st, k_nb) @ (nb_inner - nb_edge)            # into the neighbour
    nrm = np.cross(along, down)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1.0e-9 * max(float(ln.max()), 1e-300)
    nrm[ok] /= ln[ok][:, None]
    diff = first - edge
    step = a_t @ diff
    step[ok] -= np.einsum("ic,ic->i", step[ok], nrm[ok])[:, None] * nrm[ok]
    tau = along / np.maximum(np.linalg.norm(along, axis=1), 1e-300)[:, None]
    inward = -(down - np.einsum("ic,ic->i", down, tau)[:, None] * tau)         # off the edge, into the tip patch
    inward /= np.maximum(np.linalg.norm(inward, axis=1), 1e-300)[:, None]
    s_t, s_b = np.einsum("ic,ic->i", step, tau), np.einsum("ic,ic->i", step, inward)
    size, turn = np.hypot(s_t, s_b), np.radians(EDGE_ANGLE)
    flat = ok & (s_b < np.sin(turn) * size)
    flat[[0, -1]] = False
    step[flat] = size[flat, None] * (np.cos(turn) * np.sign(s_t[flat])[:, None] * tau[flat]
                                     + np.sin(turn) * inward[flat])
    new = diff.copy()
    new[1:-1] = np.linalg.lstsq(a_t[:, 1:-1], step - np.outer(a_t[:, 0], diff[0]) - np.outer(a_t[:, -1], diff[-1]),
                                rcond=None)[0]
    return edge + new


def _tip_fit(uv, tris, verts, truth, on, k_u, k_v, row0, row1, col0, col1, tol, passes=TIP_PASSES,
             max_poles=TIP_POLES, double=(), inner=None):
    """The tip patch: the tensor B-spline through the tip (truth, the smooth surface through the wraps' grid)
    as the map (uv) of its mesh lays it on the unit square, its four edges the top rows of the patches along
    the span (row0, row1 on the knots k_u; col0, col1 on k_v), held as they are. It starts on those knots;
    where it misses the tip by more than tol the knot spans there are halved, across the direction in which
    the patch bends more over its span, and the fit is repeated (the edges' poles go over to the finer knots
    unchanged as curves), in at most `passes` rounds and up to max_poles poles each way.

    double  knots of k_v taken twice (the lines the tip's rim lies on, _rim_lines: the patch is C1 across
            them)
    inner   {"row0", "row1", "col0", "col1"}: the neighbours' last rows of poles but one. The tip patch's
            first rows of poles inside its edges are then turned into the neighbours' tangent planes
            (_into_plane) and the poles further in fitted again: the tip patch joins the four patches along
            the span without a turn of the normal

    Returns (poles (v, u, 3), u knots, v knots, the largest distance of the tip from the patch, mm)."""
    ku, kv = np.asarray(k_u, dtype=float), np.sort(np.r_[np.asarray(k_v, dtype=float), list(double)])
    k_u, k_v = np.asarray(k_u, dtype=float), np.asarray(k_v, dtype=float)
    for it in range(int(passes) + 1):
        un, vn = _knot_stations(ku)[1:-1], _knot_stations(kv)[1:-1]
        data = _tip_data(uv, tris, verts, truth, on, un, vn)
        poles, gap = _tensor_fit(data, un, vn, ku, kv, _refined(row0, k_u, ku), _refined(row1, k_u, ku),
                                 _refined(col0, k_v, kv), _refined(col1, k_v, kv))
        worst = float(gap.max())
        if worst <= tol or it == passes:
            break
        a_u, a_v = _design(un, ku), _design(vn, kv)
        nrm = np.cross(_tensor(a_v, poles, _design(un, ku, deriv=1)), _tensor(_design(vn, kv, deriv=1), poles, a_u))
        nrm /= np.maximum(np.linalg.norm(nrm, axis=2), 1e-300)[:, :, None]
        uu, vv = np.unique(ku), np.unique(kv)
        iu = np.clip(np.searchsorted(uu, un, "right") - 1, 0, len(uu) - 2)
        iv = np.clip(np.searchsorted(vv, vn, "right") - 1, 0, len(vv) - 2)
        bend_u = np.abs(np.einsum("ijc,ijc->ij", _tensor(a_v, poles, _design(un, ku, deriv=2)), nrm)) \
            * (np.diff(uu)[iu] ** 2)[None, :]
        bend_v = np.abs(np.einsum("ijc,ijc->ij", _tensor(_design(vn, kv, deriv=2), poles, a_u), nrm)) \
            * (np.diff(vv)[iv] ** 2)[:, None]
        bad = gap > tol
        over_u = np.where(bad & (bend_u >= 0.5 * bend_v), gap, 0.0).max(axis=0)       # by u station
        over_v = np.where(bad & (bend_v >= 0.5 * bend_u), gap, 0.0).max(axis=1)       # by v station
        miss_u, miss_v = np.zeros(len(uu) - 1), np.zeros(len(vv) - 1)
        np.maximum.at(miss_u, iu, over_u)
        np.maximum.at(miss_v, iv, over_v)
        room_u, room_v = max(max_poles - (len(ku) - 4), 0), max(max_poles - (len(kv) - 4), 0)
        ju = np.sort(np.argsort(-miss_u)[:room_u])             # the spans that miss most, as many as fit
        jv = np.sort(np.argsort(-miss_v)[:room_v])
        ju, jv = ju[miss_u[ju] > 0.0], jv[miss_v[jv] > 0.0]
        new_u = 0.5 * (uu[ju] + uu[ju + 1])
        new_v = 0.5 * (vv[jv] + vv[jv + 1])
        if not (len(new_u) or len(new_v)):
            break
        ku, kv = np.sort(np.r_[ku, new_u]), np.sort(np.r_[kv, new_v])
    if inner is not None and min(poles.shape[:2]) >= 6:
        rows = (_into_plane(poles[1], poles[0], ku, inner["row0"], row0, k_u),
                _into_plane(poles[-2], poles[-1], ku, inner["row1"], row1, k_u))
        cols = (_into_plane(poles[:, 1], poles[:, 0], kv, inner["col0"], col0, k_v),
                _into_plane(poles[:, -2], poles[:, -1], kv, inner["col1"], col1, k_v))
        poles[1, 1:-1], poles[-2, 1:-1] = rows[0][1:-1], rows[1][1:-1]
        poles[1:-1, 1], poles[1:-1, -2] = cols[0][1:-1], cols[1][1:-1]
        for j, r in ((1, rows[0]), (-2, rows[1])):             # the four poles two edges share: their mean
            for i, c in ((1, cols[0]), (-2, cols[1])):
                poles[j, i] = 0.5 * (r[i] + c[j])
        a_u, a_v = _design(un, ku), _design(vn, kv)
        res = data - _tensor(a_v, poles, a_u)
        poles[2:-2, 2:-2] += _tensor(np.linalg.pinv(a_v[:, 2:-2]), res, np.linalg.pinv(a_u[:, 2:-2]))
        worst = _normal_gap(data, poles, a_v, a_u, _design(vn, kv, deriv=1), _design(un, ku, deriv=1))
    return poles, ku, kv, worst


def _lines_angle(poles, k_u, k_v, per=4):
    """The smallest angle (deg) between a patch's two families of lines, at `per` places in every knot span
    and on its edges; 0 when its normal turns over between two neighbouring places (by more than 135 deg:
    the lines cross there, the patch folds on itself; a sharp edge of the body itself turns it by 90 or
    so)."""
    def places(knots):
        uk = np.unique(knots)
        t = np.arange(int(per)) / float(per)
        return np.r_[(uk[:-1, None] + t[None, :] * np.diff(uk)[:, None]).ravel(), uk[-1]]

    un, vn = places(k_u), places(k_v)
    s_u = _tensor(_design(vn, k_v), poles, _design(un, k_u, deriv=1))
    s_v = _tensor(_design(vn, k_v, deriv=1), poles, _design(un, k_u))
    nrm = np.cross(s_u, s_v)
    ln = np.linalg.norm(nrm, axis=2)
    sine = ln / np.maximum(np.linalg.norm(s_u, axis=2) * np.linalg.norm(s_v, axis=2), 1e-300)
    unit = nrm / np.maximum(ln, 1e-300)[:, :, None]
    along = float(min((unit[:, 1:] * unit[:, :-1]).sum(axis=2).min(), (unit[1:] * unit[:-1]).sum(axis=2).min()))
    if along < -np.sqrt(0.5):
        return 0.0
    return float(np.degrees(np.arcsin(min(float(sine.min()), 1.0))))


def _half_cut(solid, loops, thick, cen_a_row):
    """The half body: the solid cut by the plane of symmetry, keeping the
    LE-to-TE side (the side of the loops' indices 0 to i_te). The six faces
    keep the full body's surfaces, trimmed; the section face closes it."""
    BRepPrimAPI_MakeBox = _occ("BRepPrimAPI", "BRepPrimAPI_MakeBox")
    gp_Pnt = _occ("gp", "gp_Pnt")
    Common = _occ("BRepAlgoAPI", "BRepAlgoAPI_Common")
    pts = np.vstack(loops)
    lo = pts.min(axis=0) - (0.05 * np.ptp(pts, axis=0) + 10.0)
    hi = pts.max(axis=0) + (0.05 * np.ptp(pts, axis=0) + 10.0)
    if float(np.mean(cen_a_row[:, thick])) >= 0.0:
        lo[thick] = 0.0
    else:
        hi[thick] = 0.0
    algo = Common(solid, BRepPrimAPI_MakeBox(gp_Pnt(*[float(v) for v in lo]),
                                             gp_Pnt(*[float(v) for v in hi])).Shape())
    algo.Build()
    if not algo.IsDone():
        raise RuntimeError("the half body's cut on the plane of symmetry failed")
    return algo.Shape()


def _sew_solid(parts, sew_tol):
    """The faces sewn into one shell and made a solid: (solid, free edges)."""
    BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid = _occ("BRepBuilderAPI", "BRepBuilderAPI_Sewing",
                                                           "BRepBuilderAPI_MakeSolid")
    sew = BRepBuilderAPI_Sewing(sew_tol)
    for p in parts:
        sew.Add(p)
    sew.Perform()
    free = int(sew.NbFreeEdges())
    TopExp_Explorer = _occ("TopExp", "TopExp_Explorer")
    TopAbs_SHELL = _occ("TopAbs", "TopAbs_SHELL")
    ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    shells = []
    while ex.More():
        shells.append(_topods("Shell")(ex.Current()))
        ex.Next()
    if len(shells) != 1:
        raise RuntimeError(f"sewing left {len(shells)} shells, expected 1")
    ShapeFix_Shell, ShapeFix_Solid = _occ("ShapeFix", "ShapeFix_Shell", "ShapeFix_Solid")
    fs = ShapeFix_Shell(shells[0])
    fs.Perform()
    solid = BRepBuilderAPI_MakeSolid(fs.Shell()).Solid()
    fx = ShapeFix_Solid(solid)
    fx.Perform()
    return fx.Solid(), free


def _symmetric(levels, thick, sym_tol):
    """Refuse a half body when the levels (_SmoothLoop) are not symmetric about the plane of symmetry
    (thickness 0) within sym_tol mm."""
    sym = 0.0
    for lo in levels:
        mirror = lo.dense[lo.it:][::-1].copy()
        mirror[:, thick] *= -1.0
        sym = max(sym, float(np.abs(mirror - lo.dense[:lo.it + 1]).max()))
    if sym > sym_tol:
        raise ValueError(f"no half body: the loops are not symmetric about the plane of symmetry "
                         f"(they differ by {sym:.3g} mm, the allowance is {sym_tol:g} mm)")


def _strip_widths(levels, chord_root, fr):
    """How far the LE strip and the TE strip reach to each side of their edge (mm), whether the TE strip is
    the TE's round, and the words for the notes: as wide as the edges' rounding of the horizontal levels
    (_SmoothLoop; fr None: the median of the sections that have one) or the fractions fr of the root chord.
    A TE strip much wider than the TE's rounding is narrowed to the rounding (one fraction given): a small
    round in the middle of a wide patch is more than a viewer's or a mesher's triangles of the patch
    resolve."""
    def typical(vals):
        """The rounding of the sections: the median of those that have one."""
        vals = np.array([v for v in vals if v > 0.0])
        return float(np.median(vals)) if 2 * len(vals) > len(levels) else 0.0

    te_round = False
    if fr is None:
        r_le = typical(_edge_rounding(lo, 0.0, LE_WINDOW * min(lo.s_te, lo.S - lo.s_te)) for lo in levels)
        r_te = typical(_edge_rounding(lo, lo.s_te, 0.07 * chord_root, TE_DROP) for lo in levels)
        w_le = LE_MARGIN * r_le if r_le > 0.0 else 0.07 * chord_root
        w_te = TE_MARGIN * r_te if r_te > 0.0 else 0.07 * chord_root
        te_round = r_te > 0.0
        note = (f"the strips {w_le:.6g} mm wide about the LE and {w_te:.6g} mm about the TE (their "
                f"rounding: {r_le:.4g} and {r_te:.4g} mm)")
    else:
        w_le, w_te = float(fr[0]) * chord_root, float(fr[-1]) * chord_root
        if fr.size == 1:
            r_te = typical(_edge_rounding(lo, lo.s_te, w_te, TE_DROP) for lo in levels)
            if 0.0 < TE_MARGIN * r_te < ROUND_SMALL * w_te:
                w_te, te_round = TE_MARGIN * r_te, True
        note = (f"the strips {w_le:.6g} mm wide about the LE and the TE" if w_le == w_te else
                f"the strips {w_le:.6g} mm wide about the LE and {w_te:.6g} mm about the TE"
                + (" (the TE's rounding)" if te_round else ""))
    return w_le, w_te, te_round, note


def loft_loops(loops, n_poles=40, height_axis=2, flat_tol=1.0e-6, sew_tol=1.0e-3, v_poles=30,
               strip_frac=None, half=False, sym_tol=0.2, ease="both", tip_drop=TIP_DROP, widths=None):
    """The solid through closed loops (each (n, 3), mm, the same point count,
    LE first), bottom to top, in five patches and the flat root; see the
    module docstring for how it is built.

    n_poles     B-spline poles along a section between the strips, before the
                knots added towards the strips (30 more); the strips take
                min(n_poles, 60) and 27 more
    v_poles     rows of poles along the span to start with; fewer rows than
                loops smooth extraction noise out, and rows are added where a
                sharp turn of the planform needs them (_span_fit)
    strip_frac  None (default): each strip is as wide as its edge's rounding:
                the LE strip LE_MARGIN times the arc from the LE to where the
                section's curvature has fallen to a quarter of its peak
                there, the TE strip the arc from the TE to where it has
                fallen to a tenth (the TE's round itself; the sections'
                median; an edge without a rounding, a sharp TE: 0.07 of the
                root chord). A fraction: each strip reaches that fraction of
                the root chord along the section's arc to both sides of its
                edge (the blade's constant-width rule), a TE strip more than
                twice as wide as the TE's round narrowed to the round. A
                pair (LE, TE) gives the two strips their own fractions, used
                as given
    tip_drop    under a cap: the slope of the strips' top edges in the side
                view. The tip patch's four corners, the tops of the strip
                lines, sit under the top of the LE line and of the TE line by
                tip_drop times their distance from that top along the chord
                (1: the top edges at 45 deg; a pair: the LE's and the TE's)
    half        True: the half body on the LE-to-TE side of the plane of
                symmetry (thickness 0), the loops symmetric within sym_tol (mm)
    ease        which loops give way at a step from one horizontal loop to
                the next (the one where modified sections meet the original
                ones; _ease_flats): "both" (the loops under it and over it,
                so that the surface tilts alike on both sides), "below" (the
                loops under it; those above are kept as they are) or "above"
    widths      None, or the strips' reach (LE, TE) in mm, used as given in
                place of strip_frac's (loft_split gives its two lofts the
                whole stack's, so they meet on the same seams)
    """
    if ease not in ("both", "below", "above"):
        raise ValueError('ease is "both", "below" or "above"')
    t0 = time.time()
    loops = [np.asarray(lp, dtype=float).copy() for lp in loops]
    if len(loops) < 2:
        raise ValueError("a loft needs at least two loops")
    if len({len(lp) for lp in loops}) != 1:
        raise ValueError("every loop must have the same number of points")
    thick = 1 if height_axis == 2 else 2
    z_axis = height_axis
    n = len(loops)
    n_pts = len(loops[0])
    i_te = (n_pts + 1) // 2 - 1
    size = max(np.abs(lp).max() for lp in loops)
    for lp in loops:                                   # the seams onto the plane of symmetry
        for i in (0, i_te, n_pts - 1):
            if abs(lp[i, thick]) < 1.0e-3:
                lp[i, thick] = 0.0
    flat = [np.ptp(lp[:, z_axis]) <= flat_tol * max(1.0, size) for lp in loops]
    if not flat[0]:
        raise ValueError("the root loop must be horizontal (the root is closed with a plane)")
    k = next((i for i, f in enumerate(flat) if not f), n)          # leading loops at one height each
    kr = k                                                         # the rim: the first loop that is not
    if 1 < k < n:                                                  # horizontal, or the last horizontal one
        near = 1.0e-6 * max(1.0, size)                             # when the loops above run through its LE
        if all(np.linalg.norm(loops[k][i] - loops[k - 1][i]) <= near for i in (0, i_te)):   # and TE points
            kr = k - 1                                             # (a cap's rings on a horizontal foot)
    chord_root = float(np.linalg.norm(loops[0][0] - loops[0][i_te]))
    fr = None
    if strip_frac is not None:
        fr = np.atleast_1d(np.asarray(strip_frac, dtype=float))
        if fr.size not in (1, 2) or np.any(fr <= 0.0):
            raise ValueError("strip_frac is None (the strips as wide as the edges' rounding needs), one "
                             "positive fraction, or a pair (LE, TE)")
    td = np.atleast_1d(np.asarray(tip_drop, dtype=float))
    if td.size not in (1, 2) or np.any(td <= 0.0):
        raise ValueError("tip_drop is one positive slope, or a pair (LE, TE)")
    notes = []
    span = float(max(lp[:, z_axis].max() for lp in loops) - loops[0][0, z_axis])

    # a step between horizontal loops is taken out of them first: no smooth surface follows one
    given = loops
    eased, steps = _ease_flats(loops[:k], z_axis, v_poles, SPAN_TOL * span, ease)
    if steps:
        loops = eased + loops[k:]

    # the walls' loops as their fit along the span gives them: one smooth surface for the rows, the lines up
    # the body and the tip patch's sides
    wall, wall_err = _wall_loops(loops[:k], z_axis, v_poles, SPAN_TOL * span)
    loops = wall + loops[k:]

    # the levels the four patches along the span are sampled on: every horizontal loop and, under a cap, its
    # rim (the first loop that is not horizontal: a tilted tip section, a cap's first ring)
    def levels_of(loops):
        return [_SmoothLoop(lp, i_te) for lp in loops[:kr + 1]]

    smooth = levels_of(loops)
    zf = np.array([lp[0, z_axis] for lp in loops[:kr]])

    # the strips: as wide as their edges' rounding needs, or the fractions given (_strip_widths), or as
    # loft_split gives both its lofts
    if widths is None:
        w_le, w_te, te_round, strips_note = _strip_widths(smooth[:kr], chord_root, fr)
    else:
        w_le, w_te = (float(v) for v in widths)
        te_round = False
        strips_note = f"the strips {w_le:.6g} mm wide about the LE and {w_te:.6g} mm about the TE (as given)"
    slopes = (float(td[0]), float(td[-1]))

    # per level the four patches' sample rows at shared split points; under a cap the rows are turned onto
    # the tip patch's edges (_tip_rows)
    k_cen = _refined_knots(n_poles, "central")
    k_strip = _refined_knots(max(12, min(int(n_poles), 60)), "strip")
    u_cen, u_strip = _knot_stations(k_cen), _knot_stations(k_strip)
    a_cen, a_strip = _design(u_cen, k_cen), _design(u_strip, k_strip)
    d_cen, d_strip = _design(u_cen, k_cen, deriv=1), _design(u_strip, k_strip, deriv=1)
    pinv = {"cen": np.linalg.pinv(a_cen[:, 1:-1]), "strip": np.linalg.pinv(a_strip[:, 1:-1])}
    mid = int(np.argmin(np.abs(u_strip - 0.5)))

    def rows_of(levels):
        grid = {nm: [] for nm in NAMES}
        for lo in levels:
            for nm, row in zip(NAMES, _row_arcs(lo, w_le, w_te, u_strip, u_cen)):
                grid[nm].append(row)
        grid = {nm: np.array(rows) for nm, rows in grid.items()}
        if k == n:
            return {nm: list(rows) for nm, rows in grid.items()}, None
        return _tip_rows(grid, zf, z_axis, thick, mid, slopes, 2.0e-6 * size)

    rows, tip_info = rows_of(smooth)
    n_rows = len(rows["le"])
    drops = tip_info["drops"] if k < n else (0.0, 0.0)
    if k < n:
        notes.append(f"the tip patch's corners sit {drops[0]:.5g} and {drops[1]:.5g} mm under the tops of the "
                     f"LE and TE lines, on the strip lines; {strips_note}; the rows turn onto the patches' top "
                     f"edges over the top {tip_info['band']:.4g} mm under them")
    else:
        notes.append(f"all sections horizontal: the flat top closed by a plane; {strips_note}")

    if half:
        _symmetric(smooth, thick, sym_tol)

    # the rows' poles: each arc fitted with its ends held, so neighbouring patches share their seam poles
    fams = {nm: [] for nm in NAMES}
    spacing, err, last = [], 0.0, None
    for r in range(n_rows):
        for name, a, key in (("le", a_strip, "strip"), ("cen_a", a_cen, "cen"), ("te", a_strip, "strip"),
                             ("cen_b", a_cen, "cen")):
            row = rows[name][r]
            poles = np.empty((a.shape[1], 3))
            poles[0], poles[-1] = row[0], row[-1]
            poles[1:-1] = pinv[key] @ (row - np.outer(a[:, 0], row[0]) - np.outer(a[:, -1], row[-1]))
            err = max(err, float(np.abs(a @ poles - row).max()))
            fams[name].append(poles)
        stacked = np.vstack([rows[nm][r] for nm in NAMES])
        if last is not None:
            spacing.append(float(np.linalg.norm(stacked - last, axis=1).mean()))
        last = stacked
    first_a = rows["cen_a"][0]
    data = rows_of(levels_of(given))[0] if steps else rows     # the distances reported: the loops as given

    # along the span: every family on the same parameters and knots, so the shared seams are the same curves
    vpar = np.r_[0.0, np.cumsum(np.maximum(spacing, 1e-9 * max(max(spacing), 1e-12)))]
    vpar /= vpar[-1]
    grids, vk, vd, _ = _span_fit(fams, v_poles, vpar, SPAN_TOL * span)
    b_v, d_v = _design(vpar, vk, vd), _design(vpar, vk, vd, deriv=1)
    v_err = 0.0
    for name, a, d in (("le", a_strip, d_strip), ("cen_a", a_cen, d_cen), ("te", a_strip, d_strip),
                       ("cen_b", a_cen, d_cen)):
        v_err = max(v_err, _normal_gap(np.array(data[name]), grids[name], b_v, a, d_v, d))
    v_err = max(v_err, wall_err)
    notes.append(f"along the span {len(vk) - vd - 1} rows of poles through {n_rows} rows; the rows "
                 f"within {v_err:.3g} mm of the patches (the rows off the loops' fit along the span, "
                 f"which lies within {wall_err:.3g} mm of the loops)")
    for at, jump, lo, hi, s_lo, s_hi in steps:
        notes.append(f"a step of {jump:.3g} mm between loops {at} and {at + 1}, eased out of the loops "
                     + (f"over {lo:.3g} mm below it (the loops above kept)" if s_hi == 0.0 else
                        f"over {hi:.3g} mm above it (the loops below kept)" if s_lo == 0.0 else
                        f"over {lo:.3g} mm below it ({100 * s_lo:.0f} % of it) and {hi:.3g} mm above "
                        f"({100 * s_hi:.0f} %)")
                     + ": the patches cross it smoothly")

    # the tip patch: the sides above the four patches' top rows and the cap, fitted through the map of
    # their mesh onto the patch's square
    tip, tip_err, tip_angle, tip_blend, k_tu, k_tv = None, 0.0, 90.0, (0.0, 0.0), k_cen, k_strip
    if k < n:
        js = tip_info["js"]
        verts, tris, b_idx, b_uv, idx, cols, truth, on, tip_blend = _tip_mesh(
            loops[js:kr], loops[kr:], [(lo.pt_arc, lo.s_te) for lo in smooth[js:]], i_te, z_axis, tip_info,
            (w_le, w_te))
        b_uv = _on_edges(verts[b_idx], b_uv, grids["cen_a"][-1], grids["cen_b"][-1], grids["le"][-1],
                         grids["te"][-1], k_cen, k_strip)
        uv = _mean_value_map(verts, tris, b_idx, b_uv)
        uv, v_a, v_b = _rim_lines(uv, idx, cols, k_strip)
        chord_tip = float(np.linalg.norm(tip_info["x_le"] - tip_info["x_te"]))
        tip, k_tu, k_tv, tip_err = _tip_fit(uv, tris, verts, truth, on, k_cen, k_strip, grids["cen_a"][-1],
                                            grids["cen_b"][-1], grids["le"][-1], grids["te"][-1],
                                            TIP_TOL * chord_tip, double=(v_a, v_b),
                                            inner={"row0": grids["cen_a"][-2], "row1": grids["cen_b"][-2],
                                                   "col0": grids["le"][-2], "col1": grids["te"][-2]})
        tip_angle = _lines_angle(tip, k_tu, k_tv)
        notes.append(f"the tip patch within {tip_err:.3g} mm of the tip ({tip.shape[1]} x {tip.shape[0]} poles); "
                     + (f"its two families of lines at {tip_angle:.3g} deg or more to each other (no fold)"
                        if tip_angle > 0.0 else "ITS LINES CROSS SOMEWHERE: the patch folds on itself")
                     + (f"; the cap's rounding blended over {tip_blend[0]:.2g} mm ({CAP_BLEND:.3g} of its "
                        f"tightest radius), which moved the cap by up to {tip_blend[1]:.2g} mm"
                        if tip_blend[1] > 0.0 else ""))
        v_err = max(v_err, tip_err)

    # the faces
    edges = (("le", k_strip), ("cen_a", k_cen), ("te", k_strip), ("cen_b", k_cen))
    parts = [_surface_face(grids[name], uk, vk, 3, vd) for name, uk in edges]
    if k < n:
        parts.append(_surface_face(tip, k_tu, k_tv, 3, 3))
    else:
        parts.append(_wire_face([_bspline_edge(grids[nm][-1], uk, 3) for nm, uk in edges]))
    parts.append(_wire_face([_bspline_edge(grids[nm][0], uk, 3) for nm, uk in edges]))

    solid, free = _sew_solid(parts, sew_tol)
    if half:                                           # cut on the plane of symmetry: the surfaces stay the
        solid = _half_cut(solid, loops, thick, first_a)              # full body's, the volume exactly half
        notes.append("half body: the LE-to-TE side, cut on the plane of symmetry")
    BRepCheck_Analyzer = _occ("BRepCheck", "BRepCheck_Analyzer")
    valid = bool(BRepCheck_Analyzer(solid).IsValid())
    vol, area = mass_properties(solid)
    return LoftResult(solid, valid, vol, area, free, err, k, n - k, time.time() - t0, notes,
                      count_faces(solid), v_err, bool(half), steps, (w_le, w_te), te_round,
                      drops, float(tip_err),
                      (int(tip.shape[1]), int(tip.shape[0])) if tip is not None else (0, 0), tip_angle,
                      None if tip_info is None else {kk: np.array(vv) for kk, vv in tip_info["corners"].items()},
                      (float(tip_blend[0]), float(tip_blend[1])))


def _box(shape):
    """The shape's bounding box, (2, 3): the surfaces, not their control nets."""
    Bnd_Box = _occ("Bnd", "Bnd_Box")
    mod = importlib.import_module(f"{_PKG}.BRepBndLib")
    b = Bnd_Box()
    _static(getattr(mod, "brepbndlib", None) or getattr(mod, "BRepBndLib"), "AddOptimal")(shape, b, False, False)
    return np.array(b.Get(), dtype=float).reshape(2, 3)


def loft_split(loops, split, n_poles=40, height_axis=2, flat_tol=1.0e-6, sew_tol=1.0e-3, v_poles=30,
               strip_frac=None, half=False, sym_tol=0.2, ease="both", tip_drop=TIP_DROP):
    """The solid through the loops as two lofts that meet at loop `split` (counted from 0; horizontal, as
    are all the loops under it): the loops up to it and the loops from it on, each lofted as loft_loops
    does, on its own. The surface may then turn at that loop, as the loops do: a crease along the section
    (a rudder modified below H whose curves leave the original there without its slope). One loft through
    all the loops smooths such a turn over the height (its walls and patches are fitted along the span),
    takes it for steps and ripples about it.

    Both lofts run through loop `split` exactly (loft_loops holds the first and the last loop) and have the
    same strips (as wide as the whole stack's edges need, or the fractions of the root chord given), so
    they end on the same four curves there, and the seams run on across it. The lower loft's four patches,
    the upper loft's four and its tip patch and the flat root are sewn into one solid; the two flat faces on
    the joint are left out: nine patches and the root, ten faces (an upper part without a cap: its flat top,
    eleven). half: that solid cut on the plane of symmetry, as loft_loops cuts its own.

    Returns one LoftResult: the solid; the notes of the two lofts (marked below and above the joint); steps
    counted on the whole stack; the tip's numbers from the upper loft; joint = split + 1."""
    t0 = time.time()
    loops = [np.asarray(lp, dtype=float).copy() for lp in loops]
    n, split = len(loops), int(split)
    if not 0 < split < n - 1:
        raise ValueError(f"the joint must leave loops on both sides: split {split} of {n} loops (0 .. {n - 1})")
    if len({len(lp) for lp in loops}) != 1:
        raise ValueError("every loop must have the same number of points")
    thick, z_axis = (1 if height_axis == 2 else 2), height_axis
    n_pts = len(loops[0])
    i_te = (n_pts + 1) // 2 - 1
    size = max(np.abs(lp).max() for lp in loops)
    for lp in loops:                                   # the seams onto the plane of symmetry, as loft_loops
        for i in (0, i_te, n_pts - 1):
            if abs(lp[i, thick]) < 1.0e-3:
                lp[i, thick] = 0.0
    flat = [np.ptp(lp[:, z_axis]) <= flat_tol * max(1.0, size) for lp in loops]
    if not all(flat[:split + 1]):
        raise ValueError("the loops up to the joint must all be horizontal")
    k = next((i for i, f in enumerate(flat) if not f), n)

    # the strips of both lofts: the whole stack's, as one loft would have them
    levels = [_SmoothLoop(lp, i_te) for lp in loops[:k]]
    fr = None if strip_frac is None else np.atleast_1d(np.asarray(strip_frac, dtype=float))
    if fr is not None and (fr.size not in (1, 2) or np.any(fr <= 0.0)):
        raise ValueError("strip_frac is None (the strips as wide as the edges' rounding needs), one positive "
                         "fraction, or a pair (LE, TE)")
    chord_root = float(np.linalg.norm(loops[0][0] - loops[0][i_te]))
    w_le, w_te, te_round, strips_note = _strip_widths(levels, chord_root, fr)
    if half:
        _symmetric(levels, thick, sym_tol)

    kw = dict(n_poles=n_poles, height_axis=height_axis, flat_tol=flat_tol, sew_tol=sew_tol, v_poles=v_poles,
              ease=ease, tip_drop=tip_drop, widths=(w_le, w_te))
    z_j = float(loops[split][0, z_axis])
    try:
        upper = loft_loops(loops[split:], **kw)                     # the upper first: it is the one that can fail
    except ValueError as exc:
        if "do not leave room" in str(exc):                         # _tip_rows: the joint under the tip patch
            raise ValueError(f"the joint at z {z_j:.6g} mm lies too close to the tip: the loft above it (loops "
                             f"{split + 1} to {n}) has no room for its tip patch, whose corners sit under the "
                             f"tops of the LE and TE lines by tip_drop times their distance from them along the "
                             f"chord; lower the joint") from exc
        raise ValueError(f"the loft above the joint (loops {split + 1} to {n}): {exc}") from exc
    lower = loft_loops(loops[:split + 1], **kw)

    # the faces of both but the flat ones on the joint, sewn into one solid
    tol = max(1.0e-6 * max(1.0, size), 10.0 * flat_tol * max(1.0, size))
    parts, dropped = [], 0
    for res in (lower, upper):
        for f in _faces_of(res.shape):
            box = _box(f)
            if abs(box[0, z_axis] - z_j) <= tol and abs(box[1, z_axis] - z_j) <= tol:
                dropped += 1
                continue
            parts.append(f)
    if dropped != 2:
        raise RuntimeError(f"the two lofts have {dropped} flat faces on the joint, not 2")
    solid, free = _sew_solid(parts, sew_tol)
    if half:
        solid = _half_cut(solid, loops, thick, loops[0][1:i_te])
    BRepCheck_Analyzer = _occ("BRepCheck", "BRepCheck_Analyzer")
    valid = bool(BRepCheck_Analyzer(solid).IsValid())
    vol, area = mass_properties(solid)

    notes = [f"two lofts meeting at loop {split + 1} (z {z_j:.6g} mm): the {split + 1} loops up to it and the "
             f"{n - split} from it on, each lofted on its own, so the surface turns there as the loops do; both "
             f"through that loop with the same strips ({strips_note}), the lower loft's four patches, the "
             f"upper's four and its tip patch and the flat root sewn into one solid (the joint's two flat faces "
             f"left out)"]
    notes += ["below the joint: " + t for t in lower.notes if not t.startswith("all sections horizontal")]
    notes += ["above the joint: " + t for t in upper.notes]
    if half:
        notes.append("half body: the LE-to-TE side, cut on the plane of symmetry")
    steps = list(lower.steps) + [(at + split, *rest) for at, *rest in upper.steps]
    return LoftResult(solid, valid, vol, area, free, max(lower.fit_error, upper.fit_error),
                      lower.n_smooth + upper.n_smooth - 1, upper.n_ruled, time.time() - t0, notes,
                      count_faces(solid), max(lower.v_error, upper.v_error), bool(half), steps, (w_le, w_te),
                      te_round, upper.drops, upper.tip_error, upper.tip_poles, upper.tip_angle, upper.corners,
                      upper.tip_blend, split + 1)


# --------------------------------------------------------------------------
# Files and display
# --------------------------------------------------------------------------

def write_step(shape, path):
    STEPControl_Writer, STEPControl_AsIs = _occ("STEPControl", "STEPControl_Writer", "STEPControl_AsIs")
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if int(w.Write(str(path))) != 1:                       # IFSelect_RetDone
        raise RuntimeError(f"writing {path} failed")
    return path


def read_step(path):
    STEPControl_Reader = _occ("STEPControl", "STEPControl_Reader")
    r = STEPControl_Reader()
    if int(r.ReadFile(str(path))) != 1:
        raise RuntimeError(f"reading {path} failed")
    r.TransferRoots()
    return r.OneShape()


def write_iges(shape, path):
    """Write a shape as IGES in mm, its faces as trimmed surfaces (entity 144, the form CAD and meshing
    tools read)."""
    IGESControl_Writer = _occ("IGESControl", "IGESControl_Writer")
    w = IGESControl_Writer("MM", 0)                      # 0: faces as trimmed surfaces (1: BRep solids)
    w.AddShape(shape)
    w.ComputeModel()
    if not w.Write(str(path)):
        raise RuntimeError(f"writing {path} failed")
    return path


def read_iges(path):
    IGESControl_Reader = _occ("IGESControl", "IGESControl_Reader")
    r = IGESControl_Reader()
    if int(r.ReadFile(str(path))) != 1:
        raise RuntimeError(f"reading {path} failed")
    r.TransferRoots()
    return r.OneShape()


CAD_EXTENSIONS = {".step": "STEP", ".stp": "STEP", ".igs": "IGES", ".iges": "IGES"}


def save_cad(step, path, iges=None):
    """A built STEP file written to path, by its extension (CAD_EXTENSIONS): STEP a copy of it; IGES a copy
    of iges (a build's own IGES: the blade's) when that exists, else the STEP converted (write_iges). Returns
    how it was written."""
    import shutil
    kind = CAD_EXTENSIONS.get(os.path.splitext(str(path))[1].lower())
    if kind is None:
        raise ValueError(f"{path}: a .step, .stp, .igs or .iges file")
    src = step if kind == "STEP" else iges if iges and os.path.exists(iges) else None
    if src is None:
        write_iges(read_step(step), path)
        return f"IGES converted from {os.path.basename(step)}"
    if not (os.path.exists(path) and os.path.samefile(src, path)):
        shutil.copyfile(src, path)
    return f"{kind}, a copy of {os.path.basename(src)}"


def tessellate(shape, linear=0.5, angular=0.5):
    """Triangles of a shape for display: (vertices (n, 3), triangles (m, 3)),
    outward-facing (reversed faces flipped)."""
    BRepMesh_IncrementalMesh = _occ("BRepMesh", "BRepMesh_IncrementalMesh")
    TopExp_Explorer = _occ("TopExp", "TopExp_Explorer")
    TopAbs_FACE, TopAbs_REVERSED = _occ("TopAbs", "TopAbs_FACE", "TopAbs_REVERSED")
    BRep_Tool = _occ("BRep", "BRep_Tool")
    TopLoc_Location = _occ("TopLoc", "TopLoc_Location")
    triangulation = _static(BRep_Tool, "Triangulation")
    as_face = _topods("Face")
    BRepMesh_IncrementalMesh(shape, float(linear), False, float(angular), True)
    verts, tris, base = [], [], 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        face = as_face(ex.Current())
        loc = TopLoc_Location()
        tri = triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            n = tri.NbNodes()
            pts = [tri.Node(i).Transformed(trsf) for i in range(1, n + 1)]
            verts.append(np.array([[p.X(), p.Y(), p.Z()] for p in pts]))
            t = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
            if face.Orientation() == TopAbs_REVERSED:
                t = t[:, ::-1]
            tris.append(t + base)
            base += n
        ex.Next()
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.vstack(verts), np.vstack(tris).astype(int)


def view(shape, title="XGeom CAD"):
    """Show a shape in the pythonocc viewer (needs pythonocc-core and a Qt or
    wx backend; blocks until the window is closed)."""
    if _PKG != "OCC.Core":
        raise ImportError("the viewer needs pythonocc-core (OCC.Display)")
    from OCC.Display.SimpleGui import init_display
    display, start_display, _, _ = init_display()
    display.DisplayShape(shape, update=True)
    display.FitAll()
    print(f"{title}: close the viewer window to go on")
    start_display()


def main():
    ap = argparse.ArgumentParser(description="Loft the loops of an XCAD point file into a CAD solid.")
    ap.add_argument("xcad", nargs="?", help="XCAD point file (loops starting at the LE)")
    ap.add_argument("--units", choices=("m", "mm"), default="m", help="units of the point file (default m)")
    ap.add_argument("--step", default=None, help="STEP file to write (default: next to the point file)")
    ap.add_argument("--poles", type=int, default=40, help="B-spline poles per side of a section (default 40)")
    ap.add_argument("--half", action="store_true", help="the half body, cut at the plane of symmetry")
    ap.add_argument("--ease", choices=("both", "below", "above"), default="both",
                    help="the loops that give way at a step between horizontal loops, such as modified "
                         "sections meeting the original ones (default both: under it and over it)")
    ap.add_argument("--view", action="store_true", help="show the solid in the pythonocc viewer")
    ap.add_argument("--view-step", default=None, help="only show this STEP file")
    args = ap.parse_args()
    if args.view_step:
        view(read_step(args.view_step), os.path.basename(args.view_step))
        return
    if not args.xcad:
        ap.error("give an XCAD point file or --view-step")
    res = loft_loops(read_loops(args.xcad, args.units), args.poles, half=args.half, ease=args.ease)
    step = args.step or os.path.splitext(args.xcad)[0] + ".step"
    write_step(res.shape, step)
    print(f"{args.xcad}: {res.n_smooth} horizontal + {res.n_ruled} tip loops, {res.faces} faces"
          f"{' (half body)' if res.half else ''}, valid {res.valid}, volume {res.volume:.1f} mm^3, "
          f"area {res.area:.1f} mm^2, open edges {res.free_edges}, the rows' curves within "
          f"{res.fit_error * 1e3:.1f} um, the loops and the cap within {res.v_error * 1e3:.1f} um of the patches, "
          f"{res.seconds:.1f} s -> {step}")
    for note in res.notes:
        print(f"  {note}")
    if args.view:
        view(res.shape, os.path.basename(step))


if __name__ == "__main__":
    main()
