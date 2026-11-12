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
central panels between them, and the tip patch. The four patches along the
span run from the root to the edge of the tip; the tip patch covers only the
tip face above that edge (user, 11 Nov 2026: the layout before gave the tip
patch the top 15 % of the span, as the blade's decomposition does, and its
fit rippled there).

- The rows. Every horizontal loop is a row of the four patches; under a cap
  the last row is the rim, the first loop that is not horizontal (a tilted
  tip section, the first ring of a cap). Because the rim's points sit at
  different heights, the horizontal loops in a band under it are re-cut to
  tilt onto it smoothly, each point moving along its own track through the
  loops (_fan_rows), so the rows turn without a jump while the surface stays
  the one the loops describe.
- Each row is sampled from the smooth closed curve through its points (a
  cubic spline, not their polygon) and split into four arcs at shared arc
  stations: the strips reach strip_frac of the root chord to both sides of
  the LE and of the TE (the blade's constant-width rule; a pair gives the LE
  strip and the TE strip their own), the centrals lie between them. Each arc
  is fitted with held ends, so neighbouring patches
  share their seam poles row by row, and each family is fitted along the
  span on common parameters and knots: identical boundary data, parameters
  and knots make the shared boundary curves identical, so the faces sew
  watertight.
- A small rounded TE gets the TE strip to itself (user, 12 Nov 2026: the TE
  showed a row of teeth in the mesher). With one strip_frac given, a TE
  strip more than twice as wide as the TE's rounding is narrowed to it: 1.5
  times the arc from the TE to where the curvature has fallen to a quarter
  of its peak (_edge_rounding; the wind-tunnel rudder's 0.75 mm round: 1.7
  mm to each side instead of 12.6). The surface was smooth either way; but a
  viewer or a mesher draws a patch with triangles sized for the patch, and
  a round a tenth of the patch wide in its middle comes out as teeth along
  the edge. As its own narrow patch between two straight edges, as the TE
  round is in the original CAD, it is drawn clean. A pair is used as given.
- The tip patch is one grid over the cap: its u = 0 and u = 1 edges are the
  strips' top rows, its v = 0 and v = 1 edges the central panels' top rows
  (the pole rows are shared verbatim). The cap is meshed from the rim, the
  loops above it and the ridge (one smooth curve per chord station), the
  mesh is mapped onto the patch's unit square with the rim's four arcs on
  the square's sides (mean value coordinates: smooth and one to one), and
  the grid is the least-squares tensor fit to the map's inverse. The knots
  are denser towards the patch's edges, where the tip's wall and edge
  rounding sit.
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
- half=True cuts the sewn full body on the plane of symmetry and keeps the
  LE-to-TE side: the same surfaces, trimmed, closed by one planar face. The
  loops must be symmetric about the plane within sym_tol.
- A wing or fin whose loops are all horizontal gets no tip patch: the four
  wrap patches run the whole span and the flat top is closed by a planar
  face, six faces again.
- The cap is only as fine as its loops: a rounded tip edge needs loops close
  together across it (Rudder_geom_extraction writes 40 cap loops for that).
- A strip's top row runs round its edge on the rim. Where that row doubles
  back on itself over a cap of some height (a wide strip round a thick blunt
  TE), the tip patch's grid is squeezed inside the turn and cannot follow
  the cap there; the reported distance of the tip patch from the cap shows
  it. Give that strip a smaller fraction (BB2's stern plane: 0.01 at its
  30 mm thick TE, 0.07 at the LE).

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
    te_round: bool = False         # the TE strip was narrowed to the TE's rounding


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
ROUND_DROP = 0.25                          # the TE's rounding ends where the curvature has fallen to this
                                           # share of its peak round the TE (_edge_rounding)
ROUND_MARGIN = 1.5                         # a TE strip narrowed to the rounding reaches this many times the
                                           # rounding's arc to each side of the TE
ROUND_SMALL = 0.5                          # ... which is done when that is under this share of the strip
CAP_STATIONS = 401                         # stations along the chord of the cap's mesh
CAP_ACROSS = 81                            # points from the rim to the ridge on each wrap of the cap's mesh
SPAN_TOL = 3.0e-5                          # the span fit adds knots where a pole row misses its grid by this
                                           # fraction of the span (_span_fit)


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


def _edge_rounding(loop, apex, window, n=600):
    """The arc (mm) an edge's rounding reaches to each side of the edge at arc `apex` of a _SmoothLoop: the
    farthest point within `window` of the edge where the curvature is still ROUND_DROP of its peak there (a
    rounded TE: to where the round meets the flank; a blunt base with rounded corners: past the corners).
    0 for a corner (a sharp TE has no rounding to wrap)."""
    if loop.corner and abs(apex - loop.s_te) < 1.0e-9 * loop.S:
        return 0.0
    d = np.linspace(0.0, float(window), max(int(n), int(np.ceil(float(window) / 0.02)) + 1))
    kap = np.maximum(loop.curvature(apex + d), loop.curvature(apex - d))
    drop = ROUND_DROP * kap.max()
    j = int(np.where(kap >= drop)[0][-1])
    if j + 1 >= len(d):
        return float(d[-1])
    return float(d[j] + (d[j + 1] - d[j]) * (kap[j] - drop) / max(kap[j] - kap[j + 1], 1e-300))   # where it crosses


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


def _fan_rows(flats, rim, z_axis, factor=4.0):
    """The loops the side patches run through: the horizontal loops, then the rim (the first loop that is
    not horizontal: a tilted tip section, the cap's first ring). The rim's points sit at different heights,
    so the horizontal loops under it are re-cut to tilt onto it smoothly: over a band factor times as tall
    as the rim's spread in height, each point of a loop moves up its own track through the loops (a cubic
    through that point of every loop and of the rim) by a share of the rim's height there that grows from 0
    at the band's foot to all of it at the rim, with no jump in speed at either end. The points stay on the
    surface the loops describe; only the patches' rows turn. Returns (rows, the band's height)."""
    from scipy.interpolate import CubicSpline
    flats = [np.asarray(lp, dtype=float) for lp in flats]
    rim = np.asarray(rim, dtype=float)
    zf = np.array([lp[0, z_axis] for lp in flats])
    zr = rim[:, z_axis]
    z_ref = float(zr.min())
    shear = zr - z_ref
    spread = float(shear.max())
    if len(flats) < 6 or spread <= 1.0e-9 or np.any(zr <= zf[-1]) or np.any(np.diff(zf) <= 0.0):
        return flats + [rim], 0.0
    band = min(factor * spread, 0.5 * (z_ref - zf[0]))
    z_b = z_ref - band
    j0 = int(np.searchsorted(zf, z_b + 1.0e-9))             # the band's first loop
    if j0 >= len(flats):
        return flats + [rim], 0.0
    levels = zf[j0:]
    tau = (levels - z_b) / band
    g = tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)
    js = max(j0 - 4, 0)
    stack = np.array(flats[js:])
    new = np.empty((len(levels), rim.shape[0], 3))
    for i in range(rim.shape[0]):
        track = CubicSpline(np.r_[zf[js:], zr[i]], np.vstack((stack[:, i], rim[i][None])), axis=0)
        new[:, i] = track(levels + g * shear[i])
    new[:, -1] = new[:, 0]
    return flats[:j0] + list(new) + [rim], float(band)


def _cap_mesh(top, i_te, n_xi=CAP_STATIONS, m=CAP_ACROSS):
    """The cap as a triangle mesh: top[0] the rim, the loops above it nested and pinched to its LE and TE
    points, the ridge the mid-line of the last. Per chord station (the loops' own index fraction) one C2
    spline runs from the rim on the A side through the loops and the ridge to the rim on the B side, sampled
    more finely near the rim. Returns (vertices, triangles, the vertices' index grid (stations, across), the
    LE corner's index, the TE corner's)."""
    from scipy.interpolate import CubicSpline
    top = [np.asarray(lp, dtype=float) for lp in top]
    xi = np.linspace(0.0, 1.0, n_xi)[1:-1]
    t = np.linspace(0.0, 1.0, m)
    frac = t * (0.35 + 0.65 * t)
    side_a, side_b = [], []
    for lp in top:
        a, b = lp[:i_te + 1], lp[i_te:][::-1]
        side_a.append(CubicSpline(np.linspace(0.0, 1.0, len(a)), a, axis=0)(xi))
        side_b.append(CubicSpline(np.linspace(0.0, 1.0, len(b)), b, axis=0)(xi))
    side_a, side_b = np.array(side_a), np.array(side_b)
    ridge = 0.5 * (side_a[-1] + side_b[-1])
    size = max(float(np.ptp(top[0], axis=0).max()), 1.0e-12)
    wraps = []
    for i in range(len(xi)):
        q = np.vstack((side_a[:, i], ridge[i][None], side_b[::-1, i]))
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        keep = np.r_[True, seg > 1.0e-9 * size]
        i_r = int(np.count_nonzero(keep[:len(top) + 1])) - 1         # the ridge among the points kept
        q = q[keep]
        s = _loop_arc(q)
        if s[-1] <= 1.0e-7 * size or i_r <= 0 or i_r >= len(q) - 1:
            wraps.append(None)                                      # pinched to a corner
            continue
        st = np.r_[frac * s[i_r], s[i_r] + (1.0 - frac[::-1][1:]) * (s[-1] - s[i_r])]
        if len(q) >= 4:
            wraps.append(CubicSpline(s, q, axis=0, bc_type="natural")(st))
        else:
            wraps.append(_at_arc(q, s, st))
    good = [i for i, wr in enumerate(wraps) if wr is not None]
    if len(good) < 8:
        raise ValueError("the loops above the rim do not describe a cap")
    g = np.array([wraps[i] for i in good])
    n_st, n_ac = g.shape[:2]
    verts = np.vstack((g.reshape(-1, 3), top[0][0][None], top[0][i_te][None]))
    c_le, c_te = n_st * n_ac, n_st * n_ac + 1
    idx = np.arange(n_st * n_ac).reshape(n_st, n_ac)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tris = np.vstack((np.column_stack((a.ravel(), b.ravel(), c.ravel())),
                      np.column_stack((a.ravel(), c.ravel(), d.ravel())),
                      np.column_stack((np.full(n_ac - 1, c_le), idx[0, :-1], idx[0, 1:])),
                      np.column_stack((np.full(n_ac - 1, c_te), idx[-1, 1:], idx[-1, :-1]))))
    return verts, tris, idx, c_le, c_te


def _rim_params(verts, idx, c_le, c_te, w_le, w_te):
    """The cap mesh's boundary vertices and their place on the unit square of the tip patch: the rim's arcs
    of length w_le to both sides of the LE and w_te of the TE lie on u = 0 and u = 1 (the strips' top rows:
    v from 0 at the A side's split through 0.5 at the apex to 1 at the B side's), the rest of the A side on
    v = 0 and of the B side on v = 1 (the central panels' top rows: u from 0 to 1 between the splits), every
    parameter linear in the arc as the rows' own are."""
    out_i, out_uv = [c_le, c_te], [(0.0, 0.5), (1.0, 0.5)]
    for col, side in ((0, 0.0), (-1, 1.0)):
        rim = idx[:, col]
        s = _loop_arc(np.vstack((verts[c_le][None], verts[rim], verts[c_te][None])))
        total = s[-1]
        if not max(w_le, w_te) < 0.45 * total:
            raise ValueError("the strips overlap on the tip's rim (lower strip_frac)")
        sign = 2.0 * side - 1.0
        for j, vi in enumerate(rim):
            sj = s[j + 1]
            if sj <= w_le:
                uv = (0.0, 0.5 + sign * 0.5 * sj / w_le)
            elif sj >= total - w_te:
                uv = (1.0, 0.5 + sign * 0.5 * (total - sj) / w_te)
            else:
                uv = ((sj - w_le) / (total - w_le - w_te), side)
            out_i.append(vi)
            out_uv.append(uv)
    return np.array(out_i), np.array(out_uv)


def _mean_value_map(verts, tris, bnd_idx, bnd_uv):
    """Parameters (n, 2) of a mesh's vertices with the boundary's held: every inner vertex is the mean value
    (Floater's coordinates, which follow the surface's angles and lengths) of its neighbours, so the map is
    smooth, one to one onto the square, and turns the rim's four smooth corners the way a tensor patch can
    follow."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    n = len(verts)
    p = verts[tris]
    rows, cols, vals = [], [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        e1, e2 = p[:, b] - p[:, a], p[:, c] - p[:, a]
        l1 = np.maximum(np.linalg.norm(e1, axis=1), 1e-300)
        l2 = np.maximum(np.linalg.norm(e2, axis=1), 1e-300)
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


def _locate(uv, tris, verts, un, vn):
    """The mesh's points at the parameter nodes un x vn, (len(vn), len(un), 3): each node in the triangle of
    the map that holds it (a node missed by rounding takes the nearest vertex)."""
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
    out = np.full((len(vn), len(un), 3), np.nan)
    tk = t[ok]
    out[jj[ok], ii[ok]] = (l0[ok, None] * verts[tris[tk, 0]] + l1[ok, None] * verts[tris[tk, 1]]
                           + l2[ok, None] * verts[tris[tk, 2]])
    bad = np.isnan(out).any(axis=2)
    if bad.any():
        from scipy.spatial import cKDTree
        uu, vv = np.meshgrid(un, vn)
        out[bad] = verts[cKDTree(uv).query(np.column_stack((uu[bad], vv[bad])))[1]]
    return out


def _greville(knots, degree=3):
    n = len(knots) - degree - 1
    return np.array([knots[j + 1:j + degree + 1].mean() for j in range(n)])


def _tensor(a_v, grid, a_u):
    """a_v @ grid @ a_u.T for each coordinate: the surface of a pole grid at the bases' stations, or the
    least-squares poles when the two matrices are pseudo-inverses."""
    return np.stack([a_v @ grid[:, :, c] @ a_u.T for c in range(grid.shape[2])], axis=2)


def _normal_gap(data, grid, a_v, a_u, d_v, d_u):
    """The largest distance of data (len(v), len(u), 3) from the patch of a pole grid, measured at the data's
    own parameters along the patch's normal there (a point that only slides along the patch does not count):
    a_v, a_u the bases at the parameters, d_v, d_u their derivatives."""
    res = data - _tensor(a_v, grid, a_u)
    nrm = np.cross(_tensor(a_v, grid, d_u), _tensor(d_v, grid, a_u))
    ln = np.linalg.norm(nrm, axis=2)
    okay = ln > 1.0e-9 * max(float(ln.max()), 1e-300)
    if not okay.any():
        return 0.0
    return float(np.abs(np.einsum("ijc,ijc->ij", res, nrm)[okay] / ln[okay]).max())


def _tensor_fit(data, un, vn, k_u, k_v, row0, row1, col0, col1):
    """The pole grid (v poles, u poles, 3) of the tensor B-spline fitted to data (len(vn), len(un), 3) by least
    squares with its four boundary pole rows held: row0 at v = 0, row1 at v = 1, col0 at u = 0, col1 at
    u = 1. The inner poles start from the boundary's blend, so a pole the data leaves free stays in place.
    Returns (poles, the largest distance of the data from the fitted patch along its normal, mm)."""
    a_u, a_v = _design(un, k_u), _design(vn, k_v)
    gu, gv = _greville(k_u)[None, :, None], _greville(k_v)[:, None, None]
    c00, c10, c01, c11 = row0[0], row0[-1], row1[0], row1[-1]
    poles = ((1.0 - gv) * row0[None] + gv * row1[None] + (1.0 - gu) * col0[:, None] + gu * col1[:, None]
             - ((1.0 - gu) * (1.0 - gv) * c00 + gu * (1.0 - gv) * c10 + (1.0 - gu) * gv * c01 + gu * gv * c11))
    poles[0], poles[-1], poles[:, 0], poles[:, -1] = row0, row1, col0, col1
    res = data - _tensor(a_v, poles, a_u)
    poles[1:-1, 1:-1] += _tensor(np.linalg.pinv(a_v[:, 1:-1]), res, np.linalg.pinv(a_u[:, 1:-1]))
    return poles, _normal_gap(data, poles, a_v, a_u, _design(vn, k_v, deriv=1), _design(un, k_u, deriv=1))


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


def loft_loops(loops, n_poles=40, height_axis=2, flat_tol=1.0e-6, sew_tol=1.0e-3, v_poles=30,
               strip_frac=0.07, half=False, sym_tol=0.2, ease="both"):
    """The solid through closed loops (each (n, 3), mm, the same point count,
    LE first), bottom to top, in five patches and the flat root; see the
    module docstring for how it is built.

    n_poles     B-spline poles along a section between the strips, before the
                knots added towards the strips (30 more); the strips take
                min(n_poles, 60) and 27 more
    v_poles     rows of poles along the span to start with; fewer rows than
                loops smooth extraction noise out, and rows are added where a
                sharp turn of the planform needs them (_span_fit)
    strip_frac  each strip reaches this fraction of the root chord along the
                section's arc to both sides of its edge (the blade's
                constant-width rule). With one fraction, a TE strip more than
                twice as wide as the TE's rounding is narrowed to the
                rounding (ROUND_MARGIN times its arc; the module docstring
                says why). A pair (LE, TE) gives the two strips their own
                fractions, used as given
    half        True: the half body on the LE-to-TE side of the plane of
                symmetry (thickness 0), the loops symmetric within sym_tol (mm)
    ease        which loops give way at a step from one horizontal loop to
                the next (the one where modified sections meet the original
                ones; _ease_flats): "both" (the loops under it and over it,
                so that the surface tilts alike on both sides), "below" (the
                loops under it; those above are kept as they are) or "above"
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
    chord_root = float(np.linalg.norm(loops[0][0] - loops[0][i_te]))
    fr = np.atleast_1d(np.asarray(strip_frac, dtype=float))
    if fr.size not in (1, 2) or np.any(fr <= 0.0):
        raise ValueError("strip_frac is one positive fraction, or a pair (LE, TE)")
    w_le, w_te = float(fr[0]) * chord_root, float(fr[-1]) * chord_root
    notes = []
    span = float(max(lp[:, z_axis].max() for lp in loops) - loops[0][0, z_axis])

    # a step between horizontal loops is taken out of them first: no smooth surface follows one
    given = loops
    eased, steps = _ease_flats(loops[:k], z_axis, v_poles, SPAN_TOL * span, ease)
    if steps:
        loops = eased + loops[k:]

    # the rows of the four patches along the span: every horizontal loop and, under a cap, its rim
    def rows_of(loops):
        if k < n:
            return _fan_rows(loops[:k], loops[k], z_axis)
        return list(loops), 0.0

    row_loops, band = rows_of(loops)
    smooth = [_SmoothLoop(lp, i_te) for lp in row_loops]

    # a TE strip much wider than the TE's rounding is narrowed to the rounding (one fraction given): a small
    # round in the middle of a wide patch is more than a viewer's or a mesher's triangles of the patch resolve
    te_round = False
    if fr.size == 1:
        r_te = max(_edge_rounding(lo, lo.s_te, w_te) for lo in smooth)
        if 0.0 < ROUND_MARGIN * r_te < ROUND_SMALL * w_te:
            w_te, te_round = ROUND_MARGIN * r_te, True
    widths = (f"the strips {w_le:.6g} mm wide about the LE and the TE" if w_le == w_te else
              f"the strips {w_le:.6g} mm wide about the LE and {w_te:.6g} mm about the TE"
              + (" (the TE's rounding)" if te_round else ""))
    if k < n:
        notes.append(f"the tip patch covers the cap above loop {k + 1}; {widths}"
                     + (f"; the rows tilt onto the tip's rim over the top {band:.4g} mm" if band else ""))
    else:
        notes.append(f"all sections horizontal: the flat top closed by a plane; {widths}")

    # per row: the smooth loop, its four arcs at shared split points, the arcs' poles (shared seam poles)
    k_cen = _refined_knots(n_poles, "central")
    k_strip = _refined_knots(max(12, min(int(n_poles), 60)), "strip")
    u_cen, u_strip = _knot_stations(k_cen), _knot_stations(k_strip)
    a_cen, a_strip = _design(u_cen, k_cen), _design(u_strip, k_strip)
    d_cen, d_strip = _design(u_cen, k_cen, deriv=1), _design(u_strip, k_strip, deriv=1)
    pinv = {"cen": np.linalg.pinv(a_cen[:, 1:-1]), "strip": np.linalg.pinv(a_strip[:, 1:-1])}
    fams = {"le": [], "cen_a": [], "te": [], "cen_b": []}
    data = {"le": [], "cen_a": [], "te": [], "cen_b": []}
    spacing, sym, err, first_a, last = [], 0.0, 0.0, None, None
    for lo in smooth:
        if half:
            mirror = lo.dense[lo.it:][::-1].copy()
            mirror[:, thick] *= -1.0
            sym = max(sym, float(np.abs(mirror - lo.dense[:lo.it + 1]).max()))
        arcs = _row_arcs(lo, w_le, w_te, u_strip, u_cen)
        for name, row, a, key in (("le", arcs[0], a_strip, "strip"), ("cen_a", arcs[1], a_cen, "cen"),
                                  ("te", arcs[2], a_strip, "strip"), ("cen_b", arcs[3], a_cen, "cen")):
            poles = np.empty((a.shape[1], 3))
            poles[0], poles[-1] = row[0], row[-1]
            poles[1:-1] = pinv[key] @ (row - np.outer(a[:, 0], row[0]) - np.outer(a[:, -1], row[-1]))
            err = max(err, float(np.abs(a @ poles - row).max()))
            fams[name].append(poles)
            data[name].append(row)
        stacked = np.vstack(arcs)
        if last is not None:
            spacing.append(float(np.linalg.norm(stacked - last, axis=1).mean()))
        last = stacked
        if first_a is None:
            first_a = arcs[1]
    if steps:                                          # the distances reported are those of the loops as given
        data = {"le": [], "cen_a": [], "te": [], "cen_b": []}
        for lp in rows_of(given)[0]:
            arcs = _row_arcs(_SmoothLoop(lp, i_te), w_le, w_te, u_strip, u_cen)
            for name, row in zip(("le", "cen_a", "te", "cen_b"), arcs):
                data[name].append(row)
    if half and sym > sym_tol:
        raise ValueError(f"no half body: the loops are not symmetric about the plane of symmetry "
                         f"(they differ by {sym:.3g} mm, the allowance is {sym_tol:g} mm)")

    # along the span: every family on the same parameters and knots, so the shared seams are the same curves
    vpar = np.r_[0.0, np.cumsum(np.maximum(spacing, 1e-9 * max(max(spacing), 1e-12)))]
    vpar /= vpar[-1]
    grids, vk, vd, _ = _span_fit(fams, v_poles, vpar, SPAN_TOL * span)
    b_v, d_v = _design(vpar, vk, vd), _design(vpar, vk, vd, deriv=1)
    v_err = 0.0
    for name, a, d in (("le", a_strip, d_strip), ("cen_a", a_cen, d_cen), ("te", a_strip, d_strip),
                       ("cen_b", a_cen, d_cen)):
        v_err = max(v_err, _normal_gap(np.array(data[name]), grids[name], b_v, a, d_v, d))
    notes.append(f"along the span {len(vk) - vd - 1} rows of poles through {len(row_loops)} loops; the loops "
                 f"within {v_err:.3g} mm of the patches")
    for at, jump, lo, hi, s_lo, s_hi in steps:
        notes.append(f"a step of {jump:.3g} mm between loops {at} and {at + 1}, eased out of the loops "
                     + (f"over {lo:.3g} mm below it (the loops above kept)" if s_hi == 0.0 else
                        f"over {hi:.3g} mm above it (the loops below kept)" if s_lo == 0.0 else
                        f"over {lo:.3g} mm below it ({100 * s_lo:.0f} % of it) and {hi:.3g} mm above "
                        f"({100 * s_hi:.0f} %)")
                     + ": the patches cross it smoothly")

    # the tip patch: the cap above the rim, fitted through its map onto the patch's square
    tip = None
    if k < n:
        verts, tris, idx, c_le, c_te = _cap_mesh(loops[k:], i_te)
        b_idx, b_uv = _rim_params(verts, idx, c_le, c_te, w_le, w_te)
        uv = _mean_value_map(verts, tris, b_idx, b_uv)
        un, vn = u_cen[1:-1], u_strip[1:-1]
        tip, e = _tensor_fit(_locate(uv, tris, verts, un, vn), un, vn, k_cen, k_strip,
                             grids["cen_a"][-1], grids["cen_b"][-1], grids["le"][-1], grids["te"][-1])
        notes.append(f"the tip patch within {e:.3g} mm of the cap")
        v_err = max(v_err, e)

    # the faces
    edges = (("le", k_strip), ("cen_a", k_cen), ("te", k_strip), ("cen_b", k_cen))
    parts = [_surface_face(grids[name], uk, vk, 3, vd) for name, uk in edges]
    if k < n:
        parts.append(_surface_face(tip, k_cen, k_strip, 3, 3))
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
                      count_faces(solid), v_err, bool(half), steps, (w_le, w_te), te_round)


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
