"""
xcad_loft.py

The CAD solid through the section loops of an XCAD point file, or through
loops in memory, for any geometry whose loops start at the LE, run along one
side to the TE at index (n + 1) // 2 - 1 and come back to the LE (the layout
of Rudder_geom_extraction.write_xcad and rudder_modify.write_modified_xcad).

How the solid is built
----------------------
- Each side of each loop becomes a cubic B-spline fitted by least squares
  (its end points held) with the same knots in every loop, so every surface
  below is built from compatible pole rows. The LE and TE points are snapped
  onto the plane of symmetry (thickness 0) when they are within a hair of it,
  so the seams between the two sides lie exactly on it.
- Each patch is then one bicubic B-spline surface fitted by least squares
  through its loops' pole rows (the end rows held exactly, so neighbouring
  patches share their boundary curve): degree 3 both ways, v_poles rows of
  poles at most. Fewer rows of poles than loops smooth the measurement noise
  of extracted sections out, which keeps the surfaces free of the ripples
  that interpolation (and the high spanwise degrees it needs) put near the
  rudder's tip.
- The patches (the mesh topology): the side from the root up to the collar
  (a cut collar_mm below the highest horizontal section, 0 for none), the
  tip band from the collar over the tip's side walls onto the tip section,
  the cap rim from the tip section to the line where the cap's edge
  rounding meets its deck (found per chord station from the cap loops'
  cross profile, smoothed along the chord; at the fraction crown of the
  cross arc when the cap never flattens), and the cap crown from that line
  to the ridge (the midline of the last cap loop, snapped onto the plane of
  symmetry). The rim's and the crown's rows are resampled along each chord
  station's cross profile, which also evens the cap loops' spacing out.
  Every cap loop runs through the tip section's LE and TE corners, so the
  rim and the crown pinch to those two points. Each patch is two faces, one
  per side, with their seams at the LE and the TE; a flat end loop (the
  root, a flat tip) is closed by a planar face.
- half=True builds the half body on the LE-to-TE side of the plane of
  symmetry (the working frame's z >= 0): the side faces of that side only,
  the half root (its section side closed by the straight chord line), and
  one planar face on the plane of symmetry (root chord, LE seams, ridge, TE
  seams). The loops must be symmetric about the plane within sym_tol.
- The faces are sewn into one shell and made a solid: nine faces for the
  rudder with its cap (six as the half body), four for a wing or fin with
  its flat tip (four as the half body too).

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
    fit_error: float               # largest distance of the loops' points from the section curves, mm
    n_smooth: int                  # leading loops at one height each (the horizontal sections)
    n_ruled: int                   # loops after them (the tilted tip section and the cap loops)
    seconds: float
    notes: list = field(default_factory=list)
    faces: int = 0                 # faces of the solid
    v_error: float = 0.0           # largest distance of the pole rows from the fitted patches, mm
    half: bool = False             # the half body on the LE-to-TE side


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


def _cap_rows(rows, z_axis, crown, n_rim=5, n_crown=4, angle_deg=20.0):
    """The cap's pole rows resampled along each column's cross polyline (the
    tip section's row, the cap loops' rows, the ridge row last): (rim rows,
    crown rows). The split sits where the polyline's slope along the height
    axis flattens below angle_deg (the edge rounding meeting the deck), at
    the fraction crown of the arc when it never does; both patches share the
    split row. Resampling along the arc also evens the cap loops' own uneven
    spacing out."""
    rows = np.asarray(rows, dtype=float)
    m, n_poles = rows.shape[0], rows.shape[1]
    tan_a = np.tan(np.radians(angle_deg))
    rim = np.zeros((n_rim, n_poles, 3))
    crw = np.zeros((n_crown, n_poles, 3))
    arcs, splits = [], []
    for j in range(n_poles):
        q = rows[:, j]
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        arcs.append(s)
        if s[-1] <= 1.0e-9:                        # pinched to a corner
            splits.append(np.nan)
            continue
        dz = np.abs(np.diff(q[:, z_axis]))
        steep = dz > tan_a * np.maximum(seg, 1e-12)
        last = np.where(steep)[0]
        s_split = min(s[last[-1] + 1], 0.9 * s[-1]) if len(last) else float(crown) * s[-1]
        splits.append(max(s_split, 0.1 * s[-1]) / s[-1])
    f = np.array(splits)
    good = ~np.isnan(f)
    if good.any():                                 # the split line smoothed along the chord
        idx = np.arange(n_poles)
        f[~good] = np.interp(idx[~good], idx[good], f[good])
        pad = np.r_[f[8:0:-1], f, f[-2:-10:-1]]
        f = np.convolve(pad, np.ones(9) / 9.0, mode="same")[8:8 + n_poles]
    for j in range(n_poles):
        q, s = rows[:, j], arcs[j]
        if s[-1] <= 1.0e-9:
            rim[:, j] = q[0]
            crw[:, j] = q[0]
            continue
        s_split = f[j] * s[-1]
        for out, stations in ((rim, np.linspace(0.0, s_split, n_rim)),
                              (crw, np.linspace(s_split, s[-1], n_crown))):
            for c in range(3):
                out[:, j, c] = np.interp(stations, s, q[:, c])
        rim[0, j], crw[-1, j] = q[0], q[-1]        # the tip section's row and the ridge exactly
    crw[0] = rim[-1]                               # one shared split row
    return rim, crw


def _v_params(rows):
    """Parameters 0 .. 1 for pole rows along the span: the mean distance from
    row to row, accumulated."""
    rows = np.asarray(rows, dtype=float)
    d = np.linalg.norm(np.diff(rows, axis=0), axis=2).mean(axis=1)
    d = np.maximum(d, 1e-9 * max(d.max(), 1e-12))
    t = np.r_[0.0, np.cumsum(d)]
    return t / t[-1]


def _densify(rows, factor=2.0):
    """Pole rows with linear midpoints inserted into gaps wider than factor
    times the median row-to-row distance (the step from the last horizontal
    section onto the tip section has no data of its own: without a midpoint a
    cubic fit is free to wiggle across it)."""
    rows = [np.asarray(r, dtype=float) for r in rows]
    if len(rows) < 3:
        return np.asarray(rows)
    d = [float(np.linalg.norm(b - a, axis=1).mean()) for a, b in zip(rows[:-1], rows[1:])]
    med = float(np.median(d))
    out = []
    for i, r in enumerate(rows[:-1]):
        out.append(r)
        if med > 0 and d[i] > factor * med:
            k = int(np.ceil(d[i] / med)) - 1
            for t in np.linspace(0.0, 1.0, k + 2)[1:-1]:
                out.append((1.0 - t) * r + t * rows[i + 1])
    out.append(rows[-1])
    return np.asarray(out)


def _fit_grid(rows, v_poles, degree=3):
    """A pole grid fitted through pole rows along the span by least squares,
    the end rows held exactly: (grid (n_v, n_poles, 3), v knots, v degree,
    largest distance of the rows from the grid, mm)."""
    rows = np.asarray(rows, dtype=float)
    m = len(rows)
    n_v = int(min(max(v_poles, 2), m))
    deg = int(min(degree, n_v - 1))
    if n_v == 2:
        grid = rows[[0, -1]]
        knots = np.array([0.0, 0.0, 1.0, 1.0])
        pred = np.array([grid[0] + t * (grid[1] - grid[0]) for t in _v_params(rows)])
        return grid, knots, 1, float(np.abs(pred - rows).max())
    a, knots = _basis_at(_v_params(rows), n_v, deg, spaced=True)
    flat = rows.reshape(m, -1)
    grid = _fit_poles(flat, a).reshape(n_v, rows.shape[1], 3)
    return grid, knots, deg, float(np.abs(a @ grid.reshape(n_v, -1) - flat).max())


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


class _Patch:
    """One patch of the loft: the pole grids of its two sides (up: LE to TE,
    lo: TE to LE) fitted through its loops' pole rows. A patch pinched to the
    tip section's corners (the rim, the crown) is built through ThruSections
    on its fitted rows instead of one raw B-spline face: OCC's booleans choke
    on a raw face whose boundary collapses to a point."""

    def __init__(self, name, up_rows, lo_rows, v_poles, fitter, pinched=False, thick=1):
        self.name, self.fitter, self.pinched, self.thick = name, fitter, pinched, thick
        self.up, self.vk_up, self.vdeg_up, e1 = _fit_grid(up_rows, v_poles)
        self.lo, self.vk_lo, self.vdeg_lo, e2 = _fit_grid(lo_rows, v_poles)
        self.rows_up, self.rows_lo = np.asarray(up_rows, dtype=float), np.asarray(lo_rows, dtype=float)
        self.error = max(e1, e2)

    def _closed_wires(self):
        """One closed wire per data row (its two sides on the common knots), so
        neighbouring patches share their boundary curves exactly."""
        BRepBuilderAPI_MakeWire = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeWire")
        out = []
        for r_up, r_lo in zip(self.rows_up, self.rows_lo):
            w = BRepBuilderAPI_MakeWire()
            w.Add(_bspline_edge(r_up, self.fitter.k_up, self.fitter.degree))
            w.Add(_bspline_edge(r_lo, self.fitter.k_lo, self.fitter.degree))
            out.append(w.Wire())
        return out

    def faces(self, half):
        out = [_surface_face(self.up, self.fitter.k_up, self.vk_up, self.fitter.degree, self.vdeg_up)]
        if not half:
            out.append(_surface_face(self.lo, self.fitter.k_lo, self.vk_lo, self.fitter.degree, self.vdeg_lo))
        return out

    def up_column(self, col):
        """The edge along the span at one pole column of the up side (None
        when it is pinched to a point there)."""
        poles = self.up[:, col]
        if np.ptp(poles, axis=0).max() <= 1.0e-9:
            return None
        return _bspline_edge(poles, self.vk_up, self.vdeg_up)

    def up_row(self, row):
        """The edge along the section at one pole row of the up side."""
        return _bspline_edge(self.up[row], self.fitter.k_up, self.fitter.degree)


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
               collar_mm=10.0, crown=0.5, half=False, sym_tol=0.2):
    """The solid through closed loops (each (n, 3), mm, the same point count,
    LE first), bottom to top; see the module docstring for the patches.

    v_poles     rows of poles of the side patch at most (the tip band, the rim
                and the crown use fewer, their loops being few); fewer rows
                than loops smooth extraction noise out, degree 3 either way
    collar_mm   the side is split this far (mm) below the highest horizontal
                section (0: no collar; none either when every loop is
                horizontal)
    crown       where the cap splits into rim and crown when its cross
                profile never flattens into a deck: this fraction of the
                cross arc (with a deck, the split follows the line where the
                edge rounding meets it)
    half        True: the half body on the LE-to-TE side of the plane of
                symmetry (thickness 0), the loops symmetric within sym_tol (mm)
    """
    t0 = time.time()
    loops = [np.asarray(lp, dtype=float).copy() for lp in loops]
    if len(loops) < 2:
        raise ValueError("a loft needs at least two loops")
    if len({len(lp) for lp in loops}) != 1:
        raise ValueError("every loop must have the same number of points")
    thick = 1 if height_axis == 2 else 2
    n = len(loops)
    n_pts = len(loops[0])
    i_te = (n_pts + 1) // 2 - 1
    size = max(np.abs(lp).max() for lp in loops)
    for lp in loops:                                   # the seams onto the plane of symmetry
        for i in (0, i_te, n_pts - 1):
            if abs(lp[i, thick]) < 1.0e-3:
                lp[i, thick] = 0.0
    flat = [np.ptp(lp[:, height_axis]) <= flat_tol * max(1.0, size) for lp in loops]
    k = next((i for i, f in enumerate(flat) if not f), n)          # leading loops at one height each
    fitter = _SectionFitter(n_pts, n_poles)
    up_rows, lo_rows, err = [], [], 0.0
    for lp in loops:
        p_up, p_lo, e = fitter.sides(lp)
        up_rows.append(p_up)
        lo_rows.append(p_lo)
        err = max(err, e)
    up_rows, lo_rows = np.array(up_rows), np.array(lo_rows)
    if half:
        mirror = up_rows.copy()
        mirror[:, :, thick] *= -1.0
        sym = float(np.abs(mirror - lo_rows[:, ::-1]).max())
        if sym > sym_tol:
            raise ValueError(f"no half body: the loops are not symmetric about the plane of symmetry "
                             f"(they differ by {sym:.3g} mm, the allowance is {sym_tol:g} mm)")
    notes, patches = [], []
    z_axis = height_axis

    # the side, split at the collar
    c_i = 0
    if k >= 2:
        if collar_mm and collar_mm > 0.0 and k < n:
            z_top = loops[k - 1][0, z_axis]
            heights = np.array([lp[0, z_axis] for lp in loops[:k]])
            c_i = int(np.argmin(np.abs(heights - (z_top - float(collar_mm)))))
            c_i = c_i if 1 <= c_i <= k - 2 else 0
        if c_i:
            patches.append(_Patch("side", up_rows[:c_i + 1], lo_rows[:c_i + 1], v_poles, fitter))
            notes.append(f"the side split at the collar (loop {c_i + 1}, {loops[c_i][0, z_axis]:.6g} mm)")
        end = k if k == n else k + 1                   # the tip band runs onto the tilted tip section
        b_up, b_lo = _densify(up_rows[c_i:end]), _densify(lo_rows[c_i:end])
        patches.append(_Patch("tip band" if c_i else "side", b_up, b_lo,
                              v_poles if not c_i and k == n else max(6, min(v_poles, len(b_up))), fitter))
    elif k == 1 and k < n:
        patches.append(_Patch("side", up_rows[:2], lo_rows[:2], 2, fitter))

    # the cap: rim and crown, split where the edge rounding meets the deck
    ridge = None
    if k < n:
        tip_i = k
        ridge = 0.5 * (up_rows[n - 1] + lo_rows[n - 1][::-1])
        ridge[:, thick] = 0.0
        if n - tip_i >= 3:
            rim_up, crown_up = _cap_rows(np.concatenate([up_rows[tip_i:], ridge[None]]), z_axis, crown)
            rim_lo, crown_lo = _cap_rows(np.concatenate([lo_rows[tip_i:], ridge[::-1][None]]), z_axis, crown)
            patches.append(_Patch("rim", rim_up, rim_lo, len(rim_up), fitter, pinched=True, thick=thick))
            patches.append(_Patch("crown", crown_up, crown_lo, len(crown_up), fitter, pinched=True, thick=thick))
            notes.append("cap: the rim from the tip section to the line where its edge rounding meets the "
                         "deck, the crown from it to the ridge")
        else:
            crown_up = np.concatenate([up_rows[tip_i:], ridge[None]])
            crown_lo = np.concatenate([lo_rows[tip_i:], ridge[::-1][None]])
            patches.append(_Patch("crown", crown_up, crown_lo, len(crown_up), fitter, pinched=True, thick=thick))
            notes.append("the end loop closed by the crown to its ridge")

    v_err = max(q.error for q in patches)
    parts = []
    for q in patches:
        parts += q.faces(half)

    # the flat ends (the root, and a flat tip when no other loop follows)
    ends = [(0, patches[0], 0)] + ([(n - 1, patches[-1], -1)] if k == n else [])
    for i, patch, row in ends:
        if not flat[i]:
            if half:
                raise ValueError("no half body: the end loop is not flat")
            parts.append(fitter.closing_face(loops[i]))
            notes.append(f"loop {i + 1} closed by the ruled face across it")
        elif half:
            row_i = 0 if row == 0 else len(patch.up) - 1
            chord = _line_edge(patch.up[row_i, -1], patch.up[row_i, 0])      # TE back to LE, on the plane
            parts.append(_wire_face([patch.up_row(row_i), chord]))
        else:
            parts.append(_planar_face(fitter.wire(loops[i])[0]))

    if half:                                           # the face on the plane of symmetry
        edges = []
        root = patches[0]
        edges.append(_line_edge(root.up[0, 0], root.up[0, -1]))              # the root chord, LE to TE
        for q in patches:                                                    # up the LE seams
            e = q.up_column(0)
            if e is not None:
                edges.append(e)
        if ridge is not None:
            edges.append(patches[-1].up_row(len(patches[-1].up) - 1))        # the ridge, LE to TE
        elif k == n:
            top = patches[-1]
            edges.append(_line_edge(top.up[-1, 0], top.up[-1, -1]))          # a flat tip's chord
        for q in reversed(patches):                                          # down the TE seams
            e = q.up_column(-1)
            if e is not None:
                edges.append(e)
        parts.append(_wire_face(edges))
        notes.append("half body: the LE-to-TE side, closed on the plane of symmetry")

    solid, free = _sew_solid(parts, sew_tol)
    BRepCheck_Analyzer = _occ("BRepCheck", "BRepCheck_Analyzer")
    valid = bool(BRepCheck_Analyzer(solid).IsValid())
    vol, area = mass_properties(solid)
    return LoftResult(solid, valid, vol, area, free, err, k, n - k, time.time() - t0, notes,
                      len(parts), v_err, bool(half))


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
    ap.add_argument("--view", action="store_true", help="show the solid in the pythonocc viewer")
    ap.add_argument("--view-step", default=None, help="only show this STEP file")
    args = ap.parse_args()
    if args.view_step:
        view(read_step(args.view_step), os.path.basename(args.view_step))
        return
    if not args.xcad:
        ap.error("give an XCAD point file or --view-step")
    res = loft_loops(read_loops(args.xcad, args.units), args.poles, half=args.half)
    step = args.step or os.path.splitext(args.xcad)[0] + ".step"
    write_step(res.shape, step)
    print(f"{args.xcad}: {res.n_smooth} horizontal + {res.n_ruled} tip loops, {res.faces} faces"
          f"{' (half body)' if res.half else ''}, valid {res.valid}, volume {res.volume:.1f} mm^3, "
          f"area {res.area:.1f} mm^2, open edges {res.free_edges}, section fit within "
          f"{res.fit_error * 1e3:.1f} um, the patches within {res.v_error * 1e3:.1f} um, {res.seconds:.1f} s "
          f"-> {step}")
    if args.view:
        view(res.shape, os.path.basename(step))


if __name__ == "__main__":
    main()
