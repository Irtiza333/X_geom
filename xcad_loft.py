"""
xcad_loft.py

The CAD solid through the section loops of an XCAD point file, or through
loops in memory, for any geometry whose loops start at the LE, run along one
side to the TE at index (n + 1) // 2 - 1 and come back to the LE (the layout
of Rudder_geom_extraction.write_xcad and rudder_modify.write_modified_xcad).

How the solid is built
----------------------
- Each side of each loop becomes a cubic B-spline fitted by least squares
  (its end points held) with the same knots in every loop, so the loops are
  compatible as they come and the loft needs no knot merging. That is what
  keeps it fast: lofting the rudder's 179 loops takes about 3 s, against a
  minute for 60 loops interpolated point by point.
- The leading loops that lie at one height each (the horizontal sections)
  are lofted smoothly (BRepOffsetAPI_ThruSections, two faces). The loops
  after them (the tilted tip section and the cap loops) are lofted smoothly
  among themselves (two faces), joined to the last horizontal section by one
  ruled step (the tip's side walls).
- A flat end loop (the root) is closed by a planar face; an end loop that is
  not flat (the last cap loop, a thin loop just under the crest) by the ruled
  face across it, straight lines from one side to the other.
- The faces are sewn into one shell and made a solid: eight faces for the
  rudder with its cap.

Coordinates: the XCAD frame (x chordwise, y thickness, z the height) in mm;
point files in metres are scaled by 1000, as X_CAD.py does.

    python xcad_loft.py outputs/modified/baseline_sections_xcad_m.dat   (writes the .step next to it)
    python xcad_loft.py <xcad file> --view        and shows it in the pythonocc viewer
    python xcad_loft.py --view-step <file.step>   shows a STEP file

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

def _basis(m, n_poles, degree=3):
    """Least-squares B-spline basis (m points, parameter evenly spaced by
    point index) on uniform clamped knots: (A (m, n_poles), knots)."""
    from scipy.interpolate import BSpline
    n_poles = max(int(n_poles), degree + 1)
    knots = np.r_[[0.0] * degree, np.linspace(0.0, 1.0, n_poles - degree + 1), [1.0] * degree]
    u = np.linspace(0.0, 1.0, m)
    try:
        a = BSpline.design_matrix(u, knots, degree).toarray()
    except AttributeError:                                         # scipy < 1.8
        a = np.column_stack([BSpline(knots, np.eye(n_poles)[j], degree)(u) for j in range(n_poles)])
    return a, knots


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
    n_smooth: int                  # leading loops at one height each, lofted smoothly
    n_ruled: int                   # loops after them (joined by one ruled step, then lofted smoothly)
    seconds: float
    notes: list = field(default_factory=list)


def _thru(wires, solid, ruled):
    BRepOffsetAPI_ThruSections = _occ("BRepOffsetAPI", "BRepOffsetAPI_ThruSections")
    ts = BRepOffsetAPI_ThruSections(solid, ruled, 1.0e-6)
    ts.CheckCompatibility(False)
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


def count_faces(shape):
    """Number of faces of a shape."""
    TopExp_Explorer = _occ("TopExp", "TopExp_Explorer")
    TopAbs_FACE = _occ("TopAbs", "TopAbs_FACE")
    ex, n = TopExp_Explorer(shape, TopAbs_FACE), 0
    while ex.More():
        n += 1
        ex.Next()
    return n


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


def loft_loops(loops, n_poles=40, height_axis=2, flat_tol=1.0e-6, sew_tol=1.0e-3):
    """The solid through closed loops (each (n, 3), mm, the same point count,
    LE first), bottom to top. See the module docstring."""
    t0 = time.time()
    loops = [np.asarray(lp, dtype=float) for lp in loops]
    if len(loops) < 2:
        raise ValueError("a loft needs at least two loops")
    if len({len(lp) for lp in loops}) != 1:
        raise ValueError("every loop must have the same number of points")
    flat = [np.ptp(lp[:, height_axis]) <= flat_tol * max(1.0, np.abs(lp).max()) for lp in loops]
    n = len(loops)
    k = next((i for i, f in enumerate(flat) if not f), n)          # leading loops at one height each
    fitter = _SectionFitter(len(loops[0]), n_poles)
    wires, err = [], 0.0
    for lp in loops:
        w, e = fitter.wire(lp)
        wires.append(w)
        err = max(err, e)
    notes, parts = [], []
    if k >= 2:
        parts.append(_thru(wires[:k], False, False))               # the horizontal sections, smooth
    if 1 <= k < n:
        parts.append(_thru([wires[k - 1], wires[k]], False, True))  # a ruled step onto the first other loop
        notes.append(f"loop {k + 1} joined to loop {k} by a ruled step")
    if n - max(k, 0) >= 2 and k < n:
        parts.append(_thru(wires[max(k, 0):], False, False))      # the other loops (tip, cap), smooth
    for i in (0, n - 1):                                            # the ends
        parts.append(_planar_face(wires[i]) if flat[i] else fitter.closing_face(loops[i]))
        if not flat[i]:
            notes.append(f"loop {i + 1} closed by the ruled face across it")
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
    solid = fx.Solid()
    BRepCheck_Analyzer = _occ("BRepCheck", "BRepCheck_Analyzer")
    valid = bool(BRepCheck_Analyzer(solid).IsValid())
    vol, area = mass_properties(solid)
    return LoftResult(solid, valid, vol, area, free, err, k, n - k, time.time() - t0, notes)


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
    ap.add_argument("--view", action="store_true", help="show the solid in the pythonocc viewer")
    ap.add_argument("--view-step", default=None, help="only show this STEP file")
    args = ap.parse_args()
    if args.view_step:
        view(read_step(args.view_step), os.path.basename(args.view_step))
        return
    if not args.xcad:
        ap.error("give an XCAD point file or --view-step")
    res = loft_loops(read_loops(args.xcad, args.units), args.poles)
    step = args.step or os.path.splitext(args.xcad)[0] + ".step"
    write_step(res.shape, step)
    print(f"{args.xcad}: {res.n_smooth} loops smooth + {res.n_ruled} ruled, valid {res.valid}, "
          f"volume {res.volume:.1f} mm^3, area {res.area:.1f} mm^2, open edges {res.free_edges}, "
          f"section fit within {res.fit_error * 1e3:.1f} um, {res.seconds:.1f} s -> {step}")
    if args.view:
        view(res.shape, os.path.basename(step))


if __name__ == "__main__":
    main()
