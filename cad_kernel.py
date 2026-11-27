"""
cad_kernel.py

XGeom's CAD kernel: the one module that calls OpenCascade (OCCT). The geometry
modules give it plain data (numpy arrays of points, pole grids and knot
vectors, numbers, file paths) and get back shapes, which they treat as opaque
handles and only pass back to these functions (27 Nov 2026, Eddie's request:
CAD in the browser, or under another binding, then means reimplementing these
functions, not finding OCC calls through the geometry). Either binding works:
pythonocc-core (OCC.Core, conda-forge) or cadquery-ocp (OCP). The units are
the caller's (XGeom's CAD is in mm).

Binding
    available()                       a binding can be imported
    binding()                         'OCC.Core', 'OCP' or None

Edges, wires, faces and solids
    bspline_edge(poles, knots, degree)
                                      the B-spline edge of the poles (n, 3) and the knot vector (repeated
                                      knots: their multiplicity)
    line_edge(a, b)                   the straight edge from point a to point b
    arc_edge(a, m, b)                 the circular arc from a through m to b
    wire(edges)                       the edges chained into one wire
    planar_face(wire)                 the plane face bounded by a closed flat wire
    wire_face(edges)                  the plane face bounded by the edges (one wire), its orientation fixed
    interpolate(points, t0=None, t1=None, tol=1e-9)
                                      the cubic B-spline curve through the points (end tangents t0, t1):
                                      (poles, weights, knot vector, degree)
    bspline_surface(grid, uknots, vknots, udeg=3, vdeg=3, weights=None)
                                      the B-spline surface (a handle) of the pole grid (n_v rows of n_u
                                      poles, 3), u along a row; rational with the weights (n_v, n_u)
    natural_face(surface, tol=1e-6)   the face of the whole surface
    surface_face(grid, uknots, vknots, udeg=3, vdeg=3)
                                      natural_face(bspline_surface(...))
    loft(wires, solid, ruled, max_degree=None)
                                      the surface through the wires (solid: closed by the end wires)
    cylinder_face(origin, axis, xdir, radius, wire)
                                      the face on a cylinder bounded by a wire on it, fixed
    sew(faces, tol, fix=True)         the faces sewn (a shell, or shells), fixed (ShapeFix_Shape)
    sew_solid(faces, tol)             the faces sewn into one shell and made a solid, fixed:
                                      (solid, the edges the sewing left free)
    compound(shapes)                  the shapes as one compound
    box(lo, hi)                       the box between two corners
    cylinder(base, axis, radius, height)
                                      the solid cylinder from the point base along the direction axis
    revolve(shape, origin, axis, angle)
                                      the shape turned about the axis through origin (a face: a solid)
    prism(shape, vector)              the shape swept along the vector: (prism, its first, its last shape)
    transformed(shape, m, t)          a copy moved by x -> m x + t
    nurbs(shape)                      a copy with every surface and curve a B-spline
    mirrored(shape, origin, normal)   a copy mirrored in the plane
    common(a, b, fuzzy=None), cut(a, b, fuzzy=None), fuse(a, b, fuzzy=None)
                                      the part of a inside b, a outside b, a and b together (fuzzy: the
                                      boolean's fuzzy value)
    fuse_all(shapes, fuzzy=None)      the union of the shapes in one boolean
    unify(shape)                      same-domain faces and edges merged

Measures
    shapes(shape, kind)               a shape's sub-shapes of a kind ('SOLID', 'SHELL', 'FACE', 'WIRE', ...)
    faces(shape), count_faces(shape)  a shape's faces (handles), how many
    is_same(a, b)                     a and b the same shape (any orientation)
    free_wires(shape, tol)            the closed wires along which its faces have no neighbour
    edge_points(edge, n)              n points (n, 3) along an edge, evenly in its parameter, ends included
    surface_points(surface, u, v)     the surface's points (n, 3) at the parameter pairs (u[i], v[i])
    surface_kind(face)                its surface's kind: 'plane', 'cylinder', 'cone', 'sphere', 'torus',
                                      'bezier', 'bspline', 'revolution', 'extrusion', 'offset' or 'other'
    mass_properties(shape, eps=1e-7)  (volume, area), adaptive integration to the relative error eps
    volume(shape, eps=1e-7), centroid(shape, eps=1e-7)
                                      the volume alone; the volume's centre (3,)
    is_valid(shape)                   BRepCheck_Analyzer's verdict
    bounding_box(shape)               (2, 3): the lowest and the highest corner, on the surfaces (not
                                      their control nets)
    tessellate(shape, linear, angular)
                                      (vertices (n, 3), triangles (m, 3)) for display, outward-facing

Files
    write_step(shape, path)           STEP (AP214, as built: STEPControl_AsIs)
    read_step(path)                   one shape
    write_iges(shape, path, brep=False)
                                      IGES in mm: faces as trimmed surfaces (entity 144, what CAD and
                                      meshing tools read), or brep: BRep solids
    read_iges(path)                   one shape
    view(shape, title)                the pythonocc viewer (desktop; blocks until closed)

Moving over (27 Nov 2026): xcad_loft (step 1), X_CAD_new and prop_hub (step
2), vehicle_cad, wing_modify and hull_modify (step 3) call only these. The
extraction modules still reach OpenCascade through the binding helpers _occ,
_static and _topods (also as xcad_loft._occ, ...), until they move to the
kernel's functions too.
_cad_kernel_test.py checks the kernel on its own, and that the modules moved
over call no OpenCascade themselves.
"""

from __future__ import annotations

import importlib
import os
import sys

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


# --------------------------------------------------------------------------
# The binding
# --------------------------------------------------------------------------

def available():
    """True when pythonocc-core or cadquery-ocp can be imported."""
    return _PKG is not None


def binding():
    """The binding in use: 'OCC.Core' (pythonocc-core), 'OCP' (cadquery-ocp), or None."""
    return _PKG


def _occ(module, *names):
    """Classes of a binding's module by name (the binding helpers of the modules not yet on the kernel)."""
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


# --------------------------------------------------------------------------
# Edges, wires, faces and solids
# --------------------------------------------------------------------------

def bspline_edge(poles, knots, degree):
    """The edge of the B-spline curve of the poles ((n, 3)) and the knot vector (its repeated knots give their
    multiplicities) of the degree given."""
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


def line_edge(a, b):
    """The straight edge from point a to point b."""
    gp_Pnt = _occ("gp", "gp_Pnt")
    BRepBuilderAPI_MakeEdge = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeEdge")
    return BRepBuilderAPI_MakeEdge(gp_Pnt(*[float(v) for v in a]), gp_Pnt(*[float(v) for v in b])).Edge()


def arc_edge(a, m, b):
    """The circular arc from point a through point m to point b."""
    gp_Pnt = _occ("gp", "gp_Pnt")
    GC_MakeArcOfCircle = _occ("GC", "GC_MakeArcOfCircle")
    BRepBuilderAPI_MakeEdge = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeEdge")
    pa, pm, pb = (gp_Pnt(*[float(v) for v in p]) for p in (a, m, b))
    return BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(pa, pm, pb).Value()).Edge()


def wire(edges):
    """The edges, in order, chained into one wire."""
    BRepBuilderAPI_MakeWire = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeWire")
    edges = list(edges)
    if len(edges) == 1:
        return BRepBuilderAPI_MakeWire(edges[0]).Wire()
    w = BRepBuilderAPI_MakeWire()
    for e in edges:
        w.Add(e)
    return w.Wire()


def planar_face(wire):
    """The plane face bounded by a closed flat wire."""
    BRepBuilderAPI_MakeFace = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace")
    mk = BRepBuilderAPI_MakeFace(wire, True)
    if not mk.IsDone():
        raise RuntimeError("a flat end loop did not give a planar face")
    return mk.Face()


def wire_face(edges):
    """The planar face bounded by the edges (chained into one wire), its orientation fixed (a wire chained
    from edges of other faces can come out unorientable otherwise)."""
    BRepBuilderAPI_MakeWire = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeWire")
    ShapeFix_Face = _occ("ShapeFix", "ShapeFix_Face")
    w = BRepBuilderAPI_MakeWire()
    for e in edges:
        w.Add(e)
    fix = ShapeFix_Face(planar_face(w.Wire()))
    fix.FixOrientation()
    fix.Perform()
    return fix.Face()


def interpolate(points, t0=None, t1=None, tol=1.0e-9):
    """The cubic B-spline curve through the points (n, 3) (GeomAPI_Interpolate, not periodic; tol: its
    tolerance for coincident points), with the end tangents t0 and t1 when given (as given: their length
    scales the parameter): (poles (m, 3), weights (m,), the knot vector (repeated knots: their
    multiplicity), degree)."""
    gp_Pnt, gp_Vec = _occ("gp", "gp_Pnt", "gp_Vec")
    TColgp_HArray1OfPnt = _occ("TColgp", "TColgp_HArray1OfPnt")
    GeomAPI_Interpolate = _occ("GeomAPI", "GeomAPI_Interpolate")
    arr = TColgp_HArray1OfPnt(1, len(points))
    for i, p in enumerate(points, start=1):
        arr.SetValue(i, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    it = GeomAPI_Interpolate(arr, False, float(tol))
    if t0 is not None:
        it.Load(gp_Vec(*[float(v) for v in t0]), gp_Vec(*[float(v) for v in t1]), False)
    it.Perform()
    if not it.IsDone():
        raise RuntimeError("the B-spline through the points failed")
    c = it.Curve()
    n = c.NbPoles()
    poles = np.array([[c.Pole(i).X(), c.Pole(i).Y(), c.Pole(i).Z()] for i in range(1, n + 1)])
    weights = np.array([c.Weight(i) for i in range(1, n + 1)])
    knots = np.concatenate([[c.Knot(k)] * c.Multiplicity(k) for k in range(1, c.NbKnots() + 1)])
    return poles, weights, knots, c.Degree()


def bspline_surface(grid, uknots, vknots, udeg=3, vdeg=3, weights=None):
    """The B-spline surface (a handle: natural_face, surface_points) of the pole grid (n_v rows of n_u poles,
    3), u along a row; knot vectors with their repeated knots; rational with the weights (n_v, n_u)."""
    gp_Pnt = _occ("gp", "gp_Pnt")
    TColgp_Array2OfPnt = _occ("TColgp", "TColgp_Array2OfPnt")
    Geom_BSplineSurface = _occ("Geom", "Geom_BSplineSurface")
    grid = np.asarray(grid, dtype=float)
    n_v, n_u = grid.shape[0], grid.shape[1]
    arr = TColgp_Array2OfPnt(1, n_u, 1, n_v)
    for i in range(n_u):
        for j in range(n_v):
            p = grid[j, i]
            arr.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
    uk, um = _unique_mults(uknots)
    vk, vm = _unique_mults(vknots)
    if weights is None:
        return Geom_BSplineSurface(arr, _real_array(uk), _real_array(vk), _int_array(um), _int_array(vm),
                                   int(udeg), int(vdeg))
    w = _occ("TColStd", "TColStd_Array2OfReal")(1, n_u, 1, n_v)
    for i in range(n_u):
        for j in range(n_v):
            w.SetValue(i + 1, j + 1, float(weights[j][i]))
    return Geom_BSplineSurface(arr, w, _real_array(uk), _real_array(vk), _int_array(um), _int_array(vm),
                               int(udeg), int(vdeg), False, False)


def natural_face(surface, tol=1.0e-6):
    """The face of the whole surface (bspline_surface), bounded by its parameter range; tol: the tolerance
    for degenerate edges."""
    BRepBuilderAPI_MakeFace = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace")
    return BRepBuilderAPI_MakeFace(surface, float(tol)).Face()


def surface_face(grid, uknots, vknots, udeg=3, vdeg=3):
    """The face of the B-spline surface with the pole grid (n_v rows of n_u poles, 3), u along a row; knot
    vectors with their repeated knots."""
    return natural_face(bspline_surface(grid, uknots, vknots, udeg, vdeg))


def loft(wires, solid, ruled, max_degree=None):
    """The surface through the wires in order (BRepOffsetAPI_ThruSections, precision 1e-6): ruled, straight
    between them; solid, closed by the end wires; max_degree caps its degree along the loft."""
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


def cylinder_face(origin, axis, xdir, radius, wire):
    """The face on the cylinder of the radius (its axis through the point origin along axis, its seam the
    line at xdir from the axis) bounded by the wire, which lies on it; fixed (ShapeFix_Face). RuntimeError
    when it fails."""
    gp_Pnt, gp_Dir, gp_Ax3 = _occ("gp", "gp_Pnt", "gp_Dir", "gp_Ax3")
    BRepBuilderAPI_MakeFace = _occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace")
    ax = gp_Ax3(gp_Pnt(*[float(v) for v in origin]), gp_Dir(*[float(v) for v in axis]),
                gp_Dir(*[float(v) for v in xdir]))
    mk = BRepBuilderAPI_MakeFace(_occ("Geom", "Geom_CylindricalSurface")(ax, float(radius)), wire, True)
    if not mk.IsDone():
        raise RuntimeError("a face on a cylinder bounded by a wire failed")
    fix = _occ("ShapeFix", "ShapeFix_Face")(mk.Face())
    fix.Perform()
    return fix.Face()


def sew(faces, tol, fix=True):
    """The faces sewn (BRepBuilderAPI_Sewing, tolerance tol): a shell, or a compound of shells and faces
    left apart; with fix, ShapeFix_Shape run on it."""
    BRepBuilderAPI_Sewing = _occ("BRepBuilderAPI", "BRepBuilderAPI_Sewing")
    sewing = BRepBuilderAPI_Sewing(float(tol))
    for f in faces:
        sewing.Add(f)
    sewing.Perform()
    shape = sewing.SewedShape()
    if not fix:
        return shape
    fixer = _occ("ShapeFix", "ShapeFix_Shape")(shape)
    fixer.Perform()
    return fixer.Shape()


def sew_solid(faces, tol):
    """The faces sewn into one shell (tolerance tol) and made a solid, the shell and the solid fixed
    (ShapeFix): (solid, the edges the sewing left free; 0 for a closed solid). RuntimeError when the faces
    do not sew into one shell."""
    BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid = _occ("BRepBuilderAPI", "BRepBuilderAPI_Sewing",
                                                           "BRepBuilderAPI_MakeSolid")
    sew = BRepBuilderAPI_Sewing(tol)
    for p in faces:
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


def box(lo, hi):
    """The box between the corners lo and hi."""
    BRepPrimAPI_MakeBox = _occ("BRepPrimAPI", "BRepPrimAPI_MakeBox")
    gp_Pnt = _occ("gp", "gp_Pnt")
    return BRepPrimAPI_MakeBox(gp_Pnt(*[float(v) for v in lo]), gp_Pnt(*[float(v) for v in hi])).Shape()


def compound(shapes):
    """The shapes as one compound."""
    BRep_Builder = _occ("BRep", "BRep_Builder")
    comp, builder = _occ("TopoDS", "TopoDS_Compound")(), BRep_Builder()
    builder.MakeCompound(comp)
    for s in shapes:
        builder.Add(comp, s)
    return comp


def cylinder(base, axis, radius, height):
    """The solid cylinder of the radius from the point base along the direction axis, height long."""
    gp_Pnt, gp_Ax2, gp_Dir = _occ("gp", "gp_Pnt", "gp_Ax2", "gp_Dir")
    BRepPrimAPI_MakeCylinder = _occ("BRepPrimAPI", "BRepPrimAPI_MakeCylinder")
    ax = gp_Ax2(gp_Pnt(*[float(v) for v in base]), gp_Dir(*[float(v) for v in axis]))
    return BRepPrimAPI_MakeCylinder(ax, float(radius), float(height)).Solid()


def revolve(shape, origin, axis, angle):
    """The shape turned through the angle (rad) about the axis (a direction) through the point origin: a
    face sweeps a solid."""
    gp_Pnt, gp_Ax1, gp_Dir = _occ("gp", "gp_Pnt", "gp_Ax1", "gp_Dir")
    BRepPrimAPI_MakeRevol = _occ("BRepPrimAPI", "BRepPrimAPI_MakeRevol")
    ax = gp_Ax1(gp_Pnt(*[float(v) for v in origin]), gp_Dir(*[float(v) for v in axis]))
    return BRepPrimAPI_MakeRevol(shape, ax, float(angle)).Shape()


def prism(shape, vector):
    """The shape swept along the vector (a face sweeps a solid): (the prism, its first shape, the shape it
    starts from, and its last shape, where it ends)."""
    BRepPrimAPI_MakePrism = _occ("BRepPrimAPI", "BRepPrimAPI_MakePrism")
    mk = BRepPrimAPI_MakePrism(shape, _occ("gp", "gp_Vec")(*[float(v) for v in vector]))
    first, last = mk.FirstShape(), mk.LastShape()
    return mk.Shape(), first, last


def transformed(shape, m, t):
    """A copy of the shape moved by x -> m x + t (m (3, 3): a rotation times a positive scale; t (3,))."""
    gp_Trsf = _occ("gp", "gp_Trsf")
    BRepBuilderAPI_Transform = _occ("BRepBuilderAPI", "BRepBuilderAPI_Transform")
    m, t = np.asarray(m, dtype=float), np.asarray(t, dtype=float)
    tr = gp_Trsf()
    tr.SetValues(*(float(v) for v in np.column_stack((m, t)).ravel()))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def nurbs(shape):
    """A copy of the shape with every face's surface and every edge's curve converted to a B-spline
    (BRepBuilderAPI_NurbsConvert): planes, cylinders, cones and spheres bounded to the faces' extent."""
    return _occ("BRepBuilderAPI", "BRepBuilderAPI_NurbsConvert")(shape, True).Shape()


def mirrored(shape, origin, normal):
    """A copy of the shape mirrored in the plane through the point origin with the normal given."""
    gp_Trsf, gp_Ax2, gp_Pnt, gp_Dir = _occ("gp", "gp_Trsf", "gp_Ax2", "gp_Pnt", "gp_Dir")
    BRepBuilderAPI_Transform = _occ("BRepBuilderAPI", "BRepBuilderAPI_Transform")
    tr = gp_Trsf()
    tr.SetMirror(gp_Ax2(gp_Pnt(*[float(v) for v in origin]), gp_Dir(*[float(v) for v in normal])))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def _boolean(kind, a, b, fuzzy):
    """BRepAlgoAPI_<kind> of a and b (its constructor builds it), built again with the fuzzy value when one is
    given (a binding without it keeps the first build)."""
    algo = _occ("BRepAlgoAPI", f"BRepAlgoAPI_{kind}")(a, b)
    if fuzzy is not None:
        try:
            algo.SetFuzzyValue(float(fuzzy))
            algo.Build()
        except Exception:                                   # noqa: BLE001 - as prop_hub has done since 13 Nov
            pass
    errors = getattr(algo, "HasErrors", None)                   # not in every binding
    if not algo.IsDone() or (errors is not None and errors()):
        raise RuntimeError(f"the {kind.lower()} of two shapes failed")
    return algo.Shape()


def common(a, b, fuzzy=None):
    """The part of shape a inside shape b (BRepAlgoAPI_Common); RuntimeError when it fails."""
    return _boolean("Common", a, b, fuzzy)


def cut(a, b, fuzzy=None):
    """The part of shape a outside shape b (BRepAlgoAPI_Cut); RuntimeError when it fails."""
    return _boolean("Cut", a, b, fuzzy)


def fuse(a, b, fuzzy=None):
    """Shapes a and b together (BRepAlgoAPI_Fuse); RuntimeError when it fails."""
    return _boolean("Fuse", a, b, fuzzy)


def fuse_all(shapes, fuzzy=None):
    """The union of the shapes in one boolean (BRepAlgoAPI_Fuse: the first the argument, the others the
    tools; run in parallel) with the fuzzy value when one is given; RuntimeError when it fails."""
    TopTools_ListOfShape = _occ("TopTools", "TopTools_ListOfShape")
    args, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    args.Append(shapes[0])
    for s in shapes[1:]:
        tools.Append(s)
    op = _occ("BRepAlgoAPI", "BRepAlgoAPI_Fuse")()
    op.SetArguments(args)
    op.SetTools(tools)
    if fuzzy is not None:
        op.SetFuzzyValue(float(fuzzy))
    op.SetRunParallel(True)
    op.Build()
    errors = getattr(op, "HasErrors", None)                     # not in every binding
    if not op.IsDone() or (errors is not None and errors()):
        raise RuntimeError("the boolean fuse failed")
    return op.Shape()


def unify(shape):
    """The shape with its same-domain faces and edges merged (ShapeUpgrade_UnifySameDomain, B-splines not
    concatenated)."""
    u = _occ("ShapeUpgrade", "ShapeUpgrade_UnifySameDomain")(shape, True, True, False)
    u.Build()
    return u.Shape()


# --------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------

def shapes(shape, kind):
    """The sub-shapes of a kind ('SOLID', 'SHELL', 'FACE', 'WIRE', 'EDGE', 'VERTEX'), each cast to its type, in
    the order TopExp_Explorer finds them (a shared one once for each shape that holds it: a box's 24 edges)."""
    TopExp_Explorer = _occ("TopExp", "TopExp_Explorer")
    cast = _topods(kind.capitalize())
    out, ex = [], TopExp_Explorer(shape, _occ("TopAbs", f"TopAbs_{kind}"))
    while ex.More():
        out.append(cast(ex.Current()))
        ex.Next()
    return out


def faces(shape):
    """The faces of a shape."""
    return shapes(shape, "FACE")


def is_same(a, b):
    """Are a and b the same shape (the same piece of topology in the same place, any orientation)?"""
    return bool(a.IsSame(b))


def free_wires(shape, tol):
    """The closed wires of the shape's free boundary (ShapeAnalysis_FreeBounds, tolerance tol): the loops
    along which its faces have no neighbour."""
    bounds = _occ("ShapeAnalysis", "ShapeAnalysis_FreeBounds")(shape, float(tol), False, False)
    return shapes(bounds.GetClosedWires(), "WIRE")


def count_faces(shape):
    """Number of faces of a shape."""
    return len(faces(shape))


def edge_points(edge, n):
    """n points (n, 3) along an edge, evenly in its curve's parameter, its ends included."""
    c = _occ("BRepAdaptor", "BRepAdaptor_Curve")(edge)
    out = []
    for t in np.linspace(c.FirstParameter(), c.LastParameter(), n):
        p = c.Value(float(t))
        out.append((p.X(), p.Y(), p.Z()))
    return np.array(out)


def surface_points(surface, u, v):
    """The points (n, 3) of a surface (bspline_surface) at the parameter pairs (u[i], v[i])."""
    out = []
    for a, b in zip(u, v):
        p = surface.Value(float(a), float(b))
        out.append((p.X(), p.Y(), p.Z()))
    return np.array(out, dtype=float).reshape(-1, 3)


_SURFACE_KINDS = (("Plane", "plane"), ("Cylinder", "cylinder"), ("Cone", "cone"), ("Sphere", "sphere"),
                  ("Torus", "torus"), ("BezierSurface", "bezier"), ("BSplineSurface", "bspline"),
                  ("SurfaceOfRevolution", "revolution"), ("SurfaceOfExtrusion", "extrusion"),
                  ("OffsetSurface", "offset"))


def surface_kind(face):
    """The kind of the face's surface: 'plane', 'cylinder', 'cone', 'sphere', 'torus', 'bezier', 'bspline',
    'revolution', 'extrusion', 'offset' or 'other'."""
    GeomAdaptor_Surface = _occ("GeomAdaptor", "GeomAdaptor_Surface")
    abs_ = importlib.import_module(f"{_PKG}.GeomAbs")
    kind = GeomAdaptor_Surface(_static(_occ("BRep", "BRep_Tool"), "Surface")(face)).GetType()
    for name, word in _SURFACE_KINDS:
        if kind == getattr(abs_, f"GeomAbs_{name}", None):
            return word
    return "other"


def _gprops(shape, kind, eps):
    """BRepGProp's VolumeProperties or SurfaceProperties of the shape, adaptive to the relative error eps."""
    mod = importlib.import_module(f"{_PKG}.BRepGProp")
    fn = _static(getattr(mod, "brepgprop", None) or getattr(mod, "BRepGProp"), kind)
    props = _occ("GProp", "GProp_GProps")()
    try:
        fn(shape, props, float(eps))
    except TypeError:                                       # a binding without the eps overload
        fn(shape, props)
    return props


def mass_properties(shape, eps=1.0e-7):
    """(volume, area) of a shape (the volume's absolute value), by adaptive integration to the relative error
    eps (the default Gauss rule is 0.4 % off on many-span B-spline faces)."""
    vol = _gprops(shape, "VolumeProperties", eps).Mass()
    return abs(vol), _gprops(shape, "SurfaceProperties", eps).Mass()


def volume(shape, eps=1.0e-7):
    """The shape's volume (its absolute value), adaptive integration to the relative error eps."""
    return abs(_gprops(shape, "VolumeProperties", eps).Mass())


def centroid(shape, eps=1.0e-7):
    """The centre of the shape's volume (3,), adaptive integration to the relative error eps."""
    c = _gprops(shape, "VolumeProperties", eps).CentreOfMass()
    return np.array([c.X(), c.Y(), c.Z()])


def is_valid(shape):
    """Is the shape valid (BRepCheck_Analyzer)?"""
    BRepCheck_Analyzer = _occ("BRepCheck", "BRepCheck_Analyzer")
    return bool(BRepCheck_Analyzer(shape).IsValid())


def bounding_box(shape):
    """The shape's bounding box, (2, 3): the lowest and the highest corner, on the surfaces, not their
    control nets."""
    Bnd_Box = _occ("Bnd", "Bnd_Box")
    mod = importlib.import_module(f"{_PKG}.BRepBndLib")
    b = Bnd_Box()
    _static(getattr(mod, "brepbndlib", None) or getattr(mod, "BRepBndLib"), "AddOptimal")(shape, b, False, False)
    return np.array(b.Get(), dtype=float).reshape(2, 3)


def tessellate(shape, linear=0.5, angular=0.5):
    """Triangles of a shape for display: (vertices (n, 3), triangles (m, 3)), outward-facing (reversed faces
    flipped)."""
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


# --------------------------------------------------------------------------
# Files and display
# --------------------------------------------------------------------------

def write_step(shape, path):
    """Write a shape as STEP; returns the path."""
    STEPControl_Writer, STEPControl_AsIs = _occ("STEPControl", "STEPControl_Writer", "STEPControl_AsIs")
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if int(w.Write(str(path))) != 1:                       # IFSelect_RetDone
        raise RuntimeError(f"writing {path} failed")
    return path


def read_step(path):
    """The shape of a STEP file."""
    STEPControl_Reader = _occ("STEPControl", "STEPControl_Reader")
    r = STEPControl_Reader()
    if int(r.ReadFile(str(path))) != 1:
        raise RuntimeError(f"reading {path} failed")
    r.TransferRoots()
    return r.OneShape()


def write_iges(shape, path, brep=False):
    """Write a shape as IGES in mm: its faces as trimmed surfaces (entity 144, the form CAD and meshing tools
    read), or with brep its solids as BRep solids. Returns the path."""
    IGESControl_Writer = _occ("IGESControl", "IGESControl_Writer")
    w = IGESControl_Writer("MM", 1 if brep else 0)
    w.AddShape(shape)
    w.ComputeModel()
    if not w.Write(str(path)):
        raise RuntimeError(f"writing {path} failed")
    return path


def read_iges(path):
    """The shape of an IGES file."""
    IGESControl_Reader = _occ("IGESControl", "IGESControl_Reader")
    r = IGESControl_Reader()
    if int(r.ReadFile(str(path))) != 1:
        raise RuntimeError(f"reading {path} failed")
    r.TransferRoots()
    return r.OneShape()


def view(shape, title="XGeom CAD"):
    """Show a shape in the pythonocc viewer (needs pythonocc-core and a Qt or wx backend; blocks until the
    window is closed)."""
    if _PKG != "OCC.Core":
        raise ImportError("the viewer needs pythonocc-core (OCC.Display)")
    from OCC.Display.SimpleGui import init_display
    display, start_display, _, _ = init_display()
    display.DisplayShape(shape, update=True)
    display.FitAll()
    print(f"{title}: close the viewer window to go on")
    start_display()
