"""
Rudder_geom_extraction.py

Import an existing rudder CAD file (STEP / IGES) and extract the geometric
metrics needed by the parametrisation pipeline.

Pipeline implemented here
-------------------------
Step 1 : Import the CAD.
Step 2 : Reframe so that the BOTTOM (root) LEADING-EDGE point is (0, 0, 0) and
         +X runs chordwise from LE to TE.
Step 3 : Take a horizontal planar slice at a given spanwise height y.
Step 4 : Extract the LE / TE corner points of the root section and of the cut
         section (the four "red dots") and store them.
Step 5 : Extract the 2D airfoil coordinates of the root section and of the cut
         section at `num_pts` points per section (default 200), and the whole
         stack: `num_sec` horizontal sections evenly spaced from the root up to
         a gap d below the top of the LE (d = half the spacing by default), and
         on top of them one tilted tip section, the planar cut through the top
         of the LE and the top of the TE. The tip section is also written
         untilted (rotated back about the thickness axis through its LE), so
         its airfoil can be read directly. The stack is also written in XCAD's
         point format (see `write_xcad`).
Step 6 : Parametrise that stack root-to-tip: chord, sweep, rake / dihedral,
         twist, the section tilt and the section shape (t/c, camber, LE
         radius) as spanwise distributions, each fitted with a constant, a
         straight line and a polynomial so a rudder that really is "4 deg aft
         all the way up with no rake and no twist" reports itself as exactly
         that. The tolerances only decide how a distribution is *described*;
         they never clip it. The fits use the horizontal sections only; the
         tilt of the tip section is reported on its own, as the design
         variable it is meant to become.
Step 7 : Parametrise the tip cap, the part above the tilted tip section, with
         the model in tip_cap.py: an inclined face over the tip section
         (heights at its two ends plus a small bow), crowned across the
         thickness, with a constant edge radius all round and a corner radius
         at the TE end. The CAD cap is cut normal to the tip chord and the
         model is least-squares fitted to those cross-sections. The fitted cap
         is written as a point grid, and as extra loops on top of the stack in
         a second XCAD file.

Working frame after Step 2 (standard aerodynamic convention):
    origin = root leading-edge point
    +X     = chordwise, LE -> TE, so the root TE sits at (chord, 0, 0)
    +Y     = spanwise, root -> tip
    +Z     = thickness

The incoming CAD is reframed into that, whichever way round it arrives. The
Wind_Tunnel_Rudder file has +X pointing TE -> LE, so it is rotated 180 deg
about the span axis rather than mirrored: a mirror would leave a left-handed
frame and quietly invert every surface normal and cross product downstream.
Which end is the LE is decided by bluntness, not by assuming a sign.

The exported 2D airfoil is that frame with the span coordinate dropped:
    x_af = x - x_LE(section)    (0 at LE, chord at TE)
    y_af = z
i.e. the section seen looking up the span from the root.

Requires pythonocc-core (`OCC.Core`); transparently falls back to
cadquery-ocp (`OCP`) if that is what is installed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass

import numpy as np

# Rotation helpers from the X_geom repo, and the tip-cap model; they sit next to
# this script.
try:
    from rot_axis import rot_axis                            # Rodrigues rotation matrix
    from skew_symmetric_matrix import skew_symmetric_matrix  # [v]x, the cross-product matrix
    import tip_cap                                           # Step 7, the cap model
except ImportError as _exc:                                  # pragma: no cover
    raise ImportError(
        "Rudder_geom_extraction.py needs rot_axis.py, skew_symmetric_matrix.py "
        "and tip_cap.py (from the X_geom repo) in the same folder"
    ) from _exc

# --------------------------------------------------------------------------- #
# OCCT binding shim: pythonocc-core (OCC.Core) or cadquery-ocp (OCP)
# --------------------------------------------------------------------------- #
_PKG = None
for _cand in ("OCC.Core", "OCP"):
    try:
        importlib.import_module(_cand + ".gp")
        _PKG = _cand
        break
    except ImportError:
        continue
if _PKG is None:
    raise ImportError(
        "No OCCT Python binding found. Install pythonocc-core "
        "(conda install -c conda-forge pythonocc-core) or cadquery-ocp."
    )


def _occ(module: str, *names):
    """Import `names` from the OCCT sub-module, whichever binding is present."""
    mod = importlib.import_module(f"{_PKG}.{module}")
    objs = tuple(getattr(mod, n) for n in names)
    return objs[0] if len(objs) == 1 else objs


def _static(owner, name):
    """OCP exposes static methods as `Name_s`, pythonocc as `Name`."""
    for cand in (name, name + "_s"):
        if hasattr(owner, cand):
            return getattr(owner, cand)
    raise AttributeError(f"{owner!r} has no static method {name!r}")


gp_Pnt, gp_Dir, gp_Pln, gp_Vec, gp_Trsf, gp_Ax1 = _occ(
    "gp", "gp_Pnt", "gp_Dir", "gp_Pln", "gp_Vec", "gp_Trsf", "gp_Ax1"
)
STEPControl_Reader = _occ("STEPControl", "STEPControl_Reader")
IGESControl_Reader = _occ("IGESControl", "IGESControl_Reader")
Bnd_Box = _occ("Bnd", "Bnd_Box")
TopExp_Explorer = _occ("TopExp", "TopExp_Explorer")
TopAbs_FACE, TopAbs_EDGE = _occ("TopAbs", "TopAbs_FACE", "TopAbs_EDGE")
TopoDS_Compound = _occ("TopoDS", "TopoDS_Compound")
BRep_Builder = _occ("BRep", "BRep_Builder")
BRepAdaptor_Curve, BRepAdaptor_Surface = _occ(
    "BRepAdaptor", "BRepAdaptor_Curve", "BRepAdaptor_Surface"
)
GeomAbs_Plane = _occ("GeomAbs", "GeomAbs_Plane")
BRepAlgoAPI_Section = _occ("BRepAlgoAPI", "BRepAlgoAPI_Section")
BRepBuilderAPI_Transform = _occ("BRepBuilderAPI", "BRepBuilderAPI_Transform")
GCPnts_QuasiUniformAbscissa = _occ("GCPnts", "GCPnts_QuasiUniformAbscissa")

# static helpers whose spelling differs between bindings
_bnd_mod = importlib.import_module(f"{_PKG}.BRepBndLib")
_bnd_owner = getattr(_bnd_mod, "brepbndlib", None) or getattr(_bnd_mod, "BRepBndLib")
_bnd_add = _static(_bnd_owner, "Add")
try:
    _bnd_add_optimal = _static(_bnd_owner, "AddOptimal")
except AttributeError:                                       # pragma: no cover
    _bnd_add_optimal = None

_tds_mod = importlib.import_module(f"{_PKG}.TopoDS")
_tds_owner = getattr(_tds_mod, "topods", None) or getattr(_tds_mod, "TopoDS")
_as_face = _static(_tds_owner, "Face")
_as_edge = _static(_tds_owner, "Edge")

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
CHORD_AXIS = 0   # X
SPAN_AXIS = 1    # Y
THICK_AXIS = 2   # Z

# In the working frame the chord runs LE -> TE, so the LE is the chordwise
# minimum of a section and the TE is the maximum.
LE_SIGN = -1
TE_SIGN = +1

_SPAN_DIR = [gp_Dir(1, 0, 0), gp_Dir(0, 1, 0), gp_Dir(0, 0, 1)][SPAN_AXIS]

# The tip section is tilted about the thickness axis: its plane still contains
# the thickness direction, but it climbs from the top of the LE to the top of
# the TE instead of lying at one height.
TILT_AXIS = np.eye(3)[THICK_AXIS]

DEFAULT_NUM_PTS = 200     # points per airfoil section
DEFAULT_NUM_SEC = 200     # spanwise cross-sections in the stack
_DENSE_PER_EDGE = 800     # polyline density used before resampling


# --------------------------------------------------------------------------- #
# Step 1 - import the CAD
# --------------------------------------------------------------------------- #
def load_cad(path: str):
    """Read a STEP (.stp/.step) or IGES (.igs/.iges) file into a TopoDS_Shape."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".stp", ".step"):
        reader = STEPControl_Reader()
    elif ext in (".igs", ".iges"):
        reader = IGESControl_Reader()
    else:
        raise ValueError(f"Unsupported CAD format: {ext}")

    if int(reader.ReadFile(path)) != 1:          # 1 == IFSelect_RetDone
        raise IOError(f"Could not read CAD file: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise IOError(f"CAD file produced a null shape: {path}")
    return shape


def bounding_box(shape):
    """
    Tight bounding box as two numpy points.

    `BRepBndLib.Add` without a triangulation boxes the control net of each
    B-spline face, and on a trimmed face that is the untrimmed surface. The
    wind-tunnel rudder's side surfaces are a loft that runs to about 218 mm
    before the tip trims it, so `Add` reports a top of 218.07 mm for a part
    that ends at 211.36 mm. `AddOptimal` bounds the trimmed geometry itself.
    """
    box = Bnd_Box()
    if _bnd_add_optimal is not None:
        _bnd_add_optimal(shape, box, False, False)
    else:                                                    # pragma: no cover
        _bnd_add(shape, box, False)
    box.SetGap(0.0)
    lo, hi = box.CornerMin(), box.CornerMax()
    return (np.array([lo.X(), lo.Y(), lo.Z()]),
            np.array([hi.X(), hi.Y(), hi.Z()]))


# --------------------------------------------------------------------------- #
# Section machinery
# --------------------------------------------------------------------------- #
def lateral_faces(shape, cap_tol: float = 1.0e-6):
    """
    Compound of every face except the flat root / tip caps.

    Slicing the full solid exactly on the root plane makes the section
    algorithm return the cap face boundary *and* its internal seam line.
    Removing the span-normal caps first gives a clean closed loop at any
    height, root plane included.

    A cap is detected geometrically - zero extent along the span axis - so
    it is caught whether the exporter wrote it as an analytic plane (STEP)
    or as a flat B-spline patch (IGES).
    """
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    caps = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = _as_face(exp.Current())
        lo, hi = bounding_box(face)
        is_cap = (hi[SPAN_AXIS] - lo[SPAN_AXIS]) <= cap_tol
        if not is_cap:                       # analytic-plane fallback
            surf = BRepAdaptor_Surface(face)
            if surf.GetType() == GeomAbs_Plane:
                normal = surf.Plane().Axis().Direction()
                is_cap = abs(normal.Dot(_SPAN_DIR)) > 0.999
        if is_cap:
            caps.append(0.5 * (lo[SPAN_AXIS] + hi[SPAN_AXIS]))
        else:
            builder.Add(comp, face)
        exp.Next()
    return comp, caps


def root_height(shape) -> float:
    """Spanwise station of the root plane (flat cap if present, else bbox min)."""
    _, caps = lateral_faces(shape)
    if caps:
        return float(min(caps))
    return float(bounding_box(shape)[0][SPAN_AXIS])


def _curve_polyline(curve, n_pts: int) -> np.ndarray:
    """Sample one curve into an (n, 3) polyline, uniform in arc length."""
    try:
        distrib = GCPnts_QuasiUniformAbscissa(curve, n_pts)
        params = [distrib.Parameter(i) for i in range(1, distrib.NbPoints() + 1)]
    except Exception:                                    # pragma: no cover
        params = np.linspace(curve.FirstParameter(), curve.LastParameter(), n_pts)
    pts = np.empty((len(params), 3))
    for i, u in enumerate(params):
        p = curve.Value(u)
        pts[i] = (p.X(), p.Y(), p.Z())
    return pts


def _edge_polyline(edge, n_pts: int) -> np.ndarray:
    return _curve_polyline(BRepAdaptor_Curve(edge), n_pts)


def _plane_curves(lateral, plane, what: str):
    """The curves of one planar cut through the lateral faces."""
    algo = BRepAlgoAPI_Section(lateral, plane, False)
    algo.ComputePCurveOn1(True)
    algo.Approximation(True)
    algo.Build()
    if not algo.IsDone():
        raise RuntimeError(f"Section {what} failed")

    curves = []
    exp = TopExp_Explorer(algo.Shape(), TopAbs_EDGE)
    while exp.More():
        curves.append(BRepAdaptor_Curve(_as_edge(exp.Current())))
        exp.Next()
    if not curves:
        raise RuntimeError(f"Section {what} is empty - outside the part?")
    return curves


def section_curves(lateral, y: float):
    """
    Step 3 - one horizontal planar cut at spanwise station `y`.

    Returns the section's curves. Everything else about a station (outline,
    LE, TE) is derived from this one result, so a station costs one cut.
    """
    plane = gp_Pln(gp_Pnt(*[0.0 if i != SPAN_AXIS else y for i in range(3)]), _SPAN_DIR)
    return _plane_curves(lateral, plane, f"at y = {y}")


def tilted_section_curves(lateral, point: np.ndarray, normal: np.ndarray):
    """One planar cut through `point` with unit `normal`, for the tilted section."""
    plane = gp_Pln(gp_Pnt(*[float(v) for v in point]),
                   gp_Dir(*[float(v) for v in normal]))
    return _plane_curves(lateral, plane, f"through {np.round(point, 6).tolist()}")


def section_polyline(lateral, y: float, dense: int = _DENSE_PER_EDGE) -> np.ndarray:
    """Closed, ordered (N, 3) polyline of the section outline at station `y`."""
    return _chain([_curve_polyline(c, dense) for c in section_curves(lateral, y)])


def _chain(strips) -> np.ndarray:
    """
    Order and orient a bag of polylines into one closed loop.

    The joining tolerance is relative to the section's own size: the section
    algorithm approximates its output curves, and near the tip the endpoint
    mismatch reaches ~5e-5 mm on a 180 mm chord, which is not a real gap.
    """
    extent = float(np.ptp(np.vstack(strips), axis=0).max())
    tol = max(1.0e-9, 1.0e-5 * extent)

    remaining = list(strips)
    loop = [remaining.pop(0)]
    tail = loop[-1][-1]

    while remaining:
        best_i, best_flip, best_d = None, False, np.inf
        for i, s in enumerate(remaining):
            for flip, end in ((False, s[0]), (True, s[-1])):
                d = float(np.linalg.norm(end - tail))
                if d < best_d:
                    best_i, best_flip, best_d = i, flip, d
        if best_d > tol:
            raise RuntimeError(
                f"Section outline is not closed (gap {best_d:.3e} > {tol:.3e})")
        s = remaining.pop(best_i)
        if best_flip:
            s = s[::-1]
        loop.append(s[1:])
        tail = s[-1]

    pts = np.vstack(loop)
    if np.linalg.norm(pts[0] - pts[-1]) < tol:
        pts = pts[:-1]
    return pts


def _extreme_on_curves(curves, sign: int, direction=None) -> np.ndarray:
    """
    Chordwise extremum over a section's curves (sign = LE_SIGN or TE_SIGN),
    bracketed on a dense parameter sample then refined by golden section on
    the curve itself, not read off a polyline.

    `direction` is the chordwise axis to measure along; the default is the
    global X. A tilted section passes its own chord direction instead.
    """
    e = None if direction is None else np.asarray(direction, float) / np.linalg.norm(direction)
    best_val, best_pnt = -np.inf, None
    for curve in curves:
        u0, u1 = curve.FirstParameter(), curve.LastParameter()

        if e is None:
            def f(u):
                p = curve.Value(u)
                return sign * [p.X(), p.Y(), p.Z()][CHORD_AXIS]
        else:
            def f(u):
                p = curve.Value(u)
                return sign * (e[0] * p.X() + e[1] * p.Y() + e[2] * p.Z())

        us = np.linspace(u0, u1, 400)
        vals = np.array([f(u) for u in us])
        k = int(np.argmax(vals))
        a, b = us[max(k - 1, 0)], us[min(k + 1, len(us) - 1)]

        phi = (np.sqrt(5.0) - 1.0) / 2.0
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(80):
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = f(d)
            if abs(b - a) < 1.0e-12:
                break
        u = 0.5 * (a + b)
        if f(u) > best_val:
            best_val = f(u)
            p = curve.Value(u)
            best_pnt = np.array([p.X(), p.Y(), p.Z()])

    if best_pnt is None:
        raise RuntimeError("No section geometry to take an extremum on")
    return best_pnt


def trailing_edge_point(lateral, y: float) -> np.ndarray:
    """Chordwise-aft extreme point of the section at station `y`."""
    return _extreme_on_curves(section_curves(lateral, y), TE_SIGN)


def leading_edge_point(lateral, y: float) -> np.ndarray:
    """Chordwise-forward extreme point of the section at station `y`."""
    return _extreme_on_curves(section_curves(lateral, y), LE_SIGN)


def le_end_is_chordwise_max(outline: np.ndarray, frac: float = 0.05) -> bool:
    """
    Which chordwise end of a section is the leading edge, decided by bluntness.

    Used once, on the incoming CAD, so the reframing does not have to assume
    which way round the file was drawn. A short way in from the nose an
    airfoil is several times thicker than it is the same distance in from the
    tail, whatever the sign convention of the file.
    """
    c = outline[:, CHORD_AXIS]
    t = outline[:, THICK_AXIS]
    c_lo, c_hi = float(c.min()), float(c.max())
    chord = c_hi - c_lo
    band = max(1.0e-9, 0.005 * chord)

    def thickness_at(target):
        sel = np.abs(c - target) < band
        return float(np.ptp(t[sel])) if sel.sum() >= 2 else 0.0

    return thickness_at(c_hi - frac * chord) > thickness_at(c_lo + frac * chord)


# --------------------------------------------------------------------------- #
# Where the leading edge stops
# --------------------------------------------------------------------------- #
def _point_to_polyline(q: np.ndarray, poly: np.ndarray) -> float:
    """Shortest distance from a point to a polyline, segment-exact."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom == 0.0] = 1.0
    t = np.clip(np.einsum("ij,ij->i", q - a, ab) / denom, 0.0, 1.0)
    return float(np.linalg.norm(q - (a + t[:, None] * ab), axis=1).min())


def _edge_top(shape, lateral, span: float, point_at, names, n_probe: int = 10):
    """
    Top end of the edge traced by `point_at(lateral, y)` - the LE or the TE
    point of a horizontal section - as a 3D point, with how it was found.

    Preferred route: probe the point over the lower span, find the B-rep edges
    those points actually lie on, and take the top of that curve. That is exact
    whenever the exporter wrote the edge as an edge, which is the usual case
    for a lofted surface. Falls back to bisecting where the point leaves the
    locus extrapolated from below.

    `names` = (long, short) name of the edge for the `how` string.
    """
    probes = np.linspace(0.02, 0.60, n_probe) * span
    pts = [point_at(lateral, float(y)) for y in probes]
    tol = max(1.0e-7, 1.0e-6 * span)

    best = None
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        poly = _edge_polyline(_as_edge(exp.Current()), 200)
        if any(_point_to_polyline(q, poly) < tol for q in pts):
            top = poly[int(np.argmax(poly[:, SPAN_AXIS]))]
            if best is None or top[SPAN_AXIS] > best[SPAN_AXIS]:
                best = top.copy()
        exp.Next()
    if best is not None:
        return best, f"top of the b-rep {names[0]} curve"

    # fallback - the locus is smooth below the break and leaves it after
    fit = np.polyfit(probes, [p[CHORD_AXIS] for p in pts], 1)

    def on_locus(y):
        try:
            return abs(point_at(lateral, y)[CHORD_AXIS] - np.polyval(fit, y)) < tol
        except RuntimeError:
            return False

    lo, hi = float(probes[-1]), span
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if on_locus(mid):
            lo = mid
        else:
            hi = mid
    return point_at(lateral, lo), f"bisection on the extrapolated {names[1]} locus"


def leading_edge_top_point(shape, lateral, span: float, n_probe: int = 10):
    """
    Where the leading edge ends, as a 3D point: (point, how).

    The tip surface is inclined, so above this station a horizontal cut no
    longer produces a complete airfoil - the nose of the cut is the tip blend
    rather than the true leading edge.
    """
    return _edge_top(shape, lateral, span, leading_edge_point,
                     ("leading-edge", "LE"), n_probe)


def trailing_edge_top_point(shape, lateral, span: float, n_probe: int = 10):
    """Where the trailing edge ends, as a 3D point: (point, how)."""
    return _edge_top(shape, lateral, span, trailing_edge_point,
                     ("trailing-edge", "TE"), n_probe)


def leading_edge_top(shape, lateral, span: float, n_probe: int = 10):
    """Spanwise station where the leading edge ends: (y_top, how)."""
    point, how = leading_edge_top_point(shape, lateral, span, n_probe)
    return float(point[SPAN_AXIS]), how


# --------------------------------------------------------------------------- #
# Step 2 - move the origin to the root leading edge
# --------------------------------------------------------------------------- #
def translate_shape(shape, vec: np.ndarray):
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(float(vec[0]), float(vec[1]), float(vec[2])))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def reframe_to_root_le(shape):
    """
    Put the root leading-edge point at (0, 0, 0) with +X running LE -> TE.

    If the incoming CAD already has its chord that way round this is a plain
    translation. If it runs TE -> LE, as the wind-tunnel rudder does, the
    shape is turned 180 deg about the span axis through the root LE. That is
    a rotation, not a mirror, so the frame stays right-handed; mirroring would
    give the same tidy numbers while inverting every normal downstream.

    Returns (new_shape, trsf, original_root_le_point, rotated).
    """
    lateral, _ = lateral_faces(shape)
    y_root = root_height(shape)
    curves = section_curves(lateral, y_root)
    outline = _chain([_curve_polyline(c, _DENSE_PER_EDGE) for c in curves])

    rotated = le_end_is_chordwise_max(outline)
    le = _extreme_on_curves(curves, +1 if rotated else -1)

    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(*(-le)))
    if rotated:
        spin = gp_Trsf()
        spin.SetRotation(gp_Ax1(gp_Pnt(*le), _SPAN_DIR), np.pi)
        trsf = trsf.Multiplied(spin)          # spin first, then translate

    return BRepBuilderAPI_Transform(shape, trsf, True).Shape(), trsf, le, rotated


# --------------------------------------------------------------------------- #
# Steps 4 & 5 - section data
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    y: float                 # spanwise station; on a tilted section, the height of its LE
    le: np.ndarray           # leading-edge point  (3,), where it sits on the part
    te: np.ndarray           # trailing-edge point (3,), where it sits on the part
    chord: float             # LE -> TE, chordwise, in the section's own (untilted) frame
    outline: np.ndarray      # dense closed loop, (N, 3), where it sits on the part
    airfoil_raw: np.ndarray  # (num_pts, 2) mm,  Selig order, airfoil frame (untilted)
    airfoil_norm: np.ndarray  # (num_pts, 2) normalised by chord
    tilt_deg: float = 0.0    # plane turned about the thickness axis through the LE,
                             # positive when the TE is higher; 0 for a horizontal cut

    @property
    def tilted(self) -> bool:
        return self.tilt_deg != 0.0

    def _turn(self, pts, sign: float) -> np.ndarray:
        """Rotate (N, 3) points by sign * tilt about the thickness axis through
        the LE. The pivot height is `y`, so the untilted section lies at y."""
        pts = np.asarray(pts, dtype=float)
        if not self.tilted:
            return pts.copy()
        pivot = np.zeros(3)
        pivot[CHORD_AXIS] = self.le[CHORD_AXIS]
        pivot[SPAN_AXIS] = self.y
        rot = rot_axis(TILT_AXIS, sign * np.radians(self.tilt_deg))
        return pivot + (pts - pivot) @ rot.T

    def untilt(self, pts) -> np.ndarray:
        """Points on the part -> the section's own horizontal frame at height y."""
        return self._turn(pts, -1.0)

    def retilt(self, pts) -> np.ndarray:
        """The inverse of `untilt`: back to where the section sits on the part."""
        return self._turn(pts, +1.0)

    def points_3d(self, xy: np.ndarray) -> np.ndarray:
        """
        Airfoil-frame points (x aft of this section's LE, y = thickness) -> 3D
        in the working frame, where the section sits on the part.
        """
        xy = np.asarray(xy, dtype=float)
        pts = np.empty((len(xy), 3))
        pts[:, CHORD_AXIS] = self.le[CHORD_AXIS] + xy[:, 0]
        pts[:, SPAN_AXIS] = self.y
        pts[:, THICK_AXIS] = xy[:, 1]
        return self.retilt(pts) if self.tilted else pts

    def as_dict(self):
        return {
            "y": self.y,
            "chord": self.chord,
            "le": self.le.tolist(),
            "te": self.te.tolist(),
            "tilt_deg": self.tilt_deg,
        }


def _split_surfaces(outline: np.ndarray, le: np.ndarray, te: np.ndarray):
    """Cut the closed loop at LE and TE; return (upper, lower), each LE -> TE."""
    i_le = int(np.argmin(np.linalg.norm(outline - le, axis=1)))
    i_te = int(np.argmin(np.linalg.norm(outline - te, axis=1)))

    rolled = np.roll(outline, -i_le, axis=0)
    j_te = (i_te - i_le) % len(outline)

    arc_a = np.vstack([le, rolled[1:j_te], te])            # LE -> ... -> TE
    arc_b = np.vstack([le, rolled[:j_te:-1], te])          # LE -> other way -> TE

    if arc_a[:, THICK_AXIS].mean() >= arc_b[:, THICK_AXIS].mean():
        return arc_a, arc_b
    return arc_b, arc_a


def _to_airfoil_frame(pts: np.ndarray, le: np.ndarray) -> np.ndarray:
    """3D section points -> 2D airfoil frame: drop the span coordinate and put
    the section's own LE at x = 0. x runs LE -> TE, y is thickness."""
    x = pts[:, CHORD_AXIS] - le[CHORD_AXIS]
    y = pts[:, THICK_AXIS]
    return np.column_stack([x, y])


def _resample(arc2d: np.ndarray, n: int, spacing: str) -> np.ndarray:
    """Resample an LE->TE surface to n points; cosine clusters at LE and TE."""
    seg = np.linalg.norm(np.diff(arc2d, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s /= s[-1]

    if spacing == "cosine":
        beta = np.linspace(0.0, np.pi, n)
        frac = (1.0 - np.cos(beta)) / 2.0
        x = arc2d[:, 0]
        if np.all(np.diff(x) > 0):                     # x monotone LE -> TE
            x_t = x[0] + frac * (x[-1] - x[0])
            s_t = np.interp(x_t, x, s)
        else:                                          # fall back to arc length
            s_t = frac
    elif spacing == "uniform":
        s_t = np.linspace(0.0, 1.0, n)
    else:
        raise ValueError(f"Unknown spacing: {spacing}")

    return np.column_stack([np.interp(s_t, s, arc2d[:, 0]),
                            np.interp(s_t, s, arc2d[:, 1])])


def _airfoil_coords(outline: np.ndarray, le: np.ndarray, te: np.ndarray,
                    num_pts: int, spacing: str) -> np.ndarray:
    """A horizontal section outline -> `num_pts` airfoil-frame points, Selig order."""
    upper, lower = _split_surfaces(outline, le, te)
    up2, lo2 = _to_airfoil_frame(upper, le), _to_airfoil_frame(lower, le)

    n_half = (num_pts + 1) // 2
    up = _resample(up2, n_half, spacing)[::-1]              # TE -> LE
    lo = _resample(lo2, num_pts - n_half + 1, spacing)      # LE -> TE
    return np.vstack([up, lo[1:]])                          # Selig order


def extract_section(lateral, y: float, num_pts: int = DEFAULT_NUM_PTS,
                    spacing: str = "cosine",
                    dense: int = _DENSE_PER_EDGE) -> Section:
    """Steps 3-5 for a single spanwise station, from a single planar cut."""
    curves = section_curves(lateral, y)
    outline = _chain([_curve_polyline(c, dense) for c in curves])
    le = _extreme_on_curves(curves, LE_SIGN)
    te = _extreme_on_curves(curves, TE_SIGN)
    chord = float(te[CHORD_AXIS] - le[CHORD_AXIS])

    raw = _airfoil_coords(outline, le, te, num_pts, spacing)
    return Section(y=float(y), le=le, te=te, chord=chord, outline=outline,
                   airfoil_raw=raw, airfoil_norm=raw / chord)


def tilted_plane(le_top: np.ndarray, te_top: np.ndarray):
    """
    The plane through the top of the LE and the top of the TE that still
    contains the thickness axis: (unit normal, unit chord direction, tilt_deg).

    The normal is thickness-axis x chord, formed with the skew-symmetric
    (cross-product) matrix; for a level chord it is +span, like a horizontal
    cut. The tilt is the rise of the chord over its run, positive TE up.
    """
    c = np.asarray(te_top, float) - np.asarray(le_top, float)
    normal = skew_symmetric_matrix(TILT_AXIS) @ c
    normal = normal / np.linalg.norm(normal)
    tilt_deg = float(np.degrees(np.arctan2(c[SPAN_AXIS], c[CHORD_AXIS])))
    chord_dir = rot_axis(TILT_AXIS, np.radians(tilt_deg)) @ np.eye(3)[CHORD_AXIS]
    return normal, chord_dir, tilt_deg


def extract_tilted_section(lateral, le_top: np.ndarray, te_top: np.ndarray,
                           num_pts: int = DEFAULT_NUM_PTS, spacing: str = "cosine",
                           dense: int = _DENSE_PER_EDGE) -> Section:
    """
    The tip section: one planar cut through the top of the LE and the top of
    the TE, tilted about the thickness axis (see `tilted_plane`).

    Its LE and TE are the chordwise extremes along its own chord direction.
    The outline is then turned back by -tilt about the thickness axis through
    the LE (`rot_axis`), which lays it flat at the height of the LE, and from
    there it is resampled exactly like a horizontal section. So `airfoil_raw`
    is the untilted airfoil, while `le`, `te` and `outline` stay where the
    section sits on the part; `Section.points_3d` puts any airfoil-frame
    points back on the tilted plane.
    """
    normal, chord_dir, tilt_deg = tilted_plane(le_top, te_top)
    curves = tilted_section_curves(lateral, le_top, normal)
    outline = _chain([_curve_polyline(c, dense) for c in curves])
    le = _extreme_on_curves(curves, LE_SIGN, chord_dir)
    te = _extreme_on_curves(curves, TE_SIGN, chord_dir)

    sec = Section(y=float(le[SPAN_AXIS]), le=le, te=te, chord=float("nan"),
                  outline=outline, airfoil_raw=np.empty((0, 2)),
                  airfoil_norm=np.empty((0, 2)), tilt_deg=tilt_deg)
    flat = sec.untilt(outline)
    le_f, te_f = sec.untilt(np.vstack([le, te]))
    sec.chord = float(te_f[CHORD_AXIS] - le_f[CHORD_AXIS])
    sec.airfoil_raw = _airfoil_coords(flat, le_f, te_f, num_pts, spacing)
    sec.airfoil_norm = sec.airfoil_raw / sec.chord
    return sec


def tip_spacing(y_le_top: float, num_sec: int, d: float | None = None):
    """
    Layout of the horizontal sections under the tilted tip section:
    (dz, d). They run evenly from the root to y_le_top - d.

    By default d = dz / 2, half a step below the top of the LE, with dz the
    step itself, so dz = y_le_top / (num_sec - 1/2). An explicit `d` (mm)
    fixes the gap instead and dz follows from it.
    """
    if num_sec < 2:
        raise ValueError("num_sec must be at least 2")
    if d is None:
        dz = y_le_top / (num_sec - 0.5)
        return dz, 0.5 * dz
    d = float(d)
    if not 0.0 <= d < y_le_top:
        raise ValueError(f"tip gap d = {d} must lie in [0, {y_le_top})")
    return (y_le_top - d) / (num_sec - 1), d


def extract_stack(lateral, y_top: float, num_sec: int = DEFAULT_NUM_SEC,
                  num_pts: int = DEFAULT_NUM_PTS, spacing: str = "cosine",
                  y_root: float = 0.0, progress=None):
    """
    `num_sec` horizontal cross-sections evenly spaced from the root up to
    `y_top`. Returned bottom-to-top; the writer flips them.

    Horizontal cuts stop at the top of the leading edge: above it the tip
    surface is inclined, so a horizontal cut there cuts through the tip blend
    and no longer returns a complete airfoil. `run` stops this stack d below
    the top of the LE and puts the tilted tip section on it
    (`extract_tilted_section`).
    """
    if num_sec < 2:
        raise ValueError("num_sec must be at least 2")
    ys = np.linspace(y_root, y_top, num_sec)
    out = []
    for i, y in enumerate(ys):
        out.append(extract_section(lateral, float(y), num_pts, spacing))
        if progress is not None:
            progress(i + 1, num_sec)
    return out


# --------------------------------------------------------------------------- #
# Step 6 - spanwise parametrisation
# --------------------------------------------------------------------------- #
DEFAULT_FIT_DEG = 5       # degree of the polynomial fitted to each distribution

# Bands inside which a distribution is *described* as constant or linear. They
# are tolerances on a fit residual, nothing else: a rudder that is genuinely
# swept 30 deg, raked or twisted reports 30 deg, the rake and the twist. The
# numbers are never clipped to an expectation.
TOL_ANGLE_DEG = 0.05          # sweep, dihedral, twist
TOL_LENGTH_REL = 1.0e-4       # chord (of the root chord), rake (of the span)
TOL_SHAPE = 1.0e-4            # t/c, camber/c, LE radius/c
TOL_POSITION = 1.0e-3         # chordwise position of max thickness / camber

# Below this the camber line is indistinguishable from zero, so the chordwise
# position of its maximum carries no information and is reported as NaN.
CAMBER_FLOOR = 1.0e-6         # of the chord


def _section_chord_line(section: "Section"):
    """
    A section's own chord line: (le, te, chord, twist_deg).

    The TE is the chordwise-aft extremum already found for the section. The LE
    is the outline point farthest from it, not the chordwise-forward extremum,
    so a section rotated about the span axis reports its true chord length and
    the rotation shows up as twist instead of quietly shortening the chord.
    With no twist the two LE definitions coincide.

    Twist is positive nose-up: the LE sits on the +Z side of the TE.

    Farthest-point-from-the-TE is the textbook chord, and on a symmetric
    section it is also the camber-line axis, so the twist reported for a
    symmetric rudder is its true rotation. On a cambered section the two part
    company - the geometric nose sits off the construction axis, by 0.19 deg on
    a NACA 4412 - and the twist then carries that fixed offset. It is the same
    at every station, being a property of the section shape and not of the
    stack, so twist *differences* up the span stay exact either way.
    """
    te = section.te
    le = section.outline[int(np.argmax(np.linalg.norm(section.outline - te, axis=1)))]
    dx = float(te[CHORD_AXIS] - le[CHORD_AXIS])
    dz = float(te[THICK_AXIS] - le[THICK_AXIS])
    return le, te, float(np.hypot(dx, dz)), float(np.degrees(np.arctan2(-dz, dx)))


def _chord_frame(pts: np.ndarray, le: np.ndarray, te: np.ndarray) -> np.ndarray:
    """
    Section points -> chord-aligned, chord-normalised 2D: x along LE -> TE from
    0 to 1, y perpendicular to it. Unlike `_to_airfoil_frame` this removes the
    section's twist, so thickness and camber are measured off the chord line
    itself rather than off the global X axis.
    """
    ex = np.array([te[CHORD_AXIS] - le[CHORD_AXIS], te[THICK_AXIS] - le[THICK_AXIS]])
    chord = float(np.linalg.norm(ex))
    ex = ex / chord
    ey = np.array([-ex[1], ex[0]])
    d = pts[:, [CHORD_AXIS, THICK_AXIS]] - le[[CHORD_AXIS, THICK_AXIS]]
    return np.column_stack([d @ ex, d @ ey]) / chord


def _sample_surface(xs: np.ndarray, arc: np.ndarray) -> np.ndarray:
    """One LE->TE surface sampled onto the x/c grid `xs`."""
    order = np.argsort(arc[:, 0], kind="stable")
    x, y = arc[order, 0], arc[order, 1]
    keep = np.concatenate([[True], np.diff(x) > 0.0])
    return np.interp(xs, x[keep], y[keep])


def _peak(xs: np.ndarray, v: np.ndarray):
    """
    Where `v` peaks and how big the peak is, refined below the sampling grid by
    a parabola through the peak sample and its two neighbours. Without that the
    answer can only land on a grid point, which makes a smoothly drifting
    position look like a staircase - or like a perfect constant.

    Handles a negative peak (camber below the chord line) by working on |v| and
    putting the sign back.
    """
    s = 1.0 if v[int(np.argmax(np.abs(v)))] >= 0.0 else -1.0
    w = s * v
    i = int(np.argmax(w))
    if 0 < i < len(w) - 1:
        p = np.polyfit(xs[i - 1:i + 2], w[i - 1:i + 2], 2)
        if p[0] < 0.0:
            x = -p[1] / (2.0 * p[0])
            if xs[i - 1] <= x <= xs[i + 1]:
                return float(x), float(s * np.polyval(p, x))
    return float(xs[i]), float(v[i])


def _le_radius(xs: np.ndarray, thick: np.ndarray, window: float = 0.05) -> float:
    """
    Leading-edge radius over chord, from the thickness distribution.

    Near a rounded nose the half-thickness goes as sqrt(2 r x), so
    q(x) = (t/2)^2 / (2x) tends to r as x -> 0. Fitting q over a short window
    and extrapolating to the nose is much steadier than fitting a circle to the
    nose points: the circle fit is dragged off by the first few percent of
    chord, where the section has already stopped being circular, and on a NACA
    section it lands 60 % high.
    """
    sel = (xs > 0.0) & (xs <= window)
    if sel.sum() < 4:
        return float("nan")
    s = np.sqrt(xs[sel])
    q = (0.5 * thick[sel]) ** 2 / (2.0 * xs[sel])
    r = float(np.polyval(np.polyfit(s, q, 2), 0.0))
    return r if r > 0.0 else float("nan")


def section_shape(section: "Section", le: np.ndarray, te: np.ndarray,
                  chord: float, n_x: int = 201):
    """
    Thickness and camber of one section, measured in its own chord-aligned
    frame: (t/c, x_tmax/c, camber/c signed, x_cmax/c, r_le/c).
    """
    if not np.isfinite(chord) or chord <= 1.0e-9:
        return (float("nan"),) * 5

    upper, lower = _split_surfaces(section.outline, le, te)
    up = _chord_frame(upper, le, te)
    lo = _chord_frame(lower, le, te)

    xs = (1.0 - np.cos(np.linspace(0.0, np.pi, n_x))) / 2.0
    yu, yl = _sample_surface(xs, up), _sample_surface(xs, lo)

    thick = yu - yl
    x_t, t_max = _peak(xs, thick)
    x_c, c_max = _peak(xs, 0.5 * (yu + yl))

    # On a symmetric section the camber line is zero to within the section
    # algorithm's own noise, and the position of its "maximum" is then just
    # where that noise happens to peak. Report the camber, drop the position.
    if abs(c_max) < CAMBER_FLOOR:
        x_c = float("nan")

    return t_max, x_t, c_max, x_c, _le_radius(xs, thick)


@dataclass
class Distributions:
    """
    Root-to-tip parameter distributions, one entry per spanwise station.

    Angles are in degrees, lengths in millimetres, shape parameters normalised
    by the local chord. Sweep and dihedral come in two flavours:

      cumulative - the angle the user asked for: between the spanwise vertical
                   through the root reference point and the straight line from
                   that point to the same reference point at this station. It
                   is the integrated quantity, the one that is a single fixed
                   number for a straight-edged rudder.
      local      - d(offset)/d(span) at this station, the tangent to the edge.

    For a straight edge the two are identical; where they part company the edge
    is curved, and the pair says how.
    """
    y: np.ndarray             # station height above the root [mm]
    eta: np.ndarray           # y / span
    chord: np.ndarray         # true LE -> TE length in the section plane [mm]
    chord_x: np.ndarray       # its chordwise projection [mm]
    x_le: np.ndarray
    z_le: np.ndarray
    x_te: np.ndarray
    z_te: np.ndarray
    sweep_le: np.ndarray          # cumulative, deg, positive aft
    sweep_c4: np.ndarray
    sweep_c2: np.ndarray
    sweep_te: np.ndarray
    sweep_le_local: np.ndarray    # local, deg, positive aft
    sweep_c4_local: np.ndarray
    rake: np.ndarray              # LE offset out of the planform plane [mm]
    dihedral: np.ndarray          # cumulative, deg, positive toward +Z
    dihedral_local: np.ndarray
    twist: np.ndarray             # deg, positive nose-up
    t_over_c: np.ndarray
    x_tmax: np.ndarray            # x/c of maximum thickness
    camber: np.ndarray            # max camber / c, signed
    x_cmax: np.ndarray            # x/c of maximum camber
    r_le: np.ndarray              # LE radius / c
    y_te: np.ndarray              # TE height; equal to y unless the section is tilted
    tilt_ratio: np.ndarray        # (y_te - y) / (x_te - x_le), rise over run
    tilt: np.ndarray              # deg, arctan(tilt_ratio), positive TE up
    horizontal: np.ndarray        # bool: rows the fits and the local columns use
    span: float                   # full geometric span of the part [mm]


# name, unit, tolerance key, description. Drives the .dat columns, the JSON and
# the fit report, so the three cannot drift apart.
_DIST_COLUMNS = (
    ("y",              "mm",  None,          "station height above the root"),
    ("eta",            "-",   None,          "y / span"),
    ("chord",          "mm",  "chord",       "true LE->TE length in the section plane"),
    ("chord_x",        "mm",  "chord",       "chordwise projection of the chord"),
    ("x_le",           "mm",  "chord",       "leading-edge x"),
    ("z_le",           "mm",  "rake",        "leading-edge z"),
    ("x_te",           "mm",  "chord",       "trailing-edge x"),
    ("z_te",           "mm",  "rake",        "trailing-edge z"),
    ("sweep_le",       "deg", "angle",       "cumulative LE sweep from the root LE, positive aft"),
    ("sweep_c4",       "deg", "angle",       "cumulative quarter-chord sweep, positive aft"),
    ("sweep_c2",       "deg", "angle",       "cumulative mid-chord sweep, positive aft"),
    ("sweep_te",       "deg", "angle",       "cumulative TE sweep, positive aft"),
    ("sweep_le_local", "deg", "angle",       "local LE sweep, d(x_le)/d(span)"),
    ("sweep_c4_local", "deg", "angle",       "local quarter-chord sweep"),
    ("rake",           "mm",  "rake",        "LE offset out of the planform plane (propeller rake)"),
    ("dihedral",       "deg", "angle",       "cumulative dihedral / rake angle, positive toward +Z"),
    ("dihedral_local", "deg", "angle",       "local dihedral angle"),
    ("twist",          "deg", "angle",       "section twist / pitch angle, positive nose-up"),
    ("t_over_c",       "-",   "shape",       "maximum thickness / chord"),
    ("x_tmax",         "-",   "position",    "x/c of maximum thickness"),
    ("camber",         "-",   "shape",       "maximum camber / chord, signed"),
    ("x_cmax",         "-",   "position",    "x/c of maximum camber, NaN on a symmetric section"),
    ("r_le",           "-",   "shape",       "leading-edge radius / chord"),
    # appended last so the column numbers above stay what they were
    ("y_te",           "mm",  None,          "trailing-edge height; equal to y except on the tilted tip section"),
    ("tilt_ratio",     "-",   None,          "(y_te - y) / (x_te - x_le), rise of the chord line over its run"),
    ("tilt",           "deg", None,          "section tilt about the thickness axis, arctan(tilt_ratio), positive TE up"),
)


def _slope(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    d(v)/d(y) along the span. `edge_order=2` matters: numpy's default leaves
    the first and last stations first-order accurate, and those two are exactly
    the root and tip values that get quoted.
    """
    return np.gradient(v, y, edge_order=2 if len(y) > 2 else 1)


def _cumulative_angle(dy: np.ndarray, offset: np.ndarray,
                      local: np.ndarray) -> np.ndarray:
    """
    Angle between the spanwise vertical through the root reference point and
    the line from it to the reference point at each station.

    At the root itself the secant is 0/0, so the tangent is used there: that is
    its limit as the station approaches the root, not a filled-in zero.
    """
    out = np.degrees(np.arctan2(offset, dy))
    at_root = dy <= 0.0
    out[at_root] = local[at_root]
    return out


def _flat_copy(s: Section) -> Section:
    """A tilted section laid flat (see `Section.untilt`), for measuring it."""
    le_f, te_f = s.untilt(np.vstack([s.le, s.te]))
    return Section(y=s.y, le=le_f, te=te_f, chord=s.chord,
                   outline=s.untilt(s.outline), airfoil_raw=s.airfoil_raw,
                   airfoil_norm=s.airfoil_norm)


def spanwise_distributions(stack, span: float) -> Distributions:
    """
    Step 6 - turn a section stack into root-to-tip parameter distributions.

    Every section gets a row. A tilted section (the tip section) is measured
    in its own plane, laid flat about its LE, so its chord, twist and shape
    are those of its own airfoil; its LE and TE are then put back where they
    sit on the part, and every reference point (LE, c/4, c/2, TE) is taken at
    its own height. The local (tangent) columns are derivatives up the stack
    of horizontal sections and are left NaN on a tilted row.
    """
    secs = sorted(stack, key=lambda s: (s.y, s.tilted))
    horizontal = np.array([not s.tilted for s in secs])
    if horizontal.sum() < 2:
        raise ValueError("need at least two horizontal sections to build a distribution")

    y = np.array([s.y for s in secs])
    # chordwise (X) projection of the chord; the tilt shortens it like the twist does
    chord_x = np.array([s.chord * np.cos(np.radians(s.tilt_deg)) if s.tilted else s.chord
                        for s in secs])

    le = np.empty((len(secs), 3))
    te = np.empty((len(secs), 3))
    chord = np.empty(len(secs))
    twist = np.empty(len(secs))
    shape = np.empty((len(secs), 5))
    for i, s in enumerate(secs):
        if s.tilted:
            flat = _flat_copy(s)
            le_f, te_f, chord[i], twist[i] = _section_chord_line(flat)
            shape[i] = section_shape(flat, le_f, te_f, chord[i])
            le[i], te[i] = s.retilt(np.vstack([le_f, te_f]))
        else:
            le[i], te[i], chord[i], twist[i] = _section_chord_line(s)
            shape[i] = section_shape(s, le[i], te[i], chord[i])

    # heights of the LE and TE: one station height on a horizontal section
    y_te = y.copy()
    y_te[~horizontal] = te[~horizontal, SPAN_AXIS]
    run_x = te[:, CHORD_AXIS] - le[:, CHORD_AXIS]
    tilt_ratio = (y_te - y) / run_x
    tilt = np.degrees(np.arctan2(y_te - y, run_x))

    def ref_line(frac):
        """x, z and height of the point `frac` of the way along each chord line."""
        return (le[:, CHORD_AXIS] + frac * (te[:, CHORD_AXIS] - le[:, CHORD_AXIS]),
                le[:, THICK_AXIS] + frac * (te[:, THICK_AXIS] - le[:, THICK_AXIS]),
                y + frac * (y_te - y))

    h = horizontal

    def local_angle(v, heights):
        out = np.full(len(secs), np.nan)
        out[h] = np.degrees(np.arctan(_slope(v[h], heights[h])))
        return out

    def sweep(frac):
        x_ref, _, h_ref = ref_line(frac)
        offset = x_ref - x_ref[0]
        local = local_angle(x_ref, h_ref)
        return _cumulative_angle(h_ref - h_ref[0], offset, local), local

    sweep_le, sweep_le_local = sweep(0.0)
    sweep_c4, sweep_c4_local = sweep(0.25)
    sweep_c2, _ = sweep(0.50)
    sweep_te, _ = sweep(1.00)

    dy = y - y[0]
    rake = le[:, THICK_AXIS] - le[0, THICK_AXIS]
    dihedral_local = local_angle(le[:, THICK_AXIS], y)
    dihedral = _cumulative_angle(dy, rake, dihedral_local)

    return Distributions(
        y=y, eta=y / span, chord=chord, chord_x=chord_x,
        x_le=le[:, CHORD_AXIS], z_le=le[:, THICK_AXIS],
        x_te=te[:, CHORD_AXIS], z_te=te[:, THICK_AXIS],
        sweep_le=sweep_le, sweep_c4=sweep_c4, sweep_c2=sweep_c2, sweep_te=sweep_te,
        sweep_le_local=sweep_le_local, sweep_c4_local=sweep_c4_local,
        rake=rake, dihedral=dihedral, dihedral_local=dihedral_local, twist=twist,
        t_over_c=shape[:, 0], x_tmax=shape[:, 1],
        camber=shape[:, 2], x_cmax=shape[:, 3], r_le=shape[:, 4],
        y_te=y_te, tilt_ratio=tilt_ratio, tilt=tilt, horizontal=horizontal,
        span=float(span),
    )


def planform_summary(d: Distributions) -> dict:
    """Integrated planform quantities over the horizontal sections."""
    trapz = getattr(np, "trapezoid", None) or np.trapz   # renamed in numpy 2
    h = d.horizontal
    y, c = d.y[h], d.chord[h]
    b = float(y[-1] - y[0])
    area = float(trapz(c, y))
    mac = float(trapz(c ** 2, y) / area) if area > 0 else float("nan")
    return {
        "span_geometric_mm": d.span,
        "span_extracted_mm": b,
        "chord_root_mm": float(c[0]),
        "chord_tip_mm": float(c[-1]),
        "taper_ratio": float(c[-1] / c[0]) if c[0] else float("nan"),
        "area_mm2": area,
        "mean_aerodynamic_chord_mm": mac,
        "aspect_ratio": float(b * b / area) if area > 0 else float("nan"),
        "note": "integrals cover the horizontal sections only, root up to the top "
                "horizontal station; the tilted tip section and anything above it "
                "are not included",
    }


def tip_section_summary(d: Distributions):
    """The tilted tip section, if the stack has one, as its own record."""
    rows = np.flatnonzero(~d.horizontal)
    if len(rows) == 0:
        return None
    i = int(rows[-1])
    return {
        "le_mm": [float(d.x_le[i]), float(d.y[i]), float(d.z_le[i])],
        "te_mm": [float(d.x_te[i]), float(d.y_te[i]), float(d.z_te[i])],
        "tilt_deg": float(d.tilt[i]),
        "tilt_ratio": float(d.tilt_ratio[i]),
        "chord_mm": float(d.chord[i]),
        "chord_x_mm": float(d.chord_x[i]),
        "twist_deg": float(d.twist[i]),
        "t_over_c": float(d.t_over_c[i]),
        "x_tmax": float(d.x_tmax[i]),
        "camber": float(d.camber[i]),
        "x_cmax": float(d.x_cmax[i]),
        "r_le": float(d.r_le[i]),
        "definition": "planar cut through the top of the LE and the top of the TE, "
                      "tilted about the thickness axis; shape parameters are those of "
                      "its own airfoil, measured untilted",
    }


def fit_distribution(eta: np.ndarray, values: np.ndarray, tol: float,
                     deg: int = DEFAULT_FIT_DEG) -> dict:
    """
    Describe one distribution: its spread, a straight-line fit and a polynomial
    fit in eta = y / span, plus the simplest form that holds within `tol`.

    `tol` is a reporting threshold. Nothing is snapped to the fit; the raw
    per-station values are always what gets written out.
    """
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return {"form": "undetermined", "n_valid": int(ok.sum())}

    e, v = np.asarray(eta, dtype=float)[ok], v[ok]
    mean = float(v.mean())
    dev_const = float(np.abs(v - mean).max())

    slope, intercept = np.polyfit(e, v, 1)
    res_lin = v - (slope * e + intercept)

    n_deg = int(min(deg, len(v) - 1))
    coeffs = np.polyfit(e, v, n_deg)
    res_poly = v - np.polyval(coeffs, e)

    if dev_const <= tol:
        form = "constant"
    elif np.abs(res_lin).max() <= tol:
        form = "linear"
    else:
        form = f"degree-{n_deg} polynomial"

    return {
        "form": form,
        "tolerance": float(tol),
        "n_valid": int(ok.sum()),
        "mean": mean,
        "std": float(v.std(ddof=0)),
        "min": float(v.min()),
        "max": float(v.max()),
        "root": float(v[0]),
        "tip": float(v[-1]),
        "max_dev_from_mean": dev_const,
        "linear": {
            "slope_per_eta": float(slope),
            "intercept": float(intercept),
            "max_abs_residual": float(np.abs(res_lin).max()),
            "rms_residual": float(np.sqrt(np.mean(res_lin ** 2))),
        },
        "polynomial": {
            "degree": n_deg,
            "variable": "eta = y / span",
            "coeffs_highest_power_first": [float(c) for c in coeffs],
            "max_abs_residual": float(np.abs(res_poly).max()),
            "rms_residual": float(np.sqrt(np.mean(res_poly ** 2))),
        },
    }


def fit_distributions(d: Distributions, deg: int = DEFAULT_FIT_DEG) -> dict:
    """
    Fit every distribution that carries a tolerance key, over the horizontal
    sections only: the tilted tip section is a different kind of cut, and one
    row of it would bend every fit at the top.
    """
    h = d.horizontal
    tols = {
        "angle": TOL_ANGLE_DEG,
        "chord": TOL_LENGTH_REL * float(d.chord[0]),
        "rake": TOL_LENGTH_REL * d.span,
        "shape": TOL_SHAPE,
        "position": TOL_POSITION,
    }
    return {name: fit_distribution(d.eta[h], getattr(d, name)[h], tols[key], deg)
            for name, _unit, key, _desc in _DIST_COLUMNS if key is not None}


def design_variables(d: Distributions) -> dict:
    """
    Quantities earmarked as design variables for the optimisation stage.
    For now: the tilt of the tip section.
    """
    tip = tip_section_summary(d)
    if tip is None:
        return {}
    return {
        "tip_tilt_deg": {
            "value": tip["tilt_deg"],
            "tan": tip["tilt_ratio"],
            "pivot_mm": tip["le_mm"],
            "axis": "thickness (+Z in the working frame, -Y in the XCAD file)",
            "sign": "positive raises the TE end of the tip section",
            "note": "tilt of the tip section, the last section of the stack; "
                    "a design variable in the optimisation stage",
        },
    }


def parametrise(stack, span: float, deg: int = DEFAULT_FIT_DEG) -> dict:
    """Step 6 end to end: distributions, their fits and the planform summary."""
    d = spanwise_distributions(stack, span)
    return {"distributions": d, "fits": fit_distributions(d, deg),
            "summary": planform_summary(d),
            "tip_section": tip_section_summary(d),
            "design_variables": design_variables(d)}


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _snap(v: float, places: int = 9) -> float:
    """Keep negative zero out of the files: anything that would print as zero
    at the written precision is written as +0."""
    return 0.0 if abs(v) < 0.5 * 10.0 ** (-places) else float(v)


def write_selig(path: str, coords: np.ndarray, name: str) -> None:
    with open(path, "w") as fh:
        fh.write(f"{name}\n")
        for x, y in coords:
            fh.write(f"{_snap(x, 8):12.8f}  {_snap(y, 8):12.8f}\n")


def write_raw_dat(path: str, coords: np.ndarray, y: float, notes=()) -> None:
    """Whitespace-delimited .dat, `#` comment header, loadable with np.loadtxt."""
    with open(path, "w") as fh:
        fh.write("# airfoil frame, millimetres, Selig order (TE -> upper -> LE -> lower -> TE)\n")
        fh.write(f"# section at y = {y:.6f} mm; x = 0 at this section's LE and grows aft,"
                 " y = thickness (+Z side up)\n")
        for note in notes:
            fh.write(f"# {note}\n")
        fh.write(f"# {'x_mm':>18s} {'y_mm':>18s}\n")
        for x, yy in coords:
            fh.write(f"{_snap(x):20.9f} {_snap(yy):18.9f}\n")


def write_section_stack(path: str, sections, source: str, y_top: float,
                        how: str = "", y_le_top: float | None = None,
                        layout: dict | None = None) -> None:
    """
    Every cross-section in one .dat, ordered TOP to BOTTOM.

    All metadata sits on `#` lines, so the numbers load in one call:
        np.loadtxt(path).reshape(n_sections, n_points, 2)
    with section 0 the topmost. A tilted tip section, when there is one, is
    section 0, written untilted; its header says how to put it back.
    """
    ordered = sorted(sections, key=lambda s: (-s.y, not s.tilted))
    n_pts = len(ordered[0].airfoil_raw)
    flat = [s for s in ordered if not s.tilted]
    tips = [s for s in ordered if s.tilted]

    with open(path, "w") as fh:
        fh.write("# Rudder cross-sectional airfoil coordinates\n")
        fh.write(f"# source        : {source}\n")
        fh.write("# frame         : origin at the root leading edge;"
                 " +X LE->TE, +Y root->tip, +Z thickness\n")
        fh.write(f"# n_sections    : {len(ordered)}\n")
        fh.write(f"# n_points      : {n_pts} per section\n")
        if tips:
            t = tips[0]
            fh.write("# section order : TOP to BOTTOM. Section 1 is the tilted tip section,"
                     f" LE at y = {t.y:.6f},\n")
            fh.write(f"#                 TE at y = {t.te[SPAN_AXIS]:.6f} mm"
                     f" (tilt {t.tilt_deg:.6f} deg); then {len(flat)} horizontal"
                     " sections,\n")
            fh.write(f"#                 y = {flat[0].y:.6f} down to y = {flat[-1].y:.6f} mm,"
                     " evenly spaced\n")
        else:
            fh.write(f"# section order : TOP to BOTTOM, y = {ordered[0].y:.6f}"
                     f" down to y = {ordered[-1].y:.6f} mm, evenly spaced\n")
        if layout and layout.get("tip"):
            fh.write(f"# layout        : horizontal step dz = {layout['dz']:.9f} mm; the top"
                     f" horizontal section sits\n")
            fh.write(f"#                 d = {layout['d']:.9f} mm below the top of the LE"
                     + (" (d = dz / 2)\n" if layout.get("d_default") else " (given)\n"))
        fh.write("# point order   : Selig, TE -> upper -> LE -> lower -> TE\n")
        fh.write("# columns       : x_mm y_mm in that section's own airfoil frame,"
                 " x = 0 at its LE\n")
        fh.write("#                 and growing aft, y = thickness with the +Z side"
                 " positive\n")
        fh.write("# 3D recovery   : X = x_le + x_mm,  Y = y_section,  Z = y_mm\n")
        if tips:
            fh.write("#                 for the tilted section that gives it untilted;"
                     " rotate the points by\n")
            fh.write("#                 +tilt about the thickness (Z) axis through"
                     " (x_le, y_section) to put it back\n")
        fh.write("# load          : np.loadtxt(path).reshape"
                 f"({len(ordered)}, {n_pts}, 2)\n")
        if tips:
            fh.write(f"# top station   : the top of the leading edge, y = {y_le_top:.6f} mm"
                     + (f" ({how})\n" if how else "\n"))
            fh.write("#                 horizontal cuts stop below it; above it the"
                     " inclined tip surface takes\n")
            fh.write("#                 over the nose. The tilted section closes the"
                     " stack from the top of the\n")
            fh.write("#                 LE to the top of the TE; the tip cap above it"
                     " is not covered here\n")
        elif y_le_top is not None and abs(y_top - y_le_top) > 1.0e-6:
            fh.write(f"# top station   : y = {y_top:.6f} mm, requested\n")
            fh.write(f"#                 the leading edge itself runs up to"
                     f" y = {y_le_top:.6f} mm"
                     + (f" ({how});\n" if how else ";\n"))
            fh.write("#                 above that the inclined tip surface takes over"
                     " the nose and a\n")
            fh.write("#                 horizontal cut is no longer a complete airfoil."
                     " Nothing above the\n")
            fh.write("#                 top station is covered here either way\n")
        else:
            fh.write(f"# top station   : y = {y_top:.6f} mm, the top of the leading edge"
                     + (f" ({how})\n" if how else "\n"))
            fh.write("#                 the tip surface is inclined, so horizontal cuts"
                     " above this station\n")
            fh.write("#                 cut through the tip blend instead of a complete"
                     " airfoil and are\n")
            fh.write("#                 not included here\n")

        for k, s in enumerate(ordered, start=1):
            fh.write("#\n")
            fh.write(f"# SECTION {k:d} / {len(ordered):d}"
                     f"   y = {s.y:.6f}   chord = {s.chord:.6f}"
                     f"   x_le = {s.le[CHORD_AXIS]:.6f}"
                     f"   x_te = {s.te[CHORD_AXIS]:.6f}"
                     + (f"   y_te = {s.te[SPAN_AXIS]:.6f}   tilt = {s.tilt_deg:.6f} deg"
                        "   (tilted tip section, written untilted)" if s.tilted else "")
                     + "\n")
            fh.write(f"# {'x_mm':>18s} {'y_mm':>18s}\n")
            for x, yy in s.airfoil_raw:
                fh.write(f"{_snap(x):20.9f} {_snap(yy):18.9f}\n")


XCAD_UNITS = {"m": 1.0e-3, "mm": 1.0}   # scale from the working frame (mm)

# Working frame -> XCAD frame: height (span) becomes Z. A +90 deg rotation about
# X: x' = x, y' = -z, z' = y. Swapping the Y and Z columns instead would be a
# mirror (det -1) and would reverse the loops about the span axis.
XCAD_FRAME = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0]])


def xcad_loop(sec: Section) -> np.ndarray:
    """
    One section as a closed 3D loop, in the point order of ORCA101.dat:
    start at the leading edge, run along the +Z side to the trailing edge,
    come back along the -Z side, and repeat the first point to close.

    Same point count as the Selig section (`num_pts`, closure included), with
    the trailing edge at index (num_pts + 1) // 2 - 1. Working frame, mm. A
    tilted section comes out tilted, where it sits on the part.
    """
    raw = sec.airfoil_raw                        # Selig: TE -> +Z -> LE -> -Z -> TE
    n = len(raw)
    i_le = (n + 1) // 2 - 1                      # where extract_section put the LE
    if abs(raw[i_le, 0]) > 1.0e-9:
        raise ValueError(f"section y = {sec.y}: leading edge not at index {i_le}")
    upper = raw[:i_le + 1][::-1]                 # LE -> TE along +Z
    lower = raw[i_le:][::-1]                     # TE -> LE along -Z
    loop2 = np.vstack([upper, lower[1:]])        # LE ... TE ... LE, n points
    return sec.points_3d(loop2)


def write_xcad(path: str, sections, units: str = "m", extra_loops=None) -> None:
    """
    The section stack in the point format XCAD reads (the layout of the
    ORCA101.dat propeller file), with # and the section number on its own
    line before each section:

        #1
        x y z        <- leading edge
        ...          <- -y side to the trailing edge, then back along +y
        x y z        <- leading edge again, closing the loop
        #2
        ...

    Sections run from the root (1) up to the top station, as ORCA101 runs
    from hub to tip. Columns are X Y Z in the XCAD frame, with the height as
    Z: origin at the root LE, +X chordwise LE->TE, +Z spanwise root->tip,
    and Y thickness, equal to -Z of the working frame (`XCAD_FRAME`, a
    rotation, so the frame stays right-handed and the loops turn the same
    way about the span axis as in ORCA101). Metres as in ORCA101
    (units="m") or millimetres (units="mm"). Eight decimals, single spaces,
    no header, Windows (CRLF) line endings.

    `extra_loops` (working frame, mm, each already a closed loop in the same
    point order) are written after the sections with the numbering carried on:
    that is how the tip cap's loops go on top of the stack.
    """
    if units not in XCAD_UNITS:
        raise ValueError(f"units must be one of {sorted(XCAD_UNITS)}")
    scale = XCAD_UNITS[units]
    ordered = sorted(sections, key=lambda s: s.y)
    loops = [xcad_loop(sec) for sec in ordered] + list(extra_loops or [])

    with open(path, "w", newline="\r\n") as fh:
        for k, loop in enumerate(loops, start=1):
            fh.write(f"#{k}\n")
            for x, y, z in (np.asarray(loop) @ XCAD_FRAME.T) * scale:
                fh.write(f"{_snap(x, 8):.8f} {_snap(y, 8):.8f} {_snap(z, 8):.8f}\n")


def write_corner_points(path_dat: str, path_json: str, sections, rotated) -> None:
    """
    LE and TE corner points, one spanwise station per ROW:

        x1 y1 z1 x2 y2 z2      point 1 = LE, point 2 = TE

    Row 1 is the root, the last row is the top of the leading edge, so the
    first three columns trace the LE up the span and the last three trace the
    TE. Ordered root first, the opposite of the section stack, which the user
    asked to have running top to bottom. When the stack ends in the tilted tip
    section, that is the last row, and its TE sits higher than its LE.
    """
    ordered = sorted(sections, key=lambda s: (s.y, s.tilted))
    tilted = any(s.tilted for s in ordered)

    with open(path_dat, "w") as fh:
        fh.write("# LE and TE corner points, one spanwise station per row\n")
        fh.write("# frame   : origin at the root leading edge;"
                 " +X LE->TE, +Y root->tip, +Z thickness\n")
        fh.write("# point 1 : leading edge      point 2 : trailing edge\n")
        fh.write(f"# rows    : {len(ordered)}, root first,"
                 f" y = {ordered[0].y:.6f} up to y = {ordered[-1].y:.6f} mm\n")
        if tilted:
            fh.write("#           y1 and y2 are the same station height on every"
                     " horizontal row; chord = x2 - x1\n")
            fh.write("#           the last row is the tilted tip section, top of the LE"
                     " to top of the TE:\n")
            fh.write("#           tilt = arctan((y2 - y1) / (x2 - x1)),"
                     " chord = sqrt((x2 - x1)^2 + (y2 - y1)^2)\n")
        else:
            fh.write("#           y1 and y2 are the same station height;"
                     " chord = x2 - x1\n")
        fh.write("# load    : np.loadtxt(path)  ->"
                 f" ({len(ordered)}, 6), LE = [:, :3], TE = [:, 3:]\n")
        fh.write("# {:>16s} {:>16s} {:>16s} {:>18s} {:>16s} {:>16s}\n".format(
            "x1", "y1", "z1", "x2", "y2", "z2"))
        for s in ordered:
            fh.write(f"{_snap(s.le[0]):18.9f} {_snap(s.le[1]):16.9f} {_snap(s.le[2]):16.9f}"
                     f" {_snap(s.te[0]):18.9f} {_snap(s.te[1]):16.9f} {_snap(s.te[2]):16.9f}\n")

    with open(path_json, "w") as fh:
        json.dump(
            {
                "frame": {
                    "origin": "root leading edge",
                    "x": "chordwise, LE -> TE",
                    "y": "spanwise, root -> tip",
                    "z": "thickness",
                    "cad_rotated_180deg_about_span": bool(rotated),
                },
                "columns": ["x1", "y1", "z1", "x2", "y2", "z2"],
                "point_1": "leading edge",
                "point_2": "trailing edge",
                "row_order": "root first",
                "stations": [
                    {"y": s.y, "chord": s.chord,
                     "le": s.le.tolist(), "te": s.te.tolist(),
                     **({"tilt_deg": s.tilt_deg,
                         "note": "tilted tip section; chord in its own plane"}
                        if s.tilted else {})}
                    for s in ordered
                ],
            },
            fh,
            indent=2,
        )


def write_parameters(path_dat: str, path_json: str, param: dict,
                     source: str) -> None:
    """
    The spanwise parametrisation: one row per station, one column per
    parameter, root first. The JSON alongside carries the same table plus the
    constant / linear / polynomial fit of every column.
    """
    d = param["distributions"]
    fits, summary = param["fits"], param["summary"]
    tip = param.get("tip_section")
    names = [c[0] for c in _DIST_COLUMNS]
    table = np.column_stack([getattr(d, n) for n in names])

    with open(path_dat, "w") as fh:
        fh.write("# Rudder spanwise parametrisation, one station per row, root first\n")
        fh.write(f"# source   : {source}\n")
        fh.write("# frame    : origin at the root leading edge;"
                 " +X LE->TE, +Y root->tip, +Z thickness\n")
        fh.write("# units    : lengths mm, angles deg, shape parameters / chord\n")
        fh.write("# sweep and dihedral: 'cumulative' is measured from the spanwise"
                 " vertical through the\n")
        fh.write("#            root reference point to the line joining it to that"
                 " station's reference\n")
        fh.write("#            point; 'local' is the tangent, d(offset)/d(span)."
                 " Equal for a straight edge.\n")
        fh.write("# twist    : angle of the section chord line, positive nose-up"
                 " (LE on the +Z side)\n")
        fh.write("# tilt     : slope of the chord line in the span direction, from the"
                 " LE and TE heights:\n")
        fh.write("#            tilt_ratio = (y_te - y) / (x_te - x_le), tilt ="
                 " arctan(tilt_ratio), positive TE up.\n")
        fh.write("#            In the XCAD file (height along z) these are z1 and z2"
                 " of the LE and TE.\n")
        fh.write(f"# rows     : {len(d.y)}, y = {d.y[0]:.6f} up to"
                 f" {d.y[-1]:.6f} mm\n")
        if tip is not None:
            fh.write("# tip      : the last row is the tilted tip section, top of the LE"
                     " to top of the TE,\n")
            fh.write(f"#            tilt {tip['tilt_deg']:.6f} deg (ratio"
                     f" {tip['tilt_ratio']:.9f}); every other row is horizontal,"
                     " tilt 0.\n")
            fh.write("#            Its chord, twist and shape are those of its own"
                     " airfoil, measured untilted;\n")
            fh.write("#            its reference points sit at their own heights. The"
                     " fits and the planform\n")
            fh.write("#            summary use the horizontal rows only, and the local"
                     " (tangent) columns are NaN\n")
            fh.write("#            on the tilted row. Its tilt is a design variable for"
                     " the optimisation stage.\n")
        capd = param.get("tip_cap")
        if capd:
            cp = capd["params"]
            fh.write("# cap      : the tip cap above the tilted section (Step 7, tip_cap.py):"
                     f" face {cp['h_le']:.4f} -> {cp['h_te']:.4f} mm,\n")
            fh.write(f"#            crown radius {cp['crown_radius'] or float('inf'):.3f} mm,"
                     f" edge radius {cp['edge_radius']:.4f} mm, TE corner radius"
                     f" {cp['te_radius']:.4f} mm;\n")
            fh.write(f"#            fit to {capd['fit']['max_dev_mm']:.4f} mm max,"
                     f" {capd['fit']['rms_dev_mm']:.4f} mm rms. Details in parameters.json"
                     " and cap_profile.dat.\n")
        fh.write(f"# load     : np.loadtxt(path)  ->"
                 f" ({len(d.y)}, {len(names)})\n")
        fh.write("#\n# columns\n")
        for i, (name, unit, _key, desc) in enumerate(_DIST_COLUMNS, start=1):
            fh.write(f"#   {i:2d}  {name:<16s} [{unit:>3s}]  {desc}\n")
        fh.write("#\n# fitted form of each distribution\n")
        for name in names:
            if name not in fits:
                continue
            f = fits[name]
            if "mean" not in f:                      # nothing finite to fit
                fh.write(f"#   {name:<16s} {f['form']}\n")
                continue
            fh.write(f"#   {name:<16s} {f['form']:<22s}"
                     f" mean {f['mean']:+.6f}  spread {f['max_dev_from_mean']:.3e}\n")
        fh.write("#\n")
        header = " ".join(f"{n:>20s}" for n in names)
        fh.write("#" + header[1:] + "\n")        # '#' takes one space, stays aligned
        for row in table:
            fh.write(" ".join(f"{_snap(v, 9):20.9f}" for v in row) + "\n")

    with open(path_json, "w") as fh:
        json.dump(
            {
                "source": source,
                "frame": {
                    "origin": "root leading edge",
                    "x": "chordwise, LE -> TE",
                    "y": "spanwise, root -> tip",
                    "z": "thickness",
                },
                "conventions": {
                    "sweep_cumulative": "angle between the spanwise vertical through "
                                        "the root reference point and the line from it "
                                        "to the same reference point at that station, "
                                        "positive aft",
                    "sweep_local": "arctan of d(x_ref)/d(span), positive aft",
                    "rake": "leading-edge offset out of the planform plane, +Z",
                    "dihedral": "same construction as the sweep but on the rake, "
                                "positive toward +Z; the wing name for propeller rake angle",
                    "twist": "angle of the section chord line, positive nose-up",
                    "tilt": "slope of the chord line in the span direction from the LE and "
                            "TE heights: tilt_ratio = (y_te - y) / (x_te - x_le), tilt = "
                            "arctan(tilt_ratio), positive TE up; 0 on a horizontal section",
                    "shape_parameters": "normalised by the local chord, measured in the "
                                        "section's own chord-aligned frame so twist does "
                                        "not leak into thickness or camber",
                    "tolerances": "decide only how a distribution is labelled; the "
                                  "tabulated values are the raw measurements",
                    "fit_rows": "fits, the planform summary and the local (tangent) "
                                "columns use the horizontal sections only",
                },
                "columns": [
                    {"name": n, "unit": u, "description": desc}
                    for n, u, _k, desc in _DIST_COLUMNS
                ],
                "summary": summary,
                **({"stack_layout": param["layout"]} if param.get("layout") else {}),
                **({"tip_section": tip} if tip is not None else {}),
                **({"tip_cap": param["tip_cap"]} if param.get("tip_cap") else {}),
                **({"design_variables": param["design_variables"]}
                   if param.get("design_variables") else {}),
                "fits": fits,
                "stations": [
                    {n: float(v) for n, v in zip(names, row)} for row in table
                ],
            },
            fh,
            indent=2,
        )


def write_tip_section(out_dir: str, sec: Section, source: str) -> dict:
    """
    The tilted tip section on its own, untilted so its airfoil can be read:

      section_tip_untilted_selig.dat   normalised by its chord, Selig order
      section_tip_untilted_raw_mm.dat  the same in mm
      section_tip_3d_mm.dat            3D, as it sits on the part (x y z) and
                                       unrotated (x_u y_u z_u), row for row

    The untilted points are the tilted ones turned back by -tilt about the
    thickness axis through the section's LE (`rot_axis`); they lie at the
    height of the LE. Returns the file paths.
    """
    base = os.path.join(out_dir, "section_tip")
    paths = {"selig": base + "_untilted_selig.dat",
             "raw_mm": base + "_untilted_raw_mm.dat",
             "xyz_mm": base + "_3d_mm.dat"}
    rot_note = (f"tilted tip section shown untilted: turned by {-sec.tilt_deg:.6f} deg about"
                f" the thickness axis through its LE ({sec.le[CHORD_AXIS]:.6f},"
                f" {sec.y:.6f}, 0)")
    write_selig(paths["selig"], sec.airfoil_norm,
                f"{source} tip section untilted (tilt {sec.tilt_deg:.4f} deg)"
                f" c={sec.chord:.4f}mm")
    write_raw_dat(paths["raw_mm"], sec.airfoil_raw, sec.y,
                  notes=(rot_note,
                         f"its chord in its own plane is {sec.chord:.6f} mm; on the part"
                         f" its TE is at y = {sec.te[SPAN_AXIS]:.6f} mm"))

    tilted = sec.points_3d(sec.airfoil_raw)
    untilted = sec.untilt(tilted)
    with open(paths["xyz_mm"], "w") as fh:
        fh.write("# Tilted tip section, 3D, millimetres\n")
        fh.write(f"# source   : {source}\n")
        fh.write("# frame    : origin at the root leading edge;"
                 " +X LE->TE, +Y root->tip, +Z thickness\n")
        fh.write(f"# section  : planar cut through the top of the LE"
                 f" ({sec.le[0]:.6f}, {sec.le[1]:.6f}, {sec.le[2]:.6f})\n")
        fh.write(f"#            and the top of the TE"
                 f" ({sec.te[0]:.6f}, {sec.te[1]:.6f}, {sec.te[2]:.6f}),\n")
        fh.write(f"#            tilted {sec.tilt_deg:.9f} deg about the thickness (Z)"
                 " axis, TE up\n")
        fh.write("# columns  : x y z        where the section sits on the part\n")
        fh.write(f"#            x_u y_u z_u  the same points turned by {-sec.tilt_deg:.6f}"
                 " deg about the Z axis\n")
        fh.write(f"#                         through the LE: a horizontal section at"
                 f" y = {sec.y:.6f}\n")
        fh.write("# points   : Selig order, TE -> +Z side -> LE -> -Z side -> TE, the same"
                 " rows as\n")
        fh.write("#            section_tip_untilted_raw_mm.dat\n")
        fh.write(f"# load     : np.loadtxt(path)  -> ({len(tilted)}, 6)\n")
        fh.write("# {:>16s} {:>16s} {:>16s} {:>18s} {:>16s} {:>16s}\n".format(
            "x", "y", "z", "x_u", "y_u", "z_u"))
        for p, q in zip(tilted, untilted):
            fh.write(f"{_snap(p[0]):18.9f} {_snap(p[1]):16.9f} {_snap(p[2]):16.9f}"
                     f" {_snap(q[0]):18.9f} {_snap(q[1]):16.9f} {_snap(q[2]):16.9f}\n")
    return paths


# --------------------------------------------------------------------------- #
# Step 7 - the tip cap
# --------------------------------------------------------------------------- #
DEFAULT_CAP_STATIONS = 48     # cross-sections cut through the cap for the fit
DEFAULT_CAP_LOOPS = 10        # loops over the cap in the XCAD file
DEFAULT_CAP_TOP = 0.995       # height fraction of the last loop (1 = the crest line)


def cap_base(tip: Section) -> "tip_cap.CapBase":
    """The tilted tip section as the cap's base (see tip_cap.CapBase)."""
    th = np.radians(tip.tilt_deg)
    chord_dir = rot_axis(TILT_AXIS, th) @ np.eye(3)[CHORD_AXIS]
    normal = skew_symmetric_matrix(TILT_AXIS) @ chord_dir
    pivot = np.zeros(3)
    pivot[CHORD_AXIS] = tip.le[CHORD_AXIS]
    pivot[SPAN_AXIS] = tip.y
    return tip_cap.CapBase.from_selig(tip.airfoil_raw, pivot, chord_dir,
                                      normal / np.linalg.norm(normal), TILT_AXIS)


def measure_cap(lateral, base: "tip_cap.CapBase", n_stations: int = DEFAULT_CAP_STATIONS,
                dense: int = 3000):
    """
    Cut the part normal to the tip chord at `n_stations` stations, clustered
    towards both ends of the cap, and keep what lies above the tip section: the
    cap's cross-sections, in the cap frame (distance from the tip section's
    mid-line, height above it).
    """
    k = np.arange(1, n_stations + 1)
    us = 0.5 * (1.0 - np.cos(np.pi * k / (n_stations + 1)))
    stations = []
    for u in us:
        x = float(u * base.chord)
        curves = tilted_section_curves(lateral, base.to_3d(x, 0.0, 0.0), base.t)
        pts = np.vstack([_curve_polyline(c, dense) for c in curves])
        _x, z, h = base.to_cap(pts)
        keep = h >= 0.0
        stations.append(tip_cap.CapStation(x, np.abs(z[keep] - base.mid(x)), h[keep]))
    return stations


def cap_levels(n: int = DEFAULT_CAP_LOOPS, top: float = DEFAULT_CAP_TOP):
    """Height fractions of the cap loops, closer together towards the crest."""
    k = np.arange(1, n + 1) / n
    return top * (1.0 - (1.0 - k) ** 2)


def write_cap_profile(path: str, report, params, chord: float, source: str) -> None:
    """The cap fit, one measured cross-section per row."""
    cols = ("u", "x_mm", "w_mm", "top_cad_mm", "top_model_mm", "max_dev_mm", "rms_dev_mm", "core")
    with open(path, "w") as fh:
        fh.write("# Rudder tip cap: measured cross-sections and the fitted model\n")
        fh.write(f"# source : {source}\n")
        fh.write("# frame  : cap frame; x along the tip section's chord from the top of the LE\n")
        fh.write(f"#          (0) to the top of the TE ({chord:.6f} mm), u = x / chord; heights are\n")
        fh.write("#          above the tip section's plane, along its normal\n")
        fh.write("# w      : the tip section's half-thickness at x\n")
        fh.write("# top    : height of the top of the cap (the crest) above the tip section\n")
        fh.write("# dev    : distance from the CAD cross-section's points to the model's\n")
        fh.write("# core   : 1 if the station was used to fit the face, crown and edge radius;\n")
        fh.write("#          the TE-end stations fit the TE corner radius\n")
        fh.write("# model  : " + ", ".join(f"{k} = {v if v is not None else 'flat'}"
                                          for k, v in params.as_dict().items()) + "\n")
        fh.write(f"# load   : np.loadtxt(path)  -> ({len(report)}, {len(cols)})\n")
        fh.write("#" + " ".join(f"{c:>15s}" for c in cols)[1:] + "\n")
        for r in report:
            fh.write(" ".join(f"{_snap(v, 9):15.9f}" for v in
                              (r["u"], r["x"], r["w"], r["top_cad"], r["top_model"],
                               r["max_dev"], r["rms_dev"], 1.0 if r["core"] else 0.0)) + "\n")


def write_cap_grid(path: str, g: np.ndarray, source: str) -> None:
    """The fitted cap as a structured point grid, working frame, mm."""
    nx, nj, _ = g.shape
    with open(path, "w") as fh:
        fh.write("# Rudder tip cap, fitted model, as a structured point grid\n")
        fh.write(f"# source : {source}\n")
        fh.write("# frame  : origin at the root leading edge; +X LE->TE, +Y root->tip, +Z thickness\n")
        fh.write(f"# grid   : {nx} rows along the tip chord (LE end to TE end, cosine spaced) x {nj}\n")
        fh.write("#          points across each row, from the +Z wall base over the crest to the\n")
        fh.write("#          -Z wall base. The wall-base points lie on the tip section; the first\n")
        fh.write("#          and last rows collapse to the top of the LE and the top of the TE\n")
        fh.write(f"# load   : np.loadtxt(path).reshape({nx}, {nj}, 3)\n")
        fh.write(f"# {'x_mm':>16s} {'y_mm':>16s} {'z_mm':>16s}\n")
        for x, y, z in g.reshape(-1, 3):
            fh.write(f"{_snap(x):18.9f} {_snap(y):16.9f} {_snap(z):16.9f}\n")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(cad_path: str, cut_height: float = 80.0, num_pts: int = DEFAULT_NUM_PTS,
        out_dir: str | None = None, spacing: str = "cosine", plot: bool = True,
        num_sec: int | None = DEFAULT_NUM_SEC, y_top: float | None = None,
        progress=None, fit_deg: int = DEFAULT_FIT_DEG, xcad_units: str = "m",
        tip: bool = True, tip_d: float | None = None, cap: bool = True,
        cap_loops: int = DEFAULT_CAP_LOOPS, cap_top: float = DEFAULT_CAP_TOP,
        cap_stations: int = DEFAULT_CAP_STATIONS):
    """
    Steps 1-7 end to end.

    When the stack runs to the top of the LE (no `y_top`, or `y_top` equal to
    it) and `tip` is on, it is laid out for the tilted tip section: `num_sec`
    horizontal sections from the root to d below the top of the LE (d = half
    the step unless `tip_d` gives it in mm), then the tilted section through
    the top of the LE and the top of the TE. With `tip` off, or a `y_top`
    below the top of the LE, the stack is `num_sec` horizontal sections from
    the root to the top station, as before.

    With a tip section and `cap` on, Step 7 fits the tip cap (tip_cap.py) and
    writes it: the parameters into parameters.json, the fit per cross-section
    to cap_profile.dat, the model as a point grid, and the stack with
    `cap_loops` loops over the cap (the last at height fraction `cap_top`) as a
    second XCAD file.
    """
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(cad_path)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    shape = load_cad(cad_path)                                    # Step 1
    shape, trsf, le_orig, rotated = reframe_to_root_le(shape)     # Step 2
    lateral, _ = lateral_faces(shape)

    lo, hi = bounding_box(shape)
    span = hi[SPAN_AXIS] - lo[SPAN_AXIS]
    if not (0.0 <= cut_height <= span):
        raise ValueError(f"cut height {cut_height} outside span 0 .. {span:.4f}")

    root = extract_section(lateral, 0.0, num_pts, spacing)          # Steps 3-5
    cut = extract_section(lateral, cut_height, num_pts, spacing)

    for sec, tag in ((root, "root_y0"), (cut, f"cut_y{cut_height:g}")):
        base = os.path.join(out_dir, f"section_{tag}")
        write_selig(base + "_selig.dat", sec.airfoil_norm,
                    f"{os.path.basename(cad_path)} y={sec.y:g}mm c={sec.chord:.4f}mm")
        write_raw_dat(base + "_raw_mm.dat", sec.airfoil_raw, sec.y)

    # always locate the top of the LE, even when the stack is asked to stop
    # lower, so the report and the file headers can say where it actually is
    le_top, how = leading_edge_top_point(shape, lateral, span)
    y_le_top = float(le_top[SPAN_AXIS])
    if y_top is None:
        y_top = y_le_top
    elif y_top > y_le_top + 1.0e-6:
        print(f"  warning: requested top y = {y_top:.6f} is above the top of the "
              f"leading edge at y = {y_le_top:.6f}; sections up there cut through "
              f"the tip blend and may fail")

    use_tip = bool(num_sec) and tip and abs(y_top - y_le_top) <= 1.0e-6
    layout = {"tip": use_tip, "n_horizontal": int(num_sec or 0)}
    stack = None
    tip_sec = None
    xcad_path = None
    tip_paths = None
    te_top, te_how = None, ""
    if num_sec:
        if use_tip:
            te_top, te_how = trailing_edge_top_point(shape, lateral, span)
            dz, d = tip_spacing(y_le_top, num_sec, tip_d)
            y_flat_top = y_le_top - d
            layout.update(dz=dz, d=d, d_default=tip_d is None,
                          y_top_horizontal=y_flat_top, le_top=le_top.tolist(),
                          te_top=te_top.tolist(), te_top_how=te_how)
            stack = extract_stack(lateral, y_flat_top, num_sec, num_pts, spacing,
                                  progress=progress)
            tip_sec = extract_tilted_section(lateral, le_top, te_top, num_pts, spacing)
            stack.append(tip_sec)
            layout["tilt_deg"] = tip_sec.tilt_deg
        else:
            stack = extract_stack(lateral, y_top, num_sec, num_pts, spacing,
                                  progress=progress)
            layout.update(dz=(y_top / (num_sec - 1)) if num_sec > 1 else 0.0,
                          y_top_horizontal=y_top)
        write_section_stack(os.path.join(out_dir, "sections_stack_raw_mm.dat"),
                            stack, os.path.basename(cad_path), y_top, how, y_le_top,
                            layout)
        # the same stack for XCAD: 3D closed loops, root first, numbered sections
        xcad_path = os.path.join(out_dir, f"sections_xcad_{xcad_units}.dat")
        write_xcad(xcad_path, stack, xcad_units)
        if tip_sec is not None:
            tip_paths = write_tip_section(out_dir, tip_sec, os.path.basename(cad_path))

    # Step 7 - the tip cap, on the tilted tip section
    cap_res = None
    if tip_sec is not None and cap:
        source = os.path.basename(cad_path)
        base = cap_base(tip_sec)
        measured = measure_cap(lateral, base, cap_stations)
        cap_params, cap_report = tip_cap.fit(measured, base)
        levels = cap_levels(cap_loops, cap_top)
        loops = tip_cap.loops(cap_params, base, levels)
        cap_paths = {
            "profile": os.path.join(out_dir, "cap_profile.dat"),
            "grid": os.path.join(out_dir, "tip_cap_grid_mm.dat"),
            "xcad": os.path.join(out_dir, f"sections_xcad_with_cap_{xcad_units}.dat"),
        }
        write_cap_profile(cap_paths["profile"], cap_report, cap_params, base.chord, source)
        write_cap_grid(cap_paths["grid"], tip_cap.grid(cap_params, base), source)
        write_xcad(cap_paths["xcad"], stack, xcad_units, extra_loops=loops)
        cap_res = {"params": cap_params, "report": cap_report, "base": base,
                   "measured": measured, "levels": levels, "paths": cap_paths,
                   "n_sections": len(stack)}

    # one row per station: the LE/TE edge table, root first up to the top of
    # the LE. With no stack it falls back to the two named cuts.
    stations = stack if stack else [root, cut]
    write_corner_points(os.path.join(out_dir, "corner_points.dat"),
                        os.path.join(out_dir, "corner_points.json"),
                        stations, rotated)

    # Step 6 - the parametrisation, off the same stations, no further cuts
    param = parametrise(stations, span, fit_deg)
    if use_tip:
        param["layout"] = layout
        dv = param["design_variables"].get("tip_tilt_deg")
        if dv is not None:              # the pivot is the corner itself, not a sample
            dv["pivot_mm"] = le_top.tolist()
            dv["pivot"] = "top of the LE, the end of the b-rep leading-edge curve"
    if cap_res is not None:
        param["tip_cap"] = cap_summary(cap_res, out_dir)
        param["design_variables"].update(cap_design_variables(cap_res))
    write_parameters(os.path.join(out_dir, "parameters.dat"),
                     os.path.join(out_dir, "parameters.json"),
                     param, os.path.basename(cad_path))

    if plot:
        _plot(out_dir, root, cut, span, lateral, stack, y_top, y_le_top, tip_sec)
        _plot_parameters(out_dir, param)
        if cap_res is not None:
            _plot_cap(out_dir, cap_res)

    return {"shape": shape, "lateral": lateral, "trsf": trsf,
            "le_original": le_orig, "rotated": rotated,
            "root": root, "cut": cut,
            "span": span, "out_dir": out_dir, "y_top": y_top,
            "y_le_top": y_le_top, "y_top_how": how, "stack": stack,
            "le_top": le_top, "te_top": te_top, "te_top_how": te_how,
            "tip_section": tip_sec, "tip_paths": tip_paths, "layout": layout,
            "cap": cap_res,
            "param": param, "xcad_path": xcad_path, "xcad_units": xcad_units}


def cap_summary(cap_res: dict, out_dir: str) -> dict:
    """The tip cap block of parameters.json."""
    p, rep, base = cap_res["params"], cap_res["report"], cap_res["base"]
    core = [r for r in rep if r["core"]]
    rms = lambda rows: float(np.sqrt(np.mean([r["rms_dev"] ** 2 for r in rows])))  # noqa: E731
    return {
        "definition": "the part above the tilted tip section: an inclined face over the tip "
                      "section, face(u) = h_le (1-u) + h_te u + 3u(1-u) [b1 (1-u) + b2 u], "
                      "crowned across the thickness with radius crown_radius (top of the cap "
                      "H = face + w^2 / (2 crown_radius), w = the tip section's half-thickness), "
                      "a constant edge_radius joining the side walls to the crown and rounding "
                      "the LE end, and te_radius rounding the TE end; model in tip_cap.py",
        "frame": "u = x / chord along the tip section's chord from the top of the LE; heights "
                 "above the tip section's plane along its normal; lengths in mm",
        "chord_mm": base.chord,
        "params": p.as_dict(),
        "face_inclination_deg": p.inclination_deg(base.chord),
        "fit": {
            "stations": len(rep),
            "core_stations": len(core),
            "max_dev_mm": float(max(r["max_dev"] for r in rep)),
            "rms_dev_mm": rms(rep),
            "core_max_dev_mm": float(max(r["max_dev"] for r in core)),
            "core_rms_dev_mm": rms(core),
            "note": "distance from the CAD cap's cross-section points to the model's",
        },
        "xcad": {
            "file": os.path.basename(cap_res["paths"]["xcad"]),
            "cap_loops": len(cap_res["levels"]),
            "first_cap_loop": cap_res["n_sections"] + 1,
            "height_fractions": [float(v) for v in cap_res["levels"]],
            "note": "loops over the cap after the stack, each at a fixed fraction of the local "
                    "cap height, starting at the top of the LE like the sections; they all meet "
                    "the tip section at the top of the LE and the top of the TE",
        },
        "files": {k: os.path.relpath(v, out_dir) for k, v in cap_res["paths"].items()},
    }


def cap_design_variables(cap_res: dict) -> dict:
    """The cap's design variables for the optimisation stage (the rest held fixed)."""
    p, C = cap_res["params"], cap_res["base"].chord
    return {
        "cap_face_inclination_deg": {
            "value": p.inclination_deg(C),
            "definition": "inclination of the cap's face relative to the tip section, "
                          "arctan((h_te - h_le) / chord); changing it moves h_te with h_le "
                          "and the bow held",
        },
        "cap_edge_radius_mm": {
            "value": p.edge_radius,
            "definition": "edge radius joining the side walls to the cap's crown, also "
                          "rounding the cap's LE end",
        },
        "cap_fixed": {
            "h_le_mm": p.h_le, "b1_mm": p.b1, "b2_mm": p.b2,
            "crown_radius_mm": p.crown_radius if np.isfinite(p.crown_radius) else None,
            "te_radius_mm": p.te_radius,
            "note": "cap parameters held fixed for now",
        },
    }


def planform(lateral, span: float, n_stations: int = 25, frac: float = 0.92):
    """LE and TE traces up the span, for plotting and for sweep/taper fits."""
    ys = np.linspace(0.0, span * frac, n_stations)
    le = np.array([leading_edge_point(lateral, y) for y in ys])
    te = np.array([trailing_edge_point(lateral, y) for y in ys])
    return ys, le, te


def _plot(out_dir, root, cut, span, lateral=None, stack=None, y_top=None,
          y_le_top=None, tip_sec=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    if lateral is not None:
        if y_le_top is not None:          # the edges as far as they go
            ys, le, te = planform(lateral, y_le_top, frac=1.0)
        else:
            ys, le, te = planform(lateral, span)
        ax[0].plot(le[:, CHORD_AXIS], ys, "-", color="0.35", lw=1.2, label="LE")
        ax[0].plot(te[:, CHORD_AXIS], ys, "-", color="0.35", lw=1.2, label="TE")
        ax[0].fill_betweenx(ys, te[:, CHORD_AXIS], le[:, CHORD_AXIS],
                            color="0.85", zorder=0)
    if stack:
        for s in stack:
            if s.tilted:
                continue
            ax[0].plot([s.le[CHORD_AXIS], s.te[CHORD_AXIS]], [s.y, s.y],
                       "-", color="tab:blue", lw=0.3, alpha=0.6, zorder=1)
    if tip_sec is not None:
        ax[0].plot([tip_sec.le[CHORD_AXIS], tip_sec.te[CHORD_AXIS]],
                   [tip_sec.le[SPAN_AXIS], tip_sec.te[SPAN_AXIS]],
                   "-", color="tab:orange", lw=1.6, zorder=3)
        ax[0].annotate(f"tilted tip section, {tip_sec.tilt_deg:.3f} deg",
                       (tip_sec.te[CHORD_AXIS], tip_sec.te[SPAN_AXIS]),
                       textcoords="offset points", xytext=(-2, -26), ha="right",
                       fontsize=7.5, color="tab:orange",
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    if y_le_top is not None and (y_top is None or abs(y_top - y_le_top) > 1e-6):
        ax[0].axhline(y_le_top, color="0.45", lw=1.0, ls=":", zorder=2)
        ax[0].annotate(f"top of LE, y = {y_le_top:.3f}", (0, y_le_top),
                       textcoords="offset points", xytext=(2, 4),
                       fontsize=8, color="0.35")
    if y_top is not None:
        same = y_le_top is not None and abs(y_top - y_le_top) <= 1e-6
        ax[0].axhline(y_top, color="tab:blue", lw=1.0, ls="--", zorder=2)
        ax[0].annotate(("top of LE, y = " if same else "top station, y = ")
                       + f"{y_top:.3f}", (0, y_top),
                       textcoords="offset points", xytext=(2, 4),
                       fontsize=8, color="tab:blue")
    for s in (root, cut):
        ax[0].plot([s.le[CHORD_AXIS], s.te[CHORD_AXIS]],
                   [s.le[SPAN_AXIS], s.te[SPAN_AXIS]], "o-", ms=7, color="crimson",
                   lw=1.2, zorder=3)
    ax[0].axhline(0, color="0.8", lw=0.6)
    ax[0].axvline(0, color="0.8", lw=0.6)
    ax[0].annotate("LE", root.le[[CHORD_AXIS, SPAN_AXIS]], textcoords="offset points",
                   xytext=(-4, -14), color="crimson", ha="center")
    ax[0].annotate("TE", root.te[[CHORD_AXIS, SPAN_AXIS]], textcoords="offset points",
                   xytext=(4, -14), color="crimson", ha="center")
    ax[0].set(title="planform: the four corner points", xlabel="x [mm]",
              ylabel="y (span) [mm]", ylim=(-15, span * 1.02))
    ax[0].set_aspect("equal")

    shown = [(root, f"root  y=0, c={root.chord:.3f}"),
             (cut, f"cut   y={cut.y:g}, c={cut.chord:.3f}")]
    if tip_sec is not None:
        shown.append((tip_sec, f"tip, untilted, c={tip_sec.chord:.3f}"))
    for s, lab in shown:
        ax[1].plot(s.airfoil_raw[:, 0], s.airfoil_raw[:, 1], lw=1.0, label=lab)
        ax[2].plot(s.airfoil_norm[:, 0], s.airfoil_norm[:, 1], ".-", ms=2, lw=0.7, label=lab)
    ax[1].set(title="airfoil sections [mm]", xlabel="x aft of LE [mm]", ylabel="y [mm]")
    ax[2].set(title="normalised, Selig order", xlabel="x/c", ylabel="y/c")
    for a in ax[1:]:
        a.set_aspect("equal")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sections_check.png"), dpi=150)
    plt.close(fig)


def _plot_parameters(out_dir, param):
    """Six panels of the Step-6 distributions against span."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    d_all, fits = param["distributions"], param["fits"]
    h = d_all.horizontal
    tips = np.flatnonzero(~h)

    class _Rows:
        """The horizontal rows of every distribution, for the lines."""
        def __getattr__(self, name):
            return getattr(d_all, name)[h]

    d = _Rows()
    y = d.y

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    a = ax.ravel()

    def tip_mark(axis, name, color="tab:red", label=False):
        """The tilted tip section: one marker at the height of its LE."""
        for i in tips:
            axis.plot(getattr(d_all, name)[i], d_all.y[i], "*", color=color, ms=9,
                      zorder=5, label="tilted tip section" if label else None)

    def label(name, text):
        return f"{text}  [{fits[name]['form']}]" if name in fits else text

    def widen(axis, floor):
        """Hold a panel to at least `floor` of x range. A rudder with no rake
        and no twist reads zero to about 1e-5 deg; left to autoscale, the panel
        would magnify that noise until it filled the axis and looked like a
        trend. The numbers themselves are untouched, in parameters.dat."""
        lo, hi = axis.get_xlim()
        if hi - lo < floor:
            mid = 0.5 * (lo + hi)
            axis.set_xlim(mid - 0.5 * floor, mid + 0.5 * floor)

    a[0].plot(d.chord, y, "-", color="tab:blue", lw=1.4,
              label=label("chord", "chord"))
    a[0].plot(d.chord_x, y, ":", color="0.5", lw=1.0, label="chordwise projection")
    tip_mark(a[0], "chord", label=True)
    a[0].set(title="chord distribution", xlabel="chord [mm]")

    a[1].plot(d.sweep_le, y, "-", color="tab:blue", lw=1.4,
              label=label("sweep_le", "LE, cumulative"))
    a[1].plot(d.sweep_le_local, y, "--", color="tab:blue", lw=1.0,
              label="LE, local")
    a[1].plot(d.sweep_c4, y, "-", color="tab:orange", lw=1.2,
              label=label("sweep_c4", "c/4, cumulative"))
    a[1].plot(d.sweep_te, y, "-", color="tab:green", lw=1.0,
              label=label("sweep_te", "TE, cumulative"))
    a[1].set(title="sweep, positive aft", xlabel="sweep [deg]")

    a[2].plot(d.dihedral, y, "-", color="tab:blue", lw=1.4,
              label=label("dihedral", "dihedral / rake angle, cumulative"))
    a[2].plot(d.dihedral_local, y, "--", color="tab:blue", lw=1.0, label="local")
    a[2].set(title="dihedral (propeller rake angle)", xlabel="angle [deg]")
    a2 = a[2].twiny()
    a2.plot(d.rake, y, "-", color="tab:red", lw=1.0)
    a2.set_xlabel("rake offset [mm]", color="tab:red")
    a2.tick_params(axis="x", colors="tab:red")

    a[3].plot(d.twist, y, "-", color="tab:blue", lw=1.4,
              label=label("twist", "twist"))
    a[3].plot(d_all.tilt, d_all.y, "-", color="tab:red", lw=1.0,
              label="tilt, TE up (0 below the tip section)")
    tip_mark(a[3], "tilt")
    if len(tips):
        i = tips[-1]
        a[3].annotate(f"tip tilt {d_all.tilt[i]:.3f} deg", (d_all.tilt[i], d_all.y[i]),
                      textcoords="offset points", xytext=(-6, -12), ha="right",
                      fontsize=8, color="tab:red")
    a[3].set(title="twist (nose-up) and section tilt", xlabel="angle [deg]")

    a[4].plot(d.t_over_c, y, "-", color="tab:blue", lw=1.4,
              label=label("t_over_c", "t/c"))
    a[4].plot(d.camber, y, "-", color="tab:orange", lw=1.2,
              label=label("camber", "max camber / c"))
    a[4].plot(d.r_le, y, "-", color="tab:green", lw=1.0,
              label=label("r_le", "LE radius / c"))
    tip_mark(a[4], "t_over_c", color="tab:blue", label=True)
    a[4].set(title="section shape", xlabel="fraction of chord")

    a[5].plot(d.x_le, y, "-", color="tab:blue", lw=1.4, label="LE")
    a[5].plot(d.x_le + 0.25 * (d.x_te - d.x_le), y, "--", color="tab:orange",
              lw=1.0, label="c/4")
    a[5].plot(d.x_te, y, "-", color="tab:green", lw=1.4, label="TE")
    a[5].fill_betweenx(y, d.x_le, d.x_te, color="0.88", zorder=0)
    for i in tips:
        a[5].plot([d_all.x_le[i], d_all.x_te[i]], [d_all.y[i], d_all.y_te[i]], "-",
                  color="tab:red", lw=1.4, label="tilted tip section")
    a[5].set(title="planform and reference lines", xlabel="x [mm]")
    a[5].set_aspect("equal")

    for axis in a:
        axis.set_ylabel("y (span) [mm]")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, loc="best")

    for axis in (a[1], a[2], a[3]):
        widen(axis, 1.0)                 # degrees
    widen(a2, 1.0)                       # millimetres of rake
    widen(a[4], 0.05)                    # fraction of chord

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "parameters_check.png"), dpi=150)
    plt.close(fig)


def _plot_cap(out_dir, cap_res):
    """Side view of the cap's crest and two cross-sections, CAD against the model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    p, rep, base = cap_res["params"], cap_res["report"], cap_res["base"]
    measured = cap_res["measured"]
    C = base.chord
    BLUE, AQUA, INK, GREY = "#2a78d6", "#1baf7a", "#0b0b0b", "#8a8984"

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.9], hspace=0.45, wspace=0.12)
    a0 = fig.add_subplot(gs[0, :])
    xs = np.linspace(0.0, C, 800)
    a0.plot([r["x"] for r in rep], [r["top_cad"] for r in rep], "o", ms=4, color=INK,
            label="CAD: top of the cap", zorder=3)
    a0.plot(xs, tip_cap.cap_top(xs, p, base), "-", color=BLUE, lw=2,
            label="model: top of the cap", zorder=2)
    a0.plot(xs, p.face(xs / C), "--", color=AQUA, lw=1.6,
            label="model: face (before the crown and the end radii)", zorder=1)
    a0.axhline(0.0, color="0.35", lw=1.0)
    a0.set(xlabel="along the tip section's chord, top of the LE to top of the TE [mm]",
           ylabel="height above the tip section [mm]",
           title=f"tip cap: face {p.h_le:.3f} -> {p.h_te:.3f} mm, crown radius "
                 f"{p.crown_radius:.1f} mm, edge radius {p.edge_radius:.3f} mm, TE corner radius "
                 f"{p.te_radius:.2f} mm (vertical scale exaggerated)")
    a0.title.set_fontsize(10)
    a0.legend(loc="lower right", fontsize=8.5, frameon=False)
    a0.grid(alpha=0.3)

    for col, u in enumerate((0.3, 0.8)):
        ax = fig.add_subplot(gs[1, col])
        st = min(measured, key=lambda s: abs(s.x - u * C))
        pr = tip_cap.profile(st.x, p, base, n=800)
        ax.plot(np.r_[st.s, -st.s], np.r_[st.h, st.h], ".", ms=2, color=GREY, label="CAD")
        ax.plot(np.r_[-pr[:, 0], pr[::-1, 0]], np.r_[pr[:, 1], pr[::-1, 1]], "-", color=BLUE,
                lw=1.8, label="model")
        ax.axhline(0.0, color="0.35", lw=1.0)
        r = min(rep, key=lambda q: abs(q["x"] - st.x))
        ax.set(xlabel="across the thickness [mm]", ylabel="height [mm], vertical x2",
               title=f"cross-section at {st.x / C:.0%} of the chord: max deviation "
                     f"{r['max_dev']:.3f} mm")
        ax.title.set_fontsize(10)
        ax.set_aspect(2.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower center", fontsize=8, frameon=False)
    fig.savefig(os.path.join(out_dir, "cap_check.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Extract rudder section geometry from a CAD file.")
    ap.add_argument("cad", nargs="?", default="Wind_Tunnel_Rudder.stp")
    ap.add_argument("--cut", type=float, default=80.0,
                    help="spanwise cut height in mm above the root (default 80)")
    ap.add_argument("--num-pts", type=int, default=DEFAULT_NUM_PTS,
                    help="points per airfoil section (default 200)")
    ap.add_argument("--num-sec", type=int, default=DEFAULT_NUM_SEC,
                    help="horizontal cross-sections from the root up to d below the "
                         "top of the LE, the tilted tip section going on top "
                         "(default 200; 0 skips the stack)")
    ap.add_argument("--top", type=float, default=None,
                    help="stop the stack at this height in mm instead of the top of "
                         "the LE (horizontal sections only, no tip section)")
    ap.add_argument("--tip-d", type=float, default=None,
                    help="gap d in mm between the top horizontal section and the top "
                         "of the LE (default: half the horizontal step)")
    ap.add_argument("--no-tip", action="store_true",
                    help="no tilted tip section: horizontal sections right up to the "
                         "top of the LE, as before")
    ap.add_argument("--no-cap", action="store_true",
                    help="skip Step 7, the tip cap fit")
    ap.add_argument("--cap-loops", type=int, default=DEFAULT_CAP_LOOPS,
                    help=f"loops over the cap in the XCAD file with the cap "
                         f"(default {DEFAULT_CAP_LOOPS})")
    ap.add_argument("--cap-top", type=float, default=DEFAULT_CAP_TOP,
                    help="height fraction of the last cap loop; 1 puts it on the crest "
                         f"line itself (default {DEFAULT_CAP_TOP})")
    ap.add_argument("--cap-stations", type=int, default=DEFAULT_CAP_STATIONS,
                    help=f"cross-sections cut through the cap for the fit "
                         f"(default {DEFAULT_CAP_STATIONS})")
    ap.add_argument("--spacing", choices=("cosine", "uniform"), default="cosine")
    ap.add_argument("--fit-deg", type=int, default=DEFAULT_FIT_DEG,
                    help="degree of the polynomial fitted to each spanwise "
                         f"distribution (default {DEFAULT_FIT_DEG})")
    ap.add_argument("--xcad-units", choices=("m", "mm"), default="m",
                    help="units of the XCAD section file (default m, as in ORCA101.dat)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    def progress(i, n):
        if i == n or i % max(1, n // 20) == 0:
            print(f"\r  extracting sections {i}/{n}", end="", flush=True)
        if i == n:
            print()

    res = run(args.cad, args.cut, args.num_pts, args.out_dir,
              args.spacing, not args.no_plot, args.num_sec, args.top,
              progress if args.num_sec else None, args.fit_deg,
              xcad_units=args.xcad_units, tip=not args.no_tip, tip_d=args.tip_d,
              cap=not args.no_cap, cap_loops=args.cap_loops, cap_top=args.cap_top,
              cap_stations=args.cap_stations)

    root, cut = res["root"], res["cut"]
    print(f"CAD              : {args.cad}")
    print(f"span             : {res['span']:.4f} mm")
    print("frame            : origin at the root LE, +X chordwise LE->TE, "
          "+Y span, +Z thickness")
    print(f"                   root LE was at "
          f"{np.round(res['le_original'], 6).tolist()} in the CAD"
          + ("; chord ran TE->LE there, so the shape was turned 180 deg about "
             "the span axis" if res["rotated"] else "; chord already ran LE->TE"))
    print()
    print("corner points in the new frame (x, y, z) [mm]")
    for s, tag in ((root, "root"), (cut, f"y={cut.y:g}")):
        print(f"  {tag:>8}  LE {s.le[0]:12.6f} {s.le[1]:10.6f} {s.le[2]:10.6f}")
        print(f"  {tag:>8}  TE {s.te[0]:12.6f} {s.te[1]:10.6f} {s.te[2]:10.6f}   chord {s.chord:.6f}")
    print()
    print(f"taper ratio c(cut)/c(root) : {cut.chord / root.chord:.6f}")
    print(f"LE sweep over the cut      : "
          f"{np.degrees(np.arctan2(cut.le[0] - root.le[0], cut.y - root.y)):.4f} deg aft")
    print(f"TE sweep over the cut      : "
          f"{np.degrees(np.arctan2(cut.te[0] - root.te[0], cut.y - root.y)):.4f} deg aft")

    if res["stack"]:
        st = res["stack"]
        flat = [s for s in st if not s.tilted]
        tip = res["tip_section"]
        capped = abs(res["y_top"] - res["y_le_top"]) > 1e-6
        print()
        if tip is not None:
            lay = res["layout"]
            print(f"section stack              : {len(flat)} horizontal sections, "
                  f"y = 0 .. {flat[-1].y:.6f} mm, step {lay['dz']:.6f} mm,")
            print(f"  d = {lay['d']:.6f} mm below the top of the LE"
                  + (" (half a step)" if lay["d_default"] else "")
                  + ", + 1 tilted tip section on top")
            print(f"  chord {flat[0].chord:.6f} at the root -> "
                  f"{flat[-1].chord:.6f} at y = {flat[-1].y:.4f}")
            print("top of the LE              : "
                  f"{np.round(res['le_top'], 6).tolist()}"
                  + (f" ({res['y_top_how']})" if res["y_top_how"] else ""))
            print("top of the TE              : "
                  f"{np.round(res['te_top'], 6).tolist()}"
                  + (f" ({res['te_top_how']})" if res["te_top_how"] else ""))
            print(f"tilted tip section         : LE {np.round(tip.le, 6).tolist()}, "
                  f"TE {np.round(tip.te, 6).tolist()}")
            print(f"  tilt {tip.tilt_deg:.6f} deg about the thickness axis, TE up; "
                  f"chord {tip.chord:.6f} mm in its own plane")
            print("  untilted copy            : " + ", ".join(
                os.path.basename(p) for p in res["tip_paths"].values()))
            print("  the tip cap above it (up to y = "
                  f"{res['span']:.4f} mm) is not covered yet")
        else:
            print(f"section stack              : {len(st)} sections, "
                  f"y = 0 .. {res['y_top']:.6f} mm, step {st[1].y - st[0].y:.6f} mm")
            print(f"  chord {st[0].chord:.6f} at the root -> "
                  f"{st[-1].chord:.6f} at y = {res['y_top']:.4f}")
            if capped:
                print(f"top station                : requested; the leading edge itself "
                      f"runs to y = {res['y_le_top']:.6f} mm")
            else:
                print("top station                : top of the leading edge"
                      + (f" ({res['y_top_how']})" if res["y_top_how"] else ""))
            print(f"  nothing above y = {res['y_top']:.4f} mm is covered; above "
                  f"y = {res['y_le_top']:.4f} the inclined")
            print("  tip surface means a horizontal cut is not a complete airfoil at all")
        print(f"XCAD file                  : {os.path.basename(res['xcad_path'])}, "
              f"{len(st)} numbered sections root first, {len(st[0].airfoil_raw)} points "
              f"each (closed at the LE), height along z, in {'metres' if res['xcad_units'] == 'm' else 'mm'}"
              + (", the last one tilted" if tip is not None else ""))

    _report_parameters(res["param"])
    if res["cap"] is not None:
        _report_cap(res["param"]["tip_cap"])
    print(f"\noutputs -> {res['out_dir']}")


def _report_cap(capd: dict) -> None:
    """Console form of Step 7."""
    p, f, x = capd["params"], capd["fit"], capd["xcad"]
    crown = p["crown_radius"]
    print()
    print("tip cap (Step 7)           : face over the tip section, crown, edge radius, TE radius")
    print(f"  face height above the tip section : {p['h_le']:.4f} mm at the LE end,"
          f" {p['h_te']:.4f} mm at the TE end (bow {p['b1']:+.4f}, {p['b2']:+.4f})")
    print(f"  face inclination vs the tip section: {capd['face_inclination_deg']:.4f} deg")
    print(f"  crown radius {('%.3f mm' % crown) if crown else 'flat'},"
          f" edge radius {p['edge_radius']:.4f} mm, TE corner radius {p['te_radius']:.4f} mm")
    print(f"  fit over {f['stations']} cross-sections: max {f['max_dev_mm']:.4f} mm,"
          f" rms {f['rms_dev_mm']:.4f} mm (core {f['core_max_dev_mm']:.4f} / "
          f"{f['core_rms_dev_mm']:.4f})")
    print(f"  XCAD with the cap            : {x['file']}, loops {x['first_cap_loop']}.."
          f"{x['first_cap_loop'] + x['cap_loops'] - 1} over the cap")
    print("  design variables             : cap_face_inclination_deg, cap_edge_radius_mm"
          " (with tip_tilt_deg)")


def _report_parameters(param: dict) -> None:
    """Console form of Step 6."""
    d, fits, summary = param["distributions"], param["fits"], param["summary"]
    tip = param.get("tip_section")

    print()
    print(f"parametrisation            : {len(d.y)} stations,"
          f" y = {d.y[0]:.4f} .. {d.y[-1]:.4f} mm"
          + (", the last one the tilted tip section" if tip is not None else ""))
    if tip is not None:
        print(f"  tip section: tilt {tip['tilt_deg']:.6f} deg (ratio {tip['tilt_ratio']:.6f}),"
              f" chord {tip['chord_mm']:.4f} mm, t/c {tip['t_over_c']:.5f},"
              f" twist {tip['twist_deg']:.2e} deg")
        print("  its tilt is recorded as the design variable 'tip_tilt_deg';"
              " every other row has tilt 0")
        print("  the fits and the planform summary below use the horizontal rows only")
    print(f"  root chord {summary['chord_root_mm']:.4f} mm ->"
          f" tip chord {summary['chord_tip_mm']:.4f} mm,"
          f" taper {summary['taper_ratio']:.6f}")
    print(f"  planform area {summary['area_mm2']:.2f} mm2,"
          f" MAC {summary['mean_aerodynamic_chord_mm']:.4f} mm,"
          f" aspect ratio {summary['aspect_ratio']:.4f}"
          " (horizontal sections only)")
    print()
    print(f"  {'distribution':<16s} {'form':<22s} {'root':>12s} {'tip':>12s}"
          f" {'mean':>12s} {'spread':>11s}")
    for name, unit, key, _desc in _DIST_COLUMNS:
        if key is None:
            continue
        f = fits[name]
        if "mean" not in f:
            print(f"  {name:<16s} {f['form']:<22s}"
                  f" {f['n_valid']} of {int(d.horizontal.sum())} stations finite")
            continue
        print(f"  {name:<16s} {f['form']:<22s} {f['root']:12.6f} {f['tip']:12.6f}"
              f" {f['mean']:12.6f} {f['max_dev_from_mean']:11.3e}  {unit}")
    print("  spread is the largest departure from the mean; a 'constant' row is")
    print("  flat to within its reporting tolerance, nothing has been rounded to it")


if __name__ == "__main__":
    main()
