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
         `num_sec`-section stack from the root up to the top of the LE.

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
    """Exact (no tolerance gap) bounding box as two numpy points."""
    box = Bnd_Box()
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


def section_curves(lateral, y: float):
    """
    Step 3 - one horizontal planar cut at spanwise station `y`.

    Returns the section's curves. Everything else about a station (outline,
    LE, TE) is derived from this one result, so a station costs one cut.
    """
    plane = gp_Pln(gp_Pnt(*[0.0 if i != SPAN_AXIS else y for i in range(3)]), _SPAN_DIR)
    algo = BRepAlgoAPI_Section(lateral, plane, False)
    algo.ComputePCurveOn1(True)
    algo.Approximation(True)
    algo.Build()
    if not algo.IsDone():
        raise RuntimeError(f"Section at y = {y} failed")

    curves = []
    exp = TopExp_Explorer(algo.Shape(), TopAbs_EDGE)
    while exp.More():
        curves.append(BRepAdaptor_Curve(_as_edge(exp.Current())))
        exp.Next()
    if not curves:
        raise RuntimeError(f"Section at y = {y} is empty - outside the part?")
    return curves


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


def _extreme_on_curves(curves, sign: int) -> np.ndarray:
    """
    Chordwise extremum over a section's curves (sign = -1 -> TE, +1 -> LE),
    bracketed on a dense parameter sample then refined by golden section on
    the curve itself, not read off a polyline.
    """
    best_val, best_pnt = -np.inf, None
    for curve in curves:
        u0, u1 = curve.FirstParameter(), curve.LastParameter()

        def f(u):
            p = curve.Value(u)
            return sign * [p.X(), p.Y(), p.Z()][CHORD_AXIS]

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


def leading_edge_top(shape, lateral, span: float, n_probe: int = 10):
    """
    Spanwise station where the leading edge ends.

    The tip surface is inclined, so above this station a horizontal cut no
    longer produces a complete airfoil - the nose of the cut is the tip blend
    rather than the true leading edge.

    Preferred route: probe the LE point over the lower span, find the B-rep
    edges those points actually lie on, and take the top of that curve. That
    is exact whenever the exporter wrote the leading edge as an edge, which is
    the usual case for a lofted surface. Falls back to bisecting where the LE
    point leaves the locus extrapolated from below.

    Returns (y_top, how).
    """
    probes = np.linspace(0.02, 0.60, n_probe) * span
    pts = [leading_edge_point(lateral, float(y)) for y in probes]
    tol = max(1.0e-7, 1.0e-6 * span)

    tops = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        poly = _edge_polyline(_as_edge(exp.Current()), 200)
        if any(_point_to_polyline(q, poly) < tol for q in pts):
            tops.append(float(poly[:, SPAN_AXIS].max()))
        exp.Next()
    if tops:
        return max(tops), "top of the b-rep leading-edge curve"

    # fallback - the LE locus is smooth below the break and leaves it after
    fit = np.polyfit(probes, [p[CHORD_AXIS] for p in pts], 1)

    def on_locus(y):
        try:
            return abs(leading_edge_point(lateral, y)[CHORD_AXIS]
                       - np.polyval(fit, y)) < tol
        except RuntimeError:
            return False

    lo, hi = float(probes[-1]), span
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if on_locus(mid):
            lo = mid
        else:
            hi = mid
    return lo, "bisection on the extrapolated LE locus"


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
    y: float                 # spanwise station in the reframed system
    le: np.ndarray           # leading-edge point  (3,)
    te: np.ndarray           # trailing-edge point (3,)
    chord: float
    outline: np.ndarray      # dense closed loop, (N, 3)
    airfoil_raw: np.ndarray  # (num_pts, 2) mm,  Selig order, airfoil frame
    airfoil_norm: np.ndarray  # (num_pts, 2) normalised by chord

    def as_dict(self):
        return {
            "y": self.y,
            "chord": self.chord,
            "le": self.le.tolist(),
            "te": self.te.tolist(),
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


def extract_section(lateral, y: float, num_pts: int = DEFAULT_NUM_PTS,
                    spacing: str = "cosine",
                    dense: int = _DENSE_PER_EDGE) -> Section:
    """Steps 3-5 for a single spanwise station, from a single planar cut."""
    curves = section_curves(lateral, y)
    outline = _chain([_curve_polyline(c, dense) for c in curves])
    le = _extreme_on_curves(curves, LE_SIGN)
    te = _extreme_on_curves(curves, TE_SIGN)
    chord = float(te[CHORD_AXIS] - le[CHORD_AXIS])

    upper, lower = _split_surfaces(outline, le, te)
    up2, lo2 = _to_airfoil_frame(upper, le), _to_airfoil_frame(lower, le)

    n_half = (num_pts + 1) // 2
    up = _resample(up2, n_half, spacing)[::-1]              # TE -> LE
    lo = _resample(lo2, num_pts - n_half + 1, spacing)      # LE -> TE
    raw = np.vstack([up, lo[1:]])                           # Selig order

    return Section(y=float(y), le=le, te=te, chord=chord, outline=outline,
                   airfoil_raw=raw, airfoil_norm=raw / chord)


def extract_stack(lateral, y_top: float, num_sec: int = DEFAULT_NUM_SEC,
                  num_pts: int = DEFAULT_NUM_PTS, spacing: str = "cosine",
                  y_root: float = 0.0, progress=None):
    """
    `num_sec` cross-sections evenly spaced from the root up to `y_top`, the
    top of the leading edge. Returned bottom-to-top; the writer flips them.

    Nothing above `y_top` is covered: the tip surface is inclined, so a
    horizontal cut up there cuts through the tip blend and no longer returns
    a complete airfoil. That part of the shape needs a different treatment.
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


def write_raw_dat(path: str, coords: np.ndarray, y: float) -> None:
    """Whitespace-delimited .dat, `#` comment header, loadable with np.loadtxt."""
    with open(path, "w") as fh:
        fh.write("# airfoil frame, millimetres, Selig order (TE -> upper -> LE -> lower -> TE)\n")
        fh.write(f"# section at y = {y:.6f} mm; x = 0 at this section's LE and grows aft,"
                 " y = thickness (+Z side up)\n")
        fh.write(f"# {'x_mm':>18s} {'y_mm':>18s}\n")
        for x, yy in coords:
            fh.write(f"{_snap(x):20.9f} {_snap(yy):18.9f}\n")


def write_section_stack(path: str, sections, source: str, y_top: float,
                        how: str = "", y_le_top: float | None = None) -> None:
    """
    Every cross-section in one .dat, ordered TOP to BOTTOM.

    All metadata sits on `#` lines, so the numbers load in one call:
        np.loadtxt(path).reshape(n_sections, n_points, 2)
    with section 0 the topmost.
    """
    ordered = sorted(sections, key=lambda s: -s.y)
    n_pts = len(ordered[0].airfoil_raw)

    with open(path, "w") as fh:
        fh.write("# Rudder cross-sectional airfoil coordinates\n")
        fh.write(f"# source        : {source}\n")
        fh.write("# frame         : origin at the root leading edge;"
                 " +X LE->TE, +Y root->tip, +Z thickness\n")
        fh.write(f"# n_sections    : {len(ordered)}\n")
        fh.write(f"# n_points      : {n_pts} per section\n")
        fh.write(f"# section order : TOP to BOTTOM, y = {ordered[0].y:.6f}"
                 f" down to y = {ordered[-1].y:.6f} mm, evenly spaced\n")
        fh.write("# point order   : Selig, TE -> upper -> LE -> lower -> TE\n")
        fh.write("# columns       : x_mm y_mm in that section's own airfoil frame,"
                 " x = 0 at its LE\n")
        fh.write("#                 and growing aft, y = thickness with the +Z side"
                 " positive\n")
        fh.write("# 3D recovery   : X = x_le + x_mm,  Y = y_section,  Z = y_mm\n")
        fh.write("# load          : np.loadtxt(path).reshape"
                 f"({len(ordered)}, {n_pts}, 2)\n")
        if y_le_top is not None and abs(y_top - y_le_top) > 1.0e-6:
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
                     f"   x_te = {s.te[CHORD_AXIS]:.6f}\n")
            fh.write(f"# {'x_mm':>18s} {'y_mm':>18s}\n")
            for x, yy in s.airfoil_raw:
                fh.write(f"{_snap(x):20.9f} {_snap(yy):18.9f}\n")


def write_corner_points(path_dat: str, path_json: str, sections, rotated) -> None:
    """
    LE and TE corner points, one spanwise station per ROW:

        x1 y1 z1 x2 y2 z2      point 1 = LE, point 2 = TE

    Row 1 is the root, the last row is the top of the leading edge, so the
    first three columns trace the LE up the span and the last three trace the
    TE. Ordered root first, the opposite of the section stack, which the user
    asked to have running top to bottom.
    """
    ordered = sorted(sections, key=lambda s: s.y)

    with open(path_dat, "w") as fh:
        fh.write("# LE and TE corner points, one spanwise station per row\n")
        fh.write("# frame   : origin at the root leading edge;"
                 " +X LE->TE, +Y root->tip, +Z thickness\n")
        fh.write("# point 1 : leading edge      point 2 : trailing edge\n")
        fh.write(f"# rows    : {len(ordered)}, root first,"
                 f" y = {ordered[0].y:.6f} up to y = {ordered[-1].y:.6f} mm\n")
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
                     "le": s.le.tolist(), "te": s.te.tolist()}
                    for s in ordered
                ],
            },
            fh,
            indent=2,
        )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(cad_path: str, cut_height: float = 80.0, num_pts: int = DEFAULT_NUM_PTS,
        out_dir: str | None = None, spacing: str = "cosine", plot: bool = True,
        num_sec: int | None = DEFAULT_NUM_SEC, y_top: float | None = None,
        progress=None):
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
    y_le_top, how = leading_edge_top(shape, lateral, span)
    if y_top is None:
        y_top = y_le_top
    elif y_top > y_le_top + 1.0e-6:
        print(f"  warning: requested top y = {y_top:.6f} is above the top of the "
              f"leading edge at y = {y_le_top:.6f}; sections up there cut through "
              f"the tip blend and may fail")

    stack = None
    if num_sec:
        stack = extract_stack(lateral, y_top, num_sec, num_pts, spacing,
                              progress=progress)
        write_section_stack(os.path.join(out_dir, "sections_stack_raw_mm.dat"),
                            stack, os.path.basename(cad_path), y_top, how, y_le_top)

    # one row per station: the LE/TE edge table, root first up to the top of
    # the LE. With no stack it falls back to the two named cuts.
    write_corner_points(os.path.join(out_dir, "corner_points.dat"),
                        os.path.join(out_dir, "corner_points.json"),
                        stack if stack else [root, cut], rotated)

    if plot:
        _plot(out_dir, root, cut, span, lateral, stack, y_top, y_le_top)

    return {"shape": shape, "lateral": lateral, "trsf": trsf,
            "le_original": le_orig, "rotated": rotated,
            "root": root, "cut": cut,
            "span": span, "out_dir": out_dir, "y_top": y_top,
            "y_le_top": y_le_top, "y_top_how": how, "stack": stack}


def planform(lateral, span: float, n_stations: int = 25, frac: float = 0.92):
    """LE and TE traces up the span, for plotting and for sweep/taper fits."""
    ys = np.linspace(0.0, span * frac, n_stations)
    le = np.array([leading_edge_point(lateral, y) for y in ys])
    te = np.array([trailing_edge_point(lateral, y) for y in ys])
    return ys, le, te


def _plot(out_dir, root, cut, span, lateral=None, stack=None, y_top=None,
          y_le_top=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    if lateral is not None:
        ys, le, te = planform(lateral, span)
        ax[0].plot(le[:, CHORD_AXIS], ys, "-", color="0.35", lw=1.2, label="LE")
        ax[0].plot(te[:, CHORD_AXIS], ys, "-", color="0.35", lw=1.2, label="TE")
        ax[0].fill_betweenx(ys, te[:, CHORD_AXIS], le[:, CHORD_AXIS],
                            color="0.85", zorder=0)
    if stack:
        for s in stack:
            ax[0].plot([s.le[CHORD_AXIS], s.te[CHORD_AXIS]], [s.y, s.y],
                       "-", color="tab:blue", lw=0.3, alpha=0.6, zorder=1)
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

    for s, lab in ((root, f"root  y=0, c={root.chord:.3f}"),
                   (cut, f"cut   y={cut.y:g}, c={cut.chord:.3f}")):
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


def main():
    ap = argparse.ArgumentParser(description="Extract rudder section geometry from a CAD file.")
    ap.add_argument("cad", nargs="?", default="Wind_Tunnel_Rudder.stp")
    ap.add_argument("--cut", type=float, default=80.0,
                    help="spanwise cut height in mm above the root (default 80)")
    ap.add_argument("--num-pts", type=int, default=DEFAULT_NUM_PTS,
                    help="points per airfoil section (default 200)")
    ap.add_argument("--num-sec", type=int, default=DEFAULT_NUM_SEC,
                    help="cross-sections from the root to the top of the LE "
                         "(default 200; 0 skips the stack)")
    ap.add_argument("--top", type=float, default=None,
                    help="override the top station in mm instead of detecting it")
    ap.add_argument("--spacing", choices=("cosine", "uniform"), default="cosine")
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
              progress if args.num_sec else None)

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
        capped = abs(res["y_top"] - res["y_le_top"]) > 1e-6
        print()
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
    print(f"\noutputs -> {res['out_dir']}")


if __name__ == "__main__":
    main()
