"""
blade_extraction.py

The six radial distributions of a propeller blade (pitch, chord, thickness,
camber, skew and rake over r/R), its section shapes and its diameter, backed
out of its CAD (STEP or IGES) and written as an XGeom blade component file
(xgeom_files.py, claude/xgeom-file-formats.md) that the design tool, the
optimiser and the MSc code read (blade_modify.load_component). The one-off
bb2/bb2_sections.py did this for BB2's propeller with its numbers built in;
this is the general script (13 Nov 2026, the user's request).

What the file must hold: the blades as faces or solids (the hub may be there or
not: faces of revolution about the axis are left out of the blades). Nothing
else is assumed: the axis, which end of a section is the leading edge, which way
the propeller turns and the blade count are found.

How (per blade, the one --blade picks, the first found by default):
  axis      found from the faces' edges: a blade's root edge (and a tip edge)
            lies on a cylinder about the axis, so of the candidates (the file's
            x, y and z and the principal axes of the surface points, which for
            3 blades or more stand apart in the propeller's axis) the one about
            which an edge is roundest is the axis; --axis x, y or z names it
            instead. The shape is turned so the axis is x.
  cuts      cylinders about the axis at --stations radii from the root (where
            the faces start) to the tip (the largest radius, R; --diameter sets
            D = 2R instead, the cuts stay where the blade is), clustered at both
            ends; each cut one closed loop, unrolled onto (s = r theta, x).
  LE, TE    the two points of the loop farthest apart, the leading edge the
            blunter end (the thickness a little way in from each end,
            Rudder_geom_extraction's test). The LE is upstream: if it lies at
            the smaller x the axis is turned round (a rotation, the frame stays
            right-handed). If the LE then lies at the smaller theta the propeller
            is left-handed in para.py's convention and is mirrored (y to -y),
            which the file notes: XGeom builds the mirror image.
  pitch     P = 2 pi r tan(phi), phi the nose-tail line's angle on the unrolled
            cylinder
  chord     c' = c / 1.005, c the nose-tail length: para.py and BladeSurface
            close a section at x/c' = 1.005 (the last 0.5 % is their rounded
            TE); beyond blade_surface_new.TIP_TAPER_START, c/D is the measured
            chord over BladeSurface's tip taper, so the CAD gets the measured one
  shape     camber and half thickness at para.py's 26 stations of x/c'
            (airfoils.shape); t/c and f/c their maxima over c'. The shape at
            --ref-rr (0.7) is the component's section table; every cut's goes
            to <case>_section_tables.dat (section_tables: bb2_build.py's
            per-radius shapes)
  skew      theta_root - theta of the reference point half the chord c' from
            the LE on the nose-tail line, deg (para.py puts the root's skew at
            0: the root's reference point sets theta_root)
  rake      x0 - x of the reference point + r skew tan(phi), over D, with x0
            the generator line (--x0, mm in the file's frame along the axis;
            default the root's reference point, so the root's rake is 0)
  tip       the row at r/R 1 continued from the last three cuts (a quadratic
            per column); --tip-pitch sets its P/D (a drawing's value)
  hub       not read from the file: --hub BASE TOP HEIGHT X (mm; prop_hub.py's
            cone, X the base plane on the axis in the file's frame) is written as
            hub_base_m, hub_top_m, hub_x_m and hub_height_m, else a cylinder of
            the root radius, --hub-height long (default 2.02 times the root
            section's axial extent, to 0.1 m; hub_new's margin), centred on x0

Written (--out, default outputs/blade): <case>.txt (the component file: every
distribution a table curve, radius, hub radius and hub height held, the hub,
the section tables), <case>_section.dat (the shape at --ref-rr), <case>_section_
tables.dat, <case>_cuts.dat (the loops, x y z mm in the file's frame, a block
per r/R), <case>_check.png (the distributions against r/R). Then the blade is
rebuilt from the component file (blade_modify, BladeSurface with the one
section table) and each cut is compared with it on its cylinder: the largest
and RMS distance per cut are printed (the one table's shape is the error away
from --ref-rr; with the section at each radius, as bb2_build.py builds, it is
the fit's).

    python blade_extraction.py prop.igs --case bb2_prop --out bb2/outputs --diameter 5000 \\
        --x0 1353.7785 --hub 797.2206 138.0353 1568.754 1929.9057 --tip-pitch 0.8296
    python blade_extraction.py blade.step                        # any propeller: D measured, the hub a cylinder

Needs pythonocc-core (OCC.Core) or cadquery-ocp (OCP), through xcad_loft's shim.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import airfoils                                            # noqa: E402
import blade_modify as BM                                  # noqa: E402
import xcad_loft as XL                                     # noqa: E402
import xgeom_files as XF                                   # noqa: E402

TE_END = 1.005                     # para.py's closing station, x/c'
STATIONS = airfoils.STATIONS       # para.py's 26 stations
ROOT_IN = 1.0e-4                   # the root cut this fraction of R above the faces' lowest point (a cut on
TIP_IN = 1.0e-3                    # their edge fails), the tip cut this fraction below the largest radius;
                                   # either is moved in by steps of it until the cut closes (up to END_TRIES)
END_TRIES = 8
BLUNT_FRAC = 0.05                  # the LE test: the thickness this fraction of the chord in from each end
TOUCH = 0.05                       # mm: faces whose edges come this close share a blade
TABLE_HEAD = "x/c,camber (% c),camber slope,half thickness / max thickness"
HUB_HEIGHT_FACTOR = 2.02           # a cylinder hub: this times the root section's axial extent, to 0.1 m


# --------------------------------------------------------------------------
# OCC: reading, faces, cuts
# --------------------------------------------------------------------------

def read_cad(path):
    """The shape of a STEP or IGES file."""
    if path.lower().endswith((".stp", ".step")):
        return XL.read_step(path)
    return XL.read_iges(path)


def _faces(shape):
    import vehicle_cad as VC
    return VC._shapes(shape, "FACE")


def _edges(shape):
    import vehicle_cad as VC
    return VC._shapes(shape, "EDGE")


def _polyline(edge, n):
    BRepAdaptor_Curve = XL._occ("BRepAdaptor", "BRepAdaptor_Curve")
    GCPnts = XL._occ("GCPnts", "GCPnts_QuasiUniformAbscissa")
    c = BRepAdaptor_Curve(edge)
    try:
        d = GCPnts(c, n)
        ts = [d.Parameter(i) for i in range(1, d.NbPoints() + 1)]
    except Exception:
        ts = np.linspace(c.FirstParameter(), c.LastParameter(), n)
    return np.array([[c.Value(float(t)).X(), c.Value(float(t)).Y(), c.Value(float(t)).Z()] for t in ts])


def face_points(face, n=15):
    """Points on a face: its edges sampled and its surface on an n x n grid of its parameter range."""
    BRepAdaptor_Surface = XL._occ("BRepAdaptor", "BRepAdaptor_Surface")
    a = BRepAdaptor_Surface(face)
    uu = np.linspace(a.FirstUParameter(), a.LastUParameter(), n)
    vv = np.linspace(a.FirstVParameter(), a.LastVParameter(), n)
    grid = np.array([[a.Value(float(u), float(v)).X(), a.Value(float(u), float(v)).Y(),
                      a.Value(float(u), float(v)).Z()] for u in uu for v in vv])
    return np.vstack([grid] + [_polyline(e, 30) for e in _edges(face)])


def mesh_points(shape, deflection=1.0):
    """The vertices of a triangulation of the shape (mm): its surface, trimmed as it is."""
    BRepMesh = XL._occ("BRepMesh", "BRepMesh_IncrementalMesh")
    BRep_Tool = XL._occ("BRep", "BRep_Tool")
    TopLoc_Location = XL._occ("TopLoc", "TopLoc_Location")
    BRepMesh(shape, float(deflection), False, 0.5, True)
    out = []
    for f in _faces(shape):
        loc = TopLoc_Location()
        tri = XL._static(BRep_Tool, "Triangulation")(f, loc)
        if tri is None:
            continue
        try:
            pts = [tri.Node(i) for i in range(1, tri.NbNodes() + 1)]
        except Exception:                                       # an older binding
            nodes = tri.Nodes()
            pts = [nodes.Value(i) for i in range(1, nodes.Length() + 1)]
        t = loc.Transformation()
        out.append(np.array([[q.Transformed(t).X(), q.Transformed(t).Y(), q.Transformed(t).Z()] for q in pts]))
    if not out:
        raise RuntimeError("the shape has no faces to mesh")
    return np.vstack(out)


def turned(shape, m):
    """The shape under the rotation (or reflection) matrix m, about the origin."""
    import vehicle_cad as VC
    return VC.transformed(shape, np.asarray(m, dtype=float), np.zeros(3))


def principal_axes(pts):
    """The unit vectors of the principal axes of a point cloud, the one standing apart from the other two
    first (a propeller of 3 blades or more: its axis; the two across it are alike)."""
    q = pts - pts.mean(axis=0)
    w, v = np.linalg.eigh(q.T @ q)
    k = 0 if w[1] - w[0] > w[2] - w[1] else 2                  # the odd one out: the lowest or the highest
    order = [k] + [i for i in range(3) if i != k]
    return [v[:, i] / np.linalg.norm(v[:, i]) for i in order]


def curved_edges(edges, size):
    """The edges that are curves: longer than 1e-3 of the shape's size and bowed away from their chord by more
    than 1e-3 of their length (a point or a straight line is round about any axis)."""
    out = []
    for e in edges:
        chord = e[-1] - e[0]
        length = float(np.linalg.norm(chord))
        if length < 1e-3 * size or len(e) < 3:
            continue
        t = (e - e[0]) @ chord / length ** 2
        bow = np.linalg.norm(e - e[0] - np.outer(t, chord), axis=1).max()
        if bow > 1e-3 * length:
            out.append(e)
    return out


def roundness(edges, axis):
    """How round the roundest edge is about the axis: the smallest, over the edges, of the spread of the
    points' distance from the axis over its mean (0 for a circle about it)."""
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    best = np.inf
    for e in edges:
        q = e - np.outer(e @ a, a)
        r = np.linalg.norm(q, axis=1)
        if r.mean() > 1e-9:
            best = min(best, float(np.ptp(r) / r.mean()))
    return best


def find_axis(shape, pts):
    """The propeller's axis: of the candidates (x, y, z and the principal axes of the surface points) the one
    about which an edge of the faces is roundest (a root edge, a tip edge). Returns (axis, candidates' scores
    by name)."""
    edges = curved_edges([_polyline(e, 60) for e in _edges(shape)], float(np.ptp(pts)))
    if not edges:
        raise RuntimeError("no curved edge to find the axis by: name it with --axis")
    cands = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
    for k, v in enumerate(principal_axes(pts)):
        cands[f"principal {k + 1}"] = v
    scores = {name: roundness(edges, v) for name, v in cands.items()}
    best = min(scores, key=scores.get)
    axis = cands[best]
    for name in ("x", "y", "z"):                                # a coordinate axis when it is as good
        if scores[name] <= 10.0 * scores[best] and abs(float(cands[name] @ axis)) > 0.999:
            axis, best = cands[name], name
    return axis, best, scores


def frame_to_x(axis):
    """A rotation matrix (det 1) that turns the unit vector axis onto +x."""
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    x = np.array([1.0, 0.0, 0.0])
    v, c = np.cross(a, x), float(a @ x)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def is_revolved(pts, tol=1.0e-3):
    """Are the points on a surface of revolution about x (one radius per x, within tol of the spread): a
    cylinder, a cone, a sphere or a plane normal to x (a hub's end)?"""
    x, r = pts[:, 0], np.hypot(pts[:, 1], pts[:, 2])
    size = max(1.0, float(np.ptp(pts)))
    if np.ptp(x) < tol * size or np.ptp(r) < tol * size:                # a plane normal to x, or a cylinder
        return True
    order = np.argsort(x)
    xs, rs = x[order], r[order]
    bins = np.linspace(xs[0], xs[-1], 12)
    spread = []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (xs >= a) & (xs <= b)
        if sel.sum() >= 3:
            spread.append(np.ptp(rs[sel]))
    return bool(spread) and max(spread) < 0.02 * max(np.ptp(r), np.ptp(x))


def blades_of(shape):
    """The blades of a propeller shape (its axis along x): (blades, hub_faces, points), the blades lists of
    faces that touch along their edges (faces of revolution about x are the hub's)."""
    from scipy.spatial import cKDTree
    faces = _faces(shape)
    if not faces:
        raise RuntimeError("the file has no faces")
    edges = [np.vstack([_polyline(e, 40) for e in _edges(f)]) for f in faces]
    samples = [face_points(f) for f in faces]
    hub = [i for i, p in enumerate(samples) if is_revolved(p)]
    rest = [i for i in range(len(faces)) if i not in hub]
    n = len(rest)
    adj = np.zeros((n, n), dtype=bool)
    trees = [cKDTree(edges[i]) for i in rest]
    for a in range(n):
        for b in range(a + 1, n):
            d, _ = trees[a].query(edges[rest[b]], k=1)
            if d.min() < TOUCH:
                adj[a, b] = adj[b, a] = True
    comp, groups = -np.ones(n, dtype=int), []
    for a in range(n):
        if comp[a] >= 0:
            continue
        stack, members = [a], []
        comp[a] = len(groups)
        while stack:
            k = stack.pop()
            members.append(rest[k])
            for j in np.nonzero(adj[k])[0]:
                if comp[j] < 0:
                    comp[j] = len(groups)
                    stack.append(j)
        groups.append(members)
    reach = [max(np.hypot(samples[i][:, 1], samples[i][:, 2]).max() for i in g) for g in groups]   # to the tip
    keep = [g for g, r in zip(groups, reach) if r >= 0.5 * max(reach)]       # the blades; not a hub's odd face
    small = [i for g, r in zip(groups, reach) if r < 0.5 * max(reach) for i in g]
    return [[faces[i] for i in g] for g in keep], [faces[i] for i in hub + small], samples


def compound(faces):
    BRep_Builder = XL._occ("BRep", "BRep_Builder")
    TopoDS_Compound = XL._occ("TopoDS", "TopoDS_Compound")
    c = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(c)
    for f in faces:
        b.Add(c, f)
    return c


def _chain(strips, rel=1.0e-5):
    """The polylines joined end to end into one closed loop (the closing point not repeated)."""
    tol = max(1.0e-9, rel * float(np.ptp(np.vstack(strips), axis=0).max()))
    rest = list(strips)
    loop = [rest.pop(0)]
    tail = loop[-1][-1]
    while rest:
        best = None
        for i, s in enumerate(rest):
            for flip, end in ((False, s[0]), (True, s[-1])):
                d = float(np.linalg.norm(end - tail))
                if best is None or d < best[2]:
                    best = (i, flip, d)
        i, flip, d = best
        if d > tol:
            raise RuntimeError(f"open cut (gap {d:.3e} mm)")
        s = rest.pop(i)
        s = s[::-1] if flip else s
        loop.append(s[1:])
        tail = s[-1]
    pts = np.vstack(loop)
    if np.linalg.norm(pts[0] - pts[-1]) > tol:
        raise RuntimeError("the cut does not close")
    return pts[:-1]


def cylinder_cut(shape, r, x_lo, x_hi, n=1000):
    """The closed loop where the cylinder of radius r (mm, axis x) cuts the shape: (m, 3), mm."""
    gp_Cylinder, gp_Ax3, gp_Pnt, gp_Dir = XL._occ("gp", "gp_Cylinder", "gp_Ax3", "gp_Pnt", "gp_Dir")
    MakeFace = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace")
    Section = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Section")
    cyl = gp_Cylinder(gp_Ax3(gp_Pnt(float(x_lo), 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), float(r))
    tool = MakeFace(cyl, 0.0, 2.0 * np.pi, 0.0, float(x_hi - x_lo)).Face()
    algo = Section(shape, tool, False)
    algo.ComputePCurveOn1(True)
    algo.Approximation(True)
    algo.Build()
    edges = _edges(algo.Shape())
    if not edges:
        raise RuntimeError(f"no cut at r = {r:.4f} mm")
    return _chain([_polyline(e, n) for e in edges])


# --------------------------------------------------------------------------
# One cut
# --------------------------------------------------------------------------

def unrolled(pts, r):
    """(s, x) of points near the cylinder of radius r (mm): s = r theta, theta = atan2(z, y) unwrapped."""
    return np.column_stack((r * np.unwrap(np.arctan2(pts[:, 2], pts[:, 1])), pts[:, 0]))


def _far_pair(p):
    from scipy.spatial import ConvexHull
    h = ConvexHull(p).vertices
    q = p[h]
    d = np.hypot(*(np.subtract.outer(q[:, k], q[:, k]) for k in range(2)))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return int(h[i]), int(h[j])


def blunt_end(p, i, j, frac=BLUNT_FRAC):
    """Of the two ends i, j of the unrolled loop p, the blunter one (the thickness frac of the chord in from
    it is the larger): the leading edge."""
    A, B = p[i], p[j]
    c = float(np.linalg.norm(B - A))
    ec = (B - A) / c
    en = np.array([-ec[1], ec[0]])
    u, v = (p - A) @ ec, (p - A) @ en
    band = max(1e-9, 0.005 * c)

    def thickness_at(t):
        sel = np.abs(u - t) < band
        return float(np.ptp(v[sel])) if sel.sum() >= 2 else 0.0
    return i if thickness_at(frac * c) > thickness_at(c - frac * c) else j


def section(loop, r):
    """A cut (mm, the working frame) at radius r (mm) as para.py takes it: dict with the LE and TE on the
    unrolled cylinder, c (c', mm), phi (rad), theta and x of the reference point (rad, mm), the shape (yc,
    slope, th at STATIONS, x/c' units), t_c, f_c (over c'), the coordinates xy (x/c', y/c': the LE at (0, 0),
    the TE at (1.005, 0), from the TE round to the TE) and the leading edge's place (its x and theta against
    the TE's: upstream, turning)."""
    p = unrolled(loop, r)
    i, j = _far_pair(p)
    le = blunt_end(p, i, j)
    te = j if le == i else i
    LE, TE = p[le], p[te]
    c = float(np.linalg.norm(LE - TE))
    phi = float(np.arctan2(LE[1] - TE[1], LE[0] - TE[0]))
    ec = (TE - LE) / c
    en = np.array([-np.sin(phi), np.cos(phi)])               # towards the suction side, upstream
    cp = c / TE_END
    order = (te + np.arange(len(p) + 1)) % len(p)              # from the TE round to the TE
    xy = np.column_stack(((p[order] - LE) @ ec, (p[order] - LE) @ en)) / cp
    xy[0] = xy[-1] = (TE_END, 0.0)
    xy[np.argmin(np.hypot(*xy.T))] = 0.0
    yc, slope, half, t_c, f_c = airfoils.shape(xy)
    ref = LE + 0.5 * cp * ec
    return {"r": r, "c": cp, "phi": phi, "theta": ref[0] / r, "x": ref[1], "yc": yc, "slope": slope,
            "half": half, "t_c": t_c, "f_c": f_c, "xy": xy, "le_upstream": LE[1] > TE[1],
            "le_turning": LE[0] > TE[0]}


def table_rows(sec):
    """para.py's section table of a cut (26 rows: x/c, camber % c, camber slope, half thickness / t over
    cos(atan(slope))), as airfoils.section_table writes one."""
    rows = np.column_stack((STATIONS, 100.0 * sec["yc"], sec["slope"],
                            sec["half"] / sec["t_c"] / np.cos(np.arctan(sec["slope"]))))
    rows[0] = 0.0
    return rows


def table_text(rows, note=""):
    return TABLE_HEAD + (f" ({note})" if note else "") + "\n" + "\n".join(
        ",".join(f"{v:.6f}" for v in row) for row in rows) + "\n"


def write_tables(path, rr, tables, note):
    """<case>_section_tables.dat: '# r/R <value>' and para.py's table (its header line, 26 rows) per block."""
    with open(path, "w") as fh:
        fh.write(f"# {note}\n")
        for r, rows in zip(rr, tables):
            fh.write(f"# r/R {r:.6f}\n" + table_text(rows))
    return path


def taper(rr):
    """BladeSurface's tip chord factor sqrt(1 - t^3), t = (r/R - TIP_TAPER_START) / (1 - that) (1 below it;
    1 everywhere when the taper is off)."""
    from blade_surface_new import TIP_TAPER_START as a
    rr = np.asarray(rr, dtype=float)
    if a is None:
        return np.ones(rr.shape)
    t = np.clip((rr - a) / (1.0 - a), 0.0, 1.0)
    return np.sqrt(1.0 - t ** 3)


def end_cut(blade, r_edge, step, x_lo, x_hi):
    """(r, loop) of the first closed cut from the faces' edge at r_edge inwards by steps of step (a cut on
    the edge, or on a sliver where the surface undershoots it, is open)."""
    last = None
    for k in range(1, END_TRIES + 1):
        r = r_edge + k * step
        try:
            return r, cylinder_cut(blade, r, x_lo, x_hi)
        except RuntimeError as exc:
            last = exc
    raise RuntimeError(f"no closed cut within {END_TRIES * abs(step):.3f} mm of r = {r_edge:.3f} mm: {last}")


def stations(r_root, r_tip, n):
    """n cut radii from the root to the tip, cosine-clustered at both ends (the root and tip cuts included)."""
    u = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n))))
    return r_root + (r_tip - r_root) * u


# --------------------------------------------------------------------------
# The blade frame (para.py's) and the checks
# --------------------------------------------------------------------------

def blade_to_working(pts_m, x0, theta_ref):
    """Blade-frame points (m: X downstream, the blade along +Z at zero skew) in the working frame (mm: x
    upstream, theta = atan2(z, y)): x = x0 - X, theta = theta_ref + atan2(Y, Z)."""
    p = 1000.0 * np.asarray(pts_m, dtype=float)
    s, c = np.sin(theta_ref), np.cos(theta_ref)
    return np.stack((x0 - p[..., 0], -s * p[..., 1] + c * p[..., 2], c * p[..., 1] + s * p[..., 2]), axis=-1)


def distance_to_curve(p, curve):
    """Distance of each point p to the closed polyline curve (both (n, 2))."""
    from scipy.spatial import cKDTree
    a, b = curve, np.roll(curve, -1, axis=0)
    _, k = cKDTree(curve).query(p, k=1)
    best = np.full(len(p), np.inf)
    for kk in (k, (k - 1) % len(curve)):
        ab = b[kk] - a[kk]
        t = np.clip(np.einsum("ij,ij->i", p - a[kk], ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0, 1)
        best = np.minimum(best, np.hypot(*(p - a[kk] - t[:, None] * ab).T))
    return best


def cut_deviation(loop, r, blade, R_mm, x0, theta_ref, n=4001):
    """(max, rms) distance (mm) of a cut from the BladeSurface blade at the same radius, on the cylinder."""
    eta = float(blade.eta_of_r(min(r / R_mm, blade.r_tip)))
    xi = np.linspace(0.0, 1.0, n)[:-1]
    gen = blade_to_working(blade.b(xi, np.full(xi.shape, eta)), x0, theta_ref)
    p, q = unrolled(loop, r), unrolled(gen, r)
    q[:, 0] += r * 2.0 * np.pi * np.round((p[:, 0].mean() - q[:, 0].mean()) / (r * 2.0 * np.pi))
    d = distance_to_curve(p, q)
    return float(d.max()), float(np.sqrt(np.mean(d ** 2)))


# --------------------------------------------------------------------------
# The extraction
# --------------------------------------------------------------------------

def extract(path, case="blade", out_dir=None, stations_n=40, diameter_mm=None, x0=None, hub=None,
            hub_height_mm=None, tip_pitch=None, ref_rr=0.7, axis="auto", blade_index=0, log=print):
    """Run the extraction (the module docstring): returns a dict with the component file's path, the
    distributions table, the frame (the rotation applied to the file, the mirror, x0, theta_ref), the cuts
    and the check of the rebuilt blade against them."""
    t0 = time.time()
    out_dir = out_dir or os.path.join(HERE, "outputs", "blade")
    os.makedirs(out_dir, exist_ok=True)
    shape = read_cad(path)
    faces = _faces(shape)
    log(f"{os.path.basename(path)}: {len(faces)} faces ({time.time() - t0:.1f} s)")

    # the axis, and the shape with it along x
    pts = mesh_points(shape)
    if axis == "auto":
        ax, name, scores = find_axis(shape, pts)
        ranked = sorted(scores.values())
        log(f"axis: {name} ({ax[0]:.4f}, {ax[1]:.4f}, {ax[2]:.4f}); the roundest edge about it {ranked[0]:.2e} of "
            f"its radius, about the next candidate {ranked[1]:.2e}"
            + (" (not clearly apart: name the axis with --axis)" if ranked[1] < 10.0 * ranked[0] else ""))
    else:
        ax = np.array({"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis], dtype=float)
    m = frame_to_x(ax)
    shape = turned(shape, m)
    blades, hub_faces, _ = blades_of(shape)
    sizes = sorted(len(b) for b in blades)
    if not blades:
        raise RuntimeError("no blade faces (every face is a surface of revolution about the axis)")
    log(f"{len(blades)} blades of {sizes[0] if sizes[0] == sizes[-1] else f'{sizes[0]} to {sizes[-1]}'} faces each"
        + (f"; {len(hub_faces)} hub faces (left out)" if hub_faces else "; no hub in the file"))
    if sizes[0] != sizes[-1]:
        log("  (the blades differ in their face counts: faces that touch within "
            f"{TOUCH} mm share a blade; the first blade is cut)")
    blade = compound(blades[int(blade_index) % len(blades)])
    bp = mesh_points(blade)
    ep = np.vstack([_polyline(e, 200) for e in _edges(blade)])         # the faces' edges: the root ring exactly
    r_all, r_edges = np.hypot(bp[:, 1], bp[:, 2]), np.hypot(ep[:, 1], ep[:, 2])
    r_lo, r_hi = float(r_edges.min()), float(max(r_all.max(), r_edges.max()))
    x_lo, x_hi = float(bp[:, 0].min()) - 0.5 * (r_hi - r_lo), float(bp[:, 0].max()) + 0.5 * (r_hi - r_lo)
    R = 0.5 * float(diameter_mm) if diameter_mm else r_hi
    log(f"the blade from r {r_lo:.3f} to {r_hi:.3f} mm: R {R:.3f} mm" + (" (--diameter)" if diameter_mm else ""))

    # the first cut: which end is upstream, which way the propeller turns
    r_root, loop = end_cut(blade, r_lo, ROOT_IN * R, x_lo, x_hi)
    sec = section(loop, r_root)
    flipped = mirrored = False
    if not sec["le_upstream"]:                                   # the LE at the smaller x: turn the axis round
        flip = np.diag([-1.0, -1.0, 1.0])
        shape, blade, m = turned(shape, flip), turned(blade, flip), flip @ m
        x_lo, x_hi, flipped = -x_hi, -x_lo, True
        loop = cylinder_cut(blade, r_root, x_lo, x_hi)
        sec = section(loop, r_root)
        log("the leading edge lay at the smaller x: the axis turned round (upstream is +x now)")
    if not sec["le_turning"]:                                    # the LE at the smaller theta: left-handed
        mir = np.diag([1.0, -1.0, 1.0])
        shape, blade, m = turned(shape, mir), turned(blade, mir), mir @ m
        mirrored = True
        loop = cylinder_cut(blade, r_root, x_lo, x_hi)
        sec = section(loop, r_root)
        log("a left-handed propeller in para.py's convention: mirrored (y to -y); the file describes its mirror "
            "image")
    if not (sec["le_upstream"] and sec["le_turning"]):
        raise RuntimeError("the root section's leading edge could not be placed upstream and turning")

    # the cuts
    r_tip, _ = end_cut(blade, r_hi, -TIP_IN * R, x_lo, x_hi)
    rr_cut = stations(r_root, r_tip, stations_n)
    secs, loops = [], []
    for r in rr_cut:
        loop = cylinder_cut(blade, r, x_lo, x_hi)
        secs.append(section(loop, r))
        loops.append(loop)
    theta_ref = secs[0]["theta"]
    x_ref0 = secs[0]["x"]
    x0 = float(x_ref0 if x0 is None else (-x0 if flipped else x0))
    log(f"{len(secs)} cuts, r/R {rr_cut[0] / R:.4f} to {rr_cut[-1] / R:.4f}; the root's reference point at "
        f"theta {np.degrees(theta_ref):.3f} deg, x {x_ref0:.3f} mm; the generator line x0 {x0:.3f} mm"
        + (" (the root's reference point)" if x_ref0 == x0 else "") + f" ({time.time() - t0:.1f} s)")

    # the distributions; the rows at the root (where the faces start) and at r/R 1 continued from the three
    # cuts nearest each (a quadratic per column); the root row's skew is 0 (para.py's convention), so the
    # root's reference angle theta_ref is the row's
    r_mm = np.array([s["r"] for s in secs])
    rr = r_mm / R
    phi = np.array([s["phi"] for s in secs])
    skew = theta_ref - np.array([s["theta"] for s in secs])                                   # rad
    rake = x0 - np.array([s["x"] for s in secs]) + r_mm * skew * np.tan(phi)                   # mm
    D = 2.0 * R
    cols = {"P/D": 2.0 * np.pi * r_mm * np.tan(phi) / D,
            "c/D": np.array([s["c"] for s in secs]) / D / taper(rr),
            "t/c": np.array([s["t_c"] for s in secs]),
            "f/c": np.array([s["f_c"] for s in secs]),
            "skew_deg": np.degrees(skew),
            "rake/D": rake / D}
    root_r = float(r_lo / R)
    root = {k: float(np.polyval(np.polyfit(rr[:3], v[:3], 2), root_r)) for k, v in cols.items()}
    tip = {k: float(np.polyval(np.polyfit(rr[-3:], v[-3:], 2), 1.0)) for k, v in cols.items()}
    if tip_pitch is not None:
        tip["P/D"] = float(tip_pitch)
    shift = root["skew_deg"]                                       # the root row at skew 0
    cols["skew_deg"], root["skew_deg"], tip["skew_deg"] = cols["skew_deg"] - shift, 0.0, tip["skew_deg"] - shift
    theta_ref = theta_ref - np.radians(shift)
    rr_all = np.r_[root_r, rr, 1.0]
    table = np.column_stack([rr_all] + [np.r_[root[k], cols[k], tip[k]] for k in cols])

    # the section tables
    tables = [table_rows(s) for s in secs]
    k_ref = int(np.argmin(np.abs(rr - ref_rr)))
    sec_path = os.path.join(out_dir, f"{case}_section.dat")
    with open(sec_path, "w") as fh:
        fh.write(table_text(tables[k_ref], f"{case} at r/R {rr[k_ref]:.4f}, from {os.path.basename(path)}"))
    tables = [tables[0]] + tables + [tables[-1]]                   # the root and tip rows: the nearest cuts'
    rr_tab = rr_all
    tables_path = write_tables(os.path.join(out_dir, f"{case}_section_tables.dat"), rr_tab, tables,
                               f"{case}: para.py's section table of each cut of {os.path.basename(path)} (x/c over "
                               f"c' = nose-tail length / {TE_END})")
    with open(os.path.join(out_dir, f"{case}_cuts.dat"), "w") as fh:
        fh.write(f"# {os.path.basename(path)}, one blade cut by cylinders about its axis: x y z mm in the file's "
                 f"frame, a block per r/R\n")
        mi = np.linalg.inv(m)
        for r, loop in zip(rr, loops):
            fh.write(f"# r/R {r:.6f}  r_mm {r * R:.3f}  points {len(loop)}\n")
            fh.write("\n".join(f"{q[0]:.5f} {q[1]:.5f} {q[2]:.5f}" for q in loop @ mi.T) + "\n")

    # the hub
    a, b = _root_extent(secs[0], x0)
    if hub is not None:
        base, top, height, x_base = (float(v) for v in hub)
        x_base = -x_base if flipped else x_base
        hub_rec = {"hub_height_m": height / 1000.0, "hub_base_m": base / 1000.0, "hub_top_m": top / 1000.0,
                   "hub_x_m": (x0 - x_base) / 1000.0}
        from prop_hub import HubProfile
        hp = HubProfile(base, top, height, x0 - x_base)
        log(f"hub: {hp.describe('mm')} (the blade root from x {a:.1f} to {b:.1f} mm in the blade frame)")
    else:
        h = hub_height_mm if hub_height_mm else np.ceil(HUB_HEIGHT_FACTOR * max(abs(a), abs(b)) / 100.0) * 100.0
        hub_rec = {"hub_height_m": float(h) / 1000.0}
        log(f"hub: a cylinder of the root radius ({root_r * R:.1f} mm), {h:.0f} mm long, centred on the "
            f"generator line (the blade root from x {a:.1f} to {b:.1f} mm)")

    # the component file
    ax_text = next((n for n, u in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1]))
                    if np.allclose(np.abs(ax), u, atol=1e-6)), "(" + ", ".join(f"{v:.4f}" for v in ax) + ")")
    notes = [f"{case}: the six distributions of {os.path.basename(path)} by blade_extraction.py, "
             f"{time.strftime('%d %b %Y')}; {len(secs)} cuts from r/R {rr[0]:.4f} to {rr[-1]:.4f}, the rows at "
             f"r/R {root_r:.4f} (the root, where the faces start) and 1 continued from the nearest three"
             + (f" (P/D {tip['P/D']:.4f} at the tip given)" if tip_pitch is not None else ""),
             f"the file's axis {ax_text}" + (", turned round so the leading edge is upstream" if flipped else "")
             + ("; left-handed in para.py's convention: mirrored" if mirrored else ""),
             f"generator line x0 {x0:.4f} mm along the axis (the file's frame{', turned round' if flipped else ''}), "
             f"the root's reference point at theta {np.degrees(theta_ref):.6f} deg; c/D beyond r/R "
             f"{_taper_start()} is the measured chord over BladeSurface's tip taper"]
    vary = {n: BM.Vary(False) for n in BM.VARY_NAMES}
    params = BM.BladeParams(D / 1000.0, len(blades), root_r, os.path.basename(sec_path), rr_all,
                            {c: table[:, i + 1] for i, c in enumerate(BM.CURVES)}, os.path.join(out_dir, f"{case}.txt"),
                            hub_rec["hub_height_m"], None, vary, hub_rec.get("hub_base_m"), hub_rec.get("hub_top_m"),
                            hub_rec.get("hub_x_m"), os.path.basename(tables_path))
    design, _ = BM.fit_design(params)                             # every curve the table's own
    space = BM.BladeSpace(params, design, fixed=BM.BladeSpace(params, design).values(design))
    comp_path = BM.write_component(os.path.join(out_dir, f"{case}.txt"), params, design, space, None, case, notes)
    plot_path = plot_check(os.path.join(out_dir, f"{case}_check.png"), case, table)
    log(f"wrote {comp_path}, the section files and {os.path.basename(plot_path)} ({time.time() - t0:.1f} s)")

    # the check: the blade rebuilt from the component file against the cuts, with the section tables (as the
    # tool builds it) and with the one section table at ref_rr alone
    ld = BM.load_component(comp_path)
    one = dataclasses.replace(ld.params, section_tables=None)
    dev = []
    for bs in (BM.blade_surface(ld.design, ld.params), BM.blade_surface(ld.design, one)):
        dev.append([cut_deviation(loop, s["r"], bs, R, x0, theta_ref) for s, loop in zip(secs, loops)])
    dev = np.array(dev)
    log(" r/R     P/D     c/D     t/c     f/c    skew   rake/D | rebuilt, mm max / rms: the section tables | one "
        "table")
    for k, s in enumerate(secs):
        row = table[k + 1]
        log(f"{rr[k]:.4f} {row[1]:7.4f} {row[2]:7.4f} {row[3]:7.4f} {row[4]:7.4f} {row[5]:6.2f} {row[6]:8.5f} | "
            f"{dev[0, k, 0]:6.2f} / {dev[0, k, 1]:5.2f} | {dev[1, k, 0]:6.2f} / {dev[1, k, 1]:5.2f}")
    log(f"the blade rebuilt with the section tables lies within {dev[0, :, 0].max():.3f} mm of the cuts (rms "
        f"{dev[0, :, 1].max():.3f} at worst); with the one table at r/R {rr[k_ref]:.3f} within "
        f"{dev[1, :, 0].max():.2f} mm; done in {time.time() - t0:.0f} s")
    return {"component": comp_path, "table": table, "columns": ("r/R",) + tuple(cols), "rotation": m,
            "flipped": flipped, "mirrored": mirrored, "x0": x0, "theta_ref": theta_ref, "R_mm": R,
            "blades": len(blades), "cuts": loops, "sections": secs, "deviation": dev, "hub": hub_rec,
            "paths": {"component": comp_path, "section": sec_path, "section_tables": tables_path,
                      "cuts": os.path.join(out_dir, f"{case}_cuts.dat"), "check": plot_path}}


def _taper_start():
    from blade_surface_new import TIP_TAPER_START
    return "none" if TIP_TAPER_START is None else f"{TIP_TAPER_START:g}"


def _root_extent(sec, x0):
    """The root section's axial extent (mm, the blade frame's X): its two ends."""
    half = 0.5 * TE_END * sec["c"] * abs(np.sin(sec["phi"]))
    xm = x0 - sec["x"] + (0.5 * TE_END - 0.5) * sec["c"] * np.sin(sec["phi"])
    return xm - half, xm + half


def plot_check(path, case, table):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(13, 7.5))
    FigureCanvasAgg(fig)
    axs = fig.subplots(2, 3)
    names = ("P/D", f"c/D (beyond r/R {_taper_start()}: over the tip taper)", "t/c", "f/c", "skew (deg)", "rake/D")
    for k, ax in enumerate(axs.flat):
        ax.plot(table[:, 0], table[:, k + 1], "o-", color="#2a78d6", ms=3, lw=1.2)
        if k == 1:
            ax.plot(table[:, 0], table[:, 2] * taper(table[:, 0]), "-", color="#9a9994", lw=1, label="c'/D measured")
            ax.legend(fontsize=8)
        ax.set_xlabel("r/R")
        ax.set_ylabel(names[k])
        ax.grid(True, color="#e4e3df")
    fig.suptitle(f"{case}: the distributions backed out of the CAD (the cuts and the row at r/R 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="The six distributions of a propeller blade from its CAD, as an XGeom "
                                             "blade component file")
    ap.add_argument("cad", help="the propeller's STEP or IGES file (mm)")
    ap.add_argument("--case", default=None, help="the output files' name (default: the CAD file's)")
    ap.add_argument("--out", default=None, help="the output folder (default outputs/blade)")
    ap.add_argument("--stations", type=int, default=40, help="cuts from the root to the tip (default 40)")
    ap.add_argument("--diameter", type=float, default=None, help="D in mm (default: twice the largest radius)")
    ap.add_argument("--x0", type=float, default=None,
                    help="the generator line, mm along the axis in the file's frame (default: the root's reference "
                         "point, so the root's rake is 0)")
    ap.add_argument("--hub", type=float, nargs=4, metavar=("BASE", "TOP", "HEIGHT", "X"), default=None,
                    help="a conic hub: base and top radii, base to tip, and the base plane's x in the file's "
                         "frame (mm)")
    ap.add_argument("--hub-height", type=float, default=None, help="a cylinder hub's length, mm")
    ap.add_argument("--tip-pitch", type=float, default=None, help="P/D at r/R 1 (default: continued from the cuts)")
    ap.add_argument("--ref-rr", type=float, default=0.7, help="r/R of the component's section table (default 0.7)")
    ap.add_argument("--axis", choices=("auto", "x", "y", "z"), default="auto", help="the propeller's axis")
    ap.add_argument("--blade", type=int, default=0, help="which blade to cut (default the first)")
    args = ap.parse_args(argv)
    case = args.case or os.path.splitext(os.path.basename(args.cad))[0]
    extract(args.cad, case, args.out, args.stations, args.diameter, args.x0, args.hub, args.hub_height,
            args.tip_pitch, args.ref_rr, args.axis, args.blade)


if __name__ == "__main__":
    main()
