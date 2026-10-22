"""
bb2_rudder.py

The wind-tunnel rudder (Wind_Tunnel_Rudder.stp, the XGeom project's original rudder) on BB2, placed by XGeom's
rules, compared with BB2's own stern plane (the one in hull.iges, at 45 deg of drawing F3's X stern). The
vehicles carry BB2's own stern planes (bb2_sternplanes.py): the rudder's planform differs from BB2's (its tip cut
at 94.5 deg to its TE against BB2's 90.6, its LE converging on the TE at 12.1 deg against 11.6), so no placement
lines both up.

  scale      the rudder's planform fitted to BB2's stern plane: a similarity in the stern plane's plane (scale k,
             a turn about the thickness axis, a shift) taking its straight LE and TE onto BB2's lines and the
             tops of its LE and TE onto BB2's (fit_similarity). The free fit gives 1:20.006, so SCALE = 20.
  placement  the TE line square to the axis, as BB2's: the XGeom tool's rudder with a set TE rake
             (xgeom_vehicle.place_rudder, rake RAKE on the hull of bb2_build.py), its TE line meeting the hull where
             BB2's TE meets it;
             the rudder slid along it until its root rests on the hull (its closest point on it, here the root's
             LE), the root section extruded along the TE line into the hull by the largest gap under it plus 2 %
             of the root chord, the rudder ROOT_LIFT_MM (vehicle_cad's 0.2 mm) off the hull along the extrusion.
  compare    BB2's stern plane against the rudder so placed, the rudder pitched to the hull (the XGeom tool's
             other rule: the root's LE and TE on the hull, the TE at the same x) and the best fit of the planform
             (the first placement: the fitted similarity, its root taken 510 mm into the hull at the TE). Each
             above the hull (cut by the hull with the hub): where the LE and TE meet the hull, the tops of the
             straight edges, the lean of the edges, the root chord's angle to the axis, the volume, the area in
             the stern plane's plane; cuts at s = const; the distances of SAMPLES points spread evenly over BB2's
             surface from each rudder's surface and back.

Frames. The rudder file's (mm): X from the TE to the LE, Y the span from the flat root, Z the thickness (the
XGeom tool's rudder frame, x from the LE to the TE, y the height, z the thickness, is (-X, Y, -Z)). BB2's (the
IGES frame, mm, x towards the bow); the stern plane's plane through the axis at 45 deg: x along the axis,
s = (y + z) / sqrt(2) the distance from the axis in that plane, the thickness along (0, -1, 1) / sqrt(2). The
hull tool's frame (x aft from its nose, m) is turned to the IGES frame as in bb2_build.py.

Edges are measured from cuts: BB2's stern plane by planes s = const, the rudder by planes Y = const; the LE is the
cut's largest x (X), the TE its smallest; each straight edge is the least-squares line over its straight part;
its top is where the cuts leave that line by more than EDGE_TOL of the chord.

Writes into the output folder (bb2_build.py's, with bb2_hull_params.dat and bb2_hull_hub.step): bb2_rudder.step/
.iges (the rudder with its TE square at 45 deg, with its root extension), bb2_rudder_compare.dat (one row per shape:
BB2's stern plane, then the rudders of CASES), bb2_rudder_sections.dat (the cuts) and bb2_rudder_compare.png (the
planforms in the stern plane's plane with x across, the distance map, cuts).

    python bb2_rudder.py hull.iges Wind_Tunnel_Rudder.stp [out_dir]        (about 11 min)
"""

import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # the X_geom folder
sys.path.insert(0, HERE)

import bb2_build as B                                      # noqa: E402
import bb2_sections as S                                   # noqa: E402
import hull_modify as HM                                   # noqa: E402
import vehicle_cad as VC                                   # noqa: E402
import xcad_loft as XL                                     # noqa: E402
import xgeom_vehicle as XV                                 # noqa: E402

X_HAT = np.array([1.0, 0.0, 0.0])
E_SPAN = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)         # the stern plane's span direction (45 deg)
E_THICK = np.cross(X_HAT, E_SPAN)                          # (0, -1, 1) / sqrt(2)
PHI_TOOL = -0.25 * np.pi                                   # that direction in the hull tool's frame (from +z to +y)
EDGE_TOL = 2.0e-4                                          # an edge's top: the cuts leave its line by this x chord
ROOT_SINK_MM = 50.0                                        # the best fit's extruded root this far inside the hull
BB2_SPAN = (3200.0, 6700.0)                                # s (mm) of BB2's straight edges used in their lines (the
                                                           # hull meets the LE near s 3050 and the TE near s 2020)
WT_SPAN = (5.0, 195.0)                                     # Y (mm) of the rudder's straight edges used in its lines
CHECK_SPANS = np.arange(3300.0, 6701.0, 400.0)             # s (mm) of the cuts compared
SCALE = 20.0                                               # the rudder's scale on BB2: the free fit gives 1:20.006,
                                                           # so the model is 1:20 (None: the fit's own)
SAMPLES = 1000                                             # points of each surface in the distance comparison
RAKE = 0.0                                                 # the TE line square to the axis, as BB2's (deg)
CASES = ("BB2's stern plane", "rudder, TE square to the axis", "rudder, pitched to the hull", "rudder, best fit")


# --------------------------------------------------------------------------
# Edges from cuts
# --------------------------------------------------------------------------

def plane_cut(shape, origin, normal, n=400):
    """Points of the cut of a shape by a plane (None when the plane misses it)."""
    gp_Pln, gp_Pnt, gp_Dir = XL._occ("gp", "gp_Pln", "gp_Pnt", "gp_Dir")
    Section = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Section")
    algo = Section(shape, gp_Pln(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, normal))))
    algo.Build()
    edges = S._explore(algo.Shape(), "EDGE")
    return np.vstack([S._polyline(e, n) for e in edges]) if edges else None


def bb2_cut(shape, s):
    """A shape (the STP frame) cut at span s of the stern plane's plane: (x, thickness) points."""
    p = plane_cut(shape, s * E_SPAN, E_SPAN)
    return None if p is None else np.column_stack((p @ X_HAT, p @ E_THICK))


def wt_cut(shape, y):
    """The rudder (its file frame) cut at span Y: (X, Z) points."""
    p = plane_cut(shape, (0.0, y, 0.0), (0.0, 1.0, 0.0))
    return None if p is None else p[:, [0, 2]]


def edge_line(cut, lo, hi, n=10):
    """Least-squares lines of the LE (largest first coordinate) and the TE (smallest) over spans lo .. hi:
    ((a, b) with edge = a + b span) for each."""
    v = np.linspace(lo, hi, n)
    le, te = [], []
    for s in v:
        q = cut(s)
        le.append(q[:, 0].max())
        te.append(q[:, 0].min())
    return np.polyfit(v, le, 1)[::-1], np.polyfit(v, te, 1)[::-1]


def edge_top(cut, line, which, v_on, v_off, iters=12):
    """The span where the cuts' LE (which 'le') or TE ('te') leaves its line by more than EDGE_TOL of the chord:
    bisection between a span on the line (v_on) and one off it (v_off)."""
    def off(v):
        q = cut(v)
        if q is None:
            return True
        e = q[:, 0].max() if which == "le" else q[:, 0].min()
        return abs(e - (line[0] + line[1] * v)) > EDGE_TOL * np.ptp(q[:, 0])
    for _ in range(iters):
        mid = 0.5 * (v_on + v_off)
        if off(mid):
            v_off = mid
        else:
            v_on = mid
    return v_on


def edges(cut, span, v_top):
    """{'le', 'te': lines, 'le_top', 'te_top': (x, span)} of a shape's cuts (cut(v) -> points)."""
    le, te = edge_line(cut, *span)
    out = {"le": le, "te": te}
    for k, line in (("le", le), ("te", te)):
        v = edge_top(cut, line, k, span[1], v_top)
        out[k + "_top"] = (line[0] + line[1] * v, v)
    return out


# --------------------------------------------------------------------------
# The best fit (the scale)
# --------------------------------------------------------------------------

def rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def to_plane(p, X, Y):
    """(x, s) in BB2's stern-plane coordinates of rudder points (X, Y): p = (k, angle, x0, s0)."""
    k, a, x0, s0 = p
    q = k * (rot2(a) @ np.vstack((X, Y)))
    return q[0] + x0, q[1] + s0


def line_distance(x, s, line):
    """Signed distance (mm) from points (x, s) to the line x = a + b s."""
    a, b = line
    return (x - a - b * s) / np.hypot(1.0, b)


def fit_similarity(wt, bb2, n=21, scale=None):
    """(k, angle, x0, s0) taking the rudder's edges onto BB2's (see the module docstring); scale: k held. wt and
    bb2: edges() of each."""
    y_le = np.linspace(0.0, wt["le_top"][1], n)
    y_te = np.linspace(0.0, wt["te_top"][1], n)
    x_le = wt["le"][0] + wt["le"][1] * y_le
    x_te = wt["te"][0] + wt["te"][1] * y_te

    def res(q):
        p = q if scale is None else np.r_[scale, q]
        xl, sl = to_plane(p, x_le, y_le)
        xt, st = to_plane(p, x_te, y_te)
        tops = np.concatenate([np.ravel(np.subtract(to_plane(p, *wt[k]), np.reshape(bb2[k], (2, 1))))
                               for k in ("le_top", "te_top")])
        return np.concatenate((line_distance(xl, sl, bb2["le"]), line_distance(xt, st, bb2["te"]), tops))

    k0 = np.hypot(*np.subtract(bb2["le_top"], bb2["te_top"])) / np.hypot(*np.subtract(wt["le_top"], wt["te_top"]))
    a0 = np.arctan(wt["te"][1]) - np.arctan(bb2["te"][1])
    x0, s0 = np.subtract(bb2["le_top"], k0 * (rot2(a0) @ np.asarray(wt["le_top"])))
    if scale is None:
        sol = least_squares(res, [k0, a0, x0, s0], x_scale=[1.0, 0.01, 100.0, 100.0])
        p = sol.x
    else:
        sol = least_squares(res, [a0, x0, s0], x_scale=[0.01, 100.0, 100.0])
        p = np.r_[scale, sol.x]
    r = res(sol.x)
    return p, {"le_rms": float(np.sqrt(np.mean(r[:n] ** 2))), "te_rms": float(np.sqrt(np.mean(r[n:2 * n] ** 2))),
               "le_max": float(np.abs(r[:n]).max()), "te_max": float(np.abs(r[n:2 * n]).max()),
               "le_top": float(np.hypot(*r[2 * n:2 * n + 2])), "te_top": float(np.hypot(*r[2 * n + 2:]))}


def fit_placement(p):
    """(m, t) of x_cad = m X + t for the best fit at 45 deg (m a rotation times k)."""
    k, a, x0, s0 = p
    c, s = np.cos(a), np.sin(a)
    frame = np.column_stack((X_HAT, E_SPAN, E_THICK))
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return k * frame @ turn, x0 * X_HAT + s0 * E_SPAN


def fit_root_depth(solid, m, t, x_hull, r_hull, step=5.0):
    """The best fit's root extrusion (the rudder's mm): down its span until every point of the root lies
    ROOT_SINK_MM inside the hull (x_hull, r_hull: the hull's radius)."""
    root = VC._root_face(solid, axis=1)
    pts = np.vstack([S._polyline(e, 200) for e in S._explore(root, "EDGE")])
    k = np.linalg.norm(m[:, 1])
    down = -m[:, 1] / k
    worst = 0.0
    for p in pts @ m.T + t:
        d = 0.0
        while np.hypot(*(p + d * down)[1:]) > np.interp((p + d * down)[0], x_hull, r_hull) - ROOT_SINK_MM:
            d += step
            if d > 1.0e4:
                raise RuntimeError("the rudder's root does not reach into the hull")
        worst = max(worst, d)
    return worst / k


# --------------------------------------------------------------------------
# The XGeom tool's rule
# --------------------------------------------------------------------------

def rudder_file_solid(path):
    """The rudder's solid, faces of one surface merged (its flat root is two faces in the file)."""
    Unify = XL._occ("ShapeUpgrade", "ShapeUpgrade_UnifySameDomain")
    solid = S._explore(S.read_cad(path), "SOLID")[0]
    u = Unify(solid, True, True, False)
    u.Build()
    return S._explore(u.Shape(), "SOLID")[0]


def line_on_hull(line, radius, s0=2000.0):
    """(x, s) where the edge line x = a + b s meets the hull (radius(x), mm): the fixed point s = radius(x(s))."""
    s = s0
    for _ in range(200):
        s_new = float(np.ravel(radius(line[0] + line[1] * s))[0])
        if abs(s_new - s) < 1.0e-10:
            break
        s = s_new
    return line[0] + line[1] * s, s


def rule_placement(design, x_shift, wt_solid, x_te, te_slope, scale=SCALE, rake=None):
    """The rudder by a rule of the XGeom tool (xgeom_vehicle.place_rudder on the hull tool's design, at
    PHI_TOOL): its TE line at rake (deg) from square to the axis meeting the hull at x_te (mm, the STP frame),
    resting on the hull, or (rake None) pitched to the hull with its root's TE on the hull at x_te. te_slope:
    dX/dY of the file's TE line. Returns (m, t) of x_cad = m X + t for the rudder file's points (the
    ROOT_LIFT_MM along the extrusion included), the root extrusion in the file's mm, its direction (up, the
    file's frame) and place_rudder's info."""
    root = VC._root_face(wt_solid, axis=1)
    pts = np.vstack([S._polyline(e, 200) for e in S._explore(root, "EDGE")])
    root_w = np.column_stack((-pts[:, 0], pts[:, 1], -pts[:, 2]))        # the tool's frame: x LE to TE, y up
    _, _, info = XV.place_rudder(design, [root_w], scale / 1000.0, (x_shift - x_te) / 1000.0, PHI_TOOL, rake=rake,
                                 slope=-te_slope)
    a, ec, eh, et = info["frame"]
    xa, y0 = float(root_w[:, 0].min()), float(root_w[:, 1].mean())
    m_tool = scale * np.column_stack((-ec, eh, -et))                      # the file's (X, Y, Z) = (-x, y, -z)
    t_tool = 1000.0 * a - scale * (xa * ec + y0 * eh) + VC.ROOT_LIFT_MM * info["along"]
    turn = np.diag([-1.0, -1.0, 1.0])                                     # the hull tool's frame to the STP frame
    depth = (1000.0 * info["depth"] + VC.ROOT_LIFT_MM) / scale
    up = np.array([-info["slope"], 1.0, 0.0])                             # the extrusion, up (dX/dY = -dx/dy)
    return turn @ m_tool, turn @ t_tool + np.array([x_shift, 0.0, 0.0]), depth, up, info


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------

def boolean(name, a, b):
    op = XL._occ("BRepAlgoAPI", f"BRepAlgoAPI_{name}")(a, b)
    op.Build()
    if not op.IsDone():
        raise RuntimeError(f"BRepAlgoAPI_{name} failed")
    return op.Shape()


def planform_area(shape):
    """Area (mm^2) of the shape's projection on the stern plane's plane: half the projected triangles' areas (a
    thin solid's surface covers its projection twice)."""
    v, t = XL.tessellate(shape, linear=1.0, angular=0.2)
    q = np.column_stack((v @ X_HAT, v @ E_SPAN))[t]
    a = 0.5 * np.abs((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
                     - (q[:, 2, 0] - q[:, 0, 0]) * (q[:, 1, 1] - q[:, 0, 1]))
    return 0.5 * float(a.sum())


def surface_samples(shape, radius, n=SAMPLES, seed=0):
    """n points spread evenly over a shape's surface (its tessellation: triangles drawn by their area, a point
    drawn evenly in each), those more than 5 mm outside the hull."""
    v, t = XL.tessellate(shape, linear=0.5, angular=0.2)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    k = rng.choice(len(t), size=4 * n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(4 * n))[:, None], rng.random(4 * n)[:, None]
    p = (1.0 - r1) * a[k] + r1 * (1.0 - r2) * b[k] + r1 * r2 * c[k]
    p = p[np.hypot(p[:, 1], p[:, 2]) > np.ravel(radius(p[:, 0])) + 5.0]
    return p[:n]


def distances(points, shape):
    """Distance (mm) of each point from a shape's surface (BRepExtrema_DistShapeShape)."""
    Dist = XL._occ("BRepExtrema", "BRepExtrema_DistShapeShape")
    MakeVertex = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_MakeVertex")
    gp_Pnt = XL._occ("gp", "gp_Pnt")
    shell = S._explore(shape, "SHELL")[0]
    out = []
    for p in points:
        d = Dist(MakeVertex(gp_Pnt(*map(float, p))).Vertex(), shell)
        d.Perform()
        out.append(d.Value())
    return np.array(out)


def footprint_ends(shape, radius, tol=3.0):
    """The foremost and the aftmost (x, s) of a shape's points on the hull (tessellation vertices within tol mm of
    the hull's radius): where the exposed part's LE and TE meet the hull."""
    v, _ = XL.tessellate(shape, linear=0.5, angular=0.2)
    on = np.abs(np.hypot(v[:, 1], v[:, 2]) - np.ravel(radius(v[:, 0]))) < tol
    q = plane_coords(v[on])
    return q[q[:, 0].argmax()], q[q[:, 0].argmin()]


def plane_coords(p):
    """(x, s) of STP-frame points."""
    p = np.atleast_2d(p)
    return np.column_stack((p @ X_HAT, p @ E_SPAN))


def lean(m, line):
    """The lean (deg, positive aft) from the stern plane's radial direction of a rudder edge X = a + b Y placed
    by m, and of BB2's x = a + b s (m None)."""
    if m is None:
        return float(np.degrees(np.arctan(-line[1])))
    d = m @ np.array([line[1], 1.0, 0.0])
    dx, ds = d @ X_HAT, d @ E_SPAN
    return float(np.degrees(np.arctan2(-dx, ds)))


def summary(shape_out, corners, leans, pitch, ref_pts, ref_shape, own_pts):
    """One row of bb2_rudder_compare.dat (see main)."""
    row = []
    for c in corners:                                             # root LE, root TE, tip LE, tip TE: (x, s)
        row += [float(c[0]), float(c[1])]
    row += [leans[0], leans[1], pitch, VC.volume(shape_out), planform_area(shape_out)]
    if ref_shape is None:
        return row + [0.0] * 6, None
    d1 = distances(ref_pts, shape_out)                             # BB2's surface from this rudder's
    d2 = distances(own_pts, ref_shape)                             # this rudder's surface from BB2's
    return row + [d1.mean(), np.sqrt(np.mean(d1 ** 2)), np.percentile(d1, 95), d1.max(), d2.mean(), d2.max()], d1


def plot_compare(path, outs, cuts_rows, cuts, ref_pts, d_rule, x_hull, r_hull, title):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from scipy.spatial import ConvexHull
    colors = ("#52514e", "#d95f02", "#9a9994", "#2a78d6")
    widths = (4.0, 1.8, 1.0, 1.0)
    styles = ("-", "-", "--", ":")
    fig = Figure(figsize=(15, 10))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 6, height_ratios=(2.0, 1.0))
    ax = fig.add_subplot(gs[0, :3])
    xx = np.linspace(4200.0, 9600.0, 300)
    ax.fill_between(xx, 0.0, np.interp(xx, x_hull, r_hull), color="#e4e3df", lw=0)
    ax.plot(xx, np.interp(xx, x_hull, r_hull), "-", color="#9a9994", lw=1.5, label="hull")
    for (name, shape), c, lw, ls in zip(outs, colors, widths, styles):
        v, _ = XL.tessellate(shape, linear=2.0, angular=0.3)
        q = plane_coords(v)
        h = ConvexHull(q)
        loop = q[np.r_[h.vertices, h.vertices[0]]]
        ax.plot(loop[:, 0], loop[:, 1], ls, color=c, lw=lw, label=name, alpha=0.9)
    ax.set_xlim(4200.0, 9600.0)
    ax.set_ylim(1500.0, 7200.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm, the IGES frame: towards the bow)")
    ax.set_ylabel("s (mm from the axis, 45 deg plane)")
    ax.grid(True, color="#e4e3df")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=9)
    ax2 = fig.add_subplot(gs[0, 3:])
    q = plane_coords(ref_pts)
    sc = ax2.scatter(q[:, 0], q[:, 1], c=d_rule, s=9, cmap="viridis")
    fig.colorbar(sc, ax=ax2, label=f"distance from the {outs[1][0]} (mm)")
    ax2.plot(xx, np.interp(xx, x_hull, r_hull), "-", color="#9a9994", lw=1.5)
    ax2.set_xlim(4200.0, 9600.0)
    ax2.set_ylim(1500.0, 7200.0)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("s (mm)")
    ax2.grid(True, color="#e4e3df")
    ax2.set_title(f"BB2's stern plane: each point's distance from the {outs[1][0]}", fontsize=9)
    for i, sv in enumerate((cuts_rows[0, 0], cuts_rows[len(cuts_rows) // 2, 0], cuts_rows[-1, 0])):
        ax3 = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        for (name, _), c, lw, ls, part in zip(outs, colors, (3.5, 1.6, 1.0, 1.0), styles, cuts[sv]):
            if part is not None:
                loop = B_closed(part)
                ax3.plot(np.r_[loop[:, 0], loop[0, 0]], np.r_[loop[:, 1], loop[0, 1]], ls, color=c, lw=lw,
                         label=name)
        ax3.set_aspect("equal")
        ax3.set_title(f"s = {sv:.0f} mm", fontsize=9)
        ax3.set_xlabel("x (mm)")
        ax3.set_ylabel("thickness (mm)")
        ax3.grid(True, color="#e4e3df")
        if i == 0:
            ax3.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    hull_path, wt_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "outputs")
    t0 = time.time()

    # BB2's stern plane, the hull file, the hull of bb2_build.py
    solids = S._explore(S.read_cad(hull_path), "SOLID")
    plane = min(solids, key=lambda s: VC._bbox(s)[1, 0] - VC._bbox(s)[0, 0])
    hull_shell = max(S._explore(S.read_cad(hull_path), "SHELL"), key=lambda s: VC._bbox(s)[1, 0] - VC._bbox(s)[0, 0])
    x_iges, r_iges = B.hull_profile(hull_shell)
    x_fwd = S.hub_forward_x(hull_shell)
    x_pmb, _ = B.middle_body(hull_shell)
    params = HM.read_params(os.path.join(out, "bb2_hull_params.dat"))
    x_shift = x_pmb + 1000.0 * (params.nose + params.middle)
    design = B.fit_hull(params, x_iges, r_iges, x_fwd, x_pmb, x_shift, verbose=False)
    radius = lambda x: design.radius("r", (x_shift - np.atleast_1d(np.asarray(x, dtype=float))) / 1000.0) * 1000.0
    hull_hub = S._explore(S.read_cad(os.path.join(out, "bb2_hull_hub.step")), "SOLID")[0]

    # edges, the scale and the best fit
    bb2 = edges(lambda s: bb2_cut(plane, s), BB2_SPAN, float(VC._bbox(plane)[1, 1:].max() * np.sqrt(2.0)))
    wt_solid = rudder_file_solid(wt_path)
    wt = edges(lambda y: wt_cut(wt_solid, y), WT_SPAN, float(VC._bbox(wt_solid)[1, 1]))
    free, _ = fit_similarity(wt, bb2)
    p, rep = fit_similarity(wt, bb2, scale=SCALE)
    m_fit, t_fit = fit_placement(p)
    depth_fit = fit_root_depth(wt_solid, m_fit, t_fit, x_iges, r_iges)
    print(f"BB2's stern plane: LE x = {bb2['le'][0]:.2f} {bb2['le'][1]:+.6f} s, TE x = {bb2['te'][0]:.2f} "
          f"{bb2['te'][1]:+.6f} s; the free fit 1:{free[0]:.4f}, held 1:{p[0]:g} ({time.time() - t0:.0f} s)")

    # the TE square to the axis (as BB2's) and the XGeom tool's other rule (pitched to the hull), the TE where
    # BB2's TE meets the hull
    te_hull, le_hull = line_on_hull(bb2["te"], radius), line_on_hull(bb2["le"], radius, 3000.0)
    m, t, depth, up, info = rule_placement(design, x_shift, wt_solid, te_hull[0], wt["te"][1], rake=RAKE)
    m_h, t_h, depth_h, up_h, info_h = rule_placement(design, x_shift, wt_solid, te_hull[0], wt["te"][1])
    ends = {}
    for key, inf in (("rake", info), ("hull", info_h)):
        ends[key] = (np.array([x_shift - 1000.0 * inf["x_le"], 1000.0 * inf["rho_le"]]),
                     np.array([x_shift - 1000.0 * inf["x_te"], 1000.0 * inf["rho_te"]]))
    (le_r, te_r), (le_h, te_h) = ends["rake"], ends["hull"]
    print(f"BB2's TE meets the hull at x {te_hull[0]:.2f} (s {te_hull[1]:.2f}), its LE at x {le_hull[0]:.2f} "
          f"(s {le_hull[1]:.2f})")
    print(f"TE square to the axis (rake {RAKE:g}): root chord at {info['pitch']:.3f} deg to the axis, root LE "
          f"({le_r[0]:.2f}, {le_r[1]:.2f}), TE ({te_r[0]:.2f}, {te_r[1]:.2f}), the TE {te_r[1] - te_hull[1]:.1f} mm "
          f"above the hull; root extended {1000 * info['depth']:.1f} mm along the TE line (largest gap "
          f"{1000 * info['gap']:.1f} mm)")
    print(f"pitched to the hull: TE ({te_h[0]:.2f}, {te_h[1]:.2f}) and LE ({le_h[0]:.2f}, {le_h[1]:.2f}) on the "
          f"hull, pitched {info_h['pitch']:.3f} deg (TE rake {info_h['rake']:.2f} deg), the root dips "
          f"{1000 * info_h['dip']:.1f} mm into the hull, extruded {1000 * info_h['depth']:.1f} mm")

    # the rudders: the TE square, pitched to the hull and by the fit
    rudder = VC.transformed(VC.extended_rudder(wt_solid, depth, axis=1, direction=up), m, t)
    rudder_h = VC.transformed(VC.extended_rudder(wt_solid, depth_h, axis=1, direction=up_h), m_h, t_h)
    rudder_fit = VC.transformed(VC.extended_rudder(wt_solid, depth_fit, axis=1), m_fit, t_fit)
    outs = [(CASES[0], boolean("Cut", plane, hull_hub))] + [(name, boolean("Cut", r, hull_hub)) for name, r in
                                                            zip(CASES[1:], (rudder, rudder_h, rudder_fit))]
    placements = ((m, t), (m_h, t_h), (m_fit, t_fit))
    corners = [list(footprint_ends(outs[0][1], radius)) + [bb2["le_top"], bb2["te_top"]]]
    for (mm, tt), (_, shape) in zip(placements, outs[1:]):
        tops = np.array([[wt["le_top"][0], wt["le_top"][1], 0.0], [wt["te_top"][0], wt["te_top"][1], 0.0]])
        corners.append(list(footprint_ends(shape, radius)) + list(plane_coords(tops @ mm.T + tt)))
    print("where each meets the hull (LE, TE): " + "; ".join(
        f"{name} ({c[0][0]:.1f}, {c[0][1]:.1f}), ({c[1][0]:.1f}, {c[1][1]:.1f})" for name, c in zip(CASES, corners)))
    leans = [(lean(None, bb2["le"]), lean(None, bb2["te"]))] + [(lean(mm, wt["le"]), lean(mm, wt["te"]))
                                                                 for mm, _ in placements]
    pitches = [float(np.degrees(np.arctan2(c[0][1] - c[1][1], c[0][0] - c[1][0]))) for c in corners]
    ref_pts = surface_samples(outs[0][1], radius)
    rows, d_rule = [summary(outs[0][1], corners[0], leans[0], pitches[0], None, None, None)[0]], None
    for i in range(1, len(CASES)):
        row, d1 = summary(outs[i][1], corners[i], leans[i], pitches[i], ref_pts, outs[0][1],
                          surface_samples(outs[i][1], radius))
        rows.append(row)
        d_rule = d1 if i == 1 else d_rule
    print(f"distances done ({time.time() - t0:.0f} s)")
    head = ("case root_le_x root_le_s root_te_x root_te_s tip_le_x tip_le_s tip_te_x tip_te_s le_lean_deg "
            "te_lean_deg root_pitch_deg volume_mm3 area_mm2 bb2_from_rudder_mean bb2_from_rudder_rms "
            "bb2_from_rudder_p95 bb2_from_rudder_max rudder_from_bb2_mean rudder_from_bb2_max")
    with open(os.path.join(out, "bb2_rudder_compare.dat"), "w") as fh:
        fh.write("# BB2's stern plane (hull.iges, 45 deg) against Wind_Tunnel_Rudder.stp at 1:"
                 f"{p[0]:g} (the free fit's scale 1:{free[0]:.4f}), above the hull, by bb2_rudder.py\n"
                 "# case 0: BB2's stern plane (on the vehicles); 1: the rudder with its TE line square to the axis, "
                 "as BB2's, resting on the hull; 2: the rudder pitched to the hull (root LE and TE on it); 3: the "
                 "rudder of the planform best fit (the first placement)\n"
                 "# corners (x, s) in mm in the 45 deg plane (s from the axis): the foremost and aftmost points where "
                 "each meets the hull (root LE, root TE), the tops of the straight LE and TE; leans of the edges "
                 "from the radial direction (deg, positive aft); the root chord's angle to the axis (deg); volume "
                 "and area in the 45 deg plane above the hull; distances (mm) of BB2's surface from the rudder's "
                 f"({SAMPLES} points) and back\n"
                 f"# 1: the TE line at {RAKE:g} deg meets the hull at ({te_hull[0]:.2f}, {te_hull[1]:.2f}), BB2's TE "
                 f"on the hull; root chord at {info['pitch']:.4f} deg, its LE ({le_r[0]:.2f}, {le_r[1]:.2f}) and TE "
                 f"({te_r[0]:.2f}, {te_r[1]:.2f}); extended {1000 * info['depth']:.2f} mm along the TE line (gap "
                 f"{1000 * info['gap']:.2f} mm) plus the lift {VC.ROOT_LIFT_MM:g} mm\n"
                 f"# 2: root LE ({le_h[0]:.2f}, {le_h[1]:.2f}) and TE ({te_h[0]:.2f}, {te_h[1]:.2f}) on the hull's "
                 f"profile, pitch {info_h['pitch']:.4f} deg, dip {1000 * info_h['dip']:.2f} mm, extrusion "
                 f"{1000 * info_h['depth']:.2f} mm; 3: turned {np.degrees(p[1]):.4f} deg, edges off BB2's by rms LE "
                 f"{rep['le_rms']:.2f} TE {rep['te_rms']:.2f} mm, extrusion {depth_fit * p[0]:.1f} mm\n# {head}\n")
        for i, r in enumerate(rows):
            fh.write(f"{i}  " + "  ".join(f"{v:.4f}" if abs(v) < 1e6 else f"{v:.6e}" for v in r) + "\n")

    # cuts at s = const
    cut_rows, cuts = [], {}
    for sv in CHECK_SPANS:
        parts = [bb2_cut(shape, sv) for _, shape in outs]
        cuts[sv] = parts
        r = [sv]
        for q in parts:
            r += [q[:, 0].max(), q[:, 0].min(), np.ptp(q[:, 1])] if q is not None else [np.nan] * 3
        r += [S.distance_to_curve(parts[0], B_closed(q)).max() if q is not None else np.nan for q in parts[1:]]
        cut_rows.append(r)
    cut_rows = np.array(cut_rows)
    np.savetxt(os.path.join(out, "bb2_rudder_sections.dat"), cut_rows, fmt="%.3f",
               header="cuts at s = const (mm): s, then per case (0 BB2, 1 the TE square to the axis, 2 pitched to the "
                      "hull, 3 the fit) LE x, TE x, thickness; the largest distance of BB2's cut from case 1's, 2's, "
                      "3's\ns le0 te0 t0 le1 te1 t1 le2 te2 t2 le3 te3 t3 d01 d02 d03")
    plot_compare(os.path.join(out, "bb2_rudder_compare.png"), outs, cut_rows, cuts, ref_pts, d_rule, x_iges, r_iges,
                 f"above the hull; TE square to the axis at x {te_hull[0]:.0f} (BB2's TE on the hull), resting on the "
                 f"hull at its root LE, x {le_r[0]:.0f}; root chord at {info['pitch']:.2f} deg")
    names = ("root LE", "root TE", "tip LE", "tip TE")
    for i, name in enumerate(CASES):
        r = rows[i]
        print(f"{name}: " + ", ".join(f"{names[k]} ({r[2 * k]:.0f}, {r[2 * k + 1]:.0f})" for k in range(4))
              + f"; leans LE {r[8]:.2f} TE {r[9]:.2f} deg, root {r[10]:.2f} deg; volume {r[11]:.4e} mm^3, area "
              f"{r[12]:.4e} mm^2" + ("" if i == 0 else f"; BB2 from it: mean {r[13]:.1f}, rms {r[14]:.1f}, 95 % "
                                                       f"{r[15]:.1f}, max {r[16]:.1f} mm; back: mean {r[17]:.1f}, "
                                                       f"max {r[18]:.1f} mm"))

    # the rudder with its TE square, with its root extension (the vehicles carry BB2's own stern planes)
    print(f"rudder at 45 deg with its root extension: {VC.solid_report(rudder)}")
    XL.write_step(rudder, os.path.join(out, "bb2_rudder.step"))
    XL.write_iges(rudder, os.path.join(out, "bb2_rudder.iges"))
    print(f"done in {time.time() - t0:.0f} s")


def B_closed(q):
    """A cut's points as one closed polyline: ordered by angle about their middle."""
    c = q.mean(axis=0)
    return q[np.argsort(np.arctan2(q[:, 1] - c[1], (q[:, 0] - c[0]) / max(np.ptp(q[:, 0]), 1e-9)))]


if __name__ == "__main__":
    main()
