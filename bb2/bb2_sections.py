"""
bb2_sections.py

The MARIN stock propeller S7371R of BB2 (prop.stp, full scale, mm) as XGeom blade input. Blade 1 is cut by
cylinders about the shaft (x) at the radii RR. Per cut:
  nose-tail line   the two points of the cut farthest apart on the unrolled cylinder (s = r theta, x); the LE is
                   the upstream one (larger x)
  pitch            P = 2 pi r tan(phi), phi the angle of the nose-tail line
  chord            c' = c / 1.005, c the nose-tail length: para.py and BladeSurface close a section at x/c' = 1.005,
                   so c' is the chord they take (the last 0.5 % is their rounded TE)
  shape            camber and half thickness at the same x/c' (airfoils.shape: the surfaces against sqrt(x/c')) on
                   para.py's 26 stations; f/c and t/c their maxima over c'
  skew, rake       of the reference point 0.5 c' from the LE on the nose-tail line, inverted from para.py's
                   mapping: skew = theta_ref - theta (deg, theta_ref the root's), rake = x0 - x + r skew tan(phi)
                   (x0 the generator line, F6: 31.4 mm model scale aft of the hub's forward end)

Frames. The STP/IGES frame: mm, x the shaft axis towards the bow, theta = atan2(z, y); the propeller turns towards
larger theta. The blade frame of para.py and BladeSurface (X_CAD_new's IGES, mm there): X downstream, the blade
along +Z at zero skew. A blade-frame point (X, Y, Z) is at x = x0 - X, y = -sin(t) Y + cos(t) Z,
z = cos(t) Y + sin(t) Z in the STP frame, t = theta_ref (blade_to_cad).

Inputs: the propeller (STEP) and the hull (IGES; the hub's forward end, where the hull radius equals the hub's,
places the generator line). Writes into the output folder:
  bb2_sections_raw.dat     the cuts, x y z (mm, the STP frame), a block per r/R
  bb2_section_tables.dat   the section shape at each r/R, para.py's section table format, a block per r/R
  bb2_section_table.dat    the shape at r/R 0.7 alone: the one table blade_modify (para.py, BladeSurface) takes
  bb2_blade_params.dat     the XGeom blade parameter file (blade_modify.read_params): D, Z, the root, the tables,
                           every vary line N, r/R P/D c/D t/c f/c skew_deg rake/D; c/D beyond r/R 0.9 is divided by
                           BladeSurface's tip taper, so the CAD gets the measured chord
  bb2_sections_check.png   the distributions against r/R
and prints, per cut, how far the cut lies from the blade the files give (para.py's mapping of the cut's own table;
BladeSurface with one table, bb2_section_table.dat).

    python bb2_sections.py prop.stp hull.iges [out_dir]
"""

import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # the X_geom folder

import airfoils                                            # noqa: E402
import xcad_loft as XL                                     # noqa: E402

# --------------------------------------------------------------------------
# The propeller and the hub (drawing F6, report 26053-7-SMB: model scale mm times the scale)
# --------------------------------------------------------------------------
D_MM = 5000.0
R_MM = 0.5 * D_MM
BLADES = 6
SCALE = 18.348
HUB_R_FWD = 0.5 * 86.9 * SCALE          # the hub's forward end (meets the hull), mm
HUB_R_SPLIT = 0.5 * 45.9 * SCALE        # hub and cap part, mm
HUB_L_SPLIT = 46.0 * SCALE              # forward end to the split, mm
HUB_L = 85.5 * SCALE                    # forward end to the cap's tip, mm
GEN_L = 31.4 * SCALE                    # forward end to the generator line, mm
P_TIP_MM = 226.09 * SCALE               # pitch at r/R 1 (the drawing's table)

ROOT_RR = 0.2                           # the blade's root (its faces start at r = 500 mm)
ROOT_CUT_MM = 500.01                    # the root cut (a cut on the faces' own edge fails)
BOUNDARY_RR = 0.975                     # the tip face starts there: cut just outside (BOUNDARY_CUT_MM)
BOUNDARY_CUT_MM = 2437.55
RR = np.array([ROOT_RR, 0.21, 0.22, 0.235, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.55, 0.6,
               0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.96, 0.97, BOUNDARY_RR, 0.98, 0.985,
               0.99, 0.995, 0.998, 0.999])
REF_RR = 0.7                            # the section of bb2_section_table.dat
TE_END = 1.005                          # para.py's closing station, x/c'
TAPER_START = 0.90                      # blade_surface_new.TIP_TAPER_START
STATIONS = airfoils.STATIONS            # para.py's 26 stations


def cut_radius_mm(rr):
    if rr == ROOT_RR:
        return ROOT_CUT_MM
    if rr == BOUNDARY_RR:
        return BOUNDARY_CUT_MM
    return rr * R_MM


def taper(rr):
    """BladeSurface's tip chord factor sqrt(1 - t^3), t = (r/R - 0.9) / 0.1 (1 below r/R 0.9)."""
    t = np.clip((np.asarray(rr, dtype=float) - TAPER_START) / (1.0 - TAPER_START), 0.0, 1.0)
    return np.sqrt(1.0 - t ** 3)


# --------------------------------------------------------------------------
# OCC: reading, cutting
# --------------------------------------------------------------------------

def _explore(shape, kind):
    TopExp_Explorer = XL._occ("TopExp", "TopExp_Explorer")
    top = XL._occ("TopAbs", f"TopAbs_{kind}")
    cast = XL._topods(kind.capitalize())
    ex, out = TopExp_Explorer(shape, top), []
    while ex.More():
        out.append(cast(ex.Current()))
        ex.Next()
    return out


def read_cad(path):
    """The shape of a STEP or IGES file."""
    if path.lower().endswith((".stp", ".step")):
        Reader = XL._occ("STEPControl", "STEPControl_Reader")
    else:
        Reader = XL._occ("IGESControl", "IGESControl_Reader")
    r = Reader()
    if int(r.ReadFile(path)) != 1:
        raise IOError(f"cannot read {path}")
    r.TransferRoots()
    return r.OneShape()


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


def cylinder_cut(shape, r, n=1000):
    """The closed loop where the cylinder of radius r (mm, axis x) cuts the shape: (m, 3), mm."""
    gp_Cylinder, gp_Ax3, gp_Pnt, gp_Dir = XL._occ("gp", "gp_Cylinder", "gp_Ax3", "gp_Pnt", "gp_Dir")
    MakeFace = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace")
    Section = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Section")
    cyl = gp_Cylinder(gp_Ax3(gp_Pnt(-2.0e4, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), float(r))
    tool = MakeFace(cyl, 0.0, 2.0 * np.pi, 0.0, 4.0e4).Face()
    algo = Section(shape, tool, False)
    algo.ComputePCurveOn1(True)
    algo.Approximation(True)
    algo.Build()
    edges = _explore(algo.Shape(), "EDGE")
    if not edges:
        raise RuntimeError(f"no cut at r = {r} mm")
    return _chain([_polyline(e, n) for e in edges])


def hull_radius(faces, x):
    """The hull's radius at x (mm): the largest distance from the axis on the plane x = const."""
    gp_Pln, gp_Pnt, gp_Dir = XL._occ("gp", "gp_Pln", "gp_Pnt", "gp_Dir")
    Section = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Section")
    BRepAdaptor_Curve = XL._occ("BRepAdaptor", "BRepAdaptor_Curve")
    algo = Section(faces, gp_Pln(gp_Pnt(float(x), 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)))
    algo.Build()
    best = 0.0
    for e in _explore(algo.Shape(), "EDGE"):
        c = BRepAdaptor_Curve(e)
        for t in np.linspace(c.FirstParameter(), c.LastParameter(), 41):
            p = c.Value(float(t))
            best = max(best, float(np.hypot(p.Y(), p.Z())))
    return best


def hub_forward_x(hull_shape, x_lo=1000.0, x_hi=3000.0):
    """x (mm) where the hull's radius equals HUB_R_FWD (bisection; the hull grows towards the bow there)."""
    for _ in range(30):
        mid = 0.5 * (x_lo + x_hi)
        if hull_radius(hull_shape, mid) < HUB_R_FWD:
            x_lo = mid
        else:
            x_hi = mid
    return 0.5 * (x_lo + x_hi)


# --------------------------------------------------------------------------
# One cut
# --------------------------------------------------------------------------

def _far_pair(p):
    from scipy.spatial import ConvexHull
    h = ConvexHull(p).vertices
    q = p[h]
    d = np.hypot(*(np.subtract.outer(q[:, k], q[:, k]) for k in range(2)))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return int(h[i]), int(h[j])


def section(loop, r):
    """A cut (mm, the STP frame) at radius r (mm) as para.py takes it: dict with c (c', mm), phi (rad), theta and x
    of the reference point (rad, mm), the shape (yc, slope, th at STATIONS, x/c' units), t_c, f_c (over c'), and
    the coordinates xy (x/c', y/c': the LE at (0, 0), the TE at (1.005, 0), from the TE round to the TE)."""
    th = np.unwrap(np.arctan2(loop[:, 2], loop[:, 1]))
    p = np.column_stack((r * th, loop[:, 0]))
    i, j = _far_pair(p)
    le, te = (i, j) if p[i, 1] > p[j, 1] else (j, i)
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
            "half": half, "t_c": t_c, "f_c": f_c, "xy": xy}


def table_rows(sec):
    """para.py's section table of a cut (26 rows: x/c, camber % c, camber slope, half thickness / t over
    cos(atan(slope))), as airfoils.section_table writes one."""
    rows = np.column_stack((STATIONS, 100.0 * sec["yc"], sec["slope"],
                            sec["half"] / sec["t_c"] / np.cos(np.arctan(sec["slope"]))))
    rows[0] = 0.0
    return rows


TABLE_HEAD = "x/c,camber (% c),camber slope,half thickness / max thickness"


def table_text(rows):
    return TABLE_HEAD + "\n" + "\n".join(",".join(f"{v:.6f}" for v in row) for row in rows) + "\n"


def write_tables(path, rr, tables, note):
    """bb2_section_tables.dat: '# r/R <value>' and para.py's table (its header line, 26 rows) per block."""
    with open(path, "w") as fh:
        fh.write(f"# {note}\n")
        for r, rows in zip(rr, tables):
            fh.write(f"# r/R {r:.6f}\n" + table_text(rows))
    return path


def read_tables(path):
    """[(r/R, table text)] of a bb2_section_tables.dat (each text a section table pandas reads as para.py does)."""
    out, cur = [], None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("# r/R"):
                cur = [float(s.split()[2]), []]
                out.append(cur)
            elif cur is not None and s and not s.startswith("#"):
                cur[1].append(s)
    return [(r, "\n".join(lines) + "\n") for r, lines in out]


def read_frame(params_path):
    """(x0 mm, theta_ref rad) from the parameter file's '# bb2_frame' line."""
    with open(params_path) as fh:
        for line in fh:
            parts = line[1:].split() if line.startswith("#") else []
            if parts and parts[0] == "bb2_frame":
                v = dict(zip(parts[1::2], parts[2::2]))
                return float(v["x0_mm"]), np.radians(float(v["theta_ref_deg"]))
    raise ValueError(f"{params_path}: no '# bb2_frame' line")


def blade_to_cad(pts_m, x0, theta_ref):
    """Blade-frame points (m) in the STP frame (mm)."""
    p = 1000.0 * np.asarray(pts_m, dtype=float)
    s, c = np.sin(theta_ref), np.cos(theta_ref)
    return np.stack((x0 - p[..., 0], -s * p[..., 1] + c * p[..., 2], c * p[..., 1] + s * p[..., 2]), axis=-1)


def cad_matrix(x0, theta_ref):
    """(m, t) of blade_to_cad for a shape in mm: x_cad = m x + t (a rotation: det m = 1)."""
    s, c = np.sin(theta_ref), np.cos(theta_ref)
    return np.array([[-1.0, 0.0, 0.0], [0.0, -s, c], [0.0, c, s]]), np.array([x0, 0.0, 0.0])


# --------------------------------------------------------------------------
# Checks: a cut against the blade the files give
# --------------------------------------------------------------------------

def unrolled(pts, r):
    """(s, x) of points near the cylinder of radius r (mm): s = r theta, theta unwrapped."""
    return np.column_stack((r * np.unwrap(np.arctan2(pts[:, 2], pts[:, 1])), pts[:, 0]))


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


def cut_deviation(loop, r, blade, x0, theta_ref, n=4001):
    """(max, rms) distance (mm) of a cut from the BladeSurface blade at the same radius, on the cylinder."""
    eta = float(blade.eta_of_r(r / R_MM))
    xi = np.linspace(0.0, 1.0, n)[:-1]
    gen = blade_to_cad(blade.b(xi, np.full(xi.shape, eta)), x0, theta_ref)
    p, q = unrolled(loop, r), unrolled(gen, r)
    q[:, 0] += r * 2.0 * np.pi * np.round((p[:, 0].mean() - q[:, 0].mean()) / (r * 2.0 * np.pi))
    d = distance_to_curve(p, q)
    return float(d.max()), float(np.sqrt(np.mean(d ** 2)))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    prop_path, hull_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "outputs")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    x_fwd = hub_forward_x(read_cad(hull_path))
    x0 = x_fwd - GEN_L
    print(f"hub forward end x {x_fwd:.2f} mm (hull radius {HUB_R_FWD:.2f} mm), generator line x0 {x0:.2f} mm")

    blade = _explore(read_cad(prop_path), "SHELL")[0]
    secs, loops = [], []
    for rr in RR:
        r = cut_radius_mm(rr)
        loop = cylinder_cut(blade, r)
        secs.append(section(loop, r))
        loops.append(loop)
    theta_ref = secs[0]["theta"]

    # the distributions at the cuts
    r_mm = np.array([s["r"] for s in secs])
    phi = np.array([s["phi"] for s in secs])
    skew = theta_ref - np.array([s["theta"] for s in secs])                                   # rad
    rake = x0 - np.array([s["x"] for s in secs]) + r_mm * skew * np.tan(phi)                   # mm
    cols = {"P/D": 2.0 * np.pi * r_mm * np.tan(phi) / D_MM,
            "c/D": np.array([s["c"] for s in secs]) / D_MM / taper(RR),
            "t/c": np.array([s["t_c"] for s in secs]),
            "f/c": np.array([s["f_c"] for s in secs]),
            "skew_deg": np.degrees(skew),
            "rake/D": rake / D_MM}

    # the row at the tip (r/R 1): the drawing's pitch, the others continued from the last three cuts
    tip = {}
    for k, v in cols.items():
        tip[k] = float(np.polyval(np.polyfit(RR[-3:], v[-3:], 2), 1.0))
    tip["P/D"] = P_TIP_MM / D_MM
    rr_all = np.r_[RR, 1.0]
    table = np.column_stack([rr_all] + [np.r_[cols[k], tip[k]] for k in cols])

    # files
    tables = [table_rows(s) for s in secs]
    with open(os.path.join(out, "bb2_sections_raw.dat"), "w") as fh:
        fh.write(f"# MARIN S7371R (BB2) blade 1 cut by cylinders about x: x y z mm in {os.path.basename(prop_path)}'s "
                 f"frame, one block per r/R\n")
        for rr, r, loop in zip(RR, r_mm, loops):
            fh.write(f"# r/R {rr:.6f}  r_mm {r:.3f}  points {len(loop)}\n")
            fh.write("\n".join(f"{q[0]:.5f} {q[1]:.5f} {q[2]:.5f}" for q in loop) + "\n")
    write_tables(os.path.join(out, "bb2_section_tables.dat"), RR, tables,
                 "MARIN S7371R (BB2): para.py's section table of each cut (x/c over c' = nose-tail length / 1.005)")
    k_ref = int(np.argmin(np.abs(RR - REF_RR)))
    with open(os.path.join(out, "bb2_section_table.dat"), "w") as fh:
        fh.write(table_text(tables[k_ref]).replace(TABLE_HEAD, TABLE_HEAD + f" (MARIN S7371R at r/R {REF_RR})"))
    hub_height = np.ceil(2.02 * max(abs(v) for s in secs[:1] for v in _root_extent(s, x0)) / 100.0) / 10.0
    lines = ["# XGeom propeller blade parameters",
             f"# MARIN stock propeller S7371R of BB2, cut from {os.path.basename(prop_path)} by bb2_sections.py",
             "# c/D beyond r/R 0.9 is the measured chord over BladeSurface's tip taper sqrt(1 - t^3), t = (r/R - 0.9) "
             "/ 0.1; the row at r/R 1 is continued from the last cuts (P/D the drawing's)",
             f"# bb2_frame  x0_mm {x0:.4f}  theta_ref_deg {np.degrees(theta_ref):.8f}",
             "# section_tables  bb2_section_tables.dat      (the shape per r/R: bb2_build.py)",
             f"# diameter_m     {D_MM / 1000.0:g}",
             f"# blades         {BLADES}",
             f"# root_r         {ROOT_RR:g}",
             f"# hub_height_m   {hub_height:g}",
             "# section_table  bb2_section_table.dat",
             "# Y/N per parameter: Y changes it (a design variable; low high, optional: the range of its values), "
             "N keeps it as this file has it"]
    lines += [f"# vary  {n:<11s} N" for n in ("pitch", "chord", "thickness", "camber", "skew", "rake", "radius",
                                               "hub_radius", "hub_height")]
    lines.append("# r/R  P/D  c/D  t/c  f/c  skew_deg  rake/D")
    lines += ["  ".join(f"{v:.8f}" for v in row) for row in table]
    params_path = os.path.join(out, "bb2_blade_params.dat")
    with open(params_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {params_path} and the section files ({time.time() - t0:.0f} s)")

    # checks
    import blade_modify as BM
    from blade_surface_new import BladeSurface
    params = BM.read_params(params_path)
    design, _ = BM.fit_design(params)
    one = BM.blade_surface(design, params)
    print(" r/R     P/D     c'/D    t/c     f/c    skew   rake/D | from the files: own table (para.py's mapping)  "
          "one table (BladeSurface), mm max / rms")
    for k, (rr, s, loop) in enumerate(zip(RR, secs, loops)):
        own = BladeSurface(*(_const(table[k, i]) for i in (4, 1)), _const(s["c"] / D_MM), _const(table[k, 3]),
                           _const(table[k, 5]), _const(table[k, 6]), d=D_MM / 1000.0, r_root=0.1, taper_start=None,
                           thick_floor=False, airfoil_path=io.StringIO(table_text(tables[k])))
        a = cut_deviation(loop, s["r"], own, x0, theta_ref)
        b = cut_deviation(loop, s["r"], one, x0, theta_ref)
        print(f"{rr:.4f} {table[k, 1]:7.4f} {s['c'] / D_MM:7.4f} {table[k, 3]:7.4f} {table[k, 4]:7.4f} "
              f"{table[k, 5]:6.2f} {table[k, 6]:8.5f} | {a[0]:6.2f} / {a[1]:5.2f}   {b[0]:6.2f} / {b[1]:5.2f}")
    plot_check(os.path.join(out, "bb2_sections_check.png"), table)
    print(f"done in {time.time() - t0:.0f} s")


def _const(v):
    v = float(v)
    return lambda r: np.full(np.shape(r), v) if np.ndim(r) else np.float64(v)


def _root_extent(sec, x0):
    """The root section's axial extent from the generator line (mm, the blade frame's X): its two ends."""
    half = 0.5 * TE_END * sec["c"] * np.sin(sec["phi"])
    xm = x0 - sec["x"] + (0.5 * TE_END - 0.5) * sec["c"] * np.sin(sec["phi"])
    return xm - half, xm + half


def plot_check(path, table):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(13, 7.5))
    FigureCanvasAgg(fig)
    axs = fig.subplots(2, 3)
    names = ("P/D", "c/D (beyond 0.9: over the tip taper)", "t/c", "f/c", "skew (deg)", "rake/D")
    for k, ax in enumerate(axs.flat):
        ax.plot(table[:, 0], table[:, k + 1], "o-", color="#2a78d6", ms=3, lw=1.2)
        if k == 1:
            ax.plot(table[:, 0], table[:, 2] * taper(table[:, 0]), "-", color="#9a9994", lw=1, label="c'/D")
            ax.legend(fontsize=8)
        ax.set_xlabel("r/R")
        ax.set_ylabel(names[k])
        ax.grid(True, color="#e4e3df")
    fig.suptitle("MARIN S7371R (BB2): the distributions of bb2_blade_params.dat (the cuts and the row at r/R 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


if __name__ == "__main__":
    main()
