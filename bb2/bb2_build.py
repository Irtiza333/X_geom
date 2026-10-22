"""
bb2_build.py

BB2 with its stock propeller S7371R in the STP/IGES frame (mm, x towards the bow): one blade on the conic hub and
cap of drawing F6, and the aft half of the hull, from bb2_sections.py's files.

  blade    blade_modify reads bb2_blade_params.dat (every vary line N: the file's distributions); the surface is
           blade_surface_new.BladeSurface with the section shape of each r/R of bb2_section_tables.dat, cubic in
           r/R between them (SectionsBladeSurface); its root runs ROOT_EXT_M below r/R 0.2, inside the hub, so the
           hub cuts the root sections; tip_surfaces_new.build_drdc_grids and X_CAD_new.X_CAD (no hub) give the
           five faces, closed at the extended root by the piece of that cylinder inside the ring
           (vehicle_cad.blade_solid), and moved to the STP frame (bb2_sections.cad_matrix)
  hub      a solid of revolution: from the forward end (radius HUB_R_FWD, where the hull has that radius) the cone
           of the drawing (to radius HUB_R_SPLIT HUB_L_SPLIT aft, hub and cap one solid) to the cap's tip HUB_L
           aft, the tip rounded by the sphere tangent to the cone
  hull     hull_modify fitted to hull.iges's radius from the parallel middle body to the hub's forward end (the
           tail; a dummy nose ahead of the middle body), built, turned to the IGES frame and cut at the file's
           mid-length plane
  vehicle  the three fused (vehicle_cad.fuse)
  six      the blade turned to each of prop.stp's six blade positions (BLADE_TURNS_DEG about x), fused with the
           hub; with the hull
  sector   one blade on its 1/6 of the hub, the periodic hub strip of hub_new (TM 2013-178 Sec. 10): the hub
           within p_h(x) +- 30 deg, p_h on the mid-line of the blade's footprint on the hub, continued to both ends
           of the hub with zero slope (blade_sector); as a solid (the two periodic sides included) and as its walls
           alone (the blade, the hub's strip and its piece of the forward end)
  quadrant the part in the quadrant y >= 0, z >= 0 (hull.iges's) of the hull, and of the hull with the hub; IGES
           with solids (manifold solids, shells, faces), as hull.iges is written

Writes into the output folder (bb2_sections.py's): bb2_blade_hub.step/.iges, bb2_hull.step/.iges,
bb2_hull_hub.step (the hull with the hub, no blade), bb2_vehicle.step/.iges (one blade), bb2_prop6.step/.iges and
bb2_vehicle6.step/.iges (six blades), bb2_blade_sector.step/.iges (one blade on 1/6 of the hub, a solid),
bb2_blade_sector_walls.step/.iges (its walls), bb2_hull_quadrant.iges, bb2_hull_hub_quadrant.iges,
bb2_hull_params.dat (the hull tool's station table), bb2_build_check.png (side view, x across), and prints how far
the blade lies from prop.stp's blade 1 at cylinder cuts (at the table's radii and between them) and the hull from
hull.iges.

    python bb2_build.py prop.stp hull.iges [out_dir]
"""

import io
import os
import shutil
import sys
import tempfile
import time

import numpy as np
from scipy.interpolate import RectBivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                  # the X_geom folder
sys.path.insert(0, HERE)

import blade_modify as BM                                  # noqa: E402
import bb2_sections as S                                   # noqa: E402
import hull_modify as HM                                   # noqa: E402
import vehicle_cad as VC                                   # noqa: E402
import xcad_loft as XL                                     # noqa: E402
from blade_surface_new import BladeSurface, CanonicalSectionFamily   # noqa: E402

ROOT_EXT_M = 0.10            # the blade runs this far below its root (r/R 0.2) into the hub: to r = 400 mm
N_V = 801                    # stations around a section behind SectionsBladeSurface's shapes
NOSE_M = 3.0                 # the hull's dummy nose (cut away)
NOSE_AHEAD_M = 1.0           # the middle body reaches this far ahead of the cut
TAIL_ORDERS = range(5, 11)   # hull_modify's tail curve: the lowest order within TAIL_TOL_MM of the file
TAIL_TOL_MM = 1.0
CHECK_RR = (0.205, 0.26, 0.42, 0.62, 0.86, 0.94, 0.9725, 0.9925, 0.9965, 0.9985)   # cuts between the table's
BLADE_TURNS_DEG = (0.0, 60.0, 120.0, 180.0, -120.0, -60.0)   # prop.stp's six blades: its first turned about x
SECTOR_N = (41, 41, 321)     # the periodic wedge's grids: across the radius, across the sector, along the axis
SECTOR_PAD_MM = 50.0         # the wedge reaches this far beyond the hub's ends, and 20 % beyond its radius


# --------------------------------------------------------------------------
# The blade
# --------------------------------------------------------------------------

class SectionsBladeSurface(BladeSurface):
    """BladeSurface with a section shape per r/R: tables [(r/R, section table text)], the camber and thickness
    distributions of each (CanonicalSectionFamily) on N_V stations around the section, cubic in r/R between the
    tables, the first and last held beyond them. The rest is BladeSurface's (the distributions, the tip taper and
    closure, the root extension)."""

    def __init__(self, tables, *args, **kw):
        super().__init__(*args, **kw)
        rk = np.array([r for r, _ in tables], dtype=float)
        v = np.linspace(0.0, 1.0, N_V)                       # 0 at the LE, 1 at the TE: |1 - 2 xi|
        cam, th = [], []
        for _, text in tables:
            fam = CanonicalSectionFamily(None, None, airfoil_path=io.StringIO(text))
            s = fam.s_end * 0.5 * (1.0 - np.cos(np.pi * v))
            cam.append(fam._camber_dist(s))
            th.append(fam._thickness_dist(s) * np.cos(np.arctan(fam._camber_slope(s))))
        self._cam_vr = RectBivariateSpline(v, rk, np.array(cam).T)
        self._th_vr = RectBivariateSpline(v, rk, np.array(th).T)
        self._r_range = (float(rk[0]), float(rk[-1]))

    def _shape(self, xi, r):
        """(s, F_cam, F_th) of CanonicalSectionFamily.basis, at r/R."""
        v = np.abs(1.0 - 2.0 * xi)
        rr = np.clip(r, *self._r_range)
        s = self.canon.s_end * 0.5 * (1.0 - np.cos(np.pi * v))
        f = self._cam_vr(v.ravel(), rr.ravel(), grid=False).reshape(v.shape)
        t = self._th_vr(v.ravel(), rr.ravel(), grid=False).reshape(v.shape)
        return s, f, np.where(xi < 0.5, -1.0, 1.0) * t

    def _canonical(self, xi, r):
        """BladeSurface._canonical with the shape at r/R (the tip closure as there)."""
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        xi, r = np.broadcast_arrays(xi, r)
        s, F_cam, F_th = self._shape(xi, r)
        ys = np.asarray(F_cam * self.MaxCamber(r) + F_th * self.MaxThickness(r), dtype=float)
        mask = r > self._rc
        if np.any(mask):
            c = self._closure_constants()
            rc, rt = self._rc, self.r_tip
            dr = rt - rc
            y0 = F_cam[mask] * c["mc0"]
            s0 = F_cam[mask] * c["dmc0"]
            y1 = F_cam[mask] * c["mc1"]
            s1 = 2.0 * (y1 - y0) / dr - s0
            u = np.clip((r[mask] - rc) / dr, 0.0, 1.0)
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1
            h10 = u ** 3 - 2 * u ** 2 + u
            h01 = -2 * u ** 3 + 3 * u ** 2
            h11 = u ** 3 - u ** 2
            ys_cam = h00 * y0 + h10 * dr * s0 + h01 * y1 + h11 * dr * s1
            g = np.sqrt(np.clip(1.0 - u ** 3, 0.0, None))
            ys = ys.copy()
            ys[mask] = ys_cam + F_th[mask] * np.asarray(self.MaxThickness(r[mask]), dtype=float) * g
        return np.stack([s, ys], axis=-1)


def header_value(path, key):
    with open(path) as fh:
        for line in fh:
            parts = line[1:].split() if line.startswith("#") else []
            if len(parts) >= 2 and parts[0] == key:
                return parts[1]
    raise ValueError(f"{path}: no '# {key}' line")


def blade_surface(out_dir, root_extension=ROOT_EXT_M):
    """(SectionsBladeSurface, x0 mm, theta_ref rad) of the files in out_dir."""
    path = os.path.join(out_dir, "bb2_blade_params.dat")
    params = BM.read_params(path)
    design, _ = BM.fit_design(params)                         # every vary line N: the file's own splines
    tables = S.read_tables(os.path.join(out_dir, header_value(path, "section_tables")))
    blade = SectionsBladeSurface(tables, *design.props(), d=design.diameter, r_root=design.r0,
                                 root_extension=root_extension, airfoil_path=params.section_path())
    return (blade,) + S.read_frame(path)


def blade_cad(blade, x0, theta_ref, work_dir):
    """The blade solid in the STP frame (mm): X_CAD_new's five faces closed at the extended root."""
    from tip_surfaces_new import TipConfig, build_drdc_grids
    from X_CAD_new import X_CAD
    grids = build_drdc_grids(blade, TipConfig(), verbose=False)
    tmp = tempfile.mkdtemp(prefix="x_cad_", dir=work_dir)
    try:
        shape = X_CAD(grids, 0, output_dir=tmp, hub=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    solid = VC.blade_solid(shape, 1000.0 * blade.r_root * blade.d / 2.0)
    m, t = S.cad_matrix(x0, theta_ref)
    return VC.transformed(solid, m, t)


# --------------------------------------------------------------------------
# The hub
# --------------------------------------------------------------------------

def hub_profile(x_fwd):
    """The hub's profile in the STP frame: (forward end radius, cone slope k, the tangent point (x, r), the nose
    sphere's centre x and radius, the tip x)."""
    k = (S.HUB_R_FWD - S.HUB_R_SPLIT) / S.HUB_L_SPLIT
    sin_a, cos_a = k / np.hypot(1.0, k), 1.0 / np.hypot(1.0, k)
    z_apex = S.HUB_R_FWD / k
    rho = (z_apex - S.HUB_L) * sin_a / (1.0 - sin_a)          # the sphere through the tip, tangent to the cone
    z_c = S.HUB_L - rho
    z_t, r_t = z_c + rho * sin_a, rho * cos_a
    return k, (x_fwd - z_t, r_t), x_fwd - z_c, rho, x_fwd - S.HUB_L


def hub_solid(x_fwd):
    """The hub and cap: the profile (the forward end, the cone, the nose arc, the axis) turned about x."""
    gp_Pnt, gp_Ax1, gp_Dir = XL._occ("gp", "gp_Pnt", "gp_Ax1", "gp_Dir")
    MakeEdge, MakeWire, MakeFace = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_MakeEdge", "BRepBuilderAPI_MakeWire",
                                           "BRepBuilderAPI_MakeFace")
    GC_MakeArcOfCircle = XL._occ("GC", "GC_MakeArcOfCircle")
    MakeRevol = XL._occ("BRepPrimAPI", "BRepPrimAPI_MakeRevol")
    k, (x_t, r_t), x_c, rho, x_tip = hub_profile(x_fwd)
    p0, p1 = gp_Pnt(x_fwd, 0.0, 0.0), gp_Pnt(x_fwd, S.HUB_R_FWD, 0.0)
    pt, ptip = gp_Pnt(x_t, r_t, 0.0), gp_Pnt(x_tip, 0.0, 0.0)
    a = 0.5 * (np.arctan2(r_t, x_t - x_c) + np.pi)               # the arc's middle point
    pm = gp_Pnt(x_c + rho * np.cos(a), rho * np.sin(a), 0.0)
    arc = MakeEdge(GC_MakeArcOfCircle(pt, pm, ptip).Value()).Edge()
    wire = MakeWire()
    for e in (MakeEdge(p0, p1).Edge(), MakeEdge(p1, pt).Edge(), arc, MakeEdge(ptip, p0).Edge()):
        wire.Add(e)
    face = MakeFace(wire.Wire(), True).Face()
    return MakeRevol(face, gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), 2.0 * np.pi).Shape()


# --------------------------------------------------------------------------
# The hull
# --------------------------------------------------------------------------

def hull_profile(hull_shape, n=4000):
    """(x, r) of the hull (mm, x rising) from its cut by the plane through the axis at 45 deg."""
    gp_Pln, gp_Pnt, gp_Dir = XL._occ("gp", "gp_Pln", "gp_Pnt", "gp_Dir")
    Section = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Section")
    algo = Section(hull_shape, gp_Pln(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, np.sqrt(0.5), -np.sqrt(0.5))))
    algo.Build()
    pts = np.vstack([S._polyline(e, n) for e in S._explore(algo.Shape(), "EDGE")])
    pts = pts[(pts[:, 1] > 0.0) & (pts[:, 2] > 0.0)]           # the quadrant of the hull file
    x, r = pts[:, 0], np.hypot(pts[:, 1], pts[:, 2])
    o = np.lexsort((-r, x))                                     # by x; at one x (an end face) the largest r first
    x, idx = np.unique(x[o], return_index=True)
    return x, r[o][idx]


def middle_body(hull_shape, tol=1e-3):
    """(aft end x, radius) of the hull's parallel middle body: the face of constant radius, sampled."""
    BRepAdaptor_Surface = XL._occ("BRepAdaptor", "BRepAdaptor_Surface")
    best = None
    for f in S._explore(hull_shape, "FACE"):
        a = BRepAdaptor_Surface(f)
        uu = np.linspace(a.FirstUParameter(), a.LastUParameter(), 21)
        vv = np.linspace(a.FirstVParameter(), a.LastVParameter(), 21)
        p = np.array([[a.Value(float(u), float(v)).X(), a.Value(float(u), float(v)).Y(), a.Value(float(u),
                                                                                                 float(v)).Z()]
                      for u in uu for v in vv])
        r = np.hypot(p[:, 1], p[:, 2])
        if np.ptp(r) < tol and r.mean() > 1.0 and np.ptp(p[:, 0]) > 1.0:
            if best is None or r.mean() > best[1]:
                best = (float(p[:, 0].min()), float(r.mean()))
    if best is None:
        raise RuntimeError("no parallel middle body in the hull file")
    return best


def hull_params(x_iges, r_iges, x_fwd, x_pmb, r_pmb, x_cut, n_tail=161, n_nose=41):
    """hull_modify's station table (m from the dummy nose): the nose an ellipse NOSE_M long, the middle body from
    NOSE_AHEAD_M ahead of the cut to x_pmb, the tail hull.iges's radius from x_pmb to the hub's forward end."""
    ln = NOSE_M
    lm = (x_cut + 1000.0 * NOSE_AHEAD_M - x_pmb) / 1000.0
    lt = (x_pmb - x_fwd) / 1000.0
    rs = r_pmb / 1000.0
    t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_nose)))
    xn = ln * t
    rn = rs * np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, None))
    t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_tail)))
    xc = x_pmb - (x_pmb - x_fwd) * t                             # the tail, STP x
    rt = np.interp(xc, x_iges, r_iges) / 1000.0
    rt[0], rt[-1] = rs, S.HUB_R_FWD / 1000.0
    x = np.r_[xn, ln + lm + (x_pmb - xc[1:]) / 1000.0]
    r = np.r_[rn, rt[1:]]
    return HM.HullParams(ln, lm, lt, 0.0, x, r, r.copy(), True, None,
                         [f"BB2 aft half from hull.iges by bb2_build.py: a dummy {ln:g} m nose, the middle body from "
                          f"{NOSE_AHEAD_M:g} m ahead of the mid-length cut, the tail the file's radius to the hub "
                          f"(STP x = {x_pmb + 1000.0 * (ln + lm):.3f} - 1000 x_m)"])


def fit_hull(params, x_iges, r_iges, x_fwd, x_pmb, x_shift, verbose=True):
    """hull_modify's design of the station table: the tail of the lowest order in TAIL_ORDERS within TAIL_TOL_MM of
    hull.iges (x_iges, r_iges; the STP frame), else the closest."""
    tail = (x_iges >= x_fwd) & (x_iges <= x_pmb)
    design = best = None
    for n in TAIL_ORDERS:
        d = HM.fit_design(params, orders={"tail_r": n})
        err = np.abs(d.radius("r", (x_shift - x_iges[tail]) / 1000.0) * 1000.0 - r_iges[tail]).max()
        if verbose:
            print(f"hull tail of order {n}: N2 {d.curves['tail_r'].exponent:.4f}, largest difference from hull.iges "
                  f"{err:.2f} mm")
        if design is None or err < best:
            design, best = d, err
        if err < TAIL_TOL_MM:
            break
    return design


def hull_solid(design, x_shift, x_cut):
    """The hull tool's solid turned to the IGES frame (x = x_shift - x_tool, y = -y_tool) and cut at x_cut."""
    gp_Pnt = XL._occ("gp", "gp_Pnt")
    MakeBox = XL._occ("BRepPrimAPI", "BRepPrimAPI_MakeBox")
    Cut = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Cut")
    solid = VC.transformed(HM.build_solid(design), np.diag([-1.0, -1.0, 1.0]), np.array([x_shift, 0.0, 0.0]))
    big = 2.0e4
    box = MakeBox(gp_Pnt(x_cut, -big, -big), gp_Pnt(x_cut + big, big, big)).Shape()
    op = Cut(solid, box)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("cutting the hull at mid-length failed")
    return op.Shape()


# --------------------------------------------------------------------------
# The quadrant
# --------------------------------------------------------------------------

def quadrant(shape):
    """The part of a shape in the quadrant y >= 0, z >= 0 (hull.iges's)."""
    gp_Pnt = XL._occ("gp", "gp_Pnt")
    MakeBox = XL._occ("BRepPrimAPI", "BRepPrimAPI_MakeBox")
    Common = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Common")
    b = VC._bbox(shape)
    big = float(np.abs(b).max()) + 100.0
    box = MakeBox(gp_Pnt(b[0, 0] - 100.0, 0.0, 0.0), gp_Pnt(b[1, 0] + 100.0, big, big)).Shape()
    op = Common(shape, box)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("cutting the quadrant failed")
    fix = XL._occ("ShapeFix", "ShapeFix_Shape")(op.Shape())     # the cut hub cone came out invalid once (with a rudder)
    fix.Perform()
    return fix.Shape()


def write_iges_solids(shape, path):
    """IGES in BRep mode (manifold solids, shells, faces), mm."""
    IGESControl_Writer = XL._occ("IGESControl", "IGESControl_Writer")
    w = IGESControl_Writer("MM", 1)
    w.AddShape(shape)
    w.ComputeModel()
    if not w.Write(path):
        raise IOError(f"cannot write {path}")
    return path


# --------------------------------------------------------------------------
# One blade on its 1/Z of the hub (the periodic sector)
# --------------------------------------------------------------------------

def to_blade_frame(shape, x0, theta_ref):
    """A shape of the STP frame in the blade frame (the inverse of bb2_sections.cad_matrix), mm."""
    m, t = S.cad_matrix(x0, theta_ref)
    return VC.transformed(shape, m.T, -m.T @ t)


def footprint(blade_out, n=400):
    """Points of the blade's footprint on the hub: the boundary of the blade's face on the hub's cone."""
    GeomAdaptor_Surface = XL._occ("GeomAdaptor", "GeomAdaptor_Surface")
    GeomAbs_Cone = XL._occ("GeomAbs", "GeomAbs_Cone")
    surface = XL._static(XL._occ("BRep", "BRep_Tool"), "Surface")
    faces = [f for f in S._explore(blade_out, "FACE") if GeomAdaptor_Surface(surface(f)).GetType() == GeomAbs_Cone]
    if len(faces) != 1:
        raise RuntimeError(f"the blade meets the hub's cone in {len(faces)} faces, expected 1")
    return np.vstack([S._polyline(e, n) for e in S._explore(faces[0], "EDGE")])


def sector_wedge(p_h, th_ref, x_lo, x_hi, r_out, n_blades):
    """The solid wedge theta in p_h(x) +- pi/Z about the axis (hub_new's angle: atan2(y, z) - th_ref), x_lo .. x_hi,
    radius 0 .. r_out (blade frame, mm): its two periodic sides, outer face and end caps from grids sharing their
    edges point for point (X_CAD_new.grid_to_bspline_surface), sewn."""
    from X_CAD_new import grid_to_bspline_surface
    n_r, n_s, n_x = SECTOR_N
    MakeFace = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace")
    x = np.linspace(x_lo, x_hi, n_x)
    s = np.linspace(0.0, 1.0, n_s)
    rho = np.linspace(0.0, r_out, n_r)
    ph = lambda xx: th_ref + p_h(np.clip(xx, p_h.x[0], p_h.x[-1]))
    half = np.pi / n_blades

    def point(xx, rr, th):
        return np.stack(np.broadcast_arrays(xx, rr * np.sin(th), rr * np.cos(th)), axis=-1)

    grids = [point(x[None, :], rho[:, None], (ph(x) - half)[None, :]),                        # side at -pi/Z
             point(x[None, :], rho[:, None], (ph(x) + half)[None, :]),                        # side at +pi/Z
             point(x[None, :], r_out, ph(x)[None, :] + (2.0 * s - 1.0)[:, None] * half)]    # outer face
    for xe in (x_lo, x_hi):                                                                 # end caps
        grids.append(point(xe, rho[:, None], (ph(xe) + (2.0 * s - 1.0) * half)[None, :]))
    faces = [MakeFace(grid_to_bspline_surface(g / 1000.0), 1.0e-6).Face() for g in grids]
    solid, free = VC._sewn_solid(faces)
    if free:
        raise RuntimeError(f"the periodic wedge has {free} free edges")
    return solid


def blade_sector(blade_shape, hub, x0, theta_ref, x_fwd, n_blades=S.BLADES):
    """One blade and its 1/Z of the hub, the periodic sector (STP frame, mm), as a solid, and the walls alone
    (the blade above the hub, the hub's strip, the strip's piece of the hub's forward end; no periodic faces).
    The sector is hub_new's (TM 2013-178 Sec. 10): theta within p_h(x) +- pi/Z, p_h on the mid-line of the
    blade's footprint on the hub (hub_new._centred_centre_curve), continued to both ends of the hub and meeting
    them with zero slope. Returns (solid, walls, info)."""
    import hub_new
    Cut = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Cut")
    Common = XL._occ("BRepAlgoAPI", "BRepAlgoAPI_Common")
    blade_bf, hub_bf = to_blade_frame(blade_shape, x0, theta_ref), to_blade_frame(hub, x0, theta_ref)

    def boolean(op, a, b):
        o = op(a, b)
        o.Build()
        if not o.IsDone():
            raise RuntimeError("a boolean of the sector failed")
        return o.Shape()

    blade_out = boolean(Cut, blade_bf, hub_bf)                       # the blade above the hub
    fp = footprint(blade_out)
    th_fp = np.arctan2(fp[:, 1], fp[:, 2])
    th_ref = float(np.arctan2(np.sin(th_fp).mean(), np.cos(th_fp).mean()))
    k, (x_t, r_t), x_c, rho, x_tip = hub_profile(x_fwd)
    x_lo, x_hi = x0 - x_fwd, x0 - x_tip                              # the hub's ends, blade frame
    p_h = hub_new._centred_centre_curve(fp, th_ref, x_lo, x_hi)[0]
    rel = np.arctan2(np.sin(th_fp - th_ref), np.cos(th_fp - th_ref)) - p_h(fp[:, 0])
    gap = np.degrees(np.pi / n_blades - np.abs(rel).max())
    if gap < hub_new.FOOT_MARGIN_DEG:
        raise RuntimeError(f"the footprint clears its sector's edges by {gap:.2f} deg only")
    wedge = sector_wedge(p_h, th_ref, x_lo - SECTOR_PAD_MM, x_hi + SECTOR_PAD_MM, 1.2 * S.HUB_R_FWD, n_blades)
    hub_sector = boolean(Common, hub_bf, wedge)
    solid = VC.fuse([hub_sector, blade_out])
    m, t = S.cad_matrix(x0, theta_ref)
    solid = VC.transformed(solid, m, t)

    # the walls: every face but the two periodic sides (faces on theta = p_h -+ pi/Z inside the hub)
    def periodic(f):
        q = VC._bbox(f).mean(axis=0)
        GeomAPI_ProjectPointOnSurf = XL._occ("GeomAPI", "GeomAPI_ProjectPointOnSurf")
        surface = XL._static(XL._occ("BRep", "BRep_Tool"), "Surface")
        gp_Pnt = XL._occ("gp", "gp_Pnt")
        pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, q)), surface(f))
        p = pr.NearestPoint()
        pb = m.T @ (np.array([p.X(), p.Y(), p.Z()]) - t)                 # blade frame
        th = np.arctan2(pb[1], pb[2]) - th_ref - p_h(np.clip(pb[0], x_lo, x_hi))
        th = np.arctan2(np.sin(th), np.cos(th))
        return abs(abs(th) - np.pi / n_blades) < 1.0e-4 and np.hypot(pb[1], pb[2]) > 1.0
    faces = S._explore(solid, "FACE")
    keep = [f for f in faces if not periodic(f)]
    Sewing = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_Sewing")
    sew = Sewing(VC.SEW_MM)
    for f in keep:
        sew.Add(f)
    sew.Perform()
    walls = sew.SewedShape()
    info = {"footprint_gap_deg": float(gap), "faces": len(faces), "periodic_faces": len(faces) - len(keep),
            "theta_ref_deg": float(np.degrees(th_ref)), "x_lo": x_lo, "x_hi": x_hi}
    return solid, walls, info


# --------------------------------------------------------------------------
# Checks and the plot
# --------------------------------------------------------------------------

def blade_deviation(gen, ref, rr):
    """(max, rms) distance (mm) of the reference blade's cut from the generated blade's at r/R rr."""
    r = S.cut_radius_mm(rr) if rr in S.RR else rr * S.R_MM
    p, q = S.unrolled(S.cylinder_cut(ref, r), r), S.unrolled(S.cylinder_cut(gen, r), r)
    q[:, 0] += r * 2.0 * np.pi * np.round((p[:, 0].mean() - q[:, 0].mean()) / (r * 2.0 * np.pi))
    d = S.distance_to_curve(p, q)
    return float(d.max()), float(np.sqrt(np.mean(d ** 2)))


def plot_check(path, blade, x0, theta_ref, x_iges, r_iges, design, x_shift, x_fwd, x_cut):
    """Side view (r against x, x across): hull.iges, the hull tool's hull, the hub and cap, the blade's extent at
    each radius; and the hull's radius against the file's."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(13, 8))
    FigureCanvasAgg(fig)
    ax1, ax2, ax3 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)
    xs = np.linspace(x_fwd, x_cut, 2000)
    r_fit = design.radius("r", (x_shift - xs) / 1000.0) * 1000.0
    k, (x_t, r_t), x_c, rho, x_tip = hub_profile(x_fwd)
    xh = np.linspace(x_tip, x_fwd, 600)
    rh = np.where(xh >= x_t, S.HUB_R_FWD - k * (x_fwd - xh), np.sqrt(np.clip(rho ** 2 - (xh - x_c) ** 2, 0, None)))
    eta = np.linspace(blade.eta_min, 1.0, 300)
    xi = np.linspace(0.0, 1.0, 721)
    pts = S.blade_to_cad(blade.b(xi[None, :], eta[:, None]), x0, theta_ref)
    rb = np.hypot(pts[..., 1], pts[..., 2]).mean(axis=1)
    for ax, lim in ((ax1, None), (ax2, (x_tip - 200.0, x_fwd + 1200.0))):
        ax.fill_betweenx(rb, pts[..., 0].min(axis=1), pts[..., 0].max(axis=1), color="#1b9e77", alpha=0.6, lw=0,
                         label="blade (its extent in x at each r)")
        ax.fill_between(xh, 0.0, rh, color="#d95f02", alpha=0.9, lw=0, label="hub and cap (F6)", zorder=3)
        ax.fill_between(xs, 0.0, r_fit, color="#c9d7ea", lw=0, zorder=2)
        ax.plot(x_iges, r_iges, "-", color="#9a9994", lw=3, label="hull.iges", zorder=4)
        ax.plot(xs, r_fit, "-", color="#2a78d6", lw=1.2, label="hull (hull_modify)", zorder=5)
        ax.set_xlabel("x (mm, the STP/IGES frame: towards the bow)")
        ax.set_ylabel("r (mm)")
        ax.set_aspect("equal")
        ax.grid(True, color="#e4e3df")
        if lim:
            ax.set_xlim(*lim)
            ax.set_ylim(0.0, 2600.0)
    ax1.legend(fontsize=8, loc="lower right")
    xi_ = x_iges[(x_iges >= x_fwd) & (x_iges <= x_cut)]
    ax3.plot(xi_, design.radius("r", (x_shift - xi_) / 1000.0) * 1000.0 - np.interp(xi_, x_iges, r_iges), "-",
             color="#2a78d6", lw=1)
    ax3.set_xlabel("x (mm)")
    ax3.set_ylabel("hull radius: hull_modify - hull.iges (mm)")
    ax3.grid(True, color="#e4e3df")
    fig.suptitle("BB2 aft half with the S7371R blade on the conic hub: side view (r against x)")
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
    prop_path, hull_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "outputs")
    t0 = time.time()

    # the hull file: the hub's forward end, the middle body, the cut, the profile
    hull_file = S.read_cad(hull_path)
    hull_shell = max(S._explore(hull_file, "SHELL"), key=lambda s: VC._bbox(s)[1, 0] - VC._bbox(s)[0, 0])
    x_fwd = S.hub_forward_x(hull_shell)
    x_pmb, r_pmb = middle_body(hull_shell)
    x_cut = float(VC._bbox(hull_shell)[1, 0])
    x_iges, r_iges = hull_profile(hull_shell)
    print(f"hull.iges: hub forward end x {x_fwd:.2f} mm, middle body {r_pmb:.3f} mm from x {x_pmb:.3f} mm, "
          f"cut at x {x_cut:.3f} mm ({time.time() - t0:.0f} s)")

    # the blade
    blade, x0, theta_ref = blade_surface(out)
    k, (x_t, r_t), x_c, rho, x_tip = hub_profile(x_fwd)
    ring = S.blade_to_cad(blade.b(np.linspace(0, 1, 721), blade.eta_min), x0, theta_ref)
    cone = S.HUB_R_FWD - k * (x_fwd - ring[:, 0])
    print(f"hub: cone slope {k:.5f}, nose sphere radius {rho:.1f} mm from r {r_t:.1f} mm, tip x {x_tip:.2f}; "
          f"blade root ring r {1000 * blade.r_root * blade.d / 2:.1f} mm, x {ring[:, 0].min():.1f} .. "
          f"{ring[:, 0].max():.1f}, at least {np.min(cone - np.hypot(ring[:, 1], ring[:, 2])):.1f} mm inside "
          f"the hub")
    t1 = time.time()
    blade_shape = blade_cad(blade, x0, theta_ref, out)
    print(f"blade CAD (DRDC grids, X_CAD_new, closed at the root): {time.time() - t1:.0f} s, "
          f"{VC.solid_report(blade_shape)}")
    hub = hub_solid(x_fwd)
    hub_blade = VC.fuse([hub, blade_shape])
    print(f"hub and blade: solids, valid, volume mm^3 {VC.solid_report(hub_blade)}")
    XL.write_step(hub_blade, os.path.join(out, "bb2_blade_hub.step"))
    XL.write_iges(hub_blade, os.path.join(out, "bb2_blade_hub.iges"))

    # the blade against prop.stp's blade 1
    ref = S._explore(S.read_cad(prop_path), "SHELL")[0]
    print(" r/R     blade from the files against prop.stp's blade 1: mm max / rms")
    for rr in sorted(set(S.RR) | set(CHECK_RR)):
        a = blade_deviation(blade_shape, ref, rr)
        print(f"{rr:.4f}{'' if rr in S.RR else ' (between the tables)':22s} {a[0]:6.2f} / {a[1]:5.2f}")

    # the hull
    params = hull_params(x_iges, r_iges, x_fwd, x_pmb, r_pmb, x_cut)
    HM.write_params(os.path.join(out, "bb2_hull_params.dat"), params)
    params = HM.read_params(os.path.join(out, "bb2_hull_params.dat"))
    x_shift = x_pmb + 1000.0 * (params.nose + params.middle)
    design = fit_hull(params, x_iges, r_iges, x_fwd, x_pmb, x_shift)
    problems = HM.design_problems(design)
    if problems:
        raise RuntimeError(f"hull design: {problems}")
    hull = hull_solid(design, x_shift, x_cut)
    print(f"hull (tail of order {design.curves['tail_r'].order}): solids, valid, volume mm^3 "
          f"{VC.solid_report(hull)}")
    XL.write_step(hull, os.path.join(out, "bb2_hull.step"))
    XL.write_iges(hull, os.path.join(out, "bb2_hull.iges"))

    # the vehicle
    vehicle = VC.fuse([hull, hub_blade])
    print(f"vehicle: solids, valid, volume mm^3 {VC.solid_report(vehicle)}")
    XL.write_step(vehicle, os.path.join(out, "bb2_vehicle.step"))
    XL.write_iges(vehicle, os.path.join(out, "bb2_vehicle.iges"))

    # six blades at prop.stp's positions
    blades = [VC.transformed(blade_shape, VC.turn_x(np.radians(a)), np.zeros(3)) for a in BLADE_TURNS_DEG]
    prop6 = VC.fuse([hub] + blades)
    print(f"hub and six blades: solids, valid, volume mm^3 {VC.solid_report(prop6)}")
    XL.write_step(prop6, os.path.join(out, "bb2_prop6.step"))
    XL.write_iges(prop6, os.path.join(out, "bb2_prop6.iges"))
    vehicle6 = VC.fuse([hull, prop6])
    print(f"vehicle with six blades: solids, valid, volume mm^3 {VC.solid_report(vehicle6)}")
    XL.write_step(vehicle6, os.path.join(out, "bb2_vehicle6.step"))
    XL.write_iges(vehicle6, os.path.join(out, "bb2_vehicle6.iges"))

    # one blade on its 1/6 of the hub
    sector, walls, info = blade_sector(blade_shape, hub, x0, theta_ref, x_fwd)
    print(f"one blade on 1/6 of the hub: solids, valid, volume mm^3 {VC.solid_report(sector)} (six of them "
          f"{6.0 * VC.volume(sector):.6e}, the six-blade propeller {VC.volume(prop6):.6e}); {info['faces']} faces, "
          f"{info['periodic_faces']} periodic; the footprint {info['footprint_gap_deg']:.1f} deg from the strip's "
          f"edges")
    XL.write_step(sector, os.path.join(out, "bb2_blade_sector.step"))
    write_iges_solids(sector, os.path.join(out, "bb2_blade_sector.iges"))
    XL.write_step(walls, os.path.join(out, "bb2_blade_sector_walls.step"))
    XL.write_iges(walls, os.path.join(out, "bb2_blade_sector_walls.iges"))

    # the quadrant of hull.iges: the hull, the hull with the hub (no blade)
    hull_hub = VC.fuse([hull, hub])
    print(f"hull with the hub: solids, valid, volume mm^3 {VC.solid_report(hull_hub)}")
    XL.write_step(hull_hub, os.path.join(out, "bb2_hull_hub.step"))
    for name, shape in (("hull", hull), ("hull_hub", hull_hub)):
        q = quadrant(shape)
        print(f"{name} in the quadrant y >= 0, z >= 0: solids, valid, volume mm^3 {VC.solid_report(q)}")
        write_iges_solids(q, os.path.join(out, f"bb2_{name}_quadrant.iges"))
    plot_check(os.path.join(out, "bb2_build_check.png"), blade, x0, theta_ref, x_iges, r_iges, design, x_shift,
               x_fwd, x_cut)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
