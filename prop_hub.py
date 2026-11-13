"""
prop_hub.py

The propeller hub of the XGeom blade, in two forms:

  cylinder   the DRDC hub of hub_new: the radius of the blade root ring, a
             height, centred at x = 0 (the generator line); flat ends.
  cone       (13 Nov 2026, the user's request, the hub of BB2's drawing F6)
             a cone from the base radius, the larger end upstream at the base
             plane x_base, to the top radius, closed there by a rounded nose:
             the sphere tangent to the cone that passes through the tip on the
             axis, height from the base. The three numbers base, top and
             height fix the nose. With a the cone's half angle the sphere's
             radius is rho = top / cos a (tangent at the top radius), its
             centre lies rho from the tip, and the cone passes through the
             tangent point:

                 height tan a - top sin a / (1 + sin a) = base - top,

             one root a in [0, pi/2) for top <= base (top = base: a cylinder
             with a hemispherical nose). The tangent point lies
             height - rho (1 - sin a) from the base, which must not be
             negative: the height is at least top sqrt((1 - sin a) / (1 + sin
             a)) (a nose on its own). The blade root ring (the table's first
             row, r = root_r R) lies inside the cone where the hub is wider
             and outside where it is narrower, so the blade is extended below
             its root into the hub (blade_modify.root_extension) and the hub
             trims it.

The frame is the blade frame (para.py, X_CAD_new): x the axis, downstream
positive, the hub's base upstream of its tip. Units are the caller's: the
design holds metres, the CAD millimetres (HubProfile.scaled).

    hub = HubProfile(0.7972, 0.1380, 1.5688, x_base=-0.5761)     # m: BB2's hub
    hub.radius(x)                 the hub's radius at x (0 outside the hub)
    hub.outline(n)                (x, r) along the profile, the base to the tip
    hub.nose                      (x of the sphere's centre, its radius)
    hub.x_tangent                 where the cone ends and the nose begins

CAD (pythonocc-core or cadquery-ocp through xcad_loft's shim; mm):
    hub_solid(hub)                the hub as a solid of revolution
    propeller_solid(blade, hub, Z, k=Z)
                                  k of the Z blades (the blade solid of
                                  vehicle_cad.blade_solid, its root inside the
                                  hub) fused with the hub, or with the k/Z
                                  periodic sector of it (k < Z): TM 2013-178
                                  Sec. 10's wedge, theta within p_h(x) - pi/Z
                                  .. p_h(x) + (2k - 1) pi/Z about the mid-line
                                  of the first blade's footprint on the hub
                                  (hub_new._centred_centre_curve), the other
                                  blades turned towards larger theta (theta =
                                  atan2(y, z); vehicle_cad.turn_x turns y
                                  towards z, which is towards smaller theta,
                                  so the turns are negative), so k = 1 is
                                  one blade on its 1/Z of the hub and k = Z/2
                                  half the propeller; Z turned copies of the
                                  1/Z sector tile the whole.
    write_iges_solids(shape, path) IGES in BRep mode (solids with their trimmed
                                  faces), mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECTOR_N = (41, 41, 321)      # the periodic wedge's grids: across the radius, across the sector, along the axis
SECTOR_PAD = 0.05             # the wedge reaches this fraction of the hub's height beyond its ends
SECTOR_OUT = 1.2              # ... and this times the hub's largest radius out from the axis
FUZZY_MM = 1.0e-4             # the booleans' fuzzy value


@dataclass
class HubProfile:
    """The hub's profile: base (the radius at the base plane), top (the cone's end radius; None: a cylinder of
    radius base), height (the base to the tip), x_base (the base plane on the axis; None: centred at 0)."""
    base: float
    top: float = None
    height: float = 1.0
    x_base: float = None

    def __post_init__(self):
        self.base, self.height = float(self.base), float(self.height)
        self.top = None if self.top is None else float(self.top)
        if self.x_base is None:
            self.x_base = -0.5 * self.height
        self.x_base = float(self.x_base)
        if not self.base > 0.0 or not self.height > 0.0:
            raise ValueError("the hub's base radius and height are positive")
        if self.top is not None:
            if not 0.0 < self.top <= self.base:
                raise ValueError(f"the hub's top radius ({self.top:.6g}) must lie between 0 and the base radius "
                                 f"({self.base:.6g})")
            self._solve()

    def _solve(self):
        """The cone's half angle a, the nose sphere and the tangent point (see the module docstring)."""
        from scipy.optimize import brentq
        b, t, h = self.base, self.top, self.height

        def f(a):
            return h * np.tan(a) - t * np.sin(a) / (1.0 + np.sin(a)) - (b - t)
        if b - t < 1e-12 * b:
            a = 0.0
        else:
            a = brentq(f, 0.0, 0.5 * np.pi - 1e-9)
        rho = t / np.cos(a)
        z_t = h - rho * (1.0 - np.sin(a))                  # the tangent point from the base
        if z_t < -1e-9 * h:
            least = t * np.sqrt((1.0 - np.sin(a)) / (1.0 + np.sin(a)))
            raise ValueError(f"the hub's height ({h:.6g}) is too short for its rounded nose: at least "
                             f"{least:.6g} for a top radius of {t:.6g}")
        self.half_angle = float(a)
        self.slope = float(np.tan(a))
        self.rho = float(rho)
        self.x_centre = self.x_base + h - rho
        self.x_tangent = self.x_base + max(z_t, 0.0)

    # ---------------------------------------------------------------- shape
    @property
    def cylinder(self):
        return self.top is None

    @property
    def shape(self):
        return "cylinder" if self.cylinder else "cone"

    @property
    def x_tip(self):
        return self.x_base + self.height

    @property
    def nose(self):
        """(x of the sphere's centre, its radius); None for the cylinder."""
        return None if self.cylinder else (self.x_centre, self.rho)

    @property
    def r_max(self):
        return self.base

    def radius(self, x):
        """The hub's radius at x (any array shape); 0 outside the hub."""
        x = np.asarray(x, dtype=float)
        if self.cylinder:
            return np.where((x >= self.x_base) & (x <= self.x_tip), self.base, 0.0)
        cone = self.base - self.slope * (x - self.x_base)
        sph = np.sqrt(np.clip(self.rho ** 2 - (x - self.x_centre) ** 2, 0.0, None))
        r = np.where(x <= self.x_tangent, cone, sph)
        return np.where((x >= self.x_base) & (x <= self.x_tip), r, 0.0)

    def outline(self, n=201):
        """(x, r) along the profile from the base plane to the tip: the base (x_base, 0) up the base plane, the
        cone (or the cylinder) and the nose; n points on the curved part."""
        if self.cylinder:
            return np.array([[self.x_base, 0.0], [self.x_base, self.base], [self.x_tip, self.base], [self.x_tip, 0.0]])
        n_arc = max(8, n // 4)
        xc = np.linspace(self.x_base, self.x_tangent, max(2, n - n_arc))
        a0 = np.arctan2(self.top, self.x_tangent - self.x_centre)
        ang = np.linspace(a0, 0.0, n_arc + 1)[1:]
        xs = self.x_centre + self.rho * np.cos(ang)
        rs = self.rho * np.sin(ang)
        return np.vstack(([[self.x_base, 0.0]], np.column_stack((xc, self.radius(xc))), np.column_stack((xs, rs))))

    def scaled(self, k):
        """The same hub in other units (k: the factor, 1000 for m to mm)."""
        return HubProfile(k * self.base, None if self.cylinder else k * self.top, k * self.height, k * self.x_base)

    def covers(self, x_lo, x_hi, margin=0.0):
        """Does the hub reach margin beyond x_lo .. x_hi along the axis?"""
        return self.x_base < x_lo - margin and x_hi + margin < self.x_tip

    def min_radius(self, x_lo, x_hi, n=401):
        """The hub's smallest radius over x_lo .. x_hi (0 where the hub is not)."""
        return float(self.radius(np.linspace(x_lo, x_hi, n)).min())

    def describe(self, unit="m"):
        if self.cylinder:
            return f"cylinder, radius {self.base:.4g} {unit}, {self.height:.4g} {unit} long from x {self.x_base:.4g}"
        return (f"cone, base radius {self.base:.4g} {unit} at x {self.x_base:.4g}, top radius {self.top:.4g} {unit} "
                f"{self.x_tangent - self.x_base:.4g} {unit} aft of it (half angle {np.degrees(self.half_angle):.2f} "
                f"deg), the nose a sphere of radius {self.rho:.4g} {unit} to the tip {self.height:.4g} {unit} from "
                f"the base")


# --------------------------------------------------------------------------
# CAD (mm)
# --------------------------------------------------------------------------

def _xl():
    import xcad_loft as XL
    return XL


def hub_solid(hub):
    """The hub (HubProfile in mm) as a solid of revolution about x: the cylinder from BRepPrimAPI, the cone
    with its nose the profile (the base plane, the cone, the nose arc, the axis) turned about x."""
    XL = _xl()
    gp_Pnt, gp_Ax1, gp_Ax2, gp_Dir = XL._occ("gp", "gp_Pnt", "gp_Ax1", "gp_Ax2", "gp_Dir")
    if hub.cylinder:
        BRepPrimAPI_MakeCylinder = XL._occ("BRepPrimAPI", "BRepPrimAPI_MakeCylinder")
        return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(hub.x_base, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), hub.base,
                                        hub.height).Solid()
    MakeEdge, MakeWire, MakeFace = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_MakeEdge", "BRepBuilderAPI_MakeWire",
                                           "BRepBuilderAPI_MakeFace")
    GC_MakeArcOfCircle = XL._occ("GC", "GC_MakeArcOfCircle")
    MakeRevol = XL._occ("BRepPrimAPI", "BRepPrimAPI_MakeRevol")
    x_c, rho = hub.nose
    p0, p1 = gp_Pnt(hub.x_base, 0.0, 0.0), gp_Pnt(hub.x_base, hub.base, 0.0)
    pt, ptip = gp_Pnt(hub.x_tangent, hub.top, 0.0), gp_Pnt(hub.x_tip, 0.0, 0.0)
    a = 0.5 * np.arctan2(hub.top, hub.x_tangent - x_c)            # the arc's middle point
    pm = gp_Pnt(x_c + rho * np.cos(a), rho * np.sin(a), 0.0)
    arc = MakeEdge(GC_MakeArcOfCircle(pt, pm, ptip).Value()).Edge()
    edges = [MakeEdge(p0, p1).Edge()]
    if hub.x_tangent - hub.x_base > 1e-9 * hub.height:
        edges.append(MakeEdge(p1, pt).Edge())
    edges += [arc, MakeEdge(ptip, p0).Edge()]
    wire = MakeWire()
    for e in edges:
        wire.Add(e)
    face = MakeFace(wire.Wire(), True).Face()
    return MakeRevol(face, gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), 2.0 * np.pi).Shape()


def _boolean(name, a, b, fuzzy=FUZZY_MM):
    XL = _xl()
    op = XL._occ("BRepAlgoAPI", f"BRepAlgoAPI_{name}")(a, b)
    try:
        op.SetFuzzyValue(float(fuzzy))
        op.Build()
    except Exception:
        pass
    if not op.IsDone():
        raise RuntimeError(f"the hub's boolean {name} failed")
    return op.Shape()


def _polyline(edge, n):
    XL = _xl()
    BRepAdaptor_Curve = XL._occ("BRepAdaptor", "BRepAdaptor_Curve")
    c = BRepAdaptor_Curve(edge)
    ts = np.linspace(c.FirstParameter(), c.LastParameter(), n)
    return np.array([[c.Value(float(t)).X(), c.Value(float(t)).Y(), c.Value(float(t)).Z()] for t in ts])


def footprint(blade_out, n=400):
    """Points (mm) of the blade's footprint on the hub: the boundary of the faces the hub's surface left on
    the blade cut by it (the hub's analytic faces, not the blade's B-spline ones)."""
    XL = _xl()
    GeomAdaptor_Surface = XL._occ("GeomAdaptor", "GeomAdaptor_Surface")
    GeomAbs_BSplineSurface = XL._occ("GeomAbs", "GeomAbs_BSplineSurface")
    surface = XL._static(XL._occ("BRep", "BRep_Tool"), "Surface")
    import vehicle_cad as VC
    faces = [f for f in VC._shapes(blade_out, "FACE")
             if GeomAdaptor_Surface(surface(f)).GetType() != GeomAbs_BSplineSurface]
    if not faces:
        raise RuntimeError("the blade cut by the hub has no face on the hub: does its root reach into the hub?")
    return np.vstack([_polyline(e, n) for f in faces for e in VC._shapes(f, "EDGE")])


def sector_wedge(p_h, th_ref, x_lo, x_hi, r_out, n_blades, k=1):
    """The solid wedge theta in p_h(x) - pi/Z .. p_h(x) + (2k - 1) pi/Z about the axis (hub_new's angle:
    atan2(y, z) - th_ref), x_lo .. x_hi, radius 0 .. r_out (mm): its two periodic sides, outer face and end
    caps from grids sharing their edges point for point (X_CAD_new.grid_to_bspline_surface), sewn."""
    from X_CAD_new import grid_to_bspline_surface
    import vehicle_cad as VC
    XL = _xl()
    n_r, n_s, n_x = SECTOR_N
    MakeFace = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace")
    x = np.linspace(x_lo, x_hi, n_x)
    s = np.linspace(0.0, 1.0, n_s)
    rho = np.linspace(0.0, r_out, n_r)
    ph = lambda xx: th_ref + p_h(np.clip(xx, p_h.x[0], p_h.x[-1]))
    lo, hi = -np.pi / n_blades, (2 * k - 1) * np.pi / n_blades

    def point(xx, rr, th):
        return np.stack(np.broadcast_arrays(xx, rr * np.sin(th), rr * np.cos(th)), axis=-1)

    grids = [point(x[None, :], rho[:, None], (ph(x) + lo)[None, :]),
             point(x[None, :], rho[:, None], (ph(x) + hi)[None, :]),
             point(x[None, :], r_out, ph(x)[None, :] + (lo + (hi - lo) * s)[:, None])]
    for xe in (x_lo, x_hi):
        grids.append(point(xe, rho[:, None], (ph(xe) + lo + (hi - lo) * s)[None, :]))
    faces = [MakeFace(grid_to_bspline_surface(g / 1000.0), 1.0e-6).Face() for g in grids]
    solid, free = VC._sewn_solid(faces)
    if free:
        raise RuntimeError(f"the periodic wedge has {free} free edges")
    return solid


def propeller_solid(blade, hub, n_blades, k=None):
    """k of the Z blades on the hub (mm, the blade frame): the whole propeller (k = Z or None: Z copies of the
    blade solid fused with the hub) or the periodic k/Z sector (the blade cut by the hub, its footprint's
    mid-line the centre curve p_h, the hub within the wedge of sector_wedge, the k blades turned 2 pi j / Z
    towards larger theta, all fused). blade: vehicle_cad.blade_solid's solid, its root ring inside the hub
    (or on the cylinder's surface). Returns (solid, info): info has the footprint's clearance from the
    sector's edges (deg) and the number of faces."""
    import vehicle_cad as VC
    Z = int(n_blades)
    k = Z if k is None else int(k)
    if not 1 <= k <= Z:
        raise ValueError(f"k blades out of {Z}: k must lie in 1 .. {Z}")
    solid_hub = hub_solid(hub)
    turns = [VC.transformed(blade, VC.turn_x(-2.0 * np.pi * j / Z), np.zeros(3)) for j in range(k)]  # towards
    if k == Z:
        return _one_solid(VC.fuse([solid_hub] + turns), "the propeller"), {"blades": Z}
    import hub_new
    blade_out = _boolean("Cut", blade, solid_hub) if not hub.cylinder else blade
    fp = footprint(blade_out) if not hub.cylinder else _root_ring(blade)
    th_fp = np.arctan2(fp[:, 1], fp[:, 2])
    th_ref = float(np.arctan2(np.sin(th_fp).mean(), np.cos(th_fp).mean()))
    p_h = hub_new._centred_centre_curve(fp, th_ref, hub.x_base, hub.x_tip)[0]
    rel = np.arctan2(np.sin(th_fp - th_ref), np.cos(th_fp - th_ref)) - p_h(fp[:, 0])
    gap = float(np.degrees(np.pi / Z - np.abs(rel).max()))
    if gap < hub_new.FOOT_MARGIN_DEG:
        raise RuntimeError(f"the blade's footprint on the hub clears its sector's edges by {gap:.2f} deg only "
                           f"(at least {hub_new.FOOT_MARGIN_DEG:g} deg): with {Z} blades they overlap at the root")
    pad = SECTOR_PAD * hub.height
    wedge = sector_wedge(p_h, th_ref, hub.x_base - pad, hub.x_tip + pad, SECTOR_OUT * hub.r_max, Z, k)
    hub_sector = _boolean("Common", solid_hub, wedge)
    parts = [VC.transformed(blade_out, VC.turn_x(-2.0 * np.pi * j / Z), np.zeros(3)) for j in range(k)]  # larger
    solid = _one_solid(VC.fuse([hub_sector] + parts), f"the {k}/{Z} sector")
    return solid, {"footprint_gap_deg": gap, "blades": k, "theta_ref_deg": float(np.degrees(th_ref))}


def _one_solid(shape, what):
    """The shape, which must be one solid (a fuse that left pieces apart is an error, not a result)."""
    import vehicle_cad as VC
    n = len(VC._shapes(shape, "SOLID"))
    if n != 1:
        raise RuntimeError(f"{what} came out as {n} solids, not one: the blades and the hub did not fuse")
    return shape


def _root_ring(blade, n=400):
    """The blade solid's root ring (mm): the boundary of its face on the hub cylinder (the non-B-spline face)."""
    return footprint(blade, n)


def write_iges_solids(shape, path):
    """IGES in BRep mode (manifold solids, shells, faces with their trim loops), mm."""
    XL = _xl()
    IGESControl_Writer = XL._occ("IGESControl", "IGESControl_Writer")
    w = IGESControl_Writer("MM", 1)
    w.AddShape(shape)
    w.ComputeModel()
    if not w.Write(str(path)):
        raise IOError(f"cannot write {path}")
    return path
