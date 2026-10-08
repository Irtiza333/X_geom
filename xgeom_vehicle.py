"""
xgeom_vehicle.py

The vehicle in the XGeom design tool (xgeom_tool.py, its Vehicle tab): the hull (xgeom_hull) with the propeller
(xgeom_blade) and the rudder (xgeom_rudder) mounted at its stern, each drawn from its adapter's design as it
stands. The frame is the hull's: x from the nose aft (m), y to starboard, z up.

Placement (Placement, the Vehicle tab's settings):
    prop_dx        the propeller plane (the middle of the hub, where the blades sit) on the axis, from the end of
                   the hull: x - L (m, negative forward); default 0, the end
    prop_d         the propeller's diameter in the vehicle (m): the blade design is scaled to it; default half the
                   hull's depth or breadth, the larger
    rudder_dx      the rudder root's trailing edge from the end of the hull: x - L (m); default the end of the tail
                   (the start of the cap; with no cap 5 % of L ahead of the end), further forward if the blades
                   reach there: a quarter of the propeller's diameter clear of them
    rudder_angle   around the axis from the top towards starboard (deg): 0 above, 90 to starboard, 180 under the
                   stern (default)
    rudders        1, 2 (the second opposite the first) or 4 (cruciform, 90 deg apart)
    rudder_scale   the rudder (mm) to the hull's metres: 1 keeps its size (default)
    view           "vehicle" the whole of it, "stern" from the start of the tail
The propeller's frame (para.py's) has x downstream, as the hull's, so it is moved and scaled only. The rudder's
frame (x chordwise from the LE, y the height from the root, z the thickness) is turned so that its height points
away from the axis at the angle; its root sits on the hull at the hull's distance from the axis in that direction,
the smallest along the root chord, so no part of the root stands off the hull.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import hull_modify as HM
from xgeom_common import BLUE, GREY, INK2, ORANGE, fit_box, keep_fitted
from xgeom_hull import short_ticks

COLLECTIONS = (("hull", GREY, 0.5), ("hub", INK2, 0.8), ("blades", BLUE, 0.6), ("rudders", ORANGE, 0.6))
LOOPS = 30                       # rudder loops drawn, evenly from the root to the cap
BLADE_EVERY = 4                  # every 4th section of the XCAD points file (and the last) per blade
OWNS = {"hull": ("hull",), "blade": ("blades", "hub"), "rudder": ("rudders",)}   # a component's lines


@dataclass
class Placement:
    prop_dx: float = 0.0
    prop_d: float = 0.25
    rudder_dx: float = -0.1
    rudder_angle: float = 180.0
    rudders: int = 1
    rudder_scale: float = 1.0
    view: str = "vehicle"


# --------------------------------------------------------------------------
# Each component's lines in its own frame, from its adapter
# --------------------------------------------------------------------------

def hull_part(ad, n=13, k=73, lines=8):
    """The hull (m): sections at n stations per curved part and `lines` lines along it."""
    d = ad.design
    secs = [np.vstack((s, s[:1])) for s in HM.section_points(d, HM.stations_x(d, n), k - 1)]
    x = HM.stations_x(d, 101)
    r, rp = d.radius("r", x), d.radius("rp", x)
    along = [np.column_stack((x, rp * np.cos(t), r * np.sin(t)))
             for t in np.linspace(0.0, 2.0 * np.pi, int(lines), endpoint=False)]
    return {"design": d, "lines": secs + along, "problems": HM.design_problems(d)}


def blade_part(ad, every=BLADE_EVERY):
    """One blade (m, the MSc frame): para.py's sections at every few stations of the XCAD points file and the
    last; the blade count, the diameter, the hub's radius and height; the design's problems."""
    import blade_modify as BM
    d, p = ad.design, ad.params
    out = {"design": d, "blades": int(p.blades), "diameter": float(d.diameter),
           "hub": (float(d.hub_radius), float(d.hub_height)), "sections": [], "problems": []}
    try:
        st = ad.stations()
        st = np.unique(np.r_[st[::every], st[-1]])
        out["sections"] = list(BM.blade_points(d, p, st)[0].reshape(-1, 53, 3))
        out["problems"] = BM.design_problems(d, p, ad.settings["clearance_mm"], hub=ad.settings["hub"])
    except Exception as exc:
        out["problems"].append(f"{type(exc).__name__}: {exc}")
    return out


def rudder_part(ad, n=LOOPS, every=2):
    """The rudder (mm, its working frame: x chordwise, y the height, z the thickness): n of the loops of its
    3D sections tab (modified below H, the cut, the original above with the tip and the cap), evenly from the
    root to the top, every other point; the root's x range (the lowest loop); the design's problems."""
    import rudder_modify as RM
    d, s = ad.design, ad.settings
    problems = RM.design_problems(d, ad.orig, s["n_sections"], s["te_radius_mm"])
    sk = ad.skeleton(problems)
    loops = sorted(sk["modified"] + sk["cut"] + sk["unchanged"], key=lambda lp: float(lp[:, 1].mean()))
    keep = np.unique(np.linspace(0, len(loops) - 1, min(n, len(loops))).round().astype(int))
    loops = [np.vstack((loops[i][::every], loops[i][:1])) for i in keep]
    root = loops[0]
    return {"design": d, "loops": loops, "root_x": (float(root[:, 0].min()), float(root[:, 0].max())),
            "problems": problems}


PARTS = {"hull": hull_part, "blade": blade_part, "rudder": rudder_part}


# --------------------------------------------------------------------------
# Placing them
# --------------------------------------------------------------------------

def surface_distance(design, x, phi):
    """Distance from the axis to the hull's surface at x (m) in the direction phi (rad, from +z towards +y):
    the elliptical section's r r' / sqrt((r sin phi)^2 + (r' cos phi)^2); 0 off the body."""
    r, rp = design.radius("r", x), design.radius("rp", x)
    den = np.hypot(r * np.sin(phi), rp * np.cos(phi))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0.0, r * rp / np.where(den > 0.0, den, 1.0), 0.0)


def _turn_x(angle):
    """Rotation about x for row vectors (p @ M)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def default_placement(parts):
    """The placement of the docstring's defaults for these parts."""
    d = parts["hull"]["design"]
    pl = Placement(prop_d=round(max(d.r, d.rp if d.rp is not None else d.r), 3))
    te = -d.cap if d.cap > 0 else -0.05 * d.length
    b = parts.get("blade")
    if b and b["sections"]:
        front = pl.prop_dx + pl.prop_d / b["diameter"] * min(float(sec[:, 0].min()) for sec in b["sections"])
        te = min(te, front - 0.25 * pl.prop_d)
    pl.rudder_dx = round(te, 3)
    return pl


def assemble(parts, pl):
    """The vehicle's lines in the hull's frame (m), by collection ('hull', 'hub', 'blades', 'rudders'), and
    what the view's text says (info)."""
    d = parts["hull"]["design"]
    L = d.length
    segs = {"hull": list(parts["hull"]["lines"]), "hub": [], "blades": [], "rudders": []}
    info = {"length": L}
    b = parts.get("blade")
    if b:
        s = pl.prop_d / b["diameter"]
        x0 = L + pl.prop_dx
        shift = np.array([x0, 0.0, 0.0])
        for k in range(b["blades"]):
            turn = _turn_x(2.0 * np.pi * k / b["blades"])
            segs["blades"] += [s * sec @ turn + shift for sec in b["sections"]]
        hr, hh = s * b["hub"][0], s * b["hub"][1]
        t = np.linspace(0.0, 2.0 * np.pi, 73)
        for x in (x0 - 0.5 * hh, x0 + 0.5 * hh):
            segs["hub"].append(np.column_stack((np.full_like(t, x), hr * np.sin(t), hr * np.cos(t))))
        for a in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            segs["hub"].append(np.array([[x0 - 0.5 * hh, hr * np.sin(a), hr * np.cos(a)],
                                         [x0 + 0.5 * hh, hr * np.sin(a), hr * np.cos(a)]]))
        info.update(prop_x=x0, prop_scale=s, prop_d=pl.prop_d, design_d=b["diameter"], blades=b["blades"])
    ru = parts.get("rudder")
    if ru:
        s = pl.rudder_scale / 1000.0
        xa, xb = ru["root_x"]
        shift = L + pl.rudder_dx - s * xb                     # the hull's x of the rudder frame's x = 0
        xr = shift + s * np.linspace(xa, xb, 201)             # the root chord along the hull
        ex = np.array([1.0, 0.0, 0.0])
        rho = []
        for k in range(int(pl.rudders)):
            phi = np.deg2rad(pl.rudder_angle + 360.0 * k / pl.rudders)
            rho.append(float(surface_distance(d, xr, phi).min()))
            er = np.array([0.0, np.sin(phi), np.cos(phi)])
            et = np.array([0.0, -np.cos(phi), np.sin(phi)])
            for lp in ru["loops"]:
                segs["rudders"].append(np.outer(shift + s * lp[:, 0], ex) + np.outer(rho[-1] + s * lp[:, 1], er)
                                       + np.outer(s * lp[:, 2], et))
        info.update(rudder_le=xr[0], rudder_te=xr[-1], rudder_root=rho)
    if segs["blades"] and segs["rudders"]:                     # the rudders' aft end inside the propeller's radius
        front = min(float(p[:, 0].min()) for p in segs["blades"])
        q = np.vstack(segs["rudders"])
        near = q[np.hypot(q[:, 1], q[:, 2]) <= 0.5 * pl.prop_d]
        info["gap"] = front - float(near[:, 0].max()) if len(near) else None
    return segs, info


# --------------------------------------------------------------------------
# The view
# --------------------------------------------------------------------------

class VehicleView:
    """The Vehicle tab's 3D view: the hull grey, the propeller blue with its hub, the rudders orange; the
    component being edited thicker."""

    def __init__(self, fig):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        self.fig = fig
        self.ax = ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.col = {k: Line3DCollection([np.zeros((2, 3))], colors=c, linewidths=w) for k, c, w in COLLECTIONS}
        self.width = {k: w for k, _, w in COLLECTIONS}
        for c in self.col.values():
            c.set_clip_on(False)              # a long vehicle overruns the 3D axes' square (fit_box fits it)
            ax.add_collection3d(c)
        from matplotlib.ticker import MaxNLocator
        ax.zaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel("x (m)", labelpad=10)
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=16, azim=-66)
        self.text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=9, va="top")
        self.box = None
        keep_fitted(fig, ax, lambda: self.box)

    def update(self, parts, pl, current=None):
        segs, info = assemble(parts, pl)
        d = parts["hull"]["design"]
        if pl.view == "stern":            # from the start of the tail; 3D lines are not clipped, so they are cut
            x0 = min([d.joins[2]] + [float(p[:, 0].min()) for k in ("blades", "hub", "rudders") for p in segs[k]])
            x0 -= 0.02 * d.length
            segs = {k: [p[p[:, 0] >= x0] for p in v] for k, v in segs.items()}
            segs = {k: [p for p in v if len(p) > 1] for k, v in segs.items()}
        for k, c in self.col.items():
            c.set_segments(segs[k] or [np.zeros((2, 3))])
            c.set_linewidth(self.width[k] * (2.0 if current and k in OWNS.get(current, ()) else 1.0))
        allp = np.vstack([p for v in segs.values() for p in v])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        pad = 0.02 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax = self.ax
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        short_ticks(ax.yaxis, max(-lo[1], hi[1], 1e-9) / 1.02)
        self.box = np.maximum(hi - lo, 1e-9)
        fit_box(ax, self.fig, self.box)
        size = f"D {2 * d.r:.3f} m" if d.axisymmetric else f"depth {2 * d.r:.3f} m, breadth {2 * d.rp:.3f} m"
        lines = [f"hull (grey): L {d.length:.3f} m, {size}"]
        if "prop_x" in info:
            lines.append(f"propeller (blue): {info['blades']} blades, D {info['prop_d']:.3f} m (the design's "
                         f"{info['design_d']:.3g} m x {info['prop_scale']:.3f}), plane at x {info['prop_x']:.3f} m")
        if "rudder_te" in info:
            n = len(info["rudder_root"])
            rho = info["rudder_root"]
            roots = (f"{1000 * rho[0]:.1f}" if max(rho) - min(rho) < 5e-5 else
                     ", ".join(f"{1000 * r:.1f}" for r in rho))
            lines.append(f"rudder{'s' if n > 1 else ''} (orange): root from x {info['rudder_le']:.3f} to "
                         f"{info['rudder_te']:.3f} m, on the hull at "
                         + ", ".join(f"{(pl.rudder_angle + 360.0 * k / n) % 360:g}" for k in range(n))
                         + f" deg, {roots} mm from the axis")
            gap = info.get("gap")
            if gap is not None:
                lines.append(f"rudder{'s' if n > 1 else ''} to the blades (within their radius): "
                             + (f"{1000 * gap:.0f} mm clear" if gap >= 0 else f"{-1000 * gap:.0f} mm overlap in x"))
        lines.append("the component being edited is drawn thicker; drag to turn")
        self.text.set_text("\n".join(lines))
        return info
