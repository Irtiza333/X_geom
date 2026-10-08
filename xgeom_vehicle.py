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
    rudder_dx      where the rudder root's trailing edge sits on the hull, from its end: x - L (m, -L to 0);
                   default the end of the tail (the start of the cap; with no cap 5 % of L ahead of the end),
                   further forward if the blades reach there: a quarter of the propeller's diameter clear of them
    rudder_angle   the first rudder's, around the axis from the top towards starboard (deg): 0 above, 90 to
                   starboard, 180 under the stern (default)
    rudders        how many, 1 to MAX_RUDDERS (12), evenly spaced around the axis (360/N deg apart); default 4,
                   a cross with one under the stern
    rudder_scale   the rudder (mm) to the hull's metres: 1 keeps its size (default)
    view           "vehicle" the whole of it, "stern" from the start of the tail
The propeller's frame (para.py's) has x downstream, as the hull's, so it is moved and scaled only. The rudder's
frame (x chordwise from the LE, y the height from the root, z the thickness) is turned so that its height points
away from the axis at its angle, and pitched in that half-plane so that its root follows the hull (place_rudder):
the root's TE sits on the hull at the x set and its LE on the hull one root chord ahead, along the hull's profile
at the rudder's angle (root_on_hull). The pitch, the angle of the root chord to the axis, is the hull's slope
under the root; it follows whenever the rudder is moved along x or scaled, or the hull changes. The flat root
still leaves gaps where the hull curves away under it (across its thickness, and along the chord where the
profile is concave): a copy of the root section is extruded inwards, along the rudder's height, by the largest
of them plus ROOT_OVERLAP of the root chord (root_gaps), so the rudder meets the hull with no opening. Where the
profile is convex the root dips a little into the hull between its LE and TE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import hull_modify as HM
from xgeom_common import BLUE, GREY, INK2, ORANGE, fit_box, keep_fitted
from xgeom_hull import short_ticks

EXTRUSION = "#a8461c"            # the rudders' root extrusions, a darker orange
COLLECTIONS = (("hull", GREY, 0.5), ("hub", INK2, 0.8), ("blades", BLUE, 0.6), ("rudders", ORANGE, 0.6),
               ("roots", EXTRUSION, 0.8))
ROOT_OVERLAP = 0.02              # the root extrusion reaches this fraction of the root chord into the hull
THICKNESS_FRACTIONS = np.linspace(-1.0, 1.0, 9)    # across each rudder point's thickness, the chord line included
LOOPS = 30                       # rudder loops drawn, evenly from the root to the cap
MAX_RUDDERS = 12
BLADE_EVERY = 4                  # every 4th section of the XCAD points file (and the last) per blade
OWNS = {"hull": ("hull",), "blade": ("blades", "hub"), "rudder": ("rudders", "roots")}   # a component's lines


@dataclass
class Placement:
    prop_dx: float = 0.0
    prop_d: float = 0.25
    rudder_dx: float = -0.1
    rudder_angle: float = 180.0
    rudders: int = 4
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
    last; the blade count, the diameter, the hub's radius and height; the design's problems and warnings."""
    import blade_modify as BM
    d, p = ad.design, ad.params
    out = {"design": d, "blades": int(p.blades), "diameter": float(d.diameter),
           "hub": (float(d.hub_radius), float(d.hub_height)), "sections": [], "problems": [], "warnings": []}
    try:
        st = ad.stations()
        st = np.unique(np.r_[st[::every], st[-1]])
        out["sections"] = list(BM.blade_points(d, p, st)[0].reshape(-1, 53, 3))
        out["problems"], out["warnings"] = BM.design_checks(d, p, ad.settings["clearance_mm"], hub=ad.settings["hub"])
    except Exception as exc:
        out["problems"].append(f"{type(exc).__name__}: {exc}")
    return out


def rudder_part(ad, n=LOOPS, every=2):
    """The rudder (mm, its working frame: x chordwise, y the height, z the thickness): n of the loops of its
    3D sections tab (modified below H, the cut, the original above with the tip and the cap), evenly from the
    root to the top, every other point; the whole root loop (the lowest) and its x range; the design's
    problems."""
    import rudder_modify as RM
    d, s = ad.design, ad.settings
    problems = RM.design_problems(d, ad.orig, s["n_sections"], s["te_radius_mm"])
    sk = ad.skeleton(problems)
    loops = sorted(sk["modified"] + sk["cut"] + sk["unchanged"], key=lambda lp: float(lp[:, 1].mean()))
    root = loops[0]
    keep = np.unique(np.linspace(0, len(loops) - 1, min(n, len(loops))).round().astype(int))
    loops = [np.vstack((loops[i][::every], loops[i][:1])) for i in keep]
    return {"design": d, "loops": loops, "root": root, "root_x": (float(root[:, 0].min()), float(root[:, 0].max())),
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


def surface_height(design, x, w, phi):
    """Where the hull's surface lies along er = (0, sin phi, cos phi) at the lateral offset w along
    et = (0, -cos phi, sin phi), at x (m): the largest s with s er + w et on the elliptical section, the root of
    a s^2 + b s + c = 0 from (y / r')^2 + (z / r)^2 = 1; NaN where that line misses the section (off the body,
    or wider than it). For a circle of radius R: sqrt(R^2 - w^2)."""
    x, w = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(w, dtype=float))
    r, rp = design.radius("r", x.ravel()).reshape(x.shape), design.radius("rp", x.ravel()).reshape(x.shape)
    sn, cs = np.sin(phi), np.cos(phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (sn / rp) ** 2 + (cs / r) ** 2
        b = 2.0 * w * sn * cs * (1.0 / r ** 2 - 1.0 / rp ** 2)
        c = w ** 2 * ((cs / rp) ** 2 + (sn / r) ** 2) - 1.0
        disc = b * b - 4.0 * a * c
        s = (-b + np.sqrt(np.where(disc >= 0.0, disc, 0.0))) / (2.0 * a)
    return np.where((r > 0.0) & (rp > 0.0) & (disc >= 0.0), s, np.nan)


def root_on_hull(design, x_te, chord, phi, n=400):
    """The root chord's ends on the hull, in the half-plane of a rudder at the angle phi (rad): the TE on the
    hull's profile there (surface_distance) at x_te (m), the LE the first point ahead of it along the profile a
    chord (m) away (n steps from the TE, then Brent's method). Returns (x_le, rho_le, rho_te), rho the distance
    from the axis. ValueError when x_te is off the hull or the hull ahead of it is shorter than the chord."""
    from scipy.optimize import brentq
    if not 0.0 <= x_te <= design.length:
        raise ValueError(f"the rudder's root TE at x {x_te:.4g} m is off the hull (x 0 to {design.length:.4g} m)")

    def prof(x):
        return surface_distance(design, np.atleast_1d(np.asarray(x, dtype=float)), phi)

    rho_te = float(prof(x_te)[0])

    def f(x):
        return (x_te - x) ** 2 + (prof(x) - rho_te) ** 2 - chord ** 2

    xs = np.linspace(x_te, max(x_te - 1.001 * chord, 0.0), n + 1)    # a little past a chord: a flat profile's LE
    k = np.flatnonzero(f(xs) >= 0.0)
    if not len(k):
        raise ValueError(f"the rudder's root chord ({1000 * chord:.1f} mm) is longer than the hull ahead of its TE")
    x_le = brentq(lambda x: float(f(x)[0]), xs[k[0]], xs[k[0] - 1], xtol=1e-14)
    return x_le, float(prof(x_le)[0]), rho_te


def root_gaps(design, pts, phi, eh, n=64, iterations=32):
    """The gap under each point (rows, the hull's frame, m) of a rudder at phi (rad), along -eh (its height,
    inwards): the distance to the hull's surface (surface_height) for a point outside the hull, 0 for one on or
    inside it, NaN where the line meets no hull before the axis. The first crossing among n steps, then
    bisection."""
    er = np.array([0.0, np.sin(phi), np.cos(phi)])
    et = np.array([0.0, -np.cos(phi), np.sin(phi)])
    x0, rho0, w = pts[:, 0], pts @ er, pts @ et
    hx, hr = float(eh[0]), float(eh @ er)

    def g(t, i):                       # > 0 while the point moved by t is outside the hull (no hull: outside)
        t = np.asarray(t, dtype=float)
        if t.ndim == 2:
            v = rho0[i, None] - t * hr - surface_height(design, x0[i, None] - t * hx, w[i, None], phi)
        else:
            v = rho0[i] - t * hr - surface_height(design, x0[i] - t * hx, w[i], phi)
        return np.where(np.isnan(v), np.inf, v)

    allp = np.arange(len(pts))
    t = np.linspace(0.0, 1.0, n + 1)[None, :] * (np.maximum(rho0, 0.0) / hr)[:, None]
    inside = g(t, allp) < 0.0
    first = np.where(inside.any(axis=1), inside.argmax(axis=1), -1)
    gaps = np.full(len(pts), np.nan)
    gaps[first == 0] = 0.0
    i = np.flatnonzero(first > 0)
    lo, hi = t[i, first[i] - 1], t[i, first[i]]
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        out = g(mid, i) >= 0.0
        lo, hi = np.where(out, mid, lo), np.where(out, hi, mid)
    gaps[i] = 0.5 * (lo + hi)
    return gaps


def place_rudder(design, loops, scale, x_te, phi, root=None):
    """One rudder at the angle phi (rad) on the hull: (its loops in the hull's frame (m), the root extrusion's
    lines, info). loops are the rudder's (mm, its frame), the lowest the root; root the whole root loop (mm;
    default loops[0]); scale is m per mm; x_te the hull's x of the root's TE (m).

    The root chord (from the root's LE to its TE, the extreme x of the root, at z = 0) goes onto the hull's
    profile in the rudder's half-plane (root_on_hull): ec along it, eh square to it in that plane and pointing
    away from the axis (the rudder's height), et the thickness. The pitch is the angle of the root chord to the
    axis, positive when the hull tapers aft (the rudder then leans aft). The root section is then copied and
    extruded inwards along eh by the largest gap under it (root_gaps) plus ROOT_OVERLAP of the root chord: the
    inner copy and lines joining the two at every few points. info: x_le, rho_le, rho_te, pitch (deg), gap and
    depth (m), dip (m, how far the root lies inside the hull at most, across its thickness), overhang (the
    fraction of the root with no hull under it), frame (the root LE a, ec, eh, et)."""
    root = loops[0] if root is None else np.asarray(root, dtype=float)
    er = np.array([0.0, np.sin(phi), np.cos(phi)])
    et = np.array([0.0, -np.cos(phi), np.sin(phi)])
    ex = np.array([1.0, 0.0, 0.0])
    xa, xb = float(root[:, 0].min()), float(root[:, 0].max())
    y0 = float(root[:, 1].mean())                           # the root's height in the rudder frame
    chord = scale * (xb - xa)
    x_le, rho_le, rho_te = root_on_hull(design, x_te, chord, phi)
    a = x_le * ex + rho_le * er                             # the root's LE, on the hull
    ec = ((x_te - x_le) * ex + (rho_te - rho_le) * er) / chord
    eh = ((rho_le - rho_te) * ex + (x_te - x_le) * er) / chord

    def place(lp):
        return (a + np.outer(scale * (lp[:, 0] - xa), ec) + np.outer(scale * (lp[:, 1] - y0), eh)
                + np.outer(scale * lp[:, 2], et))

    placed = [place(lp) for lp in loops]
    whole = place(root)
    gaps = root_gaps(design, whole, phi, eh)
    gap = float(np.nanmax(gaps)) if np.isfinite(gaps).any() else 0.0
    depth = gap + ROOT_OVERLAP * chord
    base = placed[0]
    inner = base - depth * eh
    lines = [inner] + [np.array([base[i], inner[i]]) for i in np.linspace(0, len(base) - 1, 16, dtype=int)]
    f = THICKNESS_FRACTIONS
    w = whole @ et
    h = surface_height(design, whole[:, 0, None] + 0.0 * f[None, :], w[:, None] * f[None, :], phi)
    inside = h - (whole @ er)[:, None]
    info = {"x_le": x_le, "rho_le": rho_le, "rho_te": rho_te,
            "pitch": float(np.degrees(np.arctan2(rho_le - rho_te, x_te - x_le))), "gap": gap, "depth": depth,
            "dip": max(0.0, float(np.nanmax(inside))) if np.isfinite(inside).any() else 0.0,
            "overhang": float(np.mean(np.isnan(surface_height(design, whole[:, 0], w, phi)))), "frame": (a, ec, eh, et)}
    return placed, lines, info


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
    segs = {"hull": list(parts["hull"]["lines"]), "hub": [], "blades": [], "rudders": [], "roots": []}
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
        x_te = L + pl.rudder_dx                               # the hull's x of the root's TE
        placed, first = [], None
        for k in range(int(pl.rudders)):
            phi = np.deg2rad(pl.rudder_angle + 360.0 * k / pl.rudders)
            if first is not None and d.axisymmetric:          # a round hull: the first rudder turned about x
                turn = _turn_x(first[3] - phi)
                loops, lines = [p @ turn for p in first[0]], [p @ turn for p in first[1]]
                pi = dict(first[2], frame=tuple(v @ turn for v in first[2]["frame"]))
            else:
                loops, lines, pi = place_rudder(d, ru["loops"], s, x_te, phi, ru.get("root"))
                first = first or (loops, lines, pi, phi)
            segs["rudders"] += loops
            segs["roots"] += lines
            placed.append(pi)
        info.update(rudder_x_te=x_te, rudder_overhang=max(q["overhang"] for q in placed),
                    rudder_frames=[q["frame"] for q in placed])
        info.update({f"rudder_{k}": [q[k] for q in placed]
                     for k in ("x_le", "rho_le", "rho_te", "pitch", "gap", "depth", "dip")})
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
            x0 = min([d.joins[2]] + [float(p[:, 0].min()) for k in ("blades", "hub", "rudders", "roots")
                                     for p in segs[k]])
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
        if "rudder_x_te" in info:
            n = len(info["rudder_pitch"])

            def rng(v, k, digits):
                v, fmt = k * np.asarray(v, dtype=float), f"{{:.{digits}f}}"
                return (fmt.format(v[0]) if np.ptp(v) < 0.5 * 10.0 ** -digits
                        else f"{fmt.format(v.min())} to {fmt.format(v.max())}")
            where = (", ".join(f"{(pl.rudder_angle + 360.0 * k / n) % 360:g}" for k in range(n)) + " deg" if n <= 4
                     else f"{pl.rudder_angle % 360:g} deg and every {360.0 / n:.4g} deg from there")
            dip = max(info["rudder_dip"])
            lines.append(f"{n} rudder{'s' if n > 1 else ''} (orange) at {where}, on the hull: root LE at x "
                         f"{rng(info['rudder_x_le'], 1, 3)} m, {rng(info['rudder_rho_le'], 1000, 1)} mm from the axis; "
                         f"TE at x {info['rudder_x_te']:.3f} m, {rng(info['rudder_rho_te'], 1000, 1)} mm")
            lines.append(f"pitched {rng(info['rudder_pitch'], 1, 1)} deg to the axis: the hull's slope under the root, "
                         f"also when moved" + (f"; the root dips up to {1000 * dip:.1f} mm into the hull"
                                               if dip >= 5e-5 else ""))
            lines.append(f"root section extruded {rng(info['rudder_depth'], 1000, 1)} mm inwards (dark orange): the "
                         f"largest gap under the root, {rng(info['rudder_gap'], 1000, 1)} mm, and "
                         f"{100 * ROOT_OVERLAP:g} % of the chord into the hull"
                         + ("" if info["rudder_overhang"] == 0 else
                            f"; {100 * info['rudder_overhang']:.0f} % of the root has no hull under it"))
            gap = info.get("gap")
            if gap is not None:
                lines.append(f"rudder{'s' if n > 1 else ''} to the blades (within their radius): "
                             + (f"{1000 * gap:.0f} mm clear" if gap >= 0 else f"{-1000 * gap:.0f} mm overlap in x"))
        lines.append("the component being edited is drawn thicker; drag to turn")
        self.text.set_text("\n".join(lines))
        return info
