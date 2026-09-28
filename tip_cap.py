"""
tip_cap.py

The rudder's tip cap, the part above the tilted tip section, described by a few
physical parameters. numpy / scipy only (no OCC), so it builds, fits and tests
anywhere; Rudder_geom_extraction.py does the CAD side (cutting the part).

Cap frame
---------
The tip section (the tilted section through the top of the LE and the top of the
TE) is the cap's base. Along its chord, x runs from the top of the LE (0) to the
top of the TE (chord), u = x / chord. h is the height above the tip section's
plane, along its normal, and z is the thickness coordinate of the working frame.
w(x) is the tip section's local half-thickness and m(x) its mid-line, both taken
from the untilted tip airfoil.

Parameters (`CapParams`, lengths in mm)
---------------------------------------
face   : the inclined face the cap is built on. Its height above the tip section is
         a cubic Bezier along the chord,
             face(u) = h_le (1-u) + h_te u + 3u(1-u) [b1 (1-u) + b2 u],
         i.e. the straight line from h_le (LE end) to h_te (TE end) plus a bow
         (b1 = b2 = 0 is a plane).
crown  : the face is crowned across the thickness with radius crown_radius, so the
         top of the cap is T(u) = face(u) + w^2 / (2 crown_radius). An infinite
         radius is a flat face.
edge   : a constant edge_radius joins the side walls to the crown all round, and
         also rounds the LE end of the cap (in the side view).
te     : the TE end is rounded in the side view with its own te_radius.

Each cross-section (a plane normal to the tip chord) is then: a straight wall at
|z - m| = w from h = 0 up to the edge radius, the edge radius, and the parabolic
crown up to the top H(u) = T(u) E_le(x) E_te(chord - x), where E is a quarter-circle
closure of length edge_radius at the LE end and te_radius at the TE end, so H goes
to zero at both ends with a vertical tangent.

Special cases: edge_radius = 0 and an infinite crown radius is a square tip; a finite
edge radius on a flat face is a flat tip with rounded edges; edge_radius equal to
the local half-thickness is a fully rounded tip.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np
from scipy.optimize import brentq, least_squares

INF = float("inf")


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class CapParams:
    h_le: float                 # face height above the tip section at the LE end [mm]
    h_te: float                 # face height above the tip section at the TE end [mm]
    b1: float = 0.0             # face bow, Bezier terms (0 = plane) [mm]
    b2: float = 0.0
    crown_radius: float = INF   # crown radius across the thickness [mm]
    edge_radius: float = 0.0    # edge radius, walls to crown and round the LE end [mm]
    te_radius: float = 0.0      # side-view corner radius at the TE end [mm]

    def face(self, u):
        u = np.asarray(u, dtype=float)
        return (self.h_le * (1.0 - u) + self.h_te * u
                + 3.0 * u * (1.0 - u) * (self.b1 * (1.0 - u) + self.b2 * u))

    def inclination_deg(self, chord: float) -> float:
        """Face inclination relative to the tip section (ends only, bow aside)."""
        return float(np.degrees(np.arctan2(self.h_te - self.h_le, chord)))

    def as_dict(self) -> dict:
        d = asdict(self)
        if not np.isfinite(d["crown_radius"]):
            d["crown_radius"] = None            # JSON has no infinity: None = flat
        return d


# --------------------------------------------------------------------------- #
# Base: the tip section the cap sits on
# --------------------------------------------------------------------------- #
class CapBase:
    """
    The tip section as the cap's base.

    x_up, y_up / x_lo, y_lo : the untilted tip airfoil's upper (+z) and lower sides,
        each LE -> TE, x from 0 at its LE to the chord, y = z (thickness), mm.
    pivot     : 3D point of the tip section's LE with z set to 0 (working frame).
    chord_dir : unit vector along the tip chord (LE top -> TE top).
    normal    : unit normal of the tip section's plane (points away from the root).
    thick_dir : unit thickness axis.
    """

    def __init__(self, x_up, y_up, x_lo, y_lo, pivot, chord_dir, normal, thick_dir):
        self.x_up, self.y_up = np.asarray(x_up, float), np.asarray(y_up, float)
        self.x_lo, self.y_lo = np.asarray(x_lo, float), np.asarray(y_lo, float)
        self.chord = float(max(self.x_up[-1], self.x_lo[-1]))
        self.pivot = np.asarray(pivot, float)
        self.t = np.asarray(chord_dir, float)
        self.n = np.asarray(normal, float)
        self.k = np.asarray(thick_dir, float)

    @classmethod
    def from_selig(cls, raw, pivot, chord_dir, normal, thick_dir):
        """From a Selig-ordered airfoil (TE -> upper -> LE -> lower -> TE), LE at x = 0."""
        raw = np.asarray(raw, float)
        i_le = int(np.argmin(raw[:, 0]))
        up = raw[:i_le + 1][::-1]
        lo = raw[i_le:]
        return cls(up[:, 0], up[:, 1], lo[:, 0], lo[:, 1], pivot, chord_dir, normal, thick_dir)

    def upper(self, x):
        return np.interp(x, self.x_up, self.y_up)

    def lower(self, x):
        return np.interp(x, self.x_lo, self.y_lo)

    def half_thickness(self, x):
        return 0.5 * (self.upper(x) - self.lower(x))

    def mid(self, x):
        return 0.5 * (self.upper(x) + self.lower(x))

    def to_3d(self, x, z, h):
        """Cap-frame (x along the chord, z thickness, h above the base) -> 3D."""
        x, z, h = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float),
                                      np.asarray(h, float))
        return (self.pivot + x[..., None] * self.t + h[..., None] * self.n
                + z[..., None] * self.k)

    def to_cap(self, pts):
        """3D points -> cap frame (x, z, h)."""
        d = np.asarray(pts, float) - self.pivot
        return d @ self.t, d @ self.k, d @ self.n


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
def _closure(d, length):
    """Quarter-circle end closure: 0 at d = 0 with a vertical tangent, 1 from d = length."""
    d = np.asarray(d, float)
    if length <= 0.0:
        return np.where(d > 0.0, 1.0, 0.0)
    t = np.clip(d / length, 0.0, 1.0)
    return np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, 1.0))


def cap_top(x, p: CapParams, base: CapBase):
    """Height of the top of the cap above the tip section, H(x)."""
    x = np.asarray(x, float)
    u = x / base.chord
    w = base.half_thickness(x)
    crown = 0.0 if not np.isfinite(p.crown_radius) else w * w / (2.0 * p.crown_radius)
    top = p.face(u) + crown
    return (np.maximum(top, 0.0) * _closure(x, p.edge_radius)
            * _closure(base.chord - x, p.te_radius))


@dataclass
class _Section:
    """One cap cross-section's geometry: wall, edge radius, crown."""
    w: float
    H: float
    r: float        # edge radius actually used (clamped to the section)
    Rc: float
    y0: float       # height where the wall meets the edge radius
    ht: float       # height where the edge radius meets the crown
    st: float       # distance from the mid-line where the edge radius meets the crown


def _solve(w: float, H: float, r: float, Rc: float) -> _Section:
    if not np.isfinite(Rc):                                   # flat face
        return _Section(w, H, r, Rc, H - r, H, w - r)
    if r <= 0.0:                                              # sharp edge
        hc = H - w * w / (2.0 * Rc)
        return _Section(w, H, 0.0, Rc, hc, hc, w)

    def f(s):                        # centre of the edge circle sits at s = w - r
        N = np.sqrt(1.0 + (s / Rc) ** 2)
        return s - r * (s / Rc) / N - (w - r)

    st = brentq(f, 0.0, w, xtol=1e-14) if f(0.0) < 0.0 < f(w) else max(w - r, 0.0)
    N = np.sqrt(1.0 + (st / Rc) ** 2)
    ht = H - st * st / (2.0 * Rc)
    return _Section(w, H, r, Rc, ht - r / N, ht, st)


def _section(w: float, H: float, edge_radius: float, crown_radius: float) -> _Section:
    """
    The cross-section at one station. The edge radius is clamped so it fits: no
    wider than the section, and no lower than the tip section itself (near the two
    ends of the cap, where the cap is shallow, it shrinks).
    """
    w, H = max(float(w), 0.0), max(float(H), 0.0)
    if w <= 0.0 or H <= 0.0:
        return _Section(w, H, 0.0, crown_radius, 0.0, 0.0, w)
    r = max(min(edge_radius, 0.999 * w, 0.999 * H), 0.0)
    sec = _solve(w, H, r, crown_radius)
    if sec.y0 < 0.0 and r > 0.0:                    # shrink r until it sits on the wall
        lo, hi = 0.0, r
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _solve(w, H, mid, crown_radius).y0 >= 0.0:
                lo = mid
            else:
                hi = mid
        sec = _solve(w, H, lo, crown_radius)
    if sec.y0 < 0.0:                                # sharp edge under a deep crown
        sec.y0 = 0.0
    return sec


def _distance(sec: _Section, s, h):
    """
    Exact distance from points (s, h) to one side of the cross-section: the wall
    segment, the edge-radius arc and the crown (a parabola, closest point by
    Newton), whichever is nearest.
    """
    s, h = np.asarray(s, float), np.asarray(h, float)
    if sec.H <= 0.0:
        return np.hypot(s - sec.w, h)
    y0 = max(sec.y0, 0.0)
    d = np.hypot(s - sec.w, h - np.clip(h, 0.0, y0))                       # wall
    if sec.r > 0.0:                                                          # arc
        cs, ch = sec.w - sec.r, sec.y0
        phi_t = np.arctan2(sec.ht - ch, sec.st - cs)
        ang = np.arctan2(h - ch, s - cs)
        on = (ang >= 0.0) & (ang <= phi_t)
        da = np.abs(np.hypot(s - cs, h - ch) - sec.r)
        d = np.where(on, np.minimum(d, da), d)
    if np.isfinite(sec.Rc):                                                  # crown
        Rc, H = sec.Rc, sec.H
        x = np.clip(s, 0.0, sec.st)
        for _ in range(12):
            g = x - s - (x / Rc) * (H - h - x * x / (2.0 * Rc))
            dg = 1.0 - (H - h) / Rc + 1.5 * x * x / (Rc * Rc)
            x = np.clip(x - g / np.where(np.abs(dg) > 1e-12, dg, 1e-12), 0.0, sec.st)
        dc = np.hypot(x - s, H - x * x / (2.0 * Rc) - h)
        dc = np.minimum(dc, np.hypot(s, h - H))                     # the crest itself
        d = np.minimum(d, dc)
    else:
        d = np.minimum(d, np.hypot(s - np.clip(s, 0.0, sec.st), h - sec.H))
    return d


def _offset(sec: _Section, h):
    """Distance from the mid-line, s, at height h (0 <= h <= H) on one side."""
    h = np.clip(np.asarray(h, float), 0.0, sec.H)
    s = np.full(h.shape, sec.w)
    if sec.H <= 0.0:
        return s
    arc = (h > sec.y0) & (h <= sec.ht)
    if sec.r > 0.0:
        s[arc] = (sec.w - sec.r) + np.sqrt(np.clip(sec.r ** 2 - (h[arc] - sec.y0) ** 2, 0.0, None))
    top = h > sec.ht
    if np.isfinite(sec.Rc):
        s[top] = np.sqrt(np.clip(2.0 * sec.Rc * (sec.H - h[top]), 0.0, None))
    else:
        s[top] = 0.0
    return s


def profile(x: float, p: CapParams, base: CapBase, n: int = 400):
    """One side of the cross-section at x as (s, h), wall base to crest."""
    sec = _section(base.half_thickness(x), float(cap_top(x, p, base)), p.edge_radius,
                   p.crown_radius)
    if sec.H <= 0.0:
        return np.array([[sec.w, 0.0]])
    # dense where the shape turns: the edge radius and the crest
    parts = [np.linspace(0.0, max(sec.y0, 0.0), n // 4, endpoint=False),
             np.linspace(max(sec.y0, 0.0), sec.ht, n // 4, endpoint=False),
             sec.ht + (sec.H - sec.ht) * np.sin(np.linspace(0.0, 0.5 * np.pi, n // 2)) ** 2]
    h = np.unique(np.concatenate(parts))
    return np.column_stack([_offset(sec, h), h])


def points(x, v, side: int, p: CapParams, base: CapBase):
    """
    Cap points at chord stations x (array) and height fractions v (scalar or
    array broadcast with x), side +1 (upper, +z) or -1 (lower), as 3D points.
    v = 0 is the wall base (the tip section itself), v = 1 the crest.
    """
    x = np.atleast_1d(np.asarray(x, float))
    v = np.broadcast_to(np.asarray(v, float), x.shape)
    H = cap_top(x, p, base)
    w = base.half_thickness(x)
    z = np.empty_like(x)
    h = v * H
    for i in range(len(x)):
        sec = _section(w[i], H[i], p.edge_radius, p.crown_radius)
        s = float(_offset(sec, h[i])) if H[i] > 0.0 else w[i]
        z[i] = base.mid(x[i]) + side * s
    # at the two ends the cap has no height: land exactly on the tip section
    edge = H <= 0.0
    if np.any(edge):
        z[edge] = (base.upper(x[edge]) if side > 0 else base.lower(x[edge]))
    return base.to_3d(x, z, h)


def loops(p: CapParams, base: CapBase, levels, x_up=None, x_lo=None):
    """
    Closed loops over the cap for a section-based rebuild (the XCAD file), one per
    height fraction in `levels`, each in the tip section's own point order: start at
    the LE, run along the +z side to the TE, come back along the -z side, and
    repeat the LE. They use the tip section's chordwise stations, so every loop has
    the same point count as the sections and meets the tip section at the top of
    the LE and the top of the TE, where the cap has no height.
    """
    x_up = base.x_up if x_up is None else np.asarray(x_up, float)
    x_lo = base.x_lo if x_lo is None else np.asarray(x_lo, float)
    out = []
    for v in levels:
        up = points(x_up, v, +1, p, base)                     # LE -> TE, +z
        lo = points(x_lo[::-1], v, -1, p, base)               # TE -> LE, -z
        out.append(np.vstack([up, lo[1:]]))
    return out


def grid(p: CapParams, base: CapBase, nx: int = 81, nv: int = 21):
    """
    The cap as a structured point grid, (nx, 2 nv - 1, 3): rows along the chord
    (cosine-spaced, LE end to TE end), columns from the +z wall base over the crest
    to the -z wall base. The end rows collapse to the top of the LE and the top of
    the TE.
    """
    x = base.chord * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, nx)))
    v = np.sin(np.linspace(0.0, 0.5 * np.pi, nv)) ** 2        # dense near the crest
    up = np.stack([points(x, vi, +1, p, base) for vi in v], axis=1)
    lo = np.stack([points(x, vi, -1, p, base) for vi in v[::-1][1:]], axis=1)
    return np.concatenate([up, lo], axis=1)


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
@dataclass
class CapStation:
    """Measured cap points at one chord station, in the cap frame."""
    x: float
    s: np.ndarray       # distance from the tip section's mid-line (|z - m|)
    h: np.ndarray       # height above the tip section


def _deviation(st: CapStation, p: CapParams, base: CapBase) -> np.ndarray:
    """Distance from each measured point to the model cross-section at its station."""
    sec = _section(base.half_thickness(st.x), float(cap_top(st.x, p, base)),
                   p.edge_radius, p.crown_radius)
    return _distance(sec, st.s, st.h)


def fit(stations, base: CapBase, x_core=(0.02, 0.95), init: CapParams | None = None):
    """
    Least-squares fit of CapParams to measured cross-sections.

    Stage 1 fits the face (h_le, h_te, b1, b2), the crown (as a curvature, so a flat
    face is reachable) and the edge radius on the stations inside `x_core` (as
    fractions of the chord), point-to-profile distances. Stage 2 fits the TE corner
    radius on the stations past the core, from their crest heights. Returns
    (params, report) with the deviation of every station.
    """
    stations = [st for st in stations if len(st.h)]
    C = base.chord
    core = [st for st in stations if x_core[0] * C <= st.x <= x_core[1] * C]
    te_end = [st for st in stations if st.x > x_core[1] * C]
    if len(core) < 4:
        raise ValueError("need at least four cap stations in the core region")

    crest = np.array([st.h.max() for st in core])
    xs = np.array([st.x for st in core])
    lin = np.polyfit(xs / C, crest, 1)
    p0 = init or CapParams(h_le=float(np.polyval(lin, 0.0)), h_te=float(np.polyval(lin, 1.0)),
                           crown_radius=60.0, edge_radius=0.5, te_radius=0.02 * C)
    w_max = float(base.half_thickness(np.linspace(0.0, C, 400)).max())

    # subsample each station for speed; keep the points near the crest and the edge
    def sub(st, n=240):
        k = np.linspace(0, len(st.h) - 1, min(n, len(st.h))).astype(int)
        return CapStation(st.x, st.s[k], st.h[k])
    core_s = [sub(st) for st in core]

    def unpack(q):
        kappa = q[4]
        return CapParams(h_le=q[0], h_te=q[1], b1=q[2], b2=q[3],
                         crown_radius=(1.0 / kappa) if kappa > 1e-9 else INF,
                         edge_radius=q[5], te_radius=p0.te_radius)

    def res1(q):
        p = unpack(q)
        return np.concatenate([_deviation(st, p, base) for st in core_s])

    kappa0 = 0.0 if not np.isfinite(p0.crown_radius) else 1.0 / p0.crown_radius
    q0 = [p0.h_le, p0.h_te, p0.b1, p0.b2, kappa0, min(p0.edge_radius, 0.5 * w_max)]
    lo = [-np.inf, -np.inf, -np.inf, -np.inf, 0.0, 0.0]
    hi = [np.inf, np.inf, np.inf, np.inf, 1.0, w_max]
    r1 = least_squares(res1, q0, bounds=(lo, hi), diff_step=1e-4, x_scale="jac")
    p = unpack(r1.x)

    if te_end:
        def res2(q):
            pt = replace(p, te_radius=float(q[0]))
            return np.array([cap_top(st.x, pt, base) - st.h.max() for st in te_end])
        r2 = least_squares(res2, [p0.te_radius], bounds=([0.0], [0.5 * C]))
        p.te_radius = float(r2.x[0])

    report = []
    for st in stations:
        d = _deviation(st, p, base)
        report.append({"x": float(st.x), "u": float(st.x / C),
                       "w": float(base.half_thickness(st.x)),
                       "top_cad": float(st.h.max()), "top_model": float(cap_top(st.x, p, base)),
                       "max_dev": float(d.max()), "rms_dev": float(np.sqrt(np.mean(d * d))),
                       "n_points": int(len(d)), "core": bool(x_core[0] * C <= st.x <= x_core[1] * C)})
    return p, report
