"""
bezier_section.py

A symmetric section from a Bezier half-section of any degree n, in fractions
of the sharp chord cs:

    P0 = (0, 0)                 the LE
    P1 .. P(n-1) = (x_i, z_i)   the inner control points, both coordinates free
    Pn = (1, 0)                 the sharp TE

mirrored for the lower surface, its sharp TE replaced by the circle of radius r
tangent to the half-section with its centre on the chord line (the fillet of
replica_funcs). The quartic of replica_funcs, P1 = (0, T1), P2 = (X1, T1),
P3 = (X2, T2), is the case n = 4. The LE is round when x_1 = 0 (P0-P1
vertical); any other x_1 gives the mirrored section a corner there.

The inner coordinates are named P1x, P1z, P2x, ... (coord_names). Every
function takes a stack of control polygons, shape (m, n + 1, 2), one section
per row, and works on all of them at once.

    ctrl = polygon([[0.0, 0.07], [0.25, 0.07], [0.6, 0.05]])      # (1, 5, 2)
    cs, r = sharp_chord(ctrl, chord=150.0, r_abs=0.75)             # mm
    xy = rounded(ctrl, r, num_pts=200) * (150.0 / te_x(ctrl, r))   # Selig order, mm
"""

from functools import lru_cache
from math import comb

import numpy as np


# --------------------------------------------------------------------------
# Control polygons
# --------------------------------------------------------------------------

def coord_names(degree):
    """Names of the inner coordinates of a degree-n half-section, in order:
    P1x, P1z, P2x, P2z, ..., P(n-1)x, P(n-1)z."""
    return [f"P{i}{a}" for i in range(1, int(degree)) for a in "xz"]


def as_stack(ctrl):
    """Control polygons as an (m, n + 1, 2) float array."""
    c = np.asarray(ctrl, dtype=float)
    if c.ndim == 2:
        c = c[None]
    if c.ndim != 3 or c.shape[2] != 2 or c.shape[1] < 3:
        raise ValueError("control polygons must have the shape (m, n + 1, 2) with n >= 2")
    return c


def polygon(inner):
    """Inner points (m, n - 1, 2), or (n - 1, 2) for one section, completed
    with P0 = (0, 0) and Pn = (1, 0): (m, n + 1, 2)."""
    q = np.asarray(inner, dtype=float)
    if q.ndim == 2:
        q = q[None]
    m = q.shape[0]
    return np.concatenate((np.zeros((m, 1, 2)), q, np.tile([[[1.0, 0.0]]], (m, 1, 1))), axis=1)


def polygon_from_values(values, degree):
    """(m, n + 1, 2) from {coordinate name: value or array (m,)}."""
    names = coord_names(degree)
    cols = np.broadcast_arrays(*[np.atleast_1d(np.asarray(values[k], dtype=float)) for k in names])
    inner = np.stack(cols, axis=1).reshape(len(cols[0]), int(degree) - 1, 2)
    return polygon(inner)


def values_from_polygon(ctrl):
    """{coordinate name: array (m,)} of the inner points."""
    c = as_stack(ctrl)
    n = c.shape[1] - 1
    return {f"P{i}{a}": c[:, i, j] for i in range(1, n) for j, a in enumerate("xz")}


@lru_cache(maxsize=None)
def _binom(n):
    return np.array([comb(n, k) for k in range(n + 1)], dtype=float)


def bernstein(t, n):
    """Bernstein basis of degree n at t (any shape): (..., n + 1)."""
    t = np.asarray(t, dtype=float)[..., None]
    k = np.arange(n + 1)
    return _binom(n) * t**k * (1.0 - t)**(n - k)


def _eval(c, t):
    """Points of the curves with control points c (m, n + 1, d) at t, which
    is (k,) for all sections or (m, k): (m, k, d)."""
    b = bernstein(t, c.shape[1] - 1)
    return np.einsum("kj,mjd->mkd" if b.ndim == 2 else "mkj,mjd->mkd", b, c)


def points(ctrl, t):
    """Half-section points at the parameters t, (k,) or (m, k): (m, k, 2)."""
    return _eval(as_stack(ctrl), t)


def derivative(ctrl, t):
    """dP/dt at the parameters t: (m, k, 2)."""
    c = as_stack(ctrl)
    return _eval((c.shape[1] - 1) * np.diff(c, axis=1), t)


def elevate(ctrl, times=1):
    """The same curves with their degree raised by `times`."""
    c = as_stack(ctrl)
    for _ in range(int(times)):
        n = c.shape[1] - 1
        a = (np.arange(1, n + 1) / (n + 1))[None, :, None]
        c = np.concatenate((c[:, :1], a * c[:, :-1] + (1.0 - a) * c[:, 1:], c[:, -1:]), axis=1)
    return c


def elevation_matrix(n_from, n_to):
    """E with elevate(P) = E @ P, from degree n_from to n_to >= n_from."""
    e = np.eye(n_from + 1)
    for n in range(n_from, n_to):
        a = np.arange(1, n + 1) / (n + 1)
        step = np.zeros((n + 2, n + 1))
        step[0, 0] = step[-1, -1] = 1.0
        step[np.arange(1, n + 1), np.arange(0, n)] = a
        step[np.arange(1, n + 1), np.arange(1, n + 1)] = 1.0 - a
        e = step @ e
    return e


def reduce(ctrl, degree):
    """The degree-`degree` curves closest (least squares on the control
    points of the elevated curve) to these, the end points kept. Exact when
    the curves came from elevating a curve of that degree."""
    c = as_stack(ctrl)
    n = c.shape[1] - 1
    degree = int(degree)
    if degree >= n:
        return elevate(c, degree - n)
    if degree < 2:
        raise ValueError("a half-section needs degree 2 or more")
    e = elevation_matrix(degree, n)
    rhs = c - e[:, :1] @ c[:, :1] - e[:, -1:] @ c[:, -1:]
    inner = np.einsum("ij,mjd->mid", np.linalg.pinv(e[:, 1:-1]), rhs)
    return np.concatenate((c[:, :1], inner, c[:, -1:]), axis=1)


def change_degree(ctrl, degree):
    """elevate or reduce to `degree`."""
    c = as_stack(ctrl)
    n = c.shape[1] - 1
    return elevate(c, int(degree) - n) if int(degree) >= n else reduce(c, degree)


# --------------------------------------------------------------------------
# Shape checks and measures
# --------------------------------------------------------------------------

_T_CHECK = np.linspace(0.0, 1.0, 241)


def x_monotone(ctrl):
    """True for each section whose x rises from 0 to 1 along the curve
    (checked on a fine grid of t), so each x has one point on it."""
    dx = derivative(ctrl, _T_CHECK)[:, :, 0]
    return np.all(dx >= -1e-12, axis=1)


def half_thickness(ctrl, n_t=401):
    """(z max, t there, z min inside) of each section's half-section, from a
    grid of n_t parameters; z min leaves out the two ends, where z is 0."""
    t = np.linspace(0.0, 1.0, int(n_t))
    z = points(ctrl, t)[:, :, 1]
    k = np.argmax(z, axis=1)
    return z[np.arange(len(z)), k], t[k], z[:, 2:-2].min(axis=1)


def le_radius(ctrl):
    """LE radius of each section, in the polygon's units (inf when P0, P1
    and P2 line up): the curvature at t = 0 is (n - 1) / n |P01 x P12| / |P01|^3."""
    c = as_stack(ctrl)
    n = c.shape[1] - 1
    a, b = c[:, 1] - c[:, 0], c[:, 2] - c[:, 1]
    cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    la = np.hypot(a[:, 0], a[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (n - 1) / n * cross / la**3
        return np.where(k > 0.0, 1.0 / k, np.inf)


# --------------------------------------------------------------------------
# Exact evaluation at given x
# --------------------------------------------------------------------------

def t_at_x(ctrl, xq, tol=1e-14, max_iter=60):
    """Curve parameter where x(t) = xq, for each section: xq is (k,) for all
    or (m, k); returns (m, k). x(t) must rise monotonically (x_monotone).
    Newton's method inside a shrinking bracket (bisection when Newton would
    leave it), started from a sampled inverse."""
    c = as_stack(ctrl)
    m = c.shape[0]
    xq = np.asarray(xq, dtype=float)
    xq = np.broadcast_to(xq, (m, xq.shape[-1])) if xq.ndim == 1 else xq
    ts = np.linspace(0.0, 1.0, 65)
    xs = np.maximum.accumulate(points(c, ts)[:, :, 0], axis=1)
    t = np.array([np.interp(xq[i], xs[i], ts) for i in range(m)])
    lo, hi = np.zeros_like(t), np.ones_like(t)
    d = (c.shape[1] - 1) * np.diff(c, axis=1)
    for _ in range(max_iter):
        f = _eval(c[:, :, :1], t)[:, :, 0] - xq
        dx = _eval(d[:, :, :1], t)[:, :, 0]
        lo = np.where(f < 0.0, t, lo)
        hi = np.where(f > 0.0, t, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_new = t - f / dx
        t_new = np.where((t_new >= lo) & (t_new <= hi), t_new, 0.5 * (lo + hi))
        t_new = np.where(f == 0.0, t, t_new)
        step = np.max(np.abs(t_new - t)) if t.size else 0.0
        t = t_new
        if step <= tol:
            break
    return t


def z_at_x(ctrl, xq):
    """Half-thickness of the (sharp) half-section at the stations xq: (m, k)."""
    c = as_stack(ctrl)
    return _eval(c[:, :, 1:], t_at_x(c, xq))[:, :, 0]


# --------------------------------------------------------------------------
# The rounded TE
# --------------------------------------------------------------------------

def fillet(ctrl, r, n_bisect=64):
    """The TE circle of radius r (sharp-chord units, scalar or (m,)) of each
    section: tangent to the half-section, centre on the chord line.

    The tangent point is where z sqrt(1 + (dz/dx)^2) = r, between the thickest
    point and the TE (bisection on the curve parameter; that quantity is the
    half-thickness at the thickest point and 0 at the TE). Returns a dict of
    arrays (m,): t, x, z (the tangent point), slope, cx (the centre), r (the
    radius rebuilt from the tangent point) and x_te = cx + r, the rounded TE.
    Raises ValueError when r does not fit under a section's half-thickness."""
    c = as_stack(ctrl)
    m = c.shape[0]
    r = np.broadcast_to(np.asarray(r, dtype=float), (m,))
    if np.any(~(r > 0.0)):
        raise ValueError("the TE radius must be positive")
    z_max, t_lo, _ = half_thickness(c)
    if np.any(r >= z_max):
        k = int(np.argmax(r >= z_max))
        raise ValueError(f"r = {r[k]:g} is not below the half-thickness {z_max[k]:.5f}, "
                         "so no tangent circle fits")
    d = (c.shape[1] - 1) * np.diff(c, axis=1)

    def gap(t):
        p = _eval(c, t[:, None])[:, 0]
        dp = _eval(d, t[:, None])[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[:, 1] * np.hypot(dp[:, 0], dp[:, 1]) / dp[:, 0] - r

    lo, hi = t_lo.copy(), np.full(m, 1.0 - 1e-13)
    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo + hi)
        g = gap(mid)
        lo = np.where(g > 0.0, mid, lo)
        hi = np.where(g > 0.0, hi, mid)
    t = 0.5 * (lo + hi)
    p = _eval(c, t[:, None])[:, 0]
    dp = _eval(d, t[:, None])[:, 0]
    slope = dp[:, 1] / dp[:, 0]
    cx = p[:, 0] + slope * p[:, 1]
    rr = np.hypot(p[:, 0] - cx, p[:, 1])
    return {"t": t, "x": p[:, 0], "z": p[:, 1], "slope": slope, "cx": cx, "r": rr, "x_te": cx + rr}


def te_x(ctrl, r):
    """x of the rounded TE (sharp-chord units) for the TE radius r."""
    return fillet(ctrl, r)["x_te"]


def sharp_chord(ctrl, chord, r_abs, tol=1e-13, max_iter=50):
    """The sharp chord cs of each section whose TE, rounded with the radius
    r_abs (units of chord), lies `chord` behind its LE: cs = chord / x_te(r_abs
    / cs), a fixed point that converges in a few steps (x_te changes little
    with r). chord and r_abs are scalars or (m,). Returns (cs, r = r_abs / cs)."""
    c = as_stack(ctrl)
    chord = np.broadcast_to(np.asarray(chord, dtype=float), (c.shape[0],)).copy()
    r_abs = np.broadcast_to(np.asarray(r_abs, dtype=float), (c.shape[0],))
    cs = chord.copy()
    for _ in range(max_iter):
        cs_new = chord / fillet(c, r_abs / cs)["x_te"]
        done = np.all(np.abs(cs_new - cs) <= tol * chord)
        cs = cs_new
        if done:
            return cs, r_abs / cs
    raise RuntimeError(f"sharp_chord: no convergence in {max_iter} steps")


def _cosine(x_end, n):
    return x_end[:, None] * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n))))[None, :]


def rounded(ctrl, r, num_pts=200):
    """The rounded sections, (m, num_pts, 2) in sharp-chord units, Selig order
    (TE -> upper -> LE -> lower -> TE), laid out as the extraction lays out its
    sections: (n + 1) // 2 cosine-spaced stations on the upper surface and
    n - (n + 1) // 2 + 1 on the lower one from the LE to the rounded TE, the LE
    shared (index (n + 1) // 2 - 1) and the TE first and last. Upstream of the
    tangent point the Bezier, downstream the circle (fillet)."""
    return _rounded(as_stack(ctrl), fillet(ctrl, r), num_pts)


def sections(ctrl, chord, r_abs, num_pts=200):
    """The rounded sections with the TE radius r_abs whose rounded TE lies
    `chord` behind the LE (scalars or (m,), same units): (selig (m, num_pts, 2)
    in those units, cs (m,) the sharp chords, r (m,) = r_abs / cs)."""
    c = as_stack(ctrl)
    chord = np.broadcast_to(np.asarray(chord, dtype=float), (c.shape[0],))
    cs, r = sharp_chord(c, chord, r_abs)
    f = fillet(c, r)
    return _rounded(c, f, num_pts) * (chord / f["x_te"])[:, None, None], cs, r


def _rounded(c, f, num_pts):
    n_up = (int(num_pts) + 1) // 2
    n_lo = int(num_pts) - n_up + 1

    def half(n):
        x = _cosine(f["x_te"], n)
        on = x <= f["x"][:, None]
        z = np.sqrt(np.clip(f["r"][:, None]**2 - (x - f["cx"][:, None])**2, 0.0, None))
        zb = z_at_x(c, np.where(on, x, 0.0))
        z = np.where(on, zb, z)
        z[:, -1] = 0.0                                   # the TE point itself
        return np.stack((x, z), axis=2)

    up, lo = half(n_up), half(n_lo)
    lo = lo[:, 1:] * [1.0, -1.0]
    return np.concatenate((up[:, ::-1], lo), axis=1) + 0.0
