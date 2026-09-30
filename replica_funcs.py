"""
replica_funcs.py

The Bezier half-section that replicates a rudder section: a quartic Bezier
from the leading edge (0, 0) to a sharp trailing edge (1, 0), in chord units,

    P0 = (0, 0)   P1 = (0, Y1)   P2 = (X2, Y1)   P3 = (X3, Y3)   P4 = (1, 0)

with four free variables xi = [Y1, X2, X3, Y3]. P0-P1 is vertical, so the
leading edge is round, with radius 4 Y1^2 / (3 X2); P1-P2 is horizontal; the
trailing-edge half angle is atan(Y3 / (1 - X3)). The section is symmetric, the
lower surface being the mirror image of this one.

Functions taking xi also take a whole swarm, xi of shape (n, 4).

The thickness-format files (Section # | X1 | X2 | T1 | T2 | r) name the same
numbers X1 = X2, X2 = X3, T1 = Y1, T2 = Y3, plus the TE radius r, all as
fractions of the sharp chord. rounded_section builds the rounded section from
them and read_thickness_params reads such a file:

    secs, params = read_thickness_params("outputs/thickness_params_H80.txt")
    xy = rounded_section(*params[-1], chord=163.3312, chord_type="actual")

With the TE radius given in mm instead, sharp_chord finds the sharp chord (and
so r) that puts the rounded TE at a given chord:

    cs, r = sharp_chord(X1, X2, T1, T2, chord=163.3312, r_abs=0.75)
    xy = rounded_section(X1, X2, T1, T2, r, chord=163.3312, chord_type="actual", num_pts=200)
"""

import math
from functools import lru_cache
from math import comb

import numpy as np

from retruncate import make_fillet, perp_method, trim_at

ORDER = 4


# --------------------------------------------------------------------------
# Bezier sampling
# --------------------------------------------------------------------------

def bernstein(t, order=ORDER):
    """Bernstein basis of the given order at t (any shape): (..., order + 1)."""
    t = np.asarray(t, dtype=float)[..., None]
    k = np.arange(order + 1)
    return _binomials(order) * t**k * (1.0 - t)**(order - k)


@lru_cache(maxsize=None)
def _binomials(order):
    return np.array([comb(order, k) for k in range(order + 1)], dtype=float)


@lru_cache(maxsize=None)
def _bernstein_matrix(n_points, order):
    return bernstein(np.linspace(0.0, 1.0, n_points + 1), order)


def Bezierxy_t(ctrl_pts, n_points, t_order):
    """n_points + 1 points along a Bezier curve, evenly spaced in t."""
    return _bernstein_matrix(n_points, t_order) @ np.asarray(ctrl_pts, dtype=float)


def ctrl_pts(xi):
    """Control points of each candidate, LE to TE: (n, 5, 2)."""
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    Y1, X2, X3, Y3 = xi.T
    zero, one = np.zeros_like(Y1), np.ones_like(Y1)
    x = np.stack((zero, zero, X2, X3, one), axis=-1)
    y = np.stack((zero, Y1, Y1, Y3, zero), axis=-1)
    return np.stack((x, y), axis=-1)


def cross_section_points(y1, x2, x3, y3, bezier_sample):
    """bezier_sample + 1 points along the half-section (LE to TE), even in t."""
    B = Bezierxy_t(ctrl_pts([y1, x2, x3, y3])[0], bezier_sample, ORDER)
    return B[:, 0], B[:, 1]


def build_half_airfoil(xi, n_bez_samples):
    """Free variables [Y1, X2, X3, Y3] -> (x, y) of the half-section, LE to TE."""
    return cross_section_points(*np.asarray(xi, dtype=float), n_bez_samples)


def expand_ctrl_pts(xi, chord):
    """The 5 x 2 control polygon, LE to TE, scaled by chord (for plotting)."""
    return ctrl_pts(xi)[0] * chord


# --------------------------------------------------------------------------
# Exact evaluation at given x (used by the fit)
# --------------------------------------------------------------------------

def is_valid(xi):
    """True where x(t) rises monotonically from 0 to 1, so each x in [0, 1]
    has exactly one point on the curve.

    dx/dt = 4t [a (1-t)^2 + 2b t(1-t) + c t^2] with a = 3 X2,
    b = 1.5 (X3 - X2), c = 1 - X3, which stays >= 0 on [0, 1] exactly when
    a >= 0, c >= 0 and b >= -sqrt(a c)."""
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    a = 3.0 * xi[:, 1]
    b = 1.5 * (xi[:, 2] - xi[:, 1])
    c = 1.0 - xi[:, 2]
    return (a >= 0.0) & (c >= 0.0) & (b >= -np.sqrt(np.clip(a * c, 0.0, None)))


def _curve(xi, t):
    """x, y, dx/dt, dy/dt at t (n, m) for each candidate (rows of xi)."""
    Y1, X2, X3, Y3 = (xi[:, k:k + 1] for k in range(4))
    u = 1.0 - t
    t2, u2 = t * t, u * u
    x = 6.0 * X2 * t2 * u2 + 4.0 * X3 * t2 * t * u + t2 * t2
    y = Y1 * (4.0 * t * u2 * u + 6.0 * t2 * u2) + 4.0 * Y3 * t2 * t * u
    dx = 12.0 * X2 * t * u * (u - t) + 4.0 * X3 * t2 * (3.0 * u - t) + 4.0 * t2 * t
    dy = 4.0 * Y1 * u * (u2 - 3.0 * t2) + 4.0 * Y3 * t2 * (3.0 * u - t)
    return x, y, dx, dy


def t_at_x(xi, xq, tol=1e-14, max_iter=60):
    """Curve parameter t where x(t) = xq, for each candidate (rows of xi) and
    each station in xq (1-D, inside [0, 1]): (n, m).

    Newton's method kept inside a shrinking bracket (a bisection step whenever
    Newton would leave it), so it converges for every candidate."""
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    xq = np.asarray(xq, dtype=float).ravel()
    X2, X3 = xi[:, 1:2], xi[:, 2:3]
    shape = (xi.shape[0], xq.size)
    lo, hi = np.zeros(shape), np.ones(shape)
    t = np.broadcast_to(np.sqrt(np.clip(xq, 0.0, 1.0)), shape).copy()
    for _ in range(max_iter):
        u = 1.0 - t
        t2 = t * t
        f = 6.0 * X2 * t2 * u * u + 4.0 * X3 * t2 * t * u + t2 * t2 - xq
        dx = 12.0 * X2 * t * u * (u - t) + 4.0 * X3 * t2 * (3.0 * u - t) + 4.0 * t2 * t
        lo = np.where(f < 0.0, t, lo)
        hi = np.where(f > 0.0, t, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_new = t - f / dx
        t_new = np.where((t_new >= lo) & (t_new <= hi), t_new, 0.5 * (lo + hi))
        t_new = np.where(f == 0.0, t, t_new)      # exact hits, e.g. the LE (dx/dt = 0 there)
        step = np.max(np.abs(t_new - t))
        t = t_new
        if step <= tol:
            break
    return t


def half_airfoil_y(xi, xq):
    """Half-thickness at the stations xq for each candidate: (n, m)."""
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    return _curve(xi, t_at_x(xi, xq))[1]


def y_and_slope(xi, x):
    """Half-thickness y and slope dy/dx of one candidate at the stations x."""
    xi = np.asarray(xi, dtype=float).reshape(1, 4)
    _, y, dx, dy = _curve(xi, t_at_x(xi, np.atleast_1d(x)))
    return y[0], dy[0] / dx[0]


def half_airfoil_jac(xi, xq):
    """Derivatives of the half-thickness at the stations xq with respect to
    [Y1, X2, X3, Y3], for one candidate: (m, 4). At a fixed x the curve
    parameter moves with X2 and X3, which brings in the slope dy/dx."""
    xi = np.asarray(xi, dtype=float).reshape(1, 4)
    t = t_at_x(xi, xq)
    _, _, dx, dy = _curve(xi, t)
    t, dx, dy = t[0], dx[0], dy[0]
    u = 1.0 - t
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(t > 0.0, dy / dx, 0.0)       # the LE station does not move
    dx_dX2, dx_dX3 = 6.0 * t * t * u * u, 4.0 * t**3 * u
    return np.column_stack((4.0 * t * u**3 + 6.0 * t * t * u * u, -slope * dx_dX2,
                            -slope * dx_dX3, 4.0 * t**3 * u))


# --------------------------------------------------------------------------
# Rounded section from X1, X2, T1, T2 and r
# --------------------------------------------------------------------------

def mirror_selig(x, y):
    """Selig-ordered points (TE -> upper -> LE -> lower -> TE) of a symmetric
    section from its upper surface given LE to TE; the LE point appears once."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.vstack((np.column_stack((x, y))[::-1], np.column_stack((x[1:], -y[1:]))))


def cut_for_radius(xi, radius, x_from=None, x_to=1.0 - 1e-12):
    """Where on the half-section (sharp-chord units) the circle tangent to it,
    with its centre on the chord line, has the given radius: y sqrt(1 + slope^2)
    = radius. That quantity falls to 0 at the sharp TE. The search runs from
    x_from (default: the point of maximum thickness) to x_to, along the curve
    parameter, so no x -> t solve is needed inside it."""
    from scipy.optimize import brentq
    xi = np.asarray(xi, dtype=float).reshape(4)
    if not radius > 0.0:
        raise ValueError(f"the TE radius must be positive, got {radius}")
    if x_from is None:
        bx, by = build_half_airfoil(xi, 400)
        k = int(np.argmax(by))
        if radius >= by[k]:
            raise ValueError(f"r = {radius:g} is not below the half-thickness {by[k]:.5f}, "
                             "so no tangent circle fits")
        t_from = k / 400.0
    else:
        t_from = float(t_at_x(xi, [x_from])[0, 0])
    t_to = float(t_at_x(xi, [x_to])[0, 0])
    Y1, X2, X3, Y3 = (float(v) for v in xi)

    def gap(t):
        u = 1.0 - t
        t2, u2 = t * t, u * u
        y = Y1 * (4.0 * t * u2 * u + 6.0 * t2 * u2) + 4.0 * Y3 * t2 * t * u
        dx = 12.0 * X2 * t * u * (u - t) + 4.0 * X3 * t2 * (3.0 * u - t) + 4.0 * t2 * t
        dy = 4.0 * Y1 * u * (u2 - 3.0 * t2) + 4.0 * Y3 * t2 * (3.0 * u - t)
        return y * math.hypot(dx, dy) / dx - radius
    t = brentq(gap, t_from, t_to, xtol=1e-15)
    return float(_curve(xi[None], np.array([[t]]))[0][0, 0])


def assemble_rounded(xi, cut_pt, center, r, n_bez_samples=200, n_arc=24):
    """The half-section cut at cut_pt and closed round the TE by the arc of
    the circle (center, r), mirrored, in Selig order: TE point, upper arc,
    upper surface to the LE, lower surface, lower arc, TE point. Sharp-chord
    units."""
    bx, by = build_half_airfoil(xi, n_bez_samples)
    te_pt = np.asarray(center, dtype=float) + [r, 0.0]
    arc = make_fillet(center, cut_pt, n_arc)
    return np.vstack((te_pt, arc, trim_at(mirror_selig(bx, by), cut_pt),
                      arc[::-1] * [1.0, -1.0], te_pt))


def fillet_geometry(X1, X2, T1, T2, r):
    """Where the TE circle of radius r meets the section of the five
    thickness-format numbers (all fractions of the sharp chord): the circle
    tangent to the half-section with its centre on the chord line.

    Returns (cut_pt, center, r, x_te): the tangent point on the upper surface,
    the centre, the radius (r itself, to rounding) and the x of the rounded TE,
    center + r. Raises ValueError when X1 and X2 fold the curve back or r does
    not fit under the half-thickness."""
    xi = np.array([T1, X1, X2, T2], dtype=float)
    if not is_valid(xi)[0]:
        raise ValueError("X1 and X2 give a curve that folds back on itself")
    x_cut = cut_for_radius(xi, r)
    y_cut, slope = y_and_slope(xi, x_cut)
    cut_pt = np.array([x_cut, y_cut[0]])
    center, r = perp_method(cut_pt, slope[0])
    return cut_pt, center, r, center[0] + r


def sharp_chord(X1, X2, T1, T2, chord, r_abs, tol=1e-13, max_iter=50):
    """The sharp chord of the section whose TE is rounded with the radius r_abs
    (same units as chord) and whose rounded TE lies `chord` behind its LE.

    The five numbers are fractions of the sharp chord cs, so r = r_abs / cs,
    and the rounded TE sits at cs * x_te(r): cs = chord / x_te(r_abs / cs).
    x_te changes little with r, so this fixed point converges in a few steps.
    Returns (cs, r), with r = r_abs / cs the radius as rounded_section takes it.
    """
    cs = float(chord)
    for _ in range(max_iter):
        x_te = fillet_geometry(X1, X2, T1, T2, r_abs / cs)[3]
        cs_new = chord / x_te
        done = abs(cs_new - cs) <= tol * chord
        cs = cs_new
        if done:
            return cs, r_abs / cs
    raise RuntimeError(f"sharp_chord: no convergence in {max_iter} steps")


def _half_at_x(xi, cut_pt, center, r, x):
    """Half-thickness of the rounded section at the stations x (LE to TE,
    the last one the rounded TE): the Bezier up to the cut, the arc after."""
    y = np.empty_like(x)
    on_bezier = x <= cut_pt[0]
    y[on_bezier] = half_airfoil_y(xi, x[on_bezier])[0]
    y[~on_bezier] = np.sqrt(np.clip(r**2 - (x[~on_bezier] - center[0])**2, 0.0, None))
    y[-1] = 0.0                                  # the TE point itself (sqrt would leave ~1e-9)
    return y


def _cosine_stations(x_te, n):
    return x_te * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n))))


def rounded_section(X1, X2, T1, T2, r, chord=1.0, chord_type="sharp", n_per_side=None,
                    n_bez_samples=200, n_arc=24, num_pts=None):
    """The rounded section defined by the five thickness-format numbers.

    X1, X2      x of the control points P2 and P3          (the fit's X2, X3)
    T1, T2      their heights, half-thickness side         (the fit's Y1, Y3)
    r           TE radius
    All five are fractions of the sharp chord. The half-section is the quartic
    Bezier P0 (0,0), P1 (0,T1), P2 (X1,T1), P3 (X2,T2), P4 (1,0), mirrored for
    the lower surface, and its sharp TE is replaced by the circle of radius r
    tangent to it with its centre on the chord line (fillet_method 2's
    construction; it also rebuilds a fillet_method 0 fit exactly).

    chord       length to scale by
    chord_type  'sharp': chord is the sharp chord, LE to P4 (cs_mm in
                thickness_params_H80_info.dat); 'actual': chord runs from the LE
                to the rounded TE, like the extraction's chord distribution
    n_per_side  None: the Bezier samples upstream of the cut (n_bez_samples + 1
                along the whole half-section) plus n_arc points on each half of
                the arc. An integer n: n points per surface, LE to TE at
                cosine-spaced x, 2n - 1 points in all.
    num_pts     an integer n instead: n points in all, laid out as the
                extraction lays out its sections (Rudder_geom_extraction.
                _airfoil_coords): (n + 1) // 2 cosine-spaced stations on the
                upper surface, n - (n + 1) // 2 + 1 on the lower one, the LE
                shared, so the LE is at index (n + 1) // 2 - 1 and the TE first
                and last. The XCAD loops need that layout.

    Returns (m, 2) points in Selig order, TE -> upper -> LE -> lower -> TE: the
    TE point first and last, the LE point once, at (0, 0).
    """
    if n_per_side is not None and num_pts is not None:
        raise ValueError("give n_per_side or num_pts, not both")
    xi = np.array([T1, X1, X2, T2], dtype=float)
    cut_pt, center, r, x_te = fillet_geometry(X1, X2, T1, T2, r)
    if chord_type == "sharp":
        scale = chord
    elif chord_type == "actual":
        scale = chord / x_te
    else:
        raise ValueError("chord_type must be 'sharp' or 'actual'")
    if n_per_side is not None:
        x = _cosine_stations(x_te, n_per_side)
        pts = mirror_selig(x, _half_at_x(xi, cut_pt, center, r, x))
    elif num_pts is not None:
        n_half = (int(num_pts) + 1) // 2
        xu = _cosine_stations(x_te, n_half)
        xl = _cosine_stations(x_te, int(num_pts) - n_half + 1)
        yu = _half_at_x(xi, cut_pt, center, r, xu)
        yl = _half_at_x(xi, cut_pt, center, r, xl)
        pts = np.vstack((np.column_stack((xu, yu))[::-1], np.column_stack((xl[1:], -yl[1:]))))
    else:
        pts = assemble_rounded(xi, cut_pt, center, r, n_bez_samples, n_arc)
    return pts * scale + 0.0                     # + 0.0: no negative zeros


def read_thickness_params(path):
    """Read a thickness-format file (Section # | X1 | X2 | T1 | T2 | r, one
    section per line). Returns (sections, params): the section numbers and an
    (n, 5) array of X1, X2, T1, T2, r, ready for rounded_section(*params[k])."""
    secs, rows = [], []
    with open(path) as fh:
        for line in fh:
            tok = line.replace("|", " ").split()
            if len(tok) == 6 and tok[0].isdigit():
                secs.append(int(tok[0]))
                rows.append([float(v) for v in tok[1:]])
    return np.array(secs), np.array(rows)
