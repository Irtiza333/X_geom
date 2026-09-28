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
"""

from functools import lru_cache
from math import comb

import numpy as np

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
