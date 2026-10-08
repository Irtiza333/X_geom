"""
suboff.py

The DARPA SUBOFF bare hull (DTRC model 5470, configuration AFF-1) from its
equations: Groves, N.C., Huang, T.T. and Chang, M.S. (1989), Geometric
characteristics of DARPA SUBOFF models (DTRC model nos. 5470 and 5471),
DTRC/SHD-1298-01, David Taylor Research Center. The equations are in feet with
x from the nose; this module works in metres (the model: length 4.3561 m,
diameter 0.508 m).

    x (ft)                   r (ft)
    0 .. 3.333333            forebody              Rmax [1.126395101 x a^4 + 0.442874707 x^2 a^3
                                                   + 1 - a^4 (1.2 x + 1)]^(1/2.1),   a = 0.3 x - 1
    3.333333 .. 10.645833    parallel middle body  Rmax
    10.645833 .. 13.979167   afterbody             Rmax [rh^2 + rh K0 xi^2 + (20 - 20 rh^2 - 4 rh K0 - K1/3) xi^3
                                                   + (-45 + 45 rh^2 + 6 rh K0 + K1) xi^4
                                                   + (36 - 36 rh^2 - 4 rh K0 - K1) xi^5
                                                   + (-10 + 10 rh^2 + rh K0 + K1/3) xi^6]^(1/2),
                                                   xi = (13.979167 - x) / 3.333333
    13.979167 .. 14.291667   aft cap               rh Rmax [1 - (3.2 x - 44.733333)^2]^(1/2)

    Rmax = 5/6 ft, rh = 0.1175, K0 = 10, K1 = 44.6244

The forebody and the afterbody meet the middle body with the same radius,
slope (0) and curvature (0); the afterbody meets the cap with the same radius
and slope (0), and the curvature jumps there. The report's bare hull: volume
0.699 m^3, wetted surface 5.988 m^2 (as tabulated in arXiv:2503.19674,
Table 1). The equations give 0.69921 m^3 and 5.98826 m^2.

    radius(x), slope(x)          r (m) and dr/dx at x (m from the nose)
    exact_hydrostatics()         volume, wetted surface, centre of buoyancy by quadrature
    stations(n)                  x and r at n stations (cosine spacing, the joins added)
    profile(n)                   the four parts as points along the curve, with end tangents (for the CAD)
"""

from __future__ import annotations

import numpy as np

FT = 0.3048                                             # m per ft
R_MAX = 5.0 / 6.0                                       # ft
X_FORE, X_MID, X_AFT, X_END = 3.333333, 10.645833, 13.979167, 14.291667   # ft: where each part ends
RH, K0, K1 = 0.1175, 10.0, 44.6244
LENGTH = X_END * FT                                     # 4.3561 m
DIAMETER = 2.0 * R_MAX * FT                             # 0.508 m
JOINS = np.array([0.0, X_FORE, X_MID, X_AFT, X_END]) * FT    # m: the nose, the three joins, the tail
PARTS = ("forebody", "parallel middle body", "afterbody", "aft cap")
REFERENCE = {"volume": 0.699, "wetted_surface": 5.988}  # m^3, m^2: the report's bare hull
SOURCE = "DARPA SUBOFF bare hull (AFF-1), equations of Groves, Huang and Chang (1989)"

_A = (20 - 20 * RH ** 2 - 4 * RH * K0 - K1 / 3, -45 + 45 * RH ** 2 + 6 * RH * K0 + K1,
      36 - 36 * RH ** 2 - 4 * RH * K0 - K1, -10 + 10 * RH ** 2 + RH * K0 + K1 / 3)   # afterbody: xi^3 .. xi^6


# --------------------------------------------------------------------------
# The parts (feet)
# --------------------------------------------------------------------------

def _fore(x):
    """Forebody: (r, dr/dx) in ft."""
    a, b = 0.3 * x - 1.0, 1.2 * x + 1.0
    f = np.clip(1.126395101 * x * a ** 4 + 0.442874707 * x ** 2 * a ** 3 + 1.0 - a ** 4 * b, 0.0, None)
    df = (1.126395101 * (a ** 4 + 1.2 * x * a ** 3) + 0.442874707 * (2.0 * x * a ** 3 + 0.9 * x ** 2 * a ** 2)
          - (1.2 * a ** 3 * b + 1.2 * a ** 4))
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = np.where(f > 0.0, R_MAX / 2.1 * f ** (1.0 / 2.1 - 1.0) * df, np.inf)
    return R_MAX * f ** (1.0 / 2.1), dr


def _aft(x):
    """Afterbody: (r, dr/dx) in ft."""
    xi = (X_AFT - x) / 3.333333
    g = RH ** 2 + RH * K0 * xi ** 2 + sum(c * xi ** (k + 3) for k, c in enumerate(_A))
    dg = 2.0 * RH * K0 * xi + sum((k + 3) * c * xi ** (k + 2) for k, c in enumerate(_A))
    g = np.clip(g, 0.0, None)
    return R_MAX * np.sqrt(g), -R_MAX * dg / (2.0 * np.sqrt(g)) / 3.333333


def _cap(x):
    """Aft cap: (r, dr/dx) in ft."""
    u = 3.2 * x - 44.733333
    h = np.clip(1.0 - u ** 2, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = np.where(h > 0.0, -RH * R_MAX * 3.2 * u / np.sqrt(h), -np.inf)
    return RH * R_MAX * np.sqrt(h), dr


def _part(x_ft):
    """Index of the part each x (ft) lies in: 0 .. 3, -1 outside the hull."""
    x = np.asarray(x_ft, dtype=float)
    k = np.searchsorted(np.array([X_FORE, X_MID, X_AFT]), x, side="left")
    return np.where((x < 0.0) | (x > X_END), -1, k)


def _eval(x, what):
    x_ft = np.atleast_1d(np.asarray(x, dtype=float)) / FT
    k = _part(x_ft)
    out = np.zeros_like(x_ft)
    for i, fn in enumerate((_fore, None, _aft, _cap)):
        m = k == i
        if not np.any(m):
            continue
        if fn is None:
            out[m] = R_MAX if what == 0 else 0.0
        else:
            out[m] = fn(x_ft[m])[what]
    out = out * FT if what == 0 else out
    return out if np.ndim(x) else float(out[0])


def radius(x):
    """r (m) at x (m from the nose); 0 outside the hull."""
    return _eval(x, 0)


def slope(x):
    """dr/dx at x (m from the nose): +inf at the nose, -inf at the tail, 0
    outside the hull. At a join, the part that ends there."""
    return _eval(x, 1)


# --------------------------------------------------------------------------
# Hydrostatics by quadrature
# --------------------------------------------------------------------------

def exact_hydrostatics():
    """Volume (m^3), wetted surface (m^2) and centre of buoyancy from the nose
    (m) of the equations, by adaptive quadrature over each part: the forebody
    in t with x = 3.333333 t^2.1 (r is then smooth in t at the nose) and the
    cap in the angle of its ellipse, so the integrands stay smooth where the
    slope is infinite. Also the length, diameter and the parallel middle body."""
    from scipy.integrate import quad

    def fore(t):                                        # (x, r, dx/dt, dr/dt) in ft
        x = X_FORE * t ** 2.1
        dx = 2.1 * X_FORE * t ** 1.1
        r, dr = _fore(np.array([x]))
        return x, float(r[0]), dx, float(dr[0]) * dx

    def after(x):
        r, dr = _aft(np.array([x]))
        return x, float(r[0]), 1.0, float(dr[0])

    th0 = np.arcsin(3.2 * X_AFT - 44.733333)

    def cap(th):                                        # u = 3.2 x - 44.733333 = sin(th)
        return ((np.sin(th) + 44.733333) / 3.2, RH * R_MAX * np.cos(th), np.cos(th) / 3.2,
                -RH * R_MAX * np.sin(th))

    parts = ((fore, 0.0, 1.0), (after, X_MID, X_AFT), (cap, th0, 0.5 * np.pi))
    vol, mom, area = 0.0, 0.0, 0.0
    for fn, a, b in parts:
        def dv(s, fn=fn):
            x, r, dx, _ = fn(s)
            return np.pi * r * r * dx

        def dm(s, fn=fn):
            x, r, dx, _ = fn(s)
            return x * np.pi * r * r * dx

        def ds(s, fn=fn):
            x, r, dx, dr = fn(s)
            return 2.0 * np.pi * r * np.hypot(dx, dr)
        kw = dict(limit=400, epsabs=0.0, epsrel=1.0e-13)
        vol += quad(dv, a, b, **kw)[0]
        mom += quad(dm, a, b, **kw)[0]
        area += quad(ds, a, b, **kw)[0]
    lp = X_MID - X_FORE                                 # the parallel middle body
    vol += np.pi * R_MAX ** 2 * lp
    mom += np.pi * R_MAX ** 2 * lp * 0.5 * (X_FORE + X_MID)
    area += 2.0 * np.pi * R_MAX * lp
    return {"length": LENGTH, "diameter": DIAMETER, "volume": vol * FT ** 3, "wetted_surface": area * FT ** 2,
            "lcb": mom / vol * FT, "pmb_start": X_FORE * FT, "pmb_end": X_MID * FT}


# --------------------------------------------------------------------------
# Stations and the profile for the CAD
# --------------------------------------------------------------------------

def stations(n=401):
    """(x, r) in m at n stations, cosine spacing from the nose to the tail,
    with the three joins added (so the middle body's ends are stations)."""
    x = np.union1d(LENGTH * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n)))), JOINS)
    r = radius(x)
    r[0] = r[-1] = 0.0
    return x, r


def _along_curve(a, b, n, dense=20001):
    """x of n points evenly spaced along the profile between a and b (m)."""
    xs = a + (b - a) * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, dense)))
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(radius(xs))))]
    x = np.interp(np.linspace(0.0, s[-1], n), s, xs)
    x[0], x[-1] = a, b
    return x


def profile(n=200):
    """The profile for the CAD, part by part: a list of (k, 2) arrays (x, r)
    in m, n points evenly spaced along the curve (the middle body: its two
    ends), neighbours sharing their end point; and the unit tangents (dx, dr)
    at the two ends of each part: vertical at the nose and the tail, along x
    at the three joins (the report's joins have zero slope)."""
    r_ends = np.array([0.0, R_MAX, R_MAX, RH * R_MAX, 0.0]) * FT     # r at the nose, the joins, the tail
    segs = []
    for k in range(4):
        a, b = JOINS[k], JOINS[k + 1]
        x = np.array([a, b]) if k == 1 else _along_curve(a, b, n)
        r = radius(x)
        r[0], r[-1] = r_ends[k], r_ends[k + 1]            # the same end points in both neighbours
        segs.append(np.column_stack((x, r)))
    tangents = [((0.0, 1.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, 0.0)),
                ((1.0, 0.0), (0.0, -1.0))]
    return segs, tangents
