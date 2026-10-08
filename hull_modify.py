"""
hull_modify.py

A hull with elliptical sections (submarines, UUVs, torpedoes, airships, fuselages) described by a few smooth
curves, which are the design variables. A hull starts from a station table (a parameter file: SUBOFF's from its
equations, or any table of half-heights and half-widths), the curves are fitted to it, and any design gives its
stations, hydrostatics, section points and CAD.

The body along x (m, from the nose), in four parts:
    nose     0 .. Ln          the half-height r and half-width r' rise from 0 to r* and r'*
    middle   Ln .. Ln + Lm    constant section r*, r'* (the parallel middle body; Lm may be 0)
    tail     .. + Lt          they fall from r*, r'* to the end radii re, re'
    cap      .. + Lc          an ellipsoidal cap closing the end (SUBOFF's); Lc = 0: the end is flat (re > 0)
                              or on the axis (re = 0)
Each section is an ellipse, r' along y (the XY view) and r along z (the XZ view); a circle where they are equal.

Curves (CST: a class function times a Bernstein polynomial, Kulfan 2008), one per part and view:
    nose  r = r* psi^N1 sum_{i=0..n} b_i B_i,n(psi),                       psi = x / Ln
          b_n = 1 and b_(n-1) = 1 + N1/n give r = r* with zero slope at the middle body
          N1: 0.5 a round nose (tip radius (r* b_0)^2 / (2 Ln)), 1 a pointed one (half-angle atan(r* b_0 / Ln))
    tail  r = re + (r* - re) (1 - psi)^N2 sum_{i=0..n} t_i B_i,n(psi),    psi from the middle body (0) to the end
          t_0 = 1 and t_1 = 1 + N2/n give r = r* with zero slope at the middle body
          N2: 2 zero slope at the end (a hub; SUBOFF's tail), 1 a cone, below 1 a rounded end
A curve's values are its exponent and its free coefficients (nose b_0 .. b_(n-2), tail t_2 .. t_n): n values for
order n. Its control points, drawn in the plots, are the coefficients at psi = i/n times the class function,
scaled to radii. An axisymmetric hull (SUBOFF) has the r curves only (r' = r).

Parameter file (a station table; read_params, write_params):
    # XGeom hull parameters
    # nose_m        1.016
    # middle_m      2.22885
    # tail_m        1.016
    # cap_m         0.09525           optional, default 0
    # x_m  r_m  rp_m                  rp_m optional: no column means axisymmetric
    0.000000  0.000000  0.000000
    ...
The fit densifies each part of the table (monotone cubic, the nose in sqrt(x)), fits the coefficients by least
squares and each exponent by a one-dimensional search.

Outputs (write_case, outputs/hull/): <case>_params.dat (the design as a station table), <case>_design.json (the
design exactly), <case>_design_space.json (with a space), <case>_stations.dat and <case>_hydrostatics.dat
(hull_extraction), <case>_sections_m.dat (section points: '#k', then x y z per point, m), <case>_check.png, and
with OCC <case>.step: one NURBS face per part, the profile B-spline times the rational circle (exact ellipses).

    python hull_modify.py --suboff                 fit SUBOFF (suboff_hull_params.dat) and write the case
    python hull_modify.py params.dat [--case NAME] [--order 5] [--no-cad]
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from dataclasses import dataclass, field
from math import comb

import numpy as np

import hull_extraction as HE
import xcad_loft as XL

OUT_DIR = os.path.join("outputs", "hull")
KINDS = ("nose", "tail")
VIEWS = ("r", "rp")                       # half-height (XZ view), half-width (XY view)
DEFAULT_ORDER = 5                         # a curve's values: its exponent and n - 1 coefficients
MAX_ORDER = 10
EXP_BOUNDS = {"nose": (0.3, 1.5), "tail": (0.3, 3.0)}
EXP_KEY = {"nose": "N1", "tail": "N2"}
LENGTHS = ("nose", "middle", "tail", "cap")
SECTION_POINTS = 64                       # points per section in the section file
FIT_POINTS = 600                          # points per part in a fit, evenly spaced along the curve
SUBOFF_PARAMS = "suboff_hull_params.dat"


# --------------------------------------------------------------------------
# CST curves
# --------------------------------------------------------------------------

def bernstein(n, u):
    """(len(u), n + 1) Bernstein basis of degree n at u in [0, 1]."""
    u = np.asarray(u, dtype=float)[:, None]
    i = np.arange(n + 1)
    return np.array([comb(n, k) for k in i], dtype=float) * u ** i * (1.0 - u) ** (n - i)


def bernstein_d(n, u):
    """d/du of the Bernstein basis of degree n."""
    b = bernstein(n - 1, u)
    z = np.zeros((len(b), 1))
    return n * (np.hstack((z, b)) - np.hstack((b, z)))


class CSTCurve:
    """A nose or tail curve, normalized: the nose gives r / r*, the tail (r - re) / (r* - re), against psi in
    [0, 1] (see the module docstring). exponent: N1 or N2; free: the free coefficients."""

    def __init__(self, kind, exponent, free):
        if kind not in KINDS:
            raise ValueError(f"kind {kind!r}: nose or tail")
        self.kind = kind
        self.exponent = float(exponent)
        self.free = np.asarray(free, dtype=float).copy()
        if len(self.free) < 1:
            raise ValueError("a curve needs at least one free coefficient (order 2)")

    @property
    def order(self):
        return len(self.free) + 1

    def coeffs(self):
        n, e = self.order, self.exponent
        if self.kind == "nose":
            return np.r_[self.free, 1.0 + e / n, 1.0]
        return np.r_[1.0, 1.0 + e / n, self.free]

    def _cls(self, psi, d=False):
        e = self.exponent
        if self.kind == "nose":
            if d:
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.where(psi > 0.0, e * psi ** (e - 1.0), np.inf if e < 1.0 else float(e == 1.0))
            return psi ** e
        if d:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(psi < 1.0, -e * (1.0 - psi) ** (e - 1.0), -np.inf if e < 1.0 else -float(e == 1.0))
        return (1.0 - psi) ** e

    def shape(self, psi):
        psi = np.clip(np.atleast_1d(np.asarray(psi, dtype=float)), 0.0, 1.0)
        return self._cls(psi) * (bernstein(self.order, psi) @ self.coeffs())

    def dshape(self, psi):
        """d shape / d psi (+-inf where the class function's slope is)."""
        psi = np.clip(np.atleast_1d(np.asarray(psi, dtype=float)), 0.0, 1.0)
        c = self.coeffs()
        s, ds = bernstein(self.order, psi) @ c, bernstein_d(self.order, psi) @ c
        with np.errstate(invalid="ignore"):
            out = self._cls(psi, True) * s + self._cls(psi) * ds
        return np.where(np.isnan(out), 0.0, out)

    def ctrl(self):
        """Control points (psi_i, value_i), normalized: the coefficients at psi = i/n times the class function."""
        n = self.order
        psi = np.arange(n + 1) / n
        return psi, self._cls(psi) * self.coeffs()

    def keys(self):
        return self.keys_of(self.kind, self.order)

    @staticmethod
    def keys_of(kind, order):
        if kind == "nose":
            return [EXP_KEY[kind]] + [f"b{i}" for i in range(order - 1)]
        return [EXP_KEY[kind]] + [f"t{i}" for i in range(2, order + 1)]

    def get(self, key):
        return self.exponent if key == EXP_KEY[self.kind] else float(self.free[self.keys().index(key) - 1])

    @classmethod
    def from_keys(cls, kind, vals, order):
        keys = cls.keys_of(kind, order)
        return cls(kind, vals[keys[0]], [vals[k] for k in keys[1:]])

    def copy(self):
        return CSTCurve(self.kind, self.exponent, self.free)

    def to_dict(self):
        return {"kind": self.kind, "order": self.order, "exponent": self.exponent, "free": self.free.tolist(),
                "coefficients": self.coeffs().tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["kind"], d["exponent"], d["free"])


def _lsq(kind, psi, y, order, e):
    n = order
    B = bernstein(n, psi)
    if kind == "nose":
        cls = psi ** e
        rhs = y - cls * (B[:, n - 1] * (1.0 + e / n) + B[:, n])
        A = cls[:, None] * B[:, :n - 1]
    else:
        cls = (1.0 - psi) ** e
        rhs = y - cls * (B[:, 0] + B[:, 1] * (1.0 + e / n))
        A = cls[:, None] * B[:, 2:]
    sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return sol, float(np.sqrt(np.mean((A @ sol - rhs) ** 2)))


def fit_cst(kind, psi, y, order=DEFAULT_ORDER, exponent=None):
    """The CST curve of an order through normalized data (psi, y): the coefficients by least squares, the
    exponent by a bounded search on the residual unless it is given. Returns (curve, rms residual)."""
    from scipy.optimize import minimize_scalar
    psi, y = np.asarray(psi, dtype=float), np.asarray(y, dtype=float)
    if exponent is None:
        lo, hi = EXP_BOUNDS[kind]
        res = minimize_scalar(lambda e: _lsq(kind, psi, y, order, e)[1], bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-6})
        exponent = float(res.x)
    sol, rms = _lsq(kind, psi, y, order, exponent)
    return CSTCurve(kind, exponent, sol), rms


# --------------------------------------------------------------------------
# The design
# --------------------------------------------------------------------------

@dataclass
class HullDesign:
    nose: float                  # Ln, m
    middle: float                # Lm, m (the parallel middle body)
    tail: float                  # Lt, m
    cap: float                   # Lc, m (0: none)
    r: float                     # r*, the largest half-height, m
    re: float                    # the tail end's half-height, m
    curves: dict                 # "nose_r", "tail_r", and unless axisymmetric "nose_rp", "tail_rp": CSTCurve
    rp: float = None             # r'*, the largest half-width, m (axisymmetric: r)
    rpe: float = None            # the tail end's half-width, m (axisymmetric: re)

    @property
    def axisymmetric(self):
        return "nose_rp" not in self.curves

    @property
    def length(self):
        return self.nose + self.middle + self.tail + self.cap

    @property
    def joins(self):
        """x of the nose, the middle body's two ends, the tail end and the end of the body (m)."""
        a = self.nose
        b = a + self.middle
        c = b + self.tail
        return np.array([0.0, a, b, c, c + self.cap])

    def copy(self):
        return HullDesign(self.nose, self.middle, self.tail, self.cap, self.r, self.re,
                          {k: c.copy() for k, c in self.curves.items()}, self.rp, self.rpe)

    def radii(self, view):
        """(largest, end) radius of a view, m."""
        if view == "r" or self.axisymmetric:
            return self.r, self.re
        return self.rp, self.rpe

    def curve(self, view, kind):
        return self.curves[f"{kind}_{'r' if self.axisymmetric else view}"]

    def radius(self, view, x):
        """Half-height (view 'r') or half-width ('rp') at x (m); 0 outside the body."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        top, end = self.radii(view)
        j = self.joins
        out = np.zeros_like(x)
        m = (x >= 0.0) & (x <= j[1])
        if np.any(m) and self.nose > 0:
            out[m] = top * self.curve(view, "nose").shape(x[m] / self.nose)
        m = (x > j[1]) & (x <= j[2])
        out[m] = top
        m = (x > j[2]) & (x <= j[3])
        if np.any(m):
            out[m] = end + (top - end) * self.curve(view, "tail").shape((x[m] - j[2]) / self.tail)
        if self.cap > 0:
            m = (x > j[3]) & (x <= j[4])
            out[m] = end * np.sqrt(np.clip(1.0 - ((x[m] - j[3]) / self.cap) ** 2, 0.0, None))
        return out

    def slope(self, view, x):
        """d radius / dx (m/m) at x; at a join, the part that ends there."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        top, end = self.radii(view)
        j = self.joins
        out = np.zeros_like(x)
        m = (x >= 0.0) & (x <= j[1])
        if np.any(m) and self.nose > 0:
            out[m] = top / self.nose * self.curve(view, "nose").dshape(x[m] / self.nose)
        m = (x > j[2]) & (x <= j[3])
        if np.any(m):
            out[m] = (top - end) / self.tail * self.curve(view, "tail").dshape((x[m] - j[2]) / self.tail)
        if self.cap > 0:
            m = (x > j[3]) & (x <= j[4])
            s = (x[m] - j[3]) / self.cap
            with np.errstate(divide="ignore", invalid="ignore"):
                out[m] = np.where(s < 1.0, -end * s / (self.cap * np.sqrt(np.clip(1.0 - s * s, 0.0, None))), -np.inf)
        return out

    def ctrl(self, view, kind):
        """Control points of a curve in m: (x, radius)."""
        top, end = self.radii(view)
        psi, v = self.curve(view, kind).ctrl()
        if kind == "nose":
            return psi * self.nose, top * v
        return self.joins[2] + psi * self.tail, end + (top - end) * v


# --------------------------------------------------------------------------
# Parameter file (a station table)
# --------------------------------------------------------------------------

@dataclass
class HullParams:
    nose: float
    middle: float
    tail: float
    cap: float
    x: np.ndarray                # m from the nose
    r: np.ndarray                # half-height, m
    rp: np.ndarray               # half-width, m
    axisymmetric: bool
    path: str = None
    notes: list = field(default_factory=list)

    @property
    def length(self):
        return self.nose + self.middle + self.tail + self.cap

    @property
    def joins(self):
        a = self.nose
        b = a + self.middle
        c = b + self.tail
        return np.array([0.0, a, b, c, c + self.cap])

    def values(self, view):
        return self.r if view == "r" or self.axisymmetric else self.rp


def read_params(path):
    """The HullParams of a parameter file (see the module docstring)."""
    head, rows = {}, []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                parts = s[1:].split()
                if len(parts) >= 2 and parts[0] in ("nose_m", "middle_m", "tail_m", "cap_m", "axisymmetric"):
                    head[parts[0]] = parts[1]
                continue
            rows.append([float(v) for v in s.split()])
    missing = [k for k in ("nose_m", "middle_m", "tail_m") if k not in head]
    if missing:
        raise ValueError(f"{path}: no {', '.join(missing)} in the header")
    tab = np.array(rows, dtype=float)
    if tab.ndim != 2 or tab.shape[1] not in (2, 3) or len(tab) < 4:
        raise ValueError(f"{path}: the table needs columns x_m r_m [rp_m] and at least 4 rows")
    if np.any(np.diff(tab[:, 0]) <= 0):
        raise ValueError(f"{path}: x must rise from row to row")
    ln, lm, lt = (float(head[k]) for k in ("nose_m", "middle_m", "tail_m"))
    lc = float(head.get("cap_m", 0.0))
    if ln <= 0 or lt <= 0 or lm < 0 or lc < 0:
        raise ValueError(f"{path}: nose and tail lengths must be positive, middle and cap not negative")
    length = ln + lm + lt + lc
    if abs(tab[0, 0]) > 1e-6 * length or abs(tab[-1, 0] - (length - lc)) > 1e-6 * length and \
            abs(tab[-1, 0] - length) > 1e-6 * length:
        raise ValueError(f"{path}: the table must run from x = 0 to the end of the tail (or of the cap)")
    rp = tab[:, 2] if tab.shape[1] == 3 else tab[:, 1].copy()
    axi = tab.shape[1] == 2 or bool(int(head.get("axisymmetric", "0"))) or np.allclose(rp, tab[:, 1], atol=1e-12)
    return HullParams(ln, lm, lt, lc, tab[:, 0], tab[:, 1], rp, axi, os.path.abspath(path))


def write_params(path, params, notes=()):
    """Write a parameter file (read_params reads it back)."""
    lines = ["# XGeom hull parameters"] + [f"# {n}" for n in list(params.notes) + list(notes)]
    lines += [f"# nose_m        {params.nose:.9g}", f"# middle_m      {params.middle:.9g}",
              f"# tail_m        {params.tail:.9g}", f"# cap_m         {params.cap:.9g}",
              f"# axisymmetric  {int(params.axisymmetric)}"]
    if params.axisymmetric:
        lines.append("# x_m  r_m")
        lines += [f"{x:.9f}  {r:.9f}" for x, r in zip(params.x, params.r)]
    else:
        lines.append("# x_m  r_m  rp_m")
        lines += [f"{x:.9f}  {r:.9f}  {rp:.9f}" for x, r, rp in zip(params.x, params.r, params.rp)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def suboff_params(n=101):
    """SUBOFF's parameter file from its equations (suboff.py): n stations per curved part (cosine spacing), the
    middle body's two ends, axisymmetric; the afterbody is the tail and SUBOFF's ellipsoidal end its cap."""
    import suboff as S
    j = S.JOINS
    t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n))))
    x = np.unique(np.concatenate([j[0] + (j[1] - j[0]) * t, j[2] + (j[3] - j[2]) * t,
                                  j[3] + (j[4] - j[3]) * t[n // 2:]]))
    r = S.radius(x)
    r[0], r[-1] = 0.0, 0.0
    return HullParams(float(j[1] - j[0]), float(j[2] - j[1]), float(j[3] - j[2]), float(j[4] - j[3]),
                      x, r, r.copy(), True, None, [S.SOURCE])


# --------------------------------------------------------------------------
# Fitting a design to a station table
# --------------------------------------------------------------------------

def _along(x, r, n):
    """n points evenly spaced along the polyline (x, r)."""
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(r)))]
    u = np.linspace(0.0, s[-1], n)
    return np.interp(u, s, x), np.interp(u, s, r)


def region_data(params, view, kind, n=FIT_POINTS):
    """Normalized data of one part for a fit: (psi, y, largest radius, end radius). The part's stations are
    densified by a monotone cubic (the nose in sqrt(x), which keeps a round nose round), then spaced evenly
    along the curve."""
    from scipy.interpolate import PchipInterpolator
    x, v = params.x, params.values(view)
    j = params.joins
    mid = (x >= j[1] - 1e-9) & (x <= j[2] + 1e-9)
    top = float(v[mid].max()) if np.any(mid) else float(np.interp(j[1], x, v))
    end = float(np.interp(j[3], x, v))
    dense = np.linspace(0.0, 1.0, 20001)
    if kind == "nose":
        m = x <= j[1] + 1e-9
        xs, vs = x[m], v[m]
        if xs[-1] < j[1] - 1e-9:
            xs, vs = np.r_[xs, j[1]], np.r_[vs, top]
        q = np.sqrt(np.clip(xs / j[1], 0.0, 1.0))
        q, idx = np.unique(q, return_index=True)
        f = PchipInterpolator(q, vs[idx])
        xd, vd = j[1] * dense ** 2, f(dense)
        xd, vd = _along(xd, vd, n)
        return xd / params.nose, vd / top, top, end
    m = (x >= j[2] - 1e-9) & (x <= j[3] + 1e-9)
    xs, vs = x[m], v[m]
    if xs[0] > j[2] + 1e-9:
        xs, vs = np.r_[j[2], xs], np.r_[top, vs]
    if xs[-1] < j[3] - 1e-9:
        xs, vs = np.r_[xs, j[3]], np.r_[vs, end]
    f = PchipInterpolator(xs, vs)
    xd = j[2] + (j[3] - j[2]) * dense
    xd, vd = _along(xd, f(xd), n)
    return (xd - j[2]) / params.tail, (vd - end) / (top - end), top, end


def fit_design(params, orders=None, exponents=None):
    """The design fitted to a parameter file: each curve of its order (default DEFAULT_ORDER; orders and
    exponents by curve name, e.g. {"nose_r": 6}; an exponent given is held in the fit)."""
    orders, exponents = orders or {}, exponents or {}
    views = ("r",) if params.axisymmetric else VIEWS
    curves, radii = {}, {}
    for view in views:
        for kind in KINDS:
            name = f"{kind}_{view}"
            psi, y, top, end = region_data(params, view, kind)
            curves[name], _ = fit_cst(kind, psi, y, int(orders.get(name, DEFAULT_ORDER)), exponents.get(name))
            radii[view] = (top, end)
    d = HullDesign(params.nose, params.middle, params.tail, params.cap, radii["r"][0], radii["r"][1], curves)
    if not params.axisymmetric:
        d.rp, d.rpe = radii["rp"]
    return d


def refit_curve(design, name, order, exponent=None, n=FIT_POINTS):
    """A curve of another order fitted to the design's present curve (its shape kept as closely as the order
    allows)."""
    kind = name.split("_")[0]
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    c = design.curves[name]
    return fit_cst(kind, psi, c.shape(psi), int(order), c.exponent if exponent is None else exponent)[0]


def fit_report(params, design):
    """{curve name: largest difference (m) between the design and the table's stations in that part}."""
    out = {}
    j = params.joins
    for view in (("r",) if design.axisymmetric else VIEWS):
        v = params.values(view)
        for kind, (a, b) in (("nose", (j[0], j[1])), ("tail", (j[2], j[3]))):
            m = (params.x >= a) & (params.x <= b)
            out[f"{kind}_{view}"] = float(np.abs(design.radius(view, params.x[m]) - v[m]).max())
    return out


# --------------------------------------------------------------------------
# The design space
# --------------------------------------------------------------------------

def global_slots(design):
    return list(LENGTHS) + (["r", "re"] if design.axisymmetric else ["r", "rp", "re", "rpe"])


def default_bounds(design):
    """Bounds by slot: nose and tail lengths 50 % to 150 %, the middle body 0 to 150 % (0 to half the nose
    when there is none), the cap 0 to 200 % (0 to 10 % of the tail when there is none); r* and r'* 75 % to
    125 %; the end radii 0 to 200 % (0 to 25 % of the largest when 0); to the mm; the exponents EXP_BOUNDS;
    each coefficient its value +-0.5, not below 0 (positive coefficients keep the radii positive)."""
    b = {}
    when_zero = {"middle": lambda: 0.5 * design.nose / 1.5, "cap": lambda: 0.05 * design.tail,
                 "re": lambda: 0.125 * design.r, "rpe": lambda: 0.125 * design.rp}
    for g, (f0, f1) in (("nose", (0.5, 1.5)), ("tail", (0.5, 1.5)), ("middle", (0.0, 1.5)), ("cap", (0.0, 2.0)),
                        ("r", (0.75, 1.25)), ("rp", (0.75, 1.25)), ("re", (0.0, 2.0)), ("rpe", (0.0, 2.0))):
        if g in ("rp", "rpe") and design.axisymmetric:
            continue
        x = float(getattr(design, g))
        if x == 0.0 and g in when_zero:
            x = float(when_zero[g]())
        b[g] = (float(np.floor(f0 * x * 1000.0) / 1000.0), float(np.ceil(f1 * x * 1000.0) / 1000.0))
    for name, c in design.curves.items():
        b[f"{name}.{EXP_KEY[c.kind]}"] = EXP_BOUNDS[c.kind]
        for k in c.keys()[1:]:
            v = c.get(k)
            b[f"{name}.{k}"] = (max(0.0, round(v - 0.5, 4)), round(v + 0.5, 4))
    return b


DEFAULT_FREE = ("nose", "middle", "tail", "nose_r", "tail_r", "nose_rp", "tail_rp")


class HullSpace:
    """A flat design vector for an optimiser and the HullDesign it stands for.

    The slots: the lengths nose, middle, tail, cap (m), the radii r, re (and rp, rpe unless axisymmetric, m),
    then curve by curve its exponent and free coefficients ("nose_r.N1", "nose_r.b0", ..., "tail_r.N2",
    "tail_r.t2", ...). The curves' orders and the axisymmetry are the design's.

    bounds   overrides of default_bounds by slot
    fixed    {slot: value} of the held slots (not in the vector); with free given instead (globals and curve
             names, default DEFAULT_FREE), every other slot is held at the design's value
    """

    def __init__(self, design, bounds=None, fixed=None, free=None):
        self.orders = {k: c.order for k, c in design.curves.items()}
        self.axisymmetric = design.axisymmetric
        b = {**default_bounds(design), **(bounds or {})}
        self._slots = [(None, g) for g in global_slots(design)]
        for name in sorted(design.curves, key=lambda k: (k.split("_")[1], KINDS.index(k.split("_")[0]))):
            self._slots += [(name, k) for k in design.curves[name].keys()]
        self.all_names = [k if c is None else f"{c}.{k}" for c, k in self._slots]
        self.all_bounds = np.array([b[n] for n in self.all_names], dtype=float)
        if fixed is None:
            free = DEFAULT_FREE if free is None else free
            vals = self.values(design)
            fixed = {n: vals[n] for n, (c, k) in zip(self.all_names, self._slots) if (c or k) not in free}
        self.fixed = {str(n): float(v) for n, v in fixed.items()}
        unknown = sorted(set(self.fixed) - set(self.all_names))
        if unknown:
            raise ValueError(f"fixed slots that this space does not have: {unknown}")
        self._free = np.array([i for i, n in enumerate(self.all_names) if n not in self.fixed], dtype=int)
        self.names = [self.all_names[i] for i in self._free]
        self.bounds = self.all_bounds[self._free].reshape(-1, 2)

    def __len__(self):
        return len(self.names)

    def values(self, design):
        """{slot name: value} of a design, every slot."""
        return {n: float(getattr(design, k)) if c is None else design.curves[c].get(k)
                for (c, k), n in zip(self._slots, self.all_names)}

    def to_vector(self, design):
        vals = self.values(design)
        return np.array([vals[n] for n in self.names])

    def full_vector(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self.names),):
            raise ValueError(f"the design vector has {len(self.names)} entries")
        full = np.array([self.fixed.get(n, np.nan) for n in self.all_names])
        full[self._free] = x
        return full

    def to_design(self, x):
        return self.design_of(dict(zip(self.all_names, self.full_vector(x))))

    def design_of(self, vals):
        """The HullDesign of {slot name: value} for every slot."""
        curves = {}
        for name, order in self.orders.items():
            kind = name.split("_")[0]
            curves[name] = CSTCurve.from_keys(kind, {k: vals[f"{name}.{k}"] for k in
                                                     CSTCurve.keys_of(kind, order)}, order)
        d = HullDesign(vals["nose"], vals["middle"], vals["tail"], vals["cap"], vals["r"], vals["re"], curves)
        if not self.axisymmetric:
            d.rp, d.rpe = vals["rp"], vals["rpe"]
        return d

    def to_dict(self, design):
        vals = self.values(design)
        return {"axisymmetric": self.axisymmetric, "orders": self.orders,
                "slots": [{"name": n, "value": vals[n], "lo": float(lo), "hi": float(hi), "free": n not in self.fixed}
                          for n, (lo, hi) in zip(self.all_names, self.all_bounds)],
                "vector": {"names": self.names, "values": [vals[n] for n in self.names],
                           "bounds": self.bounds.tolist()}}

    @classmethod
    def from_dict(cls, rec):
        """(space, design) from to_dict's output."""
        vals = {q["name"]: q["value"] for q in rec["slots"]}
        curves = {}
        for name, order in rec["orders"].items():
            kind = name.split("_")[0]
            curves[name] = CSTCurve.from_keys(kind, {k: vals[f"{name}.{k}"] for k in
                                                     CSTCurve.keys_of(kind, int(order))}, int(order))
        d = HullDesign(vals["nose"], vals["middle"], vals["tail"], vals["cap"], vals["r"], vals["re"], curves)
        if not rec["axisymmetric"]:
            d.rp, d.rpe = vals["rp"], vals["rpe"]
        space = cls(d, {q["name"]: (q["lo"], q["hi"]) for q in rec["slots"]},
                    {q["name"]: q["value"] for q in rec["slots"] if not q["free"]})
        return space, d


def space_record(space, design, params=None, settings=None):
    return {"geometry": "hull", "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "params_file": getattr(params, "path", None), "settings": dict(settings or {}),
            "space": space.to_dict(design)}


def load_hull_space(path):
    """(space, design, record) of a saved set-up (<case>_design_space.json)."""
    with open(path) as fh:
        rec = json.load(fh)
    space, design = HullSpace.from_dict(rec["space"])
    return space, design, rec


def design_record(design):
    """The design as plain data (exact)."""
    rec = {"geometry": "hull", "axisymmetric": design.axisymmetric,
           "lengths_m": {k: float(getattr(design, k)) for k in LENGTHS}, "length_m": float(design.length),
           "radii_m": {"r": design.r, "re": design.re},
           "curves": {k: c.to_dict() for k, c in design.curves.items()}}
    if not design.axisymmetric:
        rec["radii_m"].update({"rp": design.rp, "rpe": design.rpe})
    return rec


def design_from_record(rec):
    curves = {k: CSTCurve.from_dict(c) for k, c in rec["curves"].items()}
    L, R = rec["lengths_m"], rec["radii_m"]
    d = HullDesign(L["nose"], L["middle"], L["tail"], L["cap"], R["r"], R["re"], curves)
    if not rec["axisymmetric"]:
        d.rp, d.rpe = R["rp"], R["rpe"]
    return d


def read_design(path):
    with open(path) as fh:
        return design_from_record(json.load(fh))


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def stations_x(design, n=121):
    """x of n stations per part (cosine spacing in the nose, tail and cap), the joins included."""
    j = design.joins
    t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n))))
    xs = [j[0] + (j[1] - j[0]) * t, np.linspace(j[1], j[2], 5), j[2] + (j[3] - j[2]) * t]
    if design.cap > 0:
        xs.append(j[3] + (j[4] - j[3]) * np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, int(n))))
    return np.unique(np.concatenate(xs))


def ellipse_girth(a, b):
    """Perimeter of ellipses with semi-axes a, b (Ramanujan's second formula)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(s > 0, ((a - b) / s) ** 2, 0.0)
    return np.pi * s * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h)))


def stations(design, n=121):
    """hull_extraction.Stations of a design (ellipse areas and girths)."""
    x = stations_x(design, n)
    r, rp = design.radius("r", x), design.radius("rp", x)
    return HE.Stations(x, np.pi * r * rp, ellipse_girth(r, rp), 2.0 * rp, 2.0 * r, float(design.length), None,
                       "hull_modify design")


def hydrostatics(design, n=401):
    """hull_extraction.hydrostatics of a design from n stations per part; the middle body's ends exact. S is
    exact for circles and approximate for ellipses (the solid gives it exactly: write_cad)."""
    j = design.joins
    return HE.hydrostatics(stations(design, n), exact={"pmb_start": j[1], "pmb_end": j[2]})


def section_points(design, x, k=SECTION_POINTS):
    """(len(x), k, 3) points of the elliptical sections at x: (x, y, z) = (x, r' cos t, r sin t), t from 0."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t = np.linspace(0.0, 2.0 * np.pi, int(k), endpoint=False)
    r, rp = design.radius("r", x), design.radius("rp", x)
    return np.stack([np.repeat(x[:, None], len(t), 1), rp[:, None] * np.cos(t), r[:, None] * np.sin(t)], axis=2)


def write_sections(path, design, n=61, k=SECTION_POINTS):
    """The sections as points: '#k' then x y z per point (m), nose to tail, the closed ends as single points."""
    x = stations_x(design, n)
    pts = section_points(design, x, k)
    lines = []
    for i, (xi, p) in enumerate(zip(x, pts), start=1):
        lines.append(f"#{i}")
        if np.abs(p[:, 1:]).max() < 1e-12:
            p = p[:1]
        lines += [f"{a:.9f} {b:.9f} {c:.9f}" for a, b, c in p]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _profile_parts(design, n):
    """The profile part by part for the CAD: [(points (k, 3) of (x, r', r), tangent at start, tangent at end)]."""
    j = design.joins
    parts = []

    def pts_of(a, b, k):
        xs = a + (b - a) * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 4001)))
        rr, rp = design.radius("r", xs), design.radius("rp", xs)
        s = np.r_[0.0, np.cumsum(np.sqrt(np.diff(xs) ** 2 + np.diff(rr) ** 2 + np.diff(rp) ** 2))]
        x = np.interp(np.linspace(0.0, s[-1], k), s, xs)
        x[0], x[-1] = a, b
        return x

    def tangent(x0, x1):
        """direction from the point at x0 towards a point just beside it at x1"""
        p0 = np.array([x0, design.radius("rp", x0)[0], design.radius("r", x0)[0]])
        p1 = np.array([x1, design.radius("rp", x1)[0], design.radius("r", x1)[0]])
        d = p1 - p0
        return d / np.linalg.norm(d)

    eps = 1e-9 * design.length
    x = pts_of(j[0], j[1], n)
    p = np.column_stack((x, design.radius("rp", x), design.radius("r", x)))
    p[0, 1:] = 0.0
    top_r, end_r = design.radii("r")
    top_p, end_p = design.radii("rp")
    p[-1, 1:] = (top_p, top_r)
    parts.append((p, tangent(j[0], j[0] + eps), (1.0, 0.0, 0.0)))
    if design.middle > 0:
        x = np.linspace(j[1], j[2], 3)
        parts.append((np.column_stack((x, np.full(3, top_p), np.full(3, top_r))), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    x = pts_of(j[2], j[3], n)
    p = np.column_stack((x, design.radius("rp", x), design.radius("r", x)))
    p[0, 1:] = (top_p, top_r)
    p[-1, 1:] = (end_p, end_r)
    t_end = -tangent(j[3], j[3] - eps)
    parts.append((p, (1.0, 0.0, 0.0), t_end))
    if design.cap > 0 and max(end_r, end_p) > 0:
        th = np.linspace(0.0, 0.5 * np.pi, n)
        p = np.column_stack((j[3] + design.cap * np.sin(th), end_p * np.cos(th), end_r * np.cos(th)))
        p[-1, 1:] = 0.0
        parts.append((p, (1.0, 0.0, 0.0), (0.0, -end_p, -end_r)))
    elif max(end_r, end_p) > 0:
        p = np.column_stack((np.full(3, j[3]), np.linspace(end_p, 0.0, 3), np.linspace(end_r, 0.0, 3)))
        parts.append((p, (0.0, -end_p, -end_r), (0.0, -end_p, -end_r)))
    return parts


_SQ = math.sqrt(0.5)
_CIRCLE = ((1, 0, 1), (1, 1, _SQ), (0, 1, 1), (-1, 1, _SQ), (-1, 0, 1), (-1, -1, _SQ), (0, -1, 1), (1, -1, _SQ),
           (1, 0, 1))                                     # the unit circle as a quadratic NURBS (y, z, weight)


def build_solid(design, n=150, scale=1000.0):
    """The CAD solid (mm): one NURBS face per part, the profile (x, r', r) interpolated by a cubic B-spline
    (GeomAPI_Interpolate, n points evenly spaced along it, its end tangents) times the rational unit circle,
    so every section is an exact ellipse; the faces sewn into a shell and made a solid."""
    occ = XL._occ
    gp_Pnt, gp_Vec = occ("gp", "gp_Pnt", "gp_Vec")
    TColgp_HArray1OfPnt, TColgp_Array2OfPnt = occ("TColgp", "TColgp_HArray1OfPnt", "TColgp_Array2OfPnt")
    TColStd_Array1OfReal, TColStd_Array2OfReal, TColStd_Array1OfInteger = occ(
        "TColStd", "TColStd_Array1OfReal", "TColStd_Array2OfReal", "TColStd_Array1OfInteger")
    GeomAPI_Interpolate = occ("GeomAPI", "GeomAPI_Interpolate")
    Geom_BSplineSurface = occ("Geom", "Geom_BSplineSurface")
    MakeFace, Sewing, MakeSolid = occ("BRepBuilderAPI", "BRepBuilderAPI_MakeFace", "BRepBuilderAPI_Sewing",
                                      "BRepBuilderAPI_MakeSolid")
    TopExp_Explorer = occ("TopExp", "TopExp_Explorer")
    TopAbs_SHELL = occ("TopAbs", "TopAbs_SHELL")
    ShapeFix_Shell, ShapeFix_Solid = occ("ShapeFix", "ShapeFix_Shell", "ShapeFix_Solid")

    def array1(vals, cls):
        a = cls(1, len(vals))
        for i, v in enumerate(vals, start=1):
            a.SetValue(i, v)
        return a

    vknots = array1([0.0, 0.25, 0.5, 0.75, 1.0], TColStd_Array1OfReal)
    vmults = array1([3, 2, 2, 2, 3], TColStd_Array1OfInteger)
    sew = Sewing(1e-4 * scale)
    for pts, t0, t1 in _profile_parts(design, int(n)):
        arr = TColgp_HArray1OfPnt(1, len(pts))
        for i, p in enumerate(pts * scale, start=1):
            arr.SetValue(i, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        it = GeomAPI_Interpolate(arr, False, 1e-9)
        t0 = np.asarray(t0, dtype=float) / np.linalg.norm(t0)
        t1 = np.asarray(t1, dtype=float) / np.linalg.norm(t1)
        it.Load(gp_Vec(*t0), gp_Vec(*t1), False)          # unit tangents: the parameter is the chord length
        it.Perform()
        if not it.IsDone():
            raise RuntimeError("the B-spline through a profile part failed")
        c = it.Curve()
        nu = c.NbPoles()
        poles = TColgp_Array2OfPnt(1, nu, 1, len(_CIRCLE))
        w = TColStd_Array2OfReal(1, nu, 1, len(_CIRCLE))
        for i in range(1, nu + 1):
            q = c.Pole(i)
            for jj, (cy, cz, cw) in enumerate(_CIRCLE, start=1):
                poles.SetValue(i, jj, gp_Pnt(q.X(), q.Y() * cy, q.Z() * cz))
                w.SetValue(i, jj, c.Weight(i) * cw)
        uk = array1([c.Knot(k) for k in range(1, c.NbKnots() + 1)], TColStd_Array1OfReal)
        um = array1([c.Multiplicity(k) for k in range(1, c.NbKnots() + 1)], TColStd_Array1OfInteger)
        surf = Geom_BSplineSurface(poles, w, uk, vknots, um, vmults, c.Degree(), 2, False, False)
        sew.Add(MakeFace(surf, 1e-7).Face())
    sew.Perform()
    ex = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    shells = []
    while ex.More():
        shells.append(XL._topods("Shell")(ex.Current()))
        ex.Next()
    if len(shells) != 1:
        raise RuntimeError(f"sewing left {len(shells)} shells, expected 1")
    fs = ShapeFix_Shell(shells[0])
    fs.Perform()
    solid = MakeSolid(fs.Shell()).Solid()
    fx = ShapeFix_Solid(solid)
    fx.Perform()
    return fx.Solid()


# --------------------------------------------------------------------------
# Checks, files, plots
# --------------------------------------------------------------------------

def design_problems(design, n=401):
    """What makes a design unusable, as messages (none: usable)."""
    out = []
    if design.nose <= 0 or design.tail <= 0:
        out.append("the nose and the tail need positive lengths")
    if design.middle < 0 or design.cap < 0:
        out.append("the middle body and the cap cannot be negative")
    for view in (("r",) if design.axisymmetric else VIEWS):
        top, end = design.radii(view)
        name = "half-height" if view == "r" else "half-width"
        if top <= 0:
            out.append(f"the largest {name} must be positive")
        if end < 0 or end >= top:
            out.append(f"the tail end {name} must lie between 0 and the largest")
        x = stations_x(design, n)
        v = design.radius(view, x)
        inner = (x > 0) & (x < design.joins[3] - 1e-12)
        if np.any(v[inner] <= 0):
            out.append(f"the {name} reaches 0 inside the body")
        if np.any(v > 1.2 * top):
            out.append(f"the {name} rises more than 20 % above its middle-body value")
    if design.cap > 0 and max(design.radii("r")[1], design.radii("rp")[1]) <= 0:
        out.append("a cap needs a tail end above 0")
    return out


def design_params(design, n=41):
    """The design as a parameter file: n stations per part."""
    x = stations_x(design, n)
    r, rp = design.radius("r", x), design.radius("rp", x)
    r[0] = rp[0] = 0.0
    if design.cap > 0:
        r[-1] = rp[-1] = 0.0
    return HullParams(design.nose, design.middle, design.tail, design.cap, x, r, rp, design.axisymmetric)


def plot_case(path, case, design, params=None):
    """The half-profiles r (XZ view) and r' (XY view) against x, the design (blue) with its control points
    and, when given, the parameter file's stations (grey)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    views = ("r",) if design.axisymmetric else VIEWS
    fig = Figure(figsize=(13, 3.3 * len(views) + 0.6))
    FigureCanvasAgg(fig)
    axs = np.atleast_1d(fig.subplots(len(views), 1))
    xs = stations_x(design, 400)
    for ax, view in zip(axs, views):
        for jx in design.joins[1:-1]:
            ax.axvline(jx, color="#c9c8c3", lw=0.8, ls=":")
        if params is not None:
            ax.plot(params.x, params.values(view), ".", color="#9a9994", ms=3, label="parameter file")
        ax.plot(xs, design.radius(view, xs), "-", color="#2a78d6", lw=1.8, label="design")
        for kind in KINDS:
            cx, cy = design.ctrl(view, kind)
            ax.plot(cx, cy, "o--", color="#2a78d6", lw=0.7, ms=4, mfc="white")
        c_n, c_t = design.curve(view, "nose"), design.curve(view, "tail")
        ax.set_ylabel(("half-height r (m), XZ view" if view == "r" else "half-width r' (m), XY view")
                      if not design.axisymmetric else "radius r (m)")
        ax.set_title(f"nose: order {c_n.order}, N1 {c_n.exponent:.4f}    tail: order {c_t.order}, N2 "
                     f"{c_t.exponent:.4f}", fontsize=9, loc="left")
        ax.grid(True, color="#e4e3df")
    axs[-1].set_xlabel("x from the nose (m)")
    axs[0].legend(fontsize=8, loc="lower center")
    rep = fit_report(params, design) if params is not None else {}
    worst = ", ".join(f"{k} {v * 1000:.3f} mm" for k, v in rep.items())
    fig.suptitle(f"{case}: L {design.length:.4f} m (nose {design.nose:.4f}, middle {design.middle:.4f}, tail "
                 f"{design.tail:.4f}, cap {design.cap:.4f})" + (f"; largest difference from the file: {worst}"
                                                                 if worst else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def write_cad(out_dir, case, design):
    """<case>.step; returns (path, valid, volume m^3, area m^2, centre (m))."""
    solid = build_solid(design)
    path = XL.write_step(solid, os.path.join(out_dir, f"{case}.step"))
    vol, area, c = HE.solid_properties(solid)
    return path, HE.is_valid(solid), vol, area, c


def write_case(out_dir, case, design, params=None, space=None, cad=True, plot=True):
    """Write a case (see the module docstring). Returns {name: path} and the hydrostatics."""
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    notes = [f"design of case {case} (hull_modify.py), {datetime.date.today().isoformat()}"]
    files["params"] = write_params(os.path.join(out_dir, f"{case}_params.dat"), design_params(design), notes)
    with open(os.path.join(out_dir, f"{case}_design.json"), "w") as fh:
        json.dump(design_record(design), fh, indent=1)
    files["design"] = os.path.join(out_dir, f"{case}_design.json")
    if space is not None:
        with open(os.path.join(out_dir, f"{case}_design_space.json"), "w") as fh:
            json.dump(space_record(space, design, params), fh, indent=1)
        files["space"] = os.path.join(out_dir, f"{case}_design_space.json")
    st = stations(design, 401)
    exact = {"pmb_start": design.joins[1], "pmb_end": design.joins[2]}
    cadline = None
    if cad and XL.occ_available():
        path, valid, vol, area, c = write_cad(out_dir, case, design)
        files["step"] = path
        exact.update({"volume": vol, "wetted_surface": area, "lcb": float(c[0])})
        cadline = f"V, S and LCB from the solid ({'valid' if valid else 'NOT valid'})"
    hs = HE.hydrostatics(st, exact=exact)
    title = f"hull case {case}"
    files["stations"] = HE.write_stations(os.path.join(out_dir, f"{case}_stations.dat"), st, hs, title)
    files["hydrostatics"] = HE.write_hydrostatics(os.path.join(out_dir, f"{case}_hydrostatics.dat"), hs, title,
                                                  [cadline or "V, S and LCB from the stations (S approximate "
                                                   "for ellipses)"])
    files["sections"] = write_sections(os.path.join(out_dir, f"{case}_sections_m.dat"), design)
    if plot:
        files["plot"] = plot_case(os.path.join(out_dir, f"{case}_check.png"), case, design, params)
    return files, hs


def main():
    ap = argparse.ArgumentParser(description="Fit a hull's curves to a parameter file and write the case.")
    ap.add_argument("params", nargs="?", help="parameter file (a station table)")
    ap.add_argument("--suboff", action="store_true", help=f"SUBOFF ({SUBOFF_PARAMS}, written from its equations "
                                                         "if it is not there)")
    ap.add_argument("--case", default=None, help="case name (default: the file's name, suboff_fit for SUBOFF)")
    ap.add_argument("--order", type=int, default=DEFAULT_ORDER, help=f"order of every curve (default "
                                                                      f"{DEFAULT_ORDER})")
    ap.add_argument("--out", default=OUT_DIR, help=f"output folder (default {OUT_DIR})")
    ap.add_argument("--no-cad", action="store_true", help="no STEP")
    args = ap.parse_args()
    if args.suboff:
        if not os.path.exists(SUBOFF_PARAMS):
            write_params(SUBOFF_PARAMS, suboff_params())
        path, case = SUBOFF_PARAMS, args.case or "suboff_fit"
    elif args.params:
        path, case = args.params, args.case or os.path.splitext(os.path.basename(args.params))[0]
    else:
        ap.error("give a parameter file or --suboff")
    params = read_params(path)
    names = [f"{k}_{v}" for v in (("r",) if params.axisymmetric else VIEWS) for k in KINDS]
    design = fit_design(params, {n: args.order for n in names})
    files, hs = write_case(args.out, case, design, params, HullSpace(design), not args.no_cad)
    rep = fit_report(params, design)
    print(f"{case}: L {design.length:.4f} m, V {hs['volume']:.6f} m^3, S {hs['wetted_surface']:.5f} m^2, "
          f"LCB {hs['lcb'] / design.length:.4f} L, Cp {hs['cp']:.4f}")
    for k, c in design.curves.items():
        print(f"  {k}: order {c.order}, {EXP_KEY[c.kind]} {c.exponent:.4f}, largest difference from the file "
              f"{rep[k] * 1000:.3f} mm")
    for f in files.values():
        print("  " + f)


if __name__ == "__main__":
    main()
