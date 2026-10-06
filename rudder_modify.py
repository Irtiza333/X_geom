"""
rudder_modify.py

Change an existing rudder below a height H and keep it as it is above.

The rudder is the one Rudder_geom_extraction.py measured, in its working frame
(mm): origin at the root LE, x chordwise LE -> TE, y up the span (the height),
z thickness. Below H, a design variable between the user's bounds h_min and
h_max (h_max at most z_top, the top horizontal section of the extracted stack,
z_full - d):

Planform  The TE line stays. The x of the LE follows a spanwise curve (below).
Sections  Horizontal, n_sec of them from the root to H (both included). Each is
          a Bezier half-section of degree n (bezier_section.py): P0 = (0, 0)
          the LE, P1 .. P(n-1) with both coordinates free, Pn = (1, 0) the
          sharp TE, all in fractions of the local sharp chord; mirrored, its TE
          rounded by a circle of constant radius in mm, and scaled so the
          rounded TE lies on the TE line. P1x = 0 keeps the LE round.
Curves    The x of the LE and every coordinate of the inner section control
          points (P1x, P1z, P2x, P2z, ...) follow their own spanwise Bezier
          curve, the value against the height. Each curve is stored top first:
          C0 at H pinned to the original's value there, C1 on the original's
          tangent (join "G1") or free (join "G0"), the inner points free in
          height and value, the last one at the root (its value free). Each
          curve has its own number of control points.
Degree    The original sections are quartics (the PSO section fits of the
          thickness-format files: P1 = (0, T1), P2 = (X1, T1), P3 = (X2, T2)).
          At a higher degree they are the same curves raised to that degree
          (exact), so the original shape is the same whatever the degree; adding
          a section control point raises the degree by one and adds two curves
          (change_degree carries a design over).
Original  The original shape is known at every height up to z_top from Bezier
          fits to its distributions (fit_original): the LE of corner_points.dat
          and the coordinates of the per-section fits of fit_section_table (the
          thickness_params_*_info.dat tables, thickness_params_y0_y200_info.dat
          for the whole stack). Wherever H is, the pinned values and slopes come
          from these fits, and baseline_design rebuilds the original below any H.

Above H the extracted sections stay exactly as extracted: their loops are
copied line for line from the extraction's XCAD file (write_modified_xcad).
write_case writes that XCAD file with the modified sections below H, the
modified sections' control points as a table (and in the thickness format when
the design is still a quartic with P1x = 0 and P1z = P2z), the design as JSON
and a check plot.

Every spanwise Bezier is stored top first, control point 0 at H and the last at
the root, with its heights in order (never rising towards the root), so each
height has exactly one point on the curve. Heights are kept as fractions
s = y / H, so a design scales with H.

Reused from the X_geom sub-folders
----------------------------------
Bezier code/para_control_bez.py
    eval_rational_bezier (a rational cubic), generalised to any order in the
    Bezier class; and the d-fraction trick that keeps control points in order
    whatever the design variables are (p2x = p4x - (p4x - R1) * d1), used here
    for the control heights (heights_from_fractions).
Bezier code/Chord_opt_bez.py
    fitting a Bezier distribution to the original one with the PSO: fit_law's
    free_heights option (Main_PSO over the inner control heights, the values
    by linear least squares for each candidate).
Alternate truncation/Geometry.py
    the LE curve from the original LE at H down to the root, x free there; the
    sections between that curve and the TE line; the laws evaluated at y / H.
Alternate truncation/Thickness_curves.py
    one Bezier law per section parameter, its value at the top pinned.
"""

from __future__ import annotations

import copy
import datetime
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

import bezier_section as BS
from replica_funcs import bernstein, build_half_airfoil, mirror_selig

BASE_DEGREE = 4                  # the section table's fits are quartics; no lower degree is exact
DEFAULT_ORDER = 3
_TOL_Y = 1.0e-9                  # mm, slack on heights read back from files


def curve_names(degree):
    """The spanwise curves of a design with sections of this degree: "LE",
    then P1x, P1z, ..., P(n-1)x, P(n-1)z."""
    return ["LE"] + BS.coord_names(degree)


def _degree_of(names):
    idx = [int(m.group(1)) for m in (re.fullmatch(r"P(\d+)[xz]", n) for n in names) if m]
    return max(idx) + 1 if idx else BASE_DEGREE


def describe(name):
    """A line saying what curve `name` is."""
    if name == "LE":
        return "x of the leading edge (mm) against the height"
    m = re.fullmatch(r"P(\d+)([xz])", name)
    if not m:
        return name
    what = "x (chordwise)" if m.group(2) == "x" else "z (thickness)"
    return f"{what} of section control point P{m.group(1)}, fraction of the sharp chord, against the height"


# --------------------------------------------------------------------------
# Bezier curves of any order
# --------------------------------------------------------------------------

class Bezier:
    """Rational Bezier curve of any order n,

        P(t) = sum_k w_k P_k B_k,n(t) / sum_k w_k B_k,n(t),     0 <= t <= 1,

    with control points ctrl (n + 1, dim) and positive weights w (all 1 by
    default, the ordinary polynomial Bezier). eval_rational_bezier of
    Bezier code/para_control_bez.py is the cubic case."""

    def __init__(self, ctrl, weights=None):
        ctrl = np.array(ctrl, dtype=float)
        if ctrl.ndim != 2 or len(ctrl) < 2:
            raise ValueError("a Bezier curve needs at least 2 control points, as rows")
        w = np.ones(len(ctrl)) if weights is None else np.array(weights, dtype=float).ravel()
        if w.shape != (len(ctrl),) or not np.all(w > 0.0):
            raise ValueError("the weights must be positive, one per control point")
        self.ctrl, self.w = ctrl, w

    @property
    def order(self):
        return len(self.ctrl) - 1

    def _hom(self):
        """Homogeneous control points (w P, w)."""
        return np.column_stack((self.ctrl * self.w[:, None], self.w))

    @staticmethod
    def _from_hom(hom):
        return Bezier(hom[:, :-1] / hom[:, -1:], hom[:, -1])

    def __call__(self, t):
        """Points at t: (m, dim)."""
        h = bernstein(np.atleast_1d(np.asarray(t, dtype=float)), self.order) @ self._hom()
        return h[:, :-1] / h[:, -1:]

    def derivative(self, t):
        """dP/dt at t: (m, dim)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        hom = self._hom()
        h = bernstein(t, self.order) @ hom
        dh = self.order * bernstein(t, self.order - 1) @ np.diff(hom, axis=0)
        w, dw = h[:, -1:], dh[:, -1:]
        return (dh[:, :-1] - h[:, :-1] / w * dw) / w

    def split(self, t0):
        """The pieces [0, t0] and [t0, 1] of this curve, each a Bezier of the
        same order on its own [0, 1] (de Casteljau on the homogeneous points)."""
        q = self._hom()
        left, right = [q[0]], [q[-1]]
        while len(q) > 1:
            q = (1.0 - t0) * q[:-1] + t0 * q[1:]
            left.append(q[0])
            right.append(q[-1])
        return self._from_hom(np.array(left)), self._from_hom(np.array(right[::-1]))

    def elevate(self, times=1):
        """The same curve with its order raised by `times`."""
        q = self._hom()
        for _ in range(int(times)):
            n = len(q) - 1
            a = (np.arange(1, n + 1) / (n + 1))[:, None]
            q = np.vstack((q[0], a * q[:-1] + (1.0 - a) * q[1:], q[-1]))
        return self._from_hom(q)

    def solve(self, value, axis=0, tol=1e-14, max_iter=100):
        """The parameters t at which coordinate `axis` takes each of `value`.

        That coordinate must be monotone along the curve. It is whenever its
        control values are (the weights being positive), which is how every
        spanwise curve here is built: heights never rise from P0 to Pn. Each
        value must lie between the coordinate's end values. Newton's method,
        started from a dense sample and kept inside its bracket."""
        v = np.atleast_1d(np.asarray(value, dtype=float))
        c0, c1 = self.ctrl[0, axis], self.ctrl[-1, axis]
        if c0 == c1:
            raise ValueError("the coordinate has the same value at both ends of the curve")
        sgn = 1.0 if c1 > c0 else -1.0
        lo_v, hi_v = min(c0, c1), max(c0, c1)
        slack = 1e-9 * max(1.0, abs(c0), abs(c1))
        if np.any(v < lo_v - slack) or np.any(v > hi_v + slack):
            raise ValueError(f"values outside the curve's range [{lo_v:g}, {hi_v:g}]")
        if v.size == 0:
            return v.copy()
        target = sgn * np.clip(v, lo_v, hi_v)
        ts = np.linspace(0.0, 1.0, 8 * self.order + 25)
        cs = np.maximum.accumulate(sgn * self(ts)[:, axis])
        j = np.clip(np.searchsorted(cs, target), 1, len(ts) - 1)
        lo, hi = ts[j - 1], ts[j]
        f_lo, f_hi = cs[j - 1], cs[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(f_hi > f_lo, lo + (target - f_lo) / (f_hi - f_lo) * (hi - lo), lo)
        for _ in range(max_iter):
            f = sgn * self(t)[:, axis] - target
            df = sgn * self.derivative(t)[:, axis]
            lo = np.where(f < 0.0, t, lo)
            hi = np.where(f > 0.0, t, hi)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_new = t - f / df
            t_new = np.where((t_new >= lo) & (t_new <= hi), t_new, 0.5 * (lo + hi))
            t_new = np.where(f == 0.0, t, t_new)
            step = np.max(np.abs(t_new - t))
            t = t_new
            if step <= tol:
                break
        return t

    def at(self, value, axis=0):
        """Points of the curve where coordinate `axis` equals each of `value`."""
        return self(self.solve(value, axis))


def heights_from_fractions(d):
    """Control heights, top first, as fractions of the curve's height:
    s_0 = 1, s_k = s_(k-1) (1 - d_k) for the n - 1 inner points, s_n = 0.
    Whatever the d in [0, 1], the heights stay in order (the d1 trick of
    para_control_bez.py, p2x = p4x - (p4x - R1) * d1)."""
    s = [1.0]
    for dk in np.atleast_1d(np.asarray(d, dtype=float)):
        s.append(s[-1] * (1.0 - dk))
    s.append(0.0)
    return np.array(s)


def fractions_from_heights(s):
    """The inverse of heights_from_fractions."""
    s = np.asarray(s, dtype=float)
    return np.array([1.0 - s[k] / s[k - 1] if s[k - 1] > 0.0 else 0.0 for k in range(1, len(s) - 1)])


def uniform_heights(order):
    """Evenly spaced control heights, top first (s = 1 .. 0). With them and
    unit weights a law is a polynomial in y of that order."""
    return 1.0 - np.arange(order + 1) / order


# --------------------------------------------------------------------------
# Fitting laws to data
# --------------------------------------------------------------------------

def law_basis(heights, y, weights=None):
    """The matrix A with v(y_i) = A[i] @ values for a law with these control
    heights (top first) and weights: the curve's (rational) Bernstein basis
    at the parameters where it passes the heights y."""
    heights = np.asarray(heights, dtype=float)
    curve = Bezier(np.column_stack((heights, np.zeros_like(heights))), weights)
    b = bernstein(curve.solve(y), curve.order) * curve.w
    return b / b.sum(axis=1, keepdims=True)


def fit_values(heights, y, v, weights=None, pinned=None):
    """Least-squares control values of a law with given control heights and
    weights through the data (y, v). pinned = {index: value} holds some of
    them. Returns (values, residuals at the data)."""
    a = law_basis(heights, y, weights)
    v = np.asarray(v, dtype=float)
    pinned = dict(pinned or {})
    vals = np.zeros(a.shape[1])
    fixed = sorted(pinned)
    free = [k for k in range(a.shape[1]) if k not in pinned]
    for k in fixed:
        vals[k] = pinned[k]
    if free:
        rhs = v - a[:, fixed] @ vals[fixed]
        vals[free] = np.linalg.lstsq(a[:, free], rhs, rcond=None)[0]
    return vals, a @ vals - v


def fit_law(y, v, order, top=None, bottom=0.0, weights=None, free_heights=False, pso_kw=None):
    """Fit a Bezier law v(y) of the given order to the data (y, v), control
    points top first from `top` (default: the highest datum) to `bottom`.

    With evenly spaced control heights (the default) the law is the
    least-squares polynomial of that order, found exactly by linear least
    squares. free_heights also searches the n - 1 inner heights, as
    Chord_opt_bez.py fits a Bezier to an original distribution with the PSO:
    Main_PSO over their d fractions (heights_from_fractions), each candidate's
    values by linear least squares, then a bounded least-squares polish. The
    even spacing is kept when nothing beats it.

    Returns (Bezier in the (y, v) plane, residuals at the data)."""
    y, v = np.asarray(y, dtype=float), np.asarray(v, dtype=float)
    top = float(np.max(y)) if top is None else float(top)
    order = int(order)
    if order < 1:
        raise ValueError("the order of a law must be at least 1")
    if not top > bottom:
        raise ValueError("top must lie above bottom")

    def solve(d):
        h = bottom + heights_from_fractions(d) * (top - bottom)
        vals, res = fit_values(h, y, v, weights)
        return h, vals, res

    d = fractions_from_heights(uniform_heights(order))
    if free_heights and order > 1:
        from scipy.optimize import least_squares
        from Main_PSO import PSO_main
        kw = dict(swarm_size=30, num_iter=80, stall_iter=20, constraint_handling=3, seed=42,
                  verbose=False)
        kw.update(pso_kw or {})
        box = np.array([[0.01, 0.99]] * (order - 1))      # no two heights quite together
        sse = lambda X: np.array([np.sum(solve(dk)[2]**2) for dk in X])
        best = PSO_main(sse, order - 1, box, **kw).best_pos
        if sse(best[None])[0] < sse(d[None])[0]:
            d = best
        pol = least_squares(lambda dk: solve(dk)[2], d, bounds=(0.01, 0.99),
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if np.sum(pol.fun**2) < sse(d[None])[0]:
            d = pol.x
    h, vals, res = solve(d)
    return Bezier(np.column_stack((h, vals)), weights), res


# --------------------------------------------------------------------------
# The original shape
# --------------------------------------------------------------------------

def load_section_table(path):
    """The columns of a section-fit table (thickness_params_H*_info.dat: a
    '#' header whose last line starting with 'section' and naming y_mm and X1
    lists the columns) as a dict of arrays."""
    names = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                tok = line[1:].split()
                if tok and tok[0] == "section" and "y_mm" in tok and "X1" in tok:
                    names = tok
    if names is None:
        raise ValueError(f"{path}: no column header ('# section  y_mm ... X1 ...')")
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(f"{path}: {len(names)} column names but {data.shape[1]} columns")
    return {n: data[:, k] for k, n in enumerate(names)}


def quartic_polygons(X1, X2, T1, T2):
    """The control polygons (m, 5, 2) of the thickness-format parameters:
    P0 (0, 0), P1 (0, T1), P2 (X1, T1), P3 (X2, T2), P4 (1, 0)."""
    X1, X2, T1, T2 = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (X1, X2, T1, T2))
    zero = np.zeros_like(T1)
    inner = np.stack([np.stack((zero, T1), axis=1), np.stack((X1, T1), axis=1), np.stack((X2, T2), axis=1)], axis=1)
    return BS.polygon(inner)


@dataclass
class OriginalShape:
    """The original rudder's shape from the root to z_top, as fitted Bezier
    curves in the (y, value) plane, top first: x of the LE (mm) and the
    coordinates of the section control points at `degree` (fractions of the
    sharp chord). The TE line is kept as the extraction measured it."""
    z_top: float                 # the fits cover [0, z_top]; H may go up to z_top
    degree: int
    curves: dict                 # name -> Bezier over [0, z_top]
    data: dict                   # name -> (y, value) the curve was fitted to
    residuals: dict              # name -> residuals of the fit at the data
    y_rows: np.ndarray           # heights of the horizontal corner-point rows, root first
    x_te_rows: np.ndarray        # x of the TE at those heights (the TE line, kept)
    x_le_rows: np.ndarray        # x of the original LE at those heights
    sources: dict = field(default_factory=dict)
    base: dict = field(default_factory=dict, repr=False)      # the quartic rows and fit options (at_degree)

    @property
    def names(self):
        return curve_names(self.degree)

    def _heights(self, y):
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if np.any(y < -_TOL_Y) or np.any(y > self.z_top + _TOL_Y):
            raise ValueError(f"heights outside [0, z_top = {self.z_top:g}] mm, the fitted original")
        return np.clip(y, 0.0, self.z_top)

    def value(self, name, y):
        """The fitted original value of `name` at the heights y."""
        return self.curves[name].at(self._heights(y))[:, 1]

    def slope(self, name, y):
        """d value / dy of the fitted original at the heights y."""
        c = self.curves[name]
        dp = c.derivative(c.solve(self._heights(y)))
        return dp[:, 1] / dp[:, 0]

    def x_te(self, y):
        """x of the TE line at the heights y (linear between the rows)."""
        return np.interp(y, self.y_rows, self.x_te_rows)

    def at_degree(self, degree):
        """The same original with its sections raised (or reduced, down to the
        table's quartics) to another degree: its curves refitted to the
        section fits' control points at that degree."""
        degree = int(degree)
        if degree == self.degree:
            return self
        b = self.base
        return _original(self.z_top, degree, b["le"], b["y_tab"], b["ctrl4"], self.y_rows, self.x_te_rows,
                         self.x_le_rows, b["orders"], b["weights"], b["free_heights"], self.sources)


def _original(z_top, degree, le, y_tab, ctrl4, y_rows, x_te_rows, x_le_rows, orders, weights, free_heights,
              sources):
    degree = int(degree)
    if degree < BASE_DEGREE:
        raise ValueError(f"the sections need degree {BASE_DEGREE} or more (the section fits are quartics)")
    coords = BS.values_from_polygon(BS.change_degree(ctrl4, degree))
    data = {"LE": le, **{c: (y_tab, coords[c]) for c in BS.coord_names(degree)}}
    curves, residuals = {}, {}
    for c in curve_names(degree):
        curves[c], residuals[c] = fit_law(*data[c], orders.get(c, DEFAULT_ORDER), top=z_top, bottom=0.0,
                                          weights=weights.get(c), free_heights=free_heights)
    base = {"le": le, "y_tab": y_tab, "ctrl4": ctrl4, "orders": orders, "weights": weights,
            "free_heights": free_heights}
    return OriginalShape(z_top, degree, curves, data, residuals, y_rows, x_te_rows, x_le_rows, sources, base)


def fit_original(corner_points, section_table, z_top=None, orders=None, weights=None, free_heights=False,
                 degree=BASE_DEGREE):
    """Fit the original shape from the root to z_top.

    corner_points   the extraction's corner_points.dat: the LE (x1, y1) of the
                    horizontal rows up to z_top (plus the LE interpolated at
                    z_top when no row sits there) is fitted; the TE rows are
                    kept as the TE line
    section_table   the per-section fits (fit_section_table, e.g.
                    thickness_params_y0_y200_info.dat); the control points of
                    their rows up to z_top, raised to `degree`, are fitted
    z_top           the top of the fits, the highest H a design may use
                    (default: the table's top row, at most the top horizontal
                    row of corner_points)
    orders          {curve: order} of each fit (default 3)
    free_heights    fit_law's option: search the inner control heights too
    degree          of the section Bezier (4, the table's quartics, or more)
    """
    rows = np.loadtxt(corner_points, ndmin=2)
    rows = rows[np.abs(rows[:, 1] - rows[:, 4]) < _TOL_Y]          # the horizontal rows
    rows = rows[np.argsort(rows[:, 1])]
    y_rows = rows[:, 1]
    table = load_section_table(section_table)
    if z_top is None:                               # the table's top row (its heights have 4 decimals: snap to the row)
        z_top = min(float(table["y_mm"].max()), float(y_rows[-1]))
        near = y_rows[np.abs(y_rows - z_top) < 1e-3]
        z_top = float(near.max()) if len(near) else z_top
    z_top = float(z_top)
    if not 0.0 < z_top <= y_rows[-1] + _TOL_Y:
        raise ValueError(f"z_top = {z_top:g} must lie in (0, {y_rows[-1]:g}], the top horizontal station")
    below = y_rows < z_top - _TOL_Y
    le = (np.append(y_rows[below], z_top), np.append(rows[below, 0], np.interp(z_top, y_rows, rows[:, 0])))
    keep = table["y_mm"] <= z_top + 1e-6
    y_tab = table["y_mm"][keep]
    if len(y_tab) < 2 or y_tab.min() > 1e-6 or y_tab.max() < z_top - 2.0 * np.ptp(y_tab) / (len(y_tab) - 1):
        print(f"  warning: the section table covers y = {y_tab.min():g} .. {y_tab.max():g} mm, "
              f"not 0 .. {z_top:g}; its laws are extrapolated there")
    ctrl4 = quartic_polygons(table["X1"][keep], table["X2"][keep], table["T1"][keep], table["T2"][keep])
    return _original(z_top, degree, le, y_tab, ctrl4, y_rows, rows[:, 3], rows[:, 0], dict(orders or {}),
                     dict(weights or {}), free_heights,
                     {"corner_points": corner_points, "section_table": section_table})


# --------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------

@dataclass
class CurveDesign:
    """One spanwise curve of a design, top first."""
    s: np.ndarray                # control heights / H: s[0] = 1 (at H) ... s[-1] = 0 (root)
    v: np.ndarray                # control values: LE x in mm, section coordinates in fractions of the sharp chord
    w: np.ndarray | None = None  # weights (None: all 1, a polynomial Bezier)
    join: str = "G1"             # "G1": C1 on the original's tangent at H; "G0": C1 free

    @property
    def order(self):
        return len(self.s) - 1


@dataclass
class Design:
    """The modified shape below H: one CurveDesign for each of
    curve_names(degree). pin() sets the control points that follow from the
    original shape at H."""
    H: float
    curves: dict
    degree: int = None

    def __post_init__(self):
        if self.degree is None:
            self.degree = _degree_of(self.curves)

    @property
    def names(self):
        return curve_names(self.degree)

    def copy(self):
        return copy.deepcopy(self)

    def curve(self, name):
        """The curve as a Bezier in the (y, value) plane, heights in mm."""
        c = self.curves[name]
        return Bezier(np.column_stack((np.asarray(c.s, dtype=float) * self.H, c.v)), c.w)


def _pins(orig, name, s, H, join):
    """{index: value} of the control values fixed by the original at H: C0,
    and with join 'G1' C1, on the original's tangent at its own height."""
    v0 = float(orig.value(name, H)[0])
    if join == "G0":
        return {0: v0}
    if join == "G1":
        return {0: v0, 1: v0 + float(orig.slope(name, H)[0]) * (s[1] - 1.0) * H}
    raise ValueError(f"{name}: join must be 'G1' or 'G0', not {join!r}")


def pin(design, orig):
    """A copy of the design with its pinned control points set from the
    original at H: s_0 = 1 and s_n = 0; the value at H; with join 'G1' the
    value of C1 on the original's tangent at H (its height as designed).
    Checks that H lies in (0, z_top], that the design and the original have
    the same section degree and that no curve's heights rise towards the root."""
    d = design.copy()
    d.H = float(d.H)
    if not 0.0 < d.H <= orig.z_top + _TOL_Y:
        raise ValueError(f"H = {d.H:g} mm must lie in (0, z_top = {orig.z_top:g}]")
    if sorted(d.curves) != sorted(orig.names) or d.degree != orig.degree:
        raise ValueError(f"the design has sections of degree {d.degree} and the original of degree "
                         f"{orig.degree} (curves {sorted(d.curves)}); use change_degree or orig.at_degree")
    d.H = min(d.H, orig.z_top)
    for name in orig.names:
        c = d.curves[name]
        c.s, c.v = np.array(c.s, dtype=float), np.array(c.v, dtype=float)
        if len(c.s) != len(c.v) or len(c.s) < 2:
            raise ValueError(f"{name}: s and v need one entry per control point, at least 2")
        c.s[0], c.s[-1] = 1.0, 0.0
        if np.any(np.diff(c.s) > 0.0):
            raise ValueError(f"{name}: control heights must not rise from H down to the root")
        for k, val in _pins(orig, name, c.s, d.H, c.join).items():
            c.v[k] = val
    return d


def baseline_design(orig, H, orders=None, join="G1", weights=None, n_samples=401):
    """The original shape below H as a design: for each curve, evenly spaced
    control heights and the values that fit the original's fitted curve on
    [0, H] best with the pins held. When the order is at least that of the
    original's fit (unit weights) the curve is the original exactly (the
    de Casteljau split of the fit, raised to the order)."""
    H = float(H)
    orders = dict(orders or {})
    weights = weights or {}
    yd = np.linspace(0.0, min(H, orig.z_top), n_samples)
    curves = {}
    for c in orig.names:
        jn = join.get(c, "G1") if isinstance(join, dict) else join
        s = uniform_heights(int(orders.get(c, DEFAULT_ORDER)))
        w = None if weights.get(c) is None else np.asarray(weights[c], dtype=float)
        v, _ = fit_values(s * H, yd, orig.value(c, yd), w, _pins(orig, c, s, H, jn))
        curves[c] = CurveDesign(s, v, w, jn)
    return pin(Design(H, curves, orig.degree), orig)


def change_degree(design, orig, degree, n_samples=401):
    """Carry a design over to sections of another degree (one more for each
    added section control point; not below the table's quartics).

    At n_samples heights the design's section polygons are raised (exact) or
    reduced (least squares on the control points) to the new degree, and each
    new coordinate's curve is fitted to them: evenly spaced heights and the
    largest order of the design's section curves; join 'G1' where the new
    coordinate leaves H along the original's tangent, 'G0' elsewhere. The LE
    curve is kept. A design whose section curves all have evenly spaced
    heights comes over exactly. Returns (design, original at the new degree,
    largest fit residual)."""
    degree = int(degree)
    d = pin(design, orig)
    o2 = orig.at_degree(degree)
    if degree == orig.degree:
        return d, o2, 0.0
    yd = np.linspace(0.0, d.H, int(n_samples))
    val = curve_values(d, yd)
    new = BS.values_from_polygon(BS.change_degree(BS.polygon_from_values(val, d.degree), degree))
    sec = [c for c in d.names if c != "LE"]
    slope = {}
    for c in sec:                                          # d value / dy of each curve at H
        dp = d.curve(c).derivative([0.0])[0]
        slope[c] = dp[1] / dp[0]
    sp = np.zeros((1, d.degree + 1, 2))                    # slopes move like the polygon, the ends fixed
    for c in sec:
        sp[0, int(c[1:-1]), "xz".index(c[-1])] = slope[c]
    new_slope = BS.values_from_polygon(BS.change_degree(sp, degree))
    order = max(d.curves[c].order for c in sec)
    curves = {"LE": copy.deepcopy(d.curves["LE"])}
    err = 0.0
    for c in BS.coord_names(degree):
        ref = float(o2.slope(c, d.H)[0])
        join = "G1" if abs(new_slope[c][0] - ref) <= 1e-9 * max(1.0, abs(ref)) else "G0"
        s = uniform_heights(order)
        v, res = fit_values(s * d.H, yd, new[c], None, _pins(o2, c, s, d.H, join))
        curves[c] = CurveDesign(s, v, None, join)
        err = max(err, float(np.abs(res).max()))
    return pin(Design(d.H, curves, degree), o2), o2, err


def default_bounds(z_top, h_min=None, h_max=None):
    """Bounds of the design vector (DesignSpace) by group: H from h_min to
    h_max (the user's choice; default a quarter of h_max and z_top, h_max at
    most z_top); height fractions d in [0.05, 0.95]; LE control points up to
    4 h_max forward of the original LE and not behind it (the range of
    Alternate truncation/Geometry.py); section x coordinates in [0, 1] and z
    coordinates in [0, 0.2] (fractions of the sharp chord)."""
    h_max = float(z_top) if h_max is None else float(h_max)
    h_min = 0.25 * h_max if h_min is None else float(h_min)
    if not 0.0 < h_min <= h_max <= z_top + _TOL_Y:
        raise ValueError(f"H bounds [{h_min:g}, {h_max:g}] mm must satisfy 0 < h_min <= h_max <= z_top = {z_top:g}")
    return {"H": (h_min, h_max), "d": (0.05, 0.95), "LE": (-4.0 * h_max, 0.0), "x": (0.0, 1.0), "z": (0.0, 0.2)}


class DesignSpace:
    """A flat design vector for an optimiser, and the Design it stands for.

    The slots, in order: H, then for each curve of orig.names d_1 .. d_(n-1),
    the fractions that place its inner control heights (any d in [0, 1] keeps
    them in order), then its free control values: those of C2 .. Cn with join
    'G1' (C0 and C1 follow from the original at H), C1 .. Cn with 'G0'. Names:
    "H", "LE.d1", "LE.dx2", "P2z.d1", "P2z.v2", ... LE values are offsets in mm
    of the control point's x from the original LE at the control point's own
    height, negative forward; section coordinates are the fractions of the
    sharp chord themselves.

    h_min      the lowest H, h_max the highest (the user's choices; h_max at
               most orig.z_top)
    bounds     overrides entries of default_bounds(z_top, h_min, h_max) by group
               ("LE", "d" for every height fraction, "x" / "z" for every
               section x / z coordinate), by curve ("P2z") or by slot name
               ("LE.dx3": (-40, 0)); a slot beats its curve, a curve its group
    fixed      {slot name: value} of the slots held fixed: they stay out of
               the vector x (the design variables are the other slots)

    all_names / all_bounds cover every slot, names / bounds the free ones;
    len(space) is the number of design variables."""

    def __init__(self, orig, orders=None, join="G1", weights=None, bounds=None, h_min=None, h_max=None,
                 fixed=None):
        self.orig = orig
        self.curves = orig.names
        orders = dict(orders or {})
        self.orders = {c: int(orders.get(c, DEFAULT_ORDER)) for c in self.curves}
        self.join = {c: (join.get(c, "G1") if isinstance(join, dict) else join) for c in self.curves}
        weights = weights or {}
        self.weights = {c: (None if weights.get(c) is None else np.asarray(weights[c], dtype=float))
                        for c in self.curves}
        b = {**default_bounds(orig.z_top, h_min, h_max), **(bounds or {})}
        self._slots = [("", "H", 0)]
        for c in self.curves:
            n = self.orders[c]
            self._slots += [(c, "d", k) for k in range(1, n)]
            self._slots += [(c, "v", k) for k in range(2 if self.join[c] == "G1" else 1, n + 1)]
        self.all_names = [self.slot_name(*sl) for sl in self._slots]
        self.all_bounds = np.array([self._bound(b, n, sl) for n, sl in zip(self.all_names, self._slots)], dtype=float)
        self.fixed = {str(n): float(v) for n, v in (fixed or {}).items()}
        unknown = sorted(set(self.fixed) - set(self.all_names))
        if unknown:
            raise ValueError(f"fixed slots that this space does not have: {unknown}")
        self._free = np.array([i for i, n in enumerate(self.all_names) if n not in self.fixed], dtype=int)
        self.names = [self.all_names[i] for i in self._free]
        self.bounds = self.all_bounds[self._free].reshape(-1, 2)

    @staticmethod
    def _bound(b, name, slot):
        c, kind, _ = slot
        if name in b:
            return b[name]
        if kind == "H":
            return b["H"]
        if kind == "d":
            return b.get(f"{c}.d", b["d"])
        if c in b:
            return b[c]
        return b["LE"] if c == "LE" else b[c[-1]]

    @staticmethod
    def slot_name(c, kind, k):
        if kind == "H":
            return "H"
        return f"{c}.d{k}" if kind == "d" else f"{c}.{'dx' if c == 'LE' else 'v'}{k}"

    def __len__(self):
        return len(self.names)

    def full_vector(self, x):
        """Every slot's value: the design variables x and the fixed slots."""
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self.names),):
            raise ValueError(f"the design vector has {len(self.names)} entries")
        full = np.array([self.fixed.get(n, np.nan) for n in self.all_names])
        full[self._free] = x
        return full

    def to_design(self, x):
        """The (pinned) Design of the design vector x."""
        return self._design_of(self.full_vector(x))

    def design_of(self, values):
        """The (pinned) Design of {slot name: value} for every slot."""
        return self._design_of(np.array([float(values[n]) for n in self.all_names]))

    def _design_of(self, full):
        H = float(full[0])
        d = {c: [] for c in self.curves}
        vals = {c: np.zeros(self.orders[c] + 1) for c in self.curves}
        for (c, kind, k), xv in zip(self._slots[1:], full[1:]):
            if kind == "d":
                d[c].append(xv)
            else:
                vals[c][k] = xv
        curves = {}
        for c in self.curves:
            s = heights_from_fractions(d[c])
            v = vals[c]
            free = [k for (cc, kind, k) in self._slots if cc == c and kind == "v"]
            if c == "LE" and free:
                v[free] = v[free] + self.orig.value("LE", s[free] * H)
            curves[c] = CurveDesign(s, v, self.weights[c], self.join[c])
        return pin(Design(H, curves, self.orig.degree), self.orig)

    def values(self, design):
        """{slot name: value} of a design, every slot (fixed ones as the
        design has them)."""
        dd = pin(design, self.orig)
        out = {"H": dd.H}
        for (c, kind, k), name in zip(self._slots[1:], self.all_names[1:]):
            cd = dd.curves[c]
            if kind == "d":
                out[name] = float(fractions_from_heights(cd.s)[k - 1])
            elif c == "LE":
                out[name] = float(cd.v[k] - self.orig.value("LE", cd.s[k] * dd.H)[0])
            else:
                out[name] = float(cd.v[k])
        return out

    def to_vector(self, design):
        vals = self.values(design)
        return np.array([vals[n] for n in self.names])

    def to_dict(self, design):
        """The space and a design in it, as plain data (JSON): the section
        degree, orders, joins, weights, every slot with its value, bounds and
        whether it is free, and the design vector (names, values, bounds).
        from_dict reads it back."""
        vals = self.values(design)
        return {"z_top_mm": self.orig.z_top, "h_min_mm": float(self.all_bounds[0, 0]),
                "h_max_mm": float(self.all_bounds[0, 1]),
                "degree": int(self.orig.degree),
                "orders": {c: int(self.orders[c]) for c in self.curves}, "join": dict(self.join),
                "weights": {c: (None if self.weights[c] is None else self.weights[c].tolist()) for c in self.curves},
                "slots": [{"name": n, "value": vals[n], "lo": float(lo), "hi": float(hi), "free": n not in self.fixed}
                          for n, (lo, hi) in zip(self.all_names, self.all_bounds)],
                "vector": {"names": self.names, "values": [vals[n] for n in self.names],
                           "bounds": self.bounds.tolist()}}

    @classmethod
    def from_dict(cls, orig, rec):
        """(space, design) from to_dict's output; the space's original is
        orig at the record's section degree."""
        orig = orig.at_degree(int(rec.get("degree", orig.degree)))
        slots = rec["slots"]
        bounds = {q["name"]: (q["lo"], q["hi"]) for q in slots}
        fixed = {q["name"]: q["value"] for q in slots if not q["free"]}
        weights = {c: w for c, w in (rec.get("weights") or {}).items() if w is not None}
        space = cls(orig, rec["orders"], rec["join"], weights, bounds, h_min=bounds["H"][0], h_max=bounds["H"][1],
                    fixed=fixed)
        return space, space.design_of({q["name"]: q["value"] for q in slots})


def load_design_space(path):
    """(space, design, record) of a set-up the design tool saved
    (<case>_design_space.json). The original shape is fitted again from the
    files and fit settings the record names (paths relative to the folder the
    tool ran in), so the space is the one the tool set up:

        space, design, rec = load_design_space("outputs/modified/gui_design_space.json")
        x0, bounds = space.to_vector(design), space.bounds      # the free variables
        d = space.to_design(x)                                  # a candidate x
        problems = design_problems(d, space.orig, rec["settings"]["n_sections"])
        write_case(out_dir, name, d, space.orig, rec["files"]["xcad_original"], ...)
    """
    with open(path) as fh:
        rec = json.load(fh)
    f, s = rec["files"], rec["settings"]
    orig = fit_original(f["corner_points"], f["section_table"], s.get("z_top"), s.get("fit_orders"),
                        free_heights=s.get("fit_free_heights", False),
                        degree=rec["space"].get("degree", BASE_DEGREE))
    space, design = DesignSpace.from_dict(orig, rec["space"])
    return space, design, rec


def refit_curve(design, orig, name, order, n_samples=401):
    """The design with curve `name` refitted at another order: evenly spaced
    control heights and the least-squares values (pins held, weights reset)
    that follow the curve's present shape on [0, H] most closely."""
    d = pin(design, orig)
    cd = d.curves[name]
    yd = np.linspace(0.0, d.H, n_samples)
    vd = d.curve(name).at(yd)[:, 1]
    s = uniform_heights(int(order))
    v, _ = fit_values(s * d.H, yd, vd, None, _pins(orig, name, s, d.H, cd.join))
    out = d.copy()
    out.curves[name] = CurveDesign(s, v, None, cd.join)
    return pin(out, orig)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

@dataclass
class ModSection:
    """A modified section, horizontal at height y."""
    index: int                   # 1 = root
    y: float
    x_le: float
    x_te: float
    chord: float                 # LE to the rounded TE
    cs: float                    # sharp chord: ctrl and r are fractions of it
    r: float                     # TE radius / cs
    ctrl: np.ndarray             # (n + 1, 2) control points P0 .. Pn, fractions of cs
    selig: np.ndarray            # (num_pts, 2) airfoil frame, mm, Selig order

    @property
    def degree(self):
        return len(self.ctrl) - 1

    @property
    def r_mm(self):
        return self.r * self.cs

    def coords(self):
        """{P1x: .., P1z: .., ...} of the inner control points."""
        return {k: float(v[0]) for k, v in BS.values_from_polygon(self.ctrl[None]).items()}

    def params(self):
        """(X1, X2, T1, T2, r) of the thickness format, or None when the
        section is not such a quartic (degree 4, P1x = 0 and P1z = P2z)."""
        c = self.ctrl
        if self.degree != 4 or abs(c[1, 0]) > 1e-12 or abs(c[1, 1] - c[2, 1]) > 1e-12:
            return None
        return float(c[2, 0]), float(c[3, 0]), float(c[1, 1]), float(c[3, 1]), float(self.r)

    def loop(self):
        """The section as an XCAD loop, working frame, mm."""
        return xcad_loop(self.selig, self.x_le, self.y)


def curve_values(design, ys):
    """{curve: its values at the heights ys} for a pinned design."""
    return {c: design.curve(c).at(ys)[:, 1] for c in design.names}


def _problem_masks(chord, ctrl, te_radius_mm):
    """Masks of the sections that cannot be built: (LE at or behind the TE,
    polygon folding back, thickness not positive, TE radius too big)."""
    zmax, _, zmin = BS.half_thickness(ctrl)
    no_chord = ~(chord > 0.0)
    fold = ~BS.x_monotone(ctrl)
    thin = ~(zmin > 0.0)
    ok = ~no_chord & ~fold & ~thin
    r_frac = te_radius_mm / np.where(chord > 0.0, chord, np.inf)       # r / cs, cs a little above the chord
    return no_chord, fold, thin, ok & (r_frac >= 0.9 * zmax)


_PROBLEM_TEXT = ("the LE is at or behind the TE",
                 "the section control points fold the half-section back (x must rise from the LE to the TE)",
                 "the half-section touches or crosses the chord line",
                 "the TE radius does not fit under the half-thickness")


def build_sections(design, orig, ys, num_pts=200, te_radius_mm=0.75, first_index=1):
    """The sections of a pinned design at the heights ys (all at once, see
    modified_sections). Raises ValueError naming the first section that
    cannot be built and why."""
    d = pin(design, orig)
    ys = np.atleast_1d(np.asarray(ys, dtype=float))
    val = curve_values(d, ys)
    x_le, x_te = val["LE"], orig.x_te(ys)
    chord = x_te - x_le
    ctrl = BS.polygon_from_values(val, d.degree)
    for mask, text in zip(_problem_masks(chord, ctrl, te_radius_mm), _PROBLEM_TEXT):
        if np.any(mask):
            k = int(np.argmax(mask))
            raise ValueError(f"section {first_index + k} at y = {ys[k]:.4f} mm: {text} "
                             f"({', '.join(f'{n} {v[k]:.5f}' for n, v in val.items() if n != 'LE')}, "
                             f"LE x {x_le[k]:.4f}, TE x {x_te[k]:.4f})")
    selig, cs, r = BS.sections(ctrl, chord, te_radius_mm, num_pts)
    return [ModSection(first_index + k, float(ys[k]), float(x_le[k]), float(x_te[k]), float(chord[k]),
                       float(cs[k]), float(r[k]), ctrl[k], selig[k]) for k in range(len(ys))]


def modified_sections(design, orig, n_sec=50, num_pts=200, te_radius_mm=0.75):
    """The n_sec horizontal sections from the root to H (root first).

    At each height: the LE from the design's LE curve, the TE from the TE
    line, the section control points from their curves, and the sharp chord
    that puts the TE, rounded with te_radius_mm, on the TE line
    (bezier_section.sharp_chord). num_pts points per section in the
    extraction's layout."""
    d = pin(design, orig)
    return build_sections(d, orig, np.linspace(0.0, d.H, int(n_sec)), num_pts, te_radius_mm)


def section_at(design, orig, y, num_pts=200, te_radius_mm=0.75, index=0):
    """The section of a design at any height y in [0, H], built as
    modified_sections builds its own (for previews)."""
    return build_sections(design, orig, [float(y)], num_pts, te_radius_mm, index)[0]


def design_problems(design, orig, n_sec=50, te_radius_mm=0.75):
    """What would stop modified_sections, found cheaply at its n_sec heights:
    control heights out of order or H out of range, an LE at or behind the TE,
    section control points folding the half-section back, a thickness that is
    not positive, a TE radius that does not fit under the half-thickness. A
    list of messages, empty when the design builds; for an interactive tool
    and as a feasibility test for an optimiser."""
    try:
        d = pin(design, orig)
    except ValueError as exc:
        return [str(exc)]
    ys = np.linspace(0.0, d.H, int(n_sec))
    val = curve_values(d, ys)
    chord = orig.x_te(ys) - val["LE"]
    out = []
    for bad, what in zip(_problem_masks(chord, BS.polygon_from_values(val, d.degree), te_radius_mm), _PROBLEM_TEXT):
        if np.any(bad):
            y = ys[bad]
            out.append(f"{what} at {int(bad.sum())} of {len(ys)} sections (y = {y.min():.2f} .. {y.max():.2f} mm)")
    return out


def section_metrics(sec):
    """t/c, the position of the thickest point (x/c) and the LE radius (mm)
    of a section, c being its chord (LE to the rounded TE)."""
    p = BS.points(sec.ctrl, np.linspace(0.0, 1.0, 4001))[0]
    k = int(np.argmax(p[:, 1]))
    return {"t_c": 2.0 * p[k, 1] * sec.cs / sec.chord, "x_t_c": p[k, 0] * sec.cs / sec.chord,
            "rle_mm": float(BS.le_radius(sec.ctrl)[0]) * sec.cs}


# --------------------------------------------------------------------------
# The original sections (the extraction's stack file)
# --------------------------------------------------------------------------

@dataclass
class StackSection:
    y: float                     # height (of the LE, for the tilted section)
    chord: float
    x_le: float
    x_te: float
    tilt_deg: float              # 0 for a horizontal section
    xy: np.ndarray               # (num_pts, 2) airfoil frame, mm, Selig order


_SECTION_RE = re.compile(r"#\s*SECTION\s+\d+\s*/\s*\d+\s+y\s*=\s*(\S+)\s+chord\s*=\s*(\S+)"
                         r"\s+x_le\s*=\s*(\S+)\s+x_te\s*=\s*(\S+)(.*)")


def read_stack(path):
    """The sections of the extraction's stack file (sections_stack_*_raw_mm.dat),
    horizontal ones root first, then the tilted tip section if there is one."""
    heads = []
    with open(path) as fh:
        for line in fh:
            m = _SECTION_RE.match(line.strip())
            if m:
                heads.append(m)
    xy = np.loadtxt(path).reshape(len(heads), -1, 2)
    out = []
    for m, pts in zip(heads, xy):
        tilt = re.search(r"tilt\s*=\s*(\S+)", m.group(5))
        out.append(StackSection(float(m.group(1)), float(m.group(2)), float(m.group(3)),
                                float(m.group(4)), float(tilt.group(1)) if tilt else 0.0, pts))
    return sorted(out, key=lambda s: (s.tilt_deg != 0.0, s.y))


def original_section_at(stack, y):
    """The original section at height y: (x_le, xy), interpolated point by
    point between the horizontal stack sections either side (the stations of
    both follow their chords, so this is exact for a linear taper)."""
    flat = [s for s in stack if s.tilt_deg == 0.0]
    ys = np.array([s.y for s in flat])
    if not ys[0] - _TOL_Y <= y <= ys[-1] + _TOL_Y:
        raise ValueError(f"y = {y:g} lies outside the stack, {ys[0]:g} .. {ys[-1]:g} mm")
    j = int(np.clip(np.searchsorted(ys, y), 1, len(ys) - 1))
    a, b = flat[j - 1], flat[j]
    f = (y - a.y) / (b.y - a.y)
    return (1.0 - f) * a.x_le + f * b.x_le, (1.0 - f) * a.xy + f * b.xy


# --------------------------------------------------------------------------
# XCAD point files (the format of Rudder_geom_extraction.write_xcad; that
# module needs the OCC bindings, this one does not, so the few lines it takes
# are repeated here and must stay in step with it)
# --------------------------------------------------------------------------

XCAD_UNITS = {"m": 1.0e-3, "mm": 1.0}
XCAD_FRAME = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0]])


def _snap(v, places=8):
    return 0.0 if abs(v) < 0.5 * 10.0 ** (-places) else float(v)


def xcad_loop(selig, x_le, y):
    """A horizontal section at height y, Selig points in the airfoil frame
    (LE at index (n + 1) // 2 - 1), as the closed loop XCAD reads: from the
    LE along the +z side to the TE, back along the -z side, the LE repeated.
    Working frame, mm. The same as Rudder_geom_extraction.xcad_loop."""
    selig = np.asarray(selig, dtype=float)
    n = len(selig)
    i_le = (n + 1) // 2 - 1
    if abs(selig[i_le, 0]) > 1.0e-9:
        raise ValueError(f"section at y = {y:g}: leading edge not at index {i_le}")
    loop2 = np.vstack([selig[:i_le + 1][::-1], selig[i_le:][::-1][1:]])
    pts = np.empty((n, 3))
    pts[:, 0] = x_le + loop2[:, 0]
    pts[:, 1] = y
    pts[:, 2] = loop2[:, 1]
    return pts


def xcad_lines(loop, units="m"):
    """A loop (working frame, mm) as XCAD point lines: X Y Z with the height
    as Z, eight decimals, in `units`."""
    scale = XCAD_UNITS[units]
    return [f"{_snap(x):.8f} {_snap(y):.8f} {_snap(z):.8f}"
            for x, y, z in (np.asarray(loop, dtype=float) @ XCAD_FRAME.T) * scale]


def xcad_to_working(pts, units="m"):
    """XCAD-frame points in `units` -> working frame, mm."""
    return (np.asarray(pts, dtype=float) / XCAD_UNITS[units]) @ XCAD_FRAME


def read_xcad(path):
    """The loops of an XCAD point file, in file order: a list of (lines, pts),
    the loop's point lines as written (line ends dropped) and their values
    (n, 3), XCAD frame, the file's units."""
    loops = []
    with open(path, newline="") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if line.startswith("#"):
                loops.append([])
            elif line.strip():
                if not loops:
                    raise ValueError(f"{path}: points before the first # marker")
                loops[-1].append(line)
    return [(lines, np.array([[float(v) for v in ln.split()] for ln in lines])) for lines in loops]


def write_xcad_lines(path, loops):
    """Write loops (each a list of point lines) with #1, #2, ... before each,
    CRLF line ends, as write_xcad does."""
    with open(path, "w", newline="\r\n") as fh:
        for k, lines in enumerate(loops, start=1):
            fh.write(f"#{k}\n")
            fh.write("\n".join(lines) + "\n")


def kept_loops(loops, H, gap, units="m"):
    """Indices of the loops (read_xcad) kept above H: the horizontal ones lying
    wholly at or above H + gap, the others (the tilted tip section, the cap)
    lying wholly at or above H."""
    out = []
    for k, (_, pts) in enumerate(loops):
        z = pts[:, 2] / XCAD_UNITS[units]
        if z.min() >= H + (gap if np.ptp(z) <= 1e-6 * max(1.0, abs(z).max()) else 0.0) - _TOL_Y:
            out.append(k)
    return out


def write_modified_xcad(path, sections, original_xcad, H, gap_mm=None, units="m",
                        original_units="m"):
    """The modified rudder as one XCAD file: the modified sections from the
    root to H, then the loops of the original file above them (kept_loops:
    the extracted sections from H + gap up, the tilted tip section and the
    cap), copied line for line when the units match. The gap (default one
    modified step, H / (n_sec - 1)) leaves out the extracted sections just
    above H, which would sit a fraction of a millimetre from the replica at H
    and differ from it by the fit error; the tip section and the cap stay
    whatever H is. Returns what was written."""
    gap = H / (len(sections) - 1) if gap_mm is None else float(gap_mm)
    loops = read_xcad(original_xcad)
    kept = kept_loops(loops, H, gap, original_units)
    out = [xcad_lines(s.loop(), units) for s in sections]
    for k in kept:
        lines, pts = loops[k]
        out.append(lines if units == original_units
                   else xcad_lines(xcad_to_working(pts, original_units), units))
    write_xcad_lines(path, out)
    return {"path": path, "n_modified": len(sections), "gap_mm": gap, "n_original": len(loops),
            "kept": [k + 1 for k in kept],
            "first_kept_y": (float(loops[kept[0]][1][:, 2].min() / XCAD_UNITS[original_units])
                             if kept else None),
            "n_loops": len(out), "units": units}


# --------------------------------------------------------------------------
# Section fits of an extracted stack (the original's section table)
# --------------------------------------------------------------------------

def default_fit_settings():
    """The section-fit settings of control.py: (foil_settings, pso_params,
    bounds), quiet."""
    from types import SimpleNamespace
    foil = SimpleNamespace(slope_tol=0.001, n_bez_samples=200, n_target_stations=0, fillet_method=0)
    pso = SimpleNamespace(swarm_size=50, num_iter=100, C1=2.05, C2=2.05, PSO_variant=1, w_max=0.9,
                          w_min=0.5, vmax_frac=0.2, constraint_handling=3, penalty_factor=5.0,
                          craziness=0, stall_iter=15, seed=42, ConsOn=0, Obj=0, polish=True,
                          verbose=False)
    bounds = np.array([[0.00, 0.15], [0.05, 0.50], [0.30, 0.95], [0.00, 0.15]])
    return foil, pso, bounds


def _gap_to_fit(P, cs, r, rc, xr, yr):
    """Distance of the original's upper-surface points (xr, yr) to the fitted
    foil with its TE rounded (the circle of the fit: centre rc[0] - r, from
    the cut point rc[25]), all in the section's chord units."""
    te_pt, cut = rc[0], rc[25]
    center = te_pt - [r, 0.0]
    bx, by = build_half_airfoil(P, 20000)
    bez = np.column_stack((bx, by)) * cs
    phi = np.arctan2(cut[1], cut[0] - center[0])
    th = np.linspace(phi, 0.0, 4000)[1:]
    curve = np.vstack((bez[bez[:, 0] < cut[0]], cut,
                       np.column_stack((center[0] + r * np.cos(th), r * np.sin(th)))))
    a, ab = curve[:-1], np.diff(curve, axis=0)
    l2 = (ab**2).sum(axis=1)
    a, ab, l2 = a[l2 > 0], ab[l2 > 0], l2[l2 > 0]
    g = np.empty(len(xr))
    for k, q0 in enumerate(np.column_stack((xr, yr))):
        t = np.clip(((q0 - a) * ab).sum(axis=1) / l2, 0.0, 1.0)
        g[k] = np.hypot(*(a + t[:, None] * ab - q0).T).min()
    return g


SECTION_TABLE_COLUMNS = ("section", "y_mm", "chord_mm", "cs_mm", "X1", "X2", "T1", "T2", "r", "r_mm",
                         "R_orig_mm", "fit_um", "gap_rms_um", "gap_max_um", "rle_fit_mm", "rle_orig_mm")


def fit_section_table(stack_path, z_max, out_txt=None, out_dat=None, settings=None, verbose=True):
    """Fit the Bezier half-section (match_rudder_section) to every horizontal
    section of an extracted stack from the root up to z_max, as was done for
    thickness_params_H80.txt. Writes the thickness-format file (out_txt) and
    the table with heights, chords and fit errors (out_dat) when given, and
    returns the rows (dicts keyed by SECTION_TABLE_COLUMNS).

    settings = (foil_settings, pso_params, bounds) as in control.py; default
    default_fit_settings()."""
    from match_rudder_section import match_rudder_section
    from retruncate import find_te_fillet
    from temp_funcs import reference_half, untruncated_profile
    foil, pso, bounds = settings or default_fit_settings()
    secs = [s for s in read_stack(stack_path) if s.tilt_deg == 0.0 and s.y <= z_max + 1e-6]
    rows = []
    for k, s in enumerate(secs, start=1):
        xr, yr, _ = reference_half(s.xy / s.chord)              # chord units, as the Selig files
        xs, ys, cs = untruncated_profile(xr, yr, foil.slope_tol)
        P, val, r, rc, _, _ = match_rudder_section(xr, yr, xs, ys, cs, pso, foil, bounds)
        te = find_te_fillet(mirror_selig(xr, yr))
        g = _gap_to_fit(P, cs, r, rc, xr, yr)
        c = s.chord
        rows.append({"section": k, "y_mm": s.y, "chord_mm": c, "cs_mm": cs * c, "X1": P[1], "X2": P[2],
                     "T1": P[0], "T2": P[3], "r": r / cs, "r_mm": r * c, "R_orig_mm": te["R"] * c,
                     "fit_um": val * cs * c * 1e3, "gap_rms_um": np.sqrt(np.mean(g**2)) * c * 1e3,
                     "gap_max_um": g.max() * c * 1e3, "rle_fit_mm": 4 * P[0]**2 / (3 * P[1]) * cs * c,
                     "rle_orig_mm": (xr[1]**2 + yr[1]**2) / (2 * xr[1]) * c})
        if verbose:
            q = rows[-1]
            print(f"  section {k:3d}/{len(secs)}  y = {s.y:8.3f} mm  fit {q['fit_um']:6.2f} um  "
                  f"X1 {q['X1']:.5f}  X2 {q['X2']:.5f}  T1 {q['T1']:.5f}  T2 {q['T2']:.5f}")
    at_bound = [name for j, name in enumerate(("T1", "X1", "X2", "T2"))
                if all(np.isclose(q[name], bounds[j]).any() for q in rows)]
    notes = [f"source       : {os.path.basename(stack_path)}, {len(rows)} horizontal sections from "
             f"y = {rows[0]['y_mm']:g} to {rows[-1]['y_mm']:g} mm; section 1 is the root",
             "fit          : Main_PSO then a bounded least-squares polish, fillet_method "
             f"{foil.fillet_method}; bounds Y1 {list(bounds[0])}, X2 {list(bounds[1])}, "
             f"X3 {list(bounds[2])}, Y3 {list(bounds[3])}"]
    if at_bound:
        notes.append(f"note         : {', '.join(at_bound)} on a bound in every section")
    if out_txt:
        write_params_txt(out_txt, [(q["section"], q["X1"], q["X2"], q["T1"], q["T2"], q["r"]) for q in rows])
    if out_dat:
        write_section_table(out_dat, rows, notes)
    return rows


def write_section_table(path, rows, notes=()):
    """The fit table in the layout of thickness_params_H80_info.dat (readable
    by load_section_table)."""
    head = ["# Fitted thickness parameters, one row per section, root first",
            f"# written      : {datetime.date.today():%d %b %Y} by rudder_modify.fit_section_table",
            *[f"# {n}" for n in notes],
            "# parameters   : quartic Bezier half-section P0 (0,0), P1 (0,T1), P2 (X1,T1), P3 (X2,T2), "
            "P4 (1,0), mirrored",
            "#                for the lower surface, with its sharp TE rounded by the circle of radius r "
            "tangent to it.",
            "#                X1, X2, T1, T2, r are fractions of the sharp chord cs_mm (P4 is the sharp TE).",
            "# columns",
            "#   section      section number, 1 = root (y = 0)",
            "#   y_mm         span station; chord_mm the section chord (LE to rounded TE)",
            "#   cs_mm        sharp chord, the length the five parameters are fractions of",
            "#   r_mm         fitted TE radius; R_orig_mm the original section's TE radius",
            "#   fit_um       RMS vertical gap to the sharpened section at its points (the objective)",
            "#   gap_rms_um   RMS and largest distance of the original's upper-surface points",
            "#   gap_max_um   to the fitted foil (TE rounded)",
            "#   rle_fit_mm   LE radius of the fit, 4 T1^2 / (3 X1) * cs; rle_orig_mm the original's",
            "#",
            "# section      y_mm   chord_mm     cs_mm         X1         X2         T1         T2          r"
            "    r_mm  R_orig_mm  fit_um  gap_rms_um  gap_max_um  rle_fit_mm  rle_orig_mm"]
    body = [f"{q['section']:9d}  {q['y_mm']:8.4f}  {q['chord_mm']:9.4f}  {q['cs_mm']:8.4f}  {q['X1']:9.7f}"
            f"  {q['X2']:9.7f}  {q['T1']:9.7f}  {q['T2']:9.7f}  {q['r']:9.7f}  {q['r_mm']:6.4f}"
            f"  {q['R_orig_mm']:9.4f}  {q['fit_um']:6.2f}  {q['gap_rms_um']:10.2f}  {q['gap_max_um']:10.2f}"
            f"  {q['rle_fit_mm']:10.3f}  {q['rle_orig_mm']:11.3f}" for q in rows]
    with open(path, "w", newline="\r\n") as fh:
        fh.write("\n".join(head + body) + "\n")


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def write_params_txt(path, rows):
    """The thickness format of 'Example Thickness Format.txt': a header, a
    blank line, then 'Section # | X1 | X2 | T1 | T2 | r' rows. rows are
    ModSections (each a quartic with P1x = 0 and P1z = P2z, see
    ModSection.params) or (section, X1, X2, T1, T2, r) tuples."""
    lines = ["Section # | X1 | X2 | T1 | T2 | r", ""]
    for q in rows:
        if isinstance(q, ModSection):
            p = q.params()
            if p is None:
                raise ValueError(f"section {q.index} is not a quartic with P1x = 0 and P1z = P2z; "
                                 "the thickness format cannot hold it")
            k, X1, X2, T1, T2, r = (q.index, *p)
        else:
            k, X1, X2, T1, T2, r = q
        lines.append(f"{k:9d}  {X1:.7f}|{X2:.7f}|{T1:.7f}|{T2:.7f}|{r:.7f}")
    with open(path, "w", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")


def write_sections_dat(path, sections, notes=()):
    """The modified sections as a table, one row per section, root first:
    heights, LE, TE, chords, measures and the section control points."""
    names = BS.coord_names(sections[0].degree)
    head = ["# Modified sections, one row per section, root first (rudder_modify.py)",
            f"# written      : {datetime.date.today():%d %b %Y}",
            *[f"# {n}" for n in notes],
            "# frame        : working frame of the extraction, mm: origin at the root LE, x chordwise, "
            "y height",
            f"# section      : Bezier half-section of degree {sections[0].degree}, P0 (0,0) the LE, "
            f"P{sections[0].degree} (1,0) the sharp TE,",
            "#                the inner control points P1 .. as below, fractions of the sharp chord cs_mm;",
            "#                mirrored, TE rounded with radius r (fraction of cs_mm) tangent to it",
            "# columns",
            "#   section      section number, 1 = root",
            "#   y_mm         height of the section",
            "#   x_le_mm      LE (on the modified LE curve); x_te_mm the rounded TE (on the TE line)",
            "#   chord_mm     x_te_mm - x_le_mm; cs_mm the sharp chord",
            "#   r_mm         TE radius; t_c the thickness over chord_mm, at x_t_c (fraction of chord_mm)",
            "#   rle_mm       LE radius (from P0, P1, P2)",
            "#   P1x ..       the inner control points",
            "#",
            "# section      y_mm     x_le_mm     x_te_mm   chord_mm      cs_mm          r     r_mm       t_c"
            "     x_t_c    rle_mm" + "".join(f"{n:>11s}" for n in names)]
    body = []
    for s in sections:
        m = section_metrics(s)
        cv = s.coords()
        body.append(f"{s.index:9d}  {s.y:8.4f}  {s.x_le:10.5f}  {s.x_te:10.5f}  {s.chord:9.4f}  {s.cs:9.4f}"
                    f"  {s.r:9.7f}  {s.r_mm:7.4f}  {m['t_c']:8.5f}  {m['x_t_c']:8.5f}  {m['rle_mm']:8.4f}"
                    + "".join(f"  {cv[n]:9.7f}" for n in names))
    with open(path, "w", newline="\r\n") as fh:
        fh.write("\n".join(head + body) + "\n")


def design_record(design, orig):
    """The pinned design as plain lists, for JSON."""
    d = pin(design, orig)
    out = {"H_mm": d.H, "degree": int(d.degree), "curves": {}}
    for c in d.names:
        cd = d.curves[c]
        out["curves"][c] = {
            "order": cd.order, "join": cd.join, "what": describe(c),
            "s": cd.s.tolist(), "y_mm": (cd.s * d.H).tolist(), "value": cd.v.tolist(),
            "weights": (np.ones(cd.order + 1) if cd.w is None else np.asarray(cd.w)).tolist()}
    return out


_OLD_LAWS = {"T1": ("P1z", "P2z"), "X1": ("P2x",), "X2": ("P3x",), "T2": ("P3z",)}


def design_from_record(rec):
    """The Design of a design_record (or of the 'design' entry of a
    <case>_design.json). A record of the earlier layout (laws T1, X1, X2, T2)
    is read as the quartic it stood for: T1 drives P1z and P2z, P1x is 0."""
    curves = {}
    for c, r in rec["curves"].items():
        cd = CurveDesign(np.array(r["s"]), np.array(r["value"]),
                         None if np.allclose(r["weights"], 1.0) else np.array(r["weights"]), r["join"])
        for name in _OLD_LAWS.get(c, (c,)):
            curves[name] = copy.deepcopy(cd)
    if "T1" in rec["curves"]:
        le = curves["LE"]
        curves["P1x"] = CurveDesign(le.s.copy(), np.zeros_like(le.v), None, le.join)
    return Design(float(rec["H_mm"]), curves, rec.get("degree"))


def original_record(orig):
    """The original-shape fits as plain lists, for JSON."""
    out = {"z_top_mm": orig.z_top, "degree": int(orig.degree), "sources": orig.sources, "curves": {}}
    for c in orig.names:
        b = orig.curves[c]
        res = orig.residuals[c]
        out["curves"][c] = {"order": b.order, "y_mm": b.ctrl[:, 0].tolist(), "value": b.ctrl[:, 1].tolist(),
                            "weights": b.w.tolist(), "n_data": int(len(res)),
                            "rms_residual": float(np.sqrt(np.mean(res**2))),
                            "max_residual": float(np.abs(res).max())}
    return out


def write_original_fit(out_dir, orig):
    """original_fit_Zopt<z>.json (the fitted control points) and .dat (the
    data and the fits at the data heights)."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"y0_y{orig.z_top:g}"
    pj = os.path.join(out_dir, f"original_fit_{tag}.json")
    with open(pj, "w") as fh:
        json.dump(original_record(orig), fh, indent=1)
    pd_ = os.path.join(out_dir, f"original_fit_{tag}.dat")
    coords = BS.coord_names(orig.degree)
    head = [f"# Bezier fits to the original shape from the root to z_top = {orig.z_top:g} mm "
            "(rudder_modify.fit_original)",
            f"# written      : {datetime.date.today():%d %b %Y}",
            f"# sources      : {orig.sources.get('corner_points')} (LE), {orig.sources.get('section_table')} "
            f"(section control points, degree {orig.degree})",
            "# orders       : " + ", ".join(f"{c} {orig.curves[c].order}" for c in orig.names),
            "# rows         : one per section of the table; x_le_* at the same heights (interpolated "
            "between corner rows)",
            "# columns      : y_mm, then for each curve the datum and the fit (LE x in mm, section control "
            "points in fractions of the sharp chord)",
            "#",
            "#     y_mm       x_le_data        x_le_fit" + "".join(f"{c + '_data':>14s}{c + '_fit':>14s}"
                                                          for c in coords)]
    y = orig.data[coords[0]][0]
    le_data = np.interp(y, orig.y_rows, orig.x_le_rows)
    cols = [y, le_data, orig.value("LE", y)]
    for c in coords:
        cols += [orig.data[c][1], orig.value(c, y)]
    body = ["".join([f"{row[0]:10.4f}", f"{row[1]:16.9f}", f"{row[2]:16.9f}", *[f"{v:14.9f}" for v in row[3:]]])
            for row in np.column_stack(cols)]
    with open(pd_, "w", newline="\r\n") as fh:
        fh.write("\n".join(head + body) + "\n")
    return pj, pd_


def write_case(out_dir, case, design, orig, original_xcad, n_sec=50, num_pts=200, te_radius_mm=0.75,
               gap_mm=None, units="m", original_units="m", stack=None, plot=True, space=None,
               h_min=None, h_max=None):
    """Build the modified sections of a design and write, to out_dir:

    <case>_sections_xcad_<units>.dat  the whole rudder for XCAD: the modified
                                      sections, then the original loops from
                                      H + gap up (write_modified_xcad)
    <case>_sections.dat               the modified sections: heights, LE/TE,
                                      chords, t/c, LE radius and control points
    <case>_thickness_params.txt       the same in the thickness format
                                      (Section # | X1 | X2 | T1 | T2 | r), only
                                      when every section is a quartic with
                                      P1x = 0 and P1z = P2z
    <case>_design.json                the design (pinned control points) and
                                      the settings; with `space`, the design
                                      vector too
    <case>_check.png                  planform, curves and sections (plot_case)

    h_min and h_max, when given, are the bounds on H: a design outside them is
    refused.

    Returns a dict: the pinned design, the sections, the XCAD summary and
    the paths."""
    if h_min is not None and design.H < float(h_min) - _TOL_Y:
        raise ValueError(f"{case}: H = {design.H:g} mm is below h_min = {float(h_min):g} mm")
    if h_max is not None and design.H > float(h_max) + _TOL_Y:
        raise ValueError(f"{case}: H = {design.H:g} mm is above h_max = {float(h_max):g} mm")
    os.makedirs(out_dir, exist_ok=True)
    d = pin(design, orig)
    secs = modified_sections(d, orig, n_sec, num_pts, te_radius_mm)
    p = {k: os.path.join(out_dir, f"{case}_{name}") for k, name in
         (("xcad", f"sections_xcad_{units}.dat"), ("table", "sections.dat"), ("params", "thickness_params.txt"),
          ("json", "design.json"), ("png", "check.png"))}
    xinfo = write_modified_xcad(p["xcad"], secs, original_xcad, d.H, gap_mm, units, original_units)
    quartic = all(s.params() is not None for s in secs)
    if quartic:
        write_params_txt(p["params"], secs)
    else:
        p.pop("params")
    notes = [f"case         : {case}; H = {d.H:g} mm (original fitted up to {orig.z_top:g} mm), {len(secs)} sections, "
             f"TE radius {te_radius_mm:g} mm",
             f"XCAD file    : {os.path.basename(p['xcad'])}, these sections then the original loops "
             f"{xinfo['kept'][0] if xinfo['kept'] else '-'} .. {xinfo['kept'][-1] if xinfo['kept'] else '-'}"
             f" of {os.path.basename(original_xcad)}"
             + (f" (from y = {xinfo['first_kept_y']:.4f} mm)" if xinfo["kept"] else "")]
    write_sections_dat(p["table"], secs, notes)
    rec = {"case": case, "written": f"{datetime.date.today():%d %b %Y}",
           "settings": {"z_top_mm": orig.z_top, "h_min_mm": None if h_min is None else float(h_min),
                        "h_max_mm": None if h_max is None else float(h_max),
                        "degree": int(d.degree), "n_sections": len(secs), "num_pts": num_pts,
                        "te_radius_mm": te_radius_mm, "gap_mm": xinfo["gap_mm"], "xcad_units": units},
           "thickness_format": ("written" if quartic else
                                "not written: the sections are not quartics with P1x = 0 and P1z = P2z"),
           "design": design_record(d, orig),
           "original": {**original_record(orig),
                        "at_H": {c: {"value": float(orig.value(c, d.H)[0]),
                                     "slope_per_mm": float(orig.slope(c, d.H)[0])} for c in orig.names}},
           "xcad": {k: v for k, v in xinfo.items() if k != "path"},
           "files": {k: os.path.basename(v) for k, v in p.items()}}
    if space is not None:
        rec["vector"] = {"names": space.names, "values": space.to_vector(d).tolist(),
                         "bounds": space.bounds.tolist()}
    with open(p["json"], "w") as fh:
        json.dump(rec, fh, indent=1)
    if plot:
        plot_case(p["png"], case, d, orig, secs, stack)
    else:
        p.pop("png")
    return {"case": case, "design": d, "sections": secs, "xcad": xinfo, "paths": p}


def distance_to_original(sec, stack):
    """Distance (mm) of each upper-surface point of the original section at
    the height of `sec` (original_section_at) to the upper surface of `sec`,
    both where they sit in x (each from its own LE). For the baseline design
    this is the error of the replica; for a changed design, the change."""
    x_le_o, xy_o = original_section_at(stack, sec.y)
    n_up = (len(xy_o) + 1) // 2
    q = xy_o[:n_up] + [x_le_o, 0.0]
    up = sec.selig[:(len(sec.selig) + 1) // 2] + [sec.x_le, 0.0]
    a, ab = up[:-1], np.diff(up, axis=0)
    l2 = (ab**2).sum(axis=1)
    a, ab, l2 = a[l2 > 0], ab[l2 > 0], l2[l2 > 0]
    t = np.clip(np.einsum("mkd,kd->mk", q[:, None, :] - a[None], ab) / l2, 0.0, 1.0)
    return np.sqrt((((a[None] + t[..., None] * ab[None]) - q[:, None, :])**2).sum(axis=2)).min(axis=1)


def case_report(res, orig, stack=None):
    """Lines describing a written case (for printing)."""
    d, secs, x = res["design"], res["sections"], res["xcad"]
    root, top = secs[0], secs[-1]
    mr, mt = section_metrics(root), section_metrics(top)
    cr, ct = root.coords(), top.coords()
    lines = [f"case {res['case']}: H = {d.H:g} mm, {len(secs)} sections of degree {d.degree}, "
             f"step {d.H / (len(secs) - 1):.4f} mm",
             f"  LE   root x = {root.x_le:9.4f} mm (original {float(orig.value('LE', 0.0)[0]):9.4f}),"
             f" at H x = {top.x_le:.4f} mm",
             f"  chord root {root.chord:9.4f} mm, at H {top.chord:9.4f} mm; t/c root {mr['t_c']:.5f},"
             f" at H {mt['t_c']:.5f}; TE radius {root.r_mm:.4f} mm",
             "  control points (root -> H): " + ", ".join(f"{c} {cr[c]:.5f} -> {ct[c]:.5f}"
                                                         for c in BS.coord_names(d.degree)),
             f"  XCAD {os.path.basename(x['path'])}: {x['n_loops']} loops = {x['n_modified']} modified + "
             f"{len(x['kept'])} original (loops {x['kept'][0]} .. {x['kept'][-1]} of {x['n_original']}, "
             f"from y = {x['first_kept_y']:.4f} mm; gap {x['gap_mm']:.4f} mm)" if x["kept"] else
             f"  XCAD {os.path.basename(x['path'])}: {x['n_loops']} loops, no original loop kept"]
    if "params" not in res["paths"]:
        lines.append("  thickness format not written: the sections are not quartics with P1x = 0 and P1z = P2z")
    if stack is not None:
        g = np.array([distance_to_original(s, stack) for s in secs])
        rms = np.sqrt(np.mean(np.concatenate(g)**2))
        k = int(np.argmax([gi.max() for gi in g]))
        lines.append(f"  original to new, distance of the original's upper-surface points: RMS {rms * 1e3:.1f} um,"
                     f" largest {g[k].max() * 1e3:.1f} um (section {k + 1}, y = {secs[k].y:.3f} mm)")
    return lines


# --------------------------------------------------------------------------
# Check plot
# --------------------------------------------------------------------------

def plot_case(path, case, design, orig, sections, stack=None, every=7):
    """Planform and every spanwise curve (the height on the x axis) and three
    sections of a case, in one PNG. Drawn on a bare Figure (no pyplot), so it
    also runs off a GUI thread."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    d = pin(design, orig)
    H = d.H
    top = min(orig.z_top, max(1.3 * H, H + 20.0))           # the plots show the original up to here
    yy = np.linspace(0.0, H, 400)
    yz = np.linspace(0.0, top, 400)
    coords = BS.coord_names(d.degree)
    ncol = 4
    nrow = -(-(len(coords) + 2) // ncol)
    fig = Figure(figsize=(4.4 * ncol, 3.6 * nrow + 0.6))
    FigureCanvasAgg(fig)
    axs = fig.subplots(nrow, ncol).ravel()

    ax = axs[0]
    m = orig.y_rows <= top
    ax.plot(orig.y_rows[m], orig.x_le_rows[m], "k--", lw=1, label="original LE")
    ax.plot(orig.y_rows[m], orig.x_te_rows[m], "k-", lw=1, label="TE line (kept)")
    le = d.curve("LE")
    pts = le.at(yy)
    for s in sections[::every] + [sections[-1]]:
        ax.plot([s.y, s.y], [s.x_le, s.x_te], color="0.75", lw=0.6)
    ax.plot(pts[:, 0], pts[:, 1], "b-", lw=2, label=f"new LE (order {le.order}, {d.curves['LE'].join})")
    ax.plot(le.ctrl[:, 0], le.ctrl[:, 1], "o--", color="tab:blue", ms=5, lw=0.8, mfc="w", label="LE control points")
    ax.axvline(H, color="r", ls=":", lw=1, label=f"H = {H:g} mm")
    ax.set_xlabel("y, height (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_title("planform (sections every %d shown)" % every, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=6, loc="upper right")

    for ax, c in zip(axs[1:], coords):
        law = d.curve(c)
        yd, vd = orig.data[c]
        yd, vd = yd[yd <= top], vd[yd <= top]
        ax.plot(yd, vd, "k.", ms=3, label="section fits (original)")
        ax.plot(yz, orig.value(c, yz), "k-", lw=1, label=f"original (order {orig.curves[c].order})")
        ax.plot(yy, law.at(yy)[:, 1], "b-", lw=2, label=f"design (order {law.order}, {d.curves[c].join})")
        ax.plot(law.ctrl[:, 0], law.ctrl[:, 1], "o--", color="tab:blue", ms=5, lw=0.8, mfc="w",
                label="control points")
        ax.plot(law.ctrl[0, 0], law.ctrl[0, 1], "rs", ms=6, label="pinned at H")
        ax.axvline(H, color="r", ls=":", lw=1)
        shown = np.concatenate((vd, law.ctrl[:, 1], orig.value(c, yz)))
        lo, hi = shown.min(), shown.max()
        pad = 0.08 * (hi - lo) if hi - lo > 1e-6 * max(abs(hi), 1e-3) else 1e-3 * max(abs(hi), 1e-3)
        ax.set_ylim(lo - pad, hi + pad)                 # a constant curve gets a readable axis too
        ax.set_title(f"{c}: {'x' if c.endswith('x') else 'z'} of P{c[1:-1]} (fraction of the sharp chord)",
                     fontsize=9)
        ax.set_xlabel("y (mm)")
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.legend(fontsize=6)

    ax = axs[len(coords) + 1]
    pick = [sections[0], sections[len(sections) // 2], sections[-1]]
    for s, col in zip(pick, ("tab:blue", "tab:orange", "tab:green")):
        ax.plot(s.x_le + s.selig[:, 0], s.selig[:, 1], color=col, lw=1.3, label=f"new, y = {s.y:.2f} mm")
        ax.plot(s.x_le + s.ctrl[:, 0] * s.cs, s.ctrl[:, 1] * s.cs, "o:", color=col, ms=3, lw=0.7)
        if stack is not None:
            x_le_o, xy_o = original_section_at(stack, s.y)
            ax.plot(x_le_o + xy_o[:, 0], xy_o[:, 1], color=col, ls="--", lw=0.9, label="original there")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("sections at the root, H/2 and H, with their control points", fontsize=9)
    ax.legend(fontsize=6)
    for ax in axs[len(coords) + 2:]:
        ax.set_visible(False)
    fig.suptitle(f"{case}: modified below H = {H:g} mm, {len(sections)} sections of degree "
                 f"{d.degree}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
