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
          tangent (join "G1") or free (join "G0": the curve meets the original
          with its value only, and the design tool lofts its CAD below and
          above H on its own, xcad_loft.loft_split), the inner points free in
          height and value, the last one at the root (its value free). Each
          curve has its own number of control points.
Degree    The original sections are quartics (the PSO section fits of the
          thickness-format files: P1 = (0, T1), P2 = (X1, T1), P3 = (X2, T2)).
          At a higher degree they are the same curves raised to that degree
          (exact), so the original shape is the same whatever the degree; adding
          a section control point raises the degree by one and adds two curves
          (change_degree carries a design over).
Laws      A curve may instead lie between a low law and a high law (bounding
          laws, Design.laws, with_laws): expressions, in the grammar of
          xgeom_relations, of the other curves at the same height ("P3z"), of
          the original's ("orig.P2z"), of the TE line ("TE") and of the sharp
          chords ("chord", "orig.chord"; law_names). The curve's control
          values are then the fraction of the way from its low law to its high
          one, 0 .. 1, and
              value(y) = low(y) + fraction(y) [high(y) - low(y)],
          so it lies between its laws at every height whatever the fractions
          are. "orig.P2z*orig.chord/chord" as the low law of P2z keeps that
          control point at least as high in mm as the original's wherever the
          LE goes. The low law is where the fraction is 0; it may be the
          larger of the two (the fraction then runs down). Such a curve meets
          the original at H with its value only (join "G0").
Inside    inside_original measures, on the built sections, how far the
          modified rudder goes inside the original (mm; 0 when the original is
          enclosed). Laws and bounds act on control points and make most
          designs pass; the measure is what tells.
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

Set-up    A set-up is a component file (xgeom_files.py, kind rudder_h; 26 Nov
          2026, in place of the JSON set-up): the original's files, the
          section degree, the full height, the settings, each curve's order
          with its join at H or its two laws, and every variable's value,
          bounds, free flag and relation (write_component, load_component).

Written   The Bezier sections and their spanwise curves are a model of the
          rudder: their fit of the original lies a little off the extracted
          sections (the section fits' error, about 0.1 mm for the wind tunnel
          rudder). So each modified section is written to XCAD as the
          extracted section at its height plus the change the design makes to
          the model, point by point:
              written = model(design) + (extracted - model(original's fit))
          (original_offsets, ModSection.written). The baseline design then
          writes the extracted rudder itself, every design meets the extracted
          sections at H without a step, and what a design changes is exactly
          what the model changes. The control-point table, the thickness
          format and the design JSON hold the model.

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
import functools
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

import bezier_section as BS
import xgeom_files as XF
import xgeom_relations as REL
from replica_funcs import bernstein, build_half_airfoil, mirror_selig

BASE_DEGREE = 4                  # the section table's fits are quartics; no lower degree is exact
DEFAULT_ORDER = 3
_TOL_Y = 1.0e-9                  # mm, slack on heights read back from files
TE_RADIUS = 0.75                 # mm, the TE radius the builders take when none is given (build_sections)
_LAW_TOL = 1.0e-9                # a band is closed when its laws are this close (times the larger of 1 and them)
_LAW_STEPS = 60                  # steps, at most, of the fixed point that finds the chord the laws use


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
    scale: float = 1.0           # heights stretched by this from the extracted rudder (scaled)
    te_radius: float = None      # TE radius of the original sections, mm (the table's R_orig_mm), when known

    @property
    def names(self):
        return curve_names(self.degree)

    def scaled(self, k):
        """The same original stretched in height by k: every height (z_top, the rows, the data, the curves'
        control heights) times k, the x of the LE and TE and the section coordinates kept. A design on it is
        the design on this one stretched by k when its H is k times as high (pin and the sections follow
        heights in proportion). scale records the stretch from the extracted rudder."""
        k = float(k)
        if not k > 0.0:
            raise ValueError(f"a height scale is positive, not {k:g}")
        if k == 1.0:
            return self
        curves = {}
        for c, b in self.curves.items():
            ctrl = np.array(b.ctrl, dtype=float)
            ctrl[:, 0] *= k
            curves[c] = Bezier(ctrl, b.w)
        base = dict(self.base)
        if "le" in base:
            base["le"] = (np.asarray(base["le"][0], dtype=float) * k, base["le"][1])
        if "y_tab" in base:
            base["y_tab"] = np.asarray(base["y_tab"], dtype=float) * k
        return OriginalShape(self.z_top * k, self.degree, curves,
                             {c: (np.asarray(y, dtype=float) * k, v) for c, (y, v) in self.data.items()},
                             dict(self.residuals), np.asarray(self.y_rows, dtype=float) * k, self.x_te_rows,
                             self.x_le_rows, dict(self.sources), base, self.scale * k, self.te_radius)

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
                         self.x_le_rows, b["orders"], b["weights"], b["free_heights"], self.sources, self.scale,
                         self.te_radius)


def _original(z_top, degree, le, y_tab, ctrl4, y_rows, x_te_rows, x_le_rows, orders, weights, free_heights,
              sources, scale=1.0, te_radius=None):
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
    return OriginalShape(z_top, degree, curves, data, residuals, y_rows, x_te_rows, x_le_rows, sources, base,
                         float(scale), te_radius)


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
    r_te = table["R_orig_mm"][keep] if "R_orig_mm" in table else np.empty(0)       # the original sections' TE radius
    r_te = r_te[np.isfinite(r_te) & (r_te > 0.0)]
    return _original(z_top, degree, le, y_tab, ctrl4, y_rows, rows[:, 3], rows[:, 0], dict(orders or {}),
                     dict(weights or {}), free_heights,
                     {"corner_points": corner_points, "section_table": section_table},
                     te_radius=float(np.median(r_te)) if len(r_te) else None)


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
    original shape at H. laws = {curve: (low law, high law)} puts curves
    between bounding laws (with_laws): the control values of such a curve are
    its fractions of the way from its low law to its high one."""
    H: float
    curves: dict
    degree: int = None
    laws: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.degree is None:
            self.degree = _degree_of(self.curves)
        self.laws = {str(c): (str(v[0]), str(v[1])) for c, v in dict(self.laws or {}).items()}

    @property
    def names(self):
        return curve_names(self.degree)

    def copy(self):
        return copy.deepcopy(self)

    def curve(self, name):
        """The curve as a Bezier in the (y, value) plane, heights in mm (a
        curve with bounding laws: its fraction between them; curve_values
        gives the values of every curve)."""
        c = self.curves[name]
        return Bezier(np.column_stack((np.asarray(c.s, dtype=float) * self.H, c.v)), c.w)


# --------------------------------------------------------------------------
# Bounding laws
# --------------------------------------------------------------------------

def law_names(names):
    """The names a bounding law of a design with these curves may use: the
    curves themselves (the design's value at the same height), the original's
    ("orig.P2z"), "TE" (x of the TE line, mm), "chord" (the section's sharp
    chord, mm: the length its coordinates are fractions of) and "orig.chord"
    (the original's sharp chord at that height)."""
    names = list(names)
    return names + [f"orig.{c}" for c in names] + ["TE", "chord", "orig.chord"]


def _law_key(laws):
    return tuple((c, v[0], v[1]) for c, v in sorted(laws.items()))


@functools.lru_cache(maxsize=256)
def _law_plan(key, names):
    """The laws ((curve, low, high), ...) of a design with the curves `names`,
    parsed: ({curve: (low, high) as expressions}, the curves that have laws in
    the order of their dependence, whether a law uses the chord, the curves
    defined by their laws (low and high the same expression: nothing left to
    vary)). ValueError for a law that cannot be read, a curve that uses
    itself, the LE using the chord, curves that depend on each other in a loop
    (the chord counts as the LE: it follows from it)."""
    allowed = law_names(names)
    exprs = {}
    for c, low, high in key:
        if c not in names:
            raise ValueError(f"'{c}' is not a curve of this design ({', '.join(names)})")
        pair = []
        for side, text in (("low", low), ("high", high)):
            try:
                e = REL.parse_law(text, allowed, own=c)
            except ValueError as exc:
                raise ValueError(f"the {side} law of {c}, {exc}") from None
            if c == "LE" and "chord" in e.names:
                raise ValueError(f"the {side} law of LE, '{text}': the chord follows from the LE, so the laws "
                                 f"of the LE cannot use it")
            pair.append(e)
        exprs[c] = tuple(pair)
    uses = {c: [("LE" if n == "chord" else n) for e in pair for n in e.names] for c, pair in exprs.items()}
    order = REL.sequence(uses, "curves")
    uses_chord = any("chord" in e.names for pair in exprs.values() for e in pair)
    return exprs, tuple(order), uses_chord, frozenset(c for c, (lo, hi) in exprs.items() if lo.tree == hi.tree)


def law_plan(design):
    """(expressions, order, uses the chord, defined curves) of a design's laws
    (_law_plan)."""
    return _law_plan(_law_key(design.laws), tuple(design.names))


def canonical_laws(laws, names):
    """{curve: (low, high)} with every law written as the tool writes it (one
    word, as a dependent bound); ValueError says what is wrong with a law."""
    laws = {str(c): (str(v[0]), str(v[1])) for c, v in dict(laws or {}).items()}
    exprs = _law_plan(_law_key(laws), tuple(names))[0]
    return {c: (exprs[c][0].text(), exprs[c][1].text()) for c in names if c in exprs}


def _law_array(expr, env, n):
    """A law's value at the n heights of env ({name: array}); nan where it has
    none (a division by zero)."""
    try:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            v = np.asarray(expr.value(env), dtype=float)
    except ZeroDivisionError:
        return np.full(n, np.nan)
    v = np.array(np.broadcast_to(v, (n,)), dtype=float)
    v[~np.isfinite(v)] = np.nan
    return v


def _open(lo, hi):
    """Where a band is open: its laws have numbers and lie apart."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(lo) & np.isfinite(hi) & (np.abs(hi - lo) > _LAW_TOL * np.maximum(
            1.0, np.maximum(np.abs(lo), np.abs(hi))))


def _orig_radius(orig, te_radius_mm=None):
    """The TE radius of the original's model sections: its own when known
    (the section table's R_orig_mm, as original_offsets takes it), else the
    builders'."""
    if orig.te_radius is not None:
        return float(orig.te_radius)
    return TE_RADIUS if te_radius_mm is None else float(te_radius_mm)


def _te(ctrl, r, t=None):
    """(x of the rounded TE, the tangent point's parameter) of each section:
    Newton's method from t when given (bezier_section.te_x_from), fillet's
    bisection otherwise or when Newton does not settle (ValueError when the
    radius does not fit a section)."""
    out = None if t is None else BS.te_x_from(ctrl, r, t)
    if out is None:
        f = BS.fillet(ctrl, r)
        out = (f["x_te"], f["t"])
    return out


def _chord_step(ctrl, chord, r_abs, cs, t=None, ok=None):
    """(cs, t, ok): cs one step closer to the sharp chords, chord / x of the
    rounded TE (the fixed point of bezier_section.sharp_chord); t the tangent
    points' parameters, the start of the next step; ok the sections that can
    be rounded. The others keep their chord (design_problems names them)."""
    out = np.array(chord, dtype=float)
    ok = np.ones(len(out), dtype=bool) if ok is None else ok
    for again in (True, False):
        try:
            if np.any(ok):
                x_te, tk = _te(ctrl[ok], r_abs / np.maximum(cs[ok], out[ok]), None if t is None else t[ok])
                t = np.full(len(out), np.nan)
                out[ok], t[ok] = out[ok] / x_te, tk
            return out, t, ok
        except ValueError:
            if not again:
                break
            ok, t = ~np.any(_problem_masks(out, ctrl, r_abs), axis=0), None
    return np.array(chord, dtype=float), None, ok


def _rounding(ctrl, chord, r_abs, t=None):
    """(cs, t): the sharp chord of each section, as bezier_section.sharp_chord
    finds it, and the parameter of its TE circle's tangent point (None when a
    section cannot be built: that one's cs is its chord). t: the tangent
    parameters of nearly the same sections, to start from."""
    chord = np.array(chord, dtype=float)
    cs, ok = chord, None
    for _ in range(_LAW_STEPS):
        new, t, ok = _chord_step(ctrl, chord, r_abs, cs, t, ok)
        with np.errstate(invalid="ignore"):
            moved = np.any(np.abs(new - cs) > 1.0e-13 * np.abs(new))
        cs = new
        if not moved or t is None:
            break
    return cs, (t if t is not None and bool(np.all(ok)) else None)


def sharp_chord(ctrl, chord, r_abs):
    """The sharp chord of each section, as bezier_section.sharp_chord finds
    it (to rounding) in a fraction of the time; the chord itself for a
    section that cannot be built."""
    return _rounding(ctrl, chord, r_abs)[0]


_sharp = sharp_chord


_MEMO_KEEP = {"pin": 64, "pins": 16, "env": 8, "state": 6, "inside": 2}


def _memo(orig, key, make):
    """make() once for key, kept with the original (the latest few of each
    kind, key[0]: a tool asks the same things of it again and again)."""
    store = orig.__dict__.setdefault("_memo", {}).setdefault(key[0], {})
    try:
        return store[key]
    except KeyError:                                        # (a build off the GUI thread may clear it meanwhile)
        value = make()
        if len(store) >= _MEMO_KEEP[key[0]]:
            store.clear()
        store[key] = value
        return value


def _orig_env(orig, ys, exprs, te_radius_mm=None):
    """What laws may use of the original at the heights ys: "orig.<curve>",
    "TE" and, when one of `exprs` uses it, "orig.chord" (the sharp chord of
    the original's model section, rounded with the original's TE radius)."""
    ys = np.asarray(ys, dtype=float)
    chord = any("orig.chord" in e.names for pair in exprs.values() for e in pair)
    r = _orig_radius(orig, te_radius_mm)

    def make():
        env = {f"orig.{c}": orig.value(c, ys) for c in orig.names}
        env["TE"] = orig.x_te(ys)
        if chord:
            ctrl = BS.polygon_from_values({c: env[f"orig.{c}"] for c in orig.names if c != "LE"}, orig.degree)
            env["orig.chord"] = _sharp(ctrl, env["TE"] - env["orig.LE"], r)
        for v in env.values():
            v.flags.writeable = False
        return env
    return dict(_memo(orig, ("env", chord, r, ys.tobytes()), make))


def law_state(design, orig, ys, te_radius_mm=None):
    """(values, bands, chord) of a pinned design at the heights ys: {curve:
    its values}; {curve: (low, high)} for the curves with bounding laws; the
    sections' sharp chords when a law uses them (else None).

    The curves with laws are worked out in the order of their dependence,
    value = low + fraction (high - low). The chord a law names is the sharp
    chord of the section the laws give, so the two are found together: a
    fixed point from the chord to the rounded TE, a few steps (the sharp
    chord changes little with the shape). te_radius_mm is the TE radius the
    sections are built with (default TE_RADIUS, as the builders)."""
    ys = np.atleast_1d(np.asarray(ys, dtype=float))
    names = design.names
    if not design.laws:
        return {c: design.curve(c).at(ys)[:, 1] for c in names}, {}, None
    if orig is None:
        raise ValueError("a design with bounding laws needs the original shape for its values")
    r = TE_RADIUS if te_radius_mm is None else float(te_radius_mm)
    key = ("state", float(design.H), _law_key(design.laws), r, ys.tobytes(), tuple(
        (np.asarray(cd.s, dtype=float).tobytes(), np.asarray(cd.v, dtype=float).tobytes(),
         None if cd.w is None else np.asarray(cd.w, dtype=float).tobytes())
        for cd in (design.curves[c] for c in names)))
    val, bands, cs = _memo(orig, key, lambda: _law_state(design, orig, ys, r, te_radius_mm))
    return dict(val), dict(bands), cs


def _law_state(design, orig, ys, r, te_radius_mm):
    """law_state for a design that has laws (arrays not to be written to: they are kept)."""
    names = design.names
    raw = {c: design.curve(c).at(ys)[:, 1] for c in names}
    exprs, order, uses_chord, _ = law_plan(design)
    env = _orig_env(orig, ys, exprs, te_radius_mm)
    val = {c: raw[c] for c in names if c not in exprs}
    bands = {}

    def work(curves):
        for c in curves:
            lo, hi = (_law_array(e, {**env, **val}, len(ys)) for e in exprs[c])
            bands[c] = (lo, hi)
            val[c] = lo + raw[c] * (hi - lo)

    def done(cs):
        for v in [*val.values(), *(q for pair in bands.values() for q in pair)] + ([] if cs is None else [cs]):
            v.flags.writeable = False
        return {c: val[c] for c in names}, bands, cs

    if not uses_chord:
        work(order)
        return done(None)
    env["chord"] = np.full(len(ys), np.nan)
    if "LE" in exprs:                                       # the LE and what its laws use do not use the chord
        work(order[:order.index("LE") + 1])
    chord = env["TE"] - val["LE"]
    cs, t, ok = chord, None, None                           # first guess: the chord to the rounded TE
    for _ in range(_LAW_STEPS):
        env["chord"] = cs
        work(order)
        new, t, ok = _chord_step(BS.polygon_from_values(val, design.degree), chord, r, cs, t, ok)
        with np.errstate(invalid="ignore"):
            moved = np.any(np.abs(new - cs) > 1.0e-13 * np.abs(new))
        cs = new
        if not moved or t is None:
            break
    return done(cs)


def law_bands(design, orig, ys, te_radius_mm=None):
    """{curve: (low, high)} at the heights ys, for the curves of a pinned
    design that have bounding laws (law_state)."""
    return law_state(design, orig, ys, te_radius_mm)[1]


def _law_pins(design, orig, te_radius_mm=None):
    """{curve: (f0, low, high, original)} at H for the curves with laws. f0 is
    the fraction that puts the curve on the original at H, (original - low) /
    (high - low), every curve being the original's there (0 when its laws meet
    there or have no number)."""
    exprs, _, uses_chord, _ = law_plan(design)
    r = TE_RADIUS if te_radius_mm is None else float(te_radius_mm)

    def make():
        env = _orig_env(orig, np.array([float(design.H)]), exprs, te_radius_mm)
        for c in orig.names:
            env[c] = env[f"orig.{c}"]
        if uses_chord:
            ctrl = BS.polygon_from_values({c: env[c] for c in orig.names if c != "LE"}, orig.degree)
            env["chord"] = _sharp(ctrl, env["TE"] - env["LE"], r)
        out = {}
        for c, pair in exprs.items():
            lo, hi = (_law_array(e, env, 1) for e in pair)
            o = float(env[c][0])
            out[c] = (float((o - lo[0]) / (hi[0] - lo[0])) if _open(lo, hi)[0] else 0.0, float(lo[0]), float(hi[0]), o)
        return out
    return _memo(orig, ("pins", _law_key(design.laws), float(design.H), r), make)


def _pins(orig, name, s, H, join):
    """{index: value} of the control values fixed by the original at H: C0,
    and with join 'G1' C1, on the original's tangent at its own height."""
    v0, slope = _memo(orig, ("pin", name, float(H)),
                      lambda: (float(orig.value(name, H)[0]), float(orig.slope(name, H)[0])))
    if join == "G0":
        return {0: v0}
    if join == "G1":
        return {0: v0, 1: v0 + slope * (s[1] - 1.0) * H}
    raise ValueError(f"{name}: join must be 'G1' or 'G0', not {join!r}")


def pin(design, orig, te_radius_mm=None):
    """A copy of the design with its pinned control points set from the
    original at H: s_0 = 1 and s_n = 0; the value at H; with join 'G1' the
    value of C1 on the original's tangent at H (its height as designed).
    A curve with bounding laws is pinned by its fraction at H, the one that
    puts it on the original there, and joins with its value only ('G0'); a
    curve its laws define (low and high the same) keeps no control points.
    Checks that H lies in (0, z_top], that the design and the original have
    the same section degree, that no curve's heights rise towards the root
    and that the laws can be read. te_radius_mm: the TE radius the sections
    are built with (the laws' chord follows from it; default TE_RADIUS)."""
    d = design.copy()
    d.H = float(d.H)
    if not 0.0 < d.H <= orig.z_top + _TOL_Y:
        raise ValueError(f"H = {d.H:g} mm must lie in (0, z_top = {orig.z_top:g}]")
    if sorted(d.curves) != sorted(orig.names) or d.degree != orig.degree:
        raise ValueError(f"the design has sections of degree {d.degree} and the original of degree "
                         f"{orig.degree} (curves {sorted(d.curves)}); use change_degree or orig.at_degree")
    d.H = min(d.H, orig.z_top)
    unknown = sorted(set(d.laws) - set(orig.names))
    if unknown:
        raise ValueError(f"bounding laws for curves the design does not have: {unknown}")
    f0 = _law_pins(d, orig, te_radius_mm) if d.laws else {}
    defined = law_plan(d)[3] if d.laws else ()
    for name in orig.names:
        c = d.curves[name]
        if name in defined:                                 # the law is the curve
            c.s, c.v, c.w, c.join = np.array([1.0, 0.0]), np.zeros(2), None, "G0"
            continue
        c.s, c.v = np.array(c.s, dtype=float), np.array(c.v, dtype=float)
        if len(c.s) != len(c.v) or len(c.s) < 2:
            raise ValueError(f"{name}: s and v need one entry per control point, at least 2")
        c.s[0], c.s[-1] = 1.0, 0.0
        if np.any(np.diff(c.s) > 0.0):
            raise ValueError(f"{name}: control heights must not rise from H down to the root")
        if name in d.laws:
            c.join = "G0"
            c.v[0] = f0[name][0]
            continue
        for k, val in _pins(orig, name, c.s, d.H, c.join).items():
            c.v[k] = val
    return d


def baseline_design(orig, H, orders=None, join="G1", weights=None, n_samples=401, laws=None, te_radius_mm=None):
    """The original shape below H as a design: for each curve, evenly spaced
    control heights and the values that fit the original's fitted curve on
    [0, H] best with the pins held. When the order is at least that of the
    original's fit (unit weights) the curve is the original exactly (the
    de Casteljau split of the fit, raised to the order). laws = {curve: (low,
    high)}: those curves between their bounding laws, as near the original
    as the laws let them be (with_laws)."""
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
    d = pin(Design(H, curves, orig.degree), orig)
    for c, (low, high) in canonical_laws(laws, orig.names).items():
        d = with_laws(d, orig, c, low, high, te_radius_mm, n_samples)
    return d


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
    if d.laws:
        raise ValueError("a design with bounding laws keeps its section degree (the laws name its curves): "
                         "take the laws off first")
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
    coordinates in [0, 0.2] (fractions of the sharp chord); the fractions of a
    curve with bounding laws in [0, 1], the whole way between its laws."""
    h_max = float(z_top) if h_max is None else float(h_max)
    h_min = 0.25 * h_max if h_min is None else float(h_min)
    if not 0.0 < h_min <= h_max <= z_top + _TOL_Y:
        raise ValueError(f"H bounds [{h_min:g}, {h_max:g}] mm must satisfy 0 < h_min <= h_max <= z_top = {z_top:g}")
    return {"H": (h_min, h_max), "d": (0.05, 0.95), "LE": (-4.0 * h_max, 0.0), "x": (0.0, 1.0), "z": (0.0, 0.2),
            "f": (0.0, 1.0)}


class DesignSpace(REL.SpaceRelations):
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
               section x / z coordinate, "f" for every fraction between
               bounding laws), by curve ("P2z") or by slot name
               ("LE.dx3": (-40, 0)); a slot beats its curve, a curve its group
    fixed      {slot name: value} of the slots held fixed: they stay out of
               the vector x (the design variables are the other slots)
    laws       {curve: (low law, high law)}: these curves lie between their
               bounding laws (Design.laws). Their slots after the d are the
               fractions of C1 .. Cn, "P2z.f1" .. (C0 follows from the
               original at H), bounds 0 .. 1; a curve its laws define (low
               and high the same) has no slots
    te_radius_mm  the TE radius the sections are built with (the chord the
               laws name follows from it; default TE_RADIUS)

    all_names / all_bounds cover every slot, names / bounds the free ones;
    len(space) is the number of design variables.

    Dependent variables (xgeom_relations.SpaceRelations): set_relations
    (relations, dependent bounds), relations, dependent_bounds, violations(x),
    project(x). A free slot with a dependent bound is in the vector as its
    fraction between its bounds (factors; its row of bounds 0 .. 1): to_design
    places it, to_vector gives it back. A slot defined with '=' leaves the
    vector (full_vector works it out). A set-up file gives a space and a
    design in it (load_component)."""

    def __init__(self, orig, orders=None, join="G1", weights=None, bounds=None, h_min=None, h_max=None,
                 fixed=None, laws=None, te_radius_mm=None):
        self.orig = orig
        self.curves = orig.names
        self.laws = canonical_laws(laws, self.curves)
        self.defined = set(_law_plan(_law_key(self.laws), tuple(self.curves))[3])
        self.te_radius_mm = None if te_radius_mm is None else float(te_radius_mm)
        orders = dict(orders or {})
        self.orders = {c: int(orders.get(c, DEFAULT_ORDER)) for c in self.curves}
        self.join = {c: ("G0" if c in self.laws else join.get(c, "G1") if isinstance(join, dict) else join)
                     for c in self.curves}
        weights = weights or {}
        self.weights = {c: (None if weights.get(c) is None else np.asarray(weights[c], dtype=float))
                        for c in self.curves}
        b = {**default_bounds(orig.z_top, h_min, h_max), **(bounds or {})}
        self._slots = [("", "H", 0)]
        for c in self.curves:
            if c in self.defined:
                continue
            n = self.orders[c]
            self._slots += [(c, "d", k) for k in range(1, n)]
            if c in self.laws:
                self._slots += [(c, "f", k) for k in range(1, n + 1)]
            else:
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
        if kind == "f":
            return b.get(f"{c}.f", b["f"])
        if c in b:
            return b[c]
        return b["LE"] if c == "LE" else b[c[-1]]

    @staticmethod
    def slot_name(c, kind, k):
        if kind == "H":
            return "H"
        if kind == "f":
            return f"{c}.f{k}"
        return f"{c}.d{k}" if kind == "d" else f"{c}.{'dx' if c == 'LE' else 'v'}{k}"

    def __len__(self):
        return len(self.names)

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
            if c in self.defined:                           # the law is the curve
                curves[c] = CurveDesign(np.array([1.0, 0.0]), np.zeros(2), None, "G0")
                continue
            s = heights_from_fractions(d[c])
            v = vals[c]
            free = [k for (cc, kind, k) in self._slots if cc == c and kind == "v"]
            if c == "LE" and free:                          # from the original LE at their heights (H as pin
                v[free] = v[free] + self.orig.value("LE", s[free] * min(H, self.orig.z_top))    # takes it)
            curves[c] = CurveDesign(s, v, self.weights[c], self.join[c])
        return pin(Design(H, curves, self.orig.degree, self.laws), self.orig, self.te_radius_mm)

    def values(self, design):
        """{slot name: value} of a design, every slot (fixed ones as the
        design has them). The design has this space's bounding laws."""
        dd = pin(design, self.orig, self.te_radius_mm)
        if canonical_laws(dd.laws, self.curves) != self.laws:
            raise ValueError("the design's bounding laws are not this space's")
        out = {"H": dd.H}
        for (c, kind, k), name in zip(self._slots[1:], self.all_names[1:]):
            cd = dd.curves[c]
            if kind == "d":
                out[name] = float(fractions_from_heights(cd.s)[k - 1])
            elif kind == "v" and c == "LE":
                out[name] = float(cd.v[k] - self.orig.value("LE", cd.s[k] * dd.H)[0])
            else:
                out[name] = float(cd.v[k])
        return out


# --------------------------------------------------------------------------
# The component file (xgeom_files.py, kind rudder_h): the set-up
# --------------------------------------------------------------------------

FILE_KEYS = ("corner_points", "section_table", "xcad_original", "stack")     # the original's files (an extraction's)
SETTING_TYPES = {"sections": int, "points": int, "te_radius_mm": float, "half": bool, "strip_frac": str,
                 "tip_drop": str, "keep_inside": bool, "inside_tol_mm": float}
DEFAULT_SETTINGS = {"sections": 50, "points": 200, "te_radius_mm": TE_RADIUS, "half": False, "strip_frac": None,
                    "tip_drop": 1.5, "keep_inside": True, "inside_tol_mm": 0.01}
TABLE_ROWS = 21                  # rows of the file's table: the curves from the root to H
VALUE_NOISE = 1e-9               # a variable's value below this (mm, or a fraction) is written as 0
FILE_FIGURES = 1e-9              # a number read from a file is the number written within this share of it


@dataclass
class RudderParams:
    """What a component file says of the original rudder: its files (absolute
    paths: the LE and TE points, the per-section fits, the XCAD loops with
    the cap, the section stack; FILE_KEYS), the section degree, the full
    height (mm, the root to the top of the cap; None: the extracted
    rudder's) and how the original's laws are fitted."""
    files: dict
    degree: int = BASE_DEGREE
    span: float = None
    fit_orders: dict = field(default_factory=dict)
    fit_free_heights: bool = False


def full_height(xcad_original, units="m"):
    """The full height (mm) of the rudder of an extraction's XCAD file: the
    top of its loops (the cap's)."""
    return float(max(xcad_to_working(pts, units)[:, 1].max() for _, pts in read_xcad(xcad_original)))


def _pair_text(v):
    """A strip width or a tip drop as a file has it: none, a number or two."""
    return "none" if v is None else " ".join(XF.num(q) for q in np.atleast_1d(v))


def _pair_value(text, where):
    words = str(text).split()
    if words == ["none"]:
        return None
    try:
        v = [float(q) for q in words]
    except ValueError:
        v = []
    if len(v) not in (1, 2):
        raise ValueError(f"{where}: none, a number, or two (the LE's and the TE's), not '{text}'")
    return v[0] if len(v) == 1 else v


def component_params(comp):
    """(RudderParams, the settings the file gives, typed) of a rudder_h
    component (xgeom_files.read_component); the files' paths are relative to
    the component file. ValueError names what is missing or wrong."""
    if comp.kind != XF.MODIFIED:
        raise ValueError(f"{comp.path}: a {comp.kind} file, not a rudder modified below H (kind {XF.MODIFIED})")
    g, base = comp.geometry, os.path.dirname(comp.path)
    missing = [k for k in FILE_KEYS if k not in g]
    if missing:
        raise ValueError(f"{comp.path}: no {', '.join(missing)}: the original's files (an extraction's "
                         f"{', '.join(FILE_KEYS)})")
    files = {k: XF.resolve(g[k], base) for k in FILE_KEYS}
    for k in FILE_KEYS:
        if not os.path.isfile(files[k]):
            raise ValueError(f"{comp.path}: {k} {g[k]}: no such file ({files[k]})")
    try:
        degree = int(g.get("degree", BASE_DEGREE))
        span = float(g["span_mm"]) if "span_mm" in g else None
        words = g.get("fit_orders", "").split()
        orders = {c: int(o) for c, o in zip(words[::2], words[1::2])}
        if len(words) % 2 or (span is not None and not span > 0.0):
            raise ValueError
    except ValueError:
        raise ValueError(f"{comp.path}: degree is a whole number, span_mm a height in mm and fit_orders pairs of "
                         f"a curve and its order (LE 3 P1x 2 ..)") from None
    settings = XF.typed_settings(comp, SETTING_TYPES)
    for k in ("strip_frac", "tip_drop"):
        if k in settings:
            settings[k] = _pair_value(settings[k], f"{comp.path}: {k}")
    return RudderParams(files, degree, span, orders, XF.as_bool(g.get("fit_free_heights", "N"),
                                                                 f"{comp.path}: fit_free_heights")), settings


def component_original(params):
    """(the original shape of a RudderParams, fitted from its files and
    stretched to its full height; the extracted rudder's own full height,
    mm)."""
    orig = fit_original(params.files["corner_points"], params.files["section_table"], None, params.fit_orders,
                        free_heights=params.fit_free_heights, degree=params.degree)
    span0 = full_height(params.files["xcad_original"])
    k = 1.0 if params.span is None else params.span / span0
    return orig.scaled(1.0 if abs(k - 1.0) < 1e-12 else k), span0        # (a file's 12 figures: not stretched)


def component_design(comp, orig, te_radius_mm=None):
    """(space, design) of a rudder_h component on its original (at the file's
    degree and full height): H and its bounds from its var line; each curve's
    order from its curve line, with how it joins the original at H (G1, G0)
    or its two bounding laws; every variable's value, bounds, free flag,
    relation and dependent bounds from its var line. A curve with no curve
    line is at the order of the original's fit (DEFAULT_ORDER at least),
    tangent at H; a curve with no var lines is the original's there (its
    variables free, P1x's held, the default bounds). H's upper bound within
    the file's figures of the top section is the top section, and H on that
    bound is there with it (a file rounds both). ValueError names the file
    and what is wrong."""
    path, names = comp.path, orig.names
    stray = [c for c in comp.curves if c not in names]
    if stray:
        raise ValueError(f"{path}: curve lines for {', '.join(stray)}: a rudder with sections of degree "
                         f"{orig.degree} has {', '.join(names)}")
    if "H" not in comp.vars:
        raise ValueError(f"{path}: no var line for H, the height of the modified region")
    h = comp.vars["H"]
    if h.lo_expr or h.hi_expr:
        raise ValueError(f"{path}: the bounds of H are numbers (h_min, h_max)")
    orders = {c: comp.curves[c][1] - 1 if c in comp.curves else max(DEFAULT_ORDER, orig.curves[c].order)
              for c in names}
    join = {c: comp.joins.get(c, "G1") for c in names}
    bounds = {n: (q.lo, q.hi) for n, q in comp.vars.items()}
    h_value = h.value
    if abs(h.hi - orig.z_top) <= FILE_FIGURES * orig.z_top:           # h_max at the top section, as the file
        bounds["H"] = (h.lo, orig.z_top)                              # rounds it: the top section itself,
        if h_value >= min(h.hi, orig.z_top):                          # and H on that bound with it
            h_value = orig.z_top
    try:
        kw = dict(bounds=bounds, h_min=bounds["H"][0], h_max=bounds["H"][1], laws=comp.laws,
                  te_radius_mm=te_radius_mm)
        full = DesignSpace(orig, orders, join, **kw)
        vals = full.values(baseline_design(orig, h_value, orders, join, laws=comp.laws, te_radius_mm=te_radius_mm))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    unknown = [n for n in comp.vars if n not in full.all_names]
    if unknown:
        has = {c: [n.split(".")[1] for n in full.all_names if n.startswith(c + ".")] for c in names}
        raise ValueError(f"{path}: var lines for {', '.join(unknown)}, which this rudder does not have (H; "
                         + "; ".join(f"{c}: {', '.join(k) if k else 'none, set by its laws'}" for c, k in has.items())
                         + "; C1's value is no variable of a G1 curve)")
    for c in names:
        own = [n for n in full.all_names if n.startswith(c + ".")]
        given = [n for n in own if n in comp.vars]
        if given and len(given) < len(own):
            raise ValueError(f"{path}: the {c} curve has var lines for some of its variables only; missing: "
                             + ", ".join(n for n in own if n not in given))
    vals.update({n: q.value for n, q in comp.vars.items()})
    vals["H"] = h_value
    fixed = {n: vals[n] for n in full.all_names
             if not (comp.vars[n].free if n in comp.vars else not n.startswith("P1x."))}
    try:
        space = DesignSpace(orig, orders, join, fixed=fixed, **kw)
        space.set_relations({n: q.relation for n, q in comp.vars.items() if q.relation}, "",
                            {n: (q.lo_expr, q.hi_expr) for n, q in comp.vars.items() if q.lo_expr or q.hi_expr}, vals)
        return space, space.design_of(vals)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def load_component(path):
    """The rudder modified below H from its set-up, a component file of kind
    rudder_h (xgeom_files.py): an XF.Loaded with params (RudderParams: the
    original's files, degree, full height and fit), the design, the design
    space (ld.space.orig is the original, fitted from those files and
    stretched to the full height; every variable, the held ones fixed; the
    bounding laws, relations and dependent bounds) and the settings
    (SETTING_TYPES; DEFAULT_SETTINGS for those the file leaves out):

        ld = load_component("outputs/modified/add_to_original.txt")
        x0, bounds = ld.space.to_vector(ld.design), ld.space.bounds      # the free variables
        d = ld.space.to_design(x)                                        # a candidate x
        s = ld.settings
        problems = design_problems(d, ld.space.orig, s["sections"], s["te_radius_mm"])
        if s["keep_inside"]:                              # the original as the lower bound of the shape
            ok = inside_original(d, ld.space.orig, s["sections"], s["te_radius_mm"])["depth"] <= s["inside_tol_mm"]
        write_case(out_dir, name, d, ld.space.orig, ld.params.files["xcad_original"], s["sections"], s["points"],
                   s["te_radius_mm"], space=ld.space)
    """
    comp = XF.read_component(path)
    params, settings = component_params(comp)
    settings = {**DEFAULT_SETTINGS, **settings}
    space, design = component_design(comp, component_original(params)[0], settings["te_radius_mm"])
    return XF.Loaded(XF.MODIFIED, params, design, space, settings, comp.name, list(comp.notes), {})


def component(path, design, space, params, settings=None, name=None, notes=()):
    """The XF.Component that write_component writes to path: the original's
    files of params (relative to the file's folder), the section degree, the
    full height and the fit; the settings (SETTING_TYPES); each curve's
    points with its join at H or its two laws; every variable of space
    (value, bounds, free, relation); the curves' values at TABLE_ROWS
    heights from the root to H, as the design read back from the file's own
    figures has them (so a file read and written again is the same bytes,
    whatever the rudder's size)."""
    d = pin(design, space.orig, space.te_radius_mm)
    if any(cd.w is not None for cd in d.curves.values()):
        raise ValueError("a design with weighted (rational) curves is not written as a component file")
    folder = os.path.dirname(os.path.abspath(path))
    comp = XF.Component(XF.MODIFIED, name or os.path.splitext(os.path.basename(path))[0], list(notes))
    g = {k: XF.rel(os.path.abspath(params.files[k]), folder) for k in FILE_KEYS}
    g["degree"] = str(int(d.degree))
    if params.span is not None:
        g["span_mm"] = XF.num(params.span, 12)
    if params.fit_orders:
        g["fit_orders"] = " ".join(f"{c} {int(o)}" for c, o in params.fit_orders.items())
    if params.fit_free_heights:
        g["fit_free_heights"] = "Y"
    comp.geometry = g
    st = dict(settings or {})
    for k in ("strip_frac", "tip_drop"):
        if k in st:
            st[k] = _pair_text(st[k])
    comp.settings = XF.settings_text(st, SETTING_TYPES)
    for c in d.names:
        comp.curves[c] = ("bezier1", space.orders[c] + 1)
        if c in d.laws:
            comp.laws[c] = d.laws[c]
        else:
            comp.joins[c] = d.curves[c].join
    comp.vars = XF.vars_of(space, d)
    for v in comp.vars.values():                  # as the file has them: 12 figures, and a pinned value's
        v.value = 0.0 if abs(v.value) < VALUE_NOISE else float(XF.num(v.value, 12))    # rounding (1e-12 mm) 0
    comp.path = path
    d = component_design(comp, space.orig, space.te_radius_mm)[1]      # the design a reader of the file has:
    ys = np.linspace(0.0, d.H, TABLE_ROWS)                             # its table, so a rewrite is the same
    val = curve_values(d, ys, space.orig, space.te_radius_mm)
    comp.columns = XF.COLUMNS[XF.MODIFIED] + tuple(BS.coord_names(d.degree))
    comp.table = np.column_stack([ys] + [val[c] for c in d.names])
    return comp


def write_component(path, design, space, params, settings=None, name=None, notes=()):
    """Write the set-up as a component file of kind rudder_h (component),
    which load_component reads back."""
    return XF.write_component(path, component(path, design, space, params, settings, name, notes))


def refit_curve(design, orig, name, order, n_samples=401, te_radius_mm=None):
    """The design with curve `name` refitted at another order: evenly spaced
    control heights and the least-squares values (pins held, weights reset)
    that follow the curve's present shape on [0, H] most closely. A curve
    with bounding laws: its fraction, kept in 0 .. 1 (a curve its laws define
    has nothing to refit)."""
    d = pin(design, orig, te_radius_mm)
    cd = d.curves[name]
    banded = name in d.laws
    if banded and name in law_plan(d)[3]:
        return d
    yd = np.linspace(0.0, d.H, n_samples)
    vd = d.curve(name).at(yd)[:, 1]
    s = uniform_heights(int(order))
    v, _ = fit_values(s * d.H, yd, vd, None, {0: float(cd.v[0])} if banded else _pins(orig, name, s, d.H, cd.join))
    if banded:
        v[1:] = np.clip(v[1:], 0.0, 1.0)
    out = d.copy()
    out.curves[name] = CurveDesign(s, v, None, cd.join)
    return pin(out, orig, te_radius_mm)


def with_laws(design, orig, name, low=None, high=None, te_radius_mm=None, n_samples=401, order=None):
    """The design with curve `name` put between a low law and a high law
    (both given; law_names says what they may use), or with its laws taken
    off (both None), its present shape kept as closely as that allows:

    laws given  the curve's control values become the fractions between the
                laws that follow its present values on [0, H] most closely
                (least squares at its control heights, C0 pinned to the
                original at H), kept in 0 .. 1: where the curve lay outside
                its laws it now lies on the nearer one. Low and high the same
                expression define the curve, which then has no control points.
    laws off    the control values that follow its present values most
                closely, C0 on the original at H (join 'G0').

    order: the curve's order, when it has no control heights to keep (it was
    defined by its laws; default DEFAULT_ORDER). ValueError says what is
    wrong with a law."""
    d = pin(design, orig, te_radius_mm)
    if name not in d.curves:
        raise ValueError(f"'{name}' is not a curve of this design ({', '.join(d.names)})")
    low, high = (None if q is None or not str(q).strip() else str(q).strip() for q in (low, high))
    if (low is None) != (high is None):
        raise ValueError(f"{name}: a curve has a low law and a high law, or neither")
    if low is None and name not in d.laws:
        return d
    yd = np.linspace(0.0, d.H, int(n_samples))
    val, _, cs = law_state(d, orig, yd, te_radius_mm)
    out = d.copy()
    kept = name not in d.laws or name not in law_plan(d)[3]
    s = np.array(d.curves[name].s, dtype=float) if kept else uniform_heights(int(order or DEFAULT_ORDER))
    laws = dict(d.laws)
    if low is None:
        laws.pop(name)
        out.laws = laws
        v, _ = fit_values(s * d.H, yd, val[name], None, _pins(orig, name, s, d.H, "G0"))
        out.curves[name] = CurveDesign(s, v, None, "G0")
        return pin(out, orig, te_radius_mm)
    laws[name] = (low, high)
    out.laws = canonical_laws(laws, d.names)
    exprs, _, _, defined = law_plan(out)
    if name in defined:
        out.curves[name] = CurveDesign(np.array([1.0, 0.0]), np.zeros(2), None, "G0")
        return pin(out, orig, te_radius_mm)
    pair = exprs[name]
    env = {**_orig_env(orig, yd, {name: pair}, te_radius_mm), **val}
    if any("chord" in e.names for e in pair):               # the chord of the design as it is now
        env["chord"] = cs if cs is not None else _sharp(
            BS.polygon_from_values(val, d.degree), env["TE"] - val["LE"],
            TE_RADIUS if te_radius_mm is None else float(te_radius_mm))
    lo, hi = (_law_array(e, env, len(yd)) for e in pair)
    opened = _open(lo, hi)
    f = np.clip(np.where(opened, (val[name] - lo) / np.where(opened, hi - lo, 1.0), 0.0), 0.0, 1.0)
    out.curves[name] = CurveDesign(s, np.zeros(len(s)), None, "G0")
    v, _ = fit_values(s * d.H, yd, f, None, {0: _law_pins(out, orig, te_radius_mm)[name][0]})
    v[1:] = np.clip(v[1:], 0.0, 1.0)
    out.curves[name] = CurveDesign(s, v, None, "G0")
    return pin(out, orig, te_radius_mm)


def control_points(design, orig, name, te_radius_mm=None):
    """(heights, values) of a pinned design's control points of curve `name`,
    as a plot of the curve's values shows them. A curve with bounding laws:
    each control fraction placed between the laws at its own height (none for
    a curve its laws define)."""
    cd = design.curves[name]
    y = np.asarray(cd.s, dtype=float) * design.H
    if name not in design.laws:
        return y, np.array(cd.v, dtype=float)
    if name in law_plan(design)[3]:
        return np.zeros(0), np.zeros(0)
    lo, hi = law_bands(design, orig, y, te_radius_mm)[name]
    return y, lo + np.asarray(cd.v, dtype=float) * (hi - lo)


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
    selig: np.ndarray            # (num_pts, 2) the model's section: airfoil frame, mm, Selig order
    dx_le: float = 0.0           # with resid, what the extracted original has at this height that its model
    resid: np.ndarray = None     # leaves out: x of the LE (mm) and (num_pts, 2) point by point (Offsets.at)

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

    def written(self):
        """(x of the LE, the Selig points) of the section as it is written: the
        model's (x_le, selig), plus, when the section carries them, the
        offsets of the extracted original from its own model at this height
        (dx_le, resid). The baseline design then gives the extracted section
        itself and any other design that section plus the change the design
        makes to the model."""
        if self.resid is None:
            return self.x_le, self.selig
        return self.x_le + self.dx_le, self.selig + self.resid

    def loop(self):
        """The section as an XCAD loop, working frame, mm (as written())."""
        x_le, xy = self.written()
        return xcad_loop(xy, x_le, self.y)


def curve_values(design, ys, orig=None, te_radius_mm=None):
    """{curve: its values at the heights ys} for a pinned design. A design
    with bounding laws needs the original and takes the TE radius its
    sections are built with (law_state)."""
    return law_state(design, orig, ys, te_radius_mm)[0]


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


def _lawless(design, val):
    """(curve, mask) of the first curve, in the order of dependence, whose
    bounding laws give no number at some height (a division by zero); None
    when every curve has its values."""
    if not design.laws:
        return None
    for c in law_plan(design)[1]:
        bad = ~np.isfinite(val[c])
        if np.any(bad):
            return c, bad
    return None


def _model(design, orig, ys, te_radius_mm, first_index=1, pinned=False):
    """(pinned design, curve values, x of the LE, x of the TE, chords, control
    polygons) at the heights ys; ValueError naming the first section that
    cannot be built and why. pinned: the design is pinned already."""
    d = design if pinned else pin(design, orig, te_radius_mm)
    val = curve_values(d, ys, orig, te_radius_mm)
    hit = _lawless(d, val)
    if hit:
        k = int(np.argmax(hit[1]))
        raise ValueError(f"section {first_index + k} at y = {ys[k]:.4f} mm: the bounding laws of {hit[0]} give "
                         f"no number there (a division by zero)")
    x_le, x_te = val["LE"], orig.x_te(ys)
    chord = x_te - x_le
    ctrl = BS.polygon_from_values(val, d.degree)
    for mask, text in zip(_problem_masks(chord, ctrl, te_radius_mm), _PROBLEM_TEXT):
        if np.any(mask):
            k = int(np.argmax(mask))
            raise ValueError(f"section {first_index + k} at y = {ys[k]:.4f} mm: {text} "
                             f"({', '.join(f'{n} {v[k]:.5f}' for n, v in val.items() if n != 'LE')}, "
                             f"LE x {x_le[k]:.4f}, TE x {x_te[k]:.4f})")
    return d, val, x_le, x_te, chord, ctrl


def build_sections(design, orig, ys, num_pts=200, te_radius_mm=0.75, first_index=1, offsets=None):
    """The sections of a pinned design at the heights ys (all at once, see
    modified_sections). Raises ValueError naming the first section that
    cannot be built and why. With offsets (original_offsets) each section
    also carries what the extracted original has at its height that the model
    leaves out, and is written with it (ModSection.written); a section too
    thin to take them (its outline would pass through the chord line) cannot
    be built either."""
    ys = np.atleast_1d(np.asarray(ys, dtype=float))
    d, val, x_le, x_te, chord, ctrl = _model(design, orig, ys, te_radius_mm, first_index)
    selig, cs, r = BS.sections(ctrl, chord, te_radius_mm, num_pts)
    dx, res = (np.zeros(len(ys)), [None] * len(ys)) if offsets is None else offsets.at(ys, num_pts)
    if offsets is not None:                   # the written outline must stay clear of the chord line, as the model's
        z = selig[:, :, 1] + res[:, :, 1]
        i_le = (selig.shape[1] + 1) // 2 - 1
        thin = (z[:, 1:i_le].min(axis=1) <= 0.0) | (z[:, i_le + 1:-1].max(axis=1) >= 0.0)
        if np.any(thin):
            k = int(np.argmax(thin))
            raise ValueError(f"section {first_index + k} at y = {ys[k]:.4f} mm: the section is too thin to "
                             f"write. What the extracted original has that the model leaves out (up to "
                             f"{np.abs(res[k, :, 1]).max():.3f} mm here) takes its outline through the chord "
                             f"line")
    return [ModSection(first_index + k, float(ys[k]), float(x_le[k]), float(x_te[k]), float(chord[k]),
                       float(cs[k]), float(r[k]), ctrl[k], selig[k], float(dx[k]), res[k]) for k in range(len(ys))]


def modified_sections(design, orig, n_sec=50, num_pts=200, te_radius_mm=0.75, offsets=None):
    """The n_sec horizontal sections from the root to H (root first).

    At each height: the LE from the design's LE curve, the TE from the TE
    line, the section control points from their curves, and the sharp chord
    that puts the TE, rounded with te_radius_mm, on the TE line
    (bezier_section.sharp_chord). num_pts points per section in the
    extraction's layout. offsets: see build_sections."""
    d = pin(design, orig, te_radius_mm)
    return build_sections(d, orig, np.linspace(0.0, d.H, int(n_sec)), num_pts, te_radius_mm, offsets=offsets)


def section_at(design, orig, y, num_pts=200, te_radius_mm=0.75, index=0, offsets=None):
    """The section of a design at any height y in [0, H], built as
    modified_sections builds its own (for previews)."""
    return build_sections(design, orig, [float(y)], num_pts, te_radius_mm, index, offsets)[0]


def design_problems(design, orig, n_sec=50, te_radius_mm=0.75):
    """What would stop modified_sections, found cheaply at its n_sec heights:
    control heights out of order or H out of range, an LE at or behind the TE,
    section control points folding the half-section back, a thickness that is
    not positive, a TE radius that does not fit under the half-thickness; a
    bounding law that cannot be read or gives no number. A list of messages,
    empty when the design builds; for an interactive tool and as a
    feasibility test for an optimiser. (A section too thin to take the
    original's offsets is found only by building it, build_sections.)"""
    try:
        d = pin(design, orig, te_radius_mm)
    except ValueError as exc:
        return [str(exc)]
    ys = np.linspace(0.0, d.H, int(n_sec))
    val = curve_values(d, ys, orig, te_radius_mm)
    hit = _lawless(d, val)
    if hit:
        y = ys[hit[1]]
        return [f"the bounding laws of {hit[0]} give no number at {int(hit[1].sum())} of {len(ys)} sections "
                f"(y = {y.min():.2f} .. {y.max():.2f} mm): a division by zero"]
    chord = orig.x_te(ys) - val["LE"]
    out = []
    for bad, what in zip(_problem_masks(chord, BS.polygon_from_values(val, d.degree), te_radius_mm), _PROBLEM_TEXT):
        if np.any(bad):
            y = ys[bad]
            out.append(f"{what} at {int(bad.sum())} of {len(ys)} sections (y = {y.min():.2f} .. {y.max():.2f} mm)")
    return out


def law_warnings(design, orig, n_sec=50, te_radius_mm=None):
    """What the bounding laws of a design leave open, as lines for a tool,
    found at modified_sections' n_sec heights: a curve whose laws cross
    (the low law above the high one at some heights and below it at others;
    the curve still lies between them, the fraction running the other way);
    a curve that the laws keep off the original at H (they leave the
    original out there, so the curve leaves them near H; or they meet there
    away from the original, a step in the surface at H). Empty without laws,
    and when the design cannot be pinned (design_problems says why)."""
    if not design.laws:
        return []
    try:
        d = pin(design, orig, te_radius_mm)
    except ValueError:
        return []
    ys = np.linspace(0.0, d.H, int(n_sec))
    bands = law_bands(d, orig, ys, te_radius_mm)
    pins = _law_pins(d, orig, te_radius_mm)
    out = []
    for c in law_plan(d)[1]:
        lo, hi = bands[c]
        opened = _open(lo, hi)
        with np.errstate(invalid="ignore"):
            above = opened & (lo > hi)
        if np.any(above) and np.any(opened & ~above):       # one way at some heights, the other at others
            y = ys[above]
            out.append(f"{c}: its laws cross, the low law above the high one at {int(above.sum())} of {len(ys)} "
                       f"sections (y = {y.min():.2f} .. {y.max():.2f} mm)")
        f0, l0, h0, o = pins[c]
        if not (np.isfinite(l0) and np.isfinite(h0)):
            continue
        if not _open(np.array([l0]), np.array([h0]))[0]:
            if abs(l0 - o) > 1.0e-5 * max(1.0, abs(o)):
                out.append(f"{c}: its laws meet at H at {l0:.5g}, away from the original ({o:.5g}): a step in "
                           f"the surface at H")
        elif f0 < -1.0e-7 or f0 > 1.0 + 1.0e-7:
            out.append(f"{c}: the original at H ({o:.5g}) lies outside its laws there ({min(l0, h0):.5g} .. "
                       f"{max(l0, h0):.5g}), so the curve leaves them near H")
    return out


_INSIDE_EPS = 1.0e-6             # mm: a depth under the enclosure measure's own resolution counts as none


def _half_outline(ctrl, cs, x_le, t_f, n=321, n_arc=24):
    """The rounded half-outline of model sections, mm in the rudder's frame:
    (points (m, n + n_arc, 2) from the LE to the TE, the half-section at n
    parameters up to its TE circle's tangent point (parameter t_f; closer
    together towards the LE) and n_arc points of the circle; n; the circle's
    centre x and radius; x of the tangent point), to measure other outlines
    against and to be measured (_half_at)."""
    p = BS.points(ctrl, t_f[:, None] * (np.linspace(0.0, 1.0, int(n))**1.5)[None, :])
    dp = BS.derivative(ctrl, t_f[:, None])[:, 0]
    end = p[:, -1]
    cx = end[:, 0] + dp[:, 1] / dp[:, 0] * end[:, 1]
    rr = np.hypot(end[:, 0] - cx, end[:, 1])
    ang = np.arctan2(end[:, 1], end[:, 0] - cx)[:, None] * (1.0 - np.arange(1, int(n_arc) + 1) / int(n_arc))[None, :]
    arc = np.stack((cx[:, None] + rr[:, None] * np.cos(ang), rr[:, None] * np.sin(ang)), axis=2)
    pts = np.concatenate((p, arc), axis=1) * cs[:, None, None]
    pts[:, :, 0] += x_le[:, None]
    return pts, int(n), x_le + cx * cs, rr * cs, x_le + end[:, 0] * cs


def _half_at(x, outline, i):
    """Half-thickness (mm) of section i of an outline (_half_outline) at the
    stations x (mm): between its half-section's points up to the tangent
    point, on the circle itself from there, nothing ahead of the LE or
    behind the TE."""
    pts, n, cx, rr, x_tan = outline
    z = np.interp(x, pts[i, :n, 0], pts[i, :n, 1], left=0.0)
    arc = x > x_tan[i]
    z[arc] = np.sqrt(np.clip(rr[i]**2 - (x[arc] - cx[i])**2, 0.0, None))
    return z


def inside_original(design, orig, n_sec=50, te_radius_mm=0.75):
    """How far the modified rudder goes inside the original: on the model's
    sections at modified_sections' n_sec heights, the largest depth (mm) of
    the modified half-thickness under the original's at the same x, over the
    points of the original's rounded half-outline. 0 where the modified
    section encloses the original's (a depth under a millionth of a mm, the
    measure's own resolution, counts as none); the baseline design gives 0.

    Returns {"depth": the largest depth (mm), "y": the height it is at (mm),
    "x_c": its station (fraction of the original's chord there), "ys": the
    heights, "depths": the largest depth at each}. The original's sections
    take its own TE radius when known (as original_offsets). Laws and bounds
    on control points do not settle this; an optimiser keeps "depth" within
    its tolerance. ValueError when a section cannot be built
    (design_problems)."""
    d = pin(design, orig, te_radius_mm)
    ys = np.linspace(0.0, d.H, int(n_sec))
    d, val, x_le, x_te, chord, ctrl = _model(d, orig, ys, te_radius_mm, pinned=True)
    r_o = _orig_radius(orig, te_radius_mm)

    def original():                                         # the original's half-outline, and its tangent points
        vo = {c: orig.value(c, ys) for c in orig.names}
        ctrl_o = BS.polygon_from_values(vo, orig.degree)
        cs_o, t_o = _rounding(ctrl_o, x_te - vo["LE"], r_o)
        if t_o is None:
            raise ValueError("the original's own sections cannot be rounded with its TE radius")
        return _half_outline(ctrl_o, cs_o, vo["LE"], t_o)[0], t_o, vo["LE"]
    pts_o, t_o, le_o = _memo(orig, ("inside", r_o, ys.tobytes()), original)
    cs, t = _rounding(ctrl, chord, te_radius_mm, t_o)
    if t is None:
        raise ValueError("a section cannot be rounded with the TE radius (design_problems)")
    out = _half_outline(ctrl, cs, x_le, t)
    depth = np.array([pts_o[i, :, 1] - _half_at(pts_o[i, :, 0], out, i) for i in range(len(ys))])
    k = depth.argmax(axis=1)
    per = np.maximum(depth[np.arange(len(ys)), k], 0.0)
    per[per < _INSIDE_EPS] = 0.0
    i = int(np.argmax(per))
    x_c = (pts_o[i, k[i], 0] - le_o[i]) / (x_te[i] - le_o[i])
    return {"depth": float(per[i]), "y": float(ys[i]), "x_c": float(x_c), "ys": ys, "depths": per}


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


def _to_polyline(q, line):
    """Distance of each point of q (m, 2) to the polyline `line` (k, 2)."""
    a, ab = line[:-1], np.diff(line, axis=0)
    l2 = (ab**2).sum(axis=1)
    a, ab, l2 = a[l2 > 0], ab[l2 > 0], l2[l2 > 0]
    t = np.clip(np.einsum("mkd,kd->mk", q[:, None, :] - a[None], ab) / l2, 0.0, 1.0)
    return np.sqrt((((a[None] + t[..., None] * ab[None]) - q[:, None, :])**2).sum(axis=2)).min(axis=1)


def scaled_stack(stack, k):
    """The stack (read_stack) stretched in height by k (each section's height times k; its points, in the
    airfoil frame, kept), as OriginalShape.scaled stretches the original."""
    return [StackSection(s.y * float(k), s.chord, s.x_le, s.x_te, s.tilt_deg, s.xy) for s in stack]


# --------------------------------------------------------------------------
# What the model leaves out: the extracted sections against their fit
# --------------------------------------------------------------------------

_TOL_STACK = 1.0e-4              # mm, slack on the heights of extracted sections (their files print 5, 6 decimals)


def _relayout(a, num_pts):
    """Values at the points of sections in the extraction's layout, (m, n, k),
    carried to num_pts points per section: each side keeps its cosine
    stations, so a value goes over along its side's station angle (a cubic
    spline through the stations it is given at)."""
    n = a.shape[1]
    num_pts = n if num_pts is None else int(num_pts)
    if num_pts == n:
        return a
    from scipy.interpolate import CubicSpline

    def side(v, p):                                          # v (m, q, k), from the LE to the TE
        q = v.shape[1]
        return CubicSpline(np.arange(q), v, axis=1)(np.linspace(0.0, q - 1.0, p))

    iu, ju = (n + 1) // 2, (num_pts + 1) // 2
    up = side(a[:, :iu][:, ::-1], ju)[:, ::-1]
    lo = side(a[:, iu - 1:], num_pts - ju + 1)[:, 1:]
    return np.concatenate((up, lo), axis=1)


@dataclass
class Offsets:
    """What the extracted original has that its model leaves out
    (original_offsets), at the heights of its horizontal sections, root
    first."""
    y: np.ndarray                # (m,) heights, mm
    dx_le: np.ndarray            # (m,) x of the extracted LE minus the fitted LE, mm
    resid: np.ndarray            # (m, n, 2) the extracted points minus the model's: airfoil frame, mm, Selig order
    te_radius: float             # TE radius of the model sections they were measured from, mm
    gap_max: np.ndarray = None   # (m,) largest and RMS distance of the extracted section's upper-surface points
    gap_rms: np.ndarray = None   # from the model's outline, mm (the section table's gap_max_um, gap_rms_um)

    def gap(self, y_max=None):
        """(largest, RMS) distance (mm) of the extracted sections from the
        model's, over the sections up to y_max (all of them by default)."""
        m = np.ones(len(self.y), dtype=bool) if y_max is None else self.y <= y_max + _TOL_STACK
        return float(self.gap_max[m].max()), float(np.sqrt(np.mean(self.gap_rms[m]**2)))

    def at(self, ys, num_pts=None):
        """(dx_le (k,), resid (k, num_pts, 2)) at the heights ys: linear in the
        height between the sections either side (the offsets are small and
        change slowly up the span, so this is far closer than interpolating
        the sections themselves), then carried to num_pts points per section
        when the extracted sections have another count (_relayout)."""
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        if ys.min() < self.y[0] - _TOL_STACK or ys.max() > self.y[-1] + _TOL_STACK:
            raise ValueError(f"heights {ys.min():g} .. {ys.max():g} mm lie outside the extracted sections "
                             f"the offsets come from, {self.y[0]:g} .. {self.y[-1]:g} mm")
        j = np.clip(np.searchsorted(self.y, ys), 1, len(self.y) - 1)
        f = np.clip((ys - self.y[j - 1]) / (self.y[j] - self.y[j - 1]), 0.0, 1.0)
        dx = (1.0 - f) * self.dx_le[j - 1] + f * self.dx_le[j]
        res = (1.0 - f)[:, None, None] * self.resid[j - 1] + f[:, None, None] * self.resid[j]
        return dx, _relayout(res, num_pts)


def original_offsets(orig, stack, te_radius_mm=None):
    """The offsets of the extracted original from its model (Offsets): at the
    height of each horizontal section of `stack` (read_stack, stack_of_loops;
    at the original's height scale) up to z_top, the extracted section minus
    the section the model builds there from the original's fits, point by
    point (both lay their points on the same cosine stations of each side),
    and the same for x of the LE.

    The model sections take the original's TE radius (orig.te_radius, the
    section table's R_orig_mm; te_radius_mm when the table does not give it),
    so a design with another TE radius still shows that change. A section
    just above z_top is measured against the fits carried on along their
    tangents, so every height up to z_top lies between two sections.

    Offsets.at(y) added to the model's section of a design at y
    (build_sections, ModSection.written) is the extracted section there plus
    the change the design makes to the model."""
    r = orig.te_radius if orig.te_radius is not None else te_radius_mm
    if r is None:
        raise ValueError("the original's TE radius is not known (no R_orig_mm in the section table): "
                         "give te_radius_mm")
    flat = sorted((s for s in stack if s.tilt_deg == 0.0), key=lambda s: s.y)
    flat = [s for k, s in enumerate(flat) if k == 0 or s.y > flat[k - 1].y + _TOL_Y]      # one per height
    n_in = sum(1 for s in flat if s.y <= orig.z_top + _TOL_STACK)
    flat = flat[:n_in + 1]                                  # and one above z_top, when there is one
    if n_in < 1 or len(flat) < 2:
        raise ValueError(f"the extracted sections do not cover the fitted original, 0 .. {orig.z_top:g} mm: "
                         f"{len(flat)} horizontal section(s) there")
    if len({len(s.xy) for s in flat}) != 1:
        raise ValueError("the extracted sections do not all have the same number of points")
    ys = np.array([s.y for s in flat])
    yc = np.clip(ys, 0.0, orig.z_top)                       # the fits end at z_top: on along their tangents
    val = {c: orig.value(c, yc) + orig.slope(c, orig.z_top)[0] * (ys - yc) for c in orig.names}
    n = len(flat[0].xy)
    selig = BS.sections(BS.polygon_from_values(val, orig.degree), orig.x_te(ys) - val["LE"], r, n)[0]
    res = np.array([s.xy for s in flat]) - selig
    n_up = (n + 1) // 2
    x0 = res[:, n_up - 1, 0].copy()                         # the LE point's own x goes with the LE
    res[:, :, 0] -= x0[:, None]
    dx = np.array([s.x_le for s in flat]) - val["LE"] + x0
    gap = [_to_polyline(s.xy[:n_up] + [dx[k], 0.0], selig[k, :n_up]) for k, s in enumerate(flat)]
    return Offsets(ys, dx, res, float(r), np.array([g.max() for g in gap]),
                   np.array([np.sqrt(np.mean(g**2)) for g in gap]))


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


def stack_of_loops(loops):
    """The horizontal loops of an XCAD file (working frame, mm, as
    xcad_to_working gives them) as stack sections, root first: what read_stack
    gives for the extraction's stack file, from the loops themselves
    (xcad_loop undone)."""
    out = []
    for lp in loops:
        lp = np.asarray(lp, dtype=float)
        if np.ptp(lp[:, 1]) > 1e-6 * max(1.0, np.abs(lp[:, 1]).max()):
            continue
        i_le = (len(lp) + 1) // 2 - 1                       # the TE: the loop runs LE, +z side, TE, -z side, LE
        q = np.column_stack((lp[:, 0] - lp[0, 0], lp[:, 2]))
        xy = np.vstack([q[:i_le + 1][::-1], q[i_le + 1:][::-1][1:], q[i_le:i_le + 1]])
        out.append(StackSection(float(lp[0, 1]), float(xy[0, 0]), float(lp[0, 0]), float(lp[0, 0] + xy[0, 0]),
                                0.0, xy))
    return sorted(out, key=lambda s: s.y)


def stack_of_xcad(path, units="m", height_scale=1.0):
    """stack_of_loops of an XCAD point file, its heights stretched by
    height_scale."""
    k = np.array([1.0, float(height_scale), 1.0])
    return stack_of_loops([xcad_to_working(pts, units) * k for _, pts in read_xcad(path)])


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
                        original_units="m", height_scale=1.0):
    """The modified rudder as one XCAD file: the modified sections from the
    root to H (as written, ModSection.loop), then the loops of the original
    file above them (kept_loops: the extracted sections from H + gap up, the
    tilted tip section and the cap), copied line for line when the units
    match. The gap (default one modified step, H / (n_sec - 1)) leaves out
    the extracted sections just above H, which would sit a fraction of a
    millimetre from the section at H (and, for sections given without their
    offsets, differ from it by the fit error); the tip section and the cap
    stay whatever H is. height_scale (the original's scale) stretches the
    original's loops in height first. Returns what was written."""
    gap = H / (len(sections) - 1) if gap_mm is None else float(gap_mm)
    loops = read_xcad(original_xcad)
    if float(height_scale) != 1.0:                          # XCAD's Z is the height
        loops = [(None, pts * np.array([1.0, 1.0, float(height_scale)])) for _, pts in loops]
    kept = kept_loops(loops, H, gap, original_units)
    out = [xcad_lines(s.loop(), units) for s in sections]
    for k in kept:
        lines, pts = loops[k]
        out.append(lines if lines is not None and units == original_units
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


def design_record(design, orig, te_radius_mm=None):
    """The pinned design as plain lists, for JSON. A curve with bounding laws
    also has them ("low", "high"), and its "value" entries are its fractions
    between them; "laws" lists them all."""
    d = pin(design, orig, te_radius_mm)
    out = {"H_mm": d.H, "degree": int(d.degree), "curves": {}}
    for c in d.names:
        cd = d.curves[c]
        out["curves"][c] = {
            "order": cd.order, "join": cd.join, "what": describe(c),
            "s": cd.s.tolist(), "y_mm": (cd.s * d.H).tolist(), "value": cd.v.tolist(),
            "weights": (np.ones(cd.order + 1) if cd.w is None else np.asarray(cd.w)).tolist()}
        if c in d.laws:
            out["curves"][c].update(low=d.laws[c][0], high=d.laws[c][1],
                                    what=describe(c) + "; the values are its fractions between its laws")
    if d.laws:
        out["laws"] = {c: list(v) for c, v in d.laws.items()}
    return out


def design_from_record(rec):
    """The Design of a design_record (or of the 'design' entry of a
    <case>_design.json)."""
    curves = {c: CurveDesign(np.array(r["s"]), np.array(r["value"]),
                             None if np.allclose(r["weights"], 1.0) else np.array(r["weights"]), r["join"])
              for c, r in rec["curves"].items()}
    return Design(float(rec["H_mm"]), curves, rec.get("degree"), rec.get("laws") or {})


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
                                      H + gap up (write_modified_xcad). Each
                                      modified section is the extracted one at
                                      its height (from the horizontal loops of
                                      original_xcad) plus the change the design
                                      makes to the model (original_offsets), so
                                      the baseline writes the original itself
    <case>_sections.dat               the modified sections: heights, LE/TE,
                                      chords, t/c, LE radius and control points
                                      (the model)
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
    d = pin(design, orig, te_radius_mm)
    off = original_offsets(orig, stack_of_xcad(original_xcad, original_units, orig.scale), te_radius_mm)
    secs = modified_sections(d, orig, n_sec, num_pts, te_radius_mm, off)
    gap_max, gap_rms = off.gap(d.H)                         # how far the model alone is from the extracted sections
    p = {k: os.path.join(out_dir, f"{case}_{name}") for k, name in
         (("xcad", f"sections_xcad_{units}.dat"), ("table", "sections.dat"), ("params", "thickness_params.txt"),
          ("json", "design.json"), ("png", "check.png"))}
    xinfo = write_modified_xcad(p["xcad"], secs, original_xcad, d.H, gap_mm, units, original_units, orig.scale)
    xinfo.update(sections="the extracted section at each height plus the change the design makes to the model",
                 model_off_max_mm=gap_max, model_off_rms_mm=gap_rms, te_radius_original_mm=off.te_radius)
    quartic = all(s.params() is not None for s in secs)
    if quartic:
        write_params_txt(p["params"], secs)
    else:
        p.pop("params")
    notes = [f"case         : {case}; H = {d.H:g} mm (original fitted up to {orig.z_top:g} mm), {len(secs)} sections, "
             f"TE radius {te_radius_mm:g} mm" + (f"; the original stretched in height by {orig.scale:.6g}"
                                                  if orig.scale != 1.0 else ""),
             f"XCAD file    : {os.path.basename(p['xcad'])}, these sections then the original loops "
             f"{xinfo['kept'][0] if xinfo['kept'] else '-'} .. {xinfo['kept'][-1] if xinfo['kept'] else '-'}"
             f" of {os.path.basename(original_xcad)}"
             + (f" (from y = {xinfo['first_kept_y']:.4f} mm)" if xinfo["kept"] else ""),
             "XCAD loops   : each section as extracted from the original at its height, plus the change the "
             "design makes to the model;",
             f"               the model alone (the rows below) lies up to {gap_max:.3f} mm, RMS {gap_rms:.3f}, "
             f"off the extracted sections below H"]
    write_sections_dat(p["table"], secs, notes)
    rec = {"case": case, "written": f"{datetime.date.today():%d %b %Y}",
           "settings": {"z_top_mm": orig.z_top, "height_scale": orig.scale,
                        "h_min_mm": None if h_min is None else float(h_min),
                        "h_max_mm": None if h_max is None else float(h_max),
                        "degree": int(d.degree), "n_sections": len(secs), "num_pts": num_pts,
                        "te_radius_mm": te_radius_mm, "gap_mm": xinfo["gap_mm"], "xcad_units": units},
           "thickness_format": ("written" if quartic else
                                "not written: the sections are not quartics with P1x = 0 and P1z = P2z"),
           "design": design_record(d, orig, te_radius_mm),
           "original": {**original_record(orig),
                        "at_H": {c: {"value": float(orig.value(c, d.H)[0]),
                                     "slope_per_mm": float(orig.slope(c, d.H)[0])} for c in orig.names}},
           "xcad": {k: v for k, v in xinfo.items() if k != "path"},
           "files": {k: os.path.basename(v) for k, v in p.items()}}
    if space is not None:                                 # names, values, bounds; "fractions" when any
        rec["vector"] = space.vector_record(space.values(d))
    with open(p["json"], "w") as fh:
        json.dump(rec, fh, indent=1)
    if plot:
        plot_case(p["png"], case, d, orig, secs, stack, te_radius_mm=te_radius_mm)
    else:
        p.pop("png")
    return {"case": case, "design": d, "sections": secs, "xcad": xinfo, "paths": p}


def distance_to_original(sec, stack):
    """Distance (mm) of each upper-surface point of the original section at
    the height of `sec` (original_section_at) to the upper surface of `sec`
    as written (ModSection.written), both where they sit in x (each from its
    own LE). For a changed design this is the change. For the baseline it is
    nothing when the section carries the original's offsets, and the error of
    the model when it does not."""
    x_le_o, xy_o = original_section_at(stack, sec.y)
    n_up = (len(xy_o) + 1) // 2
    q = xy_o[:n_up] + [x_le_o, 0.0]
    x_le, xy = sec.written()
    return _to_polyline(q, xy[:(len(xy) + 1) // 2] + [x_le, 0.0])


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
    if "model_off_max_mm" in x:
        lines.append(f"  XCAD sections: the extracted ones plus the design's change to the model (the model alone "
                     f"lies up to {x['model_off_max_mm']:.3f} mm, RMS {x['model_off_rms_mm']:.3f}, off them)")
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

def plot_case(path, case, design, orig, sections, stack=None, every=7, te_radius_mm=None):
    """Planform and every spanwise curve (as the rudder stands: x, or the
    curve's value, across and the height up) and three sections of a case, in
    one PNG. A curve with bounding laws is drawn with them (dotted). Drawn on
    a bare Figure (no pyplot), so it also runs off a GUI thread."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    d = pin(design, orig, te_radius_mm)
    H = d.H
    top = min(orig.z_top, max(1.3 * H, H + 20.0))           # the plots show the original up to here
    yy = np.linspace(0.0, H, 400)
    yz = np.linspace(0.0, top, 400)
    val, bands, _ = law_state(d, orig, yy, te_radius_mm)

    def said(c):
        cd = d.curves[c]
        return "set by its laws" if not len(control_points(d, orig, c, te_radius_mm)[0]) else (
            f"order {cd.order}, " + ("between its laws" if c in d.laws else cd.join))
    coords = BS.coord_names(d.degree)
    ncol = 4
    nrow = -(-(len(coords) + 2) // ncol)
    fig = Figure(figsize=(4.4 * ncol, 3.6 * nrow + 0.6))
    FigureCanvasAgg(fig)
    axs = fig.subplots(nrow, ncol).ravel()

    ax = axs[0]
    m = orig.y_rows <= top
    ax.plot(orig.x_le_rows[m], orig.y_rows[m], "k--", lw=1, label="original LE")
    ax.plot(orig.x_te_rows[m], orig.y_rows[m], "k-", lw=1, label="TE line (kept)")
    cy, cv = control_points(d, orig, "LE", te_radius_mm)
    for s in sections[::every] + [sections[-1]]:
        ax.plot([s.x_le, s.x_te], [s.y, s.y], color="0.75", lw=0.6)
    ax.plot(val["LE"], yy, "b-", lw=2, label=f"new LE ({said('LE')})")
    ax.plot(cv, cy, "o--", color="tab:blue", ms=5, lw=0.8, mfc="w", label="LE control points")
    ax.axhline(H, color="r", ls=":", lw=1, label=f"H = {H:g} mm")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y, height (mm)")
    ax.set_title("planform (sections every %d shown)" % every, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=6, loc="upper right")

    for ax, c in zip(axs[1:], coords):
        cy, cv = control_points(d, orig, c, te_radius_mm)
        yd, vd = orig.data[c]
        yd, vd = yd[yd <= top], vd[yd <= top]
        ax.plot(vd, yd, "k.", ms=3, label="section fits (original)")
        ax.plot(orig.value(c, yz), yz, "k-", lw=1, label=f"original (order {orig.curves[c].order})")
        if c in bands:
            ax.plot(bands[c][0], yy, ":", color="tab:green", lw=1, label="low law")
            ax.plot(bands[c][1], yy, ":", color="tab:red", lw=1, label="high law")
        ax.plot(val[c], yy, "b-", lw=2, label=f"design ({said(c)})")
        if len(cy):
            ax.plot(cv, cy, "o--", color="tab:blue", ms=5, lw=0.8, mfc="w", label="control points")
            ax.plot(cv[0], cy[0], "rs", ms=6, label="pinned at H")
        ax.axhline(H, color="r", ls=":", lw=1)
        shown = np.concatenate((vd, cv, val[c], orig.value(c, yz)))
        shown = shown[np.isfinite(shown)]
        lo, hi = shown.min(), shown.max()
        pad = 0.08 * (hi - lo) if hi - lo > 1e-6 * max(abs(hi), 1e-3) else 1e-3 * max(abs(hi), 1e-3)
        ax.set_xlim(lo - pad, hi + pad)                 # a constant curve gets a readable axis too
        ax.set_title(f"{c}: {'x' if c.endswith('x') else 'z'} of P{c[1:-1]} (fraction of the sharp chord)",
                     fontsize=9)
        ax.set_ylabel("y (mm)")
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(4))          # the values' long labels side by side
        ax.legend(fontsize=6)

    ax = axs[len(coords) + 1]
    pick = [sections[0], sections[len(sections) // 2], sections[-1]]
    for s, col in zip(pick, ("tab:blue", "tab:orange", "tab:green")):
        x_le_w, xy_w = s.written()
        ax.plot(x_le_w + xy_w[:, 0], xy_w[:, 1], color=col, lw=1.3, label=f"new, y = {s.y:.2f} mm")
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
