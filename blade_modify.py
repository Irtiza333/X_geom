"""
blade_modify.py

A propeller blade from a parameter file, and its design space: the blade radius,
hub radius and hub height, and each radial distribution as a Bezier curve over r/R
fitted to the file, whose control points (and, for 2 segments, weights) are the
design variables. The geometry is the MSc blade code's: para.py gives the sections
(the XCAD point file and the blade clearance check of x_blade_new.X_blade),
blade_surface_new.BladeSurface, tip_surfaces_new, hub_new and X_CAD_new give the
DRDC five-surface CAD with the hub sector.

Parameter file (.dat; '#' header, np.loadtxt reads the table):

    # XGeom propeller blade parameters
    # diameter_m     1.4
    # blades         5
    # root_r         0.18
    # hub_height_m   0.625
    # section_table  airfoil_data_fixed.csv          (or:  # airfoil  naca2416)
    # r/R  P/D  c/D  t/c  f/c  skew_deg  rake/D
    0.18000000  1.24135133  0.24524880  0.21585422  0.03041617  0.27575388  0.00001628
    ...

    diameter_m     propeller diameter D, m (the blade radius R = D/2)
    blades         number of blades Z (the clearance check and the hub sector)
    root_r         r/R of the blade root, the hub radius over R (default: the
                   first row); rows below it are ignored
    hub_height_m   axial length of the hub in the CAD, m (optional, default
                   HUB_HEIGHT = hub_new.DEFAULT_HUB_HEIGHT); the hub is centred
                   at x = 0
    section_table  the section family: x/c, camber, camber slope and thickness
                   (para.py's airfoil_data_fixed.csv format), relative to the
                   parameter file's folder
    airfoil        instead of a section table: an airfoil by name (airfoils.py:
                   the UIUC database in airfoils/, else the NACA 4- and 5-digit
                   equations), e.g. naca2416, clarky, e387. It gives the shapes
                   of the camber line and the thickness; t/c and f/c below
                   scale them at each radius (a symmetric airfoil: no camber, f/c
                   has no effect)
    r/R            radius over the tip radius, rising to the tip (1)
    P/D            pitch ratio
    c/D            chord over the diameter
    t/c, f/c       maximum thickness and camber over the chord (para.py's
                   MaxThickness and MaxCamber)
    skew_deg       skew angle, deg
    rake/D         rake over the diameter

Between the rows a distribution is the cubic spline through them: the file's
shape, the reference the design is compared with.

Design (BladeDesign): the blade radius R and the hub radius (m), the hub height
(m), and per distribution a curve in the (r/R, value) plane from the root (r/R =
hub radius / R) to the tip. Positions along a curve are fractions of the span, so
a change of either radius stretches the curves over the new span. A curve has
1 or 2 segments:
  1 segment (BladeCurve): a Bezier with control points C0 at the root .. Cm at
    the tip; the inner positions are set by fractions d (C_k covers the fraction
    d_k of the span left between C_(k-1) and the tip, so any d in (0, 1) keeps
    them in order).
  2 segments (TwoSegmentCurve): the form of para_control_bez_updated (x_blade),
    rational cubic Beziers P1 .. P4 and P4 .. P7 with P2 = P3 and P5 = P6 at the
    value of the common point P4, so the curve has zero slope there; P4's
    position u4, the fractions d1 (P2 = P3 from P4 towards the root) and d2
    (P5 = P6 from P4 towards the tip), the weights w23 and w56, the values v1
    (root), v4 (P4) and v7 (tip).
fit_design fits the curves to the file (positions, weights and values by least
squares).

    params = read_params("msc_blade_params.dat")
    design, residuals = fit_design(params)                  # every curve 2 segments
    space = BladeSpace(params, design, free=("pitch", "chord"))
    x0, bounds = space.to_vector(design), space.bounds      # for the optimiser
    d = space.to_design(x)
    problems = design_problems(d, params)
    paths = write_case("outputs/blade", "case1", d, params, space, settings={"sections": 80})

The XCAD points file has para.py's sections (53 points each) at section_stations:
'sections' in all, the last 'tip_sections' from r/R 'tip_band' to the tip closer
together (defaults 56, 14, 0.93). The clearance check keeps x_blade_new's
stations (r_stations).
"""

from __future__ import annotations

import bisect
import copy
import dataclasses
import datetime
import json
import os
import shutil
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from hub_new import DEFAULT_HUB_HEIGHT
from rudder_modify import Bezier, fit_values, fractions_from_heights, heights_from_fractions, law_basis

CURVES = ("pitch", "chord", "thickness", "camber", "skew", "rake")
COLUMNS = ("r/R", "P/D", "c/D", "t/c", "f/c", "skew_deg", "rake/D")
UNITS = dict(zip(CURVES, ("P/D", "c/D", "t/c", "f/c", "deg", "rake/D")))
DESCRIPTIONS = {
    "pitch": "pitch ratio P/D",
    "chord": "chord over the diameter, c/D",
    "thickness": "maximum thickness over the chord, t/c",
    "camber": "maximum camber over the chord, f/c",
    "skew": "skew angle, deg",
    "rake": "rake over the diameter, rake/D",
}
GLOBALS = ("diameter_m", "blades", "root_r", "hub_height_m", "section_table", "airfoil")
GLOBAL_SLOTS = ("radius", "hub_radius", "hub_height")      # the blade's own design variables, m
DEFAULT_SEGMENTS = 2              # each curve 2 segments (para_control_bez_updated) unless set otherwise
DEFAULT_ORDER = 5                 # 1 segment: 6 control points
D_BOUNDS = (0.01, 0.99)           # 1 segment: the position fractions d, in the fit and the design space
SEG_D_BOUNDS = (0.01, 1.0)        # 2 segments: d1, d2 (1: P2 = P3 at the root, P5 = P6 at the tip)
SEG_U_BOUNDS = (0.02, 0.98)       # 2 segments: P4's position u4, fraction of the span
SEG_W_BOUNDS = (0.02, 5.0)        # 2 segments: the weights w23, w56 in the fit
HUB_HEIGHT = DEFAULT_HUB_HEIGHT    # m, hub_new's: the hub height when the file gives none
CLEARANCE_MM = 25.0               # x_blade_new.CLEARANCE_THRESHOLD, the MSc blade (D = 1.4 m)
SECTIONS = 56                     # sections in the XCAD points file (x_blade_new.R_VALUES: 56 from r/R 0.18)
TIP_BAND = 0.93                   # r/R where the tip band starts (x_blade_new.R_TIP_BAND)
TIP_SECTIONS = 14                 # sections in the tip band, closer together towards the tip (x_blade_new)
N_SAMPLES = 2001                  # samples of a curve behind its property function
ROW_TOL = 1e-8                    # r/R: a table row this close to root_r is the root's (the table has 8 decimals)
_PROPS = {}                       # property functions by the curve's data


# --------------------------------------------------------------------------
# The parameter file
# --------------------------------------------------------------------------

@dataclass
class BladeParams:
    """A blade parameter file: the globals and the radial table."""
    diameter: float               # m
    blades: int
    root_r: float
    section_table: str            # as written in the file ("" with an airfoil)
    r: np.ndarray                 # the table's r/R, rising
    values: dict                  # curve name -> column
    path: str = ""                # the file it was read from ("" when made here)
    hub_height: float = None      # m; None: not in the file (HUB_HEIGHT)
    airfoil: str = None           # an airfoil by name instead of a section table

    def rows(self):
        """The table rows from the root to the tip; a root between two rows gets
        a row of its own, from the splines through the whole table (a row within
        ROW_TOL of the root is the root's)."""
        keep = self.r > self.root_r + ROW_TOL
        r = np.concatenate(([self.root_r], self.r[keep]))
        i = int(np.argmin(np.abs(self.r - self.root_r)))
        out = {}
        for c in CURVES:
            at_root = self.values[c][i]
            if abs(self.r[i] - self.root_r) > ROW_TOL:
                at_root = float(CubicSpline(self.r, self.values[c])(self.root_r))
            out[c] = np.concatenate(([at_root], self.values[c][keep]))
        return r, out

    def section_path(self):
        """The section table: the airfoil's (airfoils.section_table), else the
        file's, relative to the parameter file's folder, else as written."""
        if self.airfoil:
            import airfoils
            return airfoils.section_table(self.airfoil)
        p = self.section_table
        if not os.path.isabs(p) and self.path:
            q = os.path.join(os.path.dirname(os.path.abspath(self.path)), p)
            if os.path.exists(q):
                return q
        return p

    def spline(self, name):
        """The file's distribution between its rows (cubic spline; linear beyond the ends)."""
        r, v = self.rows()
        return _extended(CubicSpline(r, v[name]), r[0], r[-1])


def _extended(spline, r0, r1):
    """spline(r) inside [r0, r1], continued linearly with its end slopes outside;
    takes and returns any array shape. Single values (BladeSurface and the DRDC
    grids ask for them by the million) go through plain float arithmetic."""
    f0, f1 = float(spline(r0)), float(spline(r1))
    s0, s1 = float(spline(r0, 1)), float(spline(r1, 1))
    xs = spline.x.tolist()
    c0, c1, c2, c3 = (row.tolist() for row in spline.c)
    last = len(xs) - 2

    def one(r):
        if r < r0:
            return f0 + s0 * (r - r0)
        if r > r1:
            return f1 + s1 * (r - r1)
        i = min(max(bisect.bisect_right(xs, r) - 1, 0), last)
        t = r - xs[i]
        return ((c0[i] * t + c1[i]) * t + c2[i]) * t + c3[i]

    def fn(r):
        a = np.asarray(r, dtype=float)
        if a.size == 1:
            v = one(float(a.reshape(-1)[0]))
            return np.full(a.shape, v) if a.ndim else np.float64(v)
        out = np.asarray(spline(np.clip(a, r0, r1)), dtype=float)
        out = np.where(a < r0, f0 + s0 * (a - r0), out)
        return np.where(a > r1, f1 + s1 * (a - r1), out)
    return fn


def read_params(path):
    """The BladeParams of a parameter file (see the module docstring)."""
    head = {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith("#"):
                continue
            parts = s[1:].split(None, 1)
            if len(parts) == 2 and parts[0] in GLOBALS:
                head[parts[0]] = parts[1].strip()
    missing = [k for k in ("diameter_m", "blades") if k not in head]
    if ("section_table" in head) == ("airfoil" in head):
        missing.append("section_table or airfoil (one of them)")
    if missing:
        raise ValueError(f"{path}: the header misses {', '.join(missing)}")
    tab = np.atleast_2d(np.loadtxt(path, comments="#"))
    if tab.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: the table needs {len(COLUMNS)} columns ({' '.join(COLUMNS)}), "
                         f"it has {tab.shape[1]}")
    r = tab[:, 0]
    if len(r) < 4 or np.any(np.diff(r) <= 0.0):
        raise ValueError(f"{path}: r/R must rise from row to row, at least 4 rows")
    root = float(head.get("root_r", r[0]))
    if not r[0] - ROW_TOL <= root < r[-1] or np.sum(r > root + ROW_TOL) < 3:
        raise ValueError(f"{path}: root_r {root:g} must lie in the table, with 4 rows from it to the tip")
    hub_h = float(head["hub_height_m"]) if "hub_height_m" in head else None
    params = BladeParams(float(head["diameter_m"]), int(head["blades"]), root, head.get("section_table", ""), r,
                         {c: tab[:, i + 1] for i, c in enumerate(CURVES)}, os.path.abspath(path), hub_h,
                         head.get("airfoil"))
    if params.airfoil:
        try:
            params.section_path()                    # the airfoil exists; its table is made
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None
    return params


def write_params(path, params, notes=()):
    """Write a parameter file (read_params reads it back). A section table
    given by an absolute path is written relative to the new file's folder."""
    table = params.section_table
    if os.path.isabs(table) and os.path.exists(table):
        try:
            table = os.path.relpath(table, os.path.dirname(os.path.abspath(path))).replace(os.sep, "/")
        except ValueError:                      # another drive
            pass
    lines = ["# XGeom propeller blade parameters"]
    lines += [f"# {n}" for n in notes]
    lines += [f"# diameter_m     {params.diameter:.10g}",
              f"# blades         {int(params.blades)}",
              f"# root_r         {params.root_r:.10g}"]
    if params.hub_height is not None:
        lines.append(f"# hub_height_m   {params.hub_height:.10g}")
    lines += [f"# airfoil        {params.airfoil}" if params.airfoil else f"# section_table  {table}",
              "# " + "  ".join(COLUMNS)]
    for i, r in enumerate(params.r):
        row = [r] + [params.values[c][i] for c in CURVES]
        lines.append("  ".join(f"{_clean(v):.8f}" for v in row))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _clean(v):
    """No negative zero at the written precision."""
    v = float(v)
    return 0.0 if abs(v) < 5e-9 else v


def cosine_stations(r0, r1, n=61):
    """n stations from r0 to r1, clustered at both ends (where propeller
    distributions change fastest)."""
    return r0 + (r1 - r0) * 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, int(n))))


def msc_params(n=61):
    """The MSc blade as the pipeline builds it (the polynomials of x_blade_new.py,
    D = 1.4 m, 5 blades, the root at R_VALUES[0], hub_new's hub height)
    tabulated at n cosine-clustered stations: the cubic splines through them
    follow the polynomials to 6e-6 of each distribution's range."""
    import x_blade_new as XB
    root = float(XB.R_VALUES[0])
    r = cosine_stations(root, 1.0, n)
    fns = {"pitch": XB.BASE_PITCH, "chord": XB.BASE_CHORD, "thickness": XB.BASE_MAX_THICKNESS,
           "camber": XB.BASE_MAX_CAMBER, "skew": XB.BASE_SKEW_ANGLE, "rake": XB.BASE_RAKE}
    return BladeParams(1.4, 5, root, "airfoil_data_fixed.csv", r, {c: np.asarray(fns[c](r)) for c in CURVES},
                       hub_height=HUB_HEIGHT)


# --------------------------------------------------------------------------
# Curves and designs
# --------------------------------------------------------------------------

def positions_from_fractions(d):
    """Control positions u (fractions of the span from the root), root first:
    u_0 = 0, u_k = 1 - (1 - u_(k-1)) (1 - d_k), u_m = 1."""
    return 1.0 - heights_from_fractions(d)


def fractions_from_positions(u):
    return fractions_from_heights(1.0 - np.asarray(u, dtype=float))


@dataclass
class BladeCurve:
    """1 segment: a Bezier with control points C0 (root) .. Cm (tip) at
    positions u (fractions of the span, 0 at the root .. 1 at the tip) with
    values v, in the distribution's units. Its design variables (keys): the
    fractions d1 .. d(m-1) that place the inner points, the values v0 .. vm."""
    u: np.ndarray
    v: np.ndarray
    segments = 1

    @property
    def order(self):
        return len(self.u) - 1

    @staticmethod
    def keys_of(order):
        return [f"d{k}" for k in range(1, order)] + [f"v{k}" for k in range(order + 1)]

    def keys(self):
        return self.keys_of(self.order)

    def get(self, key):
        k = int(key[1:])
        return float(fractions_from_positions(self.u)[k - 1]) if key[0] == "d" else float(self.v[k])

    @classmethod
    def from_keys(cls, vals, order):
        d = np.clip([vals[f"d{k}"] for k in range(1, order)], 0.0, 1.0)
        return cls(positions_from_fractions(d), np.array([vals[f"v{k}"] for k in range(order + 1)], dtype=float))

    def key(self):
        return (1, np.asarray(self.u, dtype=float).tobytes(), np.asarray(self.v, dtype=float).tobytes())

    def ctrl(self, r0, r1):
        """The control points (r/R, value), root first."""
        return np.column_stack((r0 + np.asarray(self.u, dtype=float) * (r1 - r0), self.v))

    def weights(self):
        return np.ones(len(self.u))

    def beziers(self, r0, r1):
        return [Bezier(self.ctrl(r0, r1))]

    def scaled(self, f):
        """The curve with its values times f."""
        return BladeCurve(np.array(self.u, dtype=float), np.asarray(self.v, dtype=float) * f)


@dataclass
class TwoSegmentCurve:
    """2 segments, the form of para_control_bez_updated (x_blade): rational
    cubic Beziers P1 P2 P3 P4 and P4 P5 P6 P7 with

        P1 = (r0, v1)          the root
        P2 = P3 = (r2, v4)     r2 = r4 - (r4 - r0) d1,   weight w23
        P4 = (r4, v4)          r4 = r0 + u4 (r1 - r0),   the common point
        P5 = P6 = (r5, v4)     r5 = r4 + (r1 - r4) d2,   weight w56
        P7 = (r1, v7)          the tip

    (P1, P4 and P7 of weight 1): zero slope at P4. Its design variables (keys)
    from the root to the tip: v1, d1, w23, u4, v4, d2, w56, v7."""
    v1: float
    d1: float
    w23: float
    u4: float
    v4: float
    d2: float
    w56: float
    v7: float
    segments = 2
    order = 3
    KEYS = ("v1", "d1", "w23", "u4", "v4", "d2", "w56", "v7")

    @classmethod
    def keys_of(cls, order=3):
        return list(cls.KEYS)

    def keys(self):
        return list(self.KEYS)

    def get(self, key):
        return float(getattr(self, key))

    @classmethod
    def from_keys(cls, vals, order=3):
        q = {k: float(vals[k]) for k in cls.KEYS}
        q["u4"] = min(max(q["u4"], 1e-3), 1.0 - 1e-3)               # P4 strictly inside the span
        for k in ("d1", "d2"):
            q[k] = min(max(q[k], 0.0), 1.0)
        for k in ("w23", "w56"):
            q[k] = max(q[k], 1e-6)
        return cls(**q)

    def key(self):
        return (2,) + tuple(float(getattr(self, k)) for k in self.KEYS)

    @property
    def v(self):
        """The control values v1, v4, v7."""
        return np.array([self.v1, self.v4, self.v7])

    def radii(self, r0, r1):
        """r/R of P2 = P3, P4 and P5 = P6."""
        r4 = r0 + self.u4 * (r1 - r0)
        return r4 - (r4 - r0) * self.d1, r4, r4 + (r1 - r4) * self.d2

    def ctrl(self, r0, r1):
        """P1 .. P7 (r/R, value)."""
        r2, r4, r5 = self.radii(r0, r1)
        return np.array([[r0, self.v1], [r2, self.v4], [r2, self.v4], [r4, self.v4],
                         [r5, self.v4], [r5, self.v4], [r1, self.v7]], dtype=float)

    def weights(self):
        return np.array([1.0, self.w23, self.w23, 1.0, self.w56, self.w56, 1.0])

    def beziers(self, r0, r1):
        p, w = self.ctrl(r0, r1), self.weights()
        return [Bezier(p[:4], w[:4]), Bezier(p[3:], w[3:])]

    def scaled(self, f):
        return dataclasses.replace(self, v1=self.v1 * f, v4=self.v4 * f, v7=self.v7 * f)


CURVE_TYPES = {1: BladeCurve, 2: TwoSegmentCurve}


def curve_at(curve, r0, r1, r):
    """A curve's values at the r/R in r (clipped to [r0, r1]), any array shape:
    each segment is solved for the parameter where it passes r."""
    r = np.clip(np.asarray(r, dtype=float), r0, r1)
    flat = r.ravel()
    out = np.empty(flat.size)
    left = np.ones(flat.size, dtype=bool)
    segs = curve.beziers(r0, r1)
    for i, b in enumerate(segs):
        m = left if i == len(segs) - 1 else left & (flat <= b.ctrl[-1, 0])
        if m.any():
            out[m] = b.at(np.clip(flat[m], b.ctrl[0, 0], b.ctrl[-1, 0]))[:, 1]
        left &= ~m
    return out.reshape(r.shape)


@dataclass
class BladeDesign:
    """A blade: its radius R and hub radius (m), the hub height (m, the CAD's
    hub), and one curve per distribution over r/R from the root (r0 = hub
    radius / R) to r1 (the tip). Positions along the curves are fractions of the
    span, so a change of either radius stretches them over the new span."""
    radius: float
    hub_radius: float
    hub_height: float
    r1: float
    curves: dict

    @property
    def r0(self):
        return self.hub_radius / self.radius

    @property
    def diameter(self):
        return 2.0 * self.radius

    def copy(self):
        return copy.deepcopy(self)

    def ctrl(self, name):
        """The control points (r/R, value), root first."""
        return self.curves[name].ctrl(self.r0, self.r1)

    def at(self, name, r):
        """Curve `name` at the r/R in r (clipped to the blade)."""
        return curve_at(self.curves[name], self.r0, self.r1, r)

    def prop(self, name):
        """The distribution as a function of r/R (any array shape), as para.py and
        BladeSurface take them: a cubic spline through the curve at N_SAMPLES
        evenly spaced r/R, continued linearly beyond the root and the tip."""
        key = (self.r0, self.r1, self.curves[name].key())
        fn = _PROPS.get(key)
        if fn is None:
            rr = np.linspace(self.r0, self.r1, N_SAMPLES)
            fn = _extended(CubicSpline(rr, self.at(name, rr)), self.r0, self.r1)
            if len(_PROPS) > 200:
                _PROPS.clear()
            _PROPS[key] = fn
        return fn

    def props(self):
        """MaxCamber, Pitch, ChordLength, MaxThickness, SkewAngle, Rake: the
        property functions of x_blade_new / para.py, in that order."""
        return tuple(self.prop(c) for c in ("camber", "pitch", "chord", "thickness", "skew", "rake"))

    def values(self, r):
        """{curve: values at the r/R in r}."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        return {c: self.at(c, r) for c in CURVES}


def _fit_positions(r, v, r0, r1, order, u_starts):
    """Least-squares control values for each start's positions, then the
    positions themselves by least squares (bounded fractions d); the best."""
    def solve(d):
        pos = r0 + positions_from_fractions(d) * (r1 - r0)
        vals, res = fit_values(pos, r, v)
        return vals, res
    best = None
    for u0 in u_starts:
        d0 = np.clip(fractions_from_positions(u0), D_BOUNDS[0], D_BOUNDS[1])
        if order > 1:
            sol = least_squares(lambda d: solve(d)[1], d0, bounds=D_BOUNDS, xtol=1e-12, ftol=1e-12,
                                gtol=1e-12, max_nfev=200 * order)
            d0 = np.clip(sol.x, D_BOUNDS[0] + 1e-9, D_BOUNDS[1] - 1e-9)    # u -> d round-off stays inside
        vals, res = solve(d0)
        sse = float(np.sum(res ** 2))
        if best is None or sse < best[0]:
            best = (sse, positions_from_fractions(d0), vals, res)
    return best[1], best[2], best[3]


def fit_curve(r, v, r0, r1, order):
    """A 1-segment curve of the given order through the data (r, v): positions
    and values by least squares, from four starts (evenly spaced, clustered at
    both ends, at the tip, at the root). Returns (curve, residuals at the data)."""
    order = int(order)
    if order < 1:
        raise ValueError("a curve needs at least 2 control points")
    k = np.arange(order + 1) / order
    starts = [k, 0.5 * (1.0 - np.cos(np.pi * k)), np.sin(0.5 * np.pi * k), 1.0 - np.cos(0.5 * np.pi * k)]
    u, vals, res = _fit_positions(np.asarray(r, dtype=float), np.asarray(v, dtype=float), r0, r1, order, starts)
    return BladeCurve(u, vals), res


def _two_segment_basis(r, r0, r1, u4, d1, d2, w23, w56):
    """The matrix A with v(r_i) = A[i] @ (v1, v4, v7) for the 2-segment form."""
    r2, r4, r5 = TwoSegmentCurve(0.0, d1, w23, u4, 0.0, d2, w56, 0.0).radii(r0, r1)
    a = np.zeros((len(r), 3))
    m = r <= r4
    if m.any():
        b = law_basis([r0, r2, r2, r4], r[m], [1.0, w23, w23, 1.0])
        a[m, 0], a[m, 1] = b[:, 0], b[:, 1:].sum(axis=1)
    if (~m).any():
        b = law_basis([r4, r5, r5, r1], r[~m], [1.0, w56, w56, 1.0])
        a[~m, 1], a[~m, 2] = b[:, :3].sum(axis=1), b[:, 3]
    return a


def fit_two_segment(r, v, r0, r1):
    """A 2-segment curve through the data (r, v): u4, d1, d2 and the weights by
    bounded least squares from four starts (P4 at the data's largest departure
    from the line through its ends or at mid-span; weights 0.4 or 1), the values
    v1, v4, v7 by linear least squares at every step. Returns (curve, residuals
    at the data)."""
    r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
    lo = [SEG_U_BOUNDS[0], SEG_D_BOUNDS[0], SEG_D_BOUNDS[0], SEG_W_BOUNDS[0], SEG_W_BOUNDS[0]]
    hi = [SEG_U_BOUNDS[1], SEG_D_BOUNDS[1], SEG_D_BOUNDS[1], SEG_W_BOUNDS[1], SEG_W_BOUNDS[1]]

    def solve(p):
        a = _two_segment_basis(r, r0, r1, *p)
        vals = np.linalg.lstsq(a, v, rcond=None)[0]
        return vals, a @ vals - v
    i = int(np.argmax(np.abs(v - np.interp(r, [r[0], r[-1]], [v[0], v[-1]]))))
    u_ext = float(np.clip((r[i] - r0) / (r1 - r0), 0.05, 0.95))
    best = None
    for u0 in sorted({round(u_ext, 3), 0.5}):
        for w0 in (0.4, 1.0):
            sol = least_squares(lambda p: solve(p)[1], [u0, 0.5, 0.5, w0, w0], bounds=(lo, hi),
                                xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=300)
            vals, res = solve(sol.x)
            sse = float(res @ res)
            if best is None or sse < best[0]:
                best = (sse, sol.x, vals, res)
    _, (u4, d1, d2, w23, w56), (v1, v4, v7), res = best
    return TwoSegmentCurve(float(v1), float(d1), float(w23), float(u4), float(v4), float(d2), float(w56),
                           float(v7)), res


def fit_form(r, v, r0, r1, segments, order=DEFAULT_ORDER):
    """fit_two_segment (segments 2) or fit_curve of the order (segments 1)."""
    if int(segments) == 2:
        return fit_two_segment(r, v, r0, r1)
    if int(segments) != 1:
        raise ValueError("a curve has 1 or 2 segments")
    return fit_curve(r, v, r0, r1, order)


def fit_design(params, orders=None, segments=None):
    """The file's blade as a design: radius D/2, hub radius root r/R x R, the
    hub height of the file (default HUB_HEIGHT), and each distribution's rows
    from the root to the tip fitted by a curve of segments[name] segments
    (default DEFAULT_SEGMENTS; 1 segment of orders[name], default DEFAULT_ORDER).
    Returns (design, {curve: residuals at the rows})."""
    orders, segments = dict(orders or {}), dict(segments or {})
    r, vals = params.rows()
    curves, res = {}, {}
    for c in CURVES:
        curves[c], res[c] = fit_form(r, vals[c], r[0], r[-1], segments.get(c, DEFAULT_SEGMENTS),
                                     orders.get(c, DEFAULT_ORDER))
    radius = 0.5 * params.diameter
    hub_h = HUB_HEIGHT if params.hub_height is None else params.hub_height
    return BladeDesign(radius, float(r[0]) * radius, float(hub_h), float(r[-1]), curves), res


def refit_curve(design, name, order=None, segments=None, n_samples=None):
    """The design with curve `name` fitted again to its present shape: in
    another form (segments 1 or 2) or, 1 segment, at another order (default:
    the curve's own, DEFAULT_ORDER coming from 2 segments)."""
    cur = design.curves[name]
    seg = cur.segments if segments is None else int(segments)
    if order is None:
        order = cur.order if cur.segments == 1 else DEFAULT_ORDER
    n = n_samples or (121 if seg == 2 else 401)
    rr = np.linspace(design.r0, design.r1, n)
    out = design.copy()
    out.curves[name], _ = fit_form(rr, design.at(name, rr), design.r0, design.r1, seg, order)
    return out


def refit_to_file(params, design, name, segments, order=DEFAULT_ORDER):
    """The design with curve `name` fitted again to the parameter file's rows,
    as segments (1: of the order) segments; the file's span mapped onto the
    design's (fractions of the span, as a change of the radii does)."""
    r, vals = params.rows()
    rd = design.r0 + (r - r[0]) / (r[-1] - r[0]) * (design.r1 - design.r0)
    out = design.copy()
    out.curves[name], _ = fit_form(rd, vals[name], design.r0, design.r1, segments, order)
    return out


def fit_report(params, design):
    """{curve: (largest |design - file| at the rows, the same over the file's range)}."""
    r, vals = params.rows()
    got = design.values(r)
    out = {}
    for c in CURVES:
        e = float(np.abs(got[c] - vals[c]).max())
        out[c] = (e, e / max(float(np.ptp(vals[c])), 1e-300))
    return out


# --------------------------------------------------------------------------
# The design space
# --------------------------------------------------------------------------

def default_bounds(params, design):
    """Bounds by group: "d" (1-segment position fractions) D_BOUNDS, "seg.d"
    (2-segment d1, d2) SEG_D_BOUNDS, "seg.u" (P4's position) SEG_U_BOUNDS,
    "seg.w" (the weights) from SEG_W_BOUNDS[0] to 2, or 1.5 times the largest
    fitted weight; each curve's values from the lowest to the highest of its
    rows and control values, half that range again on either side (rounded
    outward), for chord, thickness and pitch not below 0 unless a fitted control
    value is; the blade radius 75 % to 125 %, the hub radius 50 % to 150 % and
    the hub height 50 % to 200 % of the design's (to the mm)."""
    b = {"d": D_BOUNDS, "seg.d": SEG_D_BOUNDS, "seg.u": SEG_U_BOUNDS}
    w = [x for c in CURVES if design.curves[c].segments == 2 for x in (design.curves[c].w23, design.curves[c].w56)]
    b["seg.w"] = (SEG_W_BOUNDS[0], float(max([2.0] + [np.ceil(15.0 * x) / 10.0 for x in w])))
    _, vals = params.rows()
    for c in CURVES:
        allv = np.concatenate((vals[c], design.curves[c].v))
        lo, hi = float(allv.min()), float(allv.max())
        span = max(hi - lo, 1e-3 * max(abs(lo), abs(hi), 1e-6))
        lo, hi = lo - 0.5 * span, hi + 0.5 * span
        places = int(1 - np.floor(np.log10(span)))
        lo, hi = round(np.floor(lo * 10.0 ** places) / 10.0 ** places, places), \
            round(np.ceil(hi * 10.0 ** places) / 10.0 ** places, places)
        if c in ("chord", "thickness", "pitch"):                # positive, unless the fit put a control value below 0
            vmin = float(design.curves[c].v.min())
            lo = min(max(lo, 0.0), round(np.floor(vmin * 10.0 ** places) / 10.0 ** places, places))
        b[c] = (float(lo), float(hi))
    for g, (f0, f1) in (("radius", (0.75, 1.25)), ("hub_radius", (0.5, 1.5)), ("hub_height", (0.5, 2.0))):
        x = float(getattr(design, g))
        b[g] = (float(np.floor(f0 * x * 1000.0) / 1000.0), float(np.ceil(f1 * x * 1000.0) / 1000.0))
    return b


class BladeSpace:
    """A flat design vector for an optimiser, and the BladeDesign it stands for.

    The slots: the globals radius, hub_radius, hub_height (m), then curve by
    curve (CURVES order) its own: 1 segment d1 .. d(m-1) (the fractions that
    place the inner control points), v0 (root) .. vm (tip); 2 segments v1, d1,
    w23, u4, v4, d2, w56, v7. Names "radius", "pitch.d1", "pitch.v0", ... The
    curves' forms (segments, order) are the design's.

    bounds   overrides of default_bounds by group ("d", "seg.d", "seg.u",
             "seg.w"), curve ("pitch": its values; "pitch.d", "pitch.u",
             "pitch.w": its fractions, P4 position, weights) or slot
             ("pitch.v3", "radius"); a slot beats its curve, a curve its group
    fixed    {slot: value} of the held slots (not in the vector); with free
             given instead (curve names and globals), every other slot is held
             at the design's value
    """

    def __init__(self, params, design, bounds=None, fixed=None, free=None):
        self.params = params
        self.r1 = float(design.r1)
        self.forms = {c: (design.curves[c].segments, design.curves[c].order) for c in CURVES}
        b = {**default_bounds(params, design), **(bounds or {})}
        self._slots = [(None, g) for g in GLOBAL_SLOTS]
        for c in CURVES:
            self._slots += [(c, k) for k in design.curves[c].keys()]
        self.all_names = [k if c is None else f"{c}.{k}" for c, k in self._slots]
        self.all_bounds = np.array([self._bound(b, n, sl) for n, sl in zip(self.all_names, self._slots)], dtype=float)
        if fixed is None and free is not None:
            vals = self.values(design)
            fixed = {n: vals[n] for n, (c, k) in zip(self.all_names, self._slots) if (c or k) not in free}
        self.fixed = {str(n): float(v) for n, v in (fixed or {}).items()}
        unknown = sorted(set(self.fixed) - set(self.all_names))
        if unknown:
            raise ValueError(f"fixed slots that this space does not have: {unknown}")
        self._free = np.array([i for i, n in enumerate(self.all_names) if n not in self.fixed], dtype=int)
        self.names = [self.all_names[i] for i in self._free]
        self.bounds = self.all_bounds[self._free].reshape(-1, 2)

    def _bound(self, b, name, slot):
        c, k = slot
        if name in b:
            return b[name]
        if c is None:
            return b[k]
        if k[0] == "v":
            return b[c]
        if self.forms[c][0] == 1:
            return b.get(f"{c}.d", b["d"])
        return b.get(f"{c}.{k[0]}", b[f"seg.{k[0]}"])

    def __len__(self):
        return len(self.names)

    def full_vector(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self.names),):
            raise ValueError(f"the design vector has {len(self.names)} entries")
        full = np.array([self.fixed.get(n, np.nan) for n in self.all_names])
        full[self._free] = x
        return full

    def to_design(self, x):
        return self._design_of(self.full_vector(x))

    def design_of(self, values):
        """The BladeDesign of {slot name: value} for every slot."""
        return self._design_of(np.array([float(values[n]) for n in self.all_names]))

    def _design_of(self, full):
        vals = dict(zip(self.all_names, (float(x) for x in full)))
        curves = {}
        for c in CURVES:
            seg, order = self.forms[c]
            cls = CURVE_TYPES[seg]
            curves[c] = cls.from_keys({k: vals[f"{c}.{k}"] for k in cls.keys_of(order)}, order)
        return BladeDesign(vals["radius"], vals["hub_radius"], vals["hub_height"], self.r1, curves)

    def values(self, design):
        """{slot name: value} of a design, every slot."""
        return {name: float(getattr(design, k)) if c is None else design.curves[c].get(k)
                for (c, k), name in zip(self._slots, self.all_names)}

    def to_vector(self, design):
        vals = self.values(design)
        return np.array([vals[n] for n in self.names])

    def to_dict(self, design):
        vals = self.values(design)
        return {"forms": {c: f[0] for c, f in self.forms.items()}, "orders": {c: f[1] for c, f in self.forms.items()},
                "r1": self.r1,
                "slots": [{"name": n, "value": vals[n], "lo": float(lo), "hi": float(hi), "free": n not in self.fixed}
                          for n, (lo, hi) in zip(self.all_names, self.all_bounds)],
                "vector": {"names": self.names, "values": [vals[n] for n in self.names],
                           "bounds": self.bounds.tolist()}}

    @classmethod
    def from_dict(cls, params, rec):
        """(space, design) from to_dict's output. A record from before the
        globals and the 2-segment form (no "forms", "r0" for the root): every
        curve 1 segment, the globals held at the file's (root r/R the record's)."""
        slots = rec["slots"]
        forms = rec.get("forms", {})
        radius = 0.5 * params.diameter
        hub_h = HUB_HEIGHT if params.hub_height is None else params.hub_height

        def template(c):
            if int(forms.get(c, 1)) == 2:
                return TwoSegmentCurve(0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.0)
            n = int(rec["orders"][c])
            return BladeCurve(np.linspace(0.0, 1.0, n + 1), np.zeros(n + 1))
        tmpl = BladeDesign(radius, float(rec.get("r0", params.root_r)) * radius, float(hub_h),
                           float(rec.get("r1", 1.0)), {c: template(c) for c in CURVES})
        vals = {q["name"]: q["value"] for q in slots}
        bounds = {q["name"]: (q["lo"], q["hi"]) for q in slots}
        fixed = {q["name"]: q["value"] for q in slots if not q["free"]}
        for g in GLOBAL_SLOTS:
            if g not in vals:
                vals[g] = fixed[g] = float(getattr(tmpl, g))
        space = cls(params, tmpl, bounds, fixed)
        return space, space.design_of(vals)


def space_record(space, design, params, settings=None):
    """The set-up as plain data: the parameter file (path and contents' globals),
    the settings and the space with the design in it."""
    return {"geometry": "blade", "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "params_file": params.path,
            "globals": {"diameter_m": params.diameter, "blades": params.blades, "root_r": params.root_r,
                        "hub_height_m": params.hub_height, "section_table": params.section_table,
                        "airfoil": params.airfoil},
            "settings": dict(settings or {}), "space": space.to_dict(design)}


def load_blade_space(path):
    """(space, design, record) of a set-up the design tool saved
    (<case>_design_space.json); the parameter file is read again from the path
    the record names:

        space, design, rec = load_blade_space("outputs/blade/gui_design_space.json")
        x0, bounds = space.to_vector(design), space.bounds
        d = space.to_design(x)
        problems = design_problems(d, space.params)
    """
    with open(path) as fh:
        rec = json.load(fh)
    params = read_params(rec["params_file"])
    space, design = BladeSpace.from_dict(params, rec["space"])
    return space, design, rec


# --------------------------------------------------------------------------
# Geometry (the MSc code)
# --------------------------------------------------------------------------

def r_stations(root_r):
    """The radial stations of x_blade_new.R_VALUES from another root: steps of
    0.018 up to the tip band, then 14 stations clustered towards r/R = 1. The
    clearance check uses them."""
    import x_blade_new as XB
    band = XB.R_TIP_BAND
    return np.concatenate([np.arange(float(root_r), band - 1e-9, 0.018),
                           band + (1.0 - band) * np.sin(np.linspace(0.0, np.pi / 2, 14))])


def section_stations(root_r, n=SECTIONS, tip_band=TIP_BAND, tip_n=TIP_SECTIONS):
    """The n radial stations (r/R) of the XCAD points file: the last tip_n from
    tip_band to the tip, closer together towards it (sine spacing, as
    x_blade_new.R_VALUES), the others evenly spaced from the root up to the band.
    The defaults are x_blade_new's layout with even steps below the band (its
    steps are 0.018 with a last one of 0.004)."""
    n, tip_n, root_r, tip_band = int(n), int(tip_n), float(root_r), float(tip_band)
    if not root_r < tip_band < 1.0:
        raise ValueError(f"the tip band must start between the root (r/R {root_r:.3f}) and the tip")
    if tip_n < 2 or n - tip_n < 1:
        raise ValueError("sections: at least 2 in the tip band and the root section below it")
    inner = root_r + (tip_band - root_r) * np.arange(n - tip_n) / (n - tip_n)
    return np.concatenate([inner, tip_band + (1.0 - tip_band) * np.sin(np.linspace(0.0, np.pi / 2, tip_n))])


def settings_stations(root_r, settings=None):
    """section_stations from the root r/R and a settings dict ('sections',
    'tip_band', 'tip_sections'; missing ones: the defaults)."""
    st = settings or {}
    return section_stations(root_r, st.get("sections", SECTIONS), st.get("tip_band", TIP_BAND),
                            st.get("tip_sections", TIP_SECTIONS))


def blade_points(design, params, stations=None):
    """para.py's blade points (m, the MSc frame), 53 per section, at the
    stations (default r_stations(root), x_blade_new's: the clearance check).
    Returns (points, stations)."""
    from para import para
    st = r_stations(design.r0) if stations is None else np.asarray(stations, dtype=float)
    pts = para(*design.props(), st, 0, write_dat=False, d=design.diameter, airfoil_path=params.section_path())
    return pts, st


def clearance(points, blades):
    """The smallest distance between the blade and its neighbour (the blade
    turned by 360/Z deg about x), m: the check of x_blade_new.X_blade."""
    from rot_axis import rot_axis
    turned = points @ rot_axis(np.array([1.0, 0.0, 0.0]), 2.0 * np.pi / int(blades))
    return float(min(cKDTree(turned).query(points, k=1)[0].min(), cKDTree(points).query(turned, k=1)[0].min()))


def blade_surface(design, params):
    """blade_surface_new.BladeSurface of the design (the DRDC CAD path)."""
    from blade_surface_new import BladeSurface
    return BladeSurface(*design.props(), d=design.diameter, r_root=design.r0, airfoil_path=params.section_path())


def design_problems(design, params, clearance_mm=CLEARANCE_MM, points=None, gap_mm=None, hub=True):
    """What makes the design unbuildable: a hub radius not between 0 and 90 % of
    the blade radius; chord, thickness or pitch not positive inside the blade;
    blades closer than clearance_mm (the MSc check); with the hub, a hub too
    short for the blade root (hub_new: the hub, centred at x = 0, must reach 2 %
    of the root section's axial extent beyond it). points (blade_points at
    x_blade_new's stations) and gap_mm save computing them again."""
    if not 0.0 < design.r0 < 0.9 * design.r1:
        return [f"hub radius {design.hub_radius:.4g} m: it must lie between 0 and 90 % of the blade radius "
                f"({design.radius:.4g} m)"]
    out = []
    rr = np.linspace(design.r0, design.r0 + 0.999 * (design.r1 - design.r0), 400)
    for c in ("chord", "thickness", "pitch"):
        v = design.prop(c)(rr)
        if np.any(v <= 0.0):
            out.append(f"{c} not positive from r/R {rr[np.argmax(v <= 0.0)]:.3f}")
    if out:
        return out
    pts = blade_points(design, params)[0] if points is None else points
    if gap_mm is None:
        gap_mm = clearance(pts, params.blades) * 1000.0
    if gap_mm < clearance_mm:
        out.append(f"blades {gap_mm:.1f} mm apart, less than {clearance_mm:g} mm (the clearance check)")
    if hub:
        x = pts[:53, 0]                                     # the root section
        a, b = float(x.min()), float(x.max())
        half, margin = 0.5 * design.hub_height, 0.02 * (b - a)
        if not (-half < a - margin and b + margin < half):
            out.append(f"hub height {1000.0 * design.hub_height:.0f} mm does not cover the blade root (x from "
                       f"{1000.0 * a:.0f} to {1000.0 * b:.0f} mm; the hub is centred at x = 0)")
    return out


def write_points(path, points, per_section=53):
    """The blade points as an XCAD point file: '#k' before each section, then
    'x y z' per line, m, eight decimals (para.py's format)."""
    pts = np.asarray(points, dtype=float).reshape(-1, per_section, 3)
    lines = []
    for k, sec in enumerate(pts, start=1):
        lines.append(f"#{k}")
        lines += [f"{_clean(x):.8f} {_clean(y):.8f} {_clean(z):.8f}" for x, y, z in sec]
    with open(path, "w", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def design_params(design, params, n=101):
    """The design as a BladeParams (its curves at n cosine-clustered stations,
    its radius, root and hub height; the airfoil, or the section table by its
    absolute path when it is found)."""
    r = cosine_stations(design.r0, design.r1, n)
    table = params.section_table
    if not params.airfoil:
        table = params.section_path()
        table = os.path.abspath(table) if os.path.exists(table) else params.section_table
    return BladeParams(design.diameter, params.blades, design.r0, table, r, design.values(r),
                       hub_height=design.hub_height, airfoil=params.airfoil)


def design_record(design, params, space=None, problems=(), gap_mm=None, settings=None):
    """The design as plain data: the parameter file it came from, the blade's
    globals, each curve's form, variables and control points (r/R, value) with
    their weights, the fit to the file, the checks."""
    rep = fit_report(params, design)
    rec = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "params_file": params.path,
           "globals": {"radius_m": design.radius, "diameter_m": design.diameter, "hub_radius_m": design.hub_radius,
                       "root_r": design.r0, "hub_height_m": design.hub_height, "blades": params.blades,
                       "section_table": params.section_table, "airfoil": params.airfoil},
           "curves": {c: {"segments": cu.segments, "variables": {k: cu.get(k) for k in cu.keys()},
                          "control_points": design.ctrl(c).tolist(), "weights": cu.weights().tolist(),
                          "max_diff_from_file": rep[c][0]} for c, cu in design.curves.items()},
           "problems": list(problems), "clearance_mm": gap_mm, "settings": dict(settings or {})}
    if space is not None:
        rec["space"] = space.to_dict(design)
    return rec


def write_case(out_dir, case, design, params, space=None, cad=True, settings=None, plot=True):
    """Write the case: <case>_params.dat (the design as a parameter file, which
    read_params and the tool load), <case>_xcad_m.dat (the blade points at the
    settings' sections, settings_stations), <case>_design.json,
    <case>_design_space.json (with a space), <case>_check.png (plot), and with
    cad the DRDC five-surface CAD: <case>.iges (blade and hub sector,
    X_CAD_new.X_CAD) and <case>.step (the same shape). Returns the paths and the
    checks."""
    os.makedirs(out_dir, exist_ok=True)
    st = dict(settings or {})
    pts, stations = blade_points(design, params, settings_stations(design.r0, st))
    check_pts = blade_points(design, params)[0]                                  # x_blade_new's stations
    gap = clearance(check_pts, params.blades) * 1000.0
    problems = design_problems(design, params, st.get("clearance_mm", CLEARANCE_MM), points=check_pts, gap_mm=gap,
                               hub=st.get("hub", True))
    paths = {k: os.path.join(out_dir, f"{case}_{k}") for k in ("params.dat", "xcad_m.dat", "design.json")}
    write_params(paths["params.dat"], design_params(design, params),
                 notes=(f"design {case} from {os.path.basename(params.path) or 'memory'}, "
                        f"{datetime.date.today().isoformat()}",))
    write_points(paths["xcad_m.dat"], pts)
    with open(paths["design.json"], "w") as fh:
        json.dump(design_record(design, params, space, problems, gap, st), fh, indent=1)
    if space is not None:
        paths["space"] = os.path.join(out_dir, f"{case}_design_space.json")
        with open(paths["space"], "w") as fh:
            json.dump(space_record(space, design, params, st), fh, indent=1)
    if plot:
        paths["check"] = plot_case(os.path.join(out_dir, f"{case}_check.png"), case, design, params)
    out = {"paths": paths, "problems": problems, "clearance_mm": gap, "sections": len(stations)}
    if cad:
        if problems:
            raise ValueError(f"{case}: no CAD for a design with problems: {problems[0]}")
        out.update(write_cad(out_dir, case, design, params, st))
        paths.update(out.pop("cad_paths"))
    return out


def write_cad(out_dir, case, design, params, settings=None):
    """The DRDC five-surface blade and the hub sector of the design's hub
    height (tip_surfaces_new, X_CAD_new) as <case>.iges, and the same shape as
    <case>.step. X_CAD names its file from an integer case id
    (pipeline_config.cad_output_paths), so it writes into a folder of its own,
    the file is moved out of it and the folder removed (also when X_CAD fails)."""
    import tempfile
    import time
    from tip_surfaces_new import TipConfig, build_drdc_grids
    from X_CAD_new import X_CAD
    from pipeline_config import cad_output_paths
    st = dict(settings or {})
    t0 = time.time()
    grids = build_drdc_grids(blade_surface(design, params), TipConfig(**st.get("tip_config", {})), verbose=False)
    t1 = time.time()
    tmp = tempfile.mkdtemp(prefix="x_cad_", dir=out_dir)
    try:
        shape = X_CAD(grids, 0, output_dir=tmp, hub=st.get("hub", True), hub_height=design.hub_height,
                      n_blades=params.blades)
        iges = os.path.join(out_dir, f"{case}.iges")
        os.replace(cad_output_paths(0, tmp)["iges"], iges)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    step = os.path.join(out_dir, f"{case}.step")
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    w.Write(step)
    return {"cad_paths": {"iges": iges, "step": step}, "grid_seconds": t1 - t0, "cad_seconds": time.time() - t1}


def plot_case(path, case, design, params):
    """Each distribution against r/R (on the x axis), the design against the file."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(13, 7.5))
    FigureCanvasAgg(fig)
    axs = fig.subplots(2, 3)
    r, vals = params.rows()
    rf = np.linspace(r[0], r[-1], 400)
    rr = np.linspace(design.r0, design.r1, 400)
    for ax, c in zip(axs.flat, CURVES):
        ax.plot(rf, params.spline(c)(rf), "-", color="#9a9994", lw=1, label="file")
        ax.plot(r, vals[c], ".", color="#52514e", ms=3)
        ax.plot(rr, design.prop(c)(rr), "-", color="#2a78d6", lw=1.8, label="design")
        p = design.ctrl(c)
        ax.plot(p[:, 0], p[:, 1], "o--", color="#2a78d6", lw=0.8, ms=4, mfc="white")
        ax.set_xlabel("r/R")
        ax.set_ylabel(f"{c} ({UNITS[c]}), {design.curves[c].segments} segment"
                      + ("s" if design.curves[c].segments == 2 else ""))
        ax.grid(True, color="#e4e3df")
    axs.flat[0].legend(fontsize=8)
    fig.suptitle(f"{case}: distributions, design (blue, control polygon dashed) and the parameter file (grey); "
                 f"R {design.radius:.4g} m, hub {design.hub_radius:.4g} m")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path
