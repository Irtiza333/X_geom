"""
wing_modify.py

A wing or a fin from a parameter file, and its design space: the span, and each spanwise distribution as a
Bezier curve over the span fraction eta, fitted to the file, whose control points are the design variables.
The sections are a named airfoil (airfoils.py) scaled to t/c and f/c, with Bezier changes of its thickness and
camber line; lofted through the sections of the XCAD file (xcad_loft.py) they give the solid, its tip flat for
now.

Parameter file (.dat; '#' header, np.loadtxt reads the table):

    # XGeom wing parameters                       (a fin: # XGeom fin parameters; the same format)
    # span_mm     200
    # airfoil     naca2412
    # ref_xc      0.25
    # eta  chord_mm  sweep_mm  rake_mm  pitch_deg  t/c  f/c
    0.00000000  120.00000000  0.00000000  0.00000000  2.00000000  0.12000000  0.02000000
    ...

    span_mm     the span, from the root (eta 0) to the tip (eta 1), mm
    airfoil     the section by name (airfoils.py: the UIUC database in airfoils/, else the NACA 4- and 5-digit
                equations): the shapes of its thickness and its camber line
    ref_xc      the pitch axis: the point of the chord line this fraction of the chord behind the LE
                (optional, default 0.25)
    camber_xc   for a symmetric airfoil, which has no camber line of its own: f/c scales the NACA 4-digit
                camber line with its maximum this fraction of the chord behind the LE (optional, default 0.4)
    eta         the span fraction, rising from 0 (the root) to 1 (the tip)
    chord_mm    the chord
    sweep_mm    x of the LE, aft positive (the sweep)
    rake_mm     z of the LE, positive towards the side the camber bulges to (the dihedral; negative: anhedral)
    pitch_deg   the section's pitch (twist) about the pitch axis, the nose towards +z positive
    t/c, f/c    the airfoil's thickness and camber line scaled to these maxima over the chord
    dt1 .. dtk, dc1 .. dck
                optional, k columns each: the Bezier changes of the thickness and the camber line (below)

sweep and rake count from the root's row: the root's LE (before its pitch) is the origin. Between the rows a
distribution is the cubic spline through them: the file's shape, the reference the design is compared with.

Frame (mm; the rudder's working frame): the origin at the root's LE, x chordwise aft, y along the span (0 at
the root, the span at the tip), z across the section, positive on the side the camber bulges to (a wing's
upper side). The XCAD file has X = x, Y = -z, Z = y (rudder_modify.XCAD_FRAME), in m.

The section at eta lies in the plane y = eta span. With x the chord fraction from the LE,

    T(x) = t/c T0(x) + sum_k dt_k B_k,n(x)       the thickness over the chord
    C(x) = f/c C0(x) + sum_k dc_k B_k,n(x)       the camber line over the chord
    upper side z = c (C + T/2), lower side z = c (C - T/2), at x c

T0 is the airfoil's thickness with its trailing edge closed (the linear ramp x T(1) taken off) over its
maximum, C0 its camber line over its maximum (thickness and camber at the same x, as airfoils.shape and
para.py take them). B_k,n (k = 1 .. n - 1, n = k_max + 1) are the Bernstein polynomials of degree n: zero at
the LE and the TE, so the changes keep both ends and the LE's roundness (T ~ sqrt(x) there). The section is
then pitched about (ref_xc c, 0) and moved by (sweep, rake).

Design (WingDesign): the span (mm) and per distribution a curve in the (eta, value) plane, blade_modify's
BladeCurve (1 segment: C0 at the root .. Cm at the tip) or TwoSegmentCurve (2 segments: P1 .. P7, zero slope
at P4). Positions are fractions of the span, so a change of the span stretches the curves over it; their
values (mm, deg, t/c, f/c) stay. The root values of sweep and rake are pinned at 0. Each change dt_k, dc_k has
a curve of its own (zero where the file has none).

    params = read_params("wing_params.dat")
    design, residuals = fit_design(params)
    space = WingSpace(params, design, free=("chord", "pitch"))
    x0, bounds = space.to_vector(design), space.bounds         # for the optimiser
    d = space.to_design(x)
    problems = design_problems(d, params)
    out = write_case("outputs/wing", "case1", d, params, space)
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

import blade_modify as BM
from blade_modify import BladeCurve, TwoSegmentCurve, curve_at, fit_form
from replica_funcs import bernstein
from rudder_modify import fit_values

CURVES = ("chord", "sweep", "rake", "pitch", "thickness", "camber")
COLUMNS = ("eta", "chord_mm", "sweep_mm", "rake_mm", "pitch_deg", "t/c", "f/c")
UNITS = {"chord": "mm", "sweep": "mm", "rake": "mm", "pitch": "deg", "thickness": "t/c", "camber": "f/c"}
DESCRIPTIONS = {
    "chord": "the chord, mm",
    "sweep": "x of the LE, mm (the sweep, aft positive)",
    "rake": "z of the LE, mm (the dihedral, towards the cambered side positive)",
    "pitch": "the pitch (twist) about the pitch axis, deg (the nose towards +z positive)",
    "thickness": "the airfoil's maximum thickness over the chord, t/c",
    "camber": "the airfoil's maximum camber over the chord, f/c",
}
PINNED = ("sweep", "rake")        # root values 0: the root's LE is the origin
GLOBALS = ("span_mm", "airfoil", "ref_xc", "camber_xc")
REF_XC = 0.25                     # the pitch axis, fraction of the chord from the LE
CAMBER_XC = 0.4                   # a symmetric airfoil's camber line: NACA 4-digit, its maximum here
CHANGES = 3                       # change points (thickness and camber each) when the file has none
MAX_CHANGES = 8
DEFAULT_SEGMENTS = 1              # every curve 1 segment unless set otherwise
DEFAULT_ORDER = 3                 # a distribution's curve: 4 control points
CHANGE_ORDER = 2                  # a change's curve: 3 control points
CHANGE_BOUNDS = (-0.02, 0.02)     # a change's values (fractions of the chord) unless the file goes further
SECTIONS = 41                     # sections in the XCAD file, evenly from the root to the tip
POINTS = 61                       # points per side of a section, LE to TE (cosine spacing)
ROW_TOL = 1e-8
N_FIT = 41                        # evenly spaced eta (with the rows) where the curves are fitted to the file


def change_names(k):
    """The change curves of k change points: dt1 .. dtk (thickness), dc1 .. dck (camber line)."""
    return tuple(f"dt{i}" for i in range(1, k + 1)) + tuple(f"dc{i}" for i in range(1, k + 1))


def unit(name):
    return UNITS.get(name) or ("t/c" if name.startswith("dt") else "f/c")


def describe_curve(name, changes=None):
    """One line on what a curve is."""
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    what = "thickness" if name.startswith("dt") else "camber line"
    n = "" if changes is None else f" of {changes}"
    return f"the change of the {what} at Bezier point {name[2:]}{n} along the chord, fraction of the chord"


# --------------------------------------------------------------------------
# The parameter file
# --------------------------------------------------------------------------

@dataclass
class WingParams:
    """A wing parameter file: the globals and the spanwise table."""
    span: float                   # mm
    airfoil: str
    eta: np.ndarray               # rising, 0 .. 1
    values: dict                  # curve name -> column (sweep and rake from the root's row)
    changes: int = 0              # change points in the file (0: none)
    ref_xc: float = REF_XC
    camber_xc: float = CAMBER_XC
    kind: str = "wing"            # "wing" or "fin": the file's first line
    path: str = ""                # the file it was read from ("" when made here)

    @property
    def names(self):
        return CURVES + change_names(self.changes)

    def spline(self, name):
        """The file's distribution between its rows (cubic spline, linear beyond the ends); 0 for a change
        the file has none of."""
        if name not in self.values:
            return lambda e: np.zeros(np.shape(e)) if np.ndim(e) else np.float64(0.0)
        if len(self.eta) == 2:
            return lambda e: np.interp(e, self.eta, self.values[name])
        return BM._extended(CubicSpline(self.eta, self.values[name]), 0.0, 1.0)


def read_params(path):
    """The WingParams of a parameter file (see the module docstring)."""
    head, first = {}, ""
    with open(path) as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            if i == 0:
                first = s.lower()
            if not s.startswith("#"):
                continue
            parts = s[1:].split(None, 1)
            if len(parts) == 2 and parts[0] in GLOBALS:
                head[parts[0]] = parts[1].strip()
    missing = [k for k in ("span_mm", "airfoil") if k not in head]
    if missing:
        raise ValueError(f"{path}: the header misses {', '.join(missing)}")
    tab = np.atleast_2d(np.loadtxt(path, comments="#"))
    n = tab.shape[1]
    if n < len(COLUMNS) or (n - len(COLUMNS)) % 2:
        raise ValueError(f"{path}: the table needs {len(COLUMNS)} columns ({' '.join(COLUMNS)}), then as many "
                         f"dt columns as dc columns; it has {n}")
    k = (n - len(COLUMNS)) // 2
    if k > MAX_CHANGES:
        raise ValueError(f"{path}: at most {MAX_CHANGES} change points")
    eta = tab[:, 0].copy()
    if len(eta) < 2 or np.any(np.diff(eta) <= 0.0) or abs(eta[0]) > ROW_TOL or abs(eta[-1] - 1.0) > ROW_TOL:
        raise ValueError(f"{path}: eta must rise from 0 (the root) to 1 (the tip), at least 2 rows")
    eta[0], eta[-1] = 0.0, 1.0
    span, ref, cxc = float(head["span_mm"]), float(head.get("ref_xc", REF_XC)), float(head.get("camber_xc", CAMBER_XC))
    if span <= 0.0:
        raise ValueError(f"{path}: the span must be positive")
    if not 0.0 <= ref <= 1.0 or not 0.0 < cxc < 1.0:
        raise ValueError(f"{path}: ref_xc lies in [0, 1], camber_xc in (0, 1)")
    names = CURVES + change_names(k)
    values = {c: tab[:, i + 1].copy() for i, c in enumerate(names)}
    for c in PINNED:
        values[c] -= values[c][0]
    params = WingParams(span, head["airfoil"], eta, values, k, ref, cxc, "fin" if "fin" in first else "wing",
                        os.path.abspath(path))
    try:
        base_shape(params.airfoil, params.camber_xc)       # the airfoil exists and has a thickness
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    return params


def write_params(path, params, notes=()):
    """Write a parameter file (read_params reads it back): camber_xc when the airfoil is symmetric, the change
    columns when the file has change points."""
    names = params.names
    cols = COLUMNS + tuple(names[len(CURVES):])
    lines = [f"# XGeom {params.kind} parameters"]
    lines += [f"# {n}" for n in notes]
    lines += [f"# span_mm     {params.span:.10g}", f"# airfoil     {params.airfoil}", f"# ref_xc      {params.ref_xc:.10g}"]
    if base_shape(params.airfoil, params.camber_xc)[2]["symmetric"]:
        lines.append(f"# camber_xc   {params.camber_xc:.10g}")
    lines.append("# " + "  ".join(cols))
    for i, e in enumerate(params.eta):
        row = [e] + [params.values[c][i] for c in names]
        lines.append("  ".join(f"{BM._clean(v):.8f}" for v in row))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# The section
# --------------------------------------------------------------------------

@lru_cache(maxsize=32)
def base_shape(airfoil, camber_xc=CAMBER_XC):
    """(T0, C0, info) of a named airfoil: functions of the chord fraction x (any array shape) giving its
    thickness, the trailing edge closed, over its maximum, and its camber line over its maximum (a symmetric
    airfoil: the NACA 4-digit camber line with its maximum at camber_xc). Upper and lower sides are
    interpolated against sqrt(x) (monotone, airfoils.shape). info: t_max and c_max (the airfoil's own, over
    the chord), x_t and x_c (where T0 and C0 peak), te (the TE thickness taken off), symmetric, source."""
    import airfoils
    xy, src = airfoils.coordinates(airfoil)
    up, lo = airfoils.surfaces(xy)
    fu = PchipInterpolator(np.sqrt(np.clip(up[:, 0], 0.0, None)), up[:, 1], extrapolate=True)
    fl = PchipInterpolator(np.sqrt(np.clip(lo[:, 0], 0.0, None)), lo[:, 1], extrapolate=True)
    q = np.linspace(0.0, 1.0, 2001)
    x = q * q
    t, c = fu(q) - fl(q), 0.5 * (fu(q) + fl(q))
    te = float(t[-1])
    t = t - x * te                                         # the TE closed
    t_max, c_max = float(t.max()), float(c.max())
    if t_max <= 0.0:
        raise ValueError(f"airfoil '{airfoil}': no thickness")
    tiny = 1e-3 * t_max                                    # camber below this: the file's rounding
    symmetric = c_max <= tiny
    if symmetric:
        if float(c.min()) < -tiny:
            raise ValueError(f"airfoil '{airfoil}': its camber line is negative; f/c scales a camber line with "
                             f"a positive maximum")
        p = float(camber_xc)
        cn = np.where(x < p, (2.0 * p * x - x * x) / p ** 2, ((1.0 - 2.0 * p) + 2.0 * p * x - x * x) / (1.0 - p) ** 2)
    else:
        cn = c / c_max
    ft, fc = PchipInterpolator(q, t / t_max), PchipInterpolator(q, cn)

    def T0(xx):
        return ft(np.sqrt(np.clip(np.asarray(xx, dtype=float), 0.0, 1.0)))

    def C0(xx):
        return fc(np.sqrt(np.clip(np.asarray(xx, dtype=float), 0.0, 1.0)))
    info = {"t_max": t_max, "c_max": max(c_max, 0.0), "x_t": float(x[np.argmax(t)]), "x_c": float(x[np.argmax(cn)]),
            "te": te, "symmetric": bool(symmetric), "source": src}
    return T0, C0, info


def chord_points(m=POINTS):
    """m chord fractions from the LE (0) to the TE (1), closer together at both ends (cosine spacing)."""
    return 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, int(m))))


def change_basis(x, k):
    """The changes' Bernstein polynomials B_1,n .. B_k,n (n = k + 1) at x: (len(x), k)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if k == 0:
        return np.zeros((len(x), 0))
    return bernstein(x, k + 1)[:, 1:-1]


def section_shape(values, params, x, k):
    """Thickness and camber line over the chord, (T, C) each (stations, len(x)), at the chord fractions x, of
    the sections whose distributions are values (curve -> array over the stations; k change points)."""
    T0, C0, _ = base_shape(params.airfoil, params.camber_xc)
    x = np.asarray(x, dtype=float)
    B = change_basis(x, k)
    T = np.outer(values["thickness"], T0(x))
    C = np.outer(values["camber"], C0(x))
    if k:
        T += np.column_stack([values[f"dt{i}"] for i in range(1, k + 1)]) @ B.T
        C += np.column_stack([values[f"dc{i}"] for i in range(1, k + 1)]) @ B.T
    return T, C


def section_loops(design, params, eta, m=POINTS):
    """The sections at the span fractions eta (mm, the wing's frame): (stations, 2m - 1, 3), each closed, from
    the LE along the +z side to the TE (index m - 1) and back along the -z side to the LE, as XCAD reads them
    (rudder_modify.xcad_loop's layout)."""
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    return loops_of(design.values(eta), params, eta, design.span, design.changes, m)


def file_loops(params, eta, m=POINTS):
    """section_loops of the parameter file's own wing (its splines, its span)."""
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    return loops_of({c: params.spline(c)(eta) for c in params.names}, params, eta, params.span, params.changes, m)


def loops_of(v, params, eta, span, k, m=POINTS):
    """section_loops from the distributions v (curve -> values at eta), the span and k change points."""
    x = chord_points(m)
    T, C = section_shape(v, params, x, k)
    xs = np.concatenate((x, x[::-1][1:]))
    zs = np.concatenate((C + 0.5 * T, (C - 0.5 * T)[:, ::-1][:, 1:]), axis=1)
    c, th = v["chord"][:, None], np.deg2rad(v["pitch"])[:, None]
    ref = params.ref_xc
    dx, dz = c * (xs[None, :] - ref), c * zs
    X = v["sweep"][:, None] + c * ref + dx * np.cos(th) + dz * np.sin(th)
    Z = v["rake"][:, None] - dx * np.sin(th) + dz * np.cos(th)
    Y = np.broadcast_to(eta[:, None] * span, X.shape)
    return np.stack((X, Y, Z), axis=-1)


def section_metrics(design, params, eta, m=4 * POINTS):
    """Per station: the largest thickness and camber over the chord and where they are (x/c), the section's
    area (mm^2). Dict of arrays."""
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    v = design.values(eta)
    x = np.linspace(0.0, 1.0, int(m))
    T, C = section_shape(v, params, x, design.changes)
    it, ic = T.argmax(axis=1), np.abs(C).argmax(axis=1)
    rows = np.arange(len(eta))
    return {"t_max": T[rows, it], "x_t": x[it], "f_max": C[rows, ic], "x_f": x[ic],
            "area": v["chord"] ** 2 * _integrate(T, x)}


def _integrate(y, x):
    """The trapezoid rule along the last axis (numpy's, under either name)."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x, axis=-1)


def section_stations(n=SECTIONS):
    """The n span fractions of the XCAD file's sections, evenly from the root (0) to the tip (1)."""
    n = int(n)
    if n < 3:
        raise ValueError("at least 3 sections")
    return np.linspace(0.0, 1.0, n)


# --------------------------------------------------------------------------
# Curves and designs
# --------------------------------------------------------------------------

@dataclass
class WingDesign:
    """A wing: its span (mm) and one curve per distribution over eta, from the root (0) to the tip (1), the
    change curves dt1 .. dtk, dc1 .. dck included."""
    span: float
    curves: dict

    @property
    def changes(self):
        return sum(1 for c in self.curves if c.startswith("dt"))

    @property
    def names(self):
        return CURVES + change_names(self.changes)

    def copy(self):
        return copy.deepcopy(self)

    def ctrl(self, name):
        """The control points (eta, value), root first."""
        return self.curves[name].ctrl(0.0, 1.0)

    def at(self, name, eta):
        """Curve `name` at the span fractions eta (clipped to [0, 1]), any array shape."""
        return curve_at(self.curves[name], 0.0, 1.0, eta)

    def values(self, eta):
        """{curve: values at the span fractions eta}."""
        eta = np.atleast_1d(np.asarray(eta, dtype=float))
        return {c: self.at(c, eta) for c in self.names}


def zero_curve(segments=1, order=CHANGE_ORDER):
    """A curve that is 0 along the whole span: evenly placed control points (1 segment), P4 at mid-span (2)."""
    if int(segments) == 2:
        return TwoSegmentCurve(0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.0)
    return BladeCurve(np.linspace(0.0, 1.0, int(order) + 1), np.zeros(int(order) + 1))


def pinned_key(curve):
    """The slot that holds a curve's root value."""
    return "v1" if curve.segments == 2 else "v0"


def _pin_root(curve, eta, v):
    """The curve's values fitted again to (eta, v) by least squares with the root's at 0, its positions and
    weights kept."""
    if curve.segments == 1:
        vals, _ = fit_values(curve.u, eta, v, pinned={0: 0.0})
        return BladeCurve(np.array(curve.u, dtype=float), vals)
    a = BM._two_segment_basis(np.asarray(eta, dtype=float), 0.0, 1.0, curve.u4, curve.d1, curve.d2, curve.w23,
                              curve.w56)
    v4, v7 = np.linalg.lstsq(a[:, 1:], np.asarray(v, dtype=float), rcond=None)[0]
    return dataclasses.replace(curve, v1=0.0, v4=float(v4), v7=float(v7))


def fit_stations(params):
    """Where the curves are fitted to the file: its rows and N_FIT evenly spaced eta."""
    return np.unique(np.r_[params.eta, np.linspace(0.0, 1.0, N_FIT)])


def fit_curve_to(eta, v, name, segments=DEFAULT_SEGMENTS, order=None):
    """A curve of the form (segments, order) through the data (eta, v): blade_modify.fit_form, the root
    pinned at 0 for sweep and rake; a curve that is 0 along the span when the data is."""
    if order is None:
        order = DEFAULT_ORDER if name in CURVES else CHANGE_ORDER
    v = np.asarray(v, dtype=float)
    if np.all(np.abs(v) <= 1e-12):
        return zero_curve(segments, order)
    cu = None
    if int(segments) == 1:                       # evenly placed control points: exact for a polynomial of the order
        u = np.linspace(0.0, 1.0, int(order) + 1)
        vals, res = fit_values(u, eta, v)
        if np.abs(res).max() <= 1e-9 * max(float(np.ptp(v)), float(np.abs(v).max())):
            cu = BladeCurve(u, vals)
    if cu is None:
        cu, _ = fit_form(eta, v, 0.0, 1.0, segments, order)
    return _pin_root(cu, eta, v) if name in PINNED else cu


def fit_design(params, orders=None, segments=None, changes=None):
    """The file's wing as a design: its span and each distribution fitted by a curve of segments[name]
    segments (default DEFAULT_SEGMENTS; 1 segment of orders[name], default DEFAULT_ORDER, CHANGE_ORDER for
    the changes) at fit_stations; the file's change points, or `changes` (default CHANGES) of them, zero,
    when it has none. Returns (design, {curve: residuals at the rows})."""
    orders, segments = dict(orders or {}), dict(segments or {})
    k = params.changes or (CHANGES if changes is None else int(changes))
    if not 0 <= k <= MAX_CHANGES:
        raise ValueError(f"0 to {MAX_CHANGES} change points")
    eta = fit_stations(params)
    curves = {}
    for c in CURVES + change_names(k):
        curves[c] = fit_curve_to(eta, params.spline(c)(eta), c, segments.get(c, DEFAULT_SEGMENTS), orders.get(c))
    design = WingDesign(float(params.span), curves)
    res = {c: design.at(c, params.eta) - params.spline(c)(params.eta) for c in design.names}
    return design, res


def refit_curve(design, name, order=None, segments=None, n_samples=None):
    """The design with curve `name` fitted again to its present shape: in another form (segments 1 or 2) or,
    1 segment, at another order (default: the curve's own, DEFAULT_ORDER coming from 2 segments)."""
    cur = design.curves[name]
    seg = cur.segments if segments is None else int(segments)
    if order is None:
        order = cur.order if cur.segments == 1 else (DEFAULT_ORDER if name in CURVES else CHANGE_ORDER)
    eta = np.linspace(0.0, 1.0, n_samples or (121 if seg == 2 else 401))
    out = design.copy()
    out.curves[name] = fit_curve_to(eta, design.at(name, eta), name, seg, order)
    return out


def refit_to_file(params, design, name, segments, order=None):
    """The design with curve `name` fitted again to the parameter file's distribution (fit_stations)."""
    eta = fit_stations(params)
    out = design.copy()
    out.curves[name] = fit_curve_to(eta, params.spline(name)(eta), name, segments, order)
    return out


def with_changes(design, k, segments=1, order=CHANGE_ORDER, n_eta=41, n_x=201):
    """The design with k change points (0 .. MAX_CHANGES) for the thickness and the camber line each: the
    changes carried over at n_eta stations (exactly where k grows: Bezier degree elevation; by least squares
    where it shrinks) and each new change curve fitted to them (segments, order)."""
    k = int(k)
    if not 0 <= k <= MAX_CHANGES:
        raise ValueError(f"0 to {MAX_CHANGES} change points")
    k0 = design.changes
    if k == k0:
        return design.copy()
    eta, x = np.linspace(0.0, 1.0, n_eta), np.linspace(0.0, 1.0, n_x)
    b0, b1 = change_basis(x, k0), change_basis(x, k)
    out = WingDesign(design.span, {c: copy.deepcopy(design.curves[c]) for c in CURVES})
    for p in ("dt", "dc"):
        old = np.column_stack([design.at(f"{p}{i}", eta) for i in range(1, k0 + 1)]) if k0 else np.zeros((n_eta, 0))
        new = np.linalg.lstsq(b1, b0 @ old.T, rcond=None)[0].T if k else np.zeros((n_eta, 0))   # (n_eta, k)
        for i in range(1, k + 1):
            out.curves[f"{p}{i}"] = fit_curve_to(eta, new[:, i - 1], f"{p}{i}", segments, order)
    out.curves = {c: out.curves[c] for c in out.names}
    return out


def fit_report(params, design):
    """{curve: (largest |design - file| at the rows, the same over the file's range)}; a change the file has
    none of is compared with 0."""
    out = {}
    for c in design.names:
        f = params.spline(c)(params.eta)
        e = float(np.abs(design.at(c, params.eta) - f).max())
        out[c] = (e, e / max(float(np.ptp(f)), 1e-12) if np.ptp(f) > 0 else (0.0 if e < 1e-12 else np.inf))
    return out


# --------------------------------------------------------------------------
# The design space
# --------------------------------------------------------------------------

def _round_out(lo, hi):
    """lo and hi rounded outward to two figures of their difference."""
    span = max(hi - lo, 1e-12)
    places = int(1 - np.floor(np.log10(span)))
    f = 10.0 ** places
    return round(np.floor(lo * f) / f, max(places, 0)), round(np.ceil(hi * f) / f, max(places, 0))


def default_bounds(params, design):
    """Bounds by group: "d" (1-segment position fractions), "seg.d", "seg.u", "seg.w" as blade_modify's; each
    distribution's values from the lowest to the highest of its rows and control values, half that range
    again on either side, the range at least a fifth of the mean chord (chord), of the span (sweep, rake),
    4 deg (pitch), 0.04 (t/c, f/c); chord and thickness not below 0 unless a fitted control value is; the
    changes at least CHANGE_BOUNDS; the span 50 % to 150 % of the design's (to the mm)."""
    b = {"d": BM.D_BOUNDS, "seg.d": BM.SEG_D_BOUNDS, "seg.u": BM.SEG_U_BOUNDS}
    w = [x for c in design.names if design.curves[c].segments == 2
         for x in (design.curves[c].w23, design.curves[c].w56)]
    b["seg.w"] = (BM.SEG_W_BOUNDS[0], float(max([2.0] + [np.ceil(15.0 * x) / 10.0 for x in w])))
    floors = {"chord": 0.2 * float(np.mean(params.values["chord"])), "sweep": 0.2 * params.span,
              "rake": 0.2 * params.span, "pitch": 4.0, "thickness": 0.04, "camber": 0.04}
    for c in design.names:
        allv = np.concatenate((params.spline(c)(params.eta), design.curves[c].v))
        lo, hi = float(allv.min()), float(allv.max())
        if c not in CURVES:                                  # a change
            r = hi - lo
            b[c] = (min(round(lo - 0.5 * r, 4), CHANGE_BOUNDS[0]), max(round(hi + 0.5 * r, 4), CHANGE_BOUNDS[1]))
            continue
        r = max(hi - lo, floors[c])
        lo, hi = _round_out(lo - 0.5 * r, hi + 0.5 * r)
        if c in ("chord", "thickness"):                      # positive, unless the fit put a control value below 0
            vmin = float(design.curves[c].v.min())
            lo = min(max(lo, 0.0), vmin)
        b[c] = (float(lo), float(hi))
    b["span"] = (float(np.floor(0.5 * design.span)), float(np.ceil(1.5 * design.span)))
    return b


class WingSpace:
    """A flat design vector for an optimiser, and the WingDesign it stands for.

    The slots: "span" (mm), then curve by curve (design.names order) its own: 1 segment d1 .. d(m-1) (the
    fractions that place the inner control points), v0 (root) .. vm (tip); 2 segments v1, d1, w23, u4, v4, d2,
    w56, v7; the root values of sweep and rake are pinned at 0 and no slots. Names "span", "chord.d1",
    "chord.v0", ... The curves' forms (segments, order) and the number of change points are the design's.

    bounds   overrides of default_bounds by group ("d", "seg.d", "seg.u", "seg.w"), curve ("chord": its
             values; "chord.d", "chord.u", "chord.w") or slot ("chord.v3", "span"); a slot beats its curve, a
             curve its group
    fixed    {slot: value} of the held slots (not in the vector); with free given instead (curve names and
             "span"), every other slot is held at the design's value
    """

    def __init__(self, params, design, bounds=None, fixed=None, free=None):
        self.params = params
        self.curve_names = design.names
        self.forms = {c: (design.curves[c].segments, design.curves[c].order) for c in self.curve_names}
        b = {**default_bounds(params, design), **(bounds or {})}
        self._slots = [(None, "span")]
        for c in self.curve_names:
            cu = design.curves[c]
            self._slots += [(c, k) for k in cu.keys() if not (c in PINNED and k == pinned_key(cu))]
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

    @property
    def changes(self):
        return sum(1 for c in self.curve_names if c.startswith("dt"))

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
        """The WingDesign of {slot name: value} for every slot."""
        return self._design_of(np.array([float(values[n]) for n in self.all_names]))

    def _design_of(self, full):
        vals = dict(zip(self.all_names, (float(x) for x in full)))
        curves = {}
        for c in self.curve_names:
            seg, order = self.forms[c]
            cls = BM.CURVE_TYPES[seg]
            keys = cls.keys_of(order)
            pin = "v1" if seg == 2 else "v0"
            curves[c] = cls.from_keys({k: 0.0 if c in PINNED and k == pin else vals[f"{c}.{k}"] for k in keys}, order)
        return WingDesign(vals["span"], curves)

    def values(self, design):
        """{slot name: value} of a design, every slot."""
        return {name: float(design.span) if c is None else design.curves[c].get(k)
                for (c, k), name in zip(self._slots, self.all_names)}

    def to_vector(self, design):
        vals = self.values(design)
        return np.array([vals[n] for n in self.names])

    def to_dict(self, design):
        vals = self.values(design)
        return {"forms": {c: f[0] for c, f in self.forms.items()}, "orders": {c: f[1] for c, f in self.forms.items()},
                "changes": self.changes,
                "slots": [{"name": n, "value": vals[n], "lo": float(lo), "hi": float(hi), "free": n not in self.fixed}
                          for n, (lo, hi) in zip(self.all_names, self.all_bounds)],
                "vector": {"names": self.names, "values": [vals[n] for n in self.names],
                           "bounds": self.bounds.tolist()}}

    @classmethod
    def from_dict(cls, params, rec):
        """(space, design) from to_dict's output."""
        forms, orders = rec["forms"], rec["orders"]
        k = int(rec.get("changes", 0))
        curves = {c: zero_curve(forms[c], orders[c]) for c in CURVES + change_names(k)}
        tmpl = WingDesign(float(params.span), curves)
        vals = {q["name"]: q["value"] for q in rec["slots"]}
        bounds = {q["name"]: (q["lo"], q["hi"]) for q in rec["slots"]}
        fixed = {q["name"]: q["value"] for q in rec["slots"] if not q["free"]}
        space = cls(params, tmpl, bounds, fixed)
        return space, space.design_of(vals)


def space_record(space, design, params, settings=None, geometry=None):
    """The set-up as plain data: what it is for (geometry: "wing" or "fin", default the parameter file's
    kind), the parameter file (path and globals), the settings, the space with the design in it."""
    return {"geometry": geometry or params.kind, "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "params_file": params.path,
            "globals": {"span_mm": params.span, "airfoil": params.airfoil, "ref_xc": params.ref_xc,
                        "camber_xc": params.camber_xc},
            "settings": dict(settings or {}), "space": space.to_dict(design)}


def load_wing_space(path):
    """(space, design, record) of a set-up the design tool saved (<case>_design_space.json); the parameter
    file is read again from the path the record names:

        space, design, rec = load_wing_space("outputs/wing/gui_design_space.json")
        x0, bounds = space.to_vector(design), space.bounds
        d = space.to_design(x)
        problems = design_problems(d, space.params)
    """
    with open(path) as fh:
        rec = json.load(fh)
    params = read_params(rec["params_file"])
    space, design = WingSpace.from_dict(params, rec["space"])
    return space, design, rec


# --------------------------------------------------------------------------
# Checks and output
# --------------------------------------------------------------------------

def design_checks(design, params, n=SECTIONS):
    """(problems, warnings) of a design. Problems make it unbuildable: a span not positive; a chord not
    positive along the span; a section whose sides cross (its thickness not positive inside the chord) at n
    stations. No warnings yet."""
    if not design.span > 0.0:
        return [f"span {design.span:.4g} mm: it must be positive"], []
    out = []
    eta = np.linspace(0.0, 1.0, 401)
    c = design.at("chord", eta)
    if np.any(c <= 0.0):
        out.append(f"chord not positive from eta {eta[np.argmax(c <= 0.0)]:.3f}")
    es = np.linspace(0.0, 1.0, int(n))
    x = chord_points(POINTS)[1:-1]
    T, _ = section_shape(design.values(es), params, x, design.changes)
    bad = np.any(T <= 0.0, axis=1)
    if bad.any():
        i = int(np.argmax(bad))
        out.append(f"the section's sides cross (thickness not positive) at eta {es[i]:.3f}"
                   + (f" to {es[bad][-1]:.3f}" if bad.sum() > 1 else "") + f", x/c {x[np.argmax(T[i] <= 0.0)]:.2f}")
    return out, []


def design_problems(design, params, n=SECTIONS):
    """design_checks' problems, for the optimiser (a design with any is rejected)."""
    return design_checks(design, params, n)[0]


def design_params(design, params, n=41):
    """The design as a WingParams: its curves at n eta (closer together at both ends), its span; the change
    columns when any change is not zero (the file then has the design's change points)."""
    eta = BM.cosine_stations(0.0, 1.0, n)
    vals = design.values(eta)
    k = design.changes
    if not any(np.any(np.abs(design.curves[c].v) > 0.0) for c in change_names(k)):
        k = 0
    return WingParams(float(design.span), params.airfoil, eta, {c: vals[c] for c in CURVES + change_names(k)}, k,
                      params.ref_xc, params.camber_xc, params.kind)


def write_xcad(path, loops, units="m"):
    """The sections as an XCAD point file (X = x, Y = -z, Z = y; '#k' before each, CRLF), in `units`."""
    import rudder_modify as RM
    RM.write_xcad_lines(path, [RM.xcad_lines(lp, units) for lp in loops])
    return path


SECTION_COLUMNS = ("section", "eta", "y_mm", "chord_mm", "sweep_mm", "rake_mm", "pitch_deg", "t/c", "f/c",
                   "t_max/c", "x_t/c", "f_max/c", "x_f/c", "area_mm2")


def write_sections(path, design, params, eta, notes=()):
    """The XCAD sections as a table: per section its distributions, the largest thickness and camber of the
    shape (the changes included) and where they are, its area."""
    v, m = design.values(eta), section_metrics(design, params, eta)
    lines = [f"# XGeom {params.kind}: the sections of the XCAD file ({len(eta)}), mm"]
    lines += [f"# {n}" for n in notes]
    lines.append("# " + "  ".join(SECTION_COLUMNS))
    for i, e in enumerate(eta):
        row = [e, e * design.span, v["chord"][i], v["sweep"][i], v["rake"][i], v["pitch"][i], v["thickness"][i],
               v["camber"][i], m["t_max"][i], m["x_t"][i], m["f_max"][i], m["x_f"][i], m["area"][i]]
        lines.append(f"{i + 1:4d}  " + "  ".join(f"{BM._clean(q):.6f}" for q in row))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def design_record(design, params, space=None, problems=(), settings=None, warnings=()):
    """The design as plain data: the parameter file it came from, the globals, each curve's form, variables
    and control points (eta, value) with their weights, the fit to the file, the checks."""
    rep = fit_report(params, design)
    rec = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "params_file": params.path,
           "kind": params.kind,
           "globals": {"span_mm": design.span, "airfoil": params.airfoil, "ref_xc": params.ref_xc,
                       "camber_xc": params.camber_xc, "changes": design.changes},
           "curves": {c: {"segments": cu.segments, "variables": {k: cu.get(k) for k in cu.keys()},
                          "control_points": design.ctrl(c).tolist(), "weights": cu.weights().tolist(),
                          "max_diff_from_file": rep[c][0]} for c, cu in design.curves.items()},
           "problems": list(problems), "warnings": list(warnings), "settings": dict(settings or {})}
    if space is not None:
        rec["space"] = space.to_dict(design)
    return rec


def write_case(out_dir, case, design, params, space=None, cad=True, settings=None, plot=True, geometry=None):
    """Write the case: <case>_params.dat (the design as a parameter file, which read_params and the tool
    load), <case>_xcad_m.dat (the sections, settings 'sections' of them, 'points' per side), <case>_sections.dat
    (their table), <case>_design.json, <case>_design_space.json (with a space; geometry as space_record's),
    <case>_check.png (plot), and with cad the loft of the XCAD file (write_cad): <case>.step. Returns the
    paths, the checks and the sections' count, planform area (mm^2) and volume (mm^3, from the sections'
    areas)."""
    os.makedirs(out_dir, exist_ok=True)
    st = dict(settings or {})
    eta = section_stations(st.get("sections", SECTIONS))
    problems, warnings = design_checks(design, params)
    paths = {k: os.path.join(out_dir, f"{case}_{k}") for k in ("params.dat", "xcad_m.dat", "sections.dat",
                                                                "design.json")}
    note = f"design {case} from {os.path.basename(params.path) or 'memory'}, {datetime.date.today().isoformat()}"
    write_params(paths["params.dat"], design_params(design, params), notes=(note,))
    write_xcad(paths["xcad_m.dat"], section_loops(design, params, eta, st.get("points", POINTS)))
    write_sections(paths["sections.dat"], design, params, eta, notes=(note,))
    with open(paths["design.json"], "w") as fh:
        json.dump(design_record(design, params, space, problems, st, warnings), fh, indent=1)
    if space is not None:
        paths["space"] = os.path.join(out_dir, f"{case}_design_space.json")
        with open(paths["space"], "w") as fh:
            json.dump(space_record(space, design, params, st, geometry), fh, indent=1)
    if plot:
        paths["check"] = plot_case(os.path.join(out_dir, f"{case}_check.png"), case, design, params)
    ee = np.linspace(0.0, 1.0, 201)
    m = section_metrics(design, params, ee)
    out = {"paths": paths, "problems": problems, "warnings": warnings, "sections": len(eta),
           "area_mm2": float(_integrate(design.at("chord", ee), ee) * design.span),
           "volume_mm3": float(_integrate(m["area"], ee) * design.span)}
    if cad:
        if problems:
            raise ValueError(f"{case}: no CAD for a design with problems: {problems[0]}")
        out.update(write_cad(out_dir, case, paths["xcad_m.dat"]))
        paths.update(out.pop("cad_paths"))
    return out


def write_cad(out_dir, case, xcad_path):
    """The solid lofted through the XCAD file's sections (xcad_loft.loft_loops: the root and the flat tip
    closed by planar faces), mm, the XCAD frame, as <case>.step."""
    import time
    import xcad_loft as XL
    t0 = time.time()
    loft = XL.loft_loops(XL.read_loops(xcad_path, "m"))
    step = XL.write_step(loft.shape, os.path.join(out_dir, f"{case}.step"))
    return {"cad_paths": {"step": step}, "loft": loft, "cad_seconds": time.time() - t0}


def plot_case(path, case, design, params):
    """The planform (x across, the span up) and each distribution (its value across, the span up), the
    design against the file; the sections at the root, mid-span and the tip over their chords."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(15, 7.5))
    FigureCanvasAgg(fig)
    axs = fig.subplots(2, 4)
    ee = np.linspace(0.0, 1.0, 200)
    y, yf = ee * design.span, ee * params.span
    ax = axs[0, 0]
    loops = section_loops(design, params, ee, 41)
    ax.plot(loops[:, :, 0].min(axis=1), y, "-", color="#2a78d6", lw=1.8, label="design: LE, TE")
    ax.plot(loops[:, :, 0].max(axis=1), y, "-", color="#2a78d6", lw=1.8)
    v = {c: params.spline(c)(ee) for c in ("chord", "sweep")}
    ax.plot(v["sweep"], yf, "--", color="#9a9994", lw=1, label="file (unpitched)")
    ax.plot(v["sweep"] + v["chord"], yf, "--", color="#9a9994", lw=1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y, span (mm)")
    ax.legend(fontsize=7)
    ax.set_title("planform", loc="left", fontsize=10)
    for ax, c in zip(axs.flat[1:7], CURVES):
        ax.plot(params.spline(c)(ee), yf, "-", color="#9a9994", lw=1, label="file")
        ax.plot(params.values[c], params.eta * params.span, ".", color="#52514e", ms=3)
        ax.plot(design.at(c, ee), y, "-", color="#2a78d6", lw=1.8, label="design")
        p = design.ctrl(c)
        ax.plot(p[:, 1], p[:, 0] * design.span, "o--", color="#2a78d6", lw=0.8, ms=4, mfc="white")
        ax.set_xlabel(f"{c} ({UNITS[c]}), {design.curves[c].segments} segment"
                      + ("s" if design.curves[c].segments == 2 else ""))
        ax.set_ylabel("y, span (mm)")
    ax = axs[1, 3]
    x = np.linspace(0.0, 1.0, 301)
    for e, col in zip((0.0, 0.5, 1.0), ("#1f4e9a", "#3fa0c8", "#1baf7a")):
        vv = design.values([e])
        T, C = section_shape(vv, params, x, design.changes)
        base = dict(vv, **{q: np.zeros(1) for q in change_names(design.changes)})
        T0, C0 = section_shape(base, params, x, design.changes)
        ax.plot(x, (C + 0.5 * T)[0], "-", color=col, lw=1.4, label=f"eta {e:g}")
        ax.plot(x, (C - 0.5 * T)[0], "-", color=col, lw=1.4)
        ax.plot(x, (C0 + 0.5 * T0)[0], ":", color=col, lw=0.8)
        ax.plot(x, (C0 - 0.5 * T0)[0], ":", color=col, lw=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x/c")
    ax.set_ylabel("z/c")
    ax.legend(fontsize=7)
    ax.set_title(f"sections (dotted: {params.airfoil} scaled, no changes)", loc="left", fontsize=9)
    for ax in axs.flat:
        ax.grid(True, color="#e4e3df")
    fig.suptitle(f"{case}: {params.kind}, span {design.span:.4g} mm, {params.airfoil}; design (blue, control polygon "
                 f"dashed) and the parameter file (grey)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    return path
