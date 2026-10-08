"""
xgeom_hull.py

The hull in the XGeom design tool (xgeom_tool.py): a hull parameter file (hull_modify.py) as the tool's
variables, its live views and the CAD build.

A hull starts from a parameter file, a station table (the nose, middle-body, tail and cap lengths and the
half-height r at stations, with elliptical sections also the half-width r'); by default SUBOFF's
(suboff_hull_params.dat, or SUBOFF's equations when that file is not there). Its variables (hull_modify.HullSpace):
the four lengths, the largest radius r* and the tail end's radius re (with elliptical sections also r'* and r'e),
and each curve's exponent and free coefficients. The curves: nose r and tail r; with elliptical sections
(set_elliptic) also nose r' and tail r', fitted to the parameter file's half-widths (an axisymmetric file: copies
of the r curves). A new order (set_order) is fitted to the parameter file while none of the curve's variables has
changed, else to the curve's present shape. load_params reads another parameter file; save_params writes the
design as one.

The adapter has the methods of xgeom_rudder.RudderAdapter that the tool uses.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import hull_modify as HM
from xgeom_common import AQUA, BLUE, GREY, GRID, INK2, ORANGE, RED, SECTION_COLORS, Row, fit_box, keep_fitted

CURVES = ("nose_r", "tail_r", "nose_rp", "tail_rp")
LABELS = {"nose_r": "nose r", "tail_r": "tail r", "nose_rp": "nose r'", "tail_rp": "tail r'"}
GLOBAL_ROWS = (("nose", "nose (m)"), ("middle", "middle (m)"), ("tail", "tail (m)"), ("cap", "cap (m)"),
               ("r", "r* (m)"), ("re", "r end (m)"), ("rp", "r'* (m)"), ("rpe", "r' end (m)"))
VIEW_COLORS = {"r": BLUE, "rp": ORANGE}
PART_NAMES = ("nose", "middle", "tail", "cap")
PART_COLORS = (BLUE, INK2, ORANGE, AQUA)            # the parts in the 3D sections tab
SKELETON_STATIONS = 25                              # stations per curved part in the 3D sections tab


class HullAdapter:
    joins = False                 # no join switch
    segment_choice = False        # one CST curve per part and view
    build_label = "CAD"
    order_range = (2, HM.MAX_ORDER)                 # a CST curve needs one free coefficient at least

    def __init__(self, params=None):
        self.out_dir = HM.OUT_DIR
        self.settings = {"plot": True}
        self.load_params(params or HM.SUBOFF_PARAMS)

    # ---------------------------------------------------------------- the file
    def load_params(self, path):
        """Start from a parameter file: its curves fitted (order hull_modify.DEFAULT_ORDER), the lengths and the
        curves free (hull_modify.DEFAULT_FREE), the radii and the cap held."""
        if os.path.exists(path):
            params, source = HM.read_params(path), os.path.basename(path)
        elif os.path.basename(path) == HM.SUBOFF_PARAMS:
            params, source = HM.suboff_params(), "SUBOFF's equations"
        else:
            raise FileNotFoundError(path)
        self.params, self.source = params, source
        self.design = HM.fit_design(params)
        self.meta = {}
        self.as_fitted = set(self.design.curves)    # curves still as fitted to the file (no variable changed)
        self._rebuild()

    def save_params(self, path):
        """Write the design as a parameter file (read_params and load_params read it)."""
        HM.write_params(path, HM.design_params(self.design),
                        notes=(f"written by the design tool from {self.source}, {time.strftime('%d %b %Y %H:%M')}",))
        return path

    @property
    def title(self):
        return f"Hull ({self.source})"

    @property
    def names(self):
        return [c for c in CURVES if c in self.design.curves]

    def curve_label(self, curve):
        """The curve's name in the panel: nose r, or with elliptical sections nose r (XZ view) etc."""
        if self.design.axisymmetric:
            return curve.replace("_", " ")
        return LABELS[curve] + (" (XZ view)" if curve.endswith("_r") else " (XY view)")

    # ---------------------------------------------------------------- state
    def _rebuild(self):
        """The space for the present curves, keeping the bounds and free flags of every slot that is still
        there; a new slot gets hull_modify's default bounds and is free as in hull_modify.DEFAULT_FREE."""
        full = HM.HullSpace(self.design, fixed={})
        held = HM.HullSpace(self.design).fixed
        self.meta = {n: self.meta.get(n, {"lo": float(lo), "hi": float(hi), "free": n not in held})
                     for n, (lo, hi) in zip(full.all_names, full.all_bounds)}
        self.space = HM.HullSpace(self.design, bounds={n: (m["lo"], m["hi"]) for n, m in self.meta.items()},
                                  fixed={})
        self.values = self.space.values(self.design)

    def get(self, name):
        return self.values[name]

    def bounds(self, name):
        return self.meta[name]["lo"], self.meta[name]["hi"]

    def is_free(self, name):
        return self.meta[name]["free"]

    def set(self, name, value):
        """Set one variable; returns the value used (clamped to its bounds)."""
        lo, hi = self.bounds(name)
        value = float(min(max(value, lo), hi))
        vals = dict(self.values)
        vals[name] = value
        self.design = self.space.design_of(vals)
        self.values = self.space.values(self.design)
        self.as_fitted.discard(name.split(".")[0])
        return value

    def set_bounds(self, name, lo, hi):
        lo, hi = float(lo), float(hi)
        if not lo < hi:
            raise ValueError("the lower bound must be below the upper one")
        if name in ("nose", "tail", "r", "rp") and lo <= 0.0:
            raise ValueError("the nose and tail lengths and the largest radii are positive")
        if name in ("middle", "cap", "re", "rpe") and lo < 0.0:
            raise ValueError("a length or a radius is not negative")
        if name.split(".")[-1] in ("N1", "N2") and lo <= 0.0:
            raise ValueError("an exponent is positive")
        self.meta[name].update(lo=lo, hi=hi)
        self._rebuild()
        v = self.values[name]
        if not lo <= v <= hi:
            self.set(name, v)                       # clamps into the new bounds
        return lo, hi

    def set_free(self, name, free):
        self.meta[name]["free"] = bool(free)

    def part_range(self, kind):
        """x range of the nose or the tail (m)."""
        j = self.design.joins
        return (j[0], j[1]) if kind == "nose" else (j[2], j[3])

    def _file_fit(self, curve, order):
        """The curve of an order fitted to the parameter file's stations (its exponent searched)."""
        kind, view = curve.split("_")
        psi, y, _, _ = HM.region_data(self.params, view, kind)
        return HM.fit_cst(kind, psi, y, order)[0]

    def set_order(self, curve, order):
        """Number of control points of a curve = order + 1. The curve is fitted again: to the parameter file
        while it is as fitted to it (no variable of it changed since the file was read), else to its present
        shape (its exponent kept). Its slots are renewed, free if any of them was. Returns what was done, for
        the status bar."""
        order = int(order)
        lo, hi = self.order_range
        if not lo <= order <= hi:
            raise ValueError(f"a hull curve has {lo + 1} .. {hi + 1} control points (order {lo} .. {hi})")
        if order == self.order(curve):
            return "unchanged"
        kind, view = curve.split("_")
        was_free = any(m["free"] for n, m in self.meta.items() if n.startswith(curve + "."))
        x = np.linspace(*self.part_range(kind), 401)
        old = self.design.radius(view, x)
        d = self.design.copy()
        d.curves[curve] = (self._file_fit(curve, order) if curve in self.as_fitted
                           else HM.refit_curve(self.design, curve, order))
        self.design = d
        for n in [n for n in self.meta if n.startswith(curve + ".")]:
            del self.meta[n]
        self._rebuild()
        for n, m in self.meta.items():
            if n.startswith(curve + "."):
                m["free"] = was_free
        if curve in self.as_fitted:
            e = HM.fit_report(self.params, self.design)[curve]
            return f"fitted to the parameter file (largest difference {1000 * e:.2f} mm)"
        e = float(np.abs(self.design.radius(view, x) - old).max())
        return f"fitted to the curve as it was (largest change {1000 * e:.2f} mm)"

    def order(self, curve):
        return self.design.curves[curve].order

    @property
    def elliptic(self):
        return not self.design.axisymmetric

    def set_elliptic(self, on):
        """Elliptical sections (on): the half-width r' gets curves and radii of its own, nose r' and tail r'
        fitted to the parameter file's half-widths (an axisymmetric file: copies of the r curves, r'* = r*,
        r'e = re). Off: circular sections, r' = r."""
        d = self.design.copy()
        if on and d.axisymmetric:
            for kind in HM.KINDS:
                if self.params.axisymmetric:
                    d.curves[f"{kind}_rp"] = d.curves[f"{kind}_r"].copy()
                    if f"{kind}_r" in self.as_fitted:
                        self.as_fitted.add(f"{kind}_rp")
                else:
                    d.curves[f"{kind}_rp"] = self._file_fit(f"{kind}_rp", HM.DEFAULT_ORDER)
                    self.as_fitted.add(f"{kind}_rp")
            if self.params.axisymmetric:
                d.rp, d.rpe = d.r, d.re
            else:
                _, _, d.rp, d.rpe = HM.region_data(self.params, "rp", "tail", 50)
        elif not on and not d.axisymmetric:
            for kind in HM.KINDS:
                d.curves.pop(f"{kind}_rp")
                self.as_fitted.discard(f"{kind}_rp")
            d.rp = d.rpe = None
        else:
            return
        self.design = d
        self._rebuild()

    # ---------------------------------------------------------------- rows
    def global_rows(self):
        """Rows for the lengths and the radii (m)."""
        return [Row("var", label, name, 4) for name, label in GLOBAL_ROWS if name in self.meta]

    def curve_rows(self, curve):
        """Rows for one curve: its exponent, then its control points from the nose tip (or the middle body)
        along x; the two at the middle body are pinned (r* with zero slope there)."""
        c = self.design.curves[curve]
        n, key, coef = c.order, HM.EXP_KEY[c.kind], c.coeffs()
        join = f"{coef[n - 1 if c.kind == 'nose' else 1]:.4f} = 1 + {key}/{n}: zero slope at the middle body"
        rows = [Row("var", f"{key} exponent", f"{curve}.{key}", 4)]
        if c.kind == "nose":
            rows += [Row("var", f"C{i} b{i}" + (" (tip)" if i == 0 else ""), f"{curve}.b{i}", 4) for i in range(n - 1)]
            rows += [Row("pinned", f"C{n - 1}", text=join),
                     Row("pinned", f"C{n}", text="1: the largest radius, at the middle body")]
        else:
            rows += [Row("pinned", "C0", text="1: the largest radius, at the middle body"),
                     Row("pinned", "C1", text=join)]
            rows += [Row("var", f"C{i} t{i}" + (" (end)" if i == n else ""), f"{curve}.t{i}", 4)
                     for i in range(2, n + 1)]
        return rows

    def row_info(self, name):
        """Where a variable sits: the x a length ends at, the depth or breadth a radius gives, the x of a
        curve's control point (m)."""
        d = self.design
        j = d.joins
        if name in ("nose", "middle", "tail"):
            return f"x {j[PART_NAMES.index(name) + 1]:.3f}"
        if name == "cap":
            return f"L {d.length:.3f}"
        if name == "r":
            return f"2r {2.0 * d.r:.3f}"
        if name == "rp":
            return f"2r' {2.0 * d.rp:.3f}"
        if name in ("re", "rpe"):
            return f"x {j[3]:.3f}"
        curve, key = name.split(".")
        kind, view = curve.split("_")
        if key in ("N1", "N2"):
            return f"x {j[0] if kind == 'nose' else j[3]:.3f}"
        return f"x {d.ctrl(view, kind)[0][int(key[1:])]:.3f}"

    def ctrl_kinds(self, curve):
        """For each control point C0 .. Cn of a curve: 'free', 'held' (its coefficient a held variable) or
        'pinned' (the two at the middle body)."""
        c = self.design.curves[curve]
        n, out = c.order, []
        for i in range(n + 1):
            key = (f"b{i}" if i <= n - 2 else None) if c.kind == "nose" else (f"t{i}" if i >= 2 else None)
            out.append("pinned" if key is None else "free" if self.meta[f"{curve}.{key}"]["free"] else "held")
        return np.array(out)

    def n_free(self, curve=None):
        names = [n for n in self.space.all_names if curve is None or n.startswith(curve + ".")]
        return sum(1 for n in names if self.meta[n]["free"]), len(names)

    def summary(self):
        n, m = self.n_free()
        d = self.design
        size = (f"D {2 * d.r:.4g} m" if d.axisymmetric else
                f"depth {2 * d.r:.4g} m, breadth {2 * d.rp:.4g} m (elliptical sections)")
        try:
            hs = HM.hydrostatics(d, 201)
            hydro = (f"; V {hs['volume']:.4g} m^3, S {hs['wetted_surface']:.4g} m^2, LCB {hs['lcb'] / d.length:.4f} L, "
                     f"Cp {hs['cp']:.4f}")
        except Exception:                                    # a design with problems: the status bar says why
            hydro = ""
        return f"{n} of {m} variables free; L {d.length:.4g} m, {size}{hydro}"

    def describe(self, curve):
        c = self.design.curves[curve]
        kind, view = curve.split("_")
        n, key = c.order, HM.EXP_KEY[c.kind]
        what = ("the radius r" if self.design.axisymmetric else
                "the half-height r (XZ view)" if view == "r" else "the half-width r' (XY view)")
        if kind == "nose":
            text = (f"Nose: {what} from the tip (x 0) to the middle body, r = r* psi^{key} (b0 B0 + .. + bn Bn), "
                    f"Bernstein polynomials of order {n}; {key} sets the bluntness (0.5 round, 1 pointed); "
                    f"b0 .. b{n - 2} are variables, C{n - 1} and C{n} are pinned (r* with zero slope at the "
                    f"middle body)")
        else:
            text = (f"Tail: {what} from the middle body to the end, r = re + (r* - re) (1 - psi)^{key} (t0 B0 + .. "
                    f"+ tn Bn), order {n}; {key} sets the end (2 zero slope, 1 a cone, below 1 rounded); C0 and C1 "
                    f"are pinned (r* with zero slope at the middle body), t2 .. t{n} are variables")
        e = HM.fit_report(self.params, self.design).get(curve)
        if e is not None:
            text += f"; largest difference from the parameter file {1000 * e:.2f} mm"
        return text

    # ---------------------------------------------------------------- preview
    def preview(self, skeleton=False):
        """What the live views draw: the design and its problems, the half-profiles r and r' along x, the
        parameter file; with skeleton the sections of the 3D sections tab by part."""
        t0 = time.time()
        d = self.design
        x = HM.stations_x(d, 161)
        out = {"design": d, "params": self.params, "x": x, "r": d.radius("r", x), "rp": d.radius("rp", x),
               "problems": HM.design_problems(d)}
        if skeleton:
            xs = HM.stations_x(d, SKELETON_STATIONS)
            part = np.clip(np.searchsorted(d.joins[1:-1], xs, side="left"), 0, 3)    # nose 0 .. cap 3
            pts = HM.section_points(d, xs, 72)
            out["skeleton"] = {name: [np.vstack((p, p[:1])) for p, k in zip(pts, part) if k == i]
                               for i, name in enumerate(PART_NAMES)}
        out["seconds"] = time.time() - t0
        return out

    def views(self, fig):
        return HullViews(self, fig)

    def skeleton_view(self, fig):
        return HullSectionsView(self, fig)

    # ---------------------------------------------------------------- output
    def free_space(self):
        """The design space with the unticked variables held fixed: the one an optimiser gets."""
        fixed = {n: self.values[n] for n in self.space.all_names if not self.meta[n]["free"]}
        return HM.HullSpace(self.design, bounds={n: (m["lo"], m["hi"]) for n, m in self.meta.items()}, fixed=fixed)

    def space_record(self):
        """The set-up as plain data (hull_modify.load_hull_space reads it back)."""
        return HM.space_record(self.free_space(), self.design, self.params)

    def save_space(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.space_record(), fh, indent=1)
        return path

    def load_space(self, path):
        """Start from a saved set-up (<case>_design_space.json): its design, bounds and free flags; its
        parameter file is read again (none: SUBOFF's equations; not found: the design's own stations)."""
        space, design, rec = HM.load_hull_space(path)
        if rec.get("geometry") != "hull":
            raise ValueError(f"{path} is not a hull design space")
        pf = rec.get("params_file")
        if pf and os.path.exists(pf):
            self.params, self.source = HM.read_params(pf), os.path.basename(pf)
        elif not pf:
            self.params, self.source = HM.suboff_params(), "SUBOFF's equations"
        else:
            self.params, self.source = HM.design_params(design), f"{os.path.basename(path)} ({pf} not found)"
        self.design = design
        self.as_fitted = set()                      # a saved design: new orders keep its curves' shapes
        self.meta = {q["name"]: {"lo": q["lo"], "hi": q["hi"], "free": q["free"]} for q in rec["space"]["slots"]}
        self._rebuild()

    def build_task(self, case):
        """A function that writes the case as the design is now (hull_modify.write_case: the parameter file,
        the design and the set-up JSON, stations, hydrostatics, section points, the check plot and the STEP).
        The state is copied here, so it can run off the GUI thread."""
        design, params, space = self.design.copy(), self.params, self.free_space()
        out_dir, plot = self.out_dir, self.settings["plot"]

        def run():
            import xcad_loft as XL
            cad = XL.occ_available()
            files, hs = HM.write_case(out_dir, case, design, params, space, cad=cad, plot=plot)
            out = {"files": files, "hydrostatics": hs, "view": files.get("step"), "error": not cad,
                   "report": [f"{case}: L {design.length:.4f} m, V {hs['volume']:.6f} m^3, S "
                              f"{hs['wetted_surface']:.5f} m^2, LCB {hs['lcb'] / design.length:.4f} L, Cp "
                              f"{hs['cp']:.4f}", "  written: " + ", ".join(files.values())]}
            names = f"{case}_params.dat, {case}_design.json"
            if cad:
                out["message"] = (f"{names} and {case}.step (V {hs['volume']:.5g} m^3, S {hs['wetted_surface']:.5g} "
                                  f"m^2 from the solid); OCC viewer shows it")
            else:
                out["message"] = f"{names}; no CAD: needs pythonocc-core or cadquery-ocp"
            return out
        return run


def short_ticks(axis, half):
    """Two ticks, -a and a, on the short axis of a long body (three would overlap): a is half rounded down to
    the cm."""
    a = float(np.floor(half * 100.0) / 100.0) or float(half)
    axis.set_ticks([-a, a])


def _markers(ax, x, y, kinds, color, ms=6, labels=True):
    """Control points: free filled, held hollow, pinned hollow squares."""
    for kind, style, lab in (("free", dict(marker="o"), "control point: free"),
                             ("held", dict(marker="o", mfc="white"), "held"),
                             ("pinned", dict(marker="s", mfc="white"), "pinned")):
        m = kinds == kind
        if np.any(m):
            ax.plot(x[m], y[m], ls="none", color=color, ms=ms, label=lab if labels else None, **style)


class HullViews:
    """The Design tab: the half-profiles of the whole hull and the picked curve's part (both with x along
    the hull on the x axis), and in 3D the hull (sections, the top and side lines; dashed: the parameter
    file's)."""

    def __init__(self, adapter, fig):
        self.a = adapter
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.15], wspace=0.2, hspace=0.36,
                              left=0.07, right=0.98, top=0.95, bottom=0.03)
        self.ax_prof = fig.add_subplot(gs[0, 0])
        self.ax_curve = fig.add_subplot(gs[0, 1])
        self.ax_3d = fig.add_subplot(gs[1, :], projection="3d")
        self.ax_3d.view_init(elev=20, azim=-70)
        self.box = None
        keep_fitted(fig, self.ax_3d, lambda: self.box)

    def update(self, pv, curve):
        self._profiles(pv, curve)
        self._curve(pv, curve)
        self._hull(pv, curve)

    @staticmethod
    def _label(d, view):
        if d.axisymmetric:
            return "radius r"
        return "half-height r (XZ view)" if view == "r" else "half-width r' (XY view)"

    def _profiles(self, pv, curve):
        """The whole hull: r (and r') along x, the parameter file's stations, every curve's control polygon
        (the picked one's part shaded), the parts' ends dotted."""
        ax, a, d, p = self.ax_prof, self.a, pv["design"], pv["params"]
        ax.cla()
        ax.grid(True, color=GRID, lw=0.6)
        ax.axvspan(*a.part_range(curve.split("_")[0]), color=GRID, alpha=0.7, lw=0)
        j = d.joins
        for jx in j[1:-1]:
            ax.axvline(jx, color=GREY, lw=0.7, ls=":")
        for name, (x0, x1) in zip(PART_NAMES, zip(j[:-1], j[1:])):
            if x1 - x0 > 0.02 * d.length:
                ax.text(0.5 * (x0 + x1), 0.98, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=8, color=INK2)
        views = ("r",) if d.axisymmetric else HM.VIEWS
        ax.plot(p.x, p.values("r"), ".", color=GREY, ms=2.5, label="parameter file")
        if not d.axisymmetric:
            ax.plot(p.x, p.values("rp"), ".", color=GREY, ms=2.5)
        for view in views:
            col = VIEW_COLORS[view]
            ax.plot(pv["x"], pv[view], "-", color=col, lw=1.8, label=self._label(d, view))
            for kind in HM.KINDS:
                name = f"{kind}_{view}"
                cx, cy = d.ctrl(view, kind)
                ax.plot(cx, cy, "--", color=col, lw=1.2 if name == curve else 0.6)
                _markers(ax, cx, cy, a.ctrl_kinds(name), col, 4.5 if name == curve else 3.5,
                         labels=view == "r" and kind == "nose")
        top = max(float(np.max(pv["r"])), float(np.max(pv["rp"])), float(np.max(p.r)), float(np.max(p.rp)))
        ax.set_ylim(0.0, 1.18 * max(top, 1e-9))
        ax.set_xlim(-0.01 * d.length, 1.01 * max(d.length, float(p.x[-1])))
        ax.set_xlabel("x from the nose (m)")
        ax.set_ylabel("r (m)" if d.axisymmetric else "r, r' (m)")
        ax.legend(loc="center", fontsize=7, ncol=2)
        ax.set_title("half-profile" + ("" if d.axisymmetric else "s: r (XZ view, blue), r' (XY view, orange)") +
                     f"; L {d.length:.4f} m", loc="left", fontsize=10)
        if pv["problems"]:
            ax.text(0.01, 0.88, "\n".join(q[:90] for q in pv["problems"][:3]), transform=ax.transAxes, va="top",
                    ha="left", color=RED, fontsize=8)

    def _curve(self, pv, curve):
        """The picked curve's part: the parameter file's stations there, the design, the control polygon
        (free filled, held hollow, pinned squares) and r* dotted."""
        ax, a, d, p = self.ax_curve, self.a, pv["design"], pv["params"]
        ax.cla()
        ax.grid(True, color=GRID, lw=0.6)
        kind, view = curve.split("_")
        col = VIEW_COLORS[view]
        x0, x1 = a.part_range(kind)
        m = (p.x >= x0 - 1e-9) & (p.x <= x1 + 1e-9)
        ax.plot(p.x[m], p.values(view)[m], ".", color=INK2, ms=3, label="parameter file")
        t = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 300)))
        xs = x0 + (x1 - x0) * t
        ax.plot(xs, d.radius(view, xs), "-", color=col, lw=2.0, label="design")
        cx, cy = d.ctrl(view, kind)
        ax.plot(cx, cy, "--", color=col, lw=0.8)
        _markers(ax, cx, cy, a.ctrl_kinds(curve), col)
        top, end = d.radii(view)
        ax.axhline(top, color=GREY, ls=":", lw=0.8)
        c = d.curves[curve]
        ax.set_xlim(x0 - 0.04 * (x1 - x0), x1 + 0.04 * (x1 - x0))
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(0.0, lo), max(hi, 1.08 * top))
        xa, xb = ax.get_xlim()
        ya, yb = ax.get_ylim()
        for i, (xk, yk) in enumerate(zip(cx, cy)):      # near the right edge to the left, near the top below
            right, high = xk > xa + 0.8 * (xb - xa), yk > ya + 0.85 * (yb - ya)
            ax.annotate(f"C{i}", (xk, yk), textcoords="offset points", xytext=(-6 if right else 6, -6 if high else 4),
                        fontsize=8, color=col, ha="right" if right else "left", va="top" if high else "bottom")
        ax.set_xlabel("x from the nose (m)")
        ax.set_ylabel(self._label(d, view) + " (m)")
        ax.legend(loc="lower right" if kind == "nose" else "lower left", fontsize=7)
        ax.set_title(f"curve {LABELS[curve] if not d.axisymmetric else kind + ' r'}: order {c.order} "
                     f"({c.order + 1} control points), {HM.EXP_KEY[kind]} {c.exponent:.4f}", loc="left", fontsize=10)

    def _hull(self, pv, curve):
        """The hull in 3D (m): sections at a few stations, the top, bottom and side lines (red: the picked
        curve's part), the parameter file's lines dashed grey. The lines are not clipped: a 3D axes clips
        to its square, which a long hull fitted to the panel (fit_box) overruns."""
        ax, a, d, p = self.ax_3d, self.a, pv["design"], pv["params"]
        elev, azim, roll = ax.elev, ax.azim, ax.roll
        ax.cla()
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.tick_params(labelsize=8)
        j = d.joins
        xs = np.unique(np.r_[j[0] + (j[1] - j[0]) * np.array([0.15, 0.55]), j[1], j[2],
                             j[2] + (j[3] - j[2]) * np.array([0.4, 0.8]), j[3]])
        pts = []
        for sec, col in zip(HM.section_points(d, xs, 97), SECTION_COLORS):
            sec = np.vstack((sec, sec[:1]))
            ax.plot(sec[:, 0], sec[:, 1], sec[:, 2], color=col, lw=1.3, clip_on=False)
            pts.append(sec)
        x, r, rp = pv["x"], pv["r"], pv["rp"]
        z0 = np.zeros_like(x)
        kind, view = curve.split("_")
        x0, x1 = a.part_range(kind)
        pick = (x >= x0 - 1e-12) & (x <= x1 + 1e-12)
        for (yy, zz), v in (((z0, r), "r"), ((z0, -r), "r"), ((rp, z0), "rp"), ((-rp, z0), "rp")):
            ax.plot(x, yy, zz, "-", color=INK2, lw=0.9, clip_on=False)
            if v == view or d.axisymmetric:
                ax.plot(x[pick], yy[pick], zz[pick], "-", color=RED, lw=1.8, clip_on=False)
            pts.append(np.column_stack((x, yy, zz)))
        pz = np.zeros_like(p.x)
        for yy, zz in ((pz, p.r), (pz, -p.r), (p.rp, pz), (-p.rp, pz)):
            ax.plot(p.x, yy, zz, "--", color=GREY, lw=0.7, clip_on=False)
        q = np.vstack(pts)
        lo, hi = q.min(axis=0), q.max(axis=0)
        pad = 0.03 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        self.box = np.maximum(hi - lo, 1e-9)
        fit_box(ax, self.fig, self.box)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(3))
        short_ticks(ax.yaxis, max(float(np.max(rp)), 1e-9))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        part = "the nose" if kind == "nose" else "the tail"
        ax.set_title(f"the hull: sections, top, bottom and side lines (red: {part}, the picked curve's); dashed: the "
                     f"parameter file; drag to turn", fontsize=9)


class HullSectionsView:
    """The 3D sections tab: the hull as sections (elliptical), nose blue, middle body dark grey, tail orange,
    cap aqua."""

    def __init__(self, adapter, fig):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        self.a = adapter
        self.fig = fig
        self.ax = ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.col = {name: Line3DCollection([np.zeros((2, 3))], colors=c, linewidths=0.8)
                    for name, c in zip(PART_NAMES, PART_COLORS)}
        for c in self.col.values():
            c.set_clip_on(False)                    # a long hull overruns the 3D axes' square (fit_box)
            ax.add_collection3d(c)
        from matplotlib.ticker import MaxNLocator
        ax.zaxis.set_major_locator(MaxNLocator(3))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=20, azim=-60)
        self.text = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, fontsize=9, va="top")
        self.box = None
        keep_fitted(fig, ax, lambda: self.box)

    def update(self, pv):
        sk = pv["skeleton"]
        for name, c in self.col.items():
            c.set_segments(sk[name] or [np.zeros((2, 3))])
        allp = np.vstack([s for v in sk.values() for s in v])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        ax = self.ax
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        short_ticks(ax.yaxis, max(hi[1], 1e-9))
        self.box = np.maximum(hi - lo, 1e-9)
        fit_box(ax, self.fig, self.box)
        d = pv["design"]
        n = {k: len(v) for k, v in sk.items()}
        msg = (f"{sum(n.values())} sections: nose {n['nose']} (blue), middle body {n['middle']} (dark grey), tail "
               f"{n['tail']} (orange), cap {n['cap']} (aqua); L {d.length:.4f} m, "
               + (f"D {2 * d.r:.4f} m" if d.axisymmetric else f"depth {2 * d.r:.4f} m, breadth {2 * d.rp:.4f} m"))
        if pv["problems"]:
            msg += "\nproblem: " + pv["problems"][0][:100]
        self.text.set_text(msg)
