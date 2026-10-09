"""
xgeom_wing.py

The wing and the fin in the XGeom design tool (xgeom_tool.py): a wing parameter file (wing_modify.py) as the
tool's variables, its live views and the CAD build. The two share all of it; each has its own parameter file
and output folder (ROLES).

A wing or a fin starts from a parameter file: the span, the airfoil, the pitch axis and the spanwise table.
Its variables: the span (mm), and each distribution (chord, sweep, rake, pitch, thickness, camber) and each
section change (dt1 .. dtk of the thickness, dc1 .. dck of the camber line) as a Bezier curve over the span
fraction eta, fitted to the file, of 1 or 2 segments as the blade's (xgeom_blade.py). The root values of sweep
and rake are pinned at 0: the root's LE is the origin. A change of the span stretches the curves over it, their
values staying. A new form or order is fitted to the parameter file while none of the curve's variables has
changed, else to the curve's present shape. set_changes sets the number of change points along the chord (0
to 8): the changes are carried over, exactly when there are more of them. set_sections sets the sections of the
XCAD file. load_params reads another parameter file, save_params writes the design as one.

The views are drawn as the rudder's, in the wing's own frame: x (or the curve's value) across and the span up.

The adapter has the methods of xgeom_rudder.RudderAdapter that the tool uses.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import wing_modify as WM
from xgeom_common import BLUE, GREY, GRID, INK2, RED, SECTION_COLORS, Row, fit_box, keep_fitted

MAX_ORDER = 12
VIEW_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)     # sections in the Design tab's views, fractions of the span
FREE = ("chord", "pitch")                         # the curves whose variables start free
ROLES = {"wing": {"name": "Wing", "params": "wing_params.dat", "out_dir": os.path.join("outputs", "wing")},
         "fin": {"name": "Fin", "params": "fin_params.dat", "out_dir": os.path.join("outputs", "fin")}}
TWO_SEGMENT_ROWS = (("v1", "P1 value (root)"), ("d1", "P2=P3 d1"), ("w23", "P2=P3 weight"), ("u4", "P4 position"),
                    ("v4", "P4 value"), ("d2", "P5=P6 d2"), ("w56", "P5=P6 weight"), ("v7", "P7 value (tip)"))
TWO_SEGMENT_POINT = {"v1": 0, "d1": 1, "w23": 1, "u4": 3, "v4": 3, "d2": 4, "w56": 4, "v7": 6}
DIGITS = {"chord": 2, "sweep": 2, "rake": 2, "pitch": 3}  # value digits; t/c, f/c and the changes 5


class WingAdapter:
    joins = False                 # the curves have no join to switch
    segment_choice = True         # each curve 1 or 2 segments
    build_label = "CAD"
    role = "wing"

    def __init__(self, params=None):
        r = ROLES[self.role]
        self.settings = {"sections": WM.SECTIONS, "points": WM.POINTS, "plot": True, "free": FREE}
        self.out_dir = r["out_dir"]
        self.load_params(params or r["params"])

    # ---------------------------------------------------------------- the file
    def load_params(self, path):
        """Start from a parameter file: its distributions fitted (1 segment), the variables of FREE free."""
        params = WM.read_params(path)
        design, _ = WM.fit_design(params)
        self.params, self.design, self.meta = params, design, {}
        self.as_fitted = set(design.names)          # curves still as fitted to the file (no variable changed)
        self._rebuild()
        self._original = None

    def save_params(self, path):
        """Write the design as a parameter file (read_params and load_params read it)."""
        WM.write_params(path, WM.design_params(self.design, self.params),
                        notes=(f"written by the design tool from {os.path.basename(self.params.path)}, "
                               f"{time.strftime('%d %b %Y %H:%M')}",))
        return path

    @property
    def title(self):
        return f"{ROLES[self.role]['name']} ({os.path.basename(self.params.path)})"

    @property
    def names(self):
        return self.design.names

    # ---------------------------------------------------------------- state
    def _rebuild(self):
        """The space for the present curves, keeping the bounds and free flags of every slot still there."""
        full = WM.WingSpace(self.params, self.design)
        free0 = self.settings["free"]
        self.meta = {n: self.meta.get(n, {"lo": float(lo), "hi": float(hi), "free": n.split(".")[0] in free0})
                     for n, (lo, hi) in zip(full.all_names, full.all_bounds)}
        self.space = WM.WingSpace(self.params, self.design,
                                  bounds={n: (m["lo"], m["hi"]) for n, m in self.meta.items()})
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
        key = name.split(".")[-1]
        if name == "span" and lo <= 0.0:
            raise ValueError("the span is positive")
        if "." in name and key[0] == "d" and not 0.0 <= lo < hi <= 1.0:
            raise ValueError("a position fraction d lies in [0, 1]")
        if key == "u4" and not 0.0 < lo < hi < 1.0:
            raise ValueError("P4's position lies inside the span, (0, 1)")
        if key in ("w23", "w56") and lo <= 0.0:
            raise ValueError("a weight is positive")
        self.meta[name].update(lo=lo, hi=hi)
        self._rebuild()
        v = self.values[name]
        if not lo <= v <= hi:
            self.set(name, v)                       # clamps into the new bounds
        return lo, hi

    def set_free(self, name, free):
        self.meta[name]["free"] = bool(free)

    def _drop(self, curve):
        for n in [n for n in self.meta if n.startswith(curve + ".")]:
            del self.meta[n]

    def _refit(self, curve, segments, order):
        """The curve in a new form, its slots renewed (free if any of its slots was): fitted to the parameter
        file while it is as fitted to it, else to its present shape. Returns what was done."""
        was_free = any(m["free"] for n, m in self.meta.items() if n.startswith(curve + "."))
        ee = np.linspace(0.0, 1.0, 401)
        old = self.design.at(curve, ee)
        if curve in self.as_fitted:
            self.design = WM.refit_to_file(self.params, self.design, curve, segments, order)
        else:
            self.design = WM.refit_curve(self.design, curve, order=order, segments=segments)
        self._drop(curve)
        self._rebuild()
        for n, m in self.meta.items():
            if n.startswith(curve + "."):
                m["free"] = was_free
        if curve in self.as_fitted:
            e = WM.fit_report(self.params, self.design)[curve][0]
            return f"fitted to the parameter file (largest difference {e:.3g} {WM.unit(curve)})"
        e = float(np.abs(self.design.at(curve, ee) - old).max())
        return f"fitted to the curve as it was (largest change {e:.3g} {WM.unit(curve)})"

    def set_order(self, curve, order):
        """Number of control points of a 1-segment curve = order + 1; the curve is fitted again (_refit)."""
        order = int(order)
        if self.segments(curve) != 1:
            raise ValueError("a 2-segment curve has its 7 control points P1 .. P7")
        if not 1 <= order <= MAX_ORDER:
            raise ValueError(f"the order must lie in 1 .. {MAX_ORDER}")
        return self._refit(curve, 1, order)

    def order(self, curve):
        return self.design.curves[curve].order

    def segments(self, curve):
        return self.design.curves[curve].segments

    def set_segments(self, curve, n):
        """The curve as 1 segment (DEFAULT_ORDER, CHANGE_ORDER for a change) or 2 segments, fitted again."""
        n = int(n)
        if n not in (1, 2):
            raise ValueError("a curve has 1 or 2 segments")
        if n == self.segments(curve):
            return "unchanged"
        return self._refit(curve, n, WM.DEFAULT_ORDER if curve in WM.CURVES else WM.CHANGE_ORDER)

    def set_changes(self, k):
        """k change points along the chord for the thickness and the camber line each (0 to MAX_CHANGES): the
        changes carried over (wing_modify.with_changes), the new change curves held unless one of the old ones
        had a free variable. Returns what was done."""
        k = int(k)
        k0 = self.design.changes
        if k == k0:
            return "unchanged"
        old = WM.change_names(k0)
        free = any(m["free"] for n, m in self.meta.items() if n.split(".")[0] in old)
        ee, x = np.linspace(0.0, 1.0, 41), np.linspace(0.0, 1.0, 201)
        before = WM.section_shape(self.design.values(ee), self.params, x, k0)
        self.design = WM.with_changes(self.design, k)
        after = WM.section_shape(self.design.values(ee), self.params, x, k)
        for c in old:
            self._drop(c)
            self.as_fitted.discard(c)
        self._rebuild()
        for n, m in self.meta.items():
            if n.split(".")[0] in WM.change_names(k):
                m["free"] = free
        e = max(float(np.abs(a - b).max()) for a, b in zip(before, after))
        return f"the changes carried over (largest change of the sections {e:.1e} of the chord)"

    # ---------------------------------------------------------------- rows
    def global_rows(self):
        return [Row("var", "span (mm)", "span", 2)]

    def curve_label(self, curve):
        if curve in WM.CURVES:
            return curve
        return f"{curve}: {'thickness' if curve.startswith('dt') else 'camber'} change at point {curve[2:]}"

    def curve_rows(self, curve):
        """Rows for the variables of one curve, from the root to the tip; the pinned root of sweep and rake."""
        cu = self.design.curves[curve]
        digits = DIGITS.get(curve, 5)
        pinned = curve in WM.PINNED
        root_text = "0: the root's LE is the origin"
        if cu.segments == 2:
            return [Row("pinned", label, text=root_text) if pinned and k == "v1" else
                    Row("var", label, f"{curve}.{k}", digits if k[0] == "v" else 4) for k, label in TWO_SEGMENT_ROWS]
        n = cu.order
        rows = []
        for k in range(n + 1):
            if 0 < k < n:
                rows.append(Row("var", f"C{k} d", f"{curve}.d{k}", 4))
            label = f"C{k} value" + (" (root)" if k == 0 else " (tip)" if k == n else "")
            rows.append(Row("pinned", label, text=root_text) if pinned and k == 0 else
                        Row("var", label, f"{curve}.v{k}", digits))
        return rows

    def row_info(self, name):
        """Where a variable sits: its control point's y along the span (mm); for the span the planform area."""
        d = self.design
        if name == "span":
            ee = np.linspace(0.0, 1.0, 201)
            return f"S {WM._integrate(d.at('chord', ee), ee) * d.span / 100.0:.0f} cm2"
        curve, key = name.split(".")
        k = TWO_SEGMENT_POINT[key] if self.segments(curve) == 2 else int(key[1:])
        return f"y {d.ctrl(curve)[k, 0] * d.span:.1f}"

    def ctrl_free(self, curve):
        """For each control point of a curve: is any variable that moves it free? (a pinned root: no)"""
        cu = self.design.curves[curve]
        free = {k: self.meta[f"{curve}.{k}"]["free"] for k in cu.keys() if f"{curve}.{k}" in self.meta}
        if cu.segments == 2:
            groups = (("v1",), ("d1", "w23"), ("d1", "w23"), ("u4", "v4"), ("d2", "w56"), ("d2", "w56"), ("v7",))
            return np.array([any(free.get(k, False) for k in g) for g in groups])
        n = cu.order
        return np.array([free.get(f"v{k}", False) or (0 < k < n and free[f"d{k}"]) for k in range(n + 1)])

    def n_free(self, curve=None):
        names = [n for n in self.space.all_names if curve is None or n.startswith(curve + ".")]
        return sum(1 for n in names if self.meta[n]["free"]), len(names)

    def summary(self):
        n, m = self.n_free()
        d = self.design
        c = d.at("chord", [0.0, 1.0])
        return (f"{n} of {m} variables free; span {d.span:.1f} mm, chord {c[0]:.1f} mm at the root and {c[1]:.1f} mm "
                f"at the tip; {d.changes} change point{'s' if d.changes != 1 else ''} along the chord")

    def describe(self, curve):
        e, rel = WM.fit_report(self.params, self.design)[curve]
        diff = f"largest difference from the parameter file {e:.3g} {WM.unit(curve)}"
        if curve in WM.CURVES and np.isfinite(rel) and rel > 0.0:
            diff += f" ({100 * rel:.2f} % of its range)"
        what = WM.describe_curve(curve, self.design.changes) + " against the span"
        if curve in WM.PINNED:
            what += " (the root's value pinned at 0)"
        if self.segments(curve) == 2:
            return (f"{what}, 2 rational cubic Bezier segments P1-P4 and P4-P7: P2 = P3 and P5 = P6 at P4's value, "
                    f"so zero slope at P4; {diff}")
        return f"{what}, a Bezier curve from the root (C0) to the tip; {diff}"

    # ---------------------------------------------------------------- sections
    def stations(self):
        return WM.section_stations(self.settings["sections"])

    def set_sections(self, n):
        """n sections in the XCAD file, evenly from the root to the tip (3 or more)."""
        WM.section_stations(n)
        self.settings["sections"] = int(n)

    def sections_info(self):
        n, m, k = self.settings["sections"], self.settings["points"], self.design.changes
        return (f"{n} sections {self.design.span / (n - 1):.1f} mm apart along the span, {2 * m - 1} points each "
                f"({m} per side, closer together at the LE and the TE); the tip is flat for now. Change points: "
                f"{k} Bezier points along the chord that change the thickness (dt) and the camber line (dc), "
                f"each with its curve over the span")

    # ---------------------------------------------------------------- preview
    def _original_views(self):
        """The parameter file's own wing (its splines, its span), drawn dashed: cached until another file."""
        if self._original is None:
            p = self.params
            ee = np.linspace(0.0, 1.0, 61)
            loops = WM.file_loops(p, ee, 31)
            self._original = {"sections": WM.file_loops(p, VIEW_FRACTIONS, 61), "le": loops[:, 0], "te": loops[:, 30],
                              "x": (loops[:, :, 0].min(axis=1), loops[:, :, 0].max(axis=1)),
                              "z": (loops[:, :, 2].min(axis=1), loops[:, :, 2].max(axis=1)), "y": ee * p.span}
        return self._original

    def preview(self, skeleton=False):
        """What the live views draw: the design, its problems (wing_modify.design_checks), the curves, the
        file's rows and splines, the wing: sections at VIEW_FRACTIONS, LE and TE lines, the planform and front
        view outlines, the sections over their chords with and without the changes; with skeleton the XCAD
        file's sections."""
        t0 = time.time()
        d, p = self.design, self.params
        ee = np.linspace(0.0, 1.0, 121)
        out = {"design": d, "ee": ee, "curves": d.values(ee), "file": {c: p.spline(c)(ee) for c in d.names},
               "rows": (p.eta, p.values), "fractions": np.array(VIEW_FRACTIONS)}
        try:
            problems, warnings = WM.design_checks(d, p)
        except Exception as exc:
            problems, warnings = [f"{type(exc).__name__}: {exc}"], []
        try:
            loops = WM.section_loops(d, p, ee, 31)
            out.update(sections=WM.section_loops(d, p, VIEW_FRACTIONS, 61), le=loops[:, 0], te=loops[:, 30],
                       x=(loops[:, :, 0].min(axis=1), loops[:, :, 0].max(axis=1)),
                       z=(loops[:, :, 2].min(axis=1), loops[:, :, 2].max(axis=1)), y=ee * d.span)
            x = np.linspace(0.0, 1.0, 201)
            v = d.values(VIEW_FRACTIONS)
            plain = dict(v, **{c: np.zeros(len(VIEW_FRACTIONS)) for c in WM.change_names(d.changes)})
            out["shapes"] = (x, WM.section_shape(v, p, x, d.changes), WM.section_shape(plain, p, x, d.changes))
        except Exception as exc:
            out["sections"] = f"no sections: {type(exc).__name__}: {exc}"
        try:
            out["orig"] = self._original_views()
        except Exception:
            out["orig"] = None
        out["problems"], out["warnings"] = problems, warnings
        if skeleton:
            try:
                out["skeleton"] = WM.section_loops(d, p, self.stations(), self.settings["points"])
            except Exception as exc:
                out["skeleton"] = None
                problems.append(f"no sections: {type(exc).__name__}: {exc}")
        out["seconds"] = time.time() - t0
        return out

    def views(self, fig):
        return WingViews(self, fig)

    def skeleton_view(self, fig):
        return WingSkeletonView(self, fig)

    # ---------------------------------------------------------------- output
    def free_space(self):
        """The design space with the unticked variables held fixed: the one an optimiser gets."""
        fixed = {n: self.values[n] for n in self.space.all_names if not self.meta[n]["free"]}
        return WM.WingSpace(self.params, self.design, bounds={n: (m["lo"], m["hi"]) for n, m in self.meta.items()},
                            fixed=fixed)

    CASE_KEYS = ("sections", "points")

    def _case_settings(self):
        return {k: self.settings[k] for k in self.CASE_KEYS}

    def space_record(self):
        """The set-up as plain data (wing_modify.load_wing_space reads it back)."""
        return WM.space_record(self.free_space(), self.design, self.params, self._case_settings(), self.role)

    def save_space(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.space_record(), fh, indent=1)
        return path

    def load_space(self, path):
        """Start from a saved set-up (<case>_design_space.json); its parameter file is read again."""
        with open(path) as fh:
            rec = json.load(fh)
        if rec.get("geometry") not in ROLES:
            raise ValueError(f"{path} is not a wing or fin design space")
        params = WM.read_params(rec["params_file"])
        space, design = WM.WingSpace.from_dict(params, rec["space"])
        self.params, self.design, self._original = params, design, None
        self.as_fitted = set()                      # a saved design: refits keep its curves' shapes
        self.meta = {q["name"]: {"lo": q["lo"], "hi": q["hi"], "free": q["free"]} for q in rec["space"]["slots"]}
        for k in self.CASE_KEYS:
            if k in rec.get("settings", {}):
                self.settings[k] = rec["settings"][k]
        self._rebuild()

    def build_task(self, case):
        """A function that writes the case as the design is now (wing_modify.write_case: the parameter file,
        the XCAD file, the sections table, the design and the set-up JSON, the check plot), then lofts the
        XCAD file into the solid (<case>.step). The state is copied here, so it can run off the GUI thread."""
        import xcad_loft as XL
        design, params, space, st = self.design.copy(), self.params, self.free_space(), self._case_settings()
        out_dir, plot, role = self.out_dir, self.settings["plot"], self.role

        def run():
            res = WM.write_case(out_dir, case, design, params, space, cad=False, settings=st, plot=plot,
                                geometry=role)
            out = {"case": res, "view": None, "error": False,
                   "report": [f"{case}: {res['sections']} sections, planform {res['area_mm2'] / 100:.1f} cm^2, "
                              f"volume {res['volume_mm3'] / 1000:.1f} cm^3 (from the sections)",
                              "  written: " + ", ".join(res["paths"].values())]}
            files = f"{case}_params.dat, {case}_xcad_m.dat"
            if res["problems"]:
                out.update(message=f"{files}; no CAD: {res['problems'][0]}", error=True)
                return out
            if not XL.occ_available():
                out.update(message=f"{files}; no CAD: no OCCT binding (pythonocc-core or cadquery-ocp)", error=True)
                return out
            cad = WM.write_cad(out_dir, case, res["paths"]["xcad_m.dat"])
            res["paths"].update(cad["cad_paths"])
            lr = cad["loft"]
            out["report"].append(f"  CAD: the loft in {cad['cad_seconds']:.1f} s, {cad['cad_paths']['step']} (solid "
                                 f"{'valid' if lr.valid else 'NOT valid'}, {lr.volume / 1e3:.2f} cm^3)")
            out.update(view=cad["cad_paths"]["step"], loft=lr,
                       message=f"{files}, {case}.step (solid {'valid' if lr.valid else 'NOT valid'}, "
                               f"{lr.volume / 1e3:.2f} cm^3); OCC viewer shows it")
            return out
        return run


class FinAdapter(WingAdapter):
    role = "fin"


# --------------------------------------------------------------------------
# The views
# --------------------------------------------------------------------------

def _limits(ax, xs, ys, pad=0.05):
    """Axis limits around the data (lists of arrays), padded."""
    x, y = np.concatenate([np.ravel(a) for a in xs]), np.concatenate([np.ravel(a) for a in ys])
    dx, dy = max(np.ptp(x), 1e-9), max(np.ptp(y), 1e-9)
    ax.set_xlim(x.min() - pad * dx, x.max() + pad * dx)
    ax.set_ylim(y.min() - pad * dy, y.max() + pad * dy)


class WingViews:
    """The Design tab: the planform (x across, the span up), the front view (z across), the picked curve (its
    value across, the span up), the wing in 3D (sections, LE and TE lines; dashed grey the parameter file's)
    and the sections over their chords (dotted: the airfoil scaled, without the changes)."""

    def __init__(self, adapter, fig):
        self.a = adapter
        self.fig = fig
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.75, 1.0], height_ratios=[1.0, 1.15], wspace=0.3,
                              hspace=0.36, left=0.07, right=0.98, top=0.95, bottom=0.04)
        self.ax_plan = fig.add_subplot(gs[0, 0])
        self.ax_front = fig.add_subplot(gs[0, 1])
        self.ax_curve = fig.add_subplot(gs[0, 2])
        self.ax_3d = fig.add_subplot(gs[1, :2], projection="3d")
        self.ax_sec = fig.add_subplot(gs[1, 2])
        self.ax_3d.view_init(elev=24, azim=-60)
        self.box = None
        keep_fitted(fig, self.ax_3d, lambda: self.box)

    def update(self, pv, curve):
        for ax in (self.ax_plan, self.ax_front, self.ax_curve, self.ax_sec):
            ax.cla()
            ax.grid(True, color=GRID, lw=0.6)
        if isinstance(pv.get("sections"), str):
            self.ax_plan.text(0.02, 0.98, pv["sections"][:120], transform=self.ax_plan.transAxes, va="top",
                              color=RED, fontsize=8)
        else:
            self._outlines(pv, curve)
            self._sections(pv, curve)
        self._curve(pv, curve)
        self._wing(pv, curve)
        if pv["problems"]:
            self.ax_plan.text(0.02, 0.98, "\n".join(q[:80] for q in pv["problems"][:3]),
                              transform=self.ax_plan.transAxes, va="top", ha="left", color=RED, fontsize=8)

    def _outlines(self, pv, curve):
        """Planform (x across) and front view (z across), the span up: the design's outline (the sections'
        extent), its LE and TE, the pitch axis; dashed the parameter file's."""
        a, o = self.a, pv.get("orig")
        y = pv["y"]
        ax = self.ax_plan
        ax.plot(pv["x"][0], y, "-", color=RED if curve in ("chord", "sweep") else BLUE, lw=1.8, label="LE, TE")
        ax.plot(pv["x"][1], y, "-", color=RED if curve == "chord" else BLUE, lw=1.8)
        ref = pv["curves"]["sweep"] + a.params.ref_xc * pv["curves"]["chord"]
        ax.plot(ref, y, ":", color=INK2, lw=1.0, label=f"pitch axis ({a.params.ref_xc:g} c)")
        for sec in pv["sections"]:
            ax.plot([sec[:, 0].min(), sec[:, 0].max()], [sec[0, 1]] * 2, "-", color=GRID, lw=0.8)
        xs, ys = [pv["x"][0], pv["x"][1]], [y]
        if o is not None:
            ax.plot(o["x"][0], o["y"], "--", color=GREY, lw=1.0, label="parameter file")
            ax.plot(o["x"][1], o["y"], "--", color=GREY, lw=1.0)
            xs += [o["x"][0], o["x"][1]]
            ys.append(o["y"])
        _limits(ax, xs, ys)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y, span (mm)")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_title("planform", loc="left", fontsize=10)
        ax = self.ax_front
        ax.fill_betweenx(y, pv["z"][0], pv["z"][1], color="#d5e6f8", lw=0)
        ax.plot(pv["le"][:, 2], y, "-", color=RED if curve == "rake" else BLUE, lw=1.6, label="LE")
        ax.plot(pv["te"][:, 2], y, "--", color=RED if curve == "pitch" else BLUE, lw=1.0, label="TE")
        zs = [pv["z"][0], pv["z"][1]]
        if o is not None:
            ax.plot(o["le"][:, 2], o["y"], "--", color=GREY, lw=1.0)
            zs += [o["le"][:, 2]]
        _limits(ax, zs, ys)
        ax.set_xlabel("z (mm)")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_title("front view (z not to scale)", loc="left", fontsize=10)

    def _curve(self, pv, curve):
        """The picked curve, its value across and the span (mm) up: the file's rows and spline (at the
        file's span), the design and its control polygon (free filled, held hollow, pinned red)."""
        ax, a, d = self.ax_curve, self.a, pv["design"]
        eta_rows, v_rows = pv["rows"]
        span_f = a.params.span
        if curve in v_rows:
            ax.plot(v_rows[curve], eta_rows * span_f, ".", color=INK2, ms=3, label="parameter file")
        ax.plot(pv["file"][curve], pv["ee"] * span_f, "-", color=GREY, lw=1)
        ax.plot(pv["curves"][curve], pv["ee"] * d.span, "-", color=BLUE, lw=2.0, label="design")
        cu = d.curves[curve]
        p = d.ctrl(curve)
        y = p[:, 0] * d.span
        ax.plot(p[:, 1], y, "--", color=BLUE, lw=0.8)
        free = a.ctrl_free(curve)
        pinned = np.zeros(len(p), dtype=bool)
        if curve in WM.PINNED:
            pinned[0] = True
        ax.plot(p[free, 1], y[free], "o", color=BLUE, ms=6, label="control point: free")
        held = ~free & ~pinned
        ax.plot(p[held, 1], y[held], "o", color=BLUE, ms=6, mfc="white", label="held")
        if pinned.any():
            ax.plot(p[pinned, 1], y[pinned], "s", color=RED, ms=6, mfc="white", label="pinned (root LE)")
        if cu.segments == 2:
            labels = ("P1", f"P2=P3 (w {cu.w23:.2f})", None, "P4", f"P5=P6 (w {cu.w56:.2f})", None, "P7")
        else:
            labels = [f"C{k}" for k in range(len(p))]
        vals = np.concatenate([p[:, 1], pv["curves"][curve], pv["file"][curve]])
        lo, hi = float(vals.min()), float(vals.max())
        pad = max(0.08 * (hi - lo), 1e-3 if curve not in ("chord", "sweep", "rake", "pitch") else 0.5)
        ax.set_xlim(lo - pad, hi + pad)
        top = max(d.span, span_f)
        ax.set_ylim(-0.04 * top, 1.06 * top)
        for lab, (vk, yk) in zip(labels, zip(p[:, 1], y)):
            if lab:                          # near the right edge to the left, near the top below the point
                right, high = vk > hi - 0.15 * (hi - lo + 2 * pad), yk > 0.9 * top
                ax.annotate(lab, (vk, yk), textcoords="offset points", xytext=(-6 if right else 6, -6 if high else 4),
                            fontsize=8, color=BLUE, ha="right" if right else "left", va="top" if high else "bottom")
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.set_xlabel(f"{curve} ({WM.unit(curve)})")
        ax.set_ylabel("y, span (mm)")
        ax.legend(loc="best", fontsize=7)
        ax.set_title(f"curve {a.curve_label(curve)}" + (", 2 segments" if cu.segments == 2 else ""), loc="left",
                     fontsize=10)

    def _sections(self, pv, curve):
        """The sections at VIEW_FRACTIONS over their chords (x/c across, z/c up): solid as designed, dotted the
        airfoil scaled to t/c and f/c without the changes; a picked change: its point's Bezier polynomial."""
        ax = self.ax_sec
        x, (T, C), (T0, C0) = pv["shapes"]
        for i, (e, col) in enumerate(zip(pv["fractions"], SECTION_COLORS)):
            ax.plot(x, C[i] + 0.5 * T[i], "-", color=col, lw=1.3, label=f"eta {e:g}")
            ax.plot(x, C[i] - 0.5 * T[i], "-", color=col, lw=1.3)
            ax.plot(x, C0[i] + 0.5 * T0[i], ":", color=col, lw=0.8)
            ax.plot(x, C0[i] - 0.5 * T0[i], ":", color=col, lw=0.8)
        d = pv["design"]
        if curve not in WM.CURVES:
            k = int(curve[2:])
            b = WM.change_basis(x, d.changes)[:, k - 1]
            ax.plot(x, 0.25 * b * float(T.max()), "-", color=RED, lw=1.2, label=f"B{k},{d.changes + 1} (scaled)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x/c")
        ax.set_ylabel("z/c")
        ax.legend(loc="upper right", fontsize=6, ncol=2)
        ax.set_title("sections over the chord (dotted: no changes)", loc="left", fontsize=9)

    def _wing(self, pv, curve):
        """The wing in 3D (mm; x, z across, the span up): sections at VIEW_FRACTIONS, LE and TE lines, the
        parameter file's sections dashed."""
        ax = self.ax_3d
        elev, azim, roll = ax.elev, ax.azim, ax.roll
        ax.cla()
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.tick_params(labelsize=8)
        secs, pts = pv.get("sections"), []
        if isinstance(secs, str) or secs is None:
            ax.text2D(0.01, 0.97, str(secs)[:150], transform=ax.transAxes, color=RED, fontsize=8, va="top")
        else:
            for sec, col in zip(secs, SECTION_COLORS):
                ax.plot(sec[:, 0], sec[:, 2], sec[:, 1], color=col, lw=1.5)
                pts.append(sec)
                if curve == "pitch":
                    le, te = sec[0], sec[(len(sec) - 1) // 2]
                    ax.plot([le[0], te[0]], [le[2], te[2]], [le[1], te[1]], color=RED, lw=1.6)
            for key in ("le", "te"):
                q = pv[key]
                hot = curve in ("chord", "sweep", "rake") and (key == "le" or curve == "chord")
                ax.plot(q[:, 0], q[:, 2], q[:, 1], "-", color=RED if hot else INK2, lw=1.8 if hot else 1.0)
                pts.append(q)
        o = pv.get("orig")
        if o is not None:
            for sec in o["sections"]:
                ax.plot(sec[:, 0], sec[:, 2], sec[:, 1], "--", color=GREY, lw=0.7)
                pts.append(sec)
        if pts:
            p = np.vstack(pts)
            lo, hi = p.min(axis=0), p.max(axis=0)
            pad = 0.03 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[2], hi[2])
            ax.set_zlim(lo[1], hi[1])
            self.box = np.maximum(np.array([hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1]]), 1e-9)
            fit_box(ax, self.fig, self.box)
        from matplotlib.ticker import MaxNLocator
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_zlabel("y, span (mm)")
        what = {"chord": "LE and TE", "sweep": "LE", "rake": "LE", "pitch": "chord lines"}.get(curve)
        ax.set_title(f"the {ROLES[self.a.role]['name'].lower()} as the CAD builds it: sections at eta "
                     + ", ".join(f"{e:g}" for e in pv["fractions"])
                     + (f"; red: the {what}" if what else "") + "\ndashed: the parameter file's; drag to turn",
                     fontsize=9)


class WingSkeletonView:
    """The 3D sections tab: the XCAD file's sections (x, z across, the span up), the root (it meets the hull,
    closed by a flat face) red, the flat tip dark."""

    def __init__(self, adapter, fig):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        self.a = adapter
        self.fig = fig
        self.ax = ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.col = {k: Line3DCollection([np.zeros((2, 3))], colors=c, linewidths=w)
                    for k, c, w in (("sections", BLUE, 0.8), ("root", RED, 2.0), ("tip", INK2, 1.6))}
        for c in self.col.values():
            ax.add_collection3d(c)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_zlabel("y, span (mm)")
        ax.view_init(elev=22, azim=-55)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(3))
        self.text = ax.text2D(0.02, 0.99, "", transform=ax.transAxes, fontsize=9, va="top")
        self.box = None
        keep_fitted(fig, ax, lambda: self.box)

    def update(self, pv):
        sk = pv.get("skeleton")
        if sk is None:
            for c in self.col.values():
                c.set_segments([np.zeros((2, 3))])
            self.text.set_text("no sections: " + (pv["problems"][0] if pv["problems"] else ""))
            return
        q = sk[:, :, [0, 2, 1]]
        segs = {"sections": list(q[1:-1]), "root": [q[0]], "tip": [q[-1]]}
        for k, c in self.col.items():
            c.set_segments(segs[k])
        p = q.reshape(-1, 3)
        lo, hi = p.min(axis=0), p.max(axis=0)
        ax = self.ax
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        self.box = np.maximum(hi - lo, 1e-9)
        fit_box(ax, self.fig, self.box)
        d = pv["design"]
        msg = (f"{len(sk)} sections, {sk.shape[1]} points each (the XCAD file); span {d.span:.1f} mm; the root (red) "
               f"and the flat tip (dark) are closed by planar faces in the loft")
        if pv["problems"]:
            msg += "\nproblem: " + pv["problems"][0][:100]
        self.text.set_text(msg)
