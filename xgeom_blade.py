"""
xgeom_blade.py

The propeller blade in the XGeom design tool (xgeom_tool.py): a blade parameter
file (blade_modify.py) as the tool's variables, its live views and the CAD
build. The settings are blade_control.py's.

A blade always starts from a parameter file (the radial table and D, Z, the
root r/R, the hub height, the section table). Its variables: the blade radius,
hub radius and hub height (m), and each distribution (pitch, chord, thickness,
camber, skew, rake) as a Bezier curve over r/R fitted to the file, of 1 or 2
segments (set_segments):
  1 segment: control points C0 at the root .. Cm at the tip; the inner
    positions are fractions d of the span left, the values are in the
    distribution's units.
  2 segments (para_control_bez_updated, x_blade): P1 (root) .. P7 (tip), P2 =
    P3 and P5 = P6 at the value of P4, so zero slope at P4; P4's position u4,
    the fractions d1, d2, the weights w23, w56 and the values v1, v4, v7.
Each is a design variable or held. A change of either radius stretches the
curves over the new span. A new form (set_segments) or order (set_order) is
fitted to the parameter file while none of the curve's variables has changed,
else to the curve's present shape. load_params reads another parameter file;
save_params writes the design as one. set_sections sets the sections of the
XCAD points file and the 3D sections tab (blade_modify.section_stations).
set_blades sets the number of blades Z (the parameter file's by default): the
clearance check, the propeller views, the CAD's hub sector, the parameter file
the design is written as and the set-up follow it (it is a setting, not in the
design vector).

The adapter has the methods of xgeom_rudder.RudderAdapter that the tool uses.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time

import numpy as np

import blade_modify as BM
from xgeom_common import BLUE, GREY, GRID, INK2, RED, SECTION_COLORS, Row, fit_box, keep_fitted

MAX_ORDER = 12
VIEW_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)   # sections in the Design tab's 3D view, fractions of the span
GLOBAL_ROWS = (("radius", "radius R"), ("hub_radius", "hub radius"), ("hub_height", "hub height"))
TWO_SEGMENT_ROWS = (("v1", "P1 value (root)"), ("d1", "P2=P3 d1"), ("w23", "P2=P3 weight"), ("u4", "P4 position"),
                    ("v4", "P4 value"), ("d2", "P5=P6 d2"), ("w56", "P5=P6 weight"), ("v7", "P7 value (tip)"))
TWO_SEGMENT_POINT = {"v1": 0, "d1": 1, "w23": 1, "u4": 3, "v4": 3, "d2": 4, "w56": 4, "v7": 6}   # control point


class BladeAdapter:
    joins = False                 # the curves have no join to switch
    segment_choice = True         # each curve 1 or 2 segments
    build_label = "CAD"

    def __init__(self, params=None):
        import blade_control as BC
        s = BC.settings
        self.settings = {"clearance_mm": float(s.clearance_mm), "sections": int(s.sections),
                         "tip_band": float(s.tip_band), "tip_sections": int(s.tip_sections), "hub": bool(s.hub),
                         "plot": bool(s.plot), "orders": dict(BC.ORDERS), "segments": dict(BC.SEGMENTS),
                         "free": tuple(BC.FREE)}
        self.out_dir = BC.OUT_DIR
        self.load_params(params or BC.PARAMS)

    # ---------------------------------------------------------------- the file
    def load_params(self, path):
        """Start from a parameter file: its distributions fitted (blade_control's
        segments and orders), the variables of blade_control.FREE free."""
        params = BM.read_params(path)
        design, _ = BM.fit_design(params, self.settings["orders"], self.settings["segments"])
        self.params, self.design, self.meta = params, design, {}
        self.file_blades = params.blades            # the file's Z (set_blades may change the one in use)
        self.as_fitted = set(BM.CURVES)             # curves still as fitted to the file (no variable changed)
        self._rebuild()
        self._original = None

    def save_params(self, path):
        """Write the design as a parameter file (read_params and load_params read it)."""
        BM.write_params(path, BM.design_params(self.design, self.params),
                        notes=(f"written by the design tool from {os.path.basename(self.params.path)}, "
                               f"{time.strftime('%d %b %Y %H:%M')}",))
        return path

    @property
    def title(self):
        return f"Propeller blade ({os.path.basename(self.params.path)})"

    @property
    def names(self):
        return BM.CURVES

    # ---------------------------------------------------------------- state
    def _rebuild(self):
        """The space for the present curves, keeping the bounds and free flags
        of every slot that is still there."""
        full = BM.BladeSpace(self.params, self.design)
        free0 = self.settings["free"]
        self.meta = {n: self.meta.get(n, {"lo": float(lo), "hi": float(hi), "free": n.split(".")[0] in free0})
                     for n, (lo, hi) in zip(full.all_names, full.all_bounds)}
        self.space = BM.BladeSpace(self.params, self.design,
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
        if name in BM.GLOBAL_SLOTS and lo <= 0.0:
            raise ValueError("a radius or height is positive")
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

    def _refit(self, curve, segments, order):
        """The curve in a new form, its slots renewed (free if any of its slots
        was): fitted to the parameter file while it is as fitted to it (no
        variable of it changed since the file was read), else to its present
        shape. Returns what was done, for the status bar."""
        was_free = any(m["free"] for n, m in self.meta.items() if n.startswith(curve + "."))
        rr = np.linspace(self.design.r0, self.design.r1, 401)
        old = self.design.at(curve, rr)
        if curve in self.as_fitted:
            self.design = BM.refit_to_file(self.params, self.design, curve, segments, order)
        else:
            self.design = BM.refit_curve(self.design, curve, order=order, segments=segments)
        for n in [n for n in self.meta if n.startswith(curve + ".")]:
            del self.meta[n]
        self._rebuild()
        for n, m in self.meta.items():
            if n.startswith(curve + "."):
                m["free"] = was_free
        if curve in self.as_fitted:
            rel = BM.fit_report(self.params, self.design)[curve][1]
            return f"fitted to the parameter file (largest difference {100 * rel:.2f} % of its range)"
        rel = float(np.abs(self.design.at(curve, rr) - old).max() / max(np.ptp(old), 1e-300))
        return f"fitted to the curve as it was (largest change {100 * rel:.2f} % of its range)"

    def set_order(self, curve, order):
        """Number of control points of a 1-segment curve = order + 1; the curve
        is fitted again (_refit)."""
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
        """The curve as 1 segment (blade_control's order) or 2 segments, fitted
        again (_refit)."""
        n = int(n)
        if n not in (1, 2):
            raise ValueError("a curve has 1 or 2 segments")
        if n == self.segments(curve):
            return "unchanged"
        return self._refit(curve, n, self.settings["orders"].get(curve, BM.DEFAULT_ORDER))

    # ---------------------------------------------------------------- rows
    def global_rows(self):
        """Rows for the blade radius, hub radius and hub height (m)."""
        return [Row("var", f"{label} (m)", name, 4) for name, label in GLOBAL_ROWS]

    def curve_rows(self, curve):
        """Rows for the variables of one curve, from the root to the tip."""
        cu = self.design.curves[curve]
        digits = 3 if curve == "skew" else 5
        if cu.segments == 2:
            return [Row("var", label, f"{curve}.{k}", digits if k[0] == "v" else 4) for k, label in TWO_SEGMENT_ROWS]
        n = cu.order
        rows = []
        for k in range(n + 1):
            if 0 < k < n:
                rows.append(Row("var", f"C{k} d", f"{curve}.d{k}", 4))
            rows.append(Row("var", f"C{k} value" + (" (root)" if k == 0 else " (tip)" if k == n else ""),
                            f"{curve}.v{k}", digits))
        return rows

    def row_info(self, name):
        """Where a variable sits: its control point's r/R, the diameter, the
        root's r/R, the hub's ends."""
        d = self.design
        if name == "radius":
            return f"D {d.diameter:.3f}"
        if name == "hub_radius":
            return f"r/R {d.r0:.3f}"
        if name == "hub_height":
            return f"x ±{0.5 * d.hub_height:.3f}"
        curve, key = name.split(".")
        k = TWO_SEGMENT_POINT[key] if self.segments(curve) == 2 else int(key[1:])
        return f"r/R {d.ctrl(curve)[k, 0]:.3f}"

    def ctrl_free(self, curve):
        """For each control point of a curve: is any variable that moves it free?"""
        cu = self.design.curves[curve]
        free = {k: self.meta[f"{curve}.{k}"]["free"] for k in cu.keys()}
        if cu.segments == 2:
            groups = (("v1",), ("d1", "w23"), ("d1", "w23"), ("u4", "v4"), ("d2", "w56"), ("d2", "w56"), ("v7",))
            return np.array([any(free[k] for k in g) for g in groups])
        n = cu.order
        return np.array([free[f"v{k}"] or (0 < k < n and free[f"d{k}"]) for k in range(n + 1)])

    def n_free(self, curve=None):
        names = [n for n in self.space.all_names if curve is None or n.startswith(curve + ".")]
        return sum(1 for n in names if self.meta[n]["free"]), len(names)

    def summary(self):
        n, m = self.n_free()
        d = self.design
        return (f"{n} of {m} variables free; R {d.radius:.4g} m, hub radius {d.hub_radius:.4g} m (r/R {d.r0:.3f}), "
                f"hub height {d.hub_height:.4g} m, {self.params.blades} blades")

    def describe(self, curve):
        e, rel = BM.fit_report(self.params, self.design)[curve]
        diff = f"largest difference from the parameter file {e:.3g} ({100 * rel:.2f} % of its range)"
        if self.segments(curve) == 2:
            return (f"{BM.DESCRIPTIONS[curve]} against r/R, 2 rational cubic Bezier segments P1-P4 and P4-P7 "
                    f"(x_blade's form): P2 = P3 and P5 = P6 at P4's value, so zero slope at P4; d1 and d2 place "
                    f"them between P4 and the root and the tip, w23 and w56 are their weights; {diff}")
        return f"{BM.DESCRIPTIONS[curve]} against r/R, a Bezier curve from the root (C0) to the tip; {diff}"

    def set_blades(self, z):
        """The number of blades Z (blade_modify.BLADES: 2 .. 12); refused (ValueError) otherwise."""
        self.params = BM.with_blades(self.params, z)
        self._rebuild()

    # ---------------------------------------------------------------- sections
    def stations(self):
        """r/R of the XCAD sections (blade_modify.section_stations)."""
        return BM.settings_stations(self.design.r0, self.settings)

    def set_sections(self, n, tip_band, tip_n):
        """n sections in the XCAD points file, the last tip_n from r/R tip_band
        to the tip; refused (ValueError) if they do not fit the blade."""
        BM.section_stations(self.design.r0, n, tip_band, tip_n)
        self.settings.update(sections=int(n), tip_band=float(tip_band), tip_sections=int(tip_n))

    def sections_info(self):
        """The steps between the sections, r/R and mm."""
        try:
            st = self.stations()
        except ValueError as exc:
            return str(exc)
        R = 1000.0 * self.design.radius                     # tip radius, mm
        s = self.settings
        step = np.diff(st)
        tip = step[len(st) - s["tip_sections"]:]
        return (f"even steps of {step[0]:.4f} r/R ({step[0] * R:.1f} mm) from the root to r/R {s['tip_band']:g}, "
                f"then closer together: {tip.max() * R:.1f} mm down to {tip.min() * R:.2f} mm at the tip. The "
                f"clearance check keeps x_blade_new's {len(BM.r_stations(self.design.r0))} stations.")

    # ---------------------------------------------------------------- preview
    def _original_views(self):
        """The parameter file's own blade (its splines), drawn dashed: cached
        until another file is loaded."""
        if self._original is None:
            p = self.params
            props = tuple(p.spline(c) for c in ("camber", "pitch", "chord", "thickness", "skew", "rake"))
            from blade_surface_new import BladeSurface
            bs = BladeSurface(*props, d=p.diameter, r_root=p.root_r, airfoil_path=p.section_path())
            self._original = {"sections": self._sections(bs, p.root_r, p.r[-1])}
        return self._original

    def _view_radii(self, r0=None, r1=None):
        d = self.design
        r0, r1 = (d.r0 if r0 is None else r0), (d.r1 if r1 is None else r1)
        return r0 + np.asarray(VIEW_FRACTIONS) * (r1 - r0)

    def _sections(self, bs, r0=None, r1=None, n=161):
        """Closed sections of a BladeSurface at the view radii, mm: (k, n, 3)."""
        xi = np.linspace(0.0, 1.0, n)
        eta = bs.eta_of_r(np.minimum(self._view_radii(r0, r1), bs.r_tip))
        return np.stack([bs.b(xi, np.full_like(xi, e)) for e in eta]) * 1000.0

    def preview(self, skeleton=False):
        """What the live views draw: the design, its problems, the curves, the
        file's rows and splines, the blade (BladeSurface, the CAD shape): sections
        at the view radii, LE, TE, mid-chord and tip lines, the hub; with skeleton
        the blade's sections as XCAD gets them (para.py's at stations())."""
        t0 = time.time()
        d, p = self.design, self.params
        rr = np.linspace(d.r0, d.r1, 200)
        file_rr = np.linspace(p.root_r, p.r[-1], 200)
        out = {"design": d, "rr": rr, "curves": {c: d.prop(c)(rr) for c in BM.CURVES}, "file_rr": file_rr,
               "file": {c: p.spline(c)(file_rr) for c in BM.CURVES}, "rows": p.rows(), "radii": self._view_radii(),
               "hub": (1000.0 * d.hub_radius, 1000.0 * d.hub_height)}
        try:
            pts, _ = BM.blade_points(d, p)
            out["clearance_mm"] = BM.clearance(pts, p.blades) * 1000.0
            problems = BM.design_problems(d, p, self.settings["clearance_mm"], points=pts,
                                          gap_mm=out["clearance_mm"], hub=self.settings["hub"])
        except Exception as exc:
            problems = [f"{type(exc).__name__}: {exc}"]
        try:
            bs = BM.blade_surface(d, p)
            eta = np.linspace(bs.eta_min, 1.0, 120)
            xi_m = _mid_chord_xi(bs)
            mid = 0.5 * (bs.b(np.full_like(eta, xi_m), eta) + bs.b(np.full_like(eta, 1.0 - xi_m), eta))
            out.update(sections=self._sections(bs), le=bs.b(np.full_like(eta, 0.5), eta) * 1000.0,
                       te=bs.b(np.zeros_like(eta), eta) * 1000.0, mid=mid * 1000.0, tip=bs.tip_line() * 1000.0)
        except Exception as exc:
            out["sections"] = f"no blade surface: {type(exc).__name__}: {exc}"
        try:
            out["orig_sections"] = self._original_views()["sections"]
        except Exception:
            out["orig_sections"] = None
        out["problems"] = problems
        if skeleton:                                          # the XCAD sections
            try:
                out["skeleton"] = BM.blade_points(d, p, self.stations())[0].reshape(-1, 53, 3) * 1000.0
            except Exception as exc:
                out["skeleton"] = None
                problems.append(f"no sections: {type(exc).__name__}: {exc}")
        out["seconds"] = time.time() - t0
        return out

    def views(self, fig):
        return BladeViews(self, fig)

    def skeleton_view(self, fig):
        return PropellerView(self, fig)

    # ---------------------------------------------------------------- output
    def free_space(self):
        """The design space with the unticked variables held fixed: the one
        an optimiser gets."""
        fixed = {n: self.values[n] for n in self.space.all_names if not self.meta[n]["free"]}
        return BM.BladeSpace(self.params, self.design,
                             bounds={n: (m["lo"], m["hi"]) for n, m in self.meta.items()}, fixed=fixed)

    def space_record(self):
        """The set-up as plain data (blade_modify.load_blade_space reads it back)."""
        return BM.space_record(self.free_space(), self.design, self.params, self._case_settings())

    CASE_KEYS = ("clearance_mm", "sections", "tip_band", "tip_sections", "hub")

    def _case_settings(self):
        return {k: self.settings[k] for k in self.CASE_KEYS}

    def save_space(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.space_record(), fh, indent=1)
        return path

    def load_space(self, path):
        """Start from a saved set-up (<case>_design_space.json); its parameter
        file is read again."""
        with open(path) as fh:
            rec = json.load(fh)
        if rec.get("geometry") != "blade":
            raise ValueError(f"{path} is not a blade design space")
        params = BM.read_params(rec["params_file"])
        self.file_blades = params.blades
        if rec.get("globals", {}).get("blades") is not None:
            params = BM.with_blades(params, rec["globals"]["blades"])
        space, design = BM.BladeSpace.from_dict(params, rec["space"])
        self.params, self.design, self._original = params, design, None
        self.as_fitted = set()                      # a saved design: refits keep its curves' shapes
        self.meta = {q["name"]: {"lo": q["lo"], "hi": q["hi"], "free": q["free"]} for q in rec["space"]["slots"]}
        for k in self.CASE_KEYS:
            if k in rec.get("settings", {}):
                self.settings[k] = rec["settings"][k]
        self._rebuild()

    def build_task(self, case):
        """A function that writes the case as the design is now (blade_modify.
        write_case: the parameter file, the XCAD points, the design and the
        set-up JSON, the check plot) and then the CAD (<case>.iges, <case>.step).
        The state is copied here, so it can run off the GUI thread."""
        design, params, space, st = self.design.copy(), self.params, self.free_space(), self._case_settings()
        out_dir, plot = self.out_dir, self.settings["plot"]

        def run():
            res = BM.write_case(out_dir, case, design, params, space, cad=False, settings=st, plot=plot)
            out = {"case": res, "view": None, "error": False,
                   "report": [f"{case}: {res['sections']} sections, blades {res['clearance_mm']:.1f} mm apart",
                              "  written: " + ", ".join(res["paths"].values())]}
            files = f"{case}_params.dat, {case}_xcad_m.dat"
            if res["problems"]:
                out.update(message=f"{files}; no CAD: {res['problems'][0]}", error=True)
                return out
            if importlib.util.find_spec("OCC") is None:
                out.update(message=f"{files}; no CAD: needs pythonocc-core", error=True)
                return out
            cad = BM.write_cad(out_dir, case, design, params, st)
            res["paths"].update(cad["cad_paths"])
            out["report"].append(f"  CAD: grids {cad['grid_seconds']:.1f} s, faces and IGES {cad['cad_seconds']:.1f} s:"
                                 f" {cad['cad_paths']['iges']}, {cad['cad_paths']['step']}")
            out.update(view=cad["cad_paths"]["step"],
                       message=f"{files}, {os.path.basename(cad['cad_paths']['iges'])} and {case}.step; "
                               f"OCC viewer shows it")
            return out
        return run


def _mid_chord_xi(bs):
    """xi of the mid-chord station (s = 1/2) on the xi < 1/2 side."""
    s_end = bs.canon.s_end
    v = np.arccos(1.0 - 2.0 * 0.5 / s_end) / np.pi
    return 0.5 * (1.0 - v)


class BladeViews:
    """The Design tab: the expanded outline and the selected curve (both with
    the radius on the x axis), and in 3D the blade (sections, edges), the
    parameter file's blade dashed."""

    def __init__(self, adapter, fig):
        self.a = adapter
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.3], wspace=0.22, hspace=0.38,
                              left=0.09, right=0.98, top=0.95, bottom=0.03)
        self.ax_out = fig.add_subplot(gs[0, 0])
        self.ax_curve = fig.add_subplot(gs[0, 1])
        self.ax_sec = fig.add_subplot(gs[1, :], projection="3d")
        self.ax_sec.view_init(elev=20, azim=-35)
        self.box = None
        keep_fitted(fig, self.ax_sec, lambda: self.box)

    def update(self, pv, curve):
        self._outline(pv)
        self._curve(pv, curve)
        self._blade(pv, curve)

    def _outline(self, pv):
        """Expanded outline: each section's chord laid flat about its mid-chord,
        which the skew moves along the circle (arc r * skew); the file's blade at
        its own radius, the design's at its own."""
        ax, p, d = self.ax_out, self.a.params, pv["design"]
        ax.cla()
        ax.grid(True, color=GRID, lw=0.6)
        for src, rr, R, style, col, lab in (("file", pv["file_rr"], 500.0 * p.diameter, "--", GREY, "parameter file"),
                                            ("curves", pv["rr"], 1000.0 * d.radius, "-", BLUE, "design")):
            v = pv[src]
            r = rr * R                                       # mm
            mid = r * np.deg2rad(v["skew"])
            half = v["chord"] * R                            # (c/D) D / 2
            ax.plot(r, mid - half, style, color=col, lw=1.8 if src == "curves" else 1.0, label=lab)
            ax.plot(r, mid + half, style, color=col, lw=1.8 if src == "curves" else 1.0)
            if src == "curves":
                ax.plot(r, mid, ":", color=col, lw=1.0, label="mid-chord (skew)")
                for rk in pv["radii"]:
                    i = int(np.argmin(np.abs(rr - rk)))
                    ax.plot([r[i]] * 2, [mid[i] - half[i], mid[i] + half[i]], "-", color=GRID, lw=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("r (mm)")
        ax.set_ylabel("arc along the circle (mm)")
        ax.legend(loc="best", fontsize=7)
        ax.set_title("expanded outline", loc="left", fontsize=10)
        if pv["problems"]:
            ax.text(0.02, 0.98, "\n".join(q[:90] for q in pv["problems"][:3]), transform=ax.transAxes,
                    va="top", ha="left", color=RED, fontsize=8)

    def _curve(self, pv, curve):
        """The selected distribution against r/R: the file's rows and spline,
        the design and its control polygon (free filled, held hollow; 2
        segments: P1 .. P7 with the weights at P2 = P3 and P5 = P6)."""
        ax, a = self.ax_curve, self.a
        ax.cla()
        ax.grid(True, color=GRID, lw=0.6)
        r_rows, v_rows = pv["rows"]
        ax.plot(r_rows, v_rows[curve], ".", color=INK2, ms=3, label="parameter file")
        ax.plot(pv["file_rr"], pv["file"][curve], "-", color=GREY, lw=1)
        ax.plot(pv["rr"], pv["curves"][curve], "-", color=BLUE, lw=2.0, label="design")
        d = pv["design"]
        cu = d.curves[curve]
        p = d.ctrl(curve)
        ax.plot(p[:, 0], p[:, 1], "--", color=BLUE, lw=0.8)
        free = a.ctrl_free(curve)
        ax.plot(p[free, 0], p[free, 1], "o", color=BLUE, ms=6, label="control point: free")
        ax.plot(p[~free, 0], p[~free, 1], "o", color=BLUE, ms=6, mfc="white", label="held")
        if cu.segments == 2:
            labels = ("P1", f"P2=P3 (w {cu.w23:.2f})", None, "P4", f"P5=P6 (w {cu.w56:.2f})", None, "P7")
        else:
            labels = [f"C{k}" for k in range(len(p))]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for lab, (rk, vk) in zip(labels, p):
            if lab:                          # near the right edge to the left, near the top below the point
                right, high = rk > x0 + 0.8 * (x1 - x0), vk > y0 + 0.85 * (y1 - y0)
                ax.annotate(lab, (rk, vk), textcoords="offset points", xytext=(-6 if right else 6, -6 if high else 4),
                            fontsize=8, color=BLUE, ha="right" if right else "left", va="top" if high else "bottom")
        ax.set_xlabel("r/R")
        ax.set_ylabel(f"{curve} ({BM.UNITS[curve]})")
        ax.legend(loc="best", fontsize=7)
        ax.set_title(f"curve {curve}: {BM.DESCRIPTIONS[curve]}" + (", 2 segments" if cu.segments == 2 else ""),
                     loc="left", fontsize=10)

    def _blade(self, pv, curve):
        """The blade in 3D (mm, x along the shaft): sections at the view radii,
        LE, TE, tip and mid-chord lines; red: what the selected curve moves most
        (chord: LE and TE, pitch: the nose-tail lines, skew and rake: the
        mid-chord line); dashed grey: the parameter file's blade."""
        ax = self.ax_sec
        elev, azim, roll = ax.elev, ax.azim, ax.roll
        ax.cla()
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.tick_params(labelsize=8)
        secs = pv["sections"]
        pts = []
        if isinstance(secs, str):
            ax.text2D(0.01, 0.97, secs[:150], transform=ax.transAxes, color=RED, fontsize=8, va="top")
        else:
            for sec, col in zip(secs, SECTION_COLORS):
                ax.plot(sec[:, 0], sec[:, 1], sec[:, 2], color=col, lw=1.5)
                pts.append(sec)
                if curve == "pitch":
                    n = len(sec)
                    le, te = sec[(n - 1) // 2], sec[0]
                    ax.plot([le[0], te[0]], [le[1], te[1]], [le[2], te[2]], color=RED, lw=1.6)
            for key, col in (("le", RED if curve == "chord" else INK2), ("te", RED if curve == "chord" else INK2),
                             ("tip", INK2), ("mid", RED if curve in ("skew", "rake") else GREY)):
                q = pv[key]
                ax.plot(q[:, 0], q[:, 1], q[:, 2], ":" if key == "mid" and col == GREY else "-", color=col,
                        lw=1.8 if col == RED else 1.0)
                pts.append(q)
        if pv.get("orig_sections") is not None:
            for sec in pv["orig_sections"]:
                ax.plot(sec[:, 0], sec[:, 1], sec[:, 2], "--", color=GREY, lw=0.7)
                pts.append(sec)
        if pts:
            p = np.vstack(pts)
            lo, hi = p.min(axis=0), p.max(axis=0)
            pad = 0.03 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            self.box = np.maximum(hi - lo, 1e-9)
            fit_box(ax, self.fig, self.box)
        from matplotlib.ticker import MaxNLocator
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel("x, shaft (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        what = {"chord": "LE and TE", "pitch": "nose-tail lines", "skew": "mid-chord line",
                "rake": "mid-chord line"}.get(curve)
        gap = pv.get("clearance_mm")
        ax.set_title("the blade as the CAD builds it: sections at r/R "
                     + ", ".join(f"{r:.2f}" for r in pv["radii"]) + ", LE, TE, tip"
                     + (f"\nred: {what} ({curve}); " if what else "\n")
                     + "dashed: the parameter file's blade"
                     + (f"; blades {gap:.1f} mm apart" if gap is not None else "") + "; drag to turn",
                     fontsize=9)


class PropellerView:
    """The 3D tab: all Z blades as their sections (para.py's, as XCAD gets
    them), the designed blade blue, the others grey, and the hub (its radius
    and height, centred at x = 0)."""

    def __init__(self, adapter, fig):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        self.a = adapter
        self.fig = fig
        self.ax = ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.col = {k: Line3DCollection([np.zeros((2, 3))], colors=c, linewidths=w)
                    for k, c, w in (("others", GREY, 0.5), ("blade", BLUE, 0.8), ("hub", INK2, 1.0))}
        for c in self.col.values():
            ax.add_collection3d(c)
        ax.set_xlabel("x, shaft (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.view_init(elev=15, azim=-30)
        self.text = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, fontsize=9, va="top")
        self.box = None
        keep_fitted(fig, ax, lambda: self.box)

    def update(self, pv):
        sk = pv.get("skeleton")
        p = self.a.params
        if sk is None:
            for c in self.col.values():
                c.set_segments([np.zeros((2, 3))])
            self.text.set_text("no sections: " + (pv["problems"][0] if pv["problems"] else ""))
            return
        from rot_axis import rot_axis
        segs = {"blade": list(sk), "others": [], "hub": []}
        for k in range(1, p.blades):
            turn = rot_axis(np.array([1.0, 0.0, 0.0]), 2.0 * np.pi * k / p.blades)
            segs["others"] += [s @ turn for s in sk]
        hr, hh = pv["hub"]
        t = np.linspace(0.0, 2.0 * np.pi, 121)
        for x in (-0.5 * hh, 0.5 * hh):
            segs["hub"].append(np.column_stack((np.full_like(t, x), hr * np.sin(t), hr * np.cos(t))))
        for a in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            segs["hub"].append(np.array([[-0.5 * hh, hr * np.sin(a), hr * np.cos(a)],
                                         [0.5 * hh, hr * np.sin(a), hr * np.cos(a)]]))
        for k, c in self.col.items():
            c.set_segments(segs[k])
        allp = np.vstack([s for v in segs.values() for s in v])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        ax = self.ax
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        self.box = np.maximum(hi - lo, 1e-9)
        fit_box(ax, self.fig, self.box)
        d = pv["design"]
        msg = (f"{p.blades} blades, {len(sk)} sections each (the designed blade blue), D {d.diameter:.4g} m; hub "
               f"radius {hr:.0f} mm, height {hh:.0f} mm; blades {pv.get('clearance_mm', float('nan')):.1f} mm apart")
        if pv["problems"]:
            msg += "\nproblem: " + pv["problems"][0][:100]
        self.text.set_text(msg)
