"""
xgeom_rudder.py

The rudder in the XGeom design tool (xgeom_tool.py): the design space of
rudder_modify.py as the tool's variables, its live views and the XCAD build.
The files and the settings are modify_control.py's.

The shape below H: the x of the LE and every coordinate of the inner section
control points (P1x, P1z, P2x, ...) each follow a spanwise Bezier curve, the
value against the height; the section is a Bezier half-section of degree n
(P0 = (0, 0) the LE, Pn = (1, 0) the sharp TE). Adding a section control point
raises n by one and adds that point's two curves. P1x is held at 0 by default
(its slots unticked), which keeps the LE round.

An adapter gives the tool:
    title, names, degree                      what there is to edit
    get / bounds / is_free / set / set_bounds / set_free        one variable
    set_order / set_tangent / set_degree      the shape of the space
    curve_rows(curve)                         the rows of one curve (see Row)
    preview()                                 everything the live views draw
    views(fig) / skeleton_view(fig)           the drawings
    xcad_task(case)                           the XCAD file, the case files and the solid
    load_space(path)                          a design space written by an XCAD run
Blades, wings and hulls get adapters of their own with the same methods.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np

import bezier_section as BS
import rudder_modify as RM

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a9994"
MAX_DEGREE = 12
MAX_ORDER = 12
SKELETON_UNCHANGED = 90          # original loops drawn above H, at most (the tip and cap loops always)
N_VIEW = 5                       # sections drawn in the Design tab's 3D view, from the root to H
Z_STRETCH = 3.0                  # the 3D view stretches the thickness by this, so the polygons show
SECTION_COLORS = ("#1f4e9a", "#2a78d6", "#3fa0c8", "#1baf7a", "#0f7a55", "#0b5e43", "#08452f")


@dataclass
class Row:
    """One line of a curve's rows: a design variable (kind 'var') or a pinned
    control-point value shown for information (kind 'pinned')."""
    kind: str
    label: str
    name: str = ""                 # slot name for 'var'
    digits: int = 4
    text: str = ""                 # for 'pinned'


class RudderAdapter:

    def __init__(self):
        import modify_control as MC
        s = MC.settings
        self.files = {"corner_points": MC.CORNER_POINTS, "section_table": MC.SECTION_TABLE,
                      "xcad_original": MC.XCAD_ORIGINAL, "stack": MC.STACK}
        self.out_dir = MC.OUT_DIR
        self.orig = RM.fit_original(MC.CORNER_POINTS, MC.SECTION_TABLE, None, MC.FIT_ORDERS,
                                    free_heights=s.fit_free_heights, degree=getattr(MC, "DEGREE", RM.BASE_DEGREE))
        z_top = self.orig.z_top
        self.settings = {"z_top": z_top, "h_min": float(s.h_min), "h_max": min(float(s.h_max), z_top),
                         "n_sections": int(s.n_sections), "num_pts": int(s.num_pts),
                         "te_radius_mm": float(s.te_radius_mm), "gap_mm": s.gap_mm, "xcad_units": s.xcad_units,
                         "plot": bool(s.plot), "fit_orders": dict(MC.FIT_ORDERS),
                         "fit_free_heights": bool(s.fit_free_heights)}
        self.title = "Rudder, modified below H"
        self.stack = RM.read_stack(MC.STACK)
        self.loops0 = [RM.xcad_to_working(pts, "m") for _, pts in RM.read_xcad(MC.XCAD_ORIGINAL)]
        H = min(max(float(MC.H), self.settings["h_min"]), self.settings["h_max"])
        self.orders = {c: int(MC.ORDERS.get(c, RM.DEFAULT_ORDER)) for c in self.orig.names}
        self.join = {c: MC.JOIN for c in self.orig.names}
        self.design = RM.baseline_design(self.orig, H, self.orders, self.join)
        self.meta = {}
        self._rebuild()

    # ---------------------------------------------------------------- state
    @property
    def names(self):
        return self.orig.names

    @property
    def degree(self):
        return self.orig.degree

    def _rebuild(self):
        """The space for the present degree, orders and joins, keeping the
        bounds and free flags of every slot that is still there. New slots of
        P1x start held (unticked): P1x = 0 keeps the LE round."""
        h = dict(h_min=self.settings["h_min"], h_max=self.settings["h_max"])
        full = RM.DesignSpace(self.orig, self.orders, self.join, **h)
        self.meta = {n: self.meta.get(n, {"lo": float(lo), "hi": float(hi), "free": not n.startswith("P1x.")})
                     for n, (lo, hi) in zip(full.all_names, full.all_bounds)}
        self.meta["H"].update(lo=h["h_min"], hi=h["h_max"])
        self.space = RM.DesignSpace(self.orig, self.orders, self.join,
                                    bounds={n: (m["lo"], m["hi"]) for n, m in self.meta.items()}, **h)
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
        return value

    def set_bounds(self, name, lo, hi):
        lo, hi = float(lo), float(hi)
        if name == "H":
            top = self.settings["z_top"]
            if not 0.0 < lo < hi <= top + 1e-9:
                raise ValueError(f"H bounds must satisfy 0 < low < high <= {top:.4f} mm (the top horizontal section)")
            self.settings["h_min"], self.settings["h_max"] = lo, min(hi, top)
        elif not lo < hi:
            raise ValueError("the lower bound must be below the upper one")
        if re.search(r"\.d\d+$", name) and not 0.0 <= lo < hi <= 1.0:
            raise ValueError("a height fraction d lies in [0, 1]")
        self.meta[name].update(lo=lo, hi=hi)
        self._rebuild()
        v = self.values[name]
        if not lo <= v <= hi:
            self.set(name, v)                       # clamps into the new bounds
        return lo, hi

    def set_free(self, name, free):
        self.meta[name]["free"] = bool(free)

    def set_order(self, curve, order):
        """Number of control points of a curve = order + 1; the curve keeps
        its shape as closely as that order allows."""
        order = int(order)
        if not 1 <= order <= MAX_ORDER:
            raise ValueError(f"the order must lie in 1 .. {MAX_ORDER}")
        self.design = RM.refit_curve(self.design, self.orig, curve, order)
        self.orders[curve] = order
        self._drop_meta(curve)
        self._rebuild()

    def set_tangent(self, curve, on):
        """C1 on the original's tangent at H (G1, slope matched) or free (G0)."""
        d = self.design.copy()
        d.curves[curve].join = "G1" if on else "G0"
        self.design = RM.pin(d, self.orig)
        self.join[curve] = d.curves[curve].join
        self._rebuild()

    def set_degree(self, degree):
        """Add (or remove) section control points: the sections become Bezier
        curves of this degree, the design carried over (exactly when raising).
        Returns the largest change of a section coordinate it caused."""
        degree = int(degree)
        if not RM.BASE_DEGREE <= degree <= MAX_DEGREE:
            raise ValueError(f"the section degree must lie in {RM.BASE_DEGREE} .. {MAX_DEGREE} "
                             f"({RM.BASE_DEGREE + 1} .. {MAX_DEGREE + 1} control points)")
        if degree == self.degree:
            return 0.0
        self.design, self.orig, err = RM.change_degree(self.design, self.orig, degree)
        self.orders = {c: self.design.curves[c].order for c in self.orig.names}
        self.join = {c: self.design.curves[c].join for c in self.orig.names}
        for c in list(self.meta):
            if not (c == "H" or c.startswith("LE.")):
                del self.meta[c]
        self._rebuild()
        return err

    def _drop_meta(self, curve):
        for name in [n for n in self.meta if n.startswith(curve + ".")]:
            del self.meta[name]

    # ---------------------------------------------------------------- rows
    def curve_rows(self, curve):
        """Rows for the control points C0 .. Cn of one curve, H to the root."""
        cd = self.design.curves[curve]
        n, H = cd.order, self.design.H
        le = curve == "LE"
        unit, digits = (" mm", 3) if le else ("", 5)
        rows = [Row("pinned", "C0", text=f"at H = {H:.2f} mm: {cd.v[0]:.{digits}f}{unit}, the original's value")]
        for k in range(1, n + 1):
            if k < n:
                rows.append(Row("var", f"C{k} d", f"{curve}.d{k}", 4))
            if k == 1 and cd.join == "G1":
                rows.append(Row("pinned", "C1 value", text=f"{cd.v[1]:.{digits}f}{unit}, on the original's "
                                                           f"tangent at H"))
            else:
                rows.append(Row("var", f"C{k} {'dx' if le else 'value'}" + (" (root)" if k == n else ""),
                                f"{curve}.{'dx' if le else 'v'}{k}", digits))
        return rows

    def row_info(self, name):
        """Where a variable's control point sits (height, mm)."""
        if name == "H":
            return f"max {self.settings['z_top']:.1f}"
        curve, slot = name.split(".")
        cd = self.design.curves[curve]
        k = int(slot.lstrip("dxv"))
        return f"y {cd.s[k] * self.design.H:.2f}"

    def n_free(self, curve=None):
        names = [n for n in self.space.all_names if curve is None or n.startswith(curve + ".")]
        return sum(1 for n in names if self.meta[n]["free"]), len(names)

    # ---------------------------------------------------------------- preview
    def preview(self, skeleton=False):
        """What the live views draw: the pinned design, its problems, the
        curves, N_VIEW sections from the root to H, the tracks of the section
        control points along the span, and (skeleton=True) the loops of the
        3D sections tab."""
        t0 = time.time()
        d = self.design
        s = self.settings
        problems = RM.design_problems(d, self.orig, s["n_sections"], s["te_radius_mm"])
        yy = np.linspace(0.0, d.H, 160)
        curves = {c: d.curve(c).at(yy)[:, 1] for c in self.names}
        try:
            sections = RM.build_sections(d, self.orig, np.linspace(0.0, d.H, N_VIEW), s["num_pts"],
                                         s["te_radius_mm"])
        except ValueError as exc:
            sections = str(exc)
        out = {"design": d, "problems": problems, "yy": yy, "curves": curves, "sections": sections,
               "tracks": self.tracks(d)}
        if skeleton:
            out["skeleton"] = self.skeleton(problems)
        out["seconds"] = time.time() - t0
        return out

    def tracks(self, d, n=25):
        """Where each section control point P0 .. Pn sits along the span, at n
        heights from the root to H: (n, degree + 1, 3) as x, z (thickness) and
        y, mm (the polygon scaled by the sharp chord, from the LE)."""
        ys = np.linspace(0.0, d.H, n)
        val = RM.curve_values(d, ys)
        ctrl = BS.polygon_from_values(val, d.degree)
        chord = self.orig.x_te(ys) - val["LE"]
        try:
            cs = BS.sharp_chord(ctrl, chord, self.settings["te_radius_mm"])[0]
        except (ValueError, RuntimeError):
            cs = chord                                 # a design with problems: close enough to draw
        x = val["LE"][:, None] + ctrl[:, :, 0] * cs[:, None]
        return np.stack((x, ctrl[:, :, 1] * cs[:, None], np.repeat(ys[:, None], d.degree + 1, axis=1)), axis=2)

    def skeleton(self, problems=None):
        """The loops of the 3D view, working frame (mm): 'modified' (the
        n_sections sections from the root to H, as XCAD gets them; none while
        the design has problems), 'cut' (the section at H) and 'unchanged'
        (the original loops kept above H: every few horizontal ones, the tip
        section and the cap)."""
        d, s = self.design, self.settings
        if problems is None:
            problems = RM.design_problems(d, self.orig, s["n_sections"], s["te_radius_mm"])
        mod = []
        if not problems:
            secs = RM.modified_sections(d, self.orig, s["n_sections"], s["num_pts"], s["te_radius_mm"])
            mod = [sec.loop() for sec in secs]
        gap = d.H / (s["n_sections"] - 1) if s["gap_mm"] is None else float(s["gap_mm"])
        flat = [lp for lp in self.loops0 if np.ptp(lp[:, 1]) < 1e-9 and lp[0, 1] >= d.H + gap - 1e-9]
        tip = [lp for lp in self.loops0 if np.ptp(lp[:, 1]) >= 1e-9 and lp[:, 1].min() >= d.H - 1e-9]
        step = max(1, int(np.ceil(len(flat) / SKELETON_UNCHANGED)))
        flat = flat[::step] + ([flat[-1]] if flat and (len(flat) - 1) % step else [])
        if not mod:
            try:
                cut = [RM.section_at(d, self.orig, d.H, s["num_pts"], s["te_radius_mm"]).loop()]
            except ValueError:
                cut = []
            return {"modified": [], "cut": cut, "unchanged": flat + tip}
        return {"modified": mod[:-1], "cut": mod[-1:], "unchanged": flat + tip}

    def views(self, fig):
        return RudderViews(self, fig)

    def skeleton_view(self, fig):
        return SkeletonView(self, fig)

    # ---------------------------------------------------------------- output
    def free_space(self):
        """The design space with the unticked variables held fixed: the one
        an optimiser gets."""
        fixed = {n: self.values[n] for n in self.space.all_names if not self.meta[n]["free"]}
        return RM.DesignSpace(self.orig, self.orders, self.join,
                              bounds={n: (self.meta[n]["lo"], self.meta[n]["hi"]) for n in self.space.all_names},
                              h_min=self.settings["h_min"], h_max=self.settings["h_max"], fixed=fixed)

    def space_record(self):
        """The set-up as plain data: the design space (every variable with its
        value, bounds and free flag, the curves' orders and joins, the section
        degree), the settings and the input files. rudder_modify.
        load_design_space reads it back for the optimiser."""
        return {"geometry": "rudder", "written": time.strftime("%d %b %Y %H:%M"),
                "settings": dict(self.settings), "files": dict(self.files),
                "space": self.free_space().to_dict(self.design)}

    def save_space(self, path):
        """Write space_record() as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.space_record(), fh, indent=1)
        return path

    def load_space(self, path):
        """Start from a saved set-up (<case>_design_space.json)."""
        with open(path) as fh:
            rec = json.load(fh)
        if rec.get("geometry") != "rudder":
            raise ValueError(f"{path} is not a rudder design space")
        for k in ("n_sections", "te_radius_mm"):
            if k in rec.get("settings", {}):
                self.settings[k] = rec["settings"][k]
        space, design = RM.DesignSpace.from_dict(self.orig, rec["space"])
        self.orig = space.orig
        self.orders, self.join, self.design = dict(space.orders), dict(space.join), design
        self.meta = {q["name"]: {"lo": q["lo"], "hi": q["hi"], "free": q["free"]} for q in rec["space"]["slots"]}
        self.settings["h_min"], self.settings["h_max"] = self.meta["H"]["lo"], self.meta["H"]["hi"]
        self._rebuild()

    def xcad_task(self, case, plot=None, cad=True):
        """A function that writes the case as the design is now (rudder_modify.
        write_case: the XCAD file, the section table, the thickness format when
        it applies, the design JSON, plus <case>_design_space.json), then lofts
        the XCAD file into a solid (xcad_loft) and writes it as STEP next to
        it. The state is copied here, so the function can run off the GUI
        thread while the sliders move on. It returns a dict: the case, its
        report, the loft result and the STEP path."""
        import xcad_loft as XL
        s = dict(self.settings)
        plot = s["plot"] if plot is None else bool(plot)
        design, space, record = self.design.copy(), self.free_space(), self.space_record()
        orig, stack, out_dir, xcad0 = self.orig, self.stack, self.out_dir, self.files["xcad_original"]

        def run():
            t0 = time.time()
            res = RM.write_case(out_dir, case, design, orig, xcad0, s["n_sections"], s["num_pts"],
                                s["te_radius_mm"], s["gap_mm"], s["xcad_units"], stack=stack, plot=plot,
                                space=space, h_min=s["h_min"], h_max=s["h_max"])
            path = os.path.join(out_dir, f"{case}_design_space.json")
            with open(path, "w") as fh:
                json.dump(copy.deepcopy(record), fh, indent=1)
            res["paths"]["space"] = path
            out = {"case": res, "report": RM.case_report(res, orig, stack), "t_case": time.time() - t0,
                   "loft": None, "step": None, "cad_error": None}
            if not cad:
                return out
            if not XL.occ_available():
                out["cad_error"] = "no OCCT binding (pythonocc-core or cadquery-ocp): no solid"
                return out
            loft = XL.loft_loops(XL.read_loops(res["paths"]["xcad"], s["xcad_units"]))
            step = os.path.splitext(res["paths"]["xcad"])[0] + ".step"
            XL.write_step(loft.shape, step)
            out.update(loft=loft, step=step)
            return out
        return run

    def build_xcad(self, case, plot=False, cad=True):
        """xcad_task(case, plot, cad)(), in one go."""
        return self.xcad_task(case, plot, cad)()


class RudderViews:
    """The Design tab: planform, the selected curve, and in 3D the sections
    from the root to H with their control polygons (labelled P0 .. Pn) and the
    track of each control point along the span."""

    def __init__(self, adapter, fig):
        self.a = adapter
        self.fig = fig
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.3], wspace=0.22, hspace=0.38,
                              left=0.09, right=0.98, top=0.95, bottom=0.03)
        self.ax_plan = fig.add_subplot(gs[0, 0])
        self.ax_curve = fig.add_subplot(gs[0, 1])
        self.ax_sec = fig.add_subplot(gs[1, :], projection="3d")
        self.ax_sec.view_init(elev=28, azim=-58)
        self.box = None                                      # the 3D box's aspect, set when drawn
        fig.canvas.mpl_connect("motion_notify_event", self._on_turn)
        fig.canvas.mpl_connect("resize_event", lambda event: self._fit_box())
        for ax in (self.ax_plan, self.ax_curve):
            ax.grid(True, color=GRID, lw=0.6)
        o = adapter.orig
        m = o.y_rows <= o.z_top + 1e-9
        ax = self.ax_plan
        ax.plot(o.x_le_rows[m], o.y_rows[m], "--", color=INK2, lw=1, label="original LE")
        ax.plot(o.x_te_rows[m], o.y_rows[m], "-", color=INK2, lw=1, label="TE line (kept)")
        self.sec_lines = [ax.plot([], [], "-", color=GRID, lw=0.8)[0] for _ in range(12)]
        self.le_line, = ax.plot([], [], "-", color=BLUE, lw=2.0, label="LE")
        self.le_poly, = ax.plot([], [], "o--", color=BLUE, lw=0.8, ms=5, mfc="white")
        self.h_line = ax.axhline(0.0, color=RED, ls=":", lw=1)
        self.bound_lines = [ax.axhline(0.0, color=INK2, ls="-.", lw=0.8) for _ in range(2)]   # h_min, h_max
        self.h_text = ax.text(0.99, 0.0, "H", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                              color=RED, fontsize=8)
        self.prob_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", color=RED, fontsize=8)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y, height (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("planform", loc="left", fontsize=10)

        ax = self.ax_curve
        self.c_data, = ax.plot([], [], ".", color=INK2, ms=3, label="original (section fits)")
        self.c_orig, = ax.plot([], [], "-", color=INK2, lw=1)
        self.c_new, = ax.plot([], [], "-", color=BLUE, lw=2.0, label="design")
        self.c_poly, = ax.plot([], [], "--", color=BLUE, lw=0.8)
        self.c_free, = ax.plot([], [], "o", color=BLUE, ms=6, label="C1 ..: free")
        self.c_fixed, = ax.plot([], [], "o", color=BLUE, ms=6, mfc="white", label="held")
        self.c_pinned, = ax.plot([], [], "s", color=RED, ms=6, mfc="white", label="pinned at H")
        self.c_labels = []
        self.c_h = ax.axhline(0.0, color=RED, ls=":", lw=1)
        ax.set_ylabel("y (mm)")
        ax.legend(loc="best", fontsize=7)

    def _points(self, d, curve):
        """The curve's control points split into free / held / pinned, as (y, value)."""
        cd = d.curves[curve]
        pts = np.column_stack((cd.s * d.H, cd.v))
        kinds = []
        for k in range(cd.order + 1):
            vname = f"{curve}.{'dx' if curve == 'LE' else 'v'}{k}"
            if k == 0 or vname not in self.a.meta:
                kinds.append("pinned")
            else:
                kinds.append("free" if self.a.meta[vname]["free"] else "fixed")
        kinds = np.array(kinds)
        return pts, {k: pts[kinds == k] for k in ("free", "fixed", "pinned")}

    def update(self, pv, curve):
        a, o, d = self.a, self.a.orig, pv["design"]
        yy, cv = pv["yy"], pv["curves"]
        # planform
        self.le_line.set_data(cv["LE"], yy)
        le = d.curve("LE")
        self.le_poly.set_data(le.ctrl[:, 1], le.ctrl[:, 0])
        ys = np.linspace(0.0, d.H, len(self.sec_lines))
        for ln, y, xl in zip(self.sec_lines, ys, np.interp(ys, yy, cv["LE"])):
            ln.set_data([xl, o.x_te([y])[0]], [y, y])
        self.h_line.set_ydata([d.H, d.H])
        self.h_text.set_position((0.99, d.H))
        self.h_text.set_text(f"H = {d.H:.1f}")
        for ln, key in zip(self.bound_lines, ("h_min", "h_max")):
            ln.set_ydata([a.settings[key]] * 2)
        self.prob_text.set_text("\n".join(p[:90] for p in pv["problems"][:4]))
        self.ax_plan.relim()
        self.ax_plan.autoscale_view()
        # the selected curve
        ax = self.ax_curve
        top = a.settings["h_max"]                            # the curve plot shows heights up to h_max
        if curve == "LE":
            keep = o.y_rows <= top
            self.c_data.set_data(o.x_le_rows[keep], o.y_rows[keep])
            self.c_orig.set_data([], [])
            ax.set_xlabel("x of the LE (mm)")
        else:
            yd, vd = o.data[curve]
            self.c_data.set_data(vd[yd <= top], yd[yd <= top])
            yz = np.linspace(0.0, top, 160)
            self.c_orig.set_data(o.value(curve, yz), yz)
            ax.set_xlabel(f"{curve[-1]} of P{curve[1:-1]} (fraction of the sharp chord)")
        self.c_new.set_data(cv[curve], yy)
        pts, P = self._points(d, curve)
        self.c_poly.set_data(pts[:, 1], pts[:, 0])
        for key, ln in (("free", self.c_free), ("fixed", self.c_fixed), ("pinned", self.c_pinned)):
            ln.set_data(P[key][:, 1], P[key][:, 0])
        for t in self.c_labels:
            t.remove()
        self.c_labels = [ax.annotate(f"C{k}", (v, y), textcoords="offset points", xytext=(6, 4), fontsize=8,
                                     color=BLUE) for k, (y, v) in enumerate(pts)]
        self.c_h.set_ydata([d.H, d.H])
        ax.set_title(f"curve {curve}" + ("" if curve == "LE" else f": {curve[-1]} of P{curve[1:-1]}")
                     + " against the height", loc="left", fontsize=10)
        shown = np.concatenate([ln.get_xdata() for ln in (self.c_data, self.c_orig, self.c_new, self.c_poly)
                                if len(ln.get_xdata())])
        lo, hi = float(shown.min()), float(shown.max())
        pad = max(0.08 * (hi - lo), 0.5 if curve == "LE" else 0.005)    # a flat curve still gets a readable range
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.03 * top, 1.05 * top)
        ax.ticklabel_format(axis="x", useOffset=False)
        self._draw_sections(pv, curve)

    def _draw_sections(self, pv, curve):
        """The 3D panel: N_VIEW sections at their heights (dashed grey: the
        original there), each with its control polygon; the track of every
        control point along the span (grey), the one being edited in red with
        its position on each section and a bar along the coordinate (x or z)
        at the root."""
        ax, a, d = self.ax_sec, self.a, pv["design"]
        elev, azim, roll = ax.elev, ax.azim, ax.roll
        ax.cla()
        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.tick_params(labelsize=8)
        sel = 0 if curve == "LE" else int(curve[1:-1])
        tracks = pv["tracks"]
        from matplotlib.ticker import MaxNLocator
        for i in range(tracks.shape[1]):
            t = tracks[:, i]
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=RED if i == sel else GREY, lw=2.2 if i == sel else 0.8)
        pts = [tracks.reshape(-1, 3)]
        secs = pv["sections"]
        if isinstance(secs, str):
            ax.text2D(0.01, 0.97, secs[:150], transform=ax.transAxes, color=RED, fontsize=8, va="top")
        else:
            for sec, col in zip(secs, SECTION_COLORS):
                lp = sec.loop()                                   # working frame: x, y height, z
                ax.plot(lp[:, 0], lp[:, 2], lp[:, 1], color=col, lw=1.5)
                x_le_o, xy_o = RM.original_section_at(a.stack, sec.y)
                orig_pts = np.column_stack((x_le_o + xy_o[:, 0], xy_o[:, 1], np.full(len(xy_o), sec.y)))
                ax.plot(orig_pts[:, 0], orig_pts[:, 1], orig_pts[:, 2], "--", color=GREY, lw=0.7)
                pts.append(orig_pts)
                cp = np.column_stack((sec.x_le + sec.ctrl[:, 0] * sec.cs, sec.ctrl[:, 1] * sec.cs))
                ax.plot(cp[:, 0], cp[:, 1], np.full(len(cp), sec.y), "o:", color=col, lw=0.9, ms=3.5, mfc="white")
                ax.plot([cp[sel, 0]], [cp[sel, 1]], [sec.y], "o", color=RED, ms=6)
                pts.append(np.column_stack((lp[:, 0], lp[:, 2], lp[:, 1])))
            root = secs[0]
            cp = np.column_stack((root.x_le + root.ctrl[:, 0] * root.cs, root.ctrl[:, 1] * root.cs))
            for i, (x, z) in enumerate(cp):
                ax.text(x, z + 1.5, root.y, f"P{i}", color=RED if i == sel else BLUE, fontsize=9)
            x, z = cp[sel]
            if curve == "LE" or curve.endswith("x"):
                bar = np.array([[x - 10.0, z, root.y], [x + 10.0, z, root.y]])
            else:
                bar = np.array([[x, z - 6.0, root.y], [x, z + 6.0, root.y]])
            ax.plot(bar[:, 0], bar[:, 1], bar[:, 2], color=RED, lw=3)
            pts.append(bar)
        # 3D lines are not clipped: the limits take in everything drawn, and the box is fitted to the panel
        p = np.vstack(pts)
        lo, hi = p.min(axis=0), p.max(axis=0)
        pad = 0.03 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        self.box = np.maximum(hi - lo, 1e-9) * [1.0, Z_STRETCH, 1.0]
        self._fit_box()
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel("x (mm)")
        ax.set_ylabel(f"z (mm, x{Z_STRETCH:g})")
        ax.set_zlabel("y (mm)")
        what = "LE x" if curve == "LE" else f"P{sel} {curve[-1]}"
        ax.set_title(f"sections 0 .. H with control polygons P0 .. P{d.degree} (dashed: the original)\n"
                     f"grey: each control point along the span, red: {what}; thickness x{Z_STRETCH:g}, drag to turn",
                     fontsize=9)

    def _fit_box(self):
        """Fit the 3D box and its labels to the panel at the current angle: zoom
        so that they fill it, and shift the view so that they sit in its middle.
        mplot3d puts the tick and axis labels outside the box by a fraction of
        its size (at the sides, and below the box when seen from above), so the
        box grown by that fraction, plus room for the text, has to fit."""
        if self.box is None:
            return
        from mpl_toolkits.mplot3d import proj3d
        ax, dpi = self.ax_sec, self.fig.dpi
        ax.apply_aspect()
        panel, square, view = ax.get_position(original=True).transformed(self.fig.transFigure), ax.bbox, ax.viewLim
        lims = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        grow = 1.33 * dpi / (square.width + square.height) * np.diff(lims, axis=1) * [-1.0, 1.0]   # the labels' offset
        boxes = [np.array(np.meshgrid(*b)).reshape(3, -1) for b in (lims, lims + 0.5 * grow, lims + grow)]
        text, gap = 0.25 * dpi, 0.1 * dpi
        top = min(panel.y1, square.y1)                            # the title is above the square
        x0, x1, y0, y1 = panel.x0 + text, panel.x1 - text, panel.y0 + text, top - gap
        if ax.elev < 0.0:                                         # seen from below, the labels are above the box
            y0, y1 = panel.y0 + gap, top - text
        px = square.width / view.width                            # px per unit of the projection, in x and y
        zoom = 1.0
        for _ in range(3):                                        # the perspective is not quite linear in the zoom
            ax.set_box_aspect(self.box, zoom=zoom)
            (u, v), (_, v_half), (u_out, _) = (proj3d.proj_transform(*b, ax.get_proj())[:2] for b in boxes)
            lo, hi = (v_half.min(), v.max()) if ax.elev >= 0.0 else (v.min(), v_half.max())
            fit = min((x1 - x0) / (np.ptp(u_out) * px), (y1 - y0) / ((hi - lo) * px))
            zoom = float(np.clip(zoom * fit, 0.3, 3.0))
        ax.set_box_aspect(self.box, zoom=zoom)
        (u, v), (_, v_half), (u_out, _) = (proj3d.proj_transform(*b, ax.get_proj())[:2] for b in boxes)
        lo, hi = (v_half.min(), v.max()) if ax.elev >= 0.0 else (v.min(), v_half.max())
        mid_u, mid_v = 0.5 * (u_out.min() + u_out.max()), 0.5 * (lo + hi)
        fx = (0.5 * (x0 + x1) - square.x0) / square.width        # where the middle goes, as a fraction of the square
        fy = (0.5 * (y0 + y1) - square.y0) / square.height
        w, h = view.width, view.height
        view.intervalx = (mid_u - fx * w, mid_u + (1.0 - fx) * w)
        view.intervaly = (mid_v - fy * h, mid_v + (1.0 - fy) * h)

    def _on_turn(self, event):
        """While the 3D panel is turned with the mouse, keep the box fitted."""
        if self.ax_sec.button_pressed == 1:
            self._fit_box()


class SkeletonView:
    """The 3D tab: the rudder as its sections (XCAD frame: x chordwise, the
    height up), modified below H in blue, the section at H (the cut) in red,
    the unchanged original above in grey."""

    def __init__(self, adapter, fig):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        self.a = adapter
        self.ax = ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.col = {k: Line3DCollection([np.zeros((2, 3))], colors=c, linewidths=w)
                    for k, c, w in (("unchanged", GREY, 0.6), ("modified", BLUE, 0.8), ("cut", RED, 2.0))}
        for c in self.col.values():
            ax.add_collection3d(c)
        allp = np.vstack(adapter.loops0)
        self.lo, self.hi = allp.min(axis=0), allp.max(axis=0)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_zlabel("y, height (mm)")
        ax.view_init(elev=22, azim=-55)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(3))
        self.text = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, fontsize=9, va="top")

    @staticmethod
    def _frame(lp, every=2):
        q = lp[::every]
        return np.column_stack((q[:, 0], -q[:, 2], q[:, 1]))

    def update(self, pv):
        sk = pv["skeleton"]
        segs = {k: [self._frame(lp) for lp in sk[k]] for k in self.col}
        for k, c in self.col.items():
            c.set_segments(segs[k])
        f = np.vstack([s for v in segs.values() for s in v] + [self._frame(np.array([self.lo, self.hi]), 1)])
        lo, hi = f.min(axis=0), f.max(axis=0)
        self.ax.set_xlim(lo[0], hi[0])
        self.ax.set_ylim(lo[1], hi[1])
        self.ax.set_zlim(lo[2], hi[2])
        self.ax.set_box_aspect(np.maximum(hi - lo, 1e-9))
        n = {k: len(sk[k]) for k in ("modified", "cut", "unchanged")}
        msg = (f"{n['modified'] + n['cut']} modified sections (blue, the cut at H in red), "
               f"{n['unchanged']} original loops above (grey)")
        if pv["problems"]:
            msg += "\nno modified sections: " + pv["problems"][0][:100]
        self.text.set_text(msg)
