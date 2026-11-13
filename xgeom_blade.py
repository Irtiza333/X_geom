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
else to the curve's present shape. The parameter file's vary lines
(blade_vary.read_vary) set the start: a distribution flagged N is the file's
own (form 0, "file": its cubic spline, no variables; 1 or 2 segments make it a
Bezier curve fitted to the file, its variables free), one flagged Y a Bezier
curve whose variables are free, within the file's range when given; radius,
hub radius and hub height flagged Y free. A file with no vary line: every curve
fitted, blade_control.FREE free. A component file (xgeom_files.py) gives its
design, variables (bounds, free flags) and settings exactly. load_params reads
another component file or old parameter file; save_file writes the design, its
variables and the settings as a component file (Save set-up); save_params
writes the design as an old parameter file, with vary lines for it as it stands
(Y: a curve or global with a free variable). set_sections sets the sections of the
XCAD points file and the 3D sections tab (blade_modify.section_stations).
set_blades sets the number of blades Z (2 to 7; the parameter file's by
default): the clearance check, the propeller views, the CAD's hub sector, the
parameter file the design is written as and the set-up follow it (it is a
setting, not in the design vector). Blades closer than the clearance (they
interfere) are a warning, not a problem: the CAD is still built (the hub radius
moves the root section, which may make room for them).

The hub (13 Nov 2026; prop_hub.py): a cylinder of the hub radius centred on
x = 0 (the DRDC hub sector), or a cone with a rounded nose, set_hub_shape:
its base radius, top radius and base x are variables like the hub height (held
to start with; defaults when switched on: the hub radius, half of it, the hub
centred on x = 0), and the hub radius is then the blade's root radius, where
the table starts; the blade is extended into the hub and trimmed by it in the
CAD. set_part chooses what the CAD button writes (blade_modify.PARTS): the
whole propeller, half of it (an even Z) or one blade on its 1/Z of the hub (the
default; with the cylinder the DRDC files of the MSc pipeline, otherwise
solids). The vehicle builds the whole propeller.

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
CONE_ROWS = (("radius", "radius R"), ("hub_radius", "root r"), ("hub_height", "hub height"),
             ("hub_base", "hub base r"), ("hub_top", "hub top r"), ("hub_x", "hub base x"))
PART_LABELS = {"whole": "the whole propeller", "half": "half of it", "sector": "one blade on 1/Z of the hub"}
PART_SHORT = {"whole": "whole propeller", "half": "half", "sector": "1 blade on 1/Z hub"}
TWO_SEGMENT_ROWS = (("v1", "P1 value (root)"), ("d1", "P2=P3 d1"), ("w23", "P2=P3 weight"), ("u4", "P4 position"),
                    ("v4", "P4 value"), ("d2", "P5=P6 d2"), ("w56", "P5=P6 weight"), ("v7", "P7 value (tip)"))
TWO_SEGMENT_POINT = {"v1": 0, "d1": 1, "w23": 1, "u4": 3, "v4": 3, "d2": 4, "w56": 4, "v7": 6}   # control point


class BladeAdapter:
    joins = False                 # the curves have no join to switch
    segment_choice = True         # each curve 1 or 2 segments, or the file's own (0)
    segment_options = (0, 1, 2)   # 0: as the parameter file has it ("file")
    build_label = "CAD"
    parts = BM.PARTS              # what the CAD writes: the tool offers the choice (set_part)
    hub_shapes = ("cylinder", "cone")

    def __init__(self, params=None):
        import blade_control as BC
        s = BC.settings
        self.settings = {"clearance_mm": float(s.clearance_mm), "sections": int(s.sections),
                         "tip_band": float(s.tip_band), "tip_sections": int(s.tip_sections), "hub": bool(s.hub),
                         "part": str(getattr(s, "part", "sector")),
                         "plot": bool(s.plot), "orders": dict(BC.ORDERS), "segments": dict(BC.SEGMENTS),
                         "free": tuple(BC.FREE)}
        self.out_dir = BC.OUT_DIR
        self.load_params(params or BC.PARAMS)

    # ---------------------------------------------------------------- the file
    def load_params(self, path):
        """Start from a component file (blade_modify.load_component: its design, variables and settings
        exactly) or an old parameter file: its distributions fitted (blade_control's segments and orders) or
        kept as the file has them (vary N); the variables of the parameters flagged Y free (blade_control.FREE
        when the file has no vary line)."""
        ld = BM.load_component(path, self.settings["orders"], self.settings["segments"], self.settings["free"])
        self.params, self.design = ld.params, ld.design
        self.meta = {n: {"lo": float(lo), "hi": float(hi), "free": n not in ld.space.fixed}
                     for n, (lo, hi) in zip(ld.space.all_names, ld.space.all_bounds)}
        self.settings.update(ld.settings)
        self.file, self.legacy, self.notes, self.extra = os.path.abspath(path), ld.legacy, ld.notes, ld.extra
        self.file_blades = self.params.blades       # the file's Z (set_blades may change the one in use)
        self.as_fitted = set(BM.CURVES)             # curves still as the file has them (no variable changed)
        self._rebuild()
        self._original = None

    def save_file(self, path):
        """Write the design, its variables (bounds, free flags) and the settings as a component file
        (blade_modify.write_component), which load_params reads back exactly; it is the file in use then."""
        BM.write_component(path, self.params, self.design, self.free_space(), self._case_settings(),
                           os.path.splitext(os.path.basename(path))[0], self.notes, self.extra)
        self.file, self.legacy = os.path.abspath(path), False
        return path

    def save_params(self, path):
        """Write the design as a parameter file (read_params and load_params read it), with vary lines for the
        design as it stands (vary_state)."""
        BM.write_params(path, BM.design_params(self.design, self.params, vary=self.vary_state()),
                        notes=(f"written by the design tool from {os.path.basename(self.params.path)}, "
                               f"{time.strftime('%d %b %Y %H:%M')}",))
        return path

    def vary_state(self):
        """Vary lines for the design as it stands: Y for a curve or global with a free variable (its range when
        all its values share one), N for the others."""
        out = {}
        glob = self.design.hub_slots()
        for n in BM.VARY_NAMES:
            if n in BM.HUB_SLOTS and n not in glob:
                continue
            slots = [n] if n in glob else [s for s in self.space.all_names if s.startswith(n + ".")]
            on = any(self.meta[s]["free"] for s in slots)
            vals = slots if n in glob else [s for s in slots if s.split(".")[1][0] == "v"]
            rng = {(self.meta[s]["lo"], self.meta[s]["hi"]) for s in vals}
            lo, hi = next(iter(rng)) if on and len(rng) == 1 else (None, None)
            out[n] = BM.Vary(on, lo, hi)
        return out

    def _free0(self):
        """The parameters whose variables start free: the file's Y, else blade_control.FREE."""
        names = BM.vary_free(self.params)
        return names if names is not None else self.settings["free"]

    @property
    def title(self):
        return f"Propeller blade ({os.path.basename(self.file)})"

    @property
    def names(self):
        return BM.CURVES

    # ---------------------------------------------------------------- state
    def _rebuild(self):
        """The space for the present curves, keeping the bounds and free flags
        of every slot that is still there."""
        full = BM.BladeSpace(self.params, self.design, bounds=BM.vary_bounds(self.params))
        free0 = self._free0()
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
        if name in BM.GLOBAL_SLOTS + ("hub_base", "hub_top") and lo <= 0.0:
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
        was, or if it was the file's own): fitted to the parameter file while it
        is as fitted to it (no variable of it changed since the file was read),
        else to its present shape; form 0: the file's own. Returns what was done,
        for the status bar."""
        was_free = (self.design.curves[curve].segments == 0
                    or any(m["free"] for n, m in self.meta.items() if n.startswith(curve + ".")))
        rr = np.linspace(self.design.r0, self.design.r1, 401)
        old = self.design.at(curve, rr)
        if segments == 0:
            self.design = BM.refit_to_file(self.params, self.design, curve, 0)
            for n in [n for n in self.meta if n.startswith(curve + ".")]:
                del self.meta[n]
            self.as_fitted.add(curve)
            self._rebuild()
            return "as the parameter file has it (vary N): no design variables"
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

    def set_form(self, curve, segments, order=None):
        """The curve as segments (0, 1, 2) and, with 1 segment, of the order given: fitted again (_refit);
        xgeom_from_cad's best forms go in this way. Returns what was done."""
        segments = int(segments)
        if segments == 0:
            return self._refit(curve, 0, None)
        order = int(self.settings["orders"].get(curve, BM.DEFAULT_ORDER) if order is None else order)
        if segments == 1 and not 1 <= order <= MAX_ORDER:
            raise ValueError(f"the order must lie in 1 .. {MAX_ORDER}")
        return self._refit(curve, segments, order)

    def set_forms(self, forms):
        """Every curve in the form given, {curve: (segments, order)} (xgeom_from_cad's best forms), fitted
        (set_form), then the variables of the parameters _free0 names free (blade_control.FREE for an extracted
        file), the others held: a file of table curves gives no variables to take the flags from. Returns what
        was done, curve by curve."""
        done = {c: self.set_form(c, segments, order) for c, (segments, order) in forms.items()}
        free = set(self._free0())
        for n, m in self.meta.items():
            if "." in n:
                m["free"] = n.split(".")[0] in free
        return done

    def set_order(self, curve, order):
        """Number of control points of a 1-segment curve = order + 1; the curve
        is fitted again (_refit)."""
        order = int(order)
        if self.segments(curve) == 0:
            raise ValueError("the file's own distribution has no control points: 1 or 2 segments first")
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
        again (_refit), or as the parameter file has it (0)."""
        n = int(n)
        if n not in (0, 1, 2):
            raise ValueError("a curve has 1 or 2 segments, or is the file's own (0)")
        if n == self.segments(curve):
            return "unchanged"
        return self._refit(curve, n, self.settings["orders"].get(curve, BM.DEFAULT_ORDER))

    # ---------------------------------------------------------------- rows
    def global_rows(self):
        """Rows for the blade radius, hub radius and hub height (m); a conic hub: its base and top radii and
        base x too (the hub radius then the blade root's)."""
        rows = CONE_ROWS if self.design.hub_cone else GLOBAL_ROWS
        return [Row("var", f"{label} (m)", name, 4) for name, label in rows]

    @property
    def hub_shape(self):
        return self.design.hub_shape

    def set_hub_shape(self, shape):
        """The hub as a cylinder or a cone (blade_modify.BladeDesign.with_hub: the cone's base radius the hub
        radius, its top radius half of it, centred on x = 0, to start with); its variables start held. Returns
        the hub's description."""
        shape = str(shape).strip().lower()
        if shape not in self.hub_shapes:
            raise ValueError(f"a hub is one of {', '.join(self.hub_shapes)}, not '{shape}'")
        if shape == self.design.hub_shape:
            return self.design.hub.describe()
        self.design = self.design.with_hub(shape)
        for n in BM.HUB_SLOTS:
            self.meta.pop(n, None)
        self._rebuild()
        for n in BM.HUB_SLOTS:
            if n in self.meta:
                self.meta[n]["free"] = False
        return self.design.hub.describe()

    def set_part(self, part):
        """What the CAD button writes (blade_modify.cad_part): whole, half or sector; refused when half does not
        tile the propeller (an odd Z)."""
        part, _ = BM.cad_part({"part": part}, self.params.blades)
        self.settings["part"] = part
        return part

    def part_label(self, part=None, short=False):
        labels = PART_SHORT if short else PART_LABELS
        return labels[part or self.settings["part"]].replace("1/Z", f"1/{self.params.blades}")

    def curve_rows(self, curve):
        """Rows for the variables of one curve, from the root to the tip (the file's own: none)."""
        cu = self.design.curves[curve]
        digits = 3 if curve == "skew" else 5
        if cu.segments == 0:
            return [Row("pinned", "the file's", text="as the parameter file has it (vary N); 1 or 2 segments make "
                                                     "it a Bezier curve to change")]
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
            hp = d.hub
            return f"x {hp.x_base:.3f} to {hp.x_tip:.3f}" if d.hub_cone else f"x ±{0.5 * d.hub_height:.3f}"
        if name == "hub_base":
            return f"x {d.hub.x_base:.3f}"
        if name == "hub_top":
            return f"x {d.hub.x_tangent:.3f}"
        if name == "hub_x":
            return f"tip x {d.hub.x_tip:.3f}"
        curve, key = name.split(".")
        k = TWO_SEGMENT_POINT[key] if self.segments(curve) == 2 else int(key[1:])
        return f"r/R {d.ctrl(curve)[k, 0]:.3f}"

    def ctrl_free(self, curve):
        """For each control point of a curve: is any variable that moves it free?"""
        cu = self.design.curves[curve]
        if cu.segments == 0:
            return np.zeros(len(cu.r), dtype=bool)
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
        kept = [c for c in BM.CURVES if d.curves[c].segments == 0]
        hub = (f"hub a cone, base radius {d.hub_base:.4g} m, top {d.hub_top:.4g} m, {d.hub_height:.4g} m to the tip "
               f"from x {d.hub_x:.4g} m" if d.hub_cone else f"hub height {d.hub_height:.4g} m")
        return (f"{n} of {m} variables free; R {d.radius:.4g} m, {'blade root' if d.hub_cone else 'hub'} radius "
                f"{d.hub_radius:.4g} m (r/R {d.r0:.3f}), {hub}, {self.params.blades} blades; CAD: "
                f"{self.part_label()}" + (f"; as the file (vary N): {', '.join(kept)}" if kept else ""))

    def pick_note(self, curve):
        """The picker's note: the file's own curve, or no free variable."""
        if self.segments(curve) == 0:
            return "  (file)"
        return "  (held)" if self.n_free(curve)[0] == 0 else ""

    def describe(self, curve):
        if self.segments(curve) == 0:
            return (f"{BM.DESCRIPTIONS[curve]} against r/R as the parameter file has it (vary N): the cubic spline "
                    f"through its rows, not a design variable")
        e, rel = BM.fit_report(self.params, self.design)[curve]
        diff = f"largest difference from the parameter file {e:.3g} ({100 * rel:.2f} % of its range)"
        if self.segments(curve) == 2:
            return (f"{BM.DESCRIPTIONS[curve]} against r/R, 2 rational cubic Bezier segments P1-P4 and P4-P7 "
                    f"(x_blade's form): P2 = P3 and P5 = P6 at P4's value, so zero slope at P4; d1 and d2 place "
                    f"them between P4 and the root and the tip, w23 and w56 are their weights; {diff}")
        return f"{BM.DESCRIPTIONS[curve]} against r/R, a Bezier curve from the root (C0) to the tip; {diff}"

    def set_blades(self, z):
        """The number of blades Z (blade_modify.BLADES: 2 .. 7); refused (ValueError) otherwise."""
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
        """What the live views draw: the design, its problems and warnings
        (blade_modify.design_checks), the curves, the file's rows and splines, the
        blade (BladeSurface, the CAD shape): sections at the view radii, LE, TE,
        mid-chord and tip lines, the hub; with skeleton the blade's sections as
        XCAD gets them (para.py's at stations())."""
        t0 = time.time()
        d, p = self.design, self.params
        rr = np.linspace(d.r0, d.r1, 200)
        file_rr = np.linspace(p.root_r, p.r[-1], 200)
        out = {"design": d, "rr": rr, "curves": {c: d.prop(c)(rr) for c in BM.CURVES}, "file_rr": file_rr,
               "file": {c: p.spline(c)(file_rr) for c in BM.CURVES}, "rows": p.rows(), "radii": self._view_radii(),
               "hub": (1000.0 * d.hub_radius, 1000.0 * d.hub_height), "hub_outline": 1000.0 * d.hub.outline(),
               "hub_shape": d.hub_shape}
        warnings = []
        try:
            pts, _ = BM.blade_points(d, p)
            out["clearance_mm"] = BM.clearance(pts, p.blades) * 1000.0
            problems, warnings = BM.design_checks(d, p, self.settings["clearance_mm"], points=pts,
                                                  gap_mm=out["clearance_mm"], hub=self.settings["hub"])
        except Exception as exc:
            problems = [f"{type(exc).__name__}: {exc}"]
        try:
            bs = BM.blade_surface(d, p)
            eta = np.linspace(float(bs.eta_of_r(d.r0)), 1.0, 120)       # from the design root (not the extension)
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
        out["problems"], out["warnings"] = problems, warnings
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

    CASE_KEYS = ("clearance_mm", "sections", "tip_band", "tip_sections", "hub", "part")

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
        self.file, self.legacy, self.notes, self.extra = params.path, not BM.XF.is_component(params.path), [], {}
        self.as_fitted = set()                      # a saved design: refits keep its curves' shapes
        self.meta = {q["name"]: {"lo": q["lo"], "hi": q["hi"], "free": q["free"]} for q in rec["space"]["slots"]}
        for k in self.CASE_KEYS:
            if k in rec.get("settings", {}):
                self.settings[k] = rec["settings"][k]
        self._rebuild()

    def build_task(self, case, half=None, part=None):
        """A function that writes the case as the design is now (blade_modify.
        write_case: the component file, the XCAD points, the design, the check
        plot) and then the CAD (<case>.iges, <case>.step: the part of the
        settings, or part). half is ignored (a blade has no half body); the
        vehicle passes it to every component, and part 'whole' (half=False).
        The state is copied here, so it can run off the GUI thread."""
        design, params, space, st = self.design.copy(), self.params, self.free_space(), self._case_settings()
        if part is not None:
            st["part"] = BM.cad_part({"part": part}, params.blades)[0]
        elif half is False:
            st["part"] = "whole"
        out_dir, plot, extra = self.out_dir, self.settings["plot"], dict(self.extra)

        def run():
            res = BM.write_case(out_dir, case, design, params, space, cad=False, settings=st, plot=plot,
                                extra=extra)
            out = {"case": res, "view": None, "error": False,
                   "report": [f"{case}: {res['sections']} sections, blades {res['clearance_mm']:.1f} mm apart",
                              "  written: " + ", ".join(res["paths"].values())]}
            files = f"{case}.txt, {case}_xcad_m.dat"
            if res["problems"]:
                out.update(message=f"{files}; no CAD: {res['problems'][0]}", error=True)
                return out
            if importlib.util.find_spec("OCC") is None:
                out.update(message=f"{files}; no CAD: needs pythonocc-core", error=True)
                return out
            cad = BM.write_cad(out_dir, case, design, params, st)
            res["paths"].update(cad["cad_paths"])
            what = PART_LABELS[cad["part"]].replace("1/Z", f"1/{params.blades}")
            out["report"].append(f"  CAD ({what}; hub {cad['hub']}): grids {cad['grid_seconds']:.1f} s, faces and "
                                 f"IGES {cad['cad_seconds']:.1f} s: {cad['cad_paths']['iges']}, "
                                 f"{cad['cad_paths']['step']}")
            if "valid" in cad:
                out["report"].append(f"  solid {'valid' if cad['valid'] else 'NOT valid'}, "
                                     f"{cad['volume_mm3'] / 1e3:.2f} cm^3, {cad['faces']} faces"
                                     + (f", the root footprint {cad['footprint_gap_deg']:.1f} deg from the sector's "
                                        f"edges" if "footprint_gap_deg" in cad else ""))
            out.update(view=cad["cad_paths"]["step"],
                       message=f"{files}, {os.path.basename(cad['cad_paths']['iges'])} and {case}.step ({what}"
                               + (f", solid {'valid' if cad['valid'] else 'NOT valid'}" if "valid" in cad else "")
                               + "); OCC viewer shows it"
                               + (f" (warning: {res['warnings'][0]})" if res["warnings"] else ""))
            return out
        return run


def hub_lines(outline, n_rings=9, n_gen=12):
    """Lines that draw a hub from its profile (prop_hub.HubProfile.outline, (x, r), the base plane first):
    rings at the two ends of a cylinder, at n_rings stations along the lateral part of a cone and its nose,
    and n_gen generators along it (to the tip of a nose)."""
    outline = np.asarray(outline, dtype=float)
    lateral = outline[1:]                                 # without the base plane's foot on the axis
    if len(lateral) > 1 and lateral[-1, 1] < 1e-9 and abs(lateral[-1, 0] - lateral[-2, 0]) < 1e-9:
        lateral = lateral[:-1]                            # a flat end: no spoke
    x, r = lateral[:, 0], lateral[:, 1]
    t = np.linspace(0.0, 2.0 * np.pi, 121)
    if np.ptp(r) < 1e-9 * max(1.0, r.max()):              # a cylinder: its two rims
        xs, rs = np.array([x[0], x[-1]]), np.array([r[0], r[-1]])
    else:
        s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(r)))))
        u = np.linspace(0.0, 1.0, n_rings) * s[-1]
        xs, rs = np.interp(u, s, x), np.interp(u, s, r)
    out = [np.column_stack((np.full_like(t, xk), rk * np.sin(t), rk * np.cos(t))) for xk, rk in zip(xs, rs)
           if rk > 1e-9]
    for a in np.linspace(0.0, 2.0 * np.pi, n_gen, endpoint=False):
        out.append(np.column_stack((x, r * np.sin(a), r * np.cos(a))))
    return out


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
        if cu.segments == 0:                       # the file's own: no control points
            ax.set_xlabel("r/R")
            ax.set_ylabel(f"{curve} ({BM.UNITS[curve]})")
            ax.legend(loc="best", fontsize=7)
            ax.set_title(f"curve {curve}: {BM.DESCRIPTIONS[curve]}, as the parameter file has it", loc="left",
                         fontsize=10)
            return
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
        segs["hub"] = hub_lines(pv["hub_outline"])
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
        hr, hh = pv["hub"]
        hub = (f"hub a cone, base radius {1000 * d.hub_base:.0f} mm, top {1000 * d.hub_top:.0f} mm, {hh:.0f} mm long"
               if d.hub_cone else f"hub radius {hr:.0f} mm, height {hh:.0f} mm")
        msg = (f"{p.blades} blades, {len(sk)} sections each (the designed blade blue), D {d.diameter:.4g} m; {hub}; "
               f"blades {pv.get('clearance_mm', float('nan')):.1f} mm apart")
        if pv["problems"]:
            msg += "\nproblem: " + pv["problems"][0][:100]
        self.text.set_text(msg)
