"""
xgeom_wing.py

The wing and the fin in the XGeom design tool (xgeom_tool.py): a wing's component file (wing_modify.py,
xgeom_files.py) as the tool's variables, its live views and the CAD build. The two share all of it; each has its
own file and output folder (ROLES). Since 13 Nov 2026 the rudder too, as a planar lifting surface (PlanarRudderAdapter: a
component file of kind rudder, such as BB2's stern plane, bb2/outputs/bb2_sternplane.txt), the Rudder slot's
other model beside the modified-below-H one (xgeom_rudder.py); it has no default file. Since 20 Nov 2026 the
blended wing body (BwbAdapter: kind bwb, bwb_glider.txt), the Hull slot's other body beside the cylindrical hull
(xgeom_hull.py): the wing's model over the half-span from the centre line, every curve flat there (wing_modify:
1 segment with C1 at C0's value, or 2 smooth segments, form smooth2), drawn with its mirror image, its CAD the
whole body (the half and its mirror image fused) or, with the half switch, the half on one side of the centre
line.

A wing or a fin starts from a component file: the span, the airfoil, the pitch axis and the spanwise table.
Its variables: the span (mm), and each distribution (chord, sweep, rake, pitch, thickness, camber) and each
section change (dt1 .. dtk of the thickness, dc1 .. dck of the camber line) as a Bezier curve over the span
fraction eta, fitted to the file, of 1 or 2 segments as the blade's (xgeom_blade.py), or as the table has it
(form 0, "table": the cubic spline through its rows, no variables). The root values of sweep and rake are
pinned at 0: the root's LE is the origin. A change of the span stretches the curves over it, their values
staying. A new form or order is fitted to the parameter file while none of the curve's variables has changed,
else to the curve's present shape. set_changes sets the number of change points along the chord (0 to 8): the
changes are carried over, exactly when there are more of them. set_sections sets the sections of the XCAD file.
load_params reads a component file (its design, variables and settings exactly; a table file's curves fitted);
save_file writes the design, its variables and the settings as one (Save set-up). A variable is independent or
dependent on others through a relation (xgeom_relations.AdapterRelations: relation, set_relation, limits; the
dependent ones follow when the others move).

The views are drawn as the rudder's, in the wing's own frame: x (or the curve's value) across and the span up.
The Design tab has only what a design is decided on (the picked curve, the sections over the chord); the 3D
sections tab shows the whole shape.

The adapter has the methods of xgeom_rudder.RudderAdapter that the tool uses.
"""

from __future__ import annotations

import os
import time

import numpy as np

import wing_modify as WM
import xgeom_relations as REL
from xgeom_common import BLUE, GREY, GRID, INK2, RED, SECTION_COLORS, Row, fit_box, keep_fitted

GREEN = "#1f8a70"                                  # the tip cap's loops in the 3D view
MIRROR = "#9ec5f4"                                 # a blended wing body's other half (mirrored) in the 3D views

MAX_ORDER = 12
VIEW_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)     # sections in the Design tab's views, fractions of the span
FREE = ("span",) + WM.CURVES + WM.change_names(WM.MAX_CHANGES)   # whose variables start free: all of them, the
                                                  # user holds what he chooses (21 Nov 2026; chord and pitch before)
ROLES = {"wing": {"name": "Wing", "params": "wing.txt", "out_dir": os.path.join("outputs", "wing")},
         "fin": {"name": "Fin", "params": "fin.txt", "out_dir": os.path.join("outputs", "fin")},
         "rudder": {"name": "Rudder", "params": None, "out_dir": os.path.join("outputs", "rudder_planar")},
         "bwb": {"name": "Blended wing body", "params": "bwb_glider.txt", "out_dir": os.path.join("outputs", "bwb")}}
TWO_SEGMENT_ROWS = (("v1", "P1 value (root)"), ("d1", "P2=P3 d1"), ("w23", "P2=P3 weight"), ("u4", "P4 position"),
                    ("v4", "P4 value"), ("d2", "P5=P6 d2"), ("w56", "P5=P6 weight"), ("v7", "P7 value (tip)"))
TWO_SEGMENT_POINT = {"v1": 0, "d1": 1, "w23": 1, "u4": 3, "v4": 3, "d2": 4, "w56": 4, "v7": 6}
SMOOTH_ROWS = (("v0", "P0 = P1 value (centre line)"), ("d1", "P1 d"), ("d2", "P2 d"), ("v2", "P2 value"),
               ("u3", "P3 position"), ("v3", "P3 value"), ("d4", "P4 d"), ("d5", "P5 d"), ("v5", "P5 value"),
               ("v6", "P6 value (tip)"))      # smooth 2 segments (a blended wing body's): P4's value from the slope
SMOOTH_MOVES = (("v0",), ("v0", "d1"), ("d2", "v2"), ("u3", "v3"), ("d4", "v2", "v3"), ("d5", "v5"), ("v6",))
SEGMENT_TEXT = {0: "table", 1: "1", 2: "2", 3: "2 smooth"}             # the segments choice's buttons
DIGITS = {"chord": 2, "sweep": 2, "rake": 2, "pitch": 3}  # value digits; t/c, f/c and the changes 5


class WingAdapter(REL.AdapterRelations):
    joins = False                 # the curves have no join to switch
    segment_choice = True         # each curve 1 or 2 segments, or as the table has it (0)
    segment_options = (0, 1, 2)   # 0: as the table has it ("table"); 3: 2 smooth segments (a blended wing body)
    build_label = "CAD"
    role = "wing"
    free_start = FREE             # the curves free in a table file (with no var lines)
    half_text = None              # the Output box's half switch: the tool's own text

    def __init__(self, params=None):
        r = ROLES[self.role]
        self.settings = {"sections": WM.SECTIONS, "points": WM.POINTS, "plot": True, "free": self.free_start,
                         "half": False, "cap_loops": WM.CAP_LOOPS}
        self.out_dir = r["out_dir"]
        if not (params or r["params"]):
            raise ValueError(f"a planar {r['name'].lower()} starts from its component file (kind {self.role})")
        self.load_params(params or r["params"])

    # ---------------------------------------------------------------- the file
    def load_params(self, path):
        """Start from a component file (wing_modify.load_component: its design, variables and settings exactly;
        a table file's distributions fitted, 1 segment, the variables of FREE free)."""
        ld = WM.load_component(path, free=self.settings["free"])
        self.params, self.design = ld.params, ld.design
        self.meta = self.meta_of(ld.space)          # each variable's bounds, free flag and (dependent) relation
        self.settings.update({k: v for k, v in ld.settings.items() if k in self.CASE_KEYS})
        self.file, self.notes = os.path.abspath(path), ld.notes
        self.as_fitted = set(self.design.names)     # curves still as the file has them (no variable changed)
        self._rebuild()
        self._original = None

    def save_file(self, path):
        """Write the design, its variables (bounds, free flags, the dependent ones' relations) and the settings
        as a component file of this role's kind (wing_modify.write_component), which load_params reads back
        exactly; it is the file in use then."""
        WM.write_component(path, self.params, self.design, self.free_space(), self._case_settings(),
                           os.path.splitext(os.path.basename(path))[0], self.notes, self.role)
        self.file = os.path.abspath(path)
        return path

    @property
    def title(self):
        return f"{ROLES[self.role]['name']} ({os.path.basename(self.file)})"

    @property
    def name(self):
        """What the component is called (the tool's labels): Wing, Fin, Rudder, Blended wing body."""
        return ROLES[self.role]["name"]

    @staticmethod
    def segment_text(n):
        """The segments choice's button for form n."""
        return SEGMENT_TEXT[n]

    @property
    def names(self):
        return self.design.names

    # ---------------------------------------------------------------- state
    def _rebuild(self):
        """The space for the present curves, keeping the bounds, free flags and relations of every slot still
        there; the dependent variables then follow (xgeom_relations)."""
        full = WM.WingSpace(self.params, self.design)
        free0 = self.settings["free"]
        self.meta = {n: self.meta.get(n, {"lo": float(lo), "hi": float(hi), "free": n.split(".")[0] in free0})
                     for n, (lo, hi) in zip(full.all_names, full.all_bounds)}
        self.space = WM.WingSpace(self.params, self.design,
                                  bounds={n: (m["lo"], m["hi"]) for n, m in self.meta.items()})
        self.values = self.space.values(self.design)
        self.follow(note=True)

    def get(self, name):
        return self.values[name]

    def bounds(self, name):
        return self.meta[name]["lo"], self.meta[name]["hi"]

    def is_free(self, name):
        return self.meta[name]["free"]

    def set(self, name, value):
        """Set one variable; returns the value used (kept inside its limits: its bounds and, a dependent
        variable, its relations). The variables that depend on it follow."""
        vals, value, moved = self._moved(name, value)
        self.design = self.space.design_of(vals)
        self.values = self.space.values(self.design)
        for n in moved:
            self.as_fitted.discard(n.split(".")[0])
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
        """The curve in a new form, its slots renewed (free if any of its slots was, or if it was as the table
        has it): fitted to the parameter file while it is as fitted to it, else to its present shape; form 0:
        as the table has it. Returns what was done."""
        was_free = (self.design.curves[curve].segments == 0
                    or any(m["free"] for n, m in self.meta.items() if n.startswith(curve + ".")))
        ee = np.linspace(0.0, 1.0, 401)
        old = self.design.at(curve, ee)
        if segments == 0:
            self.design = WM.refit_to_file(self.params, self.design, curve, 0)
            self._drop(curve)
            self.as_fitted.add(curve)
            self._rebuild()
            return "as the table has it: no design variables"
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
        if self.segments(curve) == 0:
            raise ValueError("a curve as the table has it has no control points: 1 or 2 segments first")
        if self.segments(curve) == 3:
            raise ValueError("a smooth 2-segment curve has its 7 control points P0 .. P6")
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
        """The curve as 1 segment (DEFAULT_ORDER, CHANGE_ORDER for a change) or 2 segments (a blended wing body:
        3, the 2 smooth segments), fitted again, or as the table has it (0)."""
        n = int(n)
        if n not in self.segment_options:
            raise ValueError("a curve has " + " or ".join(SEGMENT_TEXT[k] + (" segments" if k > 1 else " segment")
                                                          for k in self.segment_options if k)
                             + ", or is as the table has it (0)")
        if n == self.segments(curve):
            return "unchanged"
        return self._refit(curve, n, WM.DEFAULT_ORDER if curve in WM.CURVES else WM.CHANGE_ORDER)

    def set_changes(self, k):
        """k change points along the chord for the thickness and the camber line each (0 to MAX_CHANGES): the
        changes carried over (wing_modify.with_changes), the new change curves free unless every variable of the
        old ones was held (the user's choice). Returns what was done."""
        k = int(k)
        k0 = self.design.changes
        if k == k0:
            return "unchanged"
        old = WM.change_names(k0)
        flags = [m["free"] for n, m in self.meta.items() if n.split(".")[0] in old]
        free = not flags or any(flags)
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
        return [Row("var", "half-span (mm)" if self.params.flat else "span (mm)", "span", 2)]

    def curve_label(self, curve):
        if curve in WM.CURVES:
            return curve
        return f"{curve}: {'thickness' if curve.startswith('dt') else 'camber'} change at point {curve[2:]}"

    def curve_rows(self, curve):
        """Rows for the variables of one curve, from the root to the tip; the pinned root of sweep and rake; a
        blended wing body's C1 value, tied to C0's (flat at the centre line)."""
        cu = self.design.curves[curve]
        flat = self.params.flat
        if cu.segments == 0:
            return [Row("pinned", "the table's", text="as the table has it (" + (
                "the monotone cubic through its rows and their mirror image, flat at the centre line"
                if flat else "the cubic spline through its rows") + "): no variables; 1 or 2 segments make it a "
                                                                    "Bezier curve to change")]
        digits = DIGITS.get(curve, 5)
        pinned = curve in WM.PINNED
        root_text = "0: the centre line's LE is the origin" if flat else "0: the root's LE is the origin"
        if cu.segments == 2:
            return [Row("pinned", label, text=root_text) if pinned and k == "v1" else
                    Row("var", label, f"{curve}.{k}", digits if k[0] == "v" else 4) for k, label in TWO_SEGMENT_ROWS]
        if cu.segments == 3:
            return [Row("pinned", label, text=root_text) if pinned and k == "v0" else
                    Row("var", label, f"{curve}.{k}", digits if k[0] == "v" else 4) for k, label in SMOOTH_ROWS]
        n = cu.order
        rows = []
        for k in range(n + 1):
            if 0 < k < n:
                rows.append(Row("var", f"C{k} d", f"{curve}.d{k}", 4))
            label = f"C{k} value" + ((" (centre line)" if flat else " (root)") if k == 0 else " (tip)" if k == n else "")
            if pinned and k == 0:
                rows.append(Row("pinned", label, text=root_text))
            elif flat and k == 1:
                rows.append(Row("pinned", label, text="C0's value: flat at the centre line"))
            else:
                rows.append(Row("var", label, f"{curve}.v{k}", digits))
        return rows

    def row_info(self, name):
        """Where a variable sits: its control point's y along the span (mm); for the span the planform area."""
        d = self.design
        if name == "span":
            ee = np.linspace(0.0, 1.0, 201)
            halves = 2.0 if self.params.flat else 1.0          # a blended wing body: both halves
            return f"S {halves * WM._integrate(d.at('chord', ee), ee) * d.span / 100.0:.0f} cm2"
        curve, key = name.split(".")
        seg = self.segments(curve)
        k = (TWO_SEGMENT_POINT[key] if seg == 2 else WM.SmoothTwoSegmentCurve.POINT[key] if seg == 3 else
             int(key[1:]))
        return f"y {d.ctrl(curve)[k, 0] * d.span:.1f}"

    def ctrl_free(self, curve):
        """For each control point of a curve: is any variable that moves it free? (a pinned root: no)"""
        cu = self.design.curves[curve]
        if cu.segments == 0:
            return np.zeros(len(cu.r), dtype=bool)
        free = {k: self.varies(f"{curve}.{k}") for k in cu.keys() if f"{curve}.{k}" in self.meta}
        if cu.segments == 2:
            groups = (("v1",), ("d1", "w23"), ("d1", "w23"), ("u4", "v4"), ("d2", "w56"), ("d2", "w56"), ("v7",))
            return np.array([any(free.get(k, False) for k in g) for g in groups])
        if cu.segments == 3:
            return np.array([any(free.get(k, False) for k in g) for g in SMOOTH_MOVES])
        n = cu.order
        tied = self.params.flat                     # C1 at C0's value: it moves with C0
        return np.array([free.get(f"v{k}", False) or (0 < k < n and free[f"d{k}"]) or (tied and k == 1 and
                                                                                       free.get("v0", False))
                         for k in range(n + 1)])

    def n_free(self, curve=None):
        names = [n for n in self.space.all_names if curve is None or n.startswith(curve + ".")]
        return sum(1 for n in names if self.varies(n)), len(names)

    def summary(self):
        d = self.design
        c = d.at("chord", [0.0, 1.0])
        if self.params.flat:
            return (f"{self.free_text()}; span {2 * d.span:.1f} mm (two halves of {d.span:.1f} mm), chord "
                    f"{c[0]:.1f} mm at the centre line and {c[1]:.1f} mm at the tips; {d.changes} change point"
                    f"{'s' if d.changes != 1 else ''} along the chord")
        return (f"{self.free_text()}; span {d.span:.1f} mm, chord {c[0]:.1f} mm at the root and {c[1]:.1f} mm "
                f"at the tip; {d.changes} change point{'s' if d.changes != 1 else ''} along the chord")

    def pick_note(self, curve):
        """The picker's note: as the table has it, or no free variable."""
        if self.segments(curve) == 0:
            return "  (table)"
        return "  (held)" if self.n_free(curve)[0] == 0 else ""

    def describe(self, curve):
        if self.segments(curve) == 0:
            return (f"{WM.describe_curve(curve, self.design.changes)} against the span as the table has it: the "
                    f"cubic spline through its rows, not a design variable")
        e, rel = WM.fit_report(self.params, self.design)[curve]
        diff = f"largest difference from the parameter file {e:.3g} {WM.unit(curve)}"
        if curve in WM.CURVES and np.isfinite(rel) and rel > 0.0:
            diff += f" ({100 * rel:.2f} % of its range)"
        what = WM.describe_curve(curve, self.design.changes) + " against the span"
        if curve in WM.PINNED:
            what += " (the centre line's value pinned at 0)" if self.params.flat else " (the root's value pinned at 0)"
        if self.segments(curve) == 2:
            return (f"{what}, 2 rational cubic Bezier segments P1-P4 and P4-P7: P2 = P3 and P5 = P6 at P4's value, "
                    f"so zero slope at P4; {diff}")
        if self.segments(curve) == 3:
            return (f"{what}, 2 cubic Bezier segments P0-P3 and P3-P6 joined smoothly at P3 (P4 on the line P2 P3: "
                    f"the slopes match), flat at the centre line (P1 at P0's value); {diff}")
        if self.params.flat:
            return f"{what}, a Bezier curve from the centre line (C0) to the tip, C1 at C0's value (flat there); {diff}"
        return f"{what}, a Bezier curve from the root (C0) to the tip; {diff}"

    # ---------------------------------------------------------------- sections
    def stations(self):
        return WM.section_stations(self.settings["sections"], self.params.tip_tilt > 0.0)

    def set_sections(self, n):
        """n sections in the XCAD file, evenly from the root to the tip (3 or more)."""
        WM.section_stations(n)
        self.settings["sections"] = int(n)

    def sections_info(self):
        n, m, k = self.settings["sections"], self.settings["points"], self.design.changes
        tilt = self.params.tip_tilt
        tip = ((f"then the tip section tilted {tilt:g} deg about its LE (the horizontal sections end half a step "
                f"under the tip's LE)" if tilt > 0.0 else "the tip flat")
               + ("" if self.params.cap is None else
                  f", then {self.settings['cap_loops']} loops over the tip cap ({WM.cap_describe(self.params.cap)})"))
        return (f"{n} sections {self.design.span / (n - 1):.1f} mm apart along the span, {2 * m - 1} points each "
                f"({m} per side, closer together at the LE and the TE); {tip}. Change points: "
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
        view outlines (the web app draws these; the tool's Design tab no longer, since 21 Nov 2026), the sections
        over their chords with and without the changes; with skeleton the XCAD file's sections."""
        t0 = time.time()
        d, p = self.design, self.params
        ee = np.linspace(0.0, 1.0, 121)
        out = {"design": d, "ee": ee, "curves": d.values(ee), "file": {c: p.spline(c)(ee) for c in d.names},
               "rows": (p.eta, p.values), "fractions": np.array(VIEW_FRACTIONS),
               "mirror": bool(p.flat)}               # a blended wing body: drawn with its mirror image (y to -y)
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
        out["problems"], out["warnings"] = problems, warnings + self.relation_warnings()
        if skeleton:
            try:
                out["skeleton"] = WM.section_loops(d, p, self.stations(), self.settings["points"])
                out["tilted"] = WM.tilted_tip(d, p, self.settings["points"])[0] if p.tip_tilt > 0.0 else None
                out["cap"] = WM.cap_loops(d, p, self.settings["points"], self.settings["cap_loops"]) if p.cap else None
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
        """The design space with the unticked variables held fixed and the dependent ones' relations: the one
        an optimiser gets."""
        fixed = {n: self.values[n] for n in self.space.all_names if not self.meta[n]["free"]}
        return self.related(WM.WingSpace(self.params, self.design,
                                         bounds={n: (m["lo"], m["hi"]) for n, m in self.meta.items()}, fixed=fixed))

    CASE_KEYS = ("sections", "points", "half", "cap_loops")

    def _case_settings(self):
        return {k: self.settings[k] for k in self.CASE_KEYS}

    def build_task(self, case, half=None):
        """A function that writes the case as the design is now (wing_modify.write_case: the component file,
        the XCAD file, the sections table, the design, the check plot), then lofts the XCAD file into the solid
        (<case>.step; the half body with the half setting, which half overrides: the vehicle builds full
        bodies). The state is copied here, so it can run off the GUI thread."""
        import xcad_loft as XL
        design, params, space, st = self.design.copy(), self.params, self.free_space(), self._case_settings()
        out_dir, plot, role = self.out_dir, self.settings["plot"], self.role
        half = bool(self.settings.get("half", False)) if half is None else bool(half)

        def run():
            res = WM.write_case(out_dir, case, design, params, space, cad=False, settings=st, plot=plot,
                                geometry=role)
            out = {"case": res, "view": None, "error": False,
                   "report": [f"{case}: {res['sections']} sections, planform {res['area_mm2'] / 100:.1f} cm^2, "
                              f"volume {res['volume_mm3'] / 1000:.1f} cm^3 (from the sections)",
                              "  written: " + ", ".join(res["paths"].values())]}
            files = f"{case}.txt, {case}_xcad_m.dat"
            if res["problems"]:
                out.update(message=f"{files}; no CAD: {res['problems'][0]}", error=True)
                return out
            if not XL.occ_available():
                out.update(message=f"{files}; no CAD: no OCCT binding (pythonocc-core or cadquery-ocp)", error=True)
                return out
            try:
                cad = WM.write_cad(out_dir, case, res["paths"]["xcad_m.dat"], half=half)
            except ValueError as exc:                                        # no half body: not symmetric
                out.update(message=f"{files}; {exc}", error=True)
                return out
            res["paths"].update(cad["cad_paths"])
            lr = cad["loft"]
            body = ("half body, " if lr.half else "") + ("solid valid" if lr.valid else "solid NOT valid")
            out["report"].append(f"  CAD: the loft in {cad['cad_seconds']:.1f} s, {cad['cad_paths']['step']} "
                                 f"({body}, {lr.volume / 1e3:.2f} cm^3)")
            out.update(view=cad["cad_paths"]["step"], loft=lr,
                       message=f"{files}, {case}.step ({body}, {lr.volume / 1e3:.2f} cm^3); OCC viewer shows it")
            return out
        return run


class FinAdapter(WingAdapter):
    role = "fin"


class PlanarRudderAdapter(WingAdapter):
    """The rudder as a planar lifting surface: a component file of kind rudder on wing_modify's engine (for
    example bb2/outputs/bb2_sternplane.txt, BB2's stern plane, bb2_aft_setup.py), the tool's Rudder slot's
    other model beside the modified-below-H one (xgeom_rudder.RudderAdapter). Save set-up writes kind rudder."""
    role = "rudder"


class BwbAdapter(WingAdapter):
    """The blended wing body (kind bwb, 20 Nov 2026; bwb_glider.txt, the underwater glider of Ye, Wang and Pan
    2021): the wing's model over the half-span from the centre line, every curve flat there, the tool's Hull slot's
    other body beside the cylindrical hull. Its curves are 1 segment (C1 at C0's value), 2 smooth segments
    (smooth2: joined smoothly at a free joint, the body's end; every curve of the sample since 21 Nov 2026) or as
    the table has it; its 3D sections tab draws it with its mirror image; its CAD (build_task) is the whole body,
    the loft of the half and its mirror image fused, or with
    the half switch the half on one side of the centre line. The report gives the volume, the wetted area and the
    centroid. Save set-up writes kind bwb."""
    role = "bwb"
    segment_options = (0, 1, 3)
    half_text = ("half body: one side of the centre line, closed by its centre section on the plane of symmetry "
                 "(for CFD on half the domain)")
    short = "BWB"

    def sections_info(self):
        n, m, k = self.settings["sections"], self.settings["points"], self.design.changes
        return (f"{n} sections {self.design.span / (n - 1):.1f} mm apart from the centre line to the tip (the half; "
                f"the CAD mirrors it), {2 * m - 1} points each ({m} per side, closer together at the LE and the TE); "
                f"the tip flat. Change points: {k} Bezier points along the chord that change the thickness (dt) and "
                f"the camber line (dc), each with its curve over the span")

    def build_task(self, case, half=None):
        """A function that writes the case as the design is now (wing_modify.write_case: the component file, the
        XCAD file of the half, the sections table, the design, the check plot; the body's volume, wetted area and
        centroid from the sections), then lofts the half and fuses it with its mirror image (write_cad, mirror):
        <case>.step, the whole body; with the half setting (which half overrides) the half alone."""
        import xcad_loft as XL
        design, params, space, st = self.design.copy(), self.params, self.free_space(), self._case_settings()
        out_dir, plot, role = self.out_dir, self.settings["plot"], self.role
        half = bool(self.settings.get("half", False)) if half is None else bool(half)

        def run():
            res = WM.write_case(out_dir, case, design, params, space, cad=False, settings=st, plot=plot,
                                geometry=role)
            out = {"case": res, "view": None, "error": False,
                   "report": [f"{case}: {res['sections']} sections (the half), both halves: planform "
                              f"{res['area_mm2'] / 100:.1f} cm^2, volume {res['volume_mm3'] / 1000:.1f} cm^3"
                              + (f", wetted area {res['body']['wetted_mm2'] / 100:.1f} cm^2, centroid x "
                                 f"{res['body']['centroid_mm'][0]:.2f} z {res['body']['centroid_mm'][2]:.2f} mm"
                                 if "body" in res else "") + " (from the sections)",
                              "  written: " + ", ".join(res["paths"].values())]}
            files = f"{case}.txt, {case}_xcad_m.dat"
            if res["problems"]:
                out.update(message=f"{files}; no CAD: {res['problems'][0]}", error=True)
                return out
            if not XL.occ_available():
                out.update(message=f"{files}; no CAD: no OCCT binding (pythonocc-core or cadquery-ocp)", error=True)
                return out
            cad = WM.write_cad(out_dir, case, res["paths"]["xcad_m.dat"], half=half, mirror=True)
            res["paths"].update(cad["cad_paths"])
            lr, whole = cad["loft"], cad.get("whole")
            if whole:
                c = whole["centroid"]
                body = (f"whole body, {whole['solids']} solid{'s' if whole['solids'] != 1 else ''} "
                        f"{'valid' if whole['valid'] else 'NOT valid'}, {whole['volume'] / 1e3:.2f} cm^3")
                out["report"].append(f"  CAD: the half lofted and fused with its mirror image in "
                                     f"{cad['cad_seconds']:.1f} s, {cad['cad_paths']['step']} ({body}, wetted area "
                                     f"{whole['area'] / 100:.1f} cm^2, centroid x {c[0]:.2f} y {c[1]:.2f} z "
                                     f"{c[2]:.2f} mm)")
                error = not (whole["valid"] and whole["solids"] == 1)
            else:
                body = (f"half body, one side of the centre line, solid {'valid' if lr.valid else 'NOT valid'}, "
                        f"{lr.volume / 1e3:.2f} cm^3")
                out["report"].append(f"  CAD: the loft in {cad['cad_seconds']:.1f} s, {cad['cad_paths']['step']} "
                                     f"({body})")
                error = not lr.valid
            out.update(view=cad["cad_paths"]["step"], loft=lr, whole=whole, error=error,
                       message=f"{files}, {case}.step ({body}); OCC viewer shows it")
            return out
        return run


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
    """The Design tab: what a design is decided on. The picked curve (its value across, the span up: the file's
    rows and spline, the design and its control polygon) and the sections over their chords (dotted: the airfoil
    scaled, without the changes). The 3D sections tab shows the whole shape; the planform, the front view and the
    wing in 3D were here until 21 Nov 2026 (the user: only plots that inform the design, nothing else)."""

    def __init__(self, adapter, fig):
        self.a = adapter
        self.fig = fig
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.22, left=0.07, right=0.98, top=0.94,
                              bottom=0.08)
        self.ax_curve = fig.add_subplot(gs[0, 0])
        self.ax_sec = fig.add_subplot(gs[0, 1])

    def update(self, pv, curve):
        for ax in (self.ax_curve, self.ax_sec):
            ax.cla()
            ax.grid(True, color=GRID, lw=0.6)
        if isinstance(pv.get("sections"), str):
            self.ax_sec.text(0.02, 0.98, pv["sections"][:120], transform=self.ax_sec.transAxes, va="top",
                             color=RED, fontsize=8, wrap=True)
        else:
            self._sections(pv, curve)
        self._curve(pv, curve)
        if pv["problems"]:
            self.ax_curve.text(0.02, 0.98, "\n".join(q[:80] for q in pv["problems"][:3]),
                               transform=self.ax_curve.transAxes, va="top", ha="left", color=RED, fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85), zorder=10)

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
        if free.any():
            ax.plot(p[free, 1], y[free], "o", color=BLUE, ms=6, label="control point: free")
        held = ~free & ~pinned
        if held.any():
            ax.plot(p[held, 1], y[held], "o", color=BLUE, ms=6, mfc="white", label="control point: held")
        if pinned.any():
            ax.plot(p[pinned, 1], y[pinned], "s", color=RED, ms=6, mfc="white",
                    label="pinned (centre line's LE)" if pv.get("mirror") else "pinned (root LE)")
        if cu.segments == 2:
            labels = ("P1", f"P2=P3 (w {cu.w23:.2f})", None, "P4", f"P5=P6 (w {cu.w56:.2f})", None, "P7")
        elif cu.segments == 3:                             # smooth 2 segments: P0 .. P6, the joint P3
            labels = ("P0", "P1", "P2", "P3 (joint)", "P4", "P5", "P6")
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
        ax.set_title(f"curve {a.curve_label(curve)}" + {2: ", 2 segments", 3: ", 2 smooth segments"}.get(cu.segments, ""),
                     loc="left", fontsize=10)

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
        z = np.concatenate([C + 0.5 * T, C - 0.5 * T, C0 + 0.5 * T0, C0 - 0.5 * T0]).ravel()
        h = max(float(np.ptp(z)), 1e-3)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(float(z.min()) - 0.15 * h, float(z.max()) + 0.7 * h)    # room for the legend at the top
        ax.set_aspect("equal", adjustable="box")           # true shape, the box as tall as the sections need,
        ax.set_anchor("C")                                 # in the middle of its column
        ax.set_xlabel("x/c")
        ax.set_ylabel("z/c")
        ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
        ax.set_title("sections over the chord (dotted: no changes)", loc="left", fontsize=10)


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
                    for k, c, w in (("mirror", MIRROR, 0.7), ("sections", BLUE, 0.8), ("root", RED, 2.0),
                                    ("tip", INK2, 1.6), ("cap", GREEN, 0.7))}
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
        cap, tilted = pv.get("cap"), pv.get("tilted")
        tip = [tilted[:, [0, 2, 1]]] if tilted is not None else [q[-1]]
        mirror = bool(pv.get("mirror"))                    # a blended wing body: the other half too, lighter
        other = [lp * np.array([1.0, 1.0, -1.0]) for lp in q[1:]] if mirror else []
        segs = {"sections": list(q[1:-1]) + ([q[-1]] if tilted is not None else []), "root": [q[0]], "tip": tip,
                "cap": list(cap[:, :, [0, 2, 1]]) if cap is not None and len(cap) else [np.zeros((2, 3))],
                "mirror": other or [np.zeros((2, 3))]}
        for k, c in self.col.items():
            c.set_segments(segs[k])
        p = np.vstack([q.reshape(-1, 3), tip[0]] + ([cap.reshape(-1, 3)[:, [0, 2, 1]]] if cap is not None and len(cap) else [])
                      + [lp for lp in other])
        lo, hi = p.min(axis=0), p.max(axis=0)
        ax = self.ax
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        self.box = np.maximum(hi - lo, 1e-9)
        fit_box(ax, self.fig, self.box)
        d = pv["design"]
        tip_name = f"the tip section tilted {self.a.params.tip_tilt:g} deg (dark)" if tilted is not None else "the tip (dark)"
        if mirror:
            msg = (f"{len(sk)} sections, {sk.shape[1]} points each (the XCAD file: the half); half-span {d.span:.1f} "
                   f"mm; the centre line's section (red); the CAD fuses the half with its mirror image (light), "
                   f"{tip_name} closed by a planar face")
        else:
            msg = (f"{len(sk)} sections, {sk.shape[1]} points each (the XCAD file); span {d.span:.1f} mm; the root "
                   f"(red) " + (f"is closed by a planar face in the loft;\n{tip_name} carries {len(cap)} cap loops "
                                f"(green): {WM.cap_describe(self.a.params.cap)}" if cap is not None and len(cap) else
                                f"and {tip_name} are closed by planar faces in the loft"))
        if pv["problems"]:
            msg += "\nproblem: " + pv["problems"][0][:100]
        self.text.set_text(msg)
