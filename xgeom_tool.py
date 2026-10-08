"""
xgeom_tool.py

XGeom design tool: set up a geometry's design space with sliders and see the
shape change live. The build button writes the case and its CAD; OCC viewer
shows the solid.

    python xgeom_tool.py                    the vehicle: hull, propeller and rudder in one
                                            window (the rudder alone where xgeom_hull.py is
                                            not there, as in the Rudder folder)
    python xgeom_tool.py rudder             the rudder alone
    python xgeom_tool.py rudder --space outputs/modified/gui_design_space.json
                                            start from a saved set-up
    python xgeom_tool.py blade              the propeller blade of blade_control.PARAMS
    python xgeom_tool.py blade --params my_blade.dat
                                            the blade of another parameter file
    python xgeom_tool.py blade --space outputs/blade/gui_design_space.json
    python xgeom_tool.py hull               the hull of suboff_hull_params.dat
    python xgeom_tool.py hull --params my_hull.dat
    python xgeom_tool.py vehicle --params my_hull.dat --space outputs/blade/gui_design_space.json
                                            the vehicle: --params (a hull's or a blade's) and
                                            --space as often as needed, each goes to its
                                            component

Vehicle
    Component    the hull, the propeller or the rudder: its left panel and its
                 Design and 3D sections tabs, as when it is alone; each keeps its
                 state while another is shown.
    Vehicle tab  the hull with the propeller and the rudder mounted at its stern
                 (xgeom_vehicle.py) as their designs stand; the component being
                 edited is drawn thicker. Placement:
                 propeller  plane at x - L: the middle of the hub on the axis, from
                            the end of the hull (m, negative forward; default 0);
                            D: its diameter in the vehicle, the design scaled to it
                            (default half the hull's depth or breadth, the larger)
                 rudder     root TE at x - L: the root's trailing edge from the end of
                            the hull (default the start of the cap, a quarter of the
                            propeller's D clear of the blades); rudders: how many, 1 to
                            12, evenly spaced around the axis (default 4, a cross);
                            first at: the first one's angle around the axis from the
                            top towards starboard (180 under the stern); scale (1: its
                            own size, mm to m). Each root sits on the hull: at the
                            hull's distance from the axis in its direction, the
                            smallest along the root.
                 show       the whole vehicle or its stern (from the start of the tail).
Left panel (rudder)
    H         the height of the modified region, between its low and high
              bounds (h_min, h_max); h_max can go up to the top horizontal section
              of the stack (z_full - d, shown under 'where').
    Section control points
              the number of control points of the section's Bezier half-
              section, P0 .. Pn (P0 = (0, 0) the LE and Pn = (1, 0) the sharp TE
              stay put; adding one keeps the shape and adds that point's x and z
              curves). Pick the LE or the x or z of an inner point to edit the
              spanwise curve it follows; "held" marks a curve with no free
              variable. P1 x is held at 0 by default (a round LE).
    Curve     the picked curve's control points C0 .. Cm, from H (C0, pinned to
              the original) down to the root (Cm); how many there are; whether
              C1 sits on the original's tangent at H (slope matched).
Left panel (blade)
    Parameter file
              the blade's parameter file (blade_modify.py: D, blades, root r/R,
              hub height, section table or airfoil name, and the radial table).
              Load reads another one (the curves are fitted to it again); Save
              writes the design as one.
    Blade and hub
              the blade radius R, the hub radius and the hub height (m; the hub
              is centred at x = 0 and must cover the blade root). A change of
              either radius stretches the distributions over the new span.
              blades Z: the number of blades, 2 to 12 (the parameter file's to
              start with); the clearance check, the propeller tab, the vehicle,
              the CAD's hub sector and the parameter file written follow it. It
              is saved with the set-up; it is not in the design vector.
    Sections  the sections of the XCAD points file (para.py's, 53 points
              each) and the 3D sections tab: how many, root to tip, and how
              many of them lie from the tip band's r/R to the tip, closer
              together towards it (the others evenly spaced from the root).
              The clearance check keeps x_blade_new's stations.
    Distributions
              pick pitch, chord, thickness, camber, skew or rake to edit its
              curve, a Bezier over r/R fitted to the file. Curve: 1 segment (C0
              at the root .. Cm at the tip, as many as set) or 2 segments
              (x_blade's form: rational cubics P1 .. P4 and P4 .. P7, P2 = P3 and
              P5 = P6 at P4's value, so zero slope at P4; the weights w23 and w56
              are variables). Switching, or a new number of control points,
              refits the curve: to the parameter file while none of its
              variables has changed, else to its present shape.
Left panel (hull)
    Parameter file
              the hull's station table (hull_modify.py: the nose, middle-body,
              tail and cap lengths, the half-height r and with elliptical sections
              the half-width r' at stations). Load reads another one (the curves
              are fitted to it again); Save writes the design as one.
    Body      the four lengths, the largest radius r* and the tail end's re (m);
              elliptical sections: the half-width r' (XY view) gets radii and
              curves of its own, r is then the half-height (XZ view).
    Curves    nose r and tail r (and nose r', tail r'), each a CST curve: its
              exponent (N1 the nose's bluntness, N2 the shape of the tail end)
              and its control values; the two control points at the middle body
              are pinned (r* with zero slope there). A new number of control
              points refits the curve as for the blade.
Each variable:  free | name | low | slider | high | value | where
    free      ticked: a design variable within low .. high; unticked: held
    d         the position of a control point, as the fraction d of the span
              left (rudder: below the point above it; blade: up to the tip;
              blade, 2 segments: d1 from P4 to the root, d2 from P4 to the tip)
    dx        rudder LE control point: x offset from the original LE, mm (negative forward)
    value     rudder: a section coordinate, fraction of the sharp chord;
              blade: the distribution's value (P/D, c/D, t/c, f/c, deg, rake/D);
              hull: a lengths or a radius (m), an exponent or a CST coefficient
    where     rudder: the height; blade: r/R; hull: the x it sits at (m)
Right panel
    Design       rudder: planform, the picked curve against the original, and in
                 3D five sections from the root to H with their control polygons
                 and each control point's track along the span. Blade: the
                 expanded outline, the picked curve against the parameter file,
                 and in 3D the blade as the CAD builds it. Hull: the half-profiles
                 of the whole hull, the picked curve's part against the parameter
                 file, and the hull in 3D. The 2D plots have the span or the length
                 on the x axis (height, r, r/R, x). Drag the 3D view to turn it.
    3D sections  the whole rudder as its sections: modified (blue), the cut at H
                 (red), unchanged (grey); all blades of the propeller; the hull's
                 sections by part.
Save set-up writes the design space as it stands (every variable with its value,
bounds and free flag, each curve's control points and order, the settings and
input files) to <output folder>/<case>_design_space.json; rudder_modify.
load_design_space, blade_modify.load_blade_space or hull_modify.load_hull_space
reads it for the optimiser, --space for the tool.
XCAD (rudder) writes the case to the output folder (XCAD file, section table,
design and the same set-up JSON), lofts the XCAD file into a solid and writes it
as STEP. CAD (blade) writes the case (parameter file, XCAD points, design and
set-up JSON) and the DRDC five-surface blade with the hub sector as IGES and
STEP. CAD (hull) writes the case (parameter file, design and set-up JSON,
stations, hydrostatics, section points, check plot) and the STEP. OCC viewer
opens the last STEP (needs pythonocc-core).
The bar at the bottom: green ok, amber a build running (the build button counts
the seconds), red a problem or an error; a design with a problem (e.g. blades
closer than the clearance) is not built until it is solved.

Other geometries plug in as adapters with the methods of
xgeom_rudder.RudderAdapter (registered in ADAPTERS) and a panel function here
(PANELS) for what is particular to them.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tkinter as tk                                                 # noqa: E402
from tkinter import filedialog, ttk                                  # noqa: E402

import matplotlib                                                    # noqa: E402
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure                                 # noqa: E402

from xgeom_common import Row                                         # noqa: E402

ADAPTERS = {"rudder": ("xgeom_rudder", "RudderAdapter"), "blade": ("xgeom_blade", "BladeAdapter"),
            "hull": ("xgeom_hull", "HullAdapter")}
VEHICLE = ("hull", "blade", "rudder")              # the vehicle's components, in the switcher's order
NAMES = {"hull": "Hull", "blade": "Propeller", "rudder": "Rudder"}
INK2, RED = "#52514e", "#e34948"
STATUS = {"ok": ("#dfeedd", "black"), "busy": ("#f6d58e", "black"), "error": ("#c62f2e", "white")}  # bar colours
VEHICLE_TAB = 2                                    # the notebook's tabs: Design, 3D sections, Vehicle


def load_adapter(name, params=None):
    module, cls = ADAPTERS[name]
    return getattr(importlib.import_module(module), cls)(**({"params": params} if params else {}))


def available(name):
    """Is the geometry's adapter in this folder (the Rudder folder has the rudder's only)?"""
    return importlib.util.find_spec(ADAPTERS[name][0]) is not None


def file_kind(path):
    """The geometry a set-up JSON ('geometry') or a parameter file (its first line) is for."""
    if path.lower().endswith(".json"):
        with open(path) as fh:
            return json.load(fh).get("geometry")
    with open(path) as fh:
        first = fh.readline().lower()
    return "hull" if "hull" in first else "blade" if "blade" in first else None


class VarRow:
    """free | name | low | slider | high | value | where, for one variable."""

    def __init__(self, app, parent, r, spec):
        self.app, self.name, self.digits = app, spec.name, spec.digits
        ad = app.ad
        self._busy = False
        self.free = tk.BooleanVar(value=ad.is_free(self.name))
        ttk.Checkbutton(parent, variable=self.free, command=self.on_free).grid(row=r, column=0)
        self.label = ttk.Label(parent, text=spec.label, width=12)
        self.label.grid(row=r, column=1, sticky="w")
        self.lo = ttk.Entry(parent, width=8, justify="right")
        self.hi = ttk.Entry(parent, width=8, justify="right")
        lo, hi = ad.bounds(self.name)
        self.scale = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal", length=120, command=self.on_scale)
        self.val = ttk.Entry(parent, width=9, justify="right")
        self.info = ttk.Label(parent, text="", foreground=INK2, width=9)
        for c, w in enumerate((self.lo, self.scale, self.hi, self.val, self.info), start=2):
            w.grid(row=r, column=c, sticky="ew", padx=1, pady=1)
        for e, fn in ((self.lo, self.on_bounds), (self.hi, self.on_bounds), (self.val, self.on_value)):
            e.bind("<Return>", fn)
            e.bind("<FocusOut>", fn)
        self.refresh()

    def refresh(self, slider=True):
        ad = self.app.ad
        v = ad.get(self.name)
        lo, hi = ad.bounds(self.name)
        self._busy = True
        try:
            self.scale.configure(from_=lo, to=hi)
            if slider:
                self.scale.set(v)
            for e, x in ((self.lo, lo), (self.hi, hi), (self.val, v)):
                if self.app.root.focus_get() is not e:
                    e.delete(0, "end")
                    e.insert(0, f"{x:.{self.digits}f}")
            self.info.configure(text=ad.row_info(self.name))
            self.label.configure(foreground="black" if self.free.get() else INK2)
        finally:
            self._busy = False

    def on_scale(self, s):
        if not self._busy:
            self.app.set_value(self.name, float(s), self)

    def on_value(self, _event=None):
        try:
            v = float(self.val.get())
        except ValueError:
            return self.refresh()
        if abs(v - self.app.ad.get(self.name)) > 0.5 * 10.0 ** (-self.digits):
            self.app.set_value(self.name, v, None)

    def on_bounds(self, _event=None):
        try:
            lo, hi = float(self.lo.get()), float(self.hi.get())
        except ValueError:
            return self.refresh()
        if (lo, hi) != tuple(round(b, self.digits) for b in self.app.ad.bounds(self.name)):
            self.app.guard(lambda: self.app.ad.set_bounds(self.name, lo, hi))
            self.app.after_change()

    def on_free(self):
        self.app.ad.set_free(self.name, self.free.get())
        self.app.refresh_labels()
        self.app.schedule_update()


class App:
    """The tool's window for one geometry, or for the vehicle (geometry 'vehicle'): its components (VEHICLE,
    those whose adapter is here) one at a time in the left panel and the Design and 3D sections tabs, and
    all of them in the Vehicle tab. space and params: a path, or for the vehicle a list of them (each goes to
    the component it is for)."""

    def __init__(self, root, adapter_name="rudder", space=None, params=None):
        self.root = root
        self.vehicle = adapter_name == "vehicle"
        self.kinds = [k for k in VEHICLE if available(k)] if self.vehicle else [adapter_name]
        if self.vehicle and "hull" not in self.kinds:
            raise RuntimeError("the vehicle needs xgeom_hull.py (the X_geom folder)")
        self.start = {k: {"params": None, "space": None} for k in self.kinds}
        for key, paths in (("params", params), ("space", space)):
            for path in ([paths] if isinstance(paths, str) else list(paths or [])):
                kind = file_kind(path) if self.vehicle else adapter_name
                if kind not in self.start:
                    raise ValueError(f"{path}: not a set-up or parameter file of " + ", ".join(self.kinds))
                self.start[kind][key] = path
        self.adapters, self.errors, self.frames, self.picked, self._vcache = {}, {}, {}, {}, {}
        self.kind, self.ad, self.placement = None, None, None
        self.curve = tk.StringVar()
        self.rows = []
        self._pending = False
        self._job = None
        self._note = None                                   # a message the next view update shows instead of ok
        self.step = None
        self._build_layout()
        self.switch(self.kinds[0])

    # ------------------------------------------------------------ components
    def _adapter(self, kind):
        """The component's adapter, made when first needed with its parameter file and set-up; for the
        vehicle None if it cannot be made (the error is kept in self.errors)."""
        if kind in self.adapters:
            return self.adapters[kind]
        if kind in self.errors:
            return None
        st = self.start[kind]
        try:
            ad = load_adapter(kind, st["params"])
            if st["space"]:
                ad.load_space(st["space"])
        except Exception as exc:
            if not self.vehicle:
                raise
            traceback.print_exc()
            self.errors[kind] = f"{type(exc).__name__}: {exc}"
            return None
        self.adapters[kind] = ad
        return ad

    def _frames(self, kind):
        """A component's Design and 3D sections figures, made when it is first shown."""
        if kind not in self.frames:
            ad = self.adapters[kind]
            f = {"design": ttk.Frame(self.tab_design), "sk": ttk.Frame(self.tab_3d)}
            f["fig"] = Figure(figsize=(10, 7.5), dpi=100)
            f["canvas"] = FigureCanvasTkAgg(f["fig"], master=f["design"])
            NavigationToolbar2Tk(f["canvas"], f["design"]).update()
            f["canvas"].get_tk_widget().pack(fill="both", expand=True)
            f["views"] = ad.views(f["fig"])
            f["fig3"] = Figure(figsize=(10, 7.5), dpi=100)
            f["canvas3"] = FigureCanvasTkAgg(f["fig3"], master=f["sk"])
            f["canvas3"].get_tk_widget().pack(fill="both", expand=True)
            f["view3"] = ad.skeleton_view(f["fig3"])
            self.frames[kind] = f
        return self.frames[kind]

    def switch(self, kind):
        """Show a component: its left panel and its Design and 3D sections tabs."""
        if kind == self.kind:
            return
        if kind not in self.adapters:
            self.say(f"loading the {NAMES.get(kind, kind).lower()} ...", busy=True)
            self.root.update_idletasks()
        ad = self._adapter(kind)
        if ad is None:
            self.say(f"{NAMES[kind]}: not loaded: {self.errors[kind]}", error=True)
            if self.kind is not None:
                self.comp.set(self.kind)
            return
        if self.kind is not None:
            self.picked[self.kind] = self.curve.get()
            self.frames[self.kind]["design"].pack_forget()
            self.frames[self.kind]["sk"].pack_forget()
        self.kind, self.ad = kind, ad
        f = self._frames(kind)
        f["design"].pack(fill="both", expand=True)
        f["sk"].pack(fill="both", expand=True)
        self.fig, self.canvas, self.views = f["fig"], f["canvas"], f["views"]
        self.fig3, self.canvas3, self.view3 = f["fig3"], f["canvas3"], f["view3"]
        self.curve.set(self.picked.get(kind, ad.names[0]))
        if self.vehicle:
            self.comp.set(kind)
        self._title()
        self.rebuild_panel()
        self.update_views()

    def _title(self):
        self.root.title(f"XGeom design tool: {'vehicle, ' if self.vehicle else ''}{self.ad.title}")

    # ------------------------------------------------------------ layout
    def _build_layout(self):
        self.status = tk.Label(self.root, text="", anchor="w", justify="left", font=("TkDefaultFont", 12, "bold"),
                               padx=10, pady=6)
        self.status.pack(fill="x", side="bottom")                  # packed first: always in view
        self.status.bind("<Configure>", lambda e: self.status.configure(wraplength=max(e.width - 30, 200)))
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left, right = ttk.Frame(paned), ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        if self.vehicle:                                           # the component switcher
            bar = ttk.Frame(left)
            bar.pack(fill="x", side="top", padx=6, pady=(6, 2))
            ttk.Label(bar, text="Component", font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=(0, 8))
            self.comp = tk.StringVar()
            for k in self.kinds:
                ttk.Radiobutton(bar, text=NAMES[k], value=k, variable=self.comp,
                                command=lambda: self.switch(self.comp.get())).pack(side="left", padx=4)
            ttk.Separator(left, orient="horizontal").pack(fill="x", side="top", pady=(2, 0))
        canvas = tk.Canvas(left, width=610, highlightthickness=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self.panel = ttk.Frame(canvas)
        self.panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.panel, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.book = ttk.Notebook(right)
        self.book.pack(fill="both", expand=True)
        self.tab_design, self.tab_3d = ttk.Frame(self.book), ttk.Frame(self.book)
        self.book.add(self.tab_design, text="Design")
        self.book.add(self.tab_3d, text="3D sections")
        if self.vehicle:
            self.tab_vehicle = ttk.Frame(self.book)
            self.book.add(self.tab_vehicle, text="Vehicle")
            self._vehicle_layout()
        self.book.bind("<<NotebookTabChanged>>", lambda e: self.schedule_update())

    def _vehicle_layout(self):
        """The Vehicle tab: the placement settings above the 3D view (xgeom_vehicle)."""
        import xgeom_vehicle as XV
        tab = self.tab_vehicle
        bar = ttk.Frame(tab)
        bar.pack(fill="x", side="top", padx=6, pady=(6, 2))
        num = dict(from_=-100.0, to=100.0, increment=0.005, width=8)
        rows = (("Propeller", (("prop_dx", "plane at x - L (m)", num),
                               ("prop_d", "D (m)", dict(num, from_=0.001))), "show"),
                ("Rudder", (("rudder_dx", "root TE at x - L (m)", num),
                            ("rudders", "rudders", dict(from_=1, to=XV.MAX_RUDDERS, increment=1, width=4)),
                            ("rudder_angle", "first at (deg)", dict(from_=-360.0, to=720.0, increment=15.0, width=6)),
                            ("rudder_scale", "scale", dict(from_=0.01, to=100.0, increment=0.05, width=6))), None))
        self.vset = {}
        for r, (title, items, extra) in enumerate(rows):
            ttk.Label(bar, text=title, font=("TkDefaultFont", 10, "bold")).grid(row=r, column=0, sticky="w")
            c = 1
            for key, text, kw in items:
                ttk.Label(bar, text=text).grid(row=r, column=c, sticky="e", padx=(12, 3))
                sb = ttk.Spinbox(bar, command=self.on_vehicle, **kw)
                sb.grid(row=r, column=c + 1, sticky="w", pady=1)
                sb.bind("<Return>", self.on_vehicle)
                sb.bind("<FocusOut>", self.on_vehicle)
                self.vset[key] = sb
                c += 2
            if extra == "show":
                self.vshow = tk.StringVar(value="vehicle")
                ttk.Label(bar, text="show").grid(row=r, column=c, sticky="e", padx=(24, 3))
                for i, (val, text) in enumerate((("vehicle", "the vehicle"), ("stern", "the stern"))):
                    ttk.Radiobutton(bar, text=text, value=val, variable=self.vshow,
                                    command=self.on_vehicle).grid(row=r, column=c + 1 + i, sticky="w", padx=2)
        self.figv = Figure(figsize=(10, 7), dpi=100)
        self.canvasv = FigureCanvasTkAgg(self.figv, master=tab)
        self.canvasv.get_tk_widget().pack(fill="both", expand=True)
        self.viewv = XV.VehicleView(self.figv)

    def rebuild_panel(self):
        """(Re)create the left panel: the geometry's own boxes (PANELS), the
        picked curve and the output."""
        case = self.case.get() if hasattr(self, "case") and self.case.winfo_exists() else "gui"
        for w in self.panel.winfo_children():
            w.destroy()
        self.rows, self.picks, self.pinned = [], {}, []
        ad = self.ad
        ttk.Label(self.panel, text=ad.title, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 0))
        self.free_label = ttk.Label(self.panel, text="", foreground=INK2, wraplength=590, justify="left")
        self.free_label.pack(anchor="w", padx=6)
        PANELS[self.kind](self)

        c = self.curve.get()
        if c not in ad.names:
            c = ad.names[0]
            self.curve.set(c)
        box = ttk.LabelFrame(self.panel, text=f"Curve {getattr(ad, 'curve_label', str)(c)}")
        box.pack(fill="x", padx=6, pady=4)
        ttk.Label(box, text=ad.describe(c), foreground=INK2, wraplength=560, justify="left").pack(fill="x", padx=4)
        top = ttk.Frame(box)
        top.pack(fill="x", pady=2)
        two = getattr(ad, "segment_choice", False) and ad.segments(c) == 2
        if getattr(ad, "segment_choice", False):
            ttk.Label(top, text="segments").pack(side="left", padx=(4, 2))
            self.nseg = tk.IntVar(value=ad.segments(c))
            for n in (1, 2):
                ttk.Radiobutton(top, text=str(n), value=n, variable=self.nseg,
                                command=self.on_segments).pack(side="left")
        if two:
            ttk.Label(top, text="P1 .. P7: P2 = P3 and P5 = P6 at P4's value (zero slope at P4)",
                      foreground=INK2).pack(side="left", padx=8)
        else:
            lo, hi = getattr(ad, "order_range", (1, 12))
            ttk.Label(top, text="control points").pack(side="left",
                                                       padx=(10 if getattr(ad, "segment_choice", False) else 4, 2))
            self.nctrl = ttk.Spinbox(top, from_=lo + 1, to=hi + 1, width=4, command=self.on_order)
            self.nctrl.set(ad.order(c) + 1)
            self.nctrl.bind("<Return>", self.on_order)
            self.nctrl.pack(side="left")
        if ad.joins:
            self.tangent = tk.BooleanVar(value=ad.design.curves[c].join == "G1")
            ttk.Checkbutton(top, text="C1 on the original's tangent at H (slope matched)", variable=self.tangent,
                            command=self.on_tangent).pack(side="left", padx=10)
        grid = ttk.Frame(box)
        grid.pack(fill="x", pady=2)
        self._header(grid)
        for r, spec in enumerate(ad.curve_rows(c), start=1):
            if spec.kind == "var":
                self.rows.append(VarRow(self, grid, r, spec))
            else:
                ttk.Label(grid, text=spec.label, foreground=RED).grid(row=r, column=1, sticky="w")
                lab = ttk.Label(grid, text=spec.text, foreground=INK2)
                lab.grid(row=r, column=2, columnspan=5, sticky="w")
                self.pinned.append(lab)

        box = ttk.LabelFrame(self.panel, text="Output")
        box.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(box)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="case").pack(side="left", padx=(4, 2))
        self.case = ttk.Entry(row, width=16)
        self.case.insert(0, case)
        self.case.pack(side="left")
        ttk.Button(row, text="Save set-up", command=self.on_save).pack(side="left", padx=(6, 0))
        self.build_btn = ttk.Button(row, text=ad.build_label, command=self.on_build, width=8,
                                    state="disabled" if self._job is not None else "normal")
        self.build_btn.pack(side="left", padx=6)
        ttk.Button(row, text="OCC viewer", command=self.open_occ_viewer).pack(side="left")
        self.refresh_labels()

    def _pick(self, parent, r, c, name, text):
        rb = ttk.Radiobutton(parent, text=text, value=name, variable=self.curve, command=self.on_curve)
        rb.grid(row=r, column=c, sticky="w", padx=4)
        self.picks[name] = (rb, text)

    @staticmethod
    def _header(parent):
        for c, t in enumerate(("free", "", "low", "", "high", "value", "where")):
            ttk.Label(parent, text=t, foreground=INK2).grid(row=0, column=c, sticky="w")

    def refresh_labels(self):
        self.free_label.configure(text=self.ad.summary())
        if getattr(self, "sec_info", None) is not None and self.sec_info.winfo_exists():
            self.sec_info.configure(text=self.ad.sections_info())     # blade: the steps follow the root
        for name, (rb, text) in self.picks.items():                 # "held": no free variable on that curve
            rb.configure(text=text + ("  (held)" if self.ad.n_free(name)[0] == 0 else ""))
        texts = [s.text for s in self.ad.curve_rows(self.curve.get()) if s.kind == "pinned"]
        for lab, t in zip(self.pinned, texts):
            lab.configure(text=t)

    # ------------------------------------------------------------ changes
    def guard(self, fn):
        try:
            return fn()
        except Exception as exc:                                  # keep the tool alive
            self.say(f"{type(exc).__name__}: {exc}", error=True)

    def set_value(self, name, value, source):
        self.guard(lambda: self.ad.set(name, value))
        for row in self.rows:
            row.refresh(slider=row is not source)
        self.refresh_labels()
        self.schedule_update()

    def after_change(self, structure=False, views=True):
        if structure:
            self.rebuild_panel()
        else:
            for row in self.rows:
                row.refresh()
            self.refresh_labels()
        if views:
            self.schedule_update()

    def on_curve(self):
        self.rebuild_panel()
        self.schedule_update()

    def on_order(self, _event=None):
        c = self.curve.get()
        try:
            n = int(self.nctrl.get()) - 1
        except ValueError:
            return
        if n != self.ad.order(c):
            msg = self.guard(lambda: self.ad.set_order(c, n))
            self.after_change(structure=True)
            if isinstance(msg, str):
                self._note = f"{c}: {n + 1} control points, {msg}"

    def on_segments(self):
        """Blade: the picked curve as 1 or 2 segments, fitted to its present shape."""
        c, n = self.curve.get(), self.nseg.get()
        if n == self.ad.segments(c):
            return
        msg = self.guard(lambda: self.ad.set_segments(c, n))
        self.after_change(structure=True)
        if msg is not None:
            self._note = f"{c}: {n} segment{'s' if n == 2 else ''}, {msg}"

    def on_tangent(self):
        c = self.curve.get()
        self.guard(lambda: self.ad.set_tangent(c, self.tangent.get()))
        self.after_change(structure=True)

    def on_points(self, _event=None):
        try:
            n = int(self.npts.get()) - 1
        except ValueError:
            return
        if n != self.ad.degree:
            err = self.guard(lambda: self.ad.set_degree(n))
            self.after_change(structure=True)
            if err is not None:
                self._note = (f"sections now of degree {self.ad.degree} ({self.ad.degree + 1} control points); "
                              f"design carried over, largest change {err:.1e} (fraction of the chord)")

    def on_sections(self, _event=None):
        """Blade: the number of sections and the tip band from the Sections box."""
        s = self.ad.settings
        try:
            new = (int(self.sec["sections"].get()), float(self.sec["tip_band"].get()),
                   int(self.sec["tip_sections"].get()))
        except ValueError:
            new = None
        except tk.TclError:                                          # the box is gone (panel rebuilt)
            return
        if new and new != (s["sections"], s["tip_band"], s["tip_sections"]):
            if self.guard(lambda: self.ad.set_sections(*new) or True):
                self.schedule_update()
        for k, sb in self.sec.items():                              # what is used
            sb.set(f"{s[k]:.3f}" if k == "tip_band" else s[k])
        self.sec_info.configure(text=self.ad.sections_info())

    def on_blades(self, _event=None):
        """Blade: the number of blades from the Blade and hub box."""
        try:
            z = int(self.nblades.get())
        except ValueError:
            z = None
        except tk.TclError:                                          # the box is gone (panel rebuilt)
            return
        if z is not None and z != self.ad.params.blades:
            if self.guard(lambda: self.ad.set_blades(z) or True):
                self.refresh_labels()
                self.schedule_update()
        self.nblades.set(self.ad.params.blades)                      # what is used

    def on_elliptic(self):
        """Hull: elliptical sections (r' of its own) or circular ones."""
        self.guard(lambda: self.ad.set_elliptic(self.elliptic.get()))
        self.after_change(structure=True)

    def on_vehicle(self, _event=None):
        """The Vehicle tab's settings: read, checked and drawn."""
        pl = self.placement
        if pl is None:
            return
        try:
            new = {k: float(self.vset[k].get()) for k in ("prop_dx", "prop_d", "rudder_dx", "rudder_angle",
                                                          "rudder_scale")}
            new["rudders"] = int(float(self.vset["rudders"].get()))
        except (ValueError, tk.TclError):
            new = None
        n_max = self._xv().MAX_RUDDERS
        if new is not None and (new["prop_d"] <= 0.0 or new["rudder_scale"] <= 0.0 or not 1 <= new["rudders"] <= n_max):
            self.say(f"the propeller's D and the rudder's scale are positive; 1 to {n_max} rudders", error=True)
            new = None
        if new is not None:
            new["rudder_angle"] %= 360.0
            new["view"] = self.vshow.get()
            if any(getattr(pl, k) != v for k, v in new.items()):
                for k, v in new.items():
                    setattr(pl, k, v)
                self.schedule_update()
        self._fill_vehicle()

    @staticmethod
    def _xv():
        import xgeom_vehicle as XV
        return XV

    def _fill_vehicle(self):
        """The Vehicle tab's boxes show the placement in use."""
        pl = self.placement
        for k, sb in self.vset.items():
            v = getattr(pl, k)
            sb.set(f"{v:d}" if k == "rudders" else f"{v:g}" if k == "rudder_angle" else
                   f"{v:.2f}" if k == "rudder_scale" else f"{v:.3f}")
        self.vshow.set(pl.view)

    def say(self, text, error=False, busy=False):
        """The status bar: green ok, amber busy (a build running), red a problem
        or an error."""
        bg, fg = STATUS["error" if error else "busy" if busy else "ok"]
        self.status.configure(text=text, background=bg, foreground=fg)

    # ------------------------------------------------------------ views
    def schedule_update(self):
        if not self._pending:
            self._pending = True
            self.root.after(15, self.update_views)

    def update_views(self):
        self._pending = False
        tab = self.book.index(self.book.select())
        if self.vehicle and tab == VEHICLE_TAB:
            try:
                self._update_vehicle()
            except Exception as exc:
                self.say(f"vehicle view failed: {type(exc).__name__}: {exc}", error=True)
                traceback.print_exc()
            return
        three_d = tab == 1
        try:
            pv = self.ad.preview(skeleton=three_d)
            if three_d:
                self.view3.update(pv)
                self.canvas3.draw_idle()
            else:
                self.views.update(pv, self.curve.get())
                self.canvas.draw_idle()
            if pv["problems"]:
                self.say(f"problem (no {self.ad.build_label} until it is solved): " + "; ".join(pv["problems"]),
                         error=True)
            elif self._job is not None:
                self.say(self._job["message"], busy=True)
            else:
                self.say(self._note or f"ok ({pv['seconds'] * 1e3:.0f} ms)")
            self._note = None
        except Exception as exc:
            self.say(f"preview failed: {type(exc).__name__}: {exc}", error=True)
            traceback.print_exc()

    def _vehicle_part(self, kind):
        """A component's lines for the Vehicle tab (xgeom_vehicle.PARTS), made again only when its design or
        its parameter file in use (the blade's Z) has changed."""
        import xgeom_vehicle as XV
        ad = self.adapters[kind]
        key = (ad.design, getattr(ad, "params", None))
        hit = self._vcache.get(kind)
        if hit is None or any(a is not b for a, b in zip(hit[0], key)):
            self._vcache[kind] = (key, XV.PARTS[kind](ad))
        return self._vcache[kind][1]

    def _update_vehicle(self):
        """The Vehicle tab: every component as its design stands, placed on the hull."""
        import xgeom_vehicle as XV
        t0 = time.time()
        for k in self.kinds:
            if k not in self.adapters and k not in self.errors:
                self.say(f"loading the {NAMES[k].lower()} for the vehicle ...", busy=True)
                self.root.update_idletasks()
                self._adapter(k)
        if "hull" not in self.adapters:
            self.say("the vehicle needs the hull: " + self.errors.get("hull", "not loaded"), error=True)
            return
        parts = {k: self._vehicle_part(k) for k in self.kinds if k in self.adapters}
        if self.placement is None:
            self.placement = XV.default_placement(parts)
            self._fill_vehicle()
        self.viewv.update(parts, self.placement, self.kind)
        self.canvasv.draw_idle()
        problems = [f"{NAMES[k].lower()}: {q}" for k, part in parts.items() for q in part["problems"]]
        problems += [f"{NAMES[k].lower()} not loaded: {e}" for k, e in self.errors.items()]
        if problems:
            self.say("problem: " + "; ".join(problems), error=True)
        elif self._job is not None:
            self.say(self._job["message"], busy=True)
        else:
            self.say(f"vehicle ok ({(time.time() - t0) * 1e3:.0f} ms)")

    # ------------------------------------------------------------ build
    def on_build(self):
        if self._job is not None:
            return
        problems = self.ad.preview()["problems"]
        label = self.ad.build_label
        if problems:
            self.say(f"{label}: the design cannot be built: " + problems[0], error=True)
            return
        case = self.case.get().strip() or "gui"
        self.build_btn.configure(state="disabled")
        job = self._job = {"done": False, "result": None, "error": None, "t0": time.time(), "label": label,
                           "message": f"{label} running: writing {case} and building the CAD ..."}
        self.say(job["message"], busy=True)
        task = self.ad.build_task(case)                                 # the design as it is now

        def work():
            try:
                job["result"] = task()
            except Exception:
                job["error"] = traceback.format_exc()
            job["done"] = True
        threading.Thread(target=work, daemon=True).start()
        self.root.after(200, self._poll)

    def _poll(self):
        job = self._job
        label = job["label"]
        if not job["done"]:
            self.build_btn.configure(text=f"{time.time() - job['t0']:.0f} s")     # the build's time so far
            self.root.after(200, self._poll)
            return
        self._job = None
        self.build_btn.configure(state="normal", text=self.ad.build_label)
        if job["error"]:
            print(job["error"])
            self.say(f"{label} failed: " + job["error"].strip().splitlines()[-1], error=True)
            return
        res = job["result"]
        print("\n".join(res["report"]))
        if res.get("view"):
            self.step = res["view"]
        self.say(f"{label} done in {time.time() - job['t0']:.1f} s: {res['message']}", error=res["error"])

    def on_save(self):
        case = self.case.get().strip() or "gui"
        path = self.guard(lambda: self.ad.save_space(os.path.join(self.ad.out_dir, f"{case}_design_space.json")))
        if path:
            self.say(f"set-up saved: {os.path.abspath(path)}")

    def on_load_params(self):
        path = filedialog.askopenfilename(title=f"{NAMES.get(self.kind, self.kind)} parameter file", initialdir=HERE,
                                          filetypes=[("parameter files", "*.dat"), ("all files", "*")])
        if path:
            self.say(f"fitting the curves to {os.path.basename(path)} ...")
            self.root.update_idletasks()
            if self.guard(lambda: self.ad.load_params(path) or True):
                self._title()
                self.after_change(structure=True)
                self.say(f"loaded {path}")

    def on_save_params(self):
        path = filedialog.asksaveasfilename(title="Save the design as a parameter file", initialdir=self.ad.out_dir,
                                            initialfile=f"{self.case.get().strip() or 'gui'}_params.dat",
                                            defaultextension=".dat", filetypes=[("parameter files", "*.dat")])
        if path and self.guard(lambda: self.ad.save_params(path)):
            self.say(f"parameter file written: {path}")

    def open_occ_viewer(self):
        if not self.step:
            self.say(f"press {self.ad.build_label} first: the viewer shows the CAD it writes", error=True)
            return
        subprocess.Popen([sys.executable, os.path.join(HERE, "xcad_loft.py"), "--view-step", self.step], cwd=HERE)
        self.say(f"OCC viewer: {os.path.basename(self.step)} (its own window, a few seconds)")


def rudder_panel(app):
    """The rudder's boxes: the region H and the section control points."""
    ad = app.ad
    box = ttk.LabelFrame(app.panel, text="Region")
    box.pack(fill="x", padx=6, pady=4)
    app._header(box)
    app.rows.append(VarRow(app, box, 1, Row("var", "H (mm)", "H", 3)))

    box = ttk.LabelFrame(app.panel, text="Section control points (pick the curve to edit)")
    box.pack(fill="x", padx=6, pady=4)
    top = ttk.Frame(box)
    top.pack(fill="x", pady=2)
    ttk.Label(top, text="control points").pack(side="left", padx=(4, 2))
    app.npts = ttk.Spinbox(top, from_=5, to=13, width=4, command=app.on_points)
    app.npts.set(ad.degree + 1)
    app.npts.bind("<Return>", app.on_points)
    app.npts.pack(side="left")
    ttk.Label(top, text=f"P0 .. P{ad.degree}, Bezier of degree {ad.degree}", foreground=INK2).pack(side="left", padx=6)
    grid = ttk.Frame(box)
    grid.pack(fill="x", padx=4, pady=2)
    app._pick(grid, 0, 1, "LE", "LE x (planform)")
    ttk.Label(grid, text="P0   (0, 0), the LE", foreground=INK2).grid(row=1, column=0, columnspan=3, sticky="w")
    for i in range(1, ad.degree):
        ttk.Label(grid, text=f"P{i}", width=4).grid(row=i + 1, column=0, sticky="w")
        app._pick(grid, i + 1, 1, f"P{i}x", "x")
        app._pick(grid, i + 1, 2, f"P{i}z", "z")
    ttk.Label(grid, text=f"P{ad.degree}   (1, 0), the sharp TE", foreground=INK2).grid(
        row=ad.degree + 1, column=0, columnspan=3, sticky="w")


def blade_panel(app):
    """The blade's boxes: the parameter file, the blade and hub, the sections
    and the distributions."""
    import blade_modify as BM
    ad = app.ad
    box = ttk.LabelFrame(app.panel, text="Parameter file")
    box.pack(fill="x", padx=6, pady=4)
    ttk.Label(box, text=os.path.relpath(ad.params.path, HERE) if ad.params.path.startswith(HERE) else ad.params.path,
              wraplength=560).pack(anchor="w", padx=4)
    p = ad.params
    hub_h = f"not given ({BM.HUB_HEIGHT:g} m)" if p.hub_height is None else f"{p.hub_height:g} m"
    if p.airfoil:
        import airfoils
        sections = "airfoil " + airfoils.describe(p.airfoil)
    else:
        sections = f"sections: {p.section_table}"
    z_file = getattr(ad, "file_blades", p.blades)
    ttk.Label(box, text=f"D {p.diameter:g} m, {z_file} blades, root r/R {p.root_r:g}, hub height {hub_h}, "
                        f"{len(p.r)} rows; {sections}", foreground=INK2, wraplength=560).pack(anchor="w", padx=4)
    row = ttk.Frame(box)
    row.pack(fill="x", pady=3)
    ttk.Button(row, text="Load ...", command=app.on_load_params).pack(side="left", padx=4)
    ttk.Button(row, text="Save ...", command=app.on_save_params).pack(side="left")
    ttk.Label(row, text="the design as a parameter file", foreground=INK2).pack(side="left", padx=6)

    box = ttk.LabelFrame(app.panel, text="Blade and hub (the curves stretch over the span between them)")
    box.pack(fill="x", padx=6, pady=4)
    app._header(box)
    specs = ad.global_rows()
    for r, spec in enumerate(specs, start=1):
        app.rows.append(VarRow(app, box, r, spec))
    row = ttk.Frame(box)
    row.grid(row=len(specs) + 1, column=0, columnspan=7, sticky="w", padx=2, pady=(4, 2))
    ttk.Label(row, text="blades Z").pack(side="left", padx=(2, 4))
    app.nblades = ttk.Spinbox(row, from_=BM.BLADES[0], to=BM.BLADES[1], width=4, command=app.on_blades)
    app.nblades.set(p.blades)
    app.nblades.bind("<Return>", app.on_blades)
    app.nblades.bind("<FocusOut>", app.on_blades)
    app.nblades.pack(side="left")
    ttk.Label(row, text=f"the file has {z_file}; the clearance check, the propeller tab, the vehicle and the "
                        f"CAD's hub sector follow it", foreground=INK2, wraplength=470,
              justify="left").pack(side="left", padx=6)

    box = ttk.LabelFrame(app.panel, text="Sections (the XCAD points file and the 3D sections tab; 53 points each)")
    box.pack(fill="x", padx=6, pady=4)
    row = ttk.Frame(box)
    row.pack(fill="x", pady=2)
    app.sec = {}
    for key, text, kw in (("sections", "sections", dict(from_=3, to=999, width=5)),
                          ("tip_sections", "the last", dict(from_=2, to=998, width=4)),
                          ("tip_band", "from r/R", dict(from_=0.5, to=0.99, increment=0.01, width=6))):
        ttk.Label(row, text=text).pack(side="left", padx=(4, 2))
        sb = ttk.Spinbox(row, command=app.on_sections, **kw)
        sb.pack(side="left")
        sb.bind("<Return>", app.on_sections)
        sb.bind("<FocusOut>", app.on_sections)
        app.sec[key] = sb
    ttk.Label(row, text="to the tip").pack(side="left", padx=4)
    app.sec_info = ttk.Label(box, foreground=INK2, wraplength=560, justify="left")
    app.sec_info.pack(anchor="w", padx=4)
    app.on_sections()                                                # fills the boxes

    box = ttk.LabelFrame(app.panel, text="Distributions (pick the curve to edit)")
    box.pack(fill="x", padx=6, pady=4)
    grid = ttk.Frame(box)
    grid.pack(fill="x", padx=4, pady=2)
    for i, c in enumerate(ad.names):
        app._pick(grid, i // 3, i % 3, c, f"{c}  {BM.UNITS[c]}")


def hull_panel(app):
    """The hull's boxes: the parameter file, the body (lengths, radii, elliptical sections) and the curves."""
    ad = app.ad
    p = ad.params
    box = ttk.LabelFrame(app.panel, text="Parameter file")
    box.pack(fill="x", padx=6, pady=4)
    where = p.path or "(SUBOFF's equations: suboff_hull_params.dat is not here)"
    ttk.Label(box, text=os.path.relpath(where, HERE) if where.startswith(HERE) else where,
              wraplength=560).pack(anchor="w", padx=4)
    ttk.Label(box, text=f"L {p.length:g} m: nose {p.nose:g}, middle {p.middle:g}, tail {p.tail:g}, cap {p.cap:g} m; "
                        f"{len(p.x)} stations, " + ("axisymmetric" if p.axisymmetric else
                                                   "elliptical sections (r and r')"),
              foreground=INK2, wraplength=560).pack(anchor="w", padx=4)
    row = ttk.Frame(box)
    row.pack(fill="x", pady=3)
    ttk.Button(row, text="Load ...", command=app.on_load_params).pack(side="left", padx=4)
    ttk.Button(row, text="Save ...", command=app.on_save_params).pack(side="left")
    ttk.Label(row, text="the design as a parameter file", foreground=INK2).pack(side="left", padx=6)

    box = ttk.LabelFrame(app.panel, text="Body: the parts' lengths and the radii (m)")
    box.pack(fill="x", padx=6, pady=4)
    app._header(box)
    specs = ad.global_rows()
    for r, spec in enumerate(specs, start=1):
        app.rows.append(VarRow(app, box, r, spec))
    app.elliptic = tk.BooleanVar(value=ad.elliptic)
    ttk.Checkbutton(box, text="elliptical sections: the half-width r' has radii and curves of its own",
                    variable=app.elliptic, command=app.on_elliptic).grid(row=len(specs) + 1, column=0, columnspan=7,
                                                                         sticky="w", padx=2, pady=(4, 2))

    box = ttk.LabelFrame(app.panel, text="Curves (pick the curve to edit)")
    box.pack(fill="x", padx=6, pady=4)
    grid = ttk.Frame(box)
    grid.pack(fill="x", padx=4, pady=2)
    for i, c in enumerate(ad.names):
        text = (c.replace("_", " ") if ad.design.axisymmetric else
                {"nose_r": "nose r (XZ)", "tail_r": "tail r (XZ)", "nose_rp": "nose r' (XY)",
                 "tail_rp": "tail r' (XY)"}[c])
        app._pick(grid, i // 2, i % 2, c, text)


PANELS = {"rudder": rudder_panel, "blade": blade_panel, "hull": hull_panel}


def main():
    default = "vehicle" if available("hull") else "rudder"
    ap = argparse.ArgumentParser(description="XGeom design tool")
    ap.add_argument("geometry", nargs="?", default=default, choices=sorted(ADAPTERS) + ["vehicle"],
                    help=f"what to design (default here: {default})")
    ap.add_argument("--space", action="append", default=[],
                    help="a <case>_design_space.json to start from (the vehicle: one per component)")
    ap.add_argument("--params", action="append", default=[],
                    help="the blade's or the hull's parameter file to start from (the vehicle: either, or both)")
    args = ap.parse_args()
    if args.geometry == "vehicle" and not available("hull"):
        ap.error("the vehicle needs xgeom_hull.py (the X_geom folder)")
    if args.geometry != "vehicle":
        if not available(args.geometry):
            ap.error(f"{ADAPTERS[args.geometry][0]}.py is not in this folder")
        if len(args.space) > 1 or len(args.params) > 1:
            ap.error("one --space and one --params for one geometry")
    space = [os.path.abspath(s) for s in args.space]
    params = [os.path.abspath(p) for p in args.params]
    if args.geometry != "vehicle":
        space, params = (space or [None])[0], (params or [None])[0]
    os.chdir(HERE)                                   # the adapters' files are relative to this folder
    root = tk.Tk()
    root.geometry(f"{min(1600, root.winfo_screenwidth() - 40)}x{min(940, root.winfo_screenheight() - 90)}+10+10")
    App(root, args.geometry, space, params)
    root.mainloop()


if __name__ == "__main__":
    main()
