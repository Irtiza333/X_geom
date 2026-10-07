"""
xgeom_tool.py

XGeom design tool: set up a geometry's design space with sliders and see the
shape change live. The build button writes the case and its CAD; OCC viewer
shows the solid.

    python xgeom_tool.py                    the rudder
    python xgeom_tool.py rudder --space outputs/modified/gui_design_space.json
                                            start from a saved set-up
    python xgeom_tool.py blade              the propeller blade of blade_control.PARAMS
    python xgeom_tool.py blade --params my_blade.dat
                                            the blade of another parameter file
    python xgeom_tool.py blade --space outputs/blade/gui_design_space.json

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
Each variable:  free | name | low | slider | high | value | where
    free      ticked: a design variable within low .. high; unticked: held
    d         the position of a control point, as the fraction d of the span
              left (rudder: below the point above it; blade: up to the tip;
              blade, 2 segments: d1 from P4 to the root, d2 from P4 to the tip)
    dx        rudder LE control point: x offset from the original LE, mm (negative forward)
    value     rudder: a section coordinate, fraction of the sharp chord;
              blade: the distribution's value (P/D, c/D, t/c, f/c, deg, rake/D)
Right panel
    Design       rudder: planform, the picked curve against the original, and in
                 3D five sections from the root to H with their control polygons
                 and each control point's track along the span. Blade: the
                 expanded outline, the picked curve against the parameter file,
                 and in 3D the blade as the CAD builds it. The 2D plots have the
                 span on the x axis (height, r, r/R). Drag the 3D view to turn it.
    3D sections  the whole rudder as its sections: modified (blue), the cut at H
                 (red), unchanged (grey); or all blades of the propeller.
Save set-up writes the design space as it stands (every variable with its value,
bounds and free flag, each curve's control points and order, the settings and
input files) to <output folder>/<case>_design_space.json; rudder_modify.
load_design_space or blade_modify.load_blade_space reads it for the optimiser,
--space for the tool.
XCAD (rudder) writes the case to the output folder (XCAD file, section table,
design and the same set-up JSON), lofts the XCAD file into a solid and writes it
as STEP. CAD (blade) writes the case (parameter file, XCAD points, design and
set-up JSON) and the DRDC five-surface blade with the hub sector as IGES and
STEP. OCC viewer opens the last STEP (needs pythonocc-core).
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

ADAPTERS = {"rudder": ("xgeom_rudder", "RudderAdapter"), "blade": ("xgeom_blade", "BladeAdapter")}
INK2, RED = "#52514e", "#e34948"
STATUS = {"ok": ("#dfeedd", "black"), "busy": ("#f6d58e", "black"), "error": ("#c62f2e", "white")}  # bar colours


def load_adapter(name, params=None):
    module, cls = ADAPTERS[name]
    return getattr(importlib.import_module(module), cls)(**({"params": params} if params else {}))


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
    def __init__(self, root, adapter_name="rudder", space=None, params=None):
        self.root = root
        self.kind = adapter_name
        self.ad = load_adapter(adapter_name, params)
        if space:
            self.ad.load_space(space)
        root.title(f"XGeom design tool: {self.ad.title}")
        self.curve = tk.StringVar(value=self.ad.names[0])
        self.rows = []
        self._pending = False
        self._job = None
        self._note = None                                   # a message the next view update shows instead of ok
        self.step = None
        self._build_layout()
        self.rebuild_panel()
        self.update_views()

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
        tab = ttk.Frame(self.book)
        self.book.add(tab, text="Design")
        self.fig = Figure(figsize=(10, 7.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        NavigationToolbar2Tk(self.canvas, tab).update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.views = self.ad.views(self.fig)
        tab = ttk.Frame(self.book)
        self.book.add(tab, text="3D sections")
        self.fig3 = Figure(figsize=(10, 7.5), dpi=100)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=tab)
        self.canvas3.get_tk_widget().pack(fill="both", expand=True)
        self.view3 = self.ad.skeleton_view(self.fig3)
        self.book.bind("<<NotebookTabChanged>>", lambda e: self.schedule_update())

    def rebuild_panel(self):
        """(Re)create the left panel: the geometry's own boxes (PANELS), the
        picked curve and the output."""
        case = self.case.get() if hasattr(self, "case") else "gui"
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
        box = ttk.LabelFrame(self.panel, text=f"Curve {c}")
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
            ttk.Label(top, text="control points").pack(side="left", padx=(10 if hasattr(self, "nseg") else 4, 2))
            self.nctrl = ttk.Spinbox(top, from_=2, to=13, width=4, command=self.on_order)
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
        three_d = self.book.index(self.book.select()) == 1
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
        job = self._job = {"done": False, "result": None, "error": None, "t0": time.time(),
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
        label = self.ad.build_label
        if not job["done"]:
            self.build_btn.configure(text=f"{time.time() - job['t0']:.0f} s")     # the build's time so far
            self.root.after(200, self._poll)
            return
        self._job = None
        self.build_btn.configure(state="normal", text=label)
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
        path = filedialog.askopenfilename(title="Blade parameter file", initialdir=HERE,
                                          filetypes=[("parameter files", "*.dat"), ("all files", "*")])
        if path:
            self.say(f"fitting the curves to {os.path.basename(path)} ...")
            self.root.update_idletasks()
            if self.guard(lambda: self.ad.load_params(path) or True):
                self.root.title(f"XGeom design tool: {self.ad.title}")
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
    ttk.Label(box, text=f"D {p.diameter:g} m, {p.blades} blades, root r/R {p.root_r:g}, hub height {hub_h}, "
                        f"{len(p.r)} rows; {sections}", foreground=INK2, wraplength=560).pack(anchor="w", padx=4)
    row = ttk.Frame(box)
    row.pack(fill="x", pady=3)
    ttk.Button(row, text="Load ...", command=app.on_load_params).pack(side="left", padx=4)
    ttk.Button(row, text="Save ...", command=app.on_save_params).pack(side="left")
    ttk.Label(row, text="the design as a parameter file", foreground=INK2).pack(side="left", padx=6)

    box = ttk.LabelFrame(app.panel, text="Blade and hub (the curves stretch over the span between them)")
    box.pack(fill="x", padx=6, pady=4)
    app._header(box)
    for r, spec in enumerate(ad.global_rows(), start=1):
        app.rows.append(VarRow(app, box, r, spec))

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


PANELS = {"rudder": rudder_panel, "blade": blade_panel}


def main():
    ap = argparse.ArgumentParser(description="XGeom design tool")
    ap.add_argument("geometry", nargs="?", default="rudder", choices=sorted(ADAPTERS))
    ap.add_argument("--space", help="a <case>_design_space.json to start from")
    ap.add_argument("--params", help="blade: the parameter file to start from")
    args = ap.parse_args()
    space = os.path.abspath(args.space) if args.space else None
    params = os.path.abspath(args.params) if args.params else None
    os.chdir(HERE)                                   # the adapters' files are relative to this folder
    root = tk.Tk()
    root.geometry(f"{min(1600, root.winfo_screenwidth() - 40)}x{min(940, root.winfo_screenheight() - 90)}+10+10")
    App(root, args.geometry, space, params)
    root.mainloop()


if __name__ == "__main__":
    main()
