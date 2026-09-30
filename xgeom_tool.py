"""
xgeom_tool.py

XGeom design tool: set up a geometry's design space with sliders and see the
shape change live. XCAD writes the XCAD point file and the CAD solid (STEP);
OCC viewer shows the solid.

    python xgeom_tool.py                    the rudder (the only geometry so far)
    python xgeom_tool.py rudder --space outputs/modified/gui_design_space.json
                                            start from a design space an XCAD run wrote

Left panel
    H         the height of the modified region; its low bound is H_min.
    Section control points
              the number of control points of the section's Bezier half-
              section, P0 .. Pn (P0 = (0, 0) the LE and Pn = (1, 0) the sharp TE
              stay put; adding one keeps the shape and adds that point's x and z
              curves). Pick the LE or the x or z of an inner point to edit the
              spanwise curve it follows; the number in brackets is how many of
              that curve's variables are free. P1 x is held at 0 by default
              (a round LE).
    Curve     the picked curve's control points C0 .. Cm, from H (C0, pinned to
              the original) down to the root (Cm); how many there are; whether
              C1 sits on the original's tangent at H (slope matched).
    Each variable:  free | name | low | slider | high | value | where
      free    ticked: a design variable within low .. high; unticked: held
      d       the height of a control point, as the fraction d of the height
              left below the point above it (keeps the points in order)
      dx      LE control point: x offset from the original LE, mm (negative forward)
      value   a section coordinate, fraction of the sharp chord
Right panel
    Design       planform, the picked curve against the original, and the
                 sections at the root, H / 2 and H with their control points.
    3D sections  the whole rudder as its sections: modified (blue), the cut at H
                 (red), unchanged (grey); drag to turn it.
XCAD writes the case to the output folder (XCAD file, section table, design and
design-space JSON), lofts the XCAD file into a solid and writes it as STEP.
OCC viewer opens the last STEP (needs pythonocc-core).

Other geometries plug in as adapters with the methods of
xgeom_rudder.RudderAdapter, registered in ADAPTERS.
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
from tkinter import ttk                                              # noqa: E402

import matplotlib                                                    # noqa: E402
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure                                 # noqa: E402

ADAPTERS = {"rudder": ("xgeom_rudder", "RudderAdapter")}
INK2, RED = "#52514e", "#e34948"


def load_adapter(name):
    module, cls = ADAPTERS[name]
    return getattr(importlib.import_module(module), cls)()


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
            self.hi.configure(state="disabled" if self.name == "H" else "normal")
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
        self.app.after_change(views=False)


class App:
    def __init__(self, root, adapter_name="rudder", space=None):
        self.root = root
        self.ad = load_adapter(adapter_name)
        if space:
            self.ad.load_space(space)
        root.title(f"XGeom design tool: {self.ad.title}")
        self.curve = tk.StringVar(value="LE")
        self.rows = []
        self._pending = False
        self._job = None
        self.step = None
        self._build_layout()
        self.rebuild_panel()
        self.update_views()

    # ------------------------------------------------------------ layout
    def _build_layout(self):
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

        self.status = ttk.Label(self.root, text="", anchor="w")
        self.status.pack(fill="x", side="bottom", padx=6, pady=2)

    def rebuild_panel(self):
        """(Re)create the left panel from the adapter."""
        case = self.case.get() if hasattr(self, "case") else "gui"
        for w in self.panel.winfo_children():
            w.destroy()
        self.rows = []
        ad = self.ad
        ttk.Label(self.panel, text=ad.title, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=6, pady=(6, 0))
        self.free_label = ttk.Label(self.panel, text="", foreground=INK2)
        self.free_label.pack(anchor="w", padx=6)

        box = ttk.LabelFrame(self.panel, text="Region")
        box.pack(fill="x", padx=6, pady=4)
        self._header(box)
        from xgeom_rudder import Row
        self.rows.append(VarRow(self, box, 1, Row("var", "H (mm)", "H", 3)))

        box = ttk.LabelFrame(self.panel, text="Section control points (pick the curve to edit)")
        box.pack(fill="x", padx=6, pady=4)
        top = ttk.Frame(box)
        top.pack(fill="x", pady=2)
        ttk.Label(top, text="control points").pack(side="left", padx=(4, 2))
        self.npts = ttk.Spinbox(top, from_=5, to=13, width=4, command=self.on_points)
        self.npts.set(ad.degree + 1)
        self.npts.bind("<Return>", self.on_points)
        self.npts.pack(side="left")
        ttk.Label(top, text=f"P0 .. P{ad.degree}, Bezier of degree {ad.degree}", foreground=INK2).pack(side="left", padx=6)
        grid = ttk.Frame(box)
        grid.pack(fill="x", padx=4, pady=2)
        self._pick(grid, 0, 1, "LE", "LE x (planform)")
        ttk.Label(grid, text="P0   (0, 0), the LE", foreground=INK2).grid(row=1, column=0, columnspan=3, sticky="w")
        for i in range(1, ad.degree):
            ttk.Label(grid, text=f"P{i}", width=4).grid(row=i + 1, column=0, sticky="w")
            self._pick(grid, i + 1, 1, f"P{i}x", "x")
            self._pick(grid, i + 1, 2, f"P{i}z", "z")
        ttk.Label(grid, text=f"P{ad.degree}   (1, 0), the sharp TE", foreground=INK2).grid(
            row=ad.degree + 1, column=0, columnspan=3, sticky="w")

        c = self.curve.get()
        if c not in ad.names:
            c = "LE"
            self.curve.set(c)
        cd = ad.design.curves[c]
        box = ttk.LabelFrame(self.panel, text=f"Curve {c}")
        box.pack(fill="x", padx=6, pady=4)
        import rudder_modify as RM
        ttk.Label(box, text=RM.describe(c), foreground=INK2, wraplength=560, justify="left").pack(fill="x", padx=4)
        top = ttk.Frame(box)
        top.pack(fill="x", pady=2)
        ttk.Label(top, text="control points").pack(side="left", padx=(4, 2))
        self.nctrl = ttk.Spinbox(top, from_=2, to=13, width=4, command=self.on_order)
        self.nctrl.set(cd.order + 1)
        self.nctrl.bind("<Return>", self.on_order)
        self.nctrl.pack(side="left")
        self.tangent = tk.BooleanVar(value=cd.join == "G1")
        ttk.Checkbutton(top, text="C1 on the original's tangent at H (slope matched)", variable=self.tangent,
                        command=self.on_tangent).pack(side="left", padx=10)
        grid = ttk.Frame(box)
        grid.pack(fill="x", pady=2)
        self._header(grid)
        self.pinned = []
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
        self.xcad_btn = ttk.Button(row, text="XCAD", command=self.on_xcad, width=8)
        self.xcad_btn.pack(side="left", padx=6)
        ttk.Button(row, text="OCC viewer", command=self.open_occ_viewer).pack(side="left")
        self.refresh_labels()

    def _pick(self, parent, r, c, name, text):
        n, m = self.ad.n_free(name)
        ttk.Radiobutton(parent, text=f"{text} ({n})", value=name, variable=self.curve,
                        command=self.on_curve).grid(row=r, column=c, sticky="w", padx=4)

    @staticmethod
    def _header(parent):
        for c, t in enumerate(("free", "", "low", "", "high", "value", "where")):
            ttk.Label(parent, text=t, foreground=INK2).grid(row=0, column=c, sticky="w")

    def refresh_labels(self):
        n, m = self.ad.n_free()
        self.free_label.configure(text=f"{n} of {m} variables free; H {self.ad.design.H:.2f} mm, sections of "
                                       f"degree {self.ad.degree}")
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
        if n != self.ad.design.curves[c].order:
            self.guard(lambda: self.ad.set_order(c, n))
            self.after_change(structure=True)

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
                self.say(f"sections now of degree {self.ad.degree} ({self.ad.degree + 1} control points); "
                         f"design carried over, largest change {err:.1e} (fraction of the chord)")

    def say(self, text, error=False):
        self.status.configure(text=text, foreground=RED if error else "black")

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
                self.say("problem: " + pv["problems"][0], error=True)
            elif self._job is None:
                self.say(f"ok ({pv['seconds'] * 1e3:.0f} ms)")
        except Exception as exc:
            self.say(f"preview failed: {type(exc).__name__}: {exc}", error=True)
            traceback.print_exc()

    # ------------------------------------------------------------ XCAD
    def on_xcad(self):
        if self._job is not None:
            return
        problems = self.ad.preview()["problems"]
        if problems:
            self.say("XCAD: the design cannot be built: " + problems[0], error=True)
            return
        case = self.case.get().strip() or "gui"
        self.xcad_btn.configure(state="disabled")
        self.say(f"XCAD: writing {case} and lofting the solid ...")
        job = self._job = {"done": False, "result": None, "error": None, "t0": time.time()}
        task = self.ad.xcad_task(case)                                  # the design as it is now

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
        if not job["done"]:
            self.root.after(200, self._poll)
            return
        self._job = None
        self.xcad_btn.configure(state="normal")
        if job["error"]:
            print(job["error"])
            self.say("XCAD failed: " + job["error"].strip().splitlines()[-1], error=True)
            return
        res = job["result"]
        print("\n".join(res["report"]))
        msg = f"XCAD done in {time.time() - job['t0']:.1f} s: {os.path.basename(res['case']['paths']['xcad'])}"
        if res.get("loft") is not None:
            lr = res["loft"]
            self.step = res["step"]
            msg += (f", {os.path.basename(self.step)} (solid {'valid' if lr.valid else 'NOT valid'}, "
                    f"{lr.volume / 1e3:.2f} cm^3); OCC viewer shows it")
        self.say(msg + (f"; {res['cad_error']}" if res.get("cad_error") else ""), error=bool(res.get("cad_error")))

    def open_occ_viewer(self):
        if not self.step:
            self.say("press XCAD first: the viewer shows the solid it writes", error=True)
            return
        subprocess.Popen([sys.executable, os.path.join(HERE, "xcad_loft.py"), "--view-step", self.step], cwd=HERE)
        self.say(f"OCC viewer: {os.path.basename(self.step)} (its own window, a few seconds)")


def main():
    ap = argparse.ArgumentParser(description="XGeom design tool")
    ap.add_argument("geometry", nargs="?", default="rudder", choices=sorted(ADAPTERS))
    ap.add_argument("--space", help="a <case>_design_space.json to start from")
    args = ap.parse_args()
    space = os.path.abspath(args.space) if args.space else None
    os.chdir(HERE)                                   # the adapters' files are relative to this folder
    root = tk.Tk()
    root.geometry(f"{min(1600, root.winfo_screenwidth() - 40)}x{min(940, root.winfo_screenheight() - 90)}+10+10")
    App(root, args.geometry, space)
    root.mainloop()


if __name__ == "__main__":
    main()
