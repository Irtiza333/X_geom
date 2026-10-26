"""
_xgeom_files_test.py

Self-check for the XGeom input files (xgeom_files.py): the component file's text (every kind, and what it
refuses), each kind through its module (blade_modify, wing_modify, hull_modify: an old parameter file, its
design written and read back exactly, rewrites identical, a hand-written file), and the assembly file.

    python _xgeom_files_test.py          the files and the modules
    python _xgeom_files_test.py --gui    also the window's Load ... and Save set-up ...
"""

import dataclasses
import filecmp
import os
import sys
import tempfile

import numpy as np

import blade_modify as BM
import hull_modify as HM
import wing_modify as WM
import xgeom_files as XF

_FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<76s} max error {err:.3e}   (tol {tol:.0e})")


def refused(fn, *args, **kw):
    try:
        fn(*args, **kw)
        return 0
    except ValueError:
        return 1


def text_file(tmp, name, lines):
    path = os.path.join(tmp, name)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def same_design(a, b, names, at, grid):
    return max(float(np.abs(at(a, c, grid) - at(b, c, grid)).max()) for c in names)


def blade_at(d, c, u):
    return d.at(c, d.r0 + u * (d.r1 - d.r0))


def test_text(tmp):
    print("\nthe component file's text")
    head = ["# XGeom component", "# kind  blade", "# diameter_m  1.4", "# blades  5", "# section  naca2412",
            "# r/R  P/D  c/D  t/c  f/c  skew_deg  rake/D"]
    rows = ["0.2  1.0  0.2  0.1  0.02  0  0", "0.5  1.1  0.3  0.06  0.02  5  0", "0.8  1.0  0.3  0.04  0.01  10  0",
            "1.0  0.9  0.1  0.02  0.005  15  0"]
    good = text_file(tmp, "good.txt", head + rows)
    comp = XF.read_component(good)
    check("a hand-written blade: kind, name from the file, geometry, columns, rows",
          [comp.kind == "blade", comp.name == "good", comp.geometry["section"] == "naca2412", len(comp.columns),
           comp.table.shape[0]], [1, 1, 1, 7, 4], 0)
    bad = {
        "not a component file": ["# XGeom blade parameters"] + head[1:] + rows,
        "no kind line": head[:1] + head[2:] + rows,
        "an unknown key": head[:5] + ["# sectoin  naca2412"] + head[5:] + rows,
        "a settings key of another kind": head[:5] + ["# change_points  3"] + head[5:] + rows,
        "a wrong column line": head[:-1] + ["# r/R  P/D  c/D  t/c  f/c  rake/D  skew_deg"] + rows,
        "a row of 6 numbers": head + rows[:-1] + ["1.0  0.9  0.1  0.02  0.005  15"],
        "a header line under the table": head + rows[:2] + ["# note  late"] + rows[2:],
        "a var line of 4 fields": head[:5] + ["# var  radius  0.7  0.5  0.9"] + head[5:] + rows,
        "a var line with low above high": head[:5] + ["# var  radius  0.7  0.9  0.5  Y"] + head[5:] + rows,
        "a curve form the kind has not": head[:5] + ["# curve  pitch  cst 4"] + head[5:] + rows,
        "bezier1 with no number of points": head[:5] + ["# curve  pitch  bezier1"] + head[5:] + rows,
        "a key given twice": head[:5] + ["# blades  6"] + head[5:] + rows,
    }
    got = [refused(XF.read_component, text_file(tmp, f"bad{i}.txt", lines)) for i, lines in enumerate(bad.values())]
    check("refused: " + ", ".join(bad), got, [1] * len(bad), 0)
    comp.notes = ["a note", "another"]
    path = XF.write_component(os.path.join(tmp, "again.txt"), comp)
    back = XF.read_component(path)
    check("written and read back: notes, geometry, table", [back.notes == comp.notes, back.geometry == comp.geometry,
                                                            float(np.abs(back.table - comp.table).max())],
          [1, 1, 0], 0)
    check("first lines: a component file, its kind; an old parameter file's kind", [
        XF.is_component(path), XF.component_kind(path) == "blade", XF.legacy_kind("msc_blade_params.dat") == "blade",
        XF.legacy_kind("fin_params.dat") == "fin", XF.legacy_kind("suboff_hull_params.dat") == "hull",
        XF.file_kind("wing_params.dat") == "wing", XF.file_kind(good) == "blade"], [1] * 7, 0)


def test_blade(tmp):
    print("\nthe blade (blade_modify)")
    src = open("msc_blade_params.dat").read()
    for n, flag in (("pitch", "Y  0.6  1.4"), ("chord", "Y"), ("skew", "Y"), ("radius", "Y")):
        src = src.replace(f"# vary  {n:<11s} N", f"# vary  {n:<11s} {flag}")
    old = text_file(tmp, "msc_y.dat", src.splitlines())
    ld = BM.load_component(old, {"skew": 4}, {"skew": 1})
    check("an old file: legacy, its curves' forms from its vary lines, the flagged ones free",
          [ld.legacy, ld.design.curves["pitch"].segments, ld.design.curves["skew"].segments,
           ld.design.curves["rake"].segments, "radius" in ld.space.names, "hub_radius" in ld.space.names,
           ld.space.bounds[ld.space.names.index("pitch.v4")][1]], [1, 2, 1, 0, 1, 0, 1.4], 0)
    vals = ld.space.values(ld.design)
    vals.update({"pitch.v4": 1.05 * vals["pitch.v4"], "chord.u4": 0.55, "skew.v2": vals["skew.v2"] + 1.0,
                 "radius": 0.72, "hub_radius": 0.13})
    d = ld.space.design_of(vals)
    st = {"sections": 40, "tip_band": 0.9, "tip_sections": 10, "clearance_mm": 20.0, "hub": False}
    p1 = BM.write_component(os.path.join(tmp, "blade.txt"), ld.params, d, ld.space, st, "blade", ["changed"])
    back = BM.load_component(p1)
    u = np.linspace(0.0, 1.0, 501)
    v1, v2 = ld.space.values(d), back.space.values(back.design)
    check("its design written and read back: the curves, every variable (to the 12 figures written)",
          [same_design(d, back.design, BM.CURVES, blade_at, u), max(abs(v1[n] - v2[n]) for n in v1),
           abs(back.design.radius - 0.72), abs(back.design.hub_radius - 0.13)], 0, 1e-8)
    check("... the forms, the free variables, the bounds, the settings, the name and notes",
          [[back.design.curves[c].segments for c in BM.CURVES] == [d.curves[c].segments for c in BM.CURVES],
           back.space.names == ld.space.names, float(np.abs(back.space.all_bounds - ld.space.all_bounds).max()),
           back.settings == st, back.name == "blade", back.notes == ["changed"], back.legacy], [1, 1, 0, 1, 1, 1, 0], 0)
    p2 = BM.write_component(os.path.join(tmp, "blade2.txt"), back.params, back.design, back.space, back.settings,
                            back.name, back.notes)
    check("written again from what was read: the same file", filecmp.cmp(p1, p2, shallow=False), 1, 0)
    lines = open(p1).read().splitlines()
    i = next(k for k, s in enumerate(lines) if not s.startswith("#")) + 10
    edited = lines[:i] + [lines[i].replace(lines[i].split()[1], f"{float(lines[i].split()[1]) + 0.01:.8f}", 1)] \
        + lines[i + 1:]
    no_vars = [s for s in edited if not s.startswith("# var    pitch.")]
    partial = [s for s in lines if not s.startswith("# var    pitch.v7")]
    clash = [s.replace("# diameter_m    1.44", "# diameter_m    1.5") for s in lines]
    check("refused: a P/D column edited under its var lines, a curve's var lines in part, D against var radius",
          [refused(BM.load_component, text_file(tmp, "e1.txt", edited)),
           refused(BM.load_component, text_file(tmp, "e2.txt", partial)),
           refused(BM.load_component, text_file(tmp, "e3.txt", clash))], [1, 1, 1], 0)
    fitted = BM.load_component(text_file(tmp, "e4.txt", no_vars))
    r = fitted.params.r
    check("the edited column with the curve's var lines deleted: fitted to it (2 segments), held (no vary line)",
          [fitted.design.curves["pitch"].segments, "pitch.v1" in fitted.space.all_names,
           "pitch.v1" in fitted.space.names,
           float(np.abs(fitted.design.at("pitch", r) - fitted.params.values["pitch"]).max()) < 0.02], [2, 1, 0, 1], 0)
    hand = text_file(tmp, "hand.txt", ["# XGeom component", "# kind  blade", "# diameter_m  1.4", "# blades  5",
                                       "# section  naca2416", "# vary  chord  Y  0.1  0.6",
                                       "# r/R  P/D  c/D  t/c  f/c  skew_deg  rake/D"]
                     + [f"{x:.4f}  {1.2 - 0.4 * x:.4f}  {0.2 + 0.3 * x * (1 - x):.4f}  {0.2 - 0.18 * x:.4f}  0.02  "
                        f"{20 * x:.4f}  0" for x in np.linspace(0.2, 1.0, 9)])
    h = BM.load_component(hand)
    check("a hand-written file with a vary line: chord a free Bezier within its range, the others the table's",
          [h.design.curves["chord"].segments, h.design.curves["pitch"].segments, h.params.airfoil == "naca2416",
           all(n.startswith("chord.") for n in h.space.names), min(lo for (lo, hi), n in zip(h.space.bounds,
                                                                                             h.space.names)
                                                                     if n.startswith("chord.v"))], [2, 0, 1, 1, 0.1],
          0)
    bb2 = os.path.join("bb2", "outputs", "bb2_blade_params.dat")
    if os.path.exists(bb2):
        lb = BM.load_component(bb2)
        pb = BM.write_component(os.path.join(tmp, "bb2.txt"), lb.params, lb.design, lb.space, {}, "bb2", lb.notes,
                                lb.extra)
        lb2 = BM.load_component(pb)
        check("BB2's blade: its section_tables kept (a path that finds the file), its notes",
              [os.path.exists(XF.resolve(lb2.extra["section_tables"], tmp)), lb2.notes == lb.notes,
               len(lb.notes) >= 2], [1, 1, 1], 0)


def test_planar(tmp):
    print("\nthe rudder, the fin and the wing (wing_modify)")
    for src, kind in (("wing_params.dat", "wing"), ("fin_params.dat", "fin"), ("fin_params.dat", "rudder")):
        ld = WM.load_component(src, free=("chord", "pitch"))
        vals = ld.space.values(ld.design)
        vals.update({"chord.v1": 1.1 * vals["chord.v1"], "span": 210.0, "dt1.v1": 0.004,
                     "pitch.v2": vals["pitch.v2"] + 0.5})
        d = ld.space.design_of(vals)
        space = ld.space
        if kind == "rudder":                              # pitch and rake as the table has them
            d = WM.refit_to_file(ld.params, WM.refit_to_file(ld.params, d, "pitch", 0), "rake", 0)
            space = WM.WingSpace(ld.params, d, bounds=dict(zip(space.all_names, map(tuple, space.all_bounds))),
                                 free=("chord",))
        p1 = WM.write_component(os.path.join(tmp, f"{kind}.txt"), ld.params, d, space, {"sections": 33}, kind,
                                ld.notes, kind)
        back = WM.load_component(p1)
        p2 = WM.write_component(os.path.join(tmp, f"{kind}2.txt"), back.params, back.design, back.space,
                                back.settings, back.name, back.notes)
        e = np.linspace(0.0, 1.0, 301)
        v1, v2 = space.values(d), back.space.values(back.design)
        check(f"{kind}: written and read back: curves (mm, deg, t/c, f/c), variables (12 figures)",
              [same_design(d, back.design, d.names, lambda q, c, x: q.at(c, x), e),
               max(abs(v1[n] - v2[n]) for n in v1)], 0, 1e-9)
        forms = [back.design.curves[c].segments for c in ("pitch", "rake")]
        check(f"{kind}: kind, span, change points, sections, forms, free, the rewrite the same file",
              [back.kind == kind, back.design.span, back.design.changes, back.settings["sections"],
               forms == ([0, 0] if kind == "rudder" else [1, 1]), back.space.names == space.names,
               filecmp.cmp(p1, p2, shallow=False)], [1, 210, 3, 33, 1, 1, 1], 0)
    comp = XF.read_component(os.path.join(tmp, "rudder.txt"))
    check("the rudder's table: 0 in pitch and rake, sweep from 0 at the root",
          [np.abs(comp.column("pitch_deg")).max(), np.abs(comp.column("rake_mm")).max(), comp.column("sweep_mm")[0]],
          0, 0)
    lines = open(os.path.join(tmp, "wing.txt")).read().splitlines()
    pinned = [s for s in lines if not s.startswith("# var    chord.v3")] + []
    root = lines[:12] + ["# var    sweep.v0   0   -20   48   N"] + lines[12:]
    check("refused: a curve's var lines in part, a var line for sweep's pinned root",
          [refused(WM.load_component, text_file(tmp, "w1.txt", pinned)),
           refused(WM.load_component, text_file(tmp, "w2.txt", root))], [1, 1], 0)


def test_hull(tmp):
    print("\nthe hull (hull_modify)")
    for src in ("suboff_hull_params.dat", os.path.join("bb2", "outputs", "bb2_hull_params.dat")):
        if not os.path.exists(src):
            continue
        ld = HM.load_component(src)
        vals = ld.space.values(ld.design)
        vals.update({"nose": 1.05 * vals["nose"], "middle": 0.9 * vals["middle"], "tail_r.N2": 2.2})
        d = ld.space.design_of(vals)
        name = os.path.basename(src)[:-len("_params.dat")]
        p1 = HM.write_component(os.path.join(tmp, f"{name}.txt"), ld.params, d, ld.space, name, ld.notes)
        back = HM.load_component(p1)
        p2 = HM.write_component(os.path.join(tmp, f"{name}2.txt"), back.params, back.design, back.space, back.name,
                                back.notes)
        x = np.linspace(0.0, d.length * (1.0 - 1e-9), 2001)
        v1, v2 = ld.space.values(d), back.space.values(back.design)
        check(f"{name}: written and read back: half-heights (m), variables; the rewrite the same file",
              [float(np.abs(d.radius("r", x) - back.design.radius("r", x)).max()),
               max(abs(v1[n] - v2[n]) for n in v1), 1 - filecmp.cmp(p1, p2, shallow=False),
               1 - (back.space.names == ld.space.names)], 0, 1e-9)
        check(f"{name}: the stations on the new lengths (from 0 to the end, rising; m), the notes kept",
              [back.params.x[0], back.params.x[-1] - (d.length - (d.cap if back.params.x[-1] < d.length else 0)),
               float(np.diff(back.params.x).min() > 0), back.notes == ld.notes], [0, 0, 1, 1], 1e-6 * d.length)
    ld = HM.load_component("suboff_hull_params.dat")
    el = ld.space.values(ld.design)
    design = HM.HullDesign(ld.design.nose, ld.design.middle, ld.design.tail, ld.design.cap, ld.design.r,
                           ld.design.re, {**{k: c.copy() for k, c in ld.design.curves.items()},
                                          "nose_rp": ld.design.curves["nose_r"].copy(),
                                          "tail_rp": ld.design.curves["tail_r"].copy()}, 0.8 * ld.design.r,
                           0.8 * ld.design.re)
    space = HM.HullSpace(design)
    p = HM.write_component(os.path.join(tmp, "elliptic.txt"), ld.params, design, space, "elliptic")
    back = HM.load_component(p)
    x = np.linspace(0.0, design.length * (1.0 - 1e-9), 801)
    check("elliptical sections: written (rp_m column, axisymmetric N) and read back",
          [back.design.axisymmetric, float(np.abs(back.design.radius("rp", x) - design.radius("rp", x)).max()) < 1e-9,
           XF.read_component(p).has("rp_m"), len(el) < len(space.all_names)], [0, 1, 1, 1], 0)
    lines = open(p).read().splitlines()
    i = next(k for k, s in enumerate(lines) if not s.startswith("#")) + 40
    lines[i] = lines[i].replace(lines[i].split()[1], f"{float(lines[i].split()[1]) + 0.001:.9f}", 1)
    check("refused: a station's r edited under the curves' var lines",
          refused(HM.load_component, text_file(tmp, "h1.txt", lines)), 1, 0)


def test_assembly(tmp):
    print("\nthe assembly file")
    asm = XF.Assembly("suboff", ["the example vehicle"], {"hull": "suboff_hull.txt", "blade": "msc_blade.txt"},
                      {"prop_dx": "0", "rudders": "4", "rudder_rake": "none"}, {"samples": "30", "seed": "1"})
    path = XF.write_assembly(os.path.join(tmp, "asm.txt"), asm)
    back = XF.read_assembly(path)
    check("written and read back: name, notes, components, placement, run; the component paths resolved",
          [back.name == "suboff", back.notes == asm.notes, back.components == asm.components,
           back.placement == asm.placement, back.run == asm.run,
           back.component_path("hull") == os.path.join(tmp, "suboff_hull.txt"), back.component_path("fin") is None,
           XF.is_assembly(path)], [1] * 8, 0)
    bad = [text_file(tmp, "a1.txt", ["# XGeom assembly", "# hulls  x.txt"]),
           text_file(tmp, "a2.txt", ["# XGeom assembly", "# hull  x.txt", "1 2 3"]),
           text_file(tmp, "a3.txt", ["# XGeom component", "# hull  x.txt"])]
    check("refused: an unknown key, a table, not an assembly file", [refused(XF.read_assembly, p) for p in bad],
          [1, 1, 1], 0)
    import xgeom_vehicle as XV
    check("the placement keys are xgeom_vehicle.Placement's fields",
          tuple(f.name for f in dataclasses.fields(XV.Placement)) == XF.PLACEMENT_KEYS, 1, 0)


def test_examples():
    print("\nthe example files of this folder")
    for path, kind in (("msc_blade.txt", "blade"), ("wing.txt", "wing"), ("fin.txt", "fin"),
                       ("suboff_hull.txt", "hull")):
        if not os.path.exists(path):
            check(f"{path} is there", 0, 1, 0)
            continue
        ld = XF.load(path)
        check(f"{path}: a {kind} component, read through its module", [ld.kind == kind, not ld.legacy], [1, 1], 0)


def test_gui(tmp):
    print("\nthe window: Save set-up ... and Load ... with component files (xgeom_tool.py)")
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:                                   # no tkinter or no display
        print(f"       skipped: {exc}")
        return
    import time
    import xgeom_tool as XT

    def settle(n=5):
        for _ in range(n):
            root.update()
            time.sleep(0.03)
    save, open_ = XT.filedialog.asksaveasfilename, XT.filedialog.askopenfilename
    try:
        app = XT.App(root, "vehicle")
        settle()
        cases = (("blade", "radius", 0.69, "pitch.v1", "msc_blade.txt"),
                 ("fin", "chord.v3", 210.0, "chord.v0", "fin.txt"),
                 ("wing", "pitch.v1", 1.5, "chord.v1", "wing.txt"),
                 ("hull", "nose", 1.1, "middle", "suboff_hull.txt"))
        for kind, name, value, held, default in cases:
            app.switch(kind)
            settle()
            ad = app.ad
            check(f"{kind}: starts from {default}; the File box names it; Save set-up opens a dialog",
                  [os.path.basename(ad.file) == default, app.save_target()[1] == default], [1, 1], 0)
            if kind == "blade":                                # pitch from the table's form to 2 segments
                app.curve.set("pitch")
                app.on_curve()
                app.nseg.set(2)
                app.on_segments()
                settle()
            ad.set_free(name, True)
            app.set_value(name, value, None)
            ad.set_free(held, not ad.is_free(held))
            flag = ad.is_free(held)
            if kind == "blade":
                app.sec["sections"].set(40)
                app.on_sections()
            elif kind != "hull":
                app.wsec["sections"].set(29)
                app.on_wing_sections()
            settle()
            target = os.path.join(tmp, f"gui_{kind}.txt")
            seen = {}
            XT.filedialog.asksaveasfilename = lambda **kw: (seen.update(kw), target)[1]
            app.on_save()
            settle()
            values, meta = dict(ad.values), {k: dict(m) for k, m in ad.meta.items()}
            check(f"{kind}: Save set-up wrote a {kind} component file where the dialog said; it is the file in use",
                  [XF.component_kind(target) == kind, seen.get("initialfile") == default, ad.file == target,
                   app.save_target()[1] == os.path.basename(target)], [1, 1, 1, 1], 0)
            for path in (os.path.join(HERE, default), target):
                XT.filedialog.askopenfilename = lambda **kw: path
                app.on_load_params()
                settle()
            ad = app.ad
            same = [abs(ad.values[k] - values[k]) <= 1e-9 * max(1.0, abs(values[k])) for k in values]
            check(f"{kind}: Load ... the default file, then the saved one: every value, bound and free flag back",
                  [all(same), ad.meta == meta, ad.is_free(held) == flag, abs(ad.get(name) - value) < 1e-9],
                  [1, 1, 1, 1], 0)
            if kind == "blade":
                check("blade: pitch 2 segments and the sections setting back from the file, in its boxes",
                      [ad.segments("pitch"), ad.settings["sections"], int(app.sec["sections"].get())], [2, 40, 40], 0)
            elif kind != "hull":
                check(f"{kind}: the sections setting back from the file, in its box",
                      [ad.settings["sections"], int(app.wsec["sections"].get())], [29, 29], 0)
    finally:
        XT.filedialog.asksaveasfilename, XT.filedialog.askopenfilename = save, open_
        root.destroy()


def main():
    print("XGeom input files: self-check")
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            test_text(tmp)
            test_blade(tmp)
            test_planar(tmp)
            test_hull(tmp)
            test_assembly(tmp)
            test_examples()
            if "--gui" in sys.argv:
                test_gui(tmp)
    finally:
        os.chdir(cwd)
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
