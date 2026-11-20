"""
_xgeom_web_contract_test.py

What the XGeom web app (the browser version on the ACIC site) relies on in this repository. The web app runs
these modules unchanged, in the browser (Pyodide), through one bridge file. This self-check calls exactly what
that bridge calls, so a change here that would break the web app shows up before it is pushed, without
knowing anything about the web app.

A FAIL is not necessarily wrong: it means "the web app's bridge needs a matching change". Rename freely
anything not checked here. Adding methods, keys, settings, preview entries or assembly fields never fails.

    python _xgeom_web_contract_test.py

Sections:
  1. imports        each component's modules import without Tk or OCC (the browser has neither)
  2. adapters       the adapter methods and their return shapes (the same ones xgeom_tool.py uses)
  3. preview        the preview() entries the web app draws, per component
  4. internals      design-object attributes the web app's 2D views still read directly (see below)
  5. files          the component and assembly files: readers, writers, constants, byte-identical rewrites
  6. vehicle        xgeom_vehicle: parts, placement, assemble and what its info holds
  7. messages       the start of the error messages the web app turns into plain language

Section 4 is the fragile one: those are design internals. The plan is for each adapter's preview() to return
what the 2D views draw (planform, sections, profile, control points) so the web app stops reading them;
each check that moves into preview() is then deleted here.
"""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_FAILED = []
_NOTES = []

# the components as the web app opens them: (module, adapter class, the sample it opens at start)
COMPONENTS = {
    "hull": ("xgeom_hull", "HullAdapter", "suboff_hull.txt"),
    "blade": ("xgeom_blade", "BladeAdapter", "msc_blade.txt"),
    "rudder": ("xgeom_rudder", "RudderAdapter", None),          # rudder-h: its own model, outputs/ beside it
    "fin": ("xgeom_wing", "FinAdapter", "fin.txt"),
    "wing": ("xgeom_wing", "WingAdapter", "wing.txt"),
}
# the geometry modules the web app checks for, per component (a missing one is reported by name)
REQUIRES = {"hull": ("hull_modify",), "blade": ("blade_modify", "blade_control", "x_blade_new", "prop_hub", "blade_sections"),
            "rudder": ("rudder_modify", "bezier_section", "modify_control"),
            "fin": ("wing_modify", "tip_cap"), "wing": ("wing_modify", "tip_cap")}

# the form changes the web app offers, where the adapter has them (it checks with hasattr)
FORM_CALLS = {
    "hull": ("set_elliptic", "set_order"),
    "blade": ("set_blades", "set_order", "set_sections", "set_segments"),
    "rudder": ("set_degree", "set_order", "set_span", "set_tangent"),
    "fin": ("set_changes", "set_order", "set_sections", "set_segments"),
    "wing": ("set_changes", "set_order", "set_sections", "set_segments"),
}

# the preview() entries the web app reads, per component. REQUIRED: read directly, so a missing one breaks the
# component's views (FAIL). DRAWN: read if there (problems, warnings, the 3D lines, the original rudder), so a
# missing one is only not shown (a note). skeleton comes with preview(skeleton=True), the CAD sections.
REQUIRED = {
    "hull": ("params", "r", "rp", "x"),
    "blade": ("curves", "file", "file_rr", "rows", "rr"),
    "rudder": ("curves", "yy"),
    "fin": ("curves", "ee", "file", "fractions", "le", "rows", "sections", "shapes", "te", "x", "y", "z"),
    "wing": ("curves", "ee", "file", "fractions", "le", "rows", "sections", "shapes", "te", "x", "y", "z"),
}
DRAWN = {
    "hull": ("problems", "seconds"),
    "blade": ("problems", "warnings", "seconds", "sections", "le", "te", "mid", "tip", "orig_sections"),
    "rudder": ("problems", "seconds", "sections", "tracks"),
    "fin": ("problems", "warnings", "seconds", "orig"),
    "wing": ("problems", "warnings", "seconds", "orig"),
}


def check(name, ok, detail=""):
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<70s} {detail}")


def note(text):
    _NOTES.append(text)
    print(f"  note  {text}")


def attempt(name, fn):
    """Run fn; a FAIL (with the error) instead of a crash, so every check still runs."""
    try:
        return fn()
    except Exception as exc:                                     # noqa: BLE001 - reported as a FAIL
        check(name, False, f"{type(exc).__name__}: {exc}")
        return None


# ----------------------------------------------------------------------------- 1. imports

_PROBE = r"""
import importlib.abc, json, os, sys
BLOCKED = ("tkinter", "_tkinter", "OCC")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is not in the browser")
        return None
sys.meta_path.insert(0, Block())
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, sys.argv[1]); os.chdir(sys.argv[1])
module, cls, params = sys.argv[2], sys.argv[3], (sys.argv[4] or None)
out = {"error": None}
try:
    mod = importlib.import_module(module)
    ad = getattr(mod, cls)(params=params) if params else getattr(mod, cls)()
    n = list(ad.meta)[0]
    ad.set(n, float(ad.get(n)))
    ad.preview()
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
out["matplotlib"] = [m for m in ("matplotlib", "matplotlib.pyplot") if m in sys.modules]
print(json.dumps(out))
"""


def test_imports():
    print("1. imports: open, set and preview each component with Tk and OCC unavailable")
    for kind, (module, cls, params) in COMPONENTS.items():
        r = subprocess.run([sys.executable, "-c", _PROBE, HERE, module, cls, params or ""],
                           capture_output=True, text=True, timeout=600)
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            out = {"error": (r.stderr.strip().splitlines() or ["no output"])[-1], "matplotlib": []}
        check(f"{kind}: imports, opens and previews without Tk or OCC", out["error"] is None, out["error"] or "")
        if "matplotlib.pyplot" in out.get("matplotlib", []):
            note(f"{kind}: matplotlib.pyplot is imported on the way (9 MB more in the browser; nothing is "
                 f"drawn with it). Importing pyplot inside the functions that plot would avoid it.")
    for kind, mods in REQUIRES.items():
        for m in mods:
            check(f"{kind}: module {m}.py is here (the web app checks for it by name)",
                  os.path.isfile(os.path.join(HERE, m + ".py")))


# ----------------------------------------------------------------------------- 2. adapters

def _open(kind):
    import importlib
    module, cls, params = COMPONENTS[kind]
    Adapter = getattr(importlib.import_module(module), cls)
    return Adapter(params=os.path.join(HERE, params)) if params else Adapter()


def _is_row(r):
    return all(hasattr(r, a) for a in ("kind", "name", "label", "digits"))


def test_adapters(tmp):
    print("2. adapters: the methods the web app calls, and what they return")
    adapters = {}
    for kind in COMPONENTS:
        ad = attempt(f"{kind}: opens from its sample", lambda: _open(kind))
        if ad is None:
            continue
        adapters[kind] = ad
        try:
            _adapter_checks(kind, ad, tmp)
        except Exception as exc:                                 # noqa: BLE001 - reported, the rest still runs
            check(f"{kind}: the adapter checks ran through", False, f"{type(exc).__name__}: {exc}")
    return adapters


def _adapter_checks(kind, ad, tmp):
    if True:
        meta = getattr(ad, "meta", None)
        check(f"{kind}: meta is variable name -> {{'free': ...}}", isinstance(meta, dict) and bool(meta)
              and all("free" in m for m in meta.values()))
        names = list(meta)
        curves = [c for c in ad.names]
        check(f"{kind}: names is the list of curve names", all(isinstance(c, str) for c in curves))
        n = names[0]
        v = attempt(f"{kind}: get(name) is a number", lambda: float(ad.get(n)))
        lo_hi = attempt(f"{kind}: bounds(name) is (low, high)", lambda: tuple(map(float, ad.bounds(n))))
        check(f"{kind}: bounds(name) is (low, high)", lo_hi is not None and len(lo_hi) == 2 and lo_hi[0] <= lo_hi[1])
        check(f"{kind}: is_free(name) is a bool", isinstance(ad.is_free(n), (bool, np.bool_)))
        info = attempt(f"{kind}: row_info(name)", lambda: ad.row_info(n))
        check(f"{kind}: row_info(name) is text or None", info is None or isinstance(info, str))
        used = attempt(f"{kind}: set(name, value) returns the value used", lambda: ad.set(n, v))
        check(f"{kind}: set(name, value) returns the value used", used is not None and abs(float(used) - v) < 1e-9)
        attempt(f"{kind}: set_free(name, flag)", lambda: ad.set_free(n, bool(ad.is_free(n))))
        attempt(f"{kind}: set_bounds(name, low, high)", lambda: ad.set_bounds(n, *lo_hi))
        nf = attempt(f"{kind}: n_free() is (free, total)", lambda: tuple(ad.n_free()))
        check(f"{kind}: n_free() is (free, total)", nf is not None and len(nf) == 2 and all(isinstance(int(x), int) for x in nf))
        if hasattr(ad, "global_rows"):                       # optional: without it the web app lists meta's plain names
            rows = attempt(f"{kind}: global_rows()", lambda: list(ad.global_rows()))
            check(f"{kind}: global_rows() rows have kind, name, label, digits", rows is not None and all(_is_row(r) for r in rows))
        for c in curves[:2]:
            cr = attempt(f"{kind}: curve_rows({c!r})", lambda: list(ad.curve_rows(c)))
            check(f"{kind}: curve_rows({c!r}) rows have kind, name, label, digits", cr is not None and all(_is_row(r) for r in cr))
            attempt(f"{kind}: n_free({c!r})", lambda: tuple(ad.n_free(c)))
        title = ad.title() if callable(getattr(ad, "title", None)) else getattr(ad, "title", None)
        check(f"{kind}: title (text, or a method giving it)", isinstance(title, str) and bool(title))
        check(f"{kind}: summary() is text", isinstance(attempt(f"{kind}: summary()", ad.summary), str))
        if curves:
            check(f"{kind}: describe(curve) is text or None",
                  attempt(f"{kind}: describe()", lambda: ad.describe(curves[0])) is None or isinstance(ad.describe(curves[0]), str))
        for m in FORM_CALLS[kind]:
            check(f"{kind}: form change {m}() is still there", callable(getattr(ad, m, None)))
        if kind == "rudder":
            rec = attempt("rudder: space_record() is JSON", lambda: json.dumps(ad.space_record(), default=str))
            check("rudder: space_record() is JSON", rec is not None)
            path = os.path.join(tmp, "rudder_setup.json")
            with open(path, "w") as fh:
                fh.write(rec or "{}")
            attempt("rudder: load_space(path) reads it back", lambda: ad.load_space(path))
        else:
            path = os.path.join(tmp, f"web_{kind}.txt")
            got = attempt(f"{kind}: save_file(path)", lambda: ad.save_file(path))
            check(f"{kind}: save_file(path) writes the file (returns its path or None)",
                  os.path.isfile(path) and (got is None or os.path.isfile(got)))
            with open(path, "rb") as fh:
                data = fh.read()
            check(f"{kind}: the saved file starts '# XGeom component', LF only",
                  data.startswith(b"# XGeom component\n") and b"\r" not in data)
            again = type(ad)(params=path)
            path2 = os.path.join(tmp, f"web_{kind}_again.txt")
            again.save_file(path2)
            with open(path2, "rb") as fh:
                data2 = fh.read()
            same = data2.replace(f"web_{kind}_again".encode(), f"web_{kind}".encode()) == data
            check(f"{kind}: open the saved file and save again: the same bytes", same)
        if kind in ("hull",):
            for part in ("nose", "tail"):
                pr = attempt(f"hull: part_range({part!r}) is (x0, x1)", lambda: tuple(map(float, ad.part_range(part))))
                check(f"hull: part_range({part!r}) is (x0, x1)", pr is not None and pr[0] < pr[1])
            check("hull: ctrl_kinds(curve) is a sequence", len(list(ad.ctrl_kinds(curves[0]))) > 0)
        if kind in ("blade", "fin", "wing"):
            cf = attempt(f"{kind}: ctrl_free(curve)", lambda: list(ad.ctrl_free(curves[0])))
            check(f"{kind}: ctrl_free(curve) is a sequence of flags", cf is not None)
        if hasattr(ad, "load_params"):
            check(f"{kind}: load_params(path) still there (old .dat parameter files)", callable(ad.load_params))


# ----------------------------------------------------------------------------- 3. preview

def test_preview(adapters):
    print("3. preview(): the entries the web app draws")
    for kind, ad in adapters.items():
        pv = attempt(f"{kind}: preview()", ad.preview)
        if pv is None:
            continue
        for key in REQUIRED[kind]:
            check(f"{kind}: preview()[{key!r}]", key in pv)
        for key in DRAWN[kind]:
            if key not in pv:
                note(f"{kind}: preview() has no {key!r} any more: the web app no longer shows it")
        if isinstance(pv.get("sections"), str):          # preview() caught a failure and kept its message there
            check(f"{kind}: preview() builds its sections and 3D lines", False, pv["sections"])
        sk = attempt(f"{kind}: preview(skeleton=True)", lambda: ad.preview(skeleton=True))
        check(f"{kind}: preview(skeleton=True) has 'skeleton' (the CAD sections)", sk is not None and "skeleton" in sk)
        check(f"{kind}: problems and warnings (where there) are lists of text",
              all(isinstance(q, str) for q in list(pv.get("problems", [])) + list(pv.get("warnings", []))))


# ----------------------------------------------------------------------------- 4. internals

def test_internals(adapters):
    print("4. internals the 2D views still read (to move into preview(); then delete the check)")
    d = {k: getattr(ad, "design", None) for k, ad in adapters.items()}
    if "hull" in d:
        h = d["hull"]
        for a in ("length", "joins", "axisymmetric", "r", "rp", "curves"):
            check(f"hull: design.{a}", hasattr(h, a))
        for m in ("ctrl", "radius", "radii"):
            check(f"hull: design.{m}()", callable(getattr(h, m, None)))
        import hull_modify as HM
        check("hull: hull_modify.section_points(design, xs, n)", callable(getattr(HM, "section_points", None)))
        p = adapters["hull"].preview()["params"]
        for a in ("x", "r", "rp"):
            check(f"hull: preview()['params'].{a}", hasattr(p, a))
        check("hull: preview()['params'].values(curve)", callable(getattr(p, "values", None)))
    for k in ("fin", "wing"):
        if k in d:
            w = d[k]
            check(f"{k}: design.span", hasattr(w, "span"))
            check(f"{k}: design.curves", hasattr(w, "curves"))
            check(f"{k}: design.ctrl(curve)", callable(getattr(w, "ctrl", None)))
            check(f"{k}: design.changes", hasattr(w, "changes"))
            check(f"{k}: adapter.params.span", hasattr(adapters[k].params, "span"))
    import wing_modify as WM
    check("wing_modify.unit(curve) and PINNED", callable(getattr(WM, "unit", None)) and hasattr(WM, "PINNED"))
    if "blade" in d:
        b = d["blade"]
        check("blade: design.curves and design.ctrl(curve)", hasattr(b, "curves") and callable(getattr(b, "ctrl", None)))
        import blade_modify as BM
        check("blade_modify.UNITS", hasattr(BM, "UNITS"))
    if "rudder" in d:
        r = d["rudder"]
        check("rudder: design.H and design.curves[c].order, .v, .s", hasattr(r, "H") and hasattr(r, "curves"))
        check("rudder: adapter.stack", hasattr(adapters["rudder"], "stack"))
        import rudder_modify as RM
        check("rudder_modify.original_section_at(stack, y)", callable(getattr(RM, "original_section_at", None)))
    for k, ad in adapters.items():
        if k in ("blade", "fin", "wing"):
            check(f"{k}: adapter.settings['sections']", "sections" in getattr(ad, "settings", {}))


# ----------------------------------------------------------------------------- 5. files

def test_files(tmp):
    print("5. files: the component and assembly files")
    import xgeom_files as XF
    for fn in ("read_component", "is_component", "component_kind", "file_kind", "legacy_kind", "is_assembly",
               "read_assembly", "write_assembly", "is_section_table"):
        check(f"xgeom_files.{fn}()", callable(getattr(XF, fn, None)))
    check("xgeom_files.COMPONENT / ASSEMBLY first lines",
          XF.COMPONENT == "# XGeom component" and XF.ASSEMBLY == "# XGeom assembly")
    check("ASSEMBLY_COMPONENTS", tuple(XF.ASSEMBLY_COMPONENTS) == ("hull", "blade", "rudder", "fin", "wing"))
    check("RUN_KEYS (samples, batch, seed, cad_timeout_s)",
          set(("samples", "batch", "seed", "cad_timeout_s")) <= set(XF.RUN_KEYS))
    fields = {f.name for f in dataclasses.fields(XF.Assembly)}
    check("Assembly(name, notes, components, placement, run)",
          {"name", "notes", "components", "placement", "run"} <= fields)
    import xgeom_vehicle as XV
    pl = {f.name for f in dataclasses.fields(XV.Placement)}
    check("every PLACEMENT_KEYS key is a Placement field (the web app maps them one for one)",
          set(XF.PLACEMENT_KEYS) <= pl, ", ".join(sorted(set(XF.PLACEMENT_KEYS) - pl)))
    for kind, (_, _, sample) in COMPONENTS.items():
        if sample:
            path = os.path.join(HERE, sample)
            check(f"{sample}: file_kind is {kind}", XF.file_kind(path) == kind)
            comp = XF.read_component(path)
            check(f"{sample}: read_component gives geometry and settings dicts",
                  isinstance(comp.geometry, dict) and isinstance(comp.settings, dict))
    blade = XF.read_component(os.path.join(HERE, "msc_blade.txt"))
    check("blade: the section key is 'section' (file beside it, else an airfoil name)", "section" in blade.geometry)
    asm = XF.Assembly(name="contract", notes=["a note"], components={"hull": "suboff_hull.txt", "fin": "fin.txt"},
                      placement={"fins": "2", "view": "stern", "rudder_rake": "none"}, run={"samples": "30"})
    p1 = XF.write_assembly(os.path.join(tmp, "a.txt"), asm) or os.path.join(tmp, "a.txt")
    back = XF.read_assembly(p1)
    check("assembly: write, read: the same components, placement and run",
          back.components == asm.components and back.placement == asm.placement and back.run == asm.run)
    p2 = XF.write_assembly(os.path.join(tmp, "b.txt"), back) or os.path.join(tmp, "b.txt")
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        b1, b2 = f1.read(), f2.read()
    check("assembly: a rewrite gives the same bytes, LF only", b1 == b2 and b"\r" not in b1)
    for sample in ("fin.txt", "wing.txt", "suboff_hull.txt", "msc_blade.txt"):
        kind = XF.file_kind(os.path.join(HERE, sample))
        module, cls, _ = COMPONENTS[kind]
        import importlib
        # in a folder of its own with what it names, so the section's path is written as in the sample
        here = os.path.join(tmp, "resave_" + kind)
        os.makedirs(here, exist_ok=True)
        for f in (sample, "airfoil_data_fixed.csv"):
            shutil.copy(os.path.join(HERE, f), here)
        ad = getattr(importlib.import_module(module), cls)(params=os.path.join(here, sample))
        out = os.path.join(here, sample)                 # over its copy: the name line follows the file name
        ad.save_file(out)
        with open(os.path.join(HERE, sample), "rb") as f1, open(out, "rb") as f2:
            shipped, written = f1.read(), f2.read()
        if b"\r\n" in shipped:
            note(f"{sample}: this checkout has CRLF line endings (Git's core.autocrlf, usual on Windows); the file "
                 f"in the repository is LF. A .gitattributes line '*.txt text eol=lf' keeps checkouts LF everywhere")
            shipped = shipped.replace(b"\r\n", b"\n")
        same = shipped == written
        if not same:
            note(f"{sample}: opened and saved unchanged, it is not the same bytes (re-save the sample with "
                 f"Save set-up so the shipped file is what the tool writes)")


# ----------------------------------------------------------------------------- 6. vehicle

def test_vehicle(adapters):
    print("6. vehicle: xgeom_vehicle")
    import xgeom_vehicle as XV
    for name in ("PARTS", "OWNS", "COLLECTIONS", "MAX_RUDDERS", "MAX_RAKE"):
        check(f"xgeom_vehicle.{name}", hasattr(XV, name))
    for fn in ("default_placement", "placement_problem", "assemble", "active"):
        check(f"xgeom_vehicle.{fn}()", callable(getattr(XV, fn, None)))
    check("PARTS has a maker for each of the five", set(COMPONENTS) <= set(XV.PARTS))
    check("COLLECTIONS rows are (key, colour, width)", all(len(c) == 3 for c in XV.COLLECTIONS))
    check("OWNS: component -> its collections", set(XV.OWNS) <= set(COMPONENTS)
          and {g for gs in XV.OWNS.values() for g in gs} <= {c[0] for c in XV.COLLECTIONS})
    parts = {}
    for k, ad in adapters.items():
        part = attempt(f"PARTS[{k!r}](adapter) makes the {k}'s part", lambda: XV.PARTS[k](ad))
        if part is not None:
            parts[k] = part
            for q in part.get("problems", ()):
                note(f"the {k}'s part on the vehicle has a problem: {q}")
    pl = attempt("default_placement(parts)", lambda: XV.default_placement(parts))
    if pl is None:
        return
    fields = {f.name for f in dataclasses.fields(pl)}
    want = {"prop_x", "prop_d", "rudder_x", "rudder_angle", "rudders", "rudder_scale", "rudder_rake", "fin_x",
            "fins", "fin_angle", "fin_scale", "wing_x", "wings", "wing_angle", "wing_scale", "view"}
    check("Placement fields", want <= fields, ", ".join(sorted(want - fields)))
    L = parts["hull"]["design"].length
    check("parts['hull']['design'].length", L > 0)
    check("placement_problem(ok values, L) is None", XV.placement_problem(dataclasses.asdict(pl), L) is None)
    bad = XV.placement_problem({**dataclasses.asdict(pl), "rudders": 40}, L)
    check("placement_problem(a bad value, L) is the reason, as text", isinstance(bad, str) and bool(bad))
    segs, info = XV.assemble(parts, pl)
    check("assemble() gives one entry per collection", {c[0] for c in XV.COLLECTIONS} <= set(segs))
    for key in ("prop_x", "blades", "prop_d", "design_d", "prop_scale", "rudder_x_te", "rudder_x_le",
                "rudder_pitch", "rudder_rake", "rudder_depth", "gap", "fin_frames", "fin_pitch", "fin_x_le",
                "fin_x_te", "wing_frames", "wing_x_le", "wing_x_te", "wing_pitch"):
        why = ""
        if key not in info:
            if key == "gap":
                why = "none: the blades and the rudders must both be placed (see the notes above)"
            elif key.startswith("prop") or key == "blades":
                why = "the propeller was not placed"
        check(f"assemble() info[{key!r}] (the web app's summary)", key in info, why)
    check("active(parts, placement) lists components", set(XV.active(parts, pl)) <= set(COMPONENTS))
    for k in ("problems", "warnings"):
        check(f"each part's {k!r} (may be absent)", all(isinstance(list(p.get(k, ())), list) for p in parts.values()))


# ----------------------------------------------------------------------------- 7. messages

def test_messages(tmp):
    print("7. messages: the start of the errors the web app explains")
    import xgeom_files as XF
    import airfoils as AF
    try:
        AF.coordinates("naca00l5x")
        msg = ""
    except ValueError as exc:
        msg = str(exc)
    check("an unknown section: \"airfoil '<name>': not in ...\"", msg.startswith("airfoil 'naca00l5x': not in"), msg[:60])
    if AF._files(AF.AIRFOIL_DIR) if hasattr(AF, "_files") else False:
        try:
            AF.coordinates("e3877")
        except ValueError as exc:
            check("close airfoil names: '; close: a, b, c'", "; close: " in str(exc))
    with open(os.path.join(HERE, "fin.txt")) as fh:
        fin = fh.read()
    p = os.path.join(tmp, "no_table.txt")
    with open(p, "w") as fh:
        fh.write("# XGeom component\n# kind          fin\n")
    try:
        XF.read_component(p)
        msg = ""
    except ValueError as exc:
        msg = str(exc)
    check("no table: '<path>: no table'", msg.endswith(": no table"), msg[-40:])
    p = os.path.join(tmp, "bad_key.txt")
    with open(p, "w") as fh:
        fh.write(fin.replace("# span_mm ", "# colour        red\n# span_mm ", 1))
    try:
        XF.read_component(p)
        msg = ""
    except ValueError as exc:
        msg = str(exc)
    check("a bad line: '<path>, line <n>: ...'", ", line 6: 'colour'" in msg, msg[len(p):len(p) + 30])


# ----------------------------------------------------------------------------- run

def main():
    print("XGeom web app contract self-check")
    import platform
    vers = []
    for m in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            vers.append(f"{m} {__import__(m).__version__}")
        except Exception:                                        # noqa: BLE001 - said as not installed
            vers.append(f"{m} not installed")
    print(f"  Python {platform.python_version()} on {platform.system()}; {', '.join(vers)}"
          f"{'; MPLBACKEND=' + os.environ['MPLBACKEND'] if 'MPLBACKEND' in os.environ else ''}")
    cwd = os.getcwd()
    os.chdir(HERE)                                    # the rudder reads outputs/ relative to it, as in the tool
    sys.path.insert(0, HERE)
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            test_imports()
            adapters = test_adapters(tmp)
            test_preview(adapters)
            test_internals(adapters)
            test_files(tmp)
            test_vehicle(adapters)
            test_messages(tmp)
    finally:
        os.chdir(cwd)
    print()
    if _NOTES:
        print(f"{len(_NOTES)} note(s): not failures, but worth a look")
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        print("A FAIL means the web app's bridge needs a matching change: say so in progress.md, or keep the old "
              "name working beside the new one.")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
