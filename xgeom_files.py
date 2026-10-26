"""
xgeom_files.py

The XGeom input files (claude/xgeom-file-formats.md): the component file, one layout for every component
(propeller blade, rudder, fin, wing, hull), and the assembly file, which places the components on the hull and
defines an optimization run. This module reads and writes their text; blade_modify, wing_modify and hull_modify
turn a component into their parameters, design and design space and back (load_component, write_component).
Standard library and numpy only.

Component file ('#' lines are the header, the rest is one table, which np.loadtxt reads):

    # XGeom component
    # kind          blade                        blade, rudder, fin, wing or hull
    # name          msc_blade
    # note          free text (any number of note lines)
    # ---- geometry
    # diameter_m    1.4
    # ...
    # ---- settings
    # sections      56
    # ...
    # ---- curves and variables
    # curve  pitch      bezier2                   table | bezier1 <points> | bezier2; the hull's: cst <order>
    # var    pitch.v1   1.24135133  0.6  1.4  Y   value, low, high, free (Y) or held (N)
    # ...
    # ---- distributions
    # r/R  P/D  c/D  t/c  f/c  skew_deg  rake/D    the last '#' line above the table names its columns
      0.18000000  1.24135133  ...

    kind    what the file describes; it decides the keys (GEOMETRY, SETTINGS) and the columns (COLUMNS)
    name    a short name (default: the file's name)
    note    free text, kept and written back
    curve   a distribution's form: table (the column as it is, the cubic spline through the rows; no
            variables), bezier1 <points> (one Bezier from the root to the tip), bezier2 (two rational cubic
            segments, x_blade's form), cst <order> (a hull curve)
    var     one variable: its value, its bounds and whether it is free (a design variable) or held
    vary    '<name> Y|N [low high]' (blade_vary.py), for hand-written files: the curves and globals with no var
            lines are free (Y) or held (N), their values within low .. high when given
    ----    a separator line, ignored
An unknown key is an error (free text goes on note lines). Geometry and settings values are kept as text here;
the modules read them.

Assembly file:

    # XGeom assembly
    # name           bb2
    # note           ...
    # hull           bb2_hull.txt                the component files, relative to the assembly file
    # blade          bb2_blade.txt
    # rudder         bb2_sternplane.txt
    # ---- placement                              the Vehicle tab's settings (xgeom_vehicle.Placement)
    # prop_dx        0
    # ...
    # ---- optimization run
    # samples        30                           round-0 designs
    # batch          10                           designs per infill round
    # seed           20261010
    # cad_timeout_s  2000

    import xgeom_files as XF
    comp = XF.read_component("msc_blade.txt")              # the text: kind, geometry, curves, vars, table
    XF.write_component("copy.txt", comp)
    ld = XF.load("msc_blade.txt")                          # through blade_modify: params, design, space, ...

An old parameter file (blade_modify, wing_modify, hull_modify .dat) still loads (load); convert writes it as a
component file:

    python xgeom_files.py convert msc_blade_params.dat msc_blade.txt
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

COMPONENT = "# XGeom component"
ASSEMBLY = "# XGeom assembly"
PLANAR = ("rudder", "fin", "wing")
KINDS = ("blade",) + PLANAR + ("hull",)
_PLANAR_GEOMETRY = ("span_mm", "section", "ref_xc", "camber_xc")
GEOMETRY = {"blade": ("diameter_m", "blades", "root_r", "hub_height_m", "section", "section_tables"),
            **{k: _PLANAR_GEOMETRY for k in PLANAR},
            "hull": ("nose_m", "middle_m", "tail_m", "cap_m", "axisymmetric")}
SETTINGS = {"blade": ("sections", "tip_band", "tip_sections", "clearance_mm", "hub"),
            **{k: ("sections", "change_points") for k in PLANAR},
            "hull": ()}
ALIASES = {"blade": {"section_table": "section", "airfoil": "section"},          # old files' keys, read as these
           **{k: {"airfoil": "section"} for k in PLANAR}, "hull": {}}
COLUMNS = {"blade": ("r/R", "P/D", "c/D", "t/c", "f/c", "skew_deg", "rake/D"),
           **{k: ("eta", "pitch_deg", "chord_mm", "t/c", "f/c", "sweep_mm", "rake_mm") for k in PLANAR},
           "hull": ("x_m", "r_m")}                 # then a planar surface's dt1 .. dtk dc1 .. dck, the hull's rp_m
CURVE_COLUMNS = {"blade": {"pitch": "P/D", "chord": "c/D", "thickness": "t/c", "camber": "f/c", "skew": "skew_deg",
                           "rake": "rake/D"},
                 **{k: {"pitch": "pitch_deg", "chord": "chord_mm", "thickness": "t/c", "camber": "f/c",
                        "sweep": "sweep_mm", "rake": "rake_mm"} for k in PLANAR}}
FORMS = {"blade": ("table", "bezier1", "bezier2"), **{k: ("table", "bezier1", "bezier2") for k in PLANAR},
         "hull": ("cst",)}
SEGMENTS = {"table": 0, "bezier1": 1, "bezier2": 2}          # the form's segments (blade_modify's curve types)
FORM_OF = {0: "table", 1: "bezier1", 2: "bezier2"}
DECIMALS = {"blade": 8, **{k: 8 for k in PLANAR}, "hull": 9}  # the table's
TITLES = {"blade": "distributions over r/R", **{k: "distributions over eta = y/span" for k in PLANAR},
          "hull": "stations"}
ASSEMBLY_COMPONENTS = ("hull", "blade", "rudder", "fin", "wing")
PLACEMENT_KEYS = ("prop_dx", "prop_d", "rudder_dx", "rudder_angle", "rudders", "rudder_scale", "rudder_rake",
                  "fin_x", "fins", "fin_angle", "fin_scale", "wing_x", "wings", "wing_angle", "wing_scale", "view")
RUN_KEYS = ("samples", "batch", "seed", "cad_timeout_s")
KEY_WIDTH = 14


@dataclass
class Var:
    """One variable: its value, bounds and whether it is free (a design variable) or held."""
    value: float
    lo: float
    hi: float
    free: bool


@dataclass
class Component:
    """A component file's content. curves: name -> (form, n), n the points of a bezier1 curve or the order of a
    cst one (else None); vars: name -> Var in file order; vary: name -> (on, low, high) of vary lines."""
    kind: str
    name: str = ""
    notes: list = field(default_factory=list)
    geometry: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    curves: dict = field(default_factory=dict)
    vars: dict = field(default_factory=dict)
    vary: dict = field(default_factory=dict)
    columns: tuple = ()
    table: np.ndarray = None
    path: str = ""

    def column(self, name):
        return np.array(self.table[:, self.columns.index(name)], dtype=float)

    def has(self, name):
        return name in self.columns


@dataclass
class Assembly:
    """An assembly file's content: the component files (kind -> path as written, relative to the file), the
    placement and the run (key -> text)."""
    name: str = ""
    notes: list = field(default_factory=list)
    components: dict = field(default_factory=dict)
    placement: dict = field(default_factory=dict)
    run: dict = field(default_factory=dict)
    path: str = ""

    def component_path(self, kind):
        """The absolute path of a component's file (None when the assembly has none)."""
        p = self.components.get(kind)
        if not p:
            return None
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(os.path.dirname(self.path), p))


@dataclass
class Loaded:
    """A component as a module reads it (load_component): its parameters (the table and the geometry), its
    design and its design space (every variable, the held ones fixed), the file's settings (typed; an old
    parameter file has none), name, notes and extra geometry keys (kept and written back)."""
    kind: str
    params: object
    design: object
    space: object
    settings: dict
    name: str
    notes: list
    extra: dict
    legacy: bool


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def first_line(path):
    """The first line of a file that is not blank, stripped."""
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.strip():
                return line.strip()
    return ""


def _norm(line):
    return " ".join(line.lower().split())


def is_component(path):
    return _norm(first_line(path)) == _norm(COMPONENT)


def is_assembly(path):
    return _norm(first_line(path)) == _norm(ASSEMBLY)


def component_kind(path):
    """The kind a component file names (None for any other file)."""
    if not is_component(path):
        return None
    with open(path, errors="replace") as fh:
        for line in fh:
            parts = line.strip().lstrip("#").split()
            if len(parts) == 2 and parts[0] == "kind":
                return parts[1]
    return None


def as_bool(text, where=""):
    t = str(text).strip().upper()
    if t in ("Y", "YES", "1", "TRUE"):
        return True
    if t in ("N", "NO", "0", "FALSE"):
        return False
    raise ValueError(f"{where}: Y or N, not '{text}'")


def num(v, digits=10):
    """A number as the files write it (0 within 1e-12 of 0: rounding noise)."""
    v = float(v)
    if abs(v) < 1e-12:
        return "0"
    return f"{v:.{digits}g}"


def yn(flag):
    return "Y" if flag else "N"


def rel(path, start):
    """path relative to the folder start, with '/' (absolute when it is on another drive)."""
    try:
        return os.path.relpath(path, start).replace(os.sep, "/")
    except ValueError:
        return path


def resolve(text, base):
    """A path as written in a file: absolute, or relative to the folder base."""
    return text if os.path.isabs(text) else os.path.normpath(os.path.join(base, text))


def is_section_table(path):
    """Is the file para.py's section table (comma-separated rows of 4 numbers after one header line) rather than
    airfoil coordinates (x y per line)?"""
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.strip():
                rows.append(line)
            if len(rows) == 2:
                break
    return len(rows) == 2 and len(rows[1].split(",")) == 4


def typed_settings(comp, types):
    """{key: value} of a component's settings, each of the type types[key] (bool: Y or N; int; float)."""
    out = {}
    for k, text in comp.settings.items():
        where = f"{comp.path}: {k}"
        if types[k] is bool:
            out[k] = as_bool(text, where)
            continue
        try:
            v = float(text)
        except ValueError:
            raise ValueError(f"{where} is a number, not '{text}'") from None
        if types[k] is int:
            if v != int(v):
                raise ValueError(f"{where} is a whole number, not '{text}'")
            v = int(v)
        out[k] = v
    return out


def settings_text(settings, types):
    """The settings as a component file writes them (the keys of types, in its order)."""
    out = {}
    for k, t in types.items():
        if k in (settings or {}) and settings[k] is not None:
            v = settings[k]
            out[k] = yn(v) if t is bool else str(int(v)) if t is int else num(v)
    return out


def legacy_header(path, keys, extra_keys=()):
    """(notes, extra) of an old parameter file: its free-text '#' lines (not the first line, a key of keys, a
    vary line, the vary lines' explanation or the column line) and the first word after each of extra_keys."""
    head, _, _ = _lines(path)
    notes, extra = [], {}
    for _, s in head[1:-1]:
        text = s.lstrip("#").strip()
        parts = text.split()
        if parts[0] in extra_keys and len(parts) > 1:
            extra[parts[0]] = parts[1]
        elif parts[0] not in keys and parts[0] != "vary" and not text.startswith("Y/N per parameter"):
            notes.append(text)
    return notes, extra


def vars_of(space, design):
    """{variable: Var} of a design space with a design in it (blade_modify.BladeSpace, wing_modify.WingSpace or
    hull_modify.HullSpace: all_names, all_bounds, fixed, values)."""
    vals = space.values(design)
    return {n: Var(float(vals[n]), float(lo), float(hi), n not in space.fixed)
            for n, (lo, hi) in zip(space.all_names, space.all_bounds)}


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _lines(path):
    """(header lines as (number, words), the column names, the table rows) of a file with a '#' header and a
    table below it."""
    with open(path, errors="replace") as fh:
        lines = fh.read().splitlines()
    data = [i for i, s in enumerate(lines) if s.strip() and not s.strip().startswith("#")]
    first = data[0] if data else len(lines)
    head = [(i + 1, s.strip()) for i, s in enumerate(lines[:first])
            if s.strip().startswith("#") and s.strip().lstrip("#").strip()]
    for i in range(first, len(lines)):
        s = lines[i].strip()
        if s.startswith("#") and s.lstrip("#").strip():
            raise ValueError(f"{path}, line {i + 1}: header lines go above the table")
    return head, data, lines


def _columns_ok(kind, cols):
    base = COLUMNS[kind]
    if tuple(cols[:len(base)]) != base:
        return False
    rest = list(cols[len(base):])
    if kind == "hull":
        return rest in ([], ["rp_m"])
    if kind in PLANAR:
        k = len(rest) // 2
        return len(rest) % 2 == 0 and rest == [f"dt{i}" for i in range(1, k + 1)] + [f"dc{i}" for i in range(1, k + 1)]
    return not rest


def read_component(path):
    """The Component of a component file (see the module docstring); ValueError naming the line for anything
    it cannot read."""
    head, data, lines = _lines(path)
    if not head or _norm(head[0][1]) != _norm(COMPONENT):
        raise ValueError(f"{path}: not a component file (its first line is not '{COMPONENT}')")
    if not data:
        raise ValueError(f"{path}: no table")
    col_no, col_line = head[-1]
    body = head[1:-1]
    kind = None
    for k, s in body:
        parts = s.lstrip("#").split()
        if parts and parts[0] == "kind":
            if len(parts) != 2 or parts[1] not in KINDS:
                raise ValueError(f"{path}, line {k}: kind is one of {', '.join(KINDS)}")
            kind = parts[1]
    if kind is None:
        raise ValueError(f"{path}: no 'kind' line ({', '.join(KINDS)})")
    comp = Component(kind, path=os.path.abspath(path))
    keys = GEOMETRY[kind] + SETTINGS[kind]
    for k, s in body:
        text = s.lstrip("#").strip()
        parts = text.split()
        where = f"{path}, line {k}"
        if not parts or parts[0].startswith("---") or parts[0] == "kind":
            continue
        key, rest = parts[0], text[len(parts[0]):].strip()
        key = ALIASES[kind].get(key, key)
        if key == "name":
            comp.name = rest
        elif key == "note":
            comp.notes.append(rest)
        elif key in keys:
            if not rest:
                raise ValueError(f"{where}: {key} needs a value")
            if key in comp.geometry or key in comp.settings:
                raise ValueError(f"{where}: {key} is given twice")
            (comp.geometry if key in GEOMETRY[kind] else comp.settings)[key] = rest
        elif key == "curve":
            _read_curve(comp, parts[1:], where)
        elif key == "var":
            _read_var(comp, parts[1:], where)
        elif key == "vary":
            _read_vary(comp, parts[1:], where)
        else:
            raise ValueError(f"{where}: '{key}' is not a key of a {kind} file (geometry: {', '.join(GEOMETRY[kind])};"
                             f" settings: {', '.join(SETTINGS[kind]) or 'none'}; curve, var, vary, name, note); free "
                             f"text goes on '# note' lines")
    cols = col_line.lstrip("#").split()
    if not _columns_ok(kind, cols):
        extra = {"hull": " [rp_m]", "blade": ""}.get(kind, " [dt1 .. dtk dc1 .. dck]")
        raise ValueError(f"{path}, line {col_no}: the column line of a {kind} file is '# {' '.join(COLUMNS[kind])}"
                         f"{extra}', not '{col_line}'")
    rows = []
    for i in data:
        try:
            row = [float(v) for v in lines[i].split()]
        except ValueError:
            raise ValueError(f"{path}, line {i + 1}: a table row is numbers only") from None
        if len(row) != len(cols):
            raise ValueError(f"{path}, line {i + 1}: {len(row)} numbers for {len(cols)} columns")
        rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"{path}: the table needs 2 rows at least")
    comp.columns, comp.table = tuple(cols), np.array(rows, dtype=float)
    comp.name = comp.name or os.path.splitext(os.path.basename(path))[0]
    return comp


def _read_curve(comp, parts, where):
    if len(parts) not in (2, 3):
        raise ValueError(f"{where}: '# curve <name> <form> [<points or order>]'")
    name, form = parts[0], parts[1]
    if form not in FORMS[comp.kind]:
        raise ValueError(f"{where}: a {comp.kind} curve is {' or '.join(FORMS[comp.kind])}, not '{form}'")
    if name in comp.curves:
        raise ValueError(f"{where}: {name} has a curve line already")
    n = None
    if form in ("bezier1", "cst"):
        if len(parts) != 3:
            raise ValueError(f"{where}: {form} needs its " + ("number of points" if form == "bezier1" else "order"))
        try:
            n = int(parts[2])
        except ValueError:
            raise ValueError(f"{where}: {parts[2]} is not a whole number") from None
        if n < 2:
            raise ValueError(f"{where}: at least 2")
    elif len(parts) == 3:
        raise ValueError(f"{where}: {form} takes no number")
    comp.curves[name] = (form, n)


def _read_var(comp, parts, where):
    if len(parts) != 5:
        raise ValueError(f"{where}: '# var <name> <value> <low> <high> <Y|N>'")
    name = parts[0]
    try:
        value, lo, hi = (float(v) for v in parts[1:4])
    except ValueError:
        raise ValueError(f"{where}: value, low and high are numbers") from None
    if not lo < hi:
        raise ValueError(f"{where}: low must lie below high")
    if name in comp.vars:
        raise ValueError(f"{where}: {name} has a var line already")
    comp.vars[name] = Var(value, lo, hi, as_bool(parts[4], where))


def _read_vary(comp, parts, where):
    if len(parts) not in (2, 4):
        raise ValueError(f"{where}: '# vary <name> Y|N [low high]'")
    lo = hi = None
    if len(parts) == 4:
        try:
            lo, hi = float(parts[2]), float(parts[3])
        except ValueError:
            raise ValueError(f"{where}: low and high are numbers") from None
        if not lo < hi:
            raise ValueError(f"{where}: low must lie below high")
    if parts[0] in comp.vary:
        raise ValueError(f"{where}: {parts[0]} has a vary line already")
    comp.vary[parts[0]] = (as_bool(parts[1], where), lo, hi)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def _key(k, v):
    return f"# {k:<{KEY_WIDTH - 1}s} {v}"


def component_text(comp):
    """The text of a component file."""
    out = [COMPONENT, _key("kind", comp.kind), _key("name", comp.name or "component")]
    out += [_key("note", n) for n in comp.notes]
    if comp.geometry:
        out.append("# ---- geometry")
        out += [_key(k, v) for k, v in comp.geometry.items()]
    if comp.settings:
        out.append("# ---- settings")
        out += [_key(k, v) for k, v in comp.settings.items()]
    if comp.curves or comp.vars:
        forms = "cst <order>" if comp.kind == "hull" else "table | bezier1 <points> | bezier2"
        out.append(f"# ---- curves ({forms}) and variables (value, low, high, free Y or held N)")
        groups = {c: [] for c in comp.curves}
        loose = []
        for n in comp.vars:
            (groups[n.split(".")[0]] if n.split(".")[0] in groups else loose).append(n)
        w = max([len(n) for n in comp.vars] + [len(c) for c in comp.curves] + [8])

        def var_line(n):
            v = comp.vars[n]
            return f"# var    {n:<{w}s}  {num(v.value, 12):>18s}  {num(v.lo):>16s}  {num(v.hi):>16s}  {yn(v.free)}"
        out += [var_line(n) for n in loose]
        for c, (form, k) in comp.curves.items():
            out.append(f"# curve  {c:<{w}s}  {form}" + (f" {k}" if k is not None else ""))
            out += [var_line(n) for n in groups[c]]
    out.append(f"# ---- {TITLES[comp.kind]}")
    d = DECIMALS[comp.kind]
    cells = [[f"{0.0 if abs(v) < 0.5 * 10.0 ** -d else v:.{d}f}" for v in row] for row in comp.table]
    widths = [max([len(c)] + [len(r[j]) for r in cells]) for j, c in enumerate(comp.columns)]
    out.append("# " + "  ".join(c.rjust(widths[j]) for j, c in enumerate(comp.columns)))
    out += ["  " + "  ".join(r[j].rjust(widths[j]) for j in range(len(widths))) for r in cells]
    return "\n".join(out) + "\n"


def write_component(path, comp):
    """Write a component file (read_component reads it back)."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(component_text(comp))
    return path


# --------------------------------------------------------------------------
# Any component through its module; old parameter files
# --------------------------------------------------------------------------

LEGACY = (("propeller blade", "blade"), ("wing", "wing"), ("fin", "fin"), ("hull", "hull"))


def legacy_kind(path):
    """The kind of an old parameter file ('# XGeom <kind> parameters' first), else None."""
    first = _norm(first_line(path))
    if not first.startswith("# xgeom") or not first.endswith("parameters"):
        return None
    return next((kind for word, kind in LEGACY if word in first), None)


def file_kind(path):
    """The kind of a component file or an old parameter file, else None."""
    return component_kind(path) or legacy_kind(path)


def load(path, **kw):
    """The Loaded of a component file or an old parameter file, read by its kind's module (blade_modify,
    wing_modify, hull_modify; kw go to its load_component)."""
    kind = file_kind(path)
    if kind is None:
        raise ValueError(f"{path}: not a component file ('{COMPONENT}') or an old XGeom parameter file")
    if kind == "blade":
        import blade_modify as module
    elif kind == "hull":
        import hull_modify as module
    else:
        import wing_modify as module
    return module.load_component(path, **kw)


def convert(path, out=None, name=None):
    """Write an old parameter file (a blade's, a wing's, a fin's or a hull's) as a component file (default:
    beside it, .txt), its curves fitted and free as the design tool starts them, the tool's settings; returns
    the new file's path."""
    kind = legacy_kind(path)
    if kind is None:
        raise ValueError(f"{path}: not an old XGeom parameter file")
    out = out or os.path.splitext(path)[0] + ".txt"
    name = name or os.path.splitext(os.path.basename(out))[0]
    if kind == "blade":
        import blade_control as BC
        import blade_modify as BM
        ld = BM.load_component(path, BC.ORDERS, BC.SEGMENTS, BC.FREE)
        s = BC.settings
        st = {"sections": s.sections, "tip_band": s.tip_band, "tip_sections": s.tip_sections,
              "clearance_mm": s.clearance_mm, "hub": s.hub}
        return BM.write_component(out, ld.params, ld.design, ld.space, st, name, ld.notes, ld.extra)
    if kind == "hull":
        import hull_modify as HM
        ld = HM.load_component(path)
        return HM.write_component(out, ld.params, ld.design, ld.space, name, ld.notes)
    import wing_modify as WM
    import xgeom_wing as XW
    ld = WM.load_component(path, free=XW.FREE)
    return WM.write_component(out, ld.params, ld.design, ld.space, {"sections": WM.SECTIONS}, name, ld.notes, kind)


# --------------------------------------------------------------------------
# The assembly file
# --------------------------------------------------------------------------

def read_assembly(path):
    """The Assembly of an assembly file (see the module docstring)."""
    head, data, lines = _lines(path)
    if data:
        raise ValueError(f"{path}, line {data[0] + 1}: an assembly file has '#' lines only")
    if not head or _norm(head[0][1]) != _norm(ASSEMBLY):
        raise ValueError(f"{path}: not an assembly file (its first line is not '{ASSEMBLY}')")
    asm = Assembly(path=os.path.abspath(path))
    for k, s in head[1:]:
        text = s.lstrip("#").strip()
        parts = text.split()
        if not parts or parts[0].startswith("---"):
            continue
        key, rest = parts[0], text[len(parts[0]):].strip()
        where = f"{path}, line {k}"
        if key == "name":
            asm.name = rest
            continue
        if key == "note":
            asm.notes.append(rest)
            continue
        group = (asm.components if key in ASSEMBLY_COMPONENTS else asm.placement if key in PLACEMENT_KEYS
                 else asm.run if key in RUN_KEYS else None)
        if group is None:
            raise ValueError(f"{where}: '{key}' is not a key of an assembly file (components: "
                             f"{', '.join(ASSEMBLY_COMPONENTS)}; placement: {', '.join(PLACEMENT_KEYS)}; run: "
                             f"{', '.join(RUN_KEYS)}; name, note)")
        if not rest:
            raise ValueError(f"{where}: {key} needs a value")
        if key in group:
            raise ValueError(f"{where}: {key} is given twice")
        group[key] = rest
    asm.name = asm.name or os.path.splitext(os.path.basename(path))[0]
    return asm


def assembly_text(asm):
    out = [ASSEMBLY, _key("name", asm.name or "assembly")]
    out += [_key("note", n) for n in asm.notes]
    out.append("# ---- components (files relative to this one)")
    out += [_key(k, asm.components[k]) for k in ASSEMBLY_COMPONENTS if k in asm.components]
    if asm.placement:
        out.append("# ---- placement (the Vehicle tab; x in m from the nose, angles in deg from the top towards "
                   "starboard)")
        out += [_key(k, asm.placement[k]) for k in PLACEMENT_KEYS if k in asm.placement]
    if asm.run:
        out.append("# ---- optimization run")
        out += [_key(k, asm.run[k]) for k in RUN_KEYS if k in asm.run]
    return "\n".join(out) + "\n"


def write_assembly(path, asm):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(assembly_text(asm))
    return path


if __name__ == "__main__":
    import sys
    if len(sys.argv) not in (3, 4) or sys.argv[1] != "convert":
        print("python xgeom_files.py convert <old parameter file> [<component file>]")
        sys.exit(1)
    print(convert(sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None))
