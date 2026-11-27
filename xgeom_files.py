"""
xgeom_files.py

The XGeom input files (claude/xgeom-file-formats.md): the component file, one layout for every component
(propeller blade, rudder, fin, wing, blended wing body, hull, and the rudder modified below H), and the assembly
file, which places the components on the hull and defines an optimization run. This module reads and writes
their text; blade_modify, wing_modify, hull_modify and rudder_modify turn a component into their parameters,
design and design space and back (load_component, write_component).
Standard library and numpy only. Files are written with LF line endings on every platform, so the same
content gives the same bytes on any machine.

Component file ('#' lines are the header, the rest is one table, which np.loadtxt reads):

    # XGeom component
    # kind          blade                        blade, rudder, fin, wing, bwb, rudder_h or hull
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
    # var    pitch.v1   1.24135133  0.6  1.4  Y   value, low, high, free (Y) or held (N): independent
    # var    pitch.v4   1.29  pitch.v1  1.4  Y    a bound that is an expression: dependent
    # var    pitch.v7   0.9  0.6  1.4  Y  <= 0.9*pitch.v1      a relation after the flag: dependent
    # ...
    # ---- distributions
    # r/R  P/D  c/D  t/c  f/c  skew_deg  rake/D    the last '#' line above the table names its columns
      0.18000000  1.24135133  ...

    kind    what the file describes; it decides the keys (GEOMETRY, SETTINGS) and the columns (COLUMNS)
    name    a short name (default: the file's name)
    note    free text, kept and written back
    curve   a distribution's form: table (the column as it is, the cubic spline through the rows; no
            variables), bezier1 <points> (one Bezier from the root to the tip), bezier2 (two rational cubic
            segments, x_blade's form), smooth2 (a blended wing body's: two cubic segments joined smoothly, flat
            at the centre line; wing_modify.SmoothTwoSegmentCurve), cst <order> (a hull curve). A curve of
            the rudder modified below H (kind rudder_h) is 'bezier1 <points>' followed by how it joins the
            original at H, G1 (on its tangent) or G0 (its value only), or by its two bounding laws, '<low law>
            <high law>' (one word each: the curve lies between them, its variables the fractions of the way)
    var     one variable, the same line in every component file: '<name> <value> <low> <high> <Y|N>
            [<relation>, ...]', its value, its bounds, whether it is free (a design variable) or held, and
            for a dependent variable its relation to others. A variable is independent with numbers for
            low and high and no relation. It depends on others (xgeom_relations.py) with
              a dependent bound: low or high an expression of other variables, written without blanks
                ('pitch.v1', '0.9*chord.v0'); the variable keeps its place between its bounds when the
                others move, and that place (0 to 1) is what an optimiser varies;
              a relation after the flag: one of < <= > >= = and an expression (numbers, names, + - * /,
                brackets), several separated by commas; a condition on top of the bounds ('=': the
                variable follows the others and is no design variable)
    vary    '<name> Y|N [low high]' (blade_vary.py): the curves and globals with no var lines are free (Y) or
            held (N), their values within low .. high when given; hand-written, or a blade's table file
            (blade_modify.write_params) that has them
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
    # ---- placement                              the Vehicle tab's settings (xgeom_vehicle.Placement):
    # prop_x         1                            positions as x/L (0 the nose, 1 the aft end): here the
    # rudder_x       0.89819                      hub's start and the rudders' root TE (BB2's aft half)
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

A table file is a component file with no curve or var lines (blade_modify, wing_modify and hull_modify
write_params write one): load_component fits the curves to its table.

Only component files are read (20 Nov 2026; the rudder modified below H since 26 Nov 2026, kind rudder_h:
rudder_modify.load_component). An old parameter file ('# XGeom <kind> parameters' first, .dat) is refused with a
message that names it (legacy_kind), and so is an old JSON set-up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

import xgeom_relations as REL

COMPONENT = "# XGeom component"
ASSEMBLY = "# XGeom assembly"
PLANAR = ("rudder", "fin", "wing", "bwb")         # bwb: a blended wing body (20 Nov 2026)
MODIFIED = "rudder_h"                             # the rudder modified below H (rudder_modify.py; 26 Nov 2026)
KINDS = ("blade",) + PLANAR + (MODIFIED, "hull")
JOINS = ("G1", "G0")                              # how a rudder_h curve joins the original at H
CAP_KEYS = {"scaled": ("cap_k_le", "cap_k_te", "cap_b1", "cap_b2", "cap_p", "cap_q"),
            "radius": ("cap_h_le_mm", "cap_h_te_mm", "cap_b1_mm", "cap_b2_mm", "cap_crown_radius_mm",
                       "cap_edge_radius_mm", "cap_te_radius_mm")}      # a planar surface's tip cap (tip_cap.py)
_PLANAR_GEOMETRY = (("span_mm", "section", "ref_xc", "camber_xc", "tip_tilt_deg", "cap") + CAP_KEYS["scaled"]
                    + CAP_KEYS["radius"])
GEOMETRY = {"blade": ("diameter_m", "blades", "root_r", "hub_height_m", "hub_base_m", "hub_top_m", "hub_x_m",
                      "section", "section_tables"),
            **{k: _PLANAR_GEOMETRY for k in PLANAR},
            MODIFIED: ("corner_points", "section_table", "xcad_original", "stack", "degree", "span_mm",
                       "fit_orders", "fit_free_heights"),
            "hull": ("nose_m", "middle_m", "tail_m", "cap_m", "axisymmetric")}
SETTINGS = {"blade": ("sections", "tip_band", "tip_sections", "clearance_mm", "hub", "part", "hub_extends"),
            **{k: ("sections", "change_points", "half", "cap_loops") for k in PLANAR},
            MODIFIED: ("sections", "points", "te_radius_mm", "half", "strip_frac", "tip_drop", "keep_inside",
                       "inside_tol_mm"),
            "hull": ("half",)}
ALIASES = {"blade": {"section_table": "section", "airfoil": "section"},          # the old parameter files' keys,
                                                                                   # read as these
           **{k: {"airfoil": "section"} for k in PLANAR}, MODIFIED: {}, "hull": {}}
COLUMNS = {"blade": ("r/R", "P/D", "c/D", "t/c", "f/c", "skew_deg", "rake/D"),
           **{k: ("eta", "pitch_deg", "chord_mm", "t/c", "f/c", "sweep_mm", "rake_mm") for k in PLANAR},
           MODIFIED: ("y_mm", "LE_mm"),            # then P1x P1z .. P(n-1)x P(n-1)z, the section's inner points
           "hull": ("x_m", "r_m")}                 # then a planar surface's dt1 .. dtk dc1 .. dck, the hull's rp_m
CURVE_COLUMNS = {"blade": {"pitch": "P/D", "chord": "c/D", "thickness": "t/c", "camber": "f/c", "skew": "skew_deg",
                           "rake": "rake/D"},
                 **{k: {"pitch": "pitch_deg", "chord": "chord_mm", "thickness": "t/c", "camber": "f/c",
                        "sweep": "sweep_mm", "rake": "rake_mm"} for k in PLANAR}}
FORMS = {"blade": ("table", "bezier1", "bezier2"), **{k: ("table", "bezier1", "bezier2") for k in PLANAR},
         "bwb": ("table", "bezier1", "smooth2"), MODIFIED: ("bezier1",), "hull": ("cst",)}
SEGMENTS = {"table": 0, "bezier1": 1, "bezier2": 2, "smooth2": 3}   # the form's code (wing_modify.CURVE_TYPES)
FORM_OF = {0: "table", 1: "bezier1", 2: "bezier2", 3: "smooth2"}
DECIMALS = {"blade": 8, **{k: 8 for k in PLANAR}, MODIFIED: 8, "hull": 9}  # the table's
TITLES = {"blade": "distributions over r/R", **{k: "distributions over eta = y/span" for k in PLANAR},
          "bwb": "distributions over eta = y/span from the centre line",
          MODIFIED: "the curves from the root to H (what the var lines give; not read back)", "hull": "stations"}
ASSEMBLY_COMPONENTS = ("hull", "blade", "rudder", "fin", "wing")
PLACEMENT_KEYS = ("prop_x", "prop_d", "rudder_x", "rudder_angle", "rudders", "rudder_scale", "rudder_rake",
                  "fin_x", "fins", "fin_angle", "fin_scale", "wing_x", "wings", "wing_angle", "wing_scale", "view")
RUN_KEYS = ("samples", "batch", "seed", "cad_timeout_s")
KEY_WIDTH = 14
VAR_TITLE = ("# ---- var <name> <value> <low> <high> <Y free | N held> [<relation>]",
             "# ----     independent: numbers for low and high, no relation",
             "# ----     dependent:   low or high an expression of other variables (one word, such as 0.9*{c}),",
             "# ----                  or a relation after the flag: an operator (< <= > >= =) and an expression,",
             "# ----                  such as <= 0.5*{a} + 2*{b}")
VAR_EXAMPLE = {**{k: dict(a="chord.v2", b="chord.v3", c="chord.v0") for k in KINDS},      # the var line's title,
               MODIFIED: dict(a="LE.dx1", b="LE.dx2", c="LE.dx3"),                       # the same in every
               "hull": dict(a="nose_r.b1", b="nose_r.b2", c="nose_r.b0")}                # component file
CURVE_TITLE = {"hull": ("# ---- curve <name> <form>: cst <order>",),
               "bwb": ("# ---- curve <name> <form>: table | bezier1 <points> | smooth2",),
               MODIFIED: ("# ---- curve <name> bezier1 <points> <G1: on the original's tangent at H | G0: its value "
                          "only>",
                          "# ----    or  <name> bezier1 <points> <low law> <high law>: the curve between two laws "
                          "(one word each),",
                          "# ----        its variables f1 .. the fractions of the way, 0 on the low law, 1 on the "
                          "high one")}


@dataclass
class Var:
    """One variable: its value, bounds and whether it is free (a design variable) or held; a dependent
    variable's relation to others and the expressions of its dependent bounds (xgeom_relations.py; '' for
    none: lo_expr and hi_expr as the file writes them, lo and hi then their values at the file's design, nan
    where the var lines alone do not give them)."""
    value: float
    lo: float
    hi: float
    free: bool
    relation: str = ""
    lo_expr: str = ""
    hi_expr: str = ""


@dataclass
class Component:
    """A component file's content. curves: name -> (form, n), n the points of a bezier1 curve or the order of a
    cst one (else None); vars: name -> Var in file order; vary: name -> (on, low, high) of vary lines; of a
    rudder modified below H (kind rudder_h): joins, name -> 'G1' or 'G0', and laws, name -> (low law, high
    law), of the curves that have them."""
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
    joins: dict = field(default_factory=dict)
    laws: dict = field(default_factory=dict)

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
    design and its design space (every variable, the held ones fixed), the file's settings (typed), name, notes
    and extra geometry keys (kept and written back)."""
    kind: str
    params: object
    design: object
    space: object
    settings: dict
    name: str
    notes: list
    extra: dict


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
        if types[k] is str:
            out[k] = text.strip()
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
            out[k] = yn(v) if t is bool else str(int(v)) if t is int else str(v) if t is str else num(v)
    return out


def vars_of(space, design):
    """{variable: Var} of a design space with a design in it (blade_modify.BladeSpace, wing_modify.WingSpace or
    hull_modify.HullSpace: all_names, all_bounds, fixed, values, and for the dependent ones relation and
    bound_text; a dependent bound's number is its value at the design)."""
    vals = space.values(design)
    at = space.bounds_of(vals)
    return {n: Var(float(vals[n]), at[n][0], at[n][1], n not in space.fixed, space.relation(n),
                   *(t or "" for t in space.bound_text(n))) for n in space.all_names}


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
    if kind == MODIFIED:                           # the inner points of a section of degree 4 or more
        k = len(rest) // 2
        return len(rest) % 2 == 0 and k >= 3 and rest == [f"P{i}{a}" for i in range(1, k + 1) for a in "xz"]
    if kind in PLANAR:
        k = len(rest) // 2
        return len(rest) % 2 == 0 and rest == [f"dt{i}" for i in range(1, k + 1)] + [f"dc{i}" for i in range(1, k + 1)]
    return not rest


def read_component(path):
    """The Component of a component file (see the module docstring); ValueError naming the line for anything
    it cannot read."""
    head, data, lines = _lines(path)
    if not head or _norm(head[0][1]) != _norm(COMPONENT):
        raise not_component(path)
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
    _dependent_bounds(comp)
    cols = col_line.lstrip("#").split()
    if not _columns_ok(kind, cols):
        extra = {"hull": " [rp_m]", "blade": "", MODIFIED: " P1x P1z .. P(n-1)x P(n-1)z"}.get(
            kind, " [dt1 .. dtk dc1 .. dck]")
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


def _read_modified_curve(comp, parts, where):
    """A curve line of a rudder modified below H: '<name> bezier1 <points> <G1|G0>' or '<name> bezier1 <points>
    <low law> <high law>' (the laws' grammar here; their names with the model, rudder_modify)."""
    if len(parts) not in (4, 5) or parts[1] != "bezier1":
        raise ValueError(f"{where}: '# curve <name> bezier1 <points> <G1|G0>' or '# curve <name> bezier1 <points> "
                         f"<low law> <high law>' (a law is one word, no blanks)")
    name = parts[0]
    if name in comp.curves:
        raise ValueError(f"{where}: {name} has a curve line already")
    try:
        n = int(parts[2])
    except ValueError:
        raise ValueError(f"{where}: {parts[2]} is not a whole number") from None
    if n < 2:
        raise ValueError(f"{where}: at least 2")
    if len(parts) == 4:
        if parts[3] not in JOINS:
            raise ValueError(f"{where}: after the points comes G1 (on the original's tangent at H) or G0 (its "
                             f"value only), or a low law and a high law; not '{parts[3]}' alone")
        comp.joins[name] = parts[3]
    else:
        for which, word in zip(("low", "high"), parts[3:]):
            try:
                REL.parse_law(word)
            except ValueError as exc:
                raise ValueError(f"{where}: the {which} law of {name}: {exc}") from None
        comp.laws[name] = (parts[3], parts[4])
    comp.curves[name] = ("bezier1", n)


def _read_curve(comp, parts, where):
    if comp.kind == MODIFIED:
        return _read_modified_curve(comp, parts, where)
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
    if len(parts) < 5:
        raise ValueError(f"{where}: '# var <name> <value> <low> <high> <Y|N> [<relation>, ...]'")
    name = parts[0]
    try:
        value = float(parts[1])
    except ValueError:
        raise ValueError(f"{where}: the value is a number") from None
    try:
        free = as_bool(parts[4])
    except ValueError:
        raise ValueError(f"{where}: after the bounds comes Y or N, not '{parts[4]}' (a bound that is an "
                         f"expression is one word, no blanks: 0.9*chord.v0)") from None
    bounds, exprs = [], []
    for which, word in zip(("low", "high"), parts[2:4]):          # each a number, or an expression of other
        try:                                                      # variables (a dependent bound): its grammar
            b = REL.parse_bound(word)                             # here, its names and value with the others
        except ValueError as exc:                                 # (_dependent_bounds, the component's space)
            raise ValueError(f"{where}: the {which} bound of {name}: {exc}") from None
        bounds.append(b if isinstance(b, float) else np.nan)
        exprs.append("" if isinstance(b, float) else word)
    if not any(exprs) and not bounds[0] < bounds[1]:
        raise ValueError(f"{where}: low must lie below high")
    if name in comp.vars:
        raise ValueError(f"{where}: {name} has a var line already")
    relation = " ".join(parts[5:])                    # a dependent variable's; its names are checked with the
    try:                                              # space (the modules' load_component), its grammar here
        REL.parse(relation)
    except ValueError as exc:
        raise ValueError(f"{where}: the relation of {name}: {exc}") from None
    comp.vars[name] = Var(value, bounds[0], bounds[1], free, relation, *exprs)


def _dependent_bounds(comp):
    """The numbers of the dependent bounds (Var.lo, Var.hi under lo_expr, hi_expr): each expression's value at
    the file's design, where the var lines give it. One that names a variable with no var line (a curve fitted
    from its column), or that cannot be worked out there, stays nan: the component's space (the modules'
    load_component) works it out with every variable, and says what is wrong with one that it cannot."""
    names = list(comp.vars)
    values = {n: q.value for n, q in comp.vars.items()}
    for n, q in comp.vars.items():
        for key, word in (("lo", q.lo_expr), ("hi", q.hi_expr)):
            if word:
                try:
                    setattr(q, key, float(REL.parse_bound(word, names, REL.curve_of(n), n).value(values)))
                except (ValueError, ZeroDivisionError):
                    pass


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
        out += ["# ---- curves and variables"]
        out += CURVE_TITLE.get(comp.kind, ("# ---- curve <name> <form>: table | bezier1 <points> | bezier2",))
        out += [t.format(**VAR_EXAMPLE[comp.kind]) for t in VAR_TITLE]
        groups = {c: [] for c in comp.curves}
        loose = []
        for n in comp.vars:
            (groups[n.split(".")[0]] if n.split(".")[0] in groups else loose).append(n)
        w = max([len(n) for n in comp.vars] + [len(c) for c in comp.curves] + [8])

        def var_line(n):
            v = comp.vars[n]
            lo, hi = v.lo_expr or num(v.lo, 12), v.hi_expr or num(v.hi, 12)    # a dependent bound: its expression
            return (f"# var    {n:<{w}s}  {num(v.value, 12):>18s}  {lo:>16s}  {hi:>16s}  {yn(v.free)}"
                    + (f"  {v.relation}" if v.relation else ""))
        out += [var_line(n) for n in loose]
        for c, (form, k) in comp.curves.items():
            tail = " " + " ".join(comp.laws[c]) if c in comp.laws else f" {comp.joins[c]}" if c in comp.joins else ""
            out.append(f"# curve  {c:<{w}s}  {form}" + (f" {k}" if k is not None else "") + tail)
            out += [var_line(n) for n in groups[c]]
    if comp.vary:
        out.append("# ---- vary (Y: changes, its values within low .. high when given; N: kept as the table has it)")
        w = max(len(n) for n in comp.vary)
        out += [f"# vary   {n:<{w}s}  {yn(on)}" + (f"  {num(lo)}  {num(hi)}" if lo is not None else "")
                for n, (on, lo, hi) in comp.vary.items()]
    out.append(f"# ---- {TITLES[comp.kind]}")
    d = DECIMALS[comp.kind]
    cells = [[f"{0.0 if abs(v) < 0.5 * 10.0 ** -d else v:.{d}f}" for v in row] for row in comp.table]
    widths = [max([len(c)] + [len(r[j]) for r in cells]) for j, c in enumerate(comp.columns)]
    out.append("# " + "  ".join(c.rjust(widths[j]) for j, c in enumerate(comp.columns)))
    out += ["  " + "  ".join(r[j].rjust(widths[j]) for j in range(len(widths))) for r in cells]
    return "\n".join(out) + "\n"


def write_component(path, comp):
    """Write a component file (read_component reads it back; LF line endings on every platform)."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write(component_text(comp))
    return path


# --------------------------------------------------------------------------
# Any component through its module; old parameter files refused
# --------------------------------------------------------------------------

LEGACY = (("propeller blade", "blade"), ("wing", "wing"), ("fin", "fin"), ("hull", "hull"))


def legacy_kind(path):
    """The kind of an old parameter file ('# XGeom <kind> parameters' first, which is no longer read), else
    None."""
    first = _norm(first_line(path))
    if not first.startswith("# xgeom") or not first.endswith("parameters"):
        return None
    return next((kind for word, kind in LEGACY if word in first), None)


def not_component(path):
    """The error for a file that is not a component file; an old parameter file, or a JSON file (an old
    set-up), is named as one."""
    old = legacy_kind(path)
    if old:
        return ValueError(f"{path}: an old {old} parameter file, which XGeom no longer reads (since 20 Nov 2026): "
                          f"it reads component files only ('{COMPONENT}' first)")
    if first_line(path).startswith("{"):
        return ValueError(f"{path}: a JSON file, not a component file: XGeom no longer reads JSON set-ups (the "
                          f"rudder's until 26 Nov 2026), it reads component files only ('{COMPONENT}' first)")
    return ValueError(f"{path}: not a component file (its first line is not '{COMPONENT}')")


def file_kind(path):
    """The kind of a component file, or of an old parameter file (the tool then routes it to its component,
    whose module refuses it), else None."""
    return component_kind(path) or legacy_kind(path)


def load(path, **kw):
    """The Loaded of a component file, read by its kind's module (blade_modify, wing_modify, hull_modify,
    rudder_modify; kw go to its load_component)."""
    kind = read_component(path).kind
    if kind == "blade":
        import blade_modify as module
    elif kind == "hull":
        import hull_modify as module
    elif kind == MODIFIED:
        import rudder_modify as module
    else:
        import wing_modify as module
    return module.load_component(path, **kw)


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
        out.append("# ---- placement (the Vehicle tab; positions x/L, 0 the nose, 1 the aft end; angles in deg from "
                   "the top towards starboard)")
        out += [_key(k, asm.placement[k]) for k in PLACEMENT_KEYS if k in asm.placement]
    if asm.run:
        out.append("# ---- optimization run")
        out += [_key(k, asm.run[k]) for k in RUN_KEYS if k in asm.run]
    return "\n".join(out) + "\n"


def write_assembly(path, asm):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write(assembly_text(asm))
    return path

