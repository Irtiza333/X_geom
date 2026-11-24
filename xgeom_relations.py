"""
xgeom_relations.py

Independent and dependent variables. Every variable of a design space has a value, the bounds low and high
and the flag free or held. An independent variable is no more than that. A dependent variable also has one or
more relations to other variables,

    <= 0.5*chord.v2 + 2*chord.v3     at most that              ('<' reads the same)
    >  pitch.u4 + 0.2                at least that             ('>=' the same)
    =  chord.v0                      that value: it follows the others and is no design variable

each an operator and an expression of other variables: numbers, names, + - * / and brackets ('0.5v4' is
0.5*v4). Several relations are separated by commas ('> 0.5*v4, < 0.9*v0'). A variable is named as the files
name it ('chord.v2', 'span'); one of the same curve by its own key too ('v2').

One structure in every set-up file (claude/xgeom-file-formats.md): the var line of a component file,

    # var  <name>  <value>  <low>  <high>  <Y|N>  [<relation>, ...]

and the slot record of the rudder's set-up JSON (its "relation"). An independent variable has no relation.

What a relation does. The limits of a dependent variable are its bounds narrowed by its relations at the
other variables' present values. In the design tool a variable moves within its limits, and when the others
move the dependent ones follow (apply): each is moved into its limits, in the order of their dependence, so
a variable may depend on dependent ones but not, in a loop, on itself. Relations that cannot hold inside the
bounds leave the variable on a bound, and the tool warns. For an optimiser a free dependent variable stays a
design variable and its relations are constraints on the design vector (SpaceRelations: violations, project);
one defined with '=' leaves the vector and is worked out from it (full_vector, to_design).

    import xgeom_relations as REL
    rels = REL.parse("> 0.5v4, <= v0 + 2", names, curve="chord")       # names: the variables there are
    REL.text(rels)                                  # '> 0.5*chord.v4, <= chord.v0 + 2'
    values, moved = REL.apply(values, bounds, {"chord.v2": rels})

    ld = wing_modify.load_component("wing.txt")
    ld.space.relations                              # {variable: its relations} of the dependent ones
    ld.space.violations(x)                          # what a design vector breaks: [(variable, text, amount)]
    x = ld.space.project(x)                         # the dependent variables moved into their limits

Standard library and numpy only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

OPS = ("<=", ">=", "<", ">", "=")
UPPER, LOWER = ("<", "<="), (">", ">=")
TOL = 1.0e-9                       # a relation holds within this, times the larger of 1 and its two sides
_SPELLINGS = (("≤", "<="), ("≥", ">="), ("==", "="))
_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?")
_PREC = {"+": 1, "-": 1, "*": 2, "/": 2}


def curve_of(name):
    """The curve a variable belongs to ('chord' of 'chord.v2'); None for a global one ('span')."""
    return name.split(".")[0] if "." in name else None


# --------------------------------------------------------------------------
# Expressions: text -> tree -> text, value
# --------------------------------------------------------------------------

def _tokens(text):
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        m = _NUMBER.match(text, i)
        if m:
            out.append(("num", float(m.group())))
            i = m.end()
            continue
        m = _NAME.match(text, i)
        if m:
            if out and out[-1][0] == "num":                  # 0.5v4 is 0.5*v4
                out.append(("op", "*"))
            out.append(("name", m.group()))
            i = m.end()
            continue
        if ch in "+-*/()":
            if ch == "(" and out and out[-1][0] == "num":    # 2(a + b) is 2*(a + b)
                out.append(("op", "*"))
            out.append(("op", ch))
            i += 1
            continue
        raise ValueError(f"'{ch}' is not part of an expression (numbers, variables, + - * / and brackets)")
    return out


class _Parser:
    """expr: term (+|- term)*; term: factor (*|/ factor)*; factor: number | name | (expr) | -factor."""

    def __init__(self, tokens, resolve):
        self.t, self.i, self.resolve = tokens, 0, resolve

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expr(self):
        node = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            node = (op, node, self.term())
        return node

    def term(self):
        node = self.factor()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            node = (op, node, self.factor())
        return node

    def factor(self):
        kind, v = self.take()
        if kind == "num":
            return ("num", v)
        if kind == "name":
            return ("var", self.resolve(v))
        if (kind, v) == ("op", "("):
            node = self.expr()
            if self.take() != ("op", ")"):
                raise ValueError("a bracket is not closed")
            return node
        if (kind, v) == ("op", "-"):
            node = self.factor()
            return ("num", -node[1]) if node[0] == "num" else ("neg", node)
        if (kind, v) == ("op", "+"):
            return self.factor()
        raise ValueError("a number or a variable is missing " + ("at the end" if kind is None else f"before '{v}'"))


def _tree(text, resolve):
    tokens = _tokens(text)
    if not tokens:
        raise ValueError("no expression")
    p = _Parser(tokens, resolve)
    node = p.expr()
    if p.i < len(tokens):
        kind, v = tokens[p.i]
        if v == ")":
            raise ValueError("a bracket too many")
        raise ValueError(f"an operator is missing before '{_number(v) if kind == 'num' else v}'")
    return node


def _number(v):
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else repr(float(v))


def _text(node, curve=None):
    """The tree as text, which _tree reads back as the same tree; the variables of curve by their keys."""
    kind = node[0]
    if kind == "num":
        return _number(node[1])
    if kind == "var":
        name = node[1]
        return name[len(curve) + 1:] if curve and name.startswith(curve + ".") else name
    if kind == "neg":
        inner = _text(node[1], curve)
        return "-" + (inner if node[1][0] == "var" else f"({inner})")
    a, b = node[1], node[2]
    ta, tb = _text(a, curve), _text(b, curve)
    if a[0] in _PREC and _PREC[a[0]] < _PREC[kind]:
        ta = f"({ta})"
    if b[0] in _PREC and _PREC[b[0]] <= _PREC[kind]:         # left to right: a - (b - c), a/(b*c)
        tb = f"({tb})"
    return f"{ta} {kind} {tb}" if _PREC[kind] == 1 else f"{ta}{kind}{tb}"


def _value(node, values):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "var":
        return float(values[node[1]])
    if kind == "neg":
        return -_value(node[1], values)
    a, b = _value(node[1], values), _value(node[2], values)
    return a + b if kind == "+" else a - b if kind == "-" else a * b if kind == "*" else a / b


def _names(node, out):
    if node[0] == "var":
        out.append(node[1])
    elif node[0] != "num":
        for child in node[1:]:
            _names(child, out)
    return out


# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Relation:
    """One relation of a dependent variable: the operator ('<', '<=', '>', '>=' or '=') and the expression
    (its tree) of other variables."""
    op: str
    tree: tuple

    @property
    def names(self):
        """The variables the expression uses, as it names them first."""
        return tuple(dict.fromkeys(_names(self.tree, [])))

    def value(self, values):
        """The expression's value at {variable: value} (ZeroDivisionError when it divides by zero there)."""
        return _value(self.tree, values)

    def text(self, curve=None):
        """'<= 0.5*chord.v2 + 2*chord.v3'; the variables of curve by their keys ('<= 0.5*v2 + 2*v3')."""
        return f"{self.op} {_text(self.tree, curve)}"


def parse(relation, names=None, curve=None, own=None):
    """The relations of a text such as '> 0.5*v4, < 0.9*v0' (an empty text: none, an independent variable).

    names   the variables there are; None: any name is taken as written (the grammar alone is checked)
    curve   the curve whose variables may be named by their keys ('v4' for 'chord.v4')
    own     the variable the relations are for: it may lead a relation ('v1 <= 0.5*v2') and no relation of
            its own may use it
    ValueError says what is wrong with the text."""
    source = str(relation or "").strip()
    for a, b in _SPELLINGS:
        source = source.replace(a, b)
    if not source:
        return ()
    listed = None if names is None else list(names)
    known = None if names is None else set(listed)

    def resolve(token):
        if known is None:
            return token
        if "." not in token and curve and f"{curve}.{token}" in known:
            return f"{curve}.{token}"
        if token in known:
            return token
        c = curve_of(token) or curve
        keys = [n.split(".", 1)[1] for n in listed if c and n.startswith(c + ".")]
        if "." in token:
            raise ValueError(f"'{token}' is not a variable ("
                             + (f"{c}'s: {', '.join(keys)})" if keys else f"there is no curve {c})"))
        other = next((n for n in listed if "." in n and curve_of(n) != curve), None)
        raise ValueError(f"'{token}' is not a variable" + (f" of this curve ({', '.join(keys)})" if keys else "")
                         + "; a variable of another curve goes by its full name"
                         + (f", such as {other}" if other else ""))

    out = []
    for part in re.split(r"[,;]", source):
        part = part.strip()
        if not part:
            continue
        m = _NAME.match(part)
        if m and part[m.end():].lstrip()[:1] in ("<", ">", "="):      # 'v1 <= ...': the variable itself first
            if own is not None and resolve(m.group()) != own:
                raise ValueError(f"'{part}': the relation is {own}'s: it starts with an operator, or with {own}")
            part = part[m.end():].lstrip()
        op = next((o for o in OPS if part.startswith(o)), None)
        if op is None:
            raise ValueError(f"'{part}': a relation starts with one of < <= > >= = (such as '> 0.5*v4')")
        try:
            rel = Relation(op, _tree(part[len(op):], resolve))
        except ValueError as exc:
            raise ValueError(f"'{part}': {exc}") from None
        if own is not None and own in rel.names:
            raise ValueError(f"'{part}': a variable's relation cannot use the variable itself")
        out.append(rel)
    if any(r.op == "=" for r in out) and len(out) > 1:
        raise ValueError("a variable defined with '=' takes no other relation")
    return tuple(out)


def text(relations, curve=None):
    """The relations as one text, as the files write it and parse reads it back ('' for none); the variables of
    curve by their keys."""
    return ", ".join(r.text(curve) for r in relations)


def order(relations):
    """The dependent variables of {variable: its relations} in the order of their dependence, each after the
    dependent ones its relations use; ValueError for a loop."""
    uses = {n: [m for r in rels for m in r.names if m in relations] for n, rels in relations.items()}
    out, state = [], {}

    def visit(n, path):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            loop = path[path.index(n):] + [n]
            raise ValueError("variables that depend on each other in a loop (" + " -> ".join(loop)
                             + "): one of them has to be independent")
        state[n] = 1
        for m in uses[n]:
            visit(m, path + [n])
        state[n] = 2
        out.append(n)

    for n in relations:
        visit(n, [])
    return out


def _tol(*xs):
    return TOL * max(1.0, *(abs(x) for x in xs))


def _sides(rels, values):
    """(the largest 'at least', the smallest 'at most', the '=' value or None) of a variable's relations at
    the values; a relation that divides by zero there is left out."""
    lower, upper, equal = -math.inf, math.inf, None
    for r in rels:
        try:
            x = r.value(values)
        except ZeroDivisionError:
            continue
        if r.op in LOWER:
            lower = max(lower, x)
        elif r.op in UPPER:
            upper = min(upper, x)
        else:
            equal = x
    return lower, upper, equal


def _holds(v, lower, upper, equal):
    if equal is not None:
        return abs(v - equal) <= _tol(v, equal)
    return ((lower == -math.inf or v >= lower - _tol(v, lower))
            and (upper == math.inf or v <= upper + _tol(v, upper)))


def limits(name, values, bounds, relations):
    """(low, high) a variable may take now: its bounds, narrowed by its relations at the other variables'
    values (an independent variable: its bounds; one defined with '=': that value twice, kept inside the
    bounds). low above high: the relations cannot hold inside the bounds."""
    lo, hi = bounds[name]
    lower, upper, equal = _sides(relations.get(name, ()), values)
    if equal is not None:
        v = min(max(equal, lo), hi)
        return v, v
    return max(lo, lower), min(hi, upper)


def nearest(name, value, values, bounds, relations):
    """The value nearest to `value` that a variable may take: inside its relations and its bounds (the smaller
    side when the relations cross, the bounds when the two cannot both hold); one defined with '=': its
    expression's, inside the bounds."""
    lo, hi = bounds[name]
    lower, upper, equal = _sides(relations.get(name, ()), values)
    v = float(value) if equal is None else equal
    return float(min(max(min(max(v, lower), upper), lo), hi))


def apply(values, bounds, relations, sequence=None):
    """The values with the dependent variables following the others: taken in the order of their dependence
    (sequence: order(relations) when it is at hand), one that breaks a relation of its own is moved to the
    nearest value that keeps them (nearest), one defined with '=' set to its expression. A variable that keeps
    its relations stays where it is, outside its bounds too. Returns (values, the names moved)."""
    vals, moved = dict(values), []
    for n in (order(relations) if sequence is None else sequence):
        v = vals[n]
        if not _holds(v, *_sides(relations[n], vals)):
            new = nearest(n, v, vals, bounds, relations)
            if new != v:
                vals[n] = new
                moved.append(n)
    return vals, moved


def violations(values, relations):
    """What the values break: [(variable, text, amount)], one entry per relation that does not hold (amount:
    by how much, in the variable's units; inf for a relation that divides by zero there). Empty: every
    relation holds."""
    out = []
    for n, rels in relations.items():
        v = float(values[n])
        for r in rels:
            try:
                x = r.value(values)
            except ZeroDivisionError:
                out.append((n, f"{n}: its relation {r.text()} divides by zero", math.inf))
                continue
            gap = v - x if r.op in UPPER else x - v if r.op in LOWER else abs(v - x)
            if gap > _tol(v, x):
                out.append((n, f"{n} = {v:.8g} does not satisfy {r.text()} (= {x:.8g})", float(gap)))
    return out


# --------------------------------------------------------------------------
# The design spaces (blade_modify.BladeSpace, wing_modify.WingSpace, hull_modify.HullSpace,
# rudder_modify.DesignSpace)
# --------------------------------------------------------------------------

class SpaceRelations:
    """The dependent variables of a design space. The space has all_names and all_bounds (every variable),
    fixed ({held variable: value}), names and bounds (the design vector) and values(design).

    relations   {variable: its relations} of the dependent ones (set_relations); empty: all independent
    A free variable with '<' or '>' relations stays in the design vector: an optimiser checks violations(x) or
    takes project(x). One defined with '=' is no design variable: it leaves the vector, and full_vector (so
    to_design) works it out from the others."""

    relations = {}

    def set_relations(self, texts, where=""):
        """Make variables dependent: {variable: relation text} (parse; an empty text: independent). The texts
        name variables in full, those of the variable's own curve by their keys too. ValueError (starting
        with where, a file's path) for a text that is no relation, a name that is no variable, or a loop."""
        names = list(self.all_names)
        pre = f"{where}: " if where else ""
        rels = {}
        for n in self.all_names:                             # in the space's order
            t = (texts or {}).get(n)
            if not t:
                continue
            try:
                r = parse(t, names, curve_of(n), n)
            except ValueError as exc:
                raise ValueError(f"{pre}the relation of {n}: {exc}") from None
            if r:
                rels[n] = r
        unknown = sorted(set(texts or {}) - set(names))
        if unknown:
            raise ValueError(f"{pre}relations for {', '.join(unknown)}, which this space does not have")
        try:
            self._order = order(rels)
        except ValueError as exc:
            raise ValueError(f"{pre}{exc}") from None
        self.relations = rels
        self._defined = [n for n in self._order if rels[n][0].op == "="]
        out = set(self.fixed) | set(self._defined)
        self._free = np.array([i for i, n in enumerate(self.all_names) if n not in out], dtype=int)
        self.names = [self.all_names[i] for i in self._free]
        self.bounds = self.all_bounds[self._free].reshape(-1, 2)
        return self

    def relation(self, name):
        """A variable's relation text, as the files write it ('' for an independent variable)."""
        return text(self.relations.get(name, ()))

    def full_vector(self, x):
        """Every variable's value: the design vector x, the held variables, and the ones defined with '='
        worked out from them."""
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self.names),):
            raise ValueError(f"the design vector has {len(self.names)} entries")
        full = np.array([self.fixed.get(n, np.nan) for n in self.all_names])
        full[self._free] = x
        if self.relations and self._defined:
            vals = dict(zip(self.all_names, full))
            for n in self._defined:
                vals[n] = self.relations[n][0].value(vals)
            full = np.array([vals[n] for n in self.all_names])
        return full

    def _all_bounds(self):
        return {n: (float(lo), float(hi)) for n, (lo, hi) in zip(self.all_names, self.all_bounds)}

    def violations(self, x):
        """What the design vector x breaks: [(variable, text, amount)] (xgeom_relations.violations); empty when
        every relation holds."""
        return violations(dict(zip(self.all_names, self.full_vector(x))), self.relations)

    def design_violations(self, design):
        """violations for a design (its values)."""
        return violations(self.values(design), self.relations)

    def project(self, x):
        """The design vector x with the dependent variables moved into their limits, in the order of their
        dependence (xgeom_relations.apply); x itself when it breaks nothing."""
        if not self.relations:
            return np.asarray(x, dtype=float)
        vals, _ = apply(dict(zip(self.all_names, self.full_vector(x))), self._all_bounds(), self.relations,
                        self._order)
        return np.array([vals[n] for n in self.names])

    def slot_records(self, vals):
        """Every variable as plain data (the set-up JSONs' "slots"): name, value, lo, hi, free, and a dependent
        variable's relation."""
        return [dict({"name": n, "value": vals[n], "lo": float(lo), "hi": float(hi), "free": n not in self.fixed},
                     **({"relation": self.relation(n)} if n in self.relations else {}))
                for n, (lo, hi) in zip(self.all_names, self.all_bounds)]


# --------------------------------------------------------------------------
# The design tool's adapters (xgeom_rudder, xgeom_blade, xgeom_hull, xgeom_wing)
# --------------------------------------------------------------------------

class AdapterRelations:
    """Dependent variables in the design tool. The adapter has meta ({variable: {"lo", "hi", "free"}, and
    "rel", a dependent variable's relation text with full names), values ({variable: value}), space
    (design_of, values) and design; as_fitted where it keeps one."""

    @staticmethod
    def meta_of(space):
        """{variable: {"lo", "hi", "free"}, and "rel" for a dependent one} of a design space."""
        meta = {}
        for n, (lo, hi) in zip(space.all_names, space.all_bounds):
            meta[n] = {"lo": float(lo), "hi": float(hi), "free": n not in space.fixed}
            if space.relation(n):
                meta[n]["rel"] = space.relation(n)
        return meta

    def _relations(self):
        """{variable: its relations}, read from meta (again when a text has changed). A relation that names a
        variable no longer there is dropped, with a note (relation_notes)."""
        texts = [(n, m["rel"]) for n, m in self.meta.items() if m.get("rel")]
        key = (tuple(self.meta), tuple(texts)) if texts else None
        if getattr(self, "_rel_key", 0) != key:
            names, rels = list(self.meta), {}
            for n, t in texts:
                try:
                    rels[n] = parse(t, names, curve_of(n), n)
                except ValueError as exc:
                    del self.meta[n]["rel"]
                    self.__dict__.setdefault("_rel_notes", []).append(f"{n} is independent again ({exc})")
            self._rel_parsed, self._rel_order = rels, order(rels)
            texts = [(n, m["rel"]) for n, m in self.meta.items() if m.get("rel")]
            self._rel_key = (tuple(self.meta), tuple(texts)) if texts else None
        return self._rel_parsed

    def _bounds(self):
        return {n: (m["lo"], m["hi"]) for n, m in self.meta.items()}

    def relation(self, name, short=True):
        """A variable's relation text ('' for an independent one); short: the variables of its own curve by
        their keys ('> 0.5*v4'), as the tool shows them, else every name in full, as the files have it."""
        return text(self._relations().get(name, ()), curve_of(name) if short else None)

    def is_dependent(self, name):
        return name in self._relations()

    def is_defined(self, name):
        """Is the variable defined with '=' (it follows the others: no design variable)?"""
        return any(r.op == "=" for r in self._relations().get(name, ()))

    def varies(self, name):
        """Is the variable a design variable: free, and not defined with '='?"""
        return bool(self.meta[name]["free"]) and not self.is_defined(name)

    def set_relation(self, name, relation):
        """Make a variable dependent (a text such as '> 0.5*v4' or '<= 0.5*chord.v2 + 2*chord.v3, > 0'), or
        independent again (an empty text); the design follows at once. Returns the relation as the tool shows
        it. ValueError for a text that is no relation, a name that is no variable, or a loop; the variable is
        then as it was."""
        rels = parse(relation, list(self.meta), curve_of(name), name)
        trial = {n: (rels if n == name else self._relations().get(n)) for n in self.meta
                 if (n == name and rels) or (n != name and n in self._relations())}
        order(trial)
        if rels:
            self.meta[name]["rel"] = text(rels)
        else:
            self.meta[name].pop("rel", None)
        self.follow()
        return self.relation(name)

    def limits(self, name):
        """(low, high) the variable may take now: its bounds, narrowed by its relations at the other
        variables' values (xgeom_relations.limits)."""
        return limits(name, self.values, self._bounds(), self._relations())

    def _moved(self, name, value):
        """(values, the value used, the names moved) for one variable set to value: kept inside its limits,
        the variables that depend on it following."""
        rels = self._relations()
        bounds = self._bounds()
        vals = dict(self.values)
        vals[name] = value = nearest(name, value, vals, bounds, rels)
        vals, moved = apply(vals, bounds, rels, self._rel_order)
        return vals, value, [name] + moved

    def follow(self, note=False):
        """Move the dependent variables into their limits as the design stands (after a new relation, new
        bounds, a curve's new form, a file read); note: say which moved (relation_notes). Returns the names
        moved."""
        rels = self._relations()
        if not rels:
            return []
        vals, moved = apply(self.values, self._bounds(), rels, self._rel_order)
        if moved:
            self.design = self.space.design_of(vals)
            self.values = self.space.values(self.design)
            fitted = getattr(self, "as_fitted", None)
            if fitted is not None:
                for n in moved:
                    fitted.discard(curve_of(n))
            if note:
                self.__dict__.setdefault("_rel_notes", []).append("moved to keep their relations: " + ", ".join(moved))
        return moved

    def free_text(self):
        """'12 of 59 variables free', and how many depend on others when any does (the adapters' summary)."""
        n, m = self.n_free()
        k = len(self._relations())
        return f"{n} of {m} variables free" + (f", {k} dependent" if k else "")

    def relation_warnings(self):
        """The relations the design does not satisfy (they cannot hold inside the bounds), as texts."""
        rels = self._relations()
        return [t for _, t, _ in violations(self.values, rels)] if rels else []

    def relation_notes(self):
        """What the relations did since the last call, as texts: one dropped with a variable that is gone,
        dependent variables moved by a file read or a curve's new form."""
        notes = self.__dict__.pop("_rel_notes", [])
        return list(notes)

    def related(self, space):
        """The design space with this adapter's relations (free_space gives it to the files and an optimiser)."""
        return space.set_relations({n: m["rel"] for n, m in self.meta.items() if m.get("rel")})
