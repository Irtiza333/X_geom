"""
xgeom_relations.py

Independent and dependent variables. Every variable of a design space has a value, the bounds low and high
and the flag free or held. An independent variable has numbers for its bounds and nothing else. A dependent
variable depends on other variables, in one or both of two ways.

A dependent bound: low or high is an expression of other variables in place of a number,

    low   pitch.v1           P4's pitch not under P1's (the bounds pitch.v1 .. 1.4)
    high  0.9*chord.v0       at most 0.9 of the root chord

The variable then lies between its bounds wherever the others are: when they move it keeps its place
between them, the fraction of the way from low to high, and that fraction, 0 to 1, is what an optimiser
varies, so every design vector inside its numbers gives a design inside the bounds. This is the MSc propeller
optimisation's way (para_control_bez_updated.py: P4's pitch p1y + (max_pitch - p1y) y4_frac; P2 and P5 placed
by the fractions d1 and d2 of the way from P4 to the ends).

A relation: a condition on top of the bounds,

    <= 0.5*chord.v2 + 2*chord.v3     at most that              ('<' reads the same)
    >  pitch.u4 + 0.2                at least that             ('>=' the same)
    =  chord.v0                      that value: it follows the others and is no design variable

The variable stays where it is until a relation would break and is then moved onto it; an optimiser checks
the relations of a design vector (violations) or moves it onto them (project): the MSc pipeline's coupled
limits (chord_sum, pitch_diff: reject or project).

An expression: numbers, names, + - * / and brackets ('0.5v4' is 0.5*v4). Several relations are separated by
commas ('> 0.5*v4, < 0.9*v0'). A variable is named as the files name it ('chord.v2', 'span'); one of the same
curve by its own key too ('v2'). A variable may depend on dependent ones but not, in a loop, on itself.

One structure in every set-up file (claude/xgeom-file-formats.md): the var line of a component file,

    # var  <name>  <value>  <low>  <high>  <Y|N>  [<relation>, ...]

low and high each a number or an expression (written without spaces), and the slot record of the rudder's
set-up JSON ("lo_expr", "hi_expr", "relation"). An independent variable has numbers and no relation.

In the design tool a variable moves within its limits (limits: its bounds, narrowed by its relations), and
when the others move the dependent ones follow, in the order of their dependence (apply): one with a
dependent bound to its place between its bounds, one whose relation would break onto it. Relations that
cannot hold inside the bounds leave the variable on a bound, and the tool warns.

    import xgeom_relations as REL
    rels = REL.parse("> 0.5v4, <= v0 + 2", names, curve="chord")       # names: the variables there are
    REL.text(rels)                                  # '> 0.5*chord.v4, <= chord.v0 + 2'
    low = REL.parse_bound("0.9v0", names, curve="chord")               # an Expression (a number: a float)

    ld = wing_modify.load_component("wing.txt")
    ld.space.relations, ld.space.dependent_bounds   # of the dependent variables
    ld.space.factors                                # the vector's variables that are fractions (bounds 0 .. 1)
    d = ld.space.to_design(x)                       # the fractions placed between their bounds
    ld.space.violations(x)                          # the relations a vector breaks: [(variable, text, amount)]
    x = ld.space.project(x)                         # the vector moved onto them

Standard library and numpy only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Real

import numpy as np

OPS = ("<=", ">=", "<", ">", "=")
UPPER, LOWER = ("<", "<="), (">", ">=")
TOL = 1.0e-9                       # a relation holds within this, times the larger of 1 and its two sides
_SPELLINGS = (("\u2264", "<="), ("\u2265", ">="), ("==", "="))
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


def _text(node, curve=None, spaced=True):
    """The tree as text, which _tree reads back as the same tree; the variables of curve by their keys; spaced:
    blanks around + and - (a relation), else none at all (a bound, one word of a var line)."""
    kind = node[0]
    if kind == "num":
        return _number(node[1])
    if kind == "var":
        name = node[1]
        return name[len(curve) + 1:] if curve and name.startswith(curve + ".") else name
    if kind == "neg":
        inner = _text(node[1], curve, spaced)
        return "-" + (inner if node[1][0] == "var" else f"({inner})")
    a, b = node[1], node[2]
    ta, tb = _text(a, curve, spaced), _text(b, curve, spaced)
    if a[0] in _PREC and _PREC[a[0]] < _PREC[kind]:
        ta = f"({ta})"
    if b[0] in _PREC and _PREC[b[0]] <= _PREC[kind]:         # left to right: a - (b - c), a/(b*c)
        tb = f"({tb})"
    return f"{ta} {kind} {tb}" if spaced and _PREC[kind] == 1 else f"{ta}{kind}{tb}"


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


def _resolver(names, curve):
    """The function that turns a name in a text into a variable: one of names (None: any name, as written),
    one of curve by its key too; ValueError for any other, saying which there are."""
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
    return resolve


# --------------------------------------------------------------------------
# Dependent bounds and relations
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Expression:
    """An expression of other variables (its tree): a dependent bound."""
    tree: tuple

    @property
    def names(self):
        """The variables the expression uses, as it names them first."""
        return tuple(dict.fromkeys(_names(self.tree, [])))

    def value(self, values):
        """Its value at {variable: value} (ZeroDivisionError when it divides by zero there)."""
        return _value(self.tree, values)

    def text(self, curve=None):
        """'0.5*chord.v2+2*chord.v3', one word, as a var line has a bound; the variables of curve by their keys
        ('0.5*v2+2*v3')."""
        return _text(self.tree, curve, spaced=False)


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


def parse_bound(bound, names=None, curve=None, own=None):
    """A bound as a var line or the tool's low and high boxes have it: a float for a number (also for an
    expression of numbers alone, '2*0.7'), an Expression for one that names variables ('0.9*v0': a dependent
    bound). names, curve, own: as parse. ValueError says what is wrong with the text."""
    if isinstance(bound, Real):
        value = float(bound)
    else:
        source = str(bound or "").strip()
        if not source:
            raise ValueError("a bound is a number or an expression of other variables")
        try:
            value = float(source)
        except ValueError:
            try:
                e = Expression(_tree(source, _resolver(names, curve)))
            except ValueError as exc:
                raise ValueError(f"'{source}': {exc}") from None
            if e.names:
                if own is not None and own in e.names:
                    raise ValueError(f"'{source}': a variable's bound cannot use the variable itself")
                return e
            try:
                value = float(e.value({}))
            except ZeroDivisionError:
                raise ValueError(f"'{source}' divides by zero") from None
    if not math.isfinite(value):
        raise ValueError("a bound is a finite number")
    return value


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
    resolve = _resolver(names, curve)
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


def order(relations, exprs=None):
    """The dependent variables, those of {variable: its relations} and of {variable: (low, high)} (the
    dependent bounds, each an Expression or None), in the order of their dependence: each after the dependent
    ones its relations and bounds use; ValueError for a loop."""
    exprs = exprs or {}
    nodes = list(dict.fromkeys(list(relations) + list(exprs)))
    known = set(nodes)
    uses = {n: [m for r in relations.get(n, ()) for m in r.names if m in known]
            + [m for e in exprs.get(n, ()) if e is not None for m in e.names if m in known] for n in nodes}
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

    for n in nodes:
        visit(n, [])
    return out


def _tol(*xs):
    return TOL * max(1.0, *(abs(x) for x in xs))


def bounds_at(name, values, bounds, exprs=None):
    """(low, high) of a variable at the values: each its number (bounds[name]), or for a dependent bound
    (exprs[name]) its expression's value there (its number where the expression divides by zero)."""
    lo, hi = bounds[name]
    for side, e in enumerate((exprs or {}).get(name, ())):
        if e is not None:
            try:
                v = e.value(values)
            except ZeroDivisionError:
                continue
            lo, hi = (v, hi) if side == 0 else (lo, v)
    return float(lo), float(hi)


def places(values, bounds, exprs):
    """{variable: (fraction, low, high)} of the variables with a dependent bound: where each lies between its
    bounds at the values, the fraction of the way from low to high (0 where the two meet)."""
    out = {}
    for n in exprs:
        lo, hi = bounds_at(n, values, bounds, exprs)
        out[n] = ((float(values[n]) - lo) / (hi - lo) if hi != lo else 0.0, lo, hi)
    return out


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


def limits(name, values, bounds, relations, exprs=None):
    """(low, high) a variable may take now: its bounds (bounds_at), narrowed by its relations at the other
    variables' values (an independent variable: its bounds; one defined with '=': that value twice, kept
    inside the bounds). low above high: the relations cannot hold inside the bounds."""
    lo, hi = bounds_at(name, values, bounds, exprs)
    lower, upper, equal = _sides(relations.get(name, ()), values)
    if equal is not None:
        v = min(max(equal, lo), hi)
        return v, v
    return max(lo, lower), min(hi, upper)


def nearest(name, value, values, bounds, relations, exprs=None):
    """The value nearest to `value` that a variable may take: inside its relations and its bounds (the smaller
    side when the relations cross, the bounds when the two cannot both hold); one defined with '=': its
    expression's, inside the bounds."""
    lo, hi = bounds_at(name, values, bounds, exprs)
    lower, upper, equal = _sides(relations.get(name, ()), values)
    v = float(value) if equal is None else equal
    return float(min(max(min(max(v, lower), upper), lo), hi))


def apply(values, bounds, relations, sequence=None, exprs=None, kept=None):
    """The values with the dependent variables following the others, taken in the order of their dependence
    (sequence: order(relations, exprs) when it is at hand):

    - a variable with a dependent bound whose bounds have moved goes to its place between them: kept, the
      places before the others moved ({variable: (fraction, low, high)}, places); without kept it stays;
    - one that breaks a relation of its own is moved to the nearest value that keeps them (nearest), one
      defined with '=' set to its expression.

    A variable whose bounds have not moved and whose relations hold stays where it is, outside its bounds too.
    Returns (values, the names moved)."""
    vals, moved = dict(values), []
    exprs, kept = exprs or {}, kept or {}
    for n in (order(relations, exprs) if sequence is None else sequence):
        v = new = vals[n]
        if n in kept:
            t, lo0, hi0 = kept[n]
            lo, hi = bounds_at(n, vals, bounds, exprs)
            if (lo, hi) != (lo0, hi0):
                new = lo + t * (hi - lo)
        rels = relations.get(n, ())
        if rels and not _holds(new, *_sides(rels, vals)):
            new = nearest(n, new, vals, bounds, relations, exprs)
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


def crossed(values, bounds, exprs):
    """The dependent bounds that leave no room at the values: [(variable, text, amount)], as violations has
    them: low above high (amount: by how much), or an expression that divides by zero there (inf). Empty:
    every variable has room between its bounds."""
    out = []
    for n, pair in exprs.items():
        for which, e in zip(("low", "high"), pair):
            if e is not None:
                try:
                    e.value(values)
                except ZeroDivisionError:
                    out.append((n, f"{n}: its {which} bound {e.text()} divides by zero", math.inf))
        lo, hi = bounds_at(n, values, bounds, exprs)
        if lo > hi + _tol(lo, hi):
            shown = [f"{w} {e.text()} = {x:.8g}" if e is not None else f"{w} {x:.8g}"
                     for w, e, x in zip(("low", "high"), pair, (lo, hi))]
            out.append((n, f"{n}: its bounds leave no room ({shown[0]} above {shown[1]})", float(lo - hi)))
    return out


# --------------------------------------------------------------------------
# The design spaces (blade_modify.BladeSpace, wing_modify.WingSpace, hull_modify.HullSpace,
# rudder_modify.DesignSpace)
# --------------------------------------------------------------------------

class SpaceRelations:
    """The dependent variables of a design space. The space has all_names and all_bounds (every variable),
    fixed ({held variable: value}), names and bounds (the design vector) and values(design).

    relations          {variable: its relations} (set_relations); empty: no relation
    dependent_bounds   {variable: (low, high)}, an Expression for a dependent bound, None for a number
    factors            the design vector's variables that are in it as fractions: the free ones with a
                       dependent bound. Their entry is the variable's place between its bounds, its row of
                       bounds 0 .. 1; to_design places it (low + fraction (high - low), at the others'
                       values), to_vector gives it back. A held variable with a dependent bound keeps the
                       place it has in the design the space was made with.
    A free variable with '<' or '>' relations stays in the design vector: an optimiser checks violations(x) or
    takes project(x). One defined with '=' is no design variable: it leaves the vector, and full_vector (so
    to_design) works it out from the others.

    all_bounds keeps a number for every bound: a dependent one's where the design the space was made with puts
    it (bounds_of(values): where any other design does)."""

    relations = {}
    dependent_bounds = {}
    factors = ()
    _order = ()
    _bounds0 = None                    # all_bounds as the space was made (set_relations starts from them)

    def set_relations(self, texts=None, where="", bounds=None, values=None):
        """Make variables dependent: these and no others (a second call starts again from the space as it was
        made).

        texts    {variable: relation text} (parse; an empty text: no relation)
        bounds   {variable: (low, high)}: an expression text for a dependent bound (parse_bound), None or ''
                 for a bound that stays its number
        values   {variable: value} of the design the space was made with (space.values(design)): with bounds,
                 the dependent bounds' numbers there (all_bounds) and the place of each held variable between
                 its bounds
        where    a file's path, to start the errors with
        The texts name variables in full, those of the variable's own curve by their keys too. ValueError for
        a text that is neither a relation nor a bound, a name that is no variable, or a loop."""
        names = list(self.all_names)
        pre = f"{where}: " if where else ""
        texts, bounds = texts or {}, bounds or {}
        unknown = sorted((set(texts) | set(bounds)) - set(names))
        if unknown:
            raise ValueError(f"{pre}relations or bounds for {', '.join(unknown)}, which this space does not have")
        rels, exprs, numbers = {}, {}, {}
        for n in names:                                      # in the space's order
            if texts.get(n):
                try:
                    r = parse(texts[n], names, curve_of(n), n)
                except ValueError as exc:
                    raise ValueError(f"{pre}the relation of {n}: {exc}") from None
                if r:
                    rels[n] = r
            pair = []
            for which, t in zip(("low", "high"), bounds.get(n) or (None, None)):
                try:
                    pair.append(None if t is None or t == "" else parse_bound(t, names, curve_of(n), n))
                except ValueError as exc:
                    raise ValueError(f"{pre}the {which} bound of {n}: {exc}") from None
            if any(isinstance(b, Expression) for b in pair):
                exprs[n] = tuple(b if isinstance(b, Expression) else None for b in pair)
            if any(isinstance(b, float) for b in pair):
                numbers[n] = pair
        try:
            sequence = order(rels, exprs)
        except ValueError as exc:
            raise ValueError(f"{pre}{exc}") from None
        if exprs and values is None:
            raise ValueError(f"{pre}dependent bounds need the design's values (set_relations(values=...))")
        if self._bounds0 is None:
            self._bounds0 = np.array(self.all_bounds, dtype=float)
        every = self._bounds0.copy()                         # (low, high) of every variable, as numbers
        index = {n: i for i, n in enumerate(names)}
        for n, pair in numbers.items():                      # a bound given as a number here
            for side, b in enumerate(pair):
                if isinstance(b, float):
                    every[index[n], side] = b
        given = {n: (float(lo), float(hi)) for n, (lo, hi) in zip(names, every)}
        for n in exprs:                                      # the dependent bounds where the design is
            every[index[n]] = bounds_at(n, values, given, exprs)
            for which, b, e in zip(("low", "high"), every[index[n]], exprs[n]):
                if not np.isfinite(b):                       # no number to fall back on (a file's bound)
                    raise ValueError(f"{pre}the {which} bound of {n}, {e.text()}, divides by zero at the design")
        self.all_bounds = every
        self._numbers = {n: (float(lo), float(hi)) for n, (lo, hi) in zip(names, every)}
        self.relations, self.dependent_bounds, self._order = rels, exprs, sequence
        self._defined = [n for n in sequence if n in rels and rels[n][0].op == "="]
        out = set(self.fixed) | set(self._defined)
        self._free = np.array([i for i, n in enumerate(names) if n not in out], dtype=int)
        self.names = [names[i] for i in self._free]
        self.bounds = every[self._free].reshape(-1, 2).copy()
        self.factors = [n for n in self.names if n in exprs]
        for n in self.factors:
            self.bounds[self.names.index(n)] = (0.0, 1.0)
        at = places(values, self._numbers, exprs) if exprs else {}
        self._held = {n: at[n][0] for n in exprs if n not in self.names and n not in self._defined}
        return self

    def relation(self, name):
        """A variable's relation text, as the files write it ('' for one with no relation)."""
        return text(self.relations.get(name, ()))

    def bound_text(self, name):
        """(low, high) of a variable's dependent bounds as the files write them ('0.9*chord.v0'); None for a
        bound that is a number."""
        return tuple(None if e is None else e.text() for e in self.dependent_bounds.get(name, (None, None)))

    def bounds_of(self, vals):
        """{variable: (low, high)} at {variable: value}: the numbers, the dependent bounds where the values put
        them."""
        if not self.dependent_bounds:
            return {n: (float(lo), float(hi)) for n, (lo, hi) in zip(self.all_names, self.all_bounds)}
        return {n: bounds_at(n, vals, self._numbers, self.dependent_bounds) for n in self.all_names}

    def full_vector(self, x):
        """Every variable's value: the design vector x and the held variables, and in the order of their
        dependence the variables with a dependent bound placed between their bounds (x's fraction; a held
        one's own) and the ones defined with '=' worked out."""
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self.names),):
            raise ValueError(f"the design vector has {len(self.names)} entries")
        full = np.array([self.fixed.get(n, np.nan) for n in self.all_names])
        full[self._free] = x
        if self.dependent_bounds or (self.relations and self._defined):
            vals = dict(zip(self.all_names, full))
            for n in self._order:
                if n in self._defined:
                    vals[n] = self.relations[n][0].value(vals)
                elif n in self.dependent_bounds:
                    t = self._held[n] if n in self._held else vals[n]
                    lo, hi = bounds_at(n, vals, self._numbers, self.dependent_bounds)
                    vals[n] = lo + t * (hi - lo)
            full = np.array([vals[n] for n in self.all_names])
        return full

    def vector_of(self, vals):
        """The design vector of {variable: value}: its variables' values, one with a dependent bound as the
        fraction of the way between its bounds there."""
        out = np.array([vals[n] for n in self.names], dtype=float)
        if self.factors:
            at = places(vals, self._numbers, self.dependent_bounds)
            for i, n in enumerate(self.names):
                if n in at:
                    out[i] = at[n][0]
        return out

    def to_vector(self, design):
        """The design vector of a design (vector_of its values)."""
        return self.vector_of(self.values(design))

    def vector_record(self, vals):
        """The design vector as plain data (the set-up JSONs' "vector"): names, values, bounds, and the names
        that are fractions when any is."""
        rec = {"names": self.names, "values": [float(v) for v in self.vector_of(vals)], "bounds": self.bounds.tolist()}
        if self.factors:
            rec["fractions"] = list(self.factors)
        return rec

    def violations(self, x):
        """What the design vector x breaks: [(variable, text, amount)], the relations that do not hold
        (xgeom_relations.violations) and the dependent bounds that leave no room there, low above high
        (crossed); empty when there is none. (Where a dependent bound has room, any fraction 0 .. 1 lies inside
        it.)"""
        return self._broken(dict(zip(self.all_names, self.full_vector(x))))

    def design_violations(self, design):
        """violations for a design (its values)."""
        return self._broken(self.values(design))

    def _broken(self, vals):
        out = violations(vals, self.relations)
        if self.dependent_bounds:
            out += crossed(vals, self._numbers, self.dependent_bounds)
        return out

    def project(self, x):
        """The design vector x moved onto its relations: the variables that break one moved to the nearest
        value that keeps it, in the order of their dependence (xgeom_relations.apply), the ones with a
        dependent bound keeping their fraction; x itself when it breaks nothing."""
        if not self.relations:
            return np.asarray(x, dtype=float)
        vals = dict(zip(self.all_names, self.full_vector(x)))
        kept = places(vals, self._numbers, self.dependent_bounds)
        vals, _ = apply(vals, self._numbers, self.relations, self._order, self.dependent_bounds, kept)
        return self.vector_of(vals)

    def slot_records(self, vals):
        """Every variable as plain data (the set-up JSONs' "slots"): name, value, lo, hi (a dependent bound:
        its number at the values), free, and for a dependent variable its relation and the expressions of its
        dependent bounds (lo_expr, hi_expr)."""
        at = self.bounds_of(vals)
        out = []
        for n in self.all_names:
            rec = {"name": n, "value": vals[n], "lo": at[n][0], "hi": at[n][1], "free": n not in self.fixed}
            if n in self.relations:
                rec["relation"] = self.relation(n)
            for key, t in zip(("lo_expr", "hi_expr"), self.bound_text(n)):
                if t is not None:
                    rec[key] = t
            out.append(rec)
        return out

    def set_slot_relations(self, slots):
        """set_relations from slot records (slot_records): their relations, dependent bounds and values."""
        return self.set_relations({q["name"]: q["relation"] for q in slots if q.get("relation")},
                                  bounds={q["name"]: (q.get("lo_expr"), q.get("hi_expr")) for q in slots
                                          if q.get("lo_expr") or q.get("hi_expr")},
                                  values={q["name"]: q["value"] for q in slots})


# --------------------------------------------------------------------------
# The design tool's adapters (xgeom_rudder, xgeom_blade, xgeom_hull, xgeom_wing)
# --------------------------------------------------------------------------

class AdapterRelations:
    """Dependent variables in the design tool. The adapter has meta ({variable: {"lo", "hi", "free"}, and for
    a dependent variable "rel", its relation text, and "lo_expr", "hi_expr", the expressions of its dependent
    bounds, all with full names; "lo" and "hi" are then the bound's number when it was last worked out),
    values ({variable: value}), space (design_of, values) and design; as_fitted where it keeps one."""

    @staticmethod
    def meta_of(space):
        """{variable: {"lo", "hi", "free"}, and "rel", "lo_expr", "hi_expr" for a dependent one} of a design
        space."""
        meta = {}
        for n, (lo, hi) in zip(space.all_names, space.all_bounds):
            meta[n] = {"lo": float(lo), "hi": float(hi), "free": n not in space.fixed}
            if space.relation(n):
                meta[n]["rel"] = space.relation(n)
            for key, t in zip(("lo_expr", "hi_expr"), space.bound_text(n)):
                if t is not None:
                    meta[n][key] = t
        return meta

    def _dependence(self):
        """(relations, dependent bounds) read from meta, again when a text or the variables have changed. An
        expression that names a variable no longer there is dropped (a relation gone, a bound its number),
        with a note (relation_notes)."""
        keys = ("rel", "lo_expr", "hi_expr")
        texts = [(n,) + tuple(m.get(k) for k in keys) for n, m in self.meta.items() if any(m.get(k) for k in keys)]
        key = (tuple(self.meta), tuple(texts)) if texts else None
        if getattr(self, "_rel_key", 0) != key:
            names, rels, exprs = list(self.meta), {}, {}
            notes = self.__dict__.setdefault("_rel_notes", [])
            for n, rel, lo, hi in texts:
                if rel:
                    try:
                        rels[n] = parse(rel, names, curve_of(n), n)
                    except ValueError as exc:
                        del self.meta[n]["rel"]
                        notes.append(f"the relation of {n} is dropped ({exc})")
                pair = []
                for k, which, t in (("lo_expr", "low", lo), ("hi_expr", "high", hi)):
                    try:
                        pair.append(parse_bound(t, names, curve_of(n), n) if t else None)
                    except ValueError as exc:
                        pair.append(None)
                        del self.meta[n][k]
                        notes.append(f"the {which} bound of {n} is a number again ({exc})")
                if any(isinstance(e, Expression) for e in pair):
                    exprs[n] = tuple(e if isinstance(e, Expression) else None for e in pair)
            self._rel_parsed, self._rel_exprs, self._rel_order = rels, exprs, order(rels, exprs)
            texts = [(n,) + tuple(m.get(k) for k in keys) for n, m in self.meta.items()
                     if any(m.get(k) for k in keys)]
            self._rel_key = (tuple(self.meta), tuple(texts)) if texts else None
        return self._rel_parsed, self._rel_exprs

    def _relations(self):
        return self._dependence()[0]

    def _bounds(self):
        """{variable: (low, high)}: the numbers kept in meta (a dependent bound: as last worked out)."""
        return {n: (m["lo"], m["hi"]) for n, m in self.meta.items()}

    def bounds(self, name):
        """(low, high) of a variable now: its numbers; a dependent bound where the other variables put it
        (meta's "lo" and "hi" are brought up to that)."""
        m = self.meta[name]
        e = self._dependence()[1].get(name)
        if e is not None:
            m["lo"], m["hi"] = bounds_at(name, self.values, {name: (m["lo"], m["hi"])}, {name: e})
        return m["lo"], m["hi"]

    def bound_text(self, name, short=True):
        """(low, high) of a variable's dependent bounds as texts ('0.9*v0'; short: the variables of its own
        curve by their keys, as the tool shows them); None for a bound that is a number."""
        pair = self._dependence()[1].get(name, (None, None))
        return tuple(None if e is None else e.text(curve_of(name) if short else None) for e in pair)

    def _bounds_given(self, name, lo, hi):
        """What set_bounds was given, each bound a number or an expression of other variables ('0.9*v0': a
        dependent bound): (low, high, keep), the two as numbers at the present design, for the adapter's own
        checks, and keep(), which records the dependent ones once those have passed (keep.dependent: is any
        of the two an expression?). ValueError for a text that is neither, a name that is no variable, or a
        loop."""
        rels, exprs = self._dependence()
        names, curve = list(self.meta), curve_of(name)
        sides = []
        for which, b in (("low", lo), ("high", hi)):
            try:
                sides.append(parse_bound(b, names, curve, name))
            except ValueError as exc:
                raise ValueError(f"the {which} bound: {exc}") from None
        pair = tuple(s if isinstance(s, Expression) else None for s in sides)
        trial = {n: e for n, e in exprs.items() if n != name}
        if any(e is not None for e in pair):
            trial[name] = pair
        order(rels, trial)
        now = []
        for which, s in zip(("low", "high"), sides):
            try:
                now.append(float(s.value(self.values)) if isinstance(s, Expression) else s)
            except ZeroDivisionError:
                raise ValueError(f"the {which} bound {s.text(curve)} divides by zero at the present design") from None

        def keep():
            for key, e in zip(("lo_expr", "hi_expr"), pair):
                if e is None:
                    self.meta[name].pop(key, None)
                else:
                    self.meta[name][key] = e.text()
        keep.dependent = any(e is not None for e in pair)
        return now[0], now[1], keep

    def relation(self, name, short=True):
        """A variable's relation text ('' for none); short: the variables of its own curve by their keys
        ('> 0.5*v4'), as the tool shows them, else every name in full, as the files have it."""
        return text(self._relations().get(name, ()), curve_of(name) if short else None)

    def is_dependent(self, name):
        """Does the variable depend on others: through a relation or a dependent bound?"""
        rels, exprs = self._dependence()
        return name in rels or name in exprs

    def is_defined(self, name):
        """Is the variable defined with '=' (it follows the others: no design variable)?"""
        return any(r.op == "=" for r in self._relations().get(name, ()))

    def varies(self, name):
        """Is the variable a design variable: free, and not defined with '='?"""
        return bool(self.meta[name]["free"]) and not self.is_defined(name)

    def set_relation(self, name, relation):
        """Give a variable a relation (a text such as '> 0.5*v4' or '<= 0.5*chord.v2 + 2*chord.v3, > 0'), or
        none (an empty text); the design follows at once. Returns the relation as the tool shows it.
        ValueError for a text that is no relation, a name that is no variable, or a loop; the variable is then
        as it was."""
        rels, exprs = self._dependence()
        new = parse(relation, list(self.meta), curve_of(name), name)
        trial = {n: (new if n == name else rels.get(n)) for n in self.meta
                 if (n == name and new) or (n != name and n in rels)}
        order(trial, exprs)
        if new:
            self.meta[name]["rel"] = text(new)
        else:
            self.meta[name].pop("rel", None)
        self.follow()
        return self.relation(name)

    def set_independent(self, name):
        """The variable independent again: its dependent bounds numbers, where they are now (set_bounds, with
        the adapter's own checks), and its relation dropped. ValueError where its bounds cannot be numbers
        (they cross, low above high, at the present design); the variable is then as it was."""
        m = self.meta[name]
        if m.get("lo_expr") or m.get("hi_expr"):
            lo, hi = self.bounds(name)
            try:
                self.set_bounds(name, lo, hi)
            except ValueError as exc:
                raise ValueError(f"{name} cannot be made independent with its bounds where they are now, "
                                 f"{lo:g} .. {hi:g} ({exc})") from None
        if self.meta[name].pop("rel", None) is not None:
            self.follow()

    def limits(self, name):
        """(low, high) the variable may take now: its bounds (a dependent one where the others put it),
        narrowed by its relations at the other variables' values (xgeom_relations.limits)."""
        rels, exprs = self._dependence()
        return limits(name, self.values, self._bounds(), rels, exprs)

    def _moved(self, name, value):
        """(values, the value used, the names moved) for one variable set to value: kept inside its limits,
        and the dependent ones following: one with a dependent bound to its place between its bounds, one
        whose relation would break onto it."""
        rels, exprs = self._dependence()
        bounds = self._bounds()
        vals = dict(self.values)
        kept = places(vals, bounds, exprs)
        kept.pop(name, None)
        vals[name] = value = nearest(name, value, vals, bounds, rels, exprs)
        vals, moved = apply(vals, bounds, rels, self._rel_order, exprs, kept)
        for n in exprs:                                      # the numbers kept for the dependent bounds
            self.meta[n]["lo"], self.meta[n]["hi"] = bounds_at(n, vals, bounds, exprs)
        return vals, value, [name] + moved

    def follow(self, note=False):
        """Move the variables that break a relation onto it as the design stands (after a new relation, new
        bounds, a curve's new form, a file read), one whose dependent bound moves with them to its place
        between its bounds; note: say which moved (relation_notes). The numbers kept for the dependent bounds
        are brought up to date. Returns the names moved."""
        rels, exprs = self._dependence()
        moved = []
        if rels:
            bounds = self._bounds()
            vals, moved = apply(self.values, bounds, rels, self._rel_order, exprs, places(self.values, bounds, exprs))
        if moved:
            self.design = self.space.design_of(vals)
            self.values = self.space.values(self.design)
            fitted = getattr(self, "as_fitted", None)
            if fitted is not None:
                for n in moved:
                    fitted.discard(curve_of(n))
            if note:
                self.__dict__.setdefault("_rel_notes", []).append("moved to keep their relations: " + ", ".join(moved))
        for n in exprs:
            self.meta[n]["lo"], self.meta[n]["hi"] = self.bounds(n)
        return moved

    def free_text(self):
        """'12 of 59 variables free', and how many depend on others when any does (the adapters' summary)."""
        n, m = self.n_free()
        rels, exprs = self._dependence()
        k = len(set(rels) | set(exprs))
        return f"{n} of {m} variables free" + (f", {k} dependent" if k else "")

    def relation_warnings(self):
        """What does not hold as the design stands, as texts: relations that cannot hold inside the bounds,
        dependent bounds that leave no room."""
        rels, exprs = self._dependence()
        out = violations(self.values, rels) if rels else []
        if exprs:
            out += crossed(self.values, self._bounds(), exprs)
        return [t for _, t, _ in out]

    def relation_notes(self):
        """What the relations did since the last call, as texts: one dropped with a variable that is gone,
        dependent variables moved by a file read or a curve's new form."""
        notes = self.__dict__.pop("_rel_notes", [])
        return list(notes)

    def related(self, space):
        """The design space with this adapter's relations and dependent bounds (free_space gives it to the
        files and an optimiser)."""
        return space.set_relations({n: m["rel"] for n, m in self.meta.items() if m.get("rel")},
                                   bounds={n: (m.get("lo_expr"), m.get("hi_expr")) for n, m in self.meta.items()
                                           if m.get("lo_expr") or m.get("hi_expr")}, values=self.values)
