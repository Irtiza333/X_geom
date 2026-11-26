"""
_xgeom_relations_test.py

Self-check for dependent variables (xgeom_relations.py): the grammar of a relation and of a bound, what a
relation does to a variable's limits, where a dependent bound puts them, and how the dependent variables follow
the others; the design spaces of every component (blade, wing, hull, the rudder modified below H) with
relations and dependent bounds (the design vector's fractions, as the MSc propeller optimisation has them); the
set-up files that carry them (the var line of a component file, the rudder's set-up JSON); the design tool's
adapters; with --gui the window's low and high boxes, dep switch and relation entry.

    python _xgeom_relations_test.py          the module, the spaces, the files and the adapters
    python _xgeom_relations_test.py --gui    also the window

The rudder modified below H needs the extraction's outputs in outputs/ and is skipped without them. Where the
other components are not (the Rudder folder: no wing_modify.py), the module and the rudder are checked.
"""

import filecmp
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import xgeom_relations as REL

_FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))
ALL = importlib.util.find_spec("wing_modify") is not None      # the blade, the wing and the hull are here
if ALL:
    import blade_modify as BM
    import blade_vary
    import hull_modify as HM
    import wing_modify as WM
    import xgeom_files as XF
NAMES = ["span", "chord.d1", "chord.d2", "chord.v0", "chord.v1", "chord.v2", "chord.v3", "pitch.v0", "pitch.v1"]
RUDDER_FILES = ["outputs/corner_points.dat", "outputs/thickness_params_y0_y200_info.dat",
                "outputs/sections_xcad_with_cap_y0_y200_m.dat", "outputs/sections_stack_y0_y200_raw_mm.dat"]


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<76s} max error {err:.3e}   (tol {tol:.0e})")


def message(fn, *args, **kw):
    """The ValueError's text ('' when fn raises none)."""
    try:
        fn(*args, **kw)
        return ""
    except ValueError as exc:
        return str(exc)


def relations_of(obj):
    """{variable: relation text} of a design space."""
    return {n: obj.relation(n) for n in obj.relations}


# --------------------------------------------------------------------------
# The module
# --------------------------------------------------------------------------

def test_grammar():
    print("\nthe grammar of a relation (xgeom_relations.parse, text)")

    def full(t, curve="chord", own="chord.v1"):
        return REL.text(REL.parse(t, NAMES, curve, own))

    check("an operator and an expression; a variable of the curve by its key, another by its full name",
          [full("> 0.5*v2") == "> 0.5*chord.v2", full("<= 0.5*v2 + 2*v3") == "<= 0.5*chord.v2 + 2*chord.v3",
           full("< pitch.v0 - span/100") == "< pitch.v0 - span/100", full(">= chord.v0") == ">= chord.v0",
           full("= v0") == "= chord.v0"], [1] * 5, 0)
    check("no spaces needed, 0.5v2 is 0.5*v2, 2(a + b) is 2*(a + b); the variable itself may lead",
          [full(">0.5v2") == "> 0.5*chord.v2", full("<2(v2+v3)") == "< 2*(chord.v2 + chord.v3)",
           full("v1 <= 0.5*v2") == "<= 0.5*chord.v2", full("chord.v1>v3") == "> chord.v3",
           full("<1e-3span") == "< 0.001*span"], [1] * 5, 0)
    check("several relations with commas (or semicolons); the spellings of at most and at least",
          [full("> 0.5v2, < 0.9*v0") == "> 0.5*chord.v2, < 0.9*chord.v0", full(">v3;<v0") == "> chord.v3, < chord.v0",
           full("\u2264 v0") == "<= chord.v0", full("\u2265 v3") == ">= chord.v3", full("== v0") == "= chord.v0",
           full("") == "", len(REL.parse("  ", NAMES)) == 0], [1] * 7, 0)
    texts = ["> v0 - (v2 - v3)/2", "> 0.2/(1 - d1)", "<= -v2*-3", "> -(-v2)", "> v0 - -0.5", "> v0/v2/v3",
             "> v0/(v2*v3)", "> v0 - (v2 + v3)", "< (v0 + v2)*(v3 - 1.5e-07)", "> 2*v0*v2 + v3/4 - 1"]
    back = []
    for t in texts:
        r = REL.parse(t, NAMES, "chord", "chord.v1")
        back += [REL.parse(REL.text(r), NAMES, "chord", "chord.v1") == r,
                 REL.parse(REL.text(r, "chord"), NAMES, "chord", "chord.v1") == r, REL.text(r, "chord") == t]
    check("the text written is read back as the same relation (full names, and the curve's keys as typed)",
          back, [1] * len(back), 0)
    vals = {"span": 200.0, "chord.d1": 0.25, "chord.d2": 0.5, "chord.v0": 120.0, "chord.v1": 110.0, "chord.v2": 90.0,
            "chord.v3": 70.0, "pitch.v0": 2.0, "pitch.v1": 1.0}
    got = [REL.parse(t, NAMES, "chord")[0].value(vals) for t in ("> v0 - (v2 - v3)/2", "< 0.2/(1 - d1)",
                                                                  "= -v2*-3 + 2(v0 - span/4)", "> v0/v2/v3")]
    check("the expression's value: + - * /, brackets, the usual order", got,
          [110.0, 0.2 / 0.75, 270.0 + 2.0 * 70.0, 120.0 / 90.0 / 70.0], 1e-12)
    check("the variables an expression uses", list(REL.parse("> v0 - v2*pitch.v0 + v0", NAMES, "chord")[0].names)
          == ["chord.v0", "chord.v2", "pitch.v0"], 1, 0)
    bad = {"0.5*v2": "starts with one of", "> p4y": "'p4y' is not a variable of this curve (d1, d2, v0, v1, v2, v3)",
           "> pitch.v9": "'pitch.v9' is not a variable (pitch's: v0, v1)", "> a.b": "there is no curve a",
           "> v2 +": "missing at the end", "> (v2": "bracket is not closed", "> v2)": "a bracket too many",
           "> v2 3": "an operator is missing before '3'", "= v0, < v2": "takes no other relation",
           "> 0.5*v1": "cannot use the variable itself", "> v2 $ 3": "'$' is not part of an expression",
           "v2 < v3": "the relation is chord.v1's", ">": "no expression"}
    msgs = {t: message(REL.parse, t, NAMES, "chord", "chord.v1") for t in bad}
    check("what is no relation is refused, saying why (no operator, no such variable, the variable itself, ...)",
          [want in msgs[t] for t, want in bad.items()], [1] * len(bad), 0)
    check("without the names (the file layer): the grammar alone is checked",
          [REL.text(REL.parse("> 0.5v9 + foo.bar")) == "> 0.5*v9 + foo.bar", bool(message(REL.parse, "> 0.5 +"))],
          [1, 1], 0)

    print("\nthe grammar of a bound (xgeom_relations.parse_bound)")

    def bound(t, curve="chord", own="chord.v1"):
        return REL.parse_bound(t, NAMES, curve, own)

    check("a number is a number: as text or as a float; an expression of numbers alone is worked out",
          [bound("0.5"), bound(3), bound("2*0.7"), bound("1e-3"), bound(" -40 ")], [0.5, 3.0, 1.4, 0.001, -40.0], 1e-15)
    e = bound("0.9v0")
    check("an expression that names variables is a dependent bound, written as one word (no blanks): names in full, "
          "or the curve's by their keys",
          [isinstance(e, REL.Expression), e.text() == "0.9*chord.v0", e.text("chord") == "0.9*v0",
           e.names == ("chord.v0",), bound("v0 - (v2 - v3)/2").text() == "chord.v0-(chord.v2-chord.v3)/2",
           bound("pitch.v0 + 2 span").text("chord") == "pitch.v0+2*span"], [1] * 6, 0)
    back = [bound(bound(t).text()) == bound(t) and bound(bound(t).text("chord")) == bound(t)
            and bound(t).text("chord") == t
            for t in ("v0-(v2-v3)/2", "0.2/(1-d1)", "-v2*-3", "v0/(v2*v3)", "(v0+v2)*(v3-1.5e-07)", "2*v0*v2+v3/4-1")]
    check("the word written is read back as the same expression", back, [1] * len(back), 0)
    check("its value at a design", [bound("v0 - (v2 - v3)/2").value(vals), e.value(vals)], [110.0, 108.0], 1e-12)
    bad = {"": "a bound is a number or an expression", "> v0": "'>' is not part of an expression",
           "0.5*v1": "cannot use the variable itself", "p4y": "'p4y' is not a variable of this curve",
           "1/0": "divides by zero", "inf": "a finite number", "v0 +": "missing at the end"}
    msgs = {t: message(REL.parse_bound, t, NAMES, "chord", "chord.v1") for t in bad}
    check("what is no bound is refused, saying why (nothing, an operator, the variable itself, no such variable, ...)",
          [want in msgs[t] for t, want in bad.items()], [1] * len(bad), 0)


def test_laws(tmp):
    print("\nmin( ) and max( ), and the laws between curves (xgeom_relations.parse_law, sequence)")
    vals = {"span": 200.0, "chord.d1": 0.25, "chord.d2": 0.5, "chord.v0": 120.0, "chord.v1": 110.0, "chord.v2": 90.0,
            "chord.v3": 70.0, "pitch.v0": 2.0, "pitch.v1": 1.0}

    def bound(t):
        return REL.parse_bound(t, NAMES, "chord", "chord.v1")

    e = bound("max(0.5v0, v3)")
    r = REL.parse("> max(0.5v2, v3), < min(v0, 2*v3) + 1", NAMES, "chord", "chord.v1")
    check("min( ) and max( ) of two or more expressions, in a bound (one word) and in a relation; the commas "
          "inside the brackets do not split the relations",
          [e.text() == "max(0.5*chord.v0,chord.v3)", e.text("chord") == "max(0.5*v0,v3)", e.value(vals), len(r),
           REL.text(r, "chord") == "> max(0.5*v2, v3), < min(v0, 2*v3) + 1", r[0].value(vals), r[1].value(vals),
           bound("min(v0, v2, v3) - max(1, 2, 0.5*3)").value(vals),
           bound("-min(v0, v2)").text("chord") == "-min(v0,v2)", bound("2max(v0, v2)").value(vals), bound("max(1, 2)")],
          [1, 1, 70.0, 2, 1, 70.0, 121.0, 68.0, 1, 240.0, 2.0], 1e-12)
    back = [bound(bound(t).text()) == bound(t) and bound(t).text("chord") == t
            for t in ("max(0.5*v0,v3)", "min(v0,max(v2,v3))/2", "v0-min(v2,v3,d1)", "max(v0+v2,v3)*min(1,d1)")]
    back += [REL.parse(REL.text(r), NAMES, "chord", "chord.v1") == r]
    check("the text written is read back as the same expression", back, [1] * len(back), 0)
    bad = {"min(v0)": "takes two or more expressions",
           "sin(v0)": "'sin(' is not a function (there are min( ) and max( ))",
           "v0, v2": "a comma belongs inside min( ) or max( )", "max(v0, v2": "bracket is not closed",
           "max(v0,)": "a number or a variable is missing", "v0(v2)": "'v0(' is not a function"}
    msgs = {t: message(REL.parse_bound, t, NAMES, "chord", "chord.v1") for t in bad}
    check("what is no min or max is refused, saying why", [want in msgs[t] for t, want in bad.items()],
          [1] * len(bad), 0)

    names = ["LE", "P2x", "P2z", "P3z", "orig.LE", "orig.P2z", "orig.chord", "chord", "TE"]
    law = REL.parse_law("(min(orig.LE + orig.P2z*orig.chord, TE) - LE)/chord", names, "P2x")
    env = {"orig.LE": np.array([0.0, 1.0]), "orig.P2z": np.array([0.2, 0.2]), "orig.chord": np.array([180.0, 170.0]),
           "TE": np.array([30.0, 200.0]), "LE": np.array([-60.0, -10.0]), "chord": np.array([240.0, 190.0])}
    one = {k: float(v[1]) for k, v in env.items()}
    zero = REL.parse_law("P2z/(LE - LE)", names).value(env | {"P2z": np.array([1.0, 2.0])})
    check("a law: an expression of the names a component gives it, a number too; its value at arrays (one per "
          "position) an array, at numbers a number; dividing by zero at arrays gives no number there, no error",
          [law.text() == "(min(orig.LE+orig.P2z*orig.chord,TE)-LE)/chord", law.names == (
              "orig.LE", "orig.P2z", "orig.chord", "TE", "LE", "chord"), *law.value(env), law.value(one),
           REL.parse_law("2", names).value({}), REL.parse_law(0.5, names).value({}), np.isfinite(zero).any()],
          [1, 1, 90.0 / 240.0, 45.0 / 190.0, 45.0 / 190.0, 2.0, 0.5, 0], 1e-12)
    bad = {("", None): "a bounding law is a number or an expression", ("foo + 1", None): "'foo' is not a name a law "
           "can use here (LE, P2x", ("0.5*P2x", "P2x"): "cannot use the curve itself", ("P2z +", None): "missing at "
           "the end", ("> P2z", None): "'>' is not part of an expression"}
    msgs = {t: message(REL.parse_law, t[0], names, t[1]) for t in bad}
    check("what is no law is refused, saying why (nothing, a name that is not there, the curve itself, ...)",
          [want in msgs[t] for t, want in bad.items()], [1] * len(bad), 0)
    seq = REL.sequence({"a": ["b", "x"], "b": ["c"], "c": [], "d": ["a", "c"]}, "curves")
    loop = message(REL.sequence, {"a": ["b"], "b": ["c"], "c": ["a"]}, "curves")
    check("sequence: each name after the ones it uses (names that are no keys left out); a loop is named",
          [seq.index("c") < seq.index("b") < seq.index("a") < seq.index("d"), len(seq),
           "curves that depend on each other in a loop (a -> b -> c -> a)" in loop], [1, 4, 1], 0)
    if not ALL:
        return
    import xgeom_wing as XW
    ad = XW.WingAdapter()
    ad.set_bounds("chord.v1", "max(0.5v0, v3)", "v0")
    ad.set_relation("chord.v2", "< min(v1, 2*v3)")
    path = ad.save_file(os.path.join(tmp, "minmax.txt"))
    lines = {ln.split()[2]: ln.split(None, 3)[3].strip() for ln in open(path) if ln.startswith("# var")}
    ad2 = XW.WingAdapter(params=path)
    lo, hi = ad2.bounds("chord.v1")
    check("in a component file: a bound with max( ) is one word of its var line, a relation with min( ) after the "
          "flag; read back the same, the variable between its bounds",
          [lines["chord.v1"].split()[1] == "max(0.5*chord.v0,chord.v3)",
           lines["chord.v2"].endswith("< min(chord.v1, 2*chord.v3)"),
           ad2.bound_text("chord.v1") == ad.bound_text("chord.v1"), ad2.relation("chord.v2") == "< min(v1, 2*v3)",
           lo - max(0.5 * ad2.get("chord.v0"), ad2.get("chord.v3")), hi - ad2.get("chord.v0"),
           lo <= ad2.get("chord.v1") <= hi], [1, 1, 1, 1, 0, 0, 1], 1e-9)


def test_follow():
    print("\nlimits, the order of dependence and following (limits, order, apply, violations)")
    bounds = {n: (0.0, 200.0) for n in NAMES}
    rels = {"chord.v1": REL.parse("< 0.9*v0", NAMES, "chord"), "chord.v2": REL.parse("< v1, > 0.25*v0", NAMES, "chord"),
            "chord.v3": REL.parse("< v2", NAMES, "chord"), "pitch.v1": REL.parse("= v0", NAMES, "pitch")}
    vals = {"span": 200.0, "chord.d1": 0.25, "chord.d2": 0.5, "chord.v0": 120.0, "chord.v1": 110.0, "chord.v2": 90.0,
            "chord.v3": 70.0, "pitch.v0": 2.0, "pitch.v1": 1.0}
    check("a variable's limits: its bounds narrowed by its relations; an independent one: its bounds; '=': the value",
          [*REL.limits("chord.v1", vals, bounds, rels), *REL.limits("chord.v2", vals, bounds, rels),
           *REL.limits("chord.v0", vals, bounds, rels), *REL.limits("pitch.v1", vals, bounds, rels)],
          [0.0, 108.0, 30.0, 110.0, 0.0, 200.0, 2.0, 2.0], 1e-12)
    seq = REL.order(rels)
    check("the order of dependence: each dependent variable after the dependent ones it uses",
          [seq.index("chord.v1") < seq.index("chord.v2") < seq.index("chord.v3"), len(seq)], [1, 4], 0)
    new, moved = REL.apply(vals, bounds, rels)
    check("following: v1 above 0.9 v0 comes down to it, pitch.v1 takes pitch.v0's value; the others stay",
          [new["chord.v1"], new["chord.v2"], new["chord.v3"], new["pitch.v1"],
           sorted(moved) == ["chord.v1", "pitch.v1"]],
          [108.0, 90.0, 70.0, 2.0, 1], 1e-12)
    low = dict(new, **{"chord.v0": 60.0})
    new2, moved2 = REL.apply(low, bounds, rels)
    check("the root value lowered to 60: v1, v2 and v3 follow down the chain, in order",
          [new2["chord.v1"], new2["chord.v2"], new2["chord.v3"], moved2 == ["chord.v1", "chord.v2", "chord.v3"]],
          [54.0, 54.0, 54.0, 1], 1e-12)
    check("a design that keeps its relations is left as it is, outside its bounds too",
          [REL.apply(new2, {n: (0.0, 50.0) for n in NAMES}, rels)[1] == [], len(REL.violations(new2, rels))], [1, 0], 0)
    out = REL.violations(vals, rels)
    check("what a design breaks: the variable, a text and by how much",
          [[n for n, _, _ in out] == ["chord.v1", "pitch.v1"], out[0][2], out[1][2],
           "chord.v1 = 110 does not satisfy < 0.9*chord.v0 (= 108)" in out[0][1]], [1, 2.0, 1.0, 1], 1e-12)
    tight = {**bounds, "chord.v2": (100.0, 200.0)}
    new3, _ = REL.apply(low, tight, rels)
    check("relations that cannot hold inside the bounds: the variable stays on the bound, and that is reported",
          [new3["chord.v2"], REL.limits("chord.v2", new3, tight, rels)[1] < 100.0,
           [n for n, _, _ in REL.violations(new3, rels)] == ["chord.v2"]], [100.0, 1, 1], 1e-12)
    check("the value nearest to the one asked for: inside the limits; '=': the expression's",
          [REL.nearest("chord.v1", 150.0, vals, bounds, rels), REL.nearest("chord.v1", -5.0, vals, bounds, rels),
           REL.nearest("chord.v0", 500.0, vals, bounds, rels), REL.nearest("pitch.v1", 7.0, vals, bounds, rels)],
          [108.0, 0.0, 200.0, 2.0], 1e-12)
    loop = message(REL.order, dict(rels, **{"chord.v0": REL.parse("> v3", NAMES, "chord")}))
    check("variables that depend on each other in a loop are refused, the loop named",
          ["in a loop" in loop, all(f"chord.v{k} -> " in loop for k in range(4)), "pitch.v1" in loop], [1, 1, 0], 0)
    zero = {"chord.d2": REL.parse("> 0.2/(1 - d1)", NAMES, "chord")}
    at1 = dict(vals, **{"chord.d1": 1.0})
    check("a relation that divides by zero where the design is: left out of the limits, and reported",
          [REL.limits("chord.d2", at1, bounds, zero) == (0.0, 200.0),
           "divides by zero" in REL.violations(at1, zero)[0][1]],
          [1, 1], 0)

    print("\ndependent bounds: where they are, a variable's place between them, following (bounds_at, places, apply)")
    ex = {"chord.v1": (REL.parse_bound("0.5v0", NAMES, "chord"), REL.parse_bound("v0", NAMES, "chord")),
          "chord.v2": (None, REL.parse_bound("v1", NAMES, "chord"))}
    at = REL.places(vals, bounds, ex)
    check("a dependent bound is its expression's value where the design is; the other side, and an independent "
          "variable, keep their numbers",
          [*REL.bounds_at("chord.v1", vals, bounds, ex), *REL.bounds_at("chord.v2", vals, bounds, ex),
           *REL.bounds_at("chord.v3", vals, bounds, ex)], [60.0, 120.0, 0.0, 110.0, 0.0, 200.0], 1e-12)
    check("a variable's place: the fraction of the way from low to high",
          [at["chord.v1"][0], at["chord.v2"][0], *at["chord.v1"][1:]], [50.0 / 60.0, 90.0 / 110.0, 60.0, 120.0], 1e-12)
    low = dict(vals, **{"chord.v0": 100.0})
    new, moved = REL.apply(low, bounds, {}, exprs=ex, kept=at)
    v1 = 50.0 + 50.0 * 50.0 / 60.0
    check("the root value moved to 100: v1 keeps its place between 0.5 v0 and v0, then v2 its place under v1",
          [new["chord.v1"], new["chord.v2"], moved == ["chord.v1", "chord.v2"], new["chord.v3"]],
          [v1, 90.0 / 110.0 * v1, 1, 70.0], 1e-12)
    check("without the places (a file read): every variable stays", REL.apply(low, bounds, {}, exprs=ex)[1] == [], 1, 0)
    top = {"chord.v1": REL.parse("< 0.8*v0", NAMES, "chord")}
    check("limits and nearest take the dependent bounds; a relation on top narrows them",
          [*REL.limits("chord.v1", vals, bounds, {}, ex), *REL.limits("chord.v1", vals, bounds, top, ex),
           REL.nearest("chord.v1", 500.0, vals, bounds, {}, ex), REL.nearest("chord.v1", 500.0, vals, bounds, top, ex),
           REL.nearest("chord.v2", 500.0, vals, bounds, {}, ex)], [60.0, 120.0, 60.0, 96.0, 120.0, 96.0, 110.0], 1e-12)
    loop = message(REL.order, {}, dict(ex, **{"chord.v0": (REL.parse_bound("v2", NAMES, "chord"), None)}))
    check("the order of dependence takes the bounds too; a loop through bounds is refused",
          [REL.order(top, ex) == ["chord.v1", "chord.v2"], "in a loop" in loop], [1, 1], 0)
    cross = {"chord.v3": (REL.parse_bound("v0", NAMES, "chord"), REL.parse_bound("v2", NAMES, "chord")),
             "chord.d2": (REL.parse_bound("0.2/(1 - d1)", NAMES, "chord"), None)}
    out = REL.crossed(at1, bounds, cross)
    check("bounds that leave no room (low above high) or divide by zero where the design is are reported; one that "
          "divides by zero keeps its number",
          [[n for n, _, _ in out] == ["chord.v3", "chord.d2"], out[0][2],
           "leave no room (low chord.v0 = 120 above high chord.v2 = 90)" in out[0][1], "divides by zero" in out[1][1],
           *REL.bounds_at("chord.d2", at1, bounds, cross), len(REL.crossed(vals, bounds, ex))],
          [1, 30.0, 1, 1, 0.0, 200.0, 0], 1e-12)


# --------------------------------------------------------------------------
# The design spaces and the component files
# --------------------------------------------------------------------------

def test_spaces(tmp):
    print("\nthe design spaces with relations (SpaceRelations: wing, blade, hull)")
    ld = WM.load_component(os.path.join(HERE, "wing.txt"))
    sp, d0 = ld.space, ld.design
    n0 = len(sp.names)
    x0 = sp.to_vector(d0)
    check("a file with no relation: every variable independent, nothing broken",
          [len(sp.relations), len(sp.violations(x0)), np.abs(sp.project(x0) - x0).max()], [0, 0, 0], 0)
    sp.set_relations({"chord.v1": "<= 0.5*v2 + 2*v3", "chord.v3": "< chord.v0", "pitch.v1": "= pitch.v0",
                      "span": "> 1.5*chord.v0"})
    check("relations set by text; a variable defined with '=' leaves the design vector",
          [relations_of(sp) == {"span": "> 1.5*chord.v0", "pitch.v1": "= pitch.v0",
                                "chord.v1": "<= 0.5*chord.v2 + 2*chord.v3", "chord.v3": "< chord.v0"},
           len(sp.names), "pitch.v1" in sp.names, sp.relation("chord.v0") == ""], [1, n0 - 1, 0, 1], 0)
    x = sp.to_vector(d0)
    d = sp.to_design(x)
    check("to_design works out the '=' variable from the vector (the file had 1 deg, pitch.v0 is 2)",
          [sp.values(d0)["pitch.v1"], sp.values(d)["pitch.v1"], len(sp.violations(x)),
           [n for n, _, _ in sp.design_violations(d0)] == ["pitch.v1"]], [1.0, 2.0, 0, 1], 1e-12)
    x1 = x.copy()
    for n, v in (("chord.v0", 60.0), ("chord.v1", 149.0), ("chord.v2", 41.0)):
        x1[sp.names.index(n)] = v                    # v3 (70) above v0; v1 under its limit, 0.5 * 41 + 2 * 70
    broken = sp.violations(x1)
    xp = sp.project(x1)
    vp = dict(zip(sp.names, xp))
    check("a design vector that breaks a relation: violations names it; project moves the dependent variables, in "
          "order (v3 down to v0, then v1 under its new limit)",
          [[n for n, _, _ in broken] == ["chord.v3"], broken[0][2], len(sp.violations(xp)), vp["chord.v3"],
           vp["chord.v1"], vp["chord.v0"], vp["chord.v2"]], [1, 10.0, 0, 60.0, 140.5, 60.0, 41.0], 1e-9)
    rec = sp.to_dict(sp.to_design(xp))
    sp2, d2 = WM.WingSpace.from_dict(ld.params, rec)
    check("the space as plain data and back (to_dict, from_dict): the relations with the slots",
          [relations_of(sp2) == relations_of(sp), sp2.names == sp.names,
           sum("relation" in q for q in rec["slots"]), np.abs(sp2.to_vector(d2) - xp).max()], [1, 1, 4, 0.0], 1e-12)
    bad = {"a name that is no variable": ({"chord.v1": "< chord.v9"}, "the relation of chord.v1"),
           "a relation for a variable the space has not": ({"chord.v9": "< 1"}, "which this space does not have"),
           "a loop": ({"chord.v1": "< v2", "chord.v2": "< v1"}, "in a loop")}
    msgs = [message(WM.load_component(os.path.join(HERE, "wing.txt")).space.set_relations, t, "f.txt")
            for t, _ in bad.values()]
    check("set_relations refuses " + ", ".join(bad) + ", naming the file",
          [want in m and m.startswith("f.txt: ") for m, (_, want) in zip(msgs, bad.values())], [1, 1, 1], 0)

    shutil.copy(os.path.join(HERE, "airfoil_data_fixed.csv"), tmp)
    shutil.copy(os.path.join(HERE, "msc_blade.txt"), tmp)
    lb = BM.load_component(os.path.join(tmp, "msc_blade.txt"))
    db = BM.fit_design(lb.params, segments={"pitch": 2, "chord": 2})[0]      # pitch and chord in x_blade's form
    sb = BM.BladeSpace(lb.params, db, free=("pitch", "chord"))
    nb = len(sb.names)
    sb.set_relations({"chord.v4": "<= 1.4866 - v1", "pitch.u4": "<= 0.7287 + v7", "pitch.v7": "= 0.5*v1",
                      "hub_height": "> 0.5*radius"})
    xb = sb.to_vector(db)
    check("the blade's space: the MSc coupled limits as relations (chord P1 + P4, pitch P4 position - P7 value)",
          [len(sb.names), sb.values(sb.to_design(xb))["pitch.v7"], len(sb.violations(xb)),
           relations_of(sb)["chord.v4"] == "<= 1.4866 - chord.v1"],
          [nb - 1, 0.5 * sb.values(db)["pitch.v1"], 0, 1], 1e-12)
    p1 = BM.write_component(os.path.join(tmp, "blade_rel.txt"), lb.params, sb.to_design(xb), sb, lb.settings,
                            "blade_rel")
    l2 = BM.load_component(p1)
    p2 = BM.write_component(os.path.join(tmp, "blade_rel_again.txt"), l2.params, l2.design, l2.space, l2.settings,
                            "blade_rel")
    check("the blade's component file: the relations written on the var lines and read back; a rewrite the same file",
          [relations_of(l2.space) == relations_of(sb), len(l2.space.names), filecmp.cmp(p1, p2, shallow=False),
           len(l2.space.design_violations(l2.design))], [1, nb - 1, 1, 0], 0)
    free = blade_vary.read_free(p1)
    check("blade_vary.read_free (Initial_sampling) reads var lines that carry a relation",
          [free["pitch"].on, free["chord"].on, free["thickness"].on], [1, 1, 0], 0)

    lh = HM.load_component(os.path.join(HERE, "suboff_hull.txt"))
    sh = lh.space
    nh = len(sh.names)
    sh.set_relations({"tail": "= nose", "nose_r.b1": "<= b0", "middle": "> 2*nose"})
    xh = sh.to_vector(lh.design)
    dh = sh.to_design(xh)
    p1 = HM.write_component(os.path.join(tmp, "hull_rel.txt"), lh.params, dh, sh, "hull_rel", settings=lh.settings)
    l2 = HM.load_component(p1)
    p2 = HM.write_component(os.path.join(tmp, "hull_rel_again.txt"), l2.params, l2.design, l2.space, "hull_rel",
                            settings=l2.settings)
    check("the hull's space and file: the tail as long as the nose ('='), no design variable; written, read, rewritten",
          [len(sh.names), dh.tail - dh.nose, relations_of(l2.space) == relations_of(sh),
           filecmp.cmp(p1, p2, shallow=False), l2.design.tail - l2.design.nose], [nh - 1, 0.0, 1, 1, 0.0], 1e-12)


def test_files(tmp):
    print("\nthe var line of a component file (xgeom_files.py, every kind the same)")
    src = open(os.path.join(HERE, "wing.txt")).read().splitlines()
    plain = [s for s in src if s.startswith("# var")]

    def with_tail(tails):
        out = []
        for s in src:
            name = s.split()[2] if s.startswith("# var") else None
            out.append(s + ("  " + tails[name] if name in tails else ""))
        path = os.path.join(tmp, "hand.txt")
        with open(path, "w") as fh:
            fh.write("\n".join(out) + "\n")
        return path

    comp = XF.read_component(os.path.join(HERE, "wing.txt"))
    check("a var line without a relation reads as before: an independent variable",
          [all(v.relation == "" for v in comp.vars.values()), len(comp.vars)], [1, len(plain)], 0)
    hand = with_tail({"chord.v3": "< chord.v0", "chord.v1": "<= 0.5v2+2v3", "pitch.v1": "pitch.v1 > pitch.v3"})
    comp = XF.read_component(hand)
    check("a hand-written relation after the flag: read with the var line, as written",
          [comp.vars["chord.v3"].relation == "< chord.v0", comp.vars["chord.v1"].relation == "<= 0.5v2+2v3",
           comp.vars["chord.v0"].relation == "", comp.vars["chord.v1"].free], [1, 1, 1, 1], 0)
    ld = WM.load_component(hand)
    check("through wing_modify: the relations in the space, names in full; the file's design kept as it is",
          [relations_of(ld.space) == {"pitch.v1": "> pitch.v3", "chord.v1": "<= 0.5*chord.v2 + 2*chord.v3",
                                      "chord.v3": "< chord.v0"}, len(ld.space.design_violations(ld.design)),
           ld.space.values(ld.design)["chord.v1"]], [1, 0, 110.0], 1e-9)
    p1 = WM.write_component(os.path.join(tmp, "wing_rel.txt"), ld.params, ld.design, ld.space, ld.settings, "wing_rel",
                            ld.notes)
    l2 = WM.load_component(p1)
    p2 = WM.write_component(os.path.join(tmp, "wing_rel_again.txt"), l2.params, l2.design, l2.space, l2.settings,
                            "wing_rel", l2.notes)
    lines = open(p1).read().splitlines()
    rel_lines = [s for s in lines if s.startswith("# var") and len(s.split()) > 7]
    kept = [s for s in lines if s.startswith("# var") and len(s.split()) == 7]
    others = [s for s in plain if s.split()[2] not in ("chord.v1", "chord.v3", "pitch.v1")]
    check("written: the relation after the flag on its var line, the other var lines as they were; a rewrite the "
          "same file", [len(rel_lines), kept == others,
                        any(s.endswith("Y  <= 0.5*chord.v2 + 2*chord.v3") for s in rel_lines),
                        filecmp.cmp(p1, p2, shallow=False)], [3, 1, 1, 1], 0)
    title = [s for s in lines if s.startswith("# ----")]
    check("the block's title lines name the structure: curve <name> <form>, var <name> <value> <low> <high> <Y|N> "
          "[<relation>], and what makes a variable dependent (a bound that is an expression, a relation)",
          [any(s.endswith("var <name> <value> <low> <high> <Y free | N held> [<relation>]") for s in title),
           any("curve <name> <form>" in s for s in title),
           any("independent: numbers for low and high, no relation" in s for s in title),
           any("dependent:   low or high an expression of other variables" in s for s in title),
           any("or a relation after the flag: an operator" in s for s in title)], [1, 1, 1, 1, 1], 0)
    broken = WM.load_component(with_tail({"chord.v1": "< 0.5*chord.v0"}))
    out = broken.space.design_violations(broken.design)
    check("a file whose design breaks its relation still loads, the design as written; design_violations says what",
          [len(out), out[0][0] == "chord.v1", out[0][2], broken.space.values(broken.design)["chord.v1"]],
          [1, 1, 50.0, 110.0], 1e-9)
    msgs = [message(XF.read_component, with_tail({"chord.v1": "0.5*chord.v0"})),
            message(XF.read_component, with_tail({"chord.v1": "< 0.5*"})),
            message(WM.load_component, with_tail({"chord.v1": "< chord.v7"})),
            message(WM.load_component, with_tail({"chord.v1": "< v2", "chord.v2": "< v1"}))]
    n = 1 + next(i for i, s in enumerate(src) if s.split()[1:3] == ["var", "chord.v1"])
    check("refused with the file named: a relation with no operator or half an expression (the line too), a name "
          "that is no variable, a loop",
          [f"hand.txt, line {n}: the relation of chord.v1" in msgs[0], f"hand.txt, line {n}: " in msgs[1],
           "hand.txt: the relation of chord.v1" in msgs[2] and "'chord.v7' is not a variable" in msgs[2],
           "hand.txt: variables that depend on each other in a loop" in msgs[3]], [1, 1, 1, 1], 0)
    for name in ("wing.txt", "fin.txt", "bwb_glider.txt", "msc_blade.txt", "suboff_hull.txt"):
        ld = XF.load(os.path.join(HERE, name))
        if ld.space.relations or ld.space.dependent_bounds:
            _FAILED.append(name)
    check("the samples have independent variables only (relations and dependent bounds are the user's to set)",
          1, 1, 0)


# --------------------------------------------------------------------------
# The adapters
# --------------------------------------------------------------------------

def test_adapters(tmp):
    print("\nthe design tool's adapters (AdapterRelations: wing, blade, hull)")
    import xgeom_blade as XB
    import xgeom_hull as XH
    import xgeom_wing as XW
    ad = XW.WingAdapter(os.path.join(HERE, "wing.txt"))
    n0, m0 = ad.n_free()
    check("a variable starts independent: no relation, its limits its bounds",
          [ad.is_dependent("chord.v1"), ad.relation("chord.v1") == "", ad.limits("chord.v1") == ad.bounds("chord.v1")],
          [0, 1, 1], 0)
    shown = ad.set_relation("chord.v1", "<0.9v0")
    check("set_relation: the variable is dependent, its limits follow the others, the design follows at once",
          [shown == "< 0.9*v0", ad.relation("chord.v1", short=False) == "< 0.9*chord.v0", ad.is_dependent("chord.v1"),
           *ad.limits("chord.v1"), ad.get("chord.v1")], [1, 1, 1, 40.0, 108.0, 108.0], 1e-9)
    ad.set_relation("chord.v2", "< v1")
    ad.set_relation("chord.v3", "< v2, > 0.25*v0")
    used = ad.set("chord.v0", 90.0)
    check("a variable moved: the ones that depend on it follow, down the chain",
          [used, ad.get("chord.v1"), ad.get("chord.v2"), ad.get("chord.v3")], [90.0, 81.0, 81.0, 70.0], 1e-9)
    check("a dependent variable moves inside its limits only; the others then stay",
          [ad.set("chord.v1", 200.0), ad.set("chord.v1", 85.0), ad.get("chord.v0"), ad.get("chord.v2")],
          [81.0, 81.0, 90.0, 81.0], 1e-9)
    ad.set("chord.v0", 150.0)
    check("the limit lifted: the dependent variables stay where they are (a limit, not a tie)",
          [ad.get("chord.v1"), *ad.limits("chord.v1")], [81.0, 40.0, 135.0], 1e-9)
    msgs = [message(ad.set_relation, "chord.v0", "> v3"), message(ad.set_relation, "chord.v2", "> 0.5p4y"),
            message(ad.set_relation, "chord.v2", "0.5*v0")]
    check("a loop, a name that is no variable, a text with no operator: refused, the variable as it was",
          ["in a loop" in msgs[0], "'p4y' is not a variable" in msgs[1], "starts with one of" in msgs[2],
           ad.relation("chord.v2") == "< v1", ad.is_dependent("chord.v0")], [1, 1, 1, 1, 0], 0)
    ad.set_relation("pitch.v1", "= v0")
    ad.set("pitch.v0", 3.0)
    check("'=': the variable takes the expression's value, follows it, and is no design variable",
          [ad.get("pitch.v1"), ad.set("pitch.v1", 0.0), ad.is_defined("pitch.v1"), ad.is_free("pitch.v1"),
           ad.varies("pitch.v1"), ad.n_free() == (n0 - 1, m0), "pitch.v1" in ad.free_space().names,
           "4 dependent" in ad.summary()], [3.0, 3.0, 1, 1, 0, 1, 0, 1], 1e-12)
    ad.set_relation("thickness.v1", "> 2*v0")
    warn = ad.preview()["warnings"]
    check("a relation that cannot hold inside the bounds: the variable on its bound, a warning (the build still runs)",
          [ad.get("thickness.v1") == ad.bounds("thickness.v1")[1], len(warn), "thickness.v1" in warn[0],
           ad.limits("thickness.v1")[0] > ad.limits("thickness.v1")[1], len(ad.preview()["problems"])],
          [1, 1, 1, 1, 0], 0)
    ad.set_relation("thickness.v1", "")
    check("an empty relation: independent again, the warning gone",
          [ad.is_dependent("thickness.v1"), len(ad.preview()["warnings"]), "rel" in ad.meta["thickness.v1"]],
          [0, 0, 0], 0)
    p1 = ad.save_file(os.path.join(tmp, "wing_ad.txt"))
    before = {n: ad.relation(n, short=False) for n in ad.meta if ad.is_dependent(n)}
    ad2 = XW.WingAdapter(p1)
    shutil.copy(p1, os.path.join(tmp, "wing_ad_first.txt"))
    ad2.save_file(p1)
    check("Save set-up and Load: the relations, values and free flags back; saved again, the same file",
          [{n: ad2.relation(n, short=False) for n in ad2.meta if ad2.is_dependent(n)} == before, len(before),
           max(abs(ad2.get(n) - ad.get(n)) for n in ad.meta), ad2.n_free() == ad.n_free(),
           filecmp.cmp(p1, os.path.join(tmp, "wing_ad_first.txt"), shallow=False)], [1, 4, 0.0, 1, 1], 1e-9)
    ad2.set_relation("pitch.v2", "< chord.v3/100")
    ad2.set_relation("pitch.v3", "< chord.v1/100")
    ad2.set_order("chord", 2)                                 # 3 control points: chord.v3 is gone
    notes = ad2.relation_notes()
    check("a curve's new form: its own relations go with its variables, and a relation elsewhere that names a "
          "variable now gone is dropped, with a note; the others stay",
          [sorted(n for n in ad2.meta if ad2.is_dependent(n)) == ["pitch.v1", "pitch.v3"], len(notes),
           notes[0].startswith("the relation of pitch.v2 is dropped") if notes else 0, len(ad2.relation_notes())],
          [1, 1, 1, 0], 0)
    hand = os.path.join(tmp, "wing_broken.txt")
    with open(os.path.join(HERE, "wing.txt")) as fh, open(hand, "w") as out:
        out.write("".join(s.rstrip("\n") + ("  < 0.5*chord.v0\n" if s.split()[1:3] == ["var", "chord.v1"] else "\n")
                          for s in fh))
    ad3 = XW.WingAdapter(hand)
    check("a hand-written relation that the file's design breaks: the tool applies it on loading (v1 to 0.5 v0) "
          "and says so", [ad3.get("chord.v1"), len(ad3.preview()["warnings"]),
                          ad3.relation_notes() == ["moved to keep their relations: chord.v1"]], [60.0, 0, 1], 1e-9)

    shutil.copy(os.path.join(HERE, "msc_blade.txt"), tmp)
    shutil.copy(os.path.join(HERE, "airfoil_data_fixed.csv"), tmp)
    ab = XB.BladeAdapter(os.path.join(tmp, "msc_blade.txt"))
    ab.set_segments("pitch", 2)
    ab.set_segments("chord", 2)
    ab.set_relation("pitch.v7", "< 0.5v1")
    ab.set_relation("chord.v4", "<= 1.4866 - v1")
    ab.set_relation("hub_height", "> 0.5*radius")
    v1 = ab.get("pitch.v1")
    check("the blade: a control value under half the root's, and a global one (the hub height) on another",
          [ab.get("pitch.v7"), ab.limits("hub_height")[0], ab.relation("hub_height") == "> 0.5*radius",
           ab.set("pitch.v1", 0.8), ab.get("pitch.v7")], [0.5 * v1, 0.35, 1, 0.8, 0.4], 1e-9)
    p1 = ab.save_file(os.path.join(tmp, "blade_ad.txt"))
    ab2 = XB.BladeAdapter(p1)
    check("the blade's set-up saved and loaded: the relations back",
          [{n: ab2.relation(n) for n in ab2.meta if ab2.is_dependent(n)}
           == {"hub_height": "> 0.5*radius", "pitch.v7": "< 0.5*v1", "chord.v4": "<= 1.4866 - v1"},
           abs(ab2.get("pitch.v7") - 0.4)], [1, 0.0], 1e-9)
    ab2.set_segments("pitch", 1)
    check("the blade's pitch in another form: its relation goes with its variables, the others stay",
          sorted(n for n in ab2.meta if ab2.is_dependent(n)) == ["chord.v4", "hub_height"], 1, 0)

    ah = XH.HullAdapter(os.path.join(HERE, "suboff_hull.txt"))
    nh = ah.n_free()[0]
    ah.set_relation("tail", "= nose")
    ah.set_relation("nose_r.b1", "<= b0")
    check("the hull: the tail as long as the nose; a CST coefficient at most the one before",
          [ah.get("tail") - ah.get("nose"), ah.set("nose", 1.2), ah.get("tail"), ah.n_free()[0],
           ah.limits("nose_r.b1")[1] - ah.get("nose_r.b0"), len(ah.preview()["warnings"])],
          [0.0, 1.2, 1.2, nh - 1, 0.0, 0], 1e-12)
    p1 = ah.save_file(os.path.join(tmp, "hull_ad.txt"))
    ah2 = XH.HullAdapter(p1)
    check("the hull's set-up saved and loaded: the relations back, the lengths as set",
          [{n: ah2.relation(n) for n in ah2.meta if ah2.is_dependent(n)} == {"tail": "= nose", "nose_r.b1": "<= b0"},
           ah2.design.tail, ah2.design.nose], [1, 1.2, 1.2], 1e-9)


# --------------------------------------------------------------------------
# Dependent bounds: the design spaces, the component files, the adapters
# --------------------------------------------------------------------------

def test_bound_spaces(tmp):
    print("\ndependent bounds in the design spaces: the design vector's fractions (SpaceRelations.factors)")
    path = os.path.join(HERE, "wing.txt")
    ld = WM.load_component(path)
    sp, d0 = ld.space, ld.design
    names0, x_old, vals0 = list(sp.names), sp.to_vector(d0), sp.values(d0)
    sp.set_relations(bounds={"chord.v1": ("0.5*v0", "chord.v0"), "chord.v2": (None, "v1")}, values=vals0)
    i0, i1, i2 = (sp.names.index(f"chord.v{k}") for k in range(3))
    x0 = sp.to_vector(d0)
    check("a free variable with a dependent bound stays in the design vector, as the fraction of the way between "
          "its bounds: its row of the vector's bounds is 0 .. 1",
          [sp.names == names0, sp.factors == ["chord.v1", "chord.v2"], *sp.bounds[i1], *sp.bounds[i2], x0[i1], x0[i2],
           np.abs(np.delete(x0 - x_old, [i1, i2])).max()],
          [1, 1, 0.0, 1.0, 0.0, 1.0, 50.0 / 60.0, (vals0["chord.v2"] - 40.0) / 70.0, 0.0], 1e-12)
    check("the bounds of every variable: all_bounds where the file's design puts them, bounds_of at any other design",
          [*sp.all_bounds[sp.all_names.index("chord.v1")],
           *sp.bounds_of(dict(vals0, **{"chord.v0": 100.0}))["chord.v1"],
           sp.bound_text("chord.v1") == ("0.5*chord.v0", "chord.v0"), sp.bound_text("chord.v2") == (None, "chord.v1"),
           sp.bound_text("chord.v0") == (None, None)], [60.0, 120.0, 50.0, 100.0, 1, 1, 1], 1e-12)
    x = x0.copy()
    x[i0], x[i1], x[i2] = 100.0, 0.25, 0.5
    v = sp.values(sp.to_design(x))
    check("to_design places the fractions: v1 a quarter of the way from 0.5 v0 to v0, v2 half way from 40 to v1; "
          "to_vector gives them back",
          [v["chord.v0"], v["chord.v1"], v["chord.v2"], np.abs(sp.to_vector(sp.to_design(x)) - x).max()],
          [100.0, 62.5, 51.25, 0.0], 1e-12)
    rng = np.random.default_rng(3)
    worst, broken, crossed = 0.0, 0, []
    for k in range(400):
        xr = x0.copy()
        for i in (i0, i1, i2):
            xr[i] = sp.bounds[i, 0] + rng.random() * (sp.bounds[i, 1] - sp.bounds[i, 0])
        if k < 200:
            xr[i0] = 80.0 + 70.0 * rng.random()          # v0 from 80: v1 (from 0.5 v0) stays above v2's low, 40
            vr = sp.values(sp.to_design(xr))
            worst = max(worst, 0.5 * vr["chord.v0"] - vr["chord.v1"], vr["chord.v1"] - vr["chord.v0"],
                        40.0 - vr["chord.v2"], vr["chord.v2"] - vr["chord.v1"])
            broken += len(sp.violations(xr))
        else:
            crossed += [(n, sp.values(sp.to_design(xr))["chord.v1"]) for n, _, _ in sp.violations(xr)]
    check("200 random vectors inside the vector's bounds (the root chord from 80): every design inside its "
          "dependent bounds, nothing to check or project",
          [worst <= 1e-12, broken], [1, 0], 0)
    check("200 over the whole of v0's range: where v1 comes under v2's low (40) v2 has no room, which violations "
          "says (bounds can cross when a number faces an expression; a vector's fractions are still placed)",
          [len(crossed) > 0, all(n == "chord.v2" and v1 < 40.0 for n, v1 in crossed)], [1, 1], 0)
    held = WM.WingSpace(ld.params, d0, fixed={"chord.v1": 110.0}).set_relations(
        bounds={"chord.v1": ("0.5*v0", "v0")}, values=vals0)
    xh = held.to_vector(d0)
    xh[held.names.index("chord.v0")] = 100.0
    check("a held variable with a dependent bound is no design variable, and keeps its place between its bounds",
          ["chord.v1" in held.names, len(held.factors), held.values(held.to_design(xh))["chord.v1"]],
          [0, 0, 50.0 + 50.0 * 50.0 / 60.0], 1e-12)
    rec = sp.to_dict(sp.to_design(x))
    slot = {q["name"]: q for q in rec["slots"]}
    sp2, d2 = WM.WingSpace.from_dict(ld.params, rec)
    check('the space as plain data and back (to_dict, from_dict): "lo_expr", "hi_expr" in the slot beside the '
          'numbers, "fractions" in the vector',
          [slot["chord.v1"]["lo_expr"] == "0.5*chord.v0", slot["chord.v1"]["hi_expr"] == "chord.v0",
           slot["chord.v1"]["lo"], slot["chord.v1"]["hi"], "lo_expr" in slot["chord.v2"], "hi_expr" in slot["chord.v0"],
           rec["vector"]["fractions"] == ["chord.v1", "chord.v2"], sp2.factors == sp.factors,
           np.abs(sp2.to_vector(d2) - x).max()], [1, 1, 50.0, 100.0, 0, 0, 1, 1, 0.0], 1e-12)
    sp.set_relations({"chord.v0": "< 0.5*span"}, bounds={"chord.v1": ("0.5*v0", "chord.v0")}, values=vals0)
    x = sp.to_vector(d0)
    x[sp.names.index("span")] = 180.0                    # v0 (120) above 0.5 span
    xp = sp.project(x)
    check("with a relation too: violations names what a vector breaks; project moves v0 onto it and v1 keeps its "
          "fraction. (set_relations again: the bound it no longer names is its number again)",
          [[n for n, _, _ in sp.violations(x)] == ["chord.v0"], xp[i0], xp[i1] - x[i1],
           sp.values(sp.to_design(xp))["chord.v1"], len(sp.violations(xp)),
           *sp.all_bounds[sp.all_names.index("chord.v2")], sp.factors == ["chord.v1"]],
          [1, 90.0, 0.0, 45.0 + 45.0 * 50.0 / 60.0, 0, 40.0, 150.0, 1], 1e-9)
    sp.set_relations(bounds={"chord.v3": ("v0", "v2")}, values=vals0)
    xc = sp.to_vector(d0)
    out = sp.violations(xc)
    xc[i0] = 60.0
    check("bounds that cross where a vector puts them (low above high) are named by violations",
          [[n for n, _, _ in out] == ["chord.v3"], out[0][2] if out else 0, len(sp.violations(xc))],
          [1, 120.0 - vals0["chord.v2"], 0], 1e-9)

    def refused(*args):
        return message(WM.load_component(path).space.set_relations, *args)

    msgs = [refused(None, "f.txt", {"chord.v1": ("0.5*v0", None)}),
            refused(None, "f.txt", {"chord.v1": ("chord.v9", None)}, vals0),
            refused(None, "f.txt", {"chord.v1": ("v2", None), "chord.v2": (None, "v1")}, vals0),
            refused({"chord.v2": "< v1"}, "f.txt", {"chord.v1": ("v2", None)}, vals0),
            refused(None, "f.txt", {"chord.v1": (None, "2*v1")}, vals0),
            refused(None, "f.txt", {"chord.v9": ("v1", None)}, vals0)]
    wants = ["dependent bounds need the design's values", "the low bound of chord.v1: 'chord.v9': 'chord.v9' is not a "
             "variable", "in a loop", "in a loop", "the high bound of chord.v1: '2*v1': a variable's bound cannot use "
             "the variable itself", "which this space does not have"]
    check("set_relations refuses, naming the file: dependent bounds without the design's values, a name that is no "
          "variable, a loop through bounds (and through a bound and a relation), the variable itself, a variable "
          "the space has not", [w in m and m.startswith("f.txt: ") for m, w in zip(msgs, wants)], [1] * 6, 0)

    shutil.copy(os.path.join(HERE, "airfoil_data_fixed.csv"), tmp)
    shutil.copy(os.path.join(HERE, "msc_blade.txt"), tmp)
    lb = BM.load_component(os.path.join(tmp, "msc_blade.txt"))
    db = BM.fit_design(lb.params, segments={"pitch": 2, "chord": 2})[0]      # pitch and chord in x_blade's form
    sb = BM.BladeSpace(lb.params, db, bounds={"pitch.v1": (0.60, 1.25)}, free=("pitch", "chord"))
    vb = sb.values(db)
    sb.set_relations(bounds={"pitch.v4": ("pitch.v1", "1.4")}, values=vb)
    k1, k4 = sb.names.index("pitch.v1"), sb.names.index("pitch.v4")
    xb = sb.to_vector(db)
    errs = []
    for _ in range(100):
        xr = xb.copy()
        xr[k1], xr[k4] = 0.60 + 0.65 * rng.random(), rng.random()                # p1y, y4_frac
        errs.append(sb.values(sb.to_design(xr))["pitch.v4"] - (xr[k1] + (1.4 - xr[k1]) * xr[k4]))
    check("the MSc propeller optimisation's P4 pitch (para_control_bez_updated.py): pitch.v4 with the bounds "
          "pitch.v1 .. 1.4 is in the vector as the MSc's y4_frac, 0 .. 1, and to_design gives p1y + (max_pitch - "
          "p1y) y4_frac",
          [sb.factors == ["pitch.v4"], *sb.bounds[k4], *sb.bounds[k1], xb[k4], np.abs(errs).max()],
          [1, 0.0, 1.0, 0.60, 1.25, (vb["pitch.v4"] - vb["pitch.v1"]) / (1.4 - vb["pitch.v1"]), 0.0], 1e-12)
    p1 = BM.write_component(os.path.join(tmp, "blade_bnd.txt"), lb.params, db, sb, lb.settings, "blade_bnd")
    l2 = BM.load_component(p1)
    p2 = BM.write_component(os.path.join(tmp, "blade_bnd_again.txt"), l2.params, l2.design, l2.space, l2.settings,
                            "blade_bnd")
    with open(p1) as fh:
        line = next(q for q in fh if q.split()[1:3] == ["var", "pitch.v4"])
    check("the blade's component file: the bound's expression in the low column of its var line; read back and "
          "rewritten, the same file",
          [line.split()[4:7] == ["pitch.v1", "1.4", "Y"], l2.space.factors == ["pitch.v4"],
           l2.space.bound_text("pitch.v4") == ("pitch.v1", None), filecmp.cmp(p1, p2, shallow=False),
           np.abs(l2.space.to_vector(l2.design) - xb).max(), len(BM.outside_bounds(l2.space, l2.design))],
          [1, 1, 1, 1, 0.0, 0], 1e-9)

    lh = HM.load_component(os.path.join(HERE, "suboff_hull.txt"))
    sh = lh.space
    sh.set_relations(bounds={"nose_r.b1": ("0.5*b0", "b0"), "tail": ("0.8*nose", "1.2*nose")},
                     values=sh.values(lh.design))
    xh = sh.to_vector(lh.design)
    place = (lh.design.tail - 0.8 * lh.design.nose) / (0.4 * lh.design.nose)       # the tail's, in the file's design
    xh[sh.names.index("nose")] = 1.2
    dh = sh.to_design(xh)
    p1 = HM.write_component(os.path.join(tmp, "hull_bnd.txt"), lh.params, dh, sh, "hull_bnd", settings=lh.settings)
    l2 = HM.load_component(p1)
    p2 = HM.write_component(os.path.join(tmp, "hull_bnd_again.txt"), l2.params, l2.design, l2.space, "hull_bnd",
                            settings=l2.settings)
    check("the hull's space and file: the tail between 0.8 and 1.2 nose lengths keeps its place when the nose "
          "changes, a CST coefficient between half the one before and it; written, read, rewritten",
          [sh.factors == ["tail", "nose_r.b1"], xh[sh.names.index("tail")], dh.tail, l2.space.factors == sh.factors,
           filecmp.cmp(p1, p2, shallow=False), l2.design.tail],
          [1, place, 0.96 + 0.48 * place, 1, 1, 0.96 + 0.48 * place], 1e-9)


def test_bound_files(tmp):
    print("\ndependent bounds on the var line of a component file: an expression in the low or high column")
    with open(os.path.join(HERE, "wing.txt")) as fh:
        src = fh.read().splitlines()
    ours = ("chord.v1", "chord.v2", "pitch.v1")

    def hand(words, tails=None, drop=None):
        """wing.txt with the low and high words of the var lines named replaced (None: as it is), a relation
        added (tails), the var lines of one curve left out (drop)."""
        out = []
        for q in src:
            w = q.split()
            if w[1:2] == ["var"]:
                if drop and w[2].startswith(drop + "."):
                    continue
                if w[2] in words:
                    w[4], w[5] = (new or old for new, old in zip(words[w[2]], w[4:6]))
                    q = "  ".join(w)
                q += "  " + (tails or {}).get(w[2], "")
            out.append(q.rstrip())
        target = os.path.join(tmp, "hand.txt")
        with open(target, "w") as fh:
            fh.write("\n".join(out) + "\n")
        return target

    def var_lines(file):
        with open(file) as fh:
            return {q.split()[2]: q.rstrip("\n") for q in fh if q.split()[1:2] == ["var"]}

    written = hand({"chord.v1": ("0.5*chord.v0", "chord.v0"), "chord.v2": (None, "v1"), "pitch.v1": ("pitch.v3", None)})
    comp = XF.read_component(written)
    q1, q2 = comp.vars["chord.v1"], comp.vars["chord.v2"]
    check("an expression in the low or high column: read with the var line as written, its number worked out at "
          "the file's design; the other var lines as before",
          [q1.lo_expr == "0.5*chord.v0", q1.hi_expr == "chord.v0", q1.lo, q1.hi, q2.lo_expr == "", q2.hi_expr == "v1",
           q2.lo, q2.hi, comp.vars["chord.v0"].hi_expr == "", q1.free],
          [1, 1, 60.0, 120.0, 1, 1, 40.0, 110.0, 1, 1], 1e-9)
    ld = WM.load_component(written)
    check("through wing_modify: the dependent bounds in the space, names in full; their variables the vector's "
          "fractions; the file's design kept as it is",
          [ld.space.bound_text("chord.v2") == (None, "chord.v1"), ld.space.bound_text("pitch.v1") == ("pitch.v3", None),
           ld.space.factors == list(ours), len(ld.space.relations), len(ld.space.design_violations(ld.design)),
           ld.space.values(ld.design)["chord.v1"]], [1, 1, 1, 0, 0, 110.0], 1e-9)
    p1 = WM.write_component(os.path.join(tmp, "wing_bnd.txt"), ld.params, ld.design, ld.space, ld.settings, "wing_bnd",
                            ld.notes)
    l2 = WM.load_component(p1)
    p2 = WM.write_component(os.path.join(tmp, "wing_bnd_again.txt"), l2.params, l2.design, l2.space, l2.settings,
                            "wing_bnd", l2.notes)
    new, old = var_lines(p1), var_lines(os.path.join(HERE, "wing.txt"))
    check("written: the expression (names in full, no blanks) in its column of the var line, the other var lines "
          "as they were; a rewrite the same file",
          [new["chord.v1"].split()[4:] == ["0.5*chord.v0", "chord.v0", "Y"], new["chord.v2"].split()[5] == "chord.v1",
           new["pitch.v1"].split()[4] == "pitch.v3", {n: q for n, q in new.items() if n not in ours}
           == {n: q for n, q in old.items() if n not in ours}, filecmp.cmp(p1, p2, shallow=False)], [1] * 5, 0)
    both = WM.load_component(hand({"chord.v1": ("0.5*chord.v0", None)}, tails={"chord.v1": "< 0.95*chord.v0"}))
    check("a dependent bound and a relation on one var line: both read (the variable a fraction in the vector, the "
          "relation an optimiser's to check)",
          [both.space.bound_text("chord.v1") == ("0.5*chord.v0", None),
           both.space.relation("chord.v1") == "< 0.95*chord.v0",
           both.space.factors == ["chord.v1"], len(both.space.violations(both.space.to_vector(both.design)))],
          [1, 1, 1, 0], 0)
    part = hand({"pitch.v1": ("0.01*chord.v0", None)}, drop="chord")
    comp = XF.read_component(part)
    ld = WM.load_component(part)
    check("a bound that names a variable with no var line (its curve fitted from the table): the file layer leaves "
          "the number open, the component's space works it out",
          [np.isnan(comp.vars["pitch.v1"].lo), "chord.v0" in comp.vars, ld.space.bound_text("pitch.v1")
           == ("0.01*chord.v0", None), ld.space.all_bounds[ld.space.all_names.index("pitch.v1"), 0]
           - 0.01 * ld.space.values(ld.design)["chord.v0"]], [1, 0, 1, 0.0], 1e-12)
    crossed = WM.load_component(hand({"chord.v3": ("chord.v0", "chord.v2")}))
    out = crossed.space.design_violations(crossed.design)
    check("a file whose dependent bounds cross at its design (low above high) still loads, the design as written; "
          "design_violations says so",
          [len(out), out[0][0] == "chord.v3" if out else 0, "leave no room" in out[0][1] if out else 0,
           crossed.space.values(crossed.design)["chord.v3"]], [1, 1, 1, 70.0], 1e-9)
    n = 1 + next(i for i, q in enumerate(src) if q.split()[1:3] == ["var", "chord.v1"])
    msgs = [message(XF.read_component, hand({"chord.v1": ("0.5*", None)})),
            message(XF.read_component, hand({"chord.v1": ("0.5 * chord.v0", None)})),
            message(WM.load_component, hand({"chord.v1": ("chord.v9", None)})),
            message(WM.load_component, hand({"chord.v1": ("v2", None), "chord.v2": (None, "v1")})),
            message(WM.load_component, hand({"chord.v1": (None, "2*chord.v1")})),
            message(WM.load_component, hand({"chord.d2": ("0.2/(chord.v0-120)", None)}))]
    check("refused with the file named: half an expression (the line too), an expression with blanks (the line "
          "too), a name that is no variable, a loop, the variable itself, a bound that divides by zero at the "
          "file's design",
          [f"hand.txt, line {n}: the low bound of chord.v1" in msgs[0],
           f"hand.txt, line {n}: after the bounds comes Y or N" in msgs[1] and "no blanks" in msgs[1],
           "hand.txt: the low bound of chord.v1: 'chord.v9': 'chord.v9' is not a variable" in msgs[2],
           "hand.txt: variables that depend on each other in a loop" in msgs[3],
           "hand.txt: the high bound of chord.v1" in msgs[4] and "cannot use the variable itself" in msgs[4],
           "hand.txt: the low bound of chord.d2, 0.2/(chord.v0-120), divides by zero at the design" in msgs[5]],
          [1] * 6, 0)


def test_bound_adapters(tmp):
    print("\ndependent bounds in the design tool's adapters (set_bounds with an expression: wing, blade, hull)")
    import xgeom_blade as XB
    import xgeom_hull as XH
    import xgeom_wing as XW
    ad = XW.WingAdapter(os.path.join(HERE, "wing.txt"))
    n0 = ad.n_free()
    got = ad.set_bounds("chord.v1", "0.5v0", "v0")
    check("set_bounds with expressions: the bounds are worked out where the design is; the variable is dependent "
          "and still a design variable",
          [*got, *ad.bounds("chord.v1"), ad.bound_text("chord.v1") == ("0.5*v0", "v0"),
           ad.bound_text("chord.v1", short=False) == ("0.5*chord.v0", "chord.v0"), ad.is_dependent("chord.v1"),
           ad.is_defined("chord.v1"), ad.varies("chord.v1"), ad.n_free() == n0, *ad.limits("chord.v1"),
           ad.get("chord.v1"), ad.bound_text("chord.v0") == (None, None)],
          [60.0, 120.0, 60.0, 120.0, 1, 1, 1, 0, 1, 1, 60.0, 120.0, 110.0, 1], 1e-9)
    used = ad.set("chord.v0", 100.0)
    m = ad.meta["chord.v1"]
    check("a variable its bounds use is moved: it keeps its place between them (5/6 of the way), and the numbers "
          "kept for its bounds follow",
          [used, ad.get("chord.v1"), *ad.bounds("chord.v1"), m["lo"], m["hi"], m["lo_expr"] == "0.5*chord.v0"],
          [100.0, 50.0 + 50.0 * 50.0 / 60.0, 50.0, 100.0, 50.0, 100.0, 1], 1e-9)
    check("it moves inside its bounds only, and the others then stay",
          [ad.set("chord.v1", 200.0), ad.set("chord.v1", 10.0), ad.set("chord.v1", 75.0), ad.get("chord.v0")],
          [100.0, 50.0, 75.0, 100.0], 1e-9)
    ad.set_bounds("chord.v2", 40, "v1")                      # v2 (93.3) comes down to its new high, v1 = 75
    ad.set("chord.v0", 120.0)
    check("a chain, v2 under v1 under v0, one bound an expression and the other a number: both follow the root",
          [ad.bound_text("chord.v2") == (None, "v1"), ad.get("chord.v1"), ad.get("chord.v2"), *ad.bounds("chord.v2")],
          [1, 90.0, 90.0, 40.0, 90.0], 1e-9)

    def state():
        return [ad.bounds(n) + ad.bound_text(n) for n in ("chord.v0", "chord.v3", "chord.d1")]

    before = state()
    msgs = [message(ad.set_bounds, "chord.v0", "v2", 150), message(ad.set_bounds, "chord.v3", "0.5p4y", 150),
            message(ad.set_bounds, "chord.v3", 40, "v3 + 1"), message(ad.set_bounds, "chord.v3", "2*v0", 150),
            message(ad.set_bounds, "chord.d1", "0.001*v0", "0.01*v0"), message(ad.set_bounds, "chord.v3", "> v0", 150),
            message(ad.set_bounds, "chord.v3", "", 150)]
    check("refused, the variable as it was: a loop, a name that is no variable, the variable itself, low above "
          "high where the design is, the adapter's own limits (a fraction d in [0, 1]), an operator, nothing",
          ["in a loop" in msgs[0], "the low bound: '0.5p4y': 'p4y' is not a variable" in msgs[1],
           "the high bound" in msgs[2] and "cannot use the variable itself" in msgs[2],
           "below the upper one" in msgs[3],
           "lies in [0, 1]" in msgs[4], "'>' is not part of an expression" in msgs[5],
           "a bound is a number or an expression" in msgs[6], state() == before], [1] * 8, 0)
    sp = ad.free_space()
    xs = sp.to_vector(ad.design)
    check("the tool's summary counts the dependent variables; free_space (the files, an optimiser) has them as "
          "fractions",
          ["2 dependent" in ad.summary(), sp.factors == ["chord.v1", "chord.v2"], xs[sp.names.index("chord.v1")],
           xs[sp.names.index("chord.v2")]], [1, 1, 0.5, 1.0], 1e-9)
    ad.set_relation("chord.v1", "< 0.7*v0")
    check("a relation on top of a dependent bound narrows the limits (the bounds stay); the variable moves onto "
          "it, and the one under it keeps its place",
          [ad.get("chord.v1"), *ad.limits("chord.v1"), *ad.bounds("chord.v1"), ad.get("chord.v2")],
          [84.0, 60.0, 84.0, 60.0, 120.0, 84.0], 1e-9)
    ad.set_bounds("chord.v3", "0.5*v0", "v2")                # v3 = 70 in 60 .. 84
    ad.set("chord.v2", 45.0)
    warn = ad.preview()["warnings"]
    check("bounds that leave no room where the design is (low above high): a warning, the build still runs; they "
          "cannot be made numbers there (set_independent refused, the variable as it was)",
          [len(warn), "chord.v3: its bounds leave no room" in warn[0] if warn else 0,
           ad.limits("chord.v3")[0] > ad.limits("chord.v3")[1], len(ad.preview()["problems"]),
           "chord.v3 cannot be made independent with its bounds where they are now, 60 .. 45"
           in message(ad.set_independent, "chord.v3"), ad.bound_text("chord.v3") == ("0.5*v0", "v2")],
          [1, 1, 1, 0, 1, 1], 0)
    ad.set("chord.v2", 84.0)
    check("room again: the warning gone, the variable back at its place", [len(ad.preview()["warnings"]),
                                                                          ad.get("chord.v3")], [0, 70.0], 1e-9)
    ad.set_independent("chord.v1")
    check("set_independent: the relation dropped and the dependent bounds numbers, where they are; the variable "
          "that has it as a bound keeps that",
          [ad.is_dependent("chord.v1"), *ad.bounds("chord.v1"), ad.bound_text("chord.v1") == (None, None),
           "lo_expr" in ad.meta["chord.v1"], "rel" in ad.meta["chord.v1"], ad.bound_text("chord.v2") == (None, "v1"),
           ad.get("chord.v1")], [0, 60.0, 120.0, 1, 0, 0, 1, 84.0], 1e-9)
    p1 = ad.save_file(os.path.join(tmp, "wing_bnd_ad.txt"))
    ad2 = XW.WingAdapter(p1)
    shutil.copy(p1, os.path.join(tmp, "wing_bnd_first.txt"))
    ad2.save_file(p1)
    check("Save set-up and Load: the dependent bounds, the values and the bounds' numbers back; saved again, the "
          "same file",
          [{n: ad2.bound_text(n) for n in ad2.meta if ad2.is_dependent(n)}
           == {"chord.v2": (None, "v1"), "chord.v3": ("0.5*v0", "v2")},
           max(abs(ad2.get(n) - ad.get(n)) for n in ad.meta),
           max(abs(a - b) for n in ad.meta for a, b in zip(ad.bounds(n), ad2.bounds(n))),
           filecmp.cmp(p1, os.path.join(tmp, "wing_bnd_first.txt"), shallow=False)], [1, 0.0, 0.0, 1], 1e-9)
    ad2.set_bounds("pitch.v2", "chord.v3/100", 5)            # the value (0) comes up to its new low, 0.7
    ad2.relation_notes()
    ad2.set_order("chord", 2)                                 # 3 control points: chord.v3 is gone
    notes = ad2.relation_notes()
    check("a curve's new form: its own dependent bounds go with its variables; a bound elsewhere that names a "
          "variable now gone is a number again, where it was, with a note",
          [sum(ad2.is_dependent(n) for n in ad2.meta), len(notes),
           notes[0].startswith("the low bound of pitch.v2 is a number again") if notes else 0, *ad2.bounds("pitch.v2"),
           ad2.bound_text("pitch.v2") == (None, None), ad2.get("pitch.v2")], [0, 1, 1, 0.7, 5.0, 1, 0.7], 1e-9)

    shutil.copy(os.path.join(HERE, "msc_blade.txt"), tmp)
    shutil.copy(os.path.join(HERE, "airfoil_data_fixed.csv"), tmp)
    ab = XB.BladeAdapter(os.path.join(tmp, "msc_blade.txt"))
    ab.set_segments("pitch", 2)
    ab.set_free("pitch.v4", True)
    ab.set_bounds("pitch.v4", "v1", 1.4)
    p1y = ab.get("pitch.v1")
    t = (ab.get("pitch.v4") - p1y) / (1.4 - p1y)
    check("the blade as the MSc optimisation has it: P4's pitch between P1's and 1.4; P1 moved, P4 keeps its "
          "fraction (y4_frac)",
          [*ab.bounds("pitch.v4"), ab.set("pitch.v1", 1.0), ab.get("pitch.v4"), *ab.bounds("pitch.v4")],
          [p1y, 1.4, 1.0, 1.0 + 0.4 * t, 1.0, 1.4], 1e-9)
    p1 = ab.save_file(os.path.join(tmp, "blade_bnd_ad.txt"))
    ab2 = XB.BladeAdapter(p1)
    check("the blade's set-up saved and loaded: the bound back, its variable a fraction for an optimiser",
          [ab2.bound_text("pitch.v4") == ("v1", None), ab2.get("pitch.v4"), ab2.free_space().factors == ["pitch.v4"]],
          [1, 1.0 + 0.4 * t, 1], 1e-9)

    ah = XH.HullAdapter(os.path.join(HERE, "suboff_hull.txt"))
    ah.set_bounds("tail", "0.8*nose", "1.2*nose")
    ah.set_bounds("nose_r.b1", "0.5*b0", "b0")
    tail = 0.96 + 0.48 * (ah.get("tail") - 0.8 * ah.get("nose")) / (0.4 * ah.get("nose"))
    check("the hull: the tail within 0.8 to 1.2 nose lengths follows the nose; a CST coefficient between half the "
          "one before and it",
          [ah.set("nose", 1.2), ah.get("tail"), *ah.bounds("tail"), ah.bounds("nose_r.b1")[1] - ah.get("nose_r.b0"),
           len(ah.preview()["warnings"])], [1.2, tail, 0.96, 1.44, 0.0, 0], 1e-9)
    p1 = ah.save_file(os.path.join(tmp, "hull_bnd_ad.txt"))
    ah2 = XH.HullAdapter(p1)
    check("the hull's set-up saved and loaded: the bounds back, the lengths as set",
          [{n: ah2.bound_text(n) for n in ah2.meta if ah2.is_dependent(n)}
           == {"tail": ("0.8*nose", "1.2*nose"), "nose_r.b1": ("0.5*b0", "b0")}, ah2.design.tail, ah2.design.nose],
          [1, tail, 1.2], 1e-9)


def test_rudder(tmp):
    print("\nthe rudder modified below H (rudder_modify.DesignSpace, xgeom_rudder.RudderAdapter, its set-up JSON)")
    missing = [f for f in RUDDER_FILES if not os.path.exists(os.path.join(HERE, f))]
    if missing:
        print("       skipped, not here: " + ", ".join(missing))
        return
    import rudder_modify as RM
    import xgeom_rudder as XR
    ad = XR.RudderAdapter()
    n0 = ad.n_free()[0]
    ad.set_relation("LE.dx3", "< dx2")
    ad.set_relation("LE.dx2", "> -0.5*H")
    ad.set_relation("P2z.v3", "= v2")
    check("relations between the LE's control points and on H; a section coordinate tied to another with '='",
          [ad.limits("LE.dx2")[0], ad.set("LE.dx2", -100.0), ad.get("LE.dx3"), ad.get("P2z.v3") - ad.get("P2z.v2"),
           ad.n_free()[0], "3 dependent" in ad.summary()], [-0.5 * ad.get("H"), -0.5 * ad.get("H"),
                                                         -0.5 * ad.get("H"), 0.0, n0 - 1, 1], 1e-9)
    h = ad.set("H", 60.0)
    check("H lowered: the LE point that depends on it follows (at least -0.5 H); the one under it stays",
          [h, ad.get("LE.dx2"), ad.get("LE.dx3")], [60.0, -30.0, -40.0], 1e-9)
    p1 = ad.save_space(os.path.join(tmp, "rudder_design_space.json"))
    with open(p1) as fh:
        rec = json.load(fh)
    rel = {q["name"]: q["relation"] for q in rec["space"]["slots"] if "relation" in q}
    check('the set-up JSON: "relation" in the slot of a dependent variable (names in full), no key in the others',
          [rel == {"LE.dx2": "> -0.5*H", "LE.dx3": "< LE.dx2", "P2z.v3": "= P2z.v2"},
           "P2z.v3" in rec["space"]["vector"]["names"], len(rec["space"]["vector"]["names"])], [1, 0, n0 - 1], 0)
    ad2 = XR.RudderAdapter()
    ad2.load_space(p1)
    p2 = ad2.save_space(os.path.join(tmp, "rudder_again_design_space.json"))
    check("the set-up loaded: the relations and the design back; saved again, the same file",
          [{n: ad2.relation(n) for n in ad2.meta if ad2.is_dependent(n)}
           == {"LE.dx2": "> -0.5*H", "LE.dx3": "< dx2", "P2z.v3": "= v2"},
           max(abs(ad2.get(n) - ad.get(n)) for n in ad.meta), filecmp.cmp(p1, p2, shallow=False)], [1, 0.0, 1], 1e-9)
    space, design, _ = RM.load_design_space(p1)
    x = space.to_vector(design)
    x1 = x.copy()
    x1[space.names.index("LE.dx3")] = 0.0
    check("for an optimiser (load_design_space): the relations, what a vector breaks, the vector projected",
          [sorted(space.relations) == ["LE.dx2", "LE.dx3", "P2z.v3"], len(space.violations(x)),
           [n for n, _, _ in space.violations(x1)] == ["LE.dx3"], space.project(x1)[space.names.index("LE.dx3")]],
          [1, 0, 1, -30.0], 1e-9)
    ad2.set_order("LE", 4)
    check("the LE curve with another number of control points: its relations go with its variables",
          sorted(n for n in ad2.meta if ad2.is_dependent(n)) == ["P2z.v3"], 1, 0)
    ar = XR.RudderAdapter()
    ar.set("LE.dx2", -100.0)
    ar.set("LE.dx3", -25.0)
    got = ar.set_bounds("LE.dx3", "dx2", 0)
    check("a dependent bound on the rudder: LE.dx3 between dx2 and 0; dx2 moved, dx3 keeps its place (3/4 of the way)",
          [*got, ar.set("LE.dx2", -200.0), ar.get("LE.dx3"), *ar.bounds("LE.dx3"),
           ar.bound_text("LE.dx3") == ("dx2", None)], [-100.0, 0.0, -200.0, -50.0, -200.0, 0.0, 1], 1e-9)
    check("the bounds of H stay numbers (h_min, h_max)",
          ["the bounds of H are numbers" in message(ar.set_bounds, "H", "-0.1*LE.dx2", 80), ar.is_dependent("H")],
          [1, 0], 0)
    p1 = ar.save_space(os.path.join(tmp, "rudder_bnd_design_space.json"))
    with open(p1) as fh:
        rec = json.load(fh)
    slot = {q["name"]: q for q in rec["space"]["slots"]}
    vec = rec["space"]["vector"]
    k = vec["names"].index("LE.dx3")
    check('the set-up JSON: "lo_expr" in the slot beside the numbers (names in full), "fractions" in the vector, '
          'the variable there as its fraction, 0 .. 1',
          [slot["LE.dx3"]["lo_expr"] == "LE.dx2", "hi_expr" in slot["LE.dx3"], slot["LE.dx3"]["lo"],
           slot["LE.dx3"]["hi"],
           vec["fractions"] == ["LE.dx3"], vec["values"][k], *vec["bounds"][k],
           sum("lo_expr" in q or "hi_expr" in q for q in rec["space"]["slots"])],
          [1, 0, -200.0, 0.0, 1, 0.75, 0.0, 1.0, 1], 1e-9)
    ar2 = XR.RudderAdapter()
    ar2.load_space(p1)
    p2 = ar2.save_space(os.path.join(tmp, "rudder_bnd_again_design_space.json"))
    space, design, _ = RM.load_design_space(p1)
    x = space.to_vector(design)
    x[space.names.index("LE.dx2")] = -80.0
    check("the set-up loaded: the bound and the design back, saved again the same file; for an optimiser "
          "(load_design_space) the variable is a fraction of the way from dx2 to 0",
          [ar2.bound_text("LE.dx3") == ("dx2", None), ar2.get("LE.dx3"), filecmp.cmp(p1, p2, shallow=False),
           space.factors == ["LE.dx3"], space.values(space.to_design(x))["LE.dx3"]], [1, -50.0, 1, 1, -20.0], 1e-9)
    rec["space"]["slots"][0]["lo_expr"] = "-0.1*LE.dx2"
    check("a set-up edited by hand to give H a dependent bound is refused",
          [rec["space"]["slots"][0]["name"] == "H",
           "the bounds of H are numbers" in message(RM.DesignSpace.from_dict, ar.orig, rec["space"])], [1, 1], 0)
    old = RM.DesignSpace(ad.orig, ad.orders, ad.join, h_min=ad.settings["h_min"], h_max=ad.settings["h_max"])
    rec = old.to_dict(ad.design)
    check("a set-up without dependent variables is written as before (no relation, lo_expr or hi_expr key in any "
          "slot, no fractions in the vector)",
          [sum(k in q for q in rec["slots"] for k in ("relation", "lo_expr", "hi_expr")), "fractions" in rec["vector"]],
          [0, 0], 0)


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

def test_gui(tmp):
    print("\nthe window: the dep switch and the relation entry of a variable's row (xgeom_tool.VarRow)")
    try:
        import tkinter as tk

        import xgeom_tool as XT
        root = tk.Tk()
    except Exception as exc:
        print(f"       skipped: {exc}")
        return

    def settle():
        for _ in range(5):
            root.update()
        time.sleep(0.05)                                 # the views' update, scheduled 15 ms after a change
        root.update()

    try:
        app = XT.App(root, "wing")
        app.curve.set("chord")
        app.on_curve()
        settle()
        rows = {r.name: r for r in app.rows}
        ad = app.ad
        r1 = rows["chord.v1"]
        check("a variable's row: its name beside it, dep unticked, no relation line",
              [r1.dep.get(), r1.rel.winfo_ismapped(), "span" in rows, "chord.v0" in rows], [0, 0, 1, 1], 0)
        r1.dep.set(True)
        r1.on_dep()
        settle()
        check("dep ticked: the relation line opens under the row; the variable is independent until one is entered",
              [r1.rel.winfo_ismapped(), ad.is_dependent("chord.v1"), "e.g." in r1.rel_info.cget("text")], [1, 0, 1], 0)
        r1.rel.insert(0, "<0.9v0")
        r1.on_relation()
        settle()
        check("a relation entered: the variable follows (108), the slider runs over its limits, they are shown",
              [ad.relation("chord.v1") == "< 0.9*v0", ad.get("chord.v1"), float(r1.scale.cget("to")),
               float(r1.scale.cget("from")), r1.rel_info.cget("text") == "now 40.00 .. 108.00",
               "1 dependent" in app.free_label.cget("text")], [1, 108.0, 108.0, 40.0, 1, 1], 1e-9)
        app.set_value("chord.v0", 100.0, None)
        settle()
        check("the root value moved to 100: the dependent row follows to 90, its slider's range with it",
              [ad.get("chord.v1"), float(r1.val.get()), float(r1.scale.cget("to"))], [90.0, 90.0, 90.0], 1e-9)
        r1.rel.delete(0, "end")
        r1.rel.insert(0, "> 0.5p4y")
        r1.on_relation()
        settle()
        check("a text that is no relation: named in the status bar, the relation as it was",
              ["'p4y' is not a variable" in app.status.cget("text"), ad.relation("chord.v1") == "< 0.9*v0"], [1, 1], 0)
        r0 = rows["chord.v2"]
        r0.dep.set(True)
        r0.on_dep()
        r0.rel.insert(0, "= v1")
        r0.on_relation()
        settle()
        check("'=': the row's slider, value and free box are off; the row shows the value it follows",
              [ad.get("chord.v2"), r0.scale.instate(["disabled"]), r0.val.instate(["disabled"]),
               r0.free_box.instate(["disabled"]), r0.rel_info.cget("text") == "now 90.00"], [90.0, 1, 1, 1, 1], 1e-9)
        target = os.path.join(tmp, "gui_rel.txt")
        save, open_ = XT.filedialog.asksaveasfilename, XT.filedialog.askopenfilename
        try:
            XT.filedialog.asksaveasfilename = lambda **kw: target
            app.on_save()
            settle()
            XT.filedialog.askopenfilename = lambda **kw: os.path.join(HERE, "wing.txt")
            app.on_load_params()
            settle()
            none = sum(app.ad.is_dependent(n) for n in app.ad.meta)
            XT.filedialog.askopenfilename = lambda **kw: target
            app.on_load_params()
            settle()
        finally:
            XT.filedialog.asksaveasfilename, XT.filedialog.askopenfilename = save, open_
        rows = {r.name: r for r in app.rows}
        check("Save set-up, Load ... of the sample (no dependent variable), Load ... of the saved file: the "
              "relations back, their rows with dep ticked and the relation shown",
              [none, app.ad.relation("chord.v1") == "< 0.9*v0", app.ad.relation("chord.v2") == "= v1",
               rows["chord.v1"].dep.get(), rows["chord.v1"].rel.get() == "< 0.9*v0",
               rows["chord.v1"].rel.winfo_ismapped(), rows["chord.v0"].dep.get()], [0, 1, 1, 1, 1, 1, 0], 0)
        r1 = rows["chord.v1"]
        r1.dep.set(False)
        r1.on_dep()
        settle()
        rows = {r.name: r for r in app.rows}
        check("dep unticked: the variable is independent again, its relation line gone, its slider over low .. high",
              [app.ad.is_dependent("chord.v1"), rows["chord.v1"].rel.winfo_ismapped(),
               float(rows["chord.v1"].scale.cget("to"))], [0, 0, 150.0], 1e-9)
        app.nctrl.set(3)
        app.on_order()
        settle()
        check("the chord with 3 control points: the rows are made again, no relation left on the curve",
              [sum(app.ad.is_dependent(n) for n in app.ad.meta),
               len([r for r in app.rows if r.name.startswith("chord.")])], [0, 4], 0)

        print("\nthe window: a dependent bound typed in a row's low or high box")

        def load(file):
            keep = XT.filedialog.askopenfilename
            try:
                XT.filedialog.askopenfilename = lambda **kw: file
                app.on_load_params()
                settle()
            finally:
                XT.filedialog.askopenfilename = keep
            return {r.name: r for r in app.rows}, app.ad

        def enter(box, text, row):
            box.delete(0, "end")
            box.insert(0, text)
            row.on_bounds()
            settle()

        rows, ad = load(os.path.join(HERE, "wing.txt"))
        r1 = rows["chord.v1"]
        enter(r1.lo, "0.5v0", r1)
        check("an expression typed in the low box: a dependent bound. The box shows it as the tool writes it, dep "
              "is ticked, the line under the row shows the limits now and the slider runs over them",
              [ad.bound_text("chord.v1") == ("0.5*v0", None), r1.lo.get() == "0.5*v0", r1.hi.get() == "150.00",
               r1.dep.get(), r1.rel.winfo_ismapped(), r1.rel.get() == "",
               r1.rel_info.cget("text") == "now 60.00 .. 150.00", float(r1.scale.cget("from")),
               "1 dependent" in app.free_label.cget("text"), r1.full.winfo_ismapped()],
              [1, 1, 1, 1, 1, 1, 1, 60.0, 1, 0], 1e-9)
        app.set_value("chord.v0", 100.0, None)
        settle()
        place = 50.0 + 100.0 * 50.0 / 90.0
        check("the root value moved to 100: the row keeps its place between its bounds (5/9 of the way, now from 50)",
              [ad.get("chord.v1"), float(r1.val.get()), float(r1.scale.cget("from")),
               r1.rel_info.cget("text") == "now 50.00 .. 150.00"], [place, round(place, 2), 50.0, 1], 1e-9)
        enter(r1.hi, "v0 +", r1)
        check("a text that is no bound: named in the status bar, the bounds as they were",
              ["the high bound" in app.status.cget("text"), ad.bound_text("chord.v1") == ("0.5*v0", None),
               ad.bounds("chord.v1")[1]], [1, 1, 150.0], 1e-9)
        design = ad.design
        enter(r1.hi, "150", r1)
        check("the same number typed another way (150 for 150.00) changes nothing",
              [ad.design is design, ad.bound_text("chord.v1") == ("0.5*v0", None)], [1, 1], 0)
        enter(r1.hi, "chord.v0 + 0.1*span", r1)
        check("an expression too long for its box: the bounds in full on a line under the row",
              [ad.bound_text("chord.v1") == ("0.5*v0", "v0+0.1*span"), r1.hi.get() == "v0+0.1*span",
               r1.full.winfo_ismapped(), r1.full.cget("text") == "bounds:  high v0+0.1*span", ad.bounds("chord.v1")[1]],
              [1, 1, 1, 1, 120.0], 1e-9)
        r3 = rows["chord.v3"]
        enter(r3.lo, "0.5*v1", r3)                       # 52.8 now; v3 is 70
        enter(r3.hi, "v2", r3)                           # 93.3 now
        app.set_value("chord.v2", 45.0, None)            # under v3's low: its bounds cross
        settle()
        r3.dep.set(False)
        r3.on_dep()
        settle()
        check("bounds that cross (low above high): the row says so, the status bar warns, and dep cannot be "
              "unticked there (the bounds cannot be numbers), which the status bar says",
              [r3.rel_info.cget("text") == "no room now: low above high", r3.dep.get(),
               "chord.v3 cannot be made independent" in app.status.cget("text"),
               ad.bound_text("chord.v3") == ("0.5*v1", "v2")], [1, 1, 1, 1], 0)
        app.set_value("chord.v2", 93.0, None)
        settle()
        r3.dep.set(False)
        r3.on_dep()
        settle()
        check("room again: dep unticked makes that row independent", [ad.is_dependent("chord.v3"), r3.dep.get()],
              [0, 0], 0)
        r1.dep.set(False)
        r1.on_dep()
        settle()
        rows = {r.name: r for r in app.rows}
        r1 = rows["chord.v1"]
        check("dep unticked: independent again, the bounds numbers where they were, the boxes show them, the lines "
              "under the row gone",
              [ad.is_dependent("chord.v1"), r1.lo.get() == "50.00", r1.hi.get() == "120.00", r1.rel.winfo_ismapped(),
               r1.full.winfo_ismapped(), *ad.bounds("chord.v1")], [0, 1, 1, 0, 0, 50.0, 120.0], 1e-9)
        enter(rows["chord.v2"].hi, "v1", rows["chord.v2"])
        target = os.path.join(tmp, "gui_bnd.txt")
        keep = XT.filedialog.asksaveasfilename
        try:
            XT.filedialog.asksaveasfilename = lambda **kw: target
            app.on_save()
            settle()
        finally:
            XT.filedialog.asksaveasfilename = keep
        rows, ad = load(os.path.join(HERE, "wing.txt"))
        none = sum(ad.is_dependent(n) for n in ad.meta)
        rows, ad = load(target)
        check("Save set-up, Load ... of the sample (no dependent variable), Load ... of the saved file: the bound "
              "back in its box, dep ticked",
              [none, ad.bound_text("chord.v2") == (None, "v1"), rows["chord.v2"].hi.get() == "v1",
               rows["chord.v2"].lo.get() == "40.00", rows["chord.v2"].dep.get(), rows["chord.v1"].dep.get()],
              [0, 1, 1, 1, 1, 0], 0)
    finally:
        root.destroy()


def main():
    print("XGeom dependent variables: self-check")
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            test_grammar()
            test_laws(tmp)
            test_follow()
            if ALL:
                test_spaces(tmp)
                test_files(tmp)
                test_adapters(tmp)
                test_bound_spaces(tmp)
                test_bound_files(tmp)
                test_bound_adapters(tmp)
            else:
                print("\nthe blade, the wing and the hull: skipped, not here (no wing_modify.py)")
            test_rudder(tmp)
            if "--gui" in sys.argv and ALL:
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
