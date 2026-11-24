"""
_xgeom_relations_test.py

Self-check for dependent variables (xgeom_relations.py): the grammar of a relation, what a relation does to a
variable's limits and how the dependent variables follow the others; the design spaces of every component
(blade, wing, hull, the rudder modified below H) with relations; the set-up files that carry them (the var
line of a component file, the rudder's set-up JSON); the design tool's adapters; with --gui the window's dep
switch and relation entry.

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
           full("≤ v0") == "<= chord.v0", full("≥ v3") == ">= chord.v3", full("== v0") == "= chord.v0",
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
          "[<relation>]", [any("var <name> <value> <low> <high> <Y free | N held> [<relation>]" in s for s in title),
                           any("curve <name> <form>" in s for s in title)], [1, 1], 0)
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
        if ld.space.relations:
            _FAILED.append(name)
    check("the samples have independent variables only (relations are the user's to set)", 1, 1, 0)


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
           notes[0].startswith("pitch.v2 is independent again") if notes else 0, len(ad2.relation_notes())],
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
    old = RM.DesignSpace(ad.orig, ad.orders, ad.join, h_min=ad.settings["h_min"], h_max=ad.settings["h_max"])
    check("a set-up without relations is written as before (no relation key in any slot)",
          sum("relation" in q for q in old.to_dict(ad.design)["slots"]), 0, 0)


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
    finally:
        root.destroy()


def main():
    print("XGeom dependent variables: self-check")
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            test_grammar()
            test_follow()
            if ALL:
                test_spaces(tmp)
                test_files(tmp)
                test_adapters(tmp)
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
