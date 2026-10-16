"""
_xgeom_blade_test.py

Self-check for the propeller blade in XGeom: the parameter file and its fitted
curves of 1 and 2 segments (blade_modify.py), its vary lines (Y/N per parameter,
blade_vary.py, blade_control.py, and para_control in the MSc pipeline:
para_control_bez_updated, x_blade_new, Initial_sampling.py with the CAD left
out), the blade radius, hub radius and hub height, the geometry through the MSc
code (para.py, blade_surface_new.py), the design tool's blade adapter
(xgeom_blade.py) and, with --gui, the window; with --cad the DRDC CAD (IGES and
STEP, pythonocc-core, about two minutes).

    python _xgeom_blade_test.py                 file, vary lines, curves, design space, geometry, adapter
    python _xgeom_blade_test.py --gui           also the window
    python _xgeom_blade_test.py --cad           also the CAD
    python _xgeom_blade_test.py --section-table PATH
                                                another section table than airfoil_data_fixed.csv

The geometry and adapter checks need the section table and are skipped without
it; the Initial_sampling.py checks need pythonocc-core (it imports X_CAD_new).
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import blade_modify as BM
import blade_vary as BV
import rudder_modify as RM

_FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))
ONE = {c: 1 for c in BM.CURVES}                      # every curve 1 segment


def check(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = float(np.abs(got - want).max()) if got.size else 0.0
    ok = err <= tol
    if not ok:
        _FAILED.append(name)
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<66s} max error {err:.3e}   (tol {tol:.0e})")


def msc_file(tmp, table=None):
    """The MSc blade's parameter file, with another section table if given."""
    p = BM.msc_params()
    if table:
        p.section_table = os.path.abspath(table)
    return BM.write_params(os.path.join(tmp, "msc.dat"), p)


def vary_file(tmp, name, lines, table=None):
    """The MSc blade's parameter file with '# vary' lines (e.g. 'pitch Y 0.6 1.6') above its table."""
    head, rest = open(msc_file(tmp, table)).read().split("# r/R", 1)
    path = os.path.join(tmp, name)
    with open(path, "w") as fh:
        fh.write(head + "".join(f"# vary {s}\n" for s in lines) + "# r/R" + rest)
    return path


ALL_N = [f"{n} N" for n in BM.VARY_NAMES]          # every vary line N: the original blade


def refused(fn, *args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1


def test_file(tmp):
    print("\nthe parameter file")
    import x_blade_new as XB
    path = msc_file(tmp)
    p = BM.read_params(path)
    p0 = BM.msc_params()
    check("written and read back: globals (hub height too) and the table (8 decimals)",
          [p.diameter - 1.4, p.blades - 5, p.root_r - XB.R_VALUES[0], p.hub_height - BM.HUB_HEIGHT,
           p.section_table == "airfoil_data_fixed.csv",
           max(np.abs(p.values[c] - p0.values[c]).max() for c in BM.CURVES), np.abs(p.r - p0.r).max()],
          [0, 0, 0, 0, 1, 0, 0], 5e-9)
    rf = np.linspace(XB.R_VALUES[0], 1.0, 5001)
    fns = {"pitch": XB.BASE_PITCH, "chord": XB.BASE_CHORD, "thickness": XB.BASE_MAX_THICKNESS,
           "camber": XB.BASE_MAX_CAMBER, "skew": XB.BASE_SKEW_ANGLE, "rake": XB.BASE_RAKE}
    check("the file's splines follow the MSc polynomials (fraction of each range)",
          [np.abs(p.spline(c)(rf) - fns[c](rf)).max() / np.ptp(fns[c](rf)) for c in BM.CURVES], 0.0, 1e-5)
    q = BM.read_params(path)
    q.root_r, q.hub_height = 0.2, None
    BM.write_params(os.path.join(tmp, "root02.dat"), q)
    q2 = BM.read_params(os.path.join(tmp, "root02.dat"))
    r, v = q2.rows()
    check("a root between two rows gets a row of its own (value from the spline); no hub height: None",
          [r[0], r[1] > 0.2, v["pitch"][0] - fns["pitch"](0.2), q2.hub_height is None,
           BM.fit_design(q2, segments=ONE)[0].hub_height], [0.2, 1, 0, 1, BM.HUB_HEIGHT], 1e-5)
    bad = os.path.join(tmp, "bad.dat")
    out = []
    for text in ("# diameter_m 1.4\n# blades 5\n0.2 1 2 3 4 5 6\n",
                 "# diameter_m 1.4\n# blades 5\n# section_table x.csv\n0.2 1 2 3\n0.3 1 2 3\n0.4 1 2 3\n0.5 1 2 3\n"):
        with open(bad, "w") as fh:
            fh.write(text)
        out.append(refused(BM.read_params, bad))
    check("a file without the section table, or with 4 columns, is refused", out, [1, 1], 0)
    q = BM.read_params(path)
    q.section_table, q.airfoil = "", "NACA 2416"
    pa = BM.write_params(os.path.join(tmp, "naca.dat"), q)
    qa = BM.read_params(pa)
    tab = np.loadtxt(qa.section_path(), delimiter=",", skiprows=1)
    pts, _ = BM.blade_points(BM.fit_design(qa, segments=ONE)[0], qa)
    out = []
    for head in ("# section_table airfoil_data_fixed.csv\n# airfoil naca2412\n", "# airfoil nosuchfoil\n"):
        with open(bad, "w") as fh:
            fh.write("# diameter_m 1.4\n# blades 5\n" + head + open(pa).read().split("# r/R", 1)[1].split("\n", 1)[1])
        out.append(refused(BM.read_params, bad))
    check("an airfoil by name (NACA 2416): read back, its section table (26 rows), the blade; both keys or an "
          "unknown airfoil refused", [qa.airfoil == "NACA 2416", qa.section_table == "", tab.shape[0],
                                      np.all(np.isfinite(pts))] + out, [1, 1, 26, 1, 1, 1], 0)


def test_vary(tmp):
    print("\nthe vary lines (Y/N per parameter)")
    p = BM.read_params(vary_file(tmp, "vary.dat", ["pitch Y 0.6 1.6", "chord y", "skew N", "radius yes 0.6 0.8"]))
    check("read: Y (y, yes) and N, a range; a name not listed is N; a file with no vary line: None",
          [BM.vary_free(p) == ("pitch", "chord", "radius"), BM.vary_bounds(p) == {"pitch": (0.6, 1.6),
                                                                                 "radius": (0.6, 0.8)},
           p.vary["thickness"].on, p.vary["skew"].on, BM.vary_free(BM.read_params(msc_file(tmp))) is None],
          [1, 1, 0, 0, 1], 0)
    out = [refused(BM.read_params, vary_file(tmp, "bad_vary.dat", lines))
           for lines in (["twist Y"], ["pitch Y", "pitch N"], ["pitch maybe"], ["pitch Y 1.6 0.6"], ["pitch Y 0.6"],
                         ["pitch Y low high"])]
    check("refused: an unknown name, a second line for one, a flag not Y or N, low not below high, one bound, "
          "not numbers", out, [1] * 6, 0)
    alone = os.path.join(tmp, "alone")                  # blade_vary.py by itself, as in a pipeline folder
    os.makedirs(alone, exist_ok=True)
    with open(os.path.join(HERE, "blade_vary.py")) as src, open(os.path.join(alone, "blade_vary.py"), "w") as dst:
        dst.write(src.read())
    run = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import blade_vary "
                          "as V; v = V.read_vary(sys.argv[2]); print(' '.join(n for n in V.VARY_NAMES if v[n].on))",
                          alone, p.path], capture_output=True, text=True)
    check("blade_vary.py alone (standard library only) reads them; blade_modify's names and reader are its own",
          [run.stdout.strip() == "pitch chord radius", BM.read_vary is BV.read_vary,
           BM.VARY_NAMES == BM.CURVES + BM.GLOBAL_SLOTS], [1, 1, 1], 0)
    design, _ = BM.fit_design(p, segments={"chord": 1, "thickness": 1})
    kept = [c for c in BM.CURVES if c not in ("pitch", "chord")]
    rr = np.linspace(design.r0, design.r1, 2001)
    rep = BM.fit_report(p, design)
    check("fit_design: pitch 2 segments, chord 1 (SEGMENTS); the N curves the file's own (also with SEGMENTS), "
          "exactly its splines",
          [design.curves["pitch"].segments, design.curves["chord"].segments]
          + [design.curves[c].segments for c in kept]
          + [max(np.abs(design.prop(c)(rr) - p.spline(c)(rr)).max() for c in kept), max(rep[c][0] for c in kept)],
          [2, 1, 0, 0, 0, 0, 0, 0], 1e-12)
    space = BM.file_space(p, design)
    bnd = dict(zip(space.names, map(tuple, space.bounds)))
    x0 = space.to_vector(design)
    check("file_space: the Y parameters free (pitch 8, chord 10, the radius), no slot of a file's own curve; "
          "pitch's values and the radius in the file's ranges; round trip; the blade inside them",
          [len(space), sum(n.startswith("pitch.") for n in space.names), sum(n.startswith("chord.") for n in space.names),
           "radius" in space.names, any(n.split(".")[0] in kept for n in space.all_names),
           all(bnd[n] == (0.6, 1.6) for n in space.names if n.startswith("pitch.v")), bnd["radius"] == (0.6, 0.8),
           np.abs(space.to_vector(space.to_design(x0)) - x0).max(), len(BM.outside_bounds(space, design))],
          [19, 8, 10, 1, 0, 1, 1, 0, 0], 1e-14)
    narrow = BM.read_params(vary_file(tmp, "narrow.dat", ["pitch Y 1.0 1.2"]))
    dn, _ = BM.fit_design(narrow)
    out_n = BM.outside_bounds(BM.file_space(narrow, dn), dn)
    alln = BM.read_params(vary_file(tmp, "all_n.dat", ALL_N))
    da, _ = BM.fit_design(alln)
    check("a range the file's blade leaves (pitch 1.0 .. 1.2): outside_bounds names pitch's values; every line N: "
          "nothing free, every curve the file's own, the blade exactly the file's",
          [len(out_n) > 0, all(n.startswith("pitch.v") for n in out_n), BM.vary_free(alln) == (),
           len(BM.file_space(alln, da)), all(da.curves[c].segments == 0 for c in BM.CURVES),
           max(BM.fit_report(alln, da)[c][0] for c in BM.CURVES)], [1, 1, 1, 0, 1, 0], 1e-12)
    q = BM.read_params(BM.write_params(os.path.join(tmp, "vary_back.dat"), BM.design_params(design, p)))
    big = design.copy()
    big.radius, big.hub_radius = 0.77, 1.2 * design.hub_radius
    frac = np.linspace(0.0, 1.0, 11)
    check("written (design_params) and read back: the same vary lines; the file's own curves stretch with the "
          "blade and hub radius", [q.vary == p.vary] + [np.abs(big.at(c, big.r0 + frac * (big.r1 - big.r0))
                                                            - design.at(c, design.r0 + frac * (design.r1 - design.r0))
                                                            ).max() for c in kept], [1, 0, 0, 0, 0], 1e-12)
    repo = BM.read_params(os.path.join(HERE, "msc_blade_params.dat"))
    m = BM.msc_params()
    check("msc_blade_params.dat: its vary lines read (the flags are the user's); its table the MSc polynomials' (8 "
          "decimals, written on another machine)",
          [repo.vary is not None, max(np.abs(repo.values[c] - m.values[c]).max() for c in BM.CURVES)], [1, 0], 2e-8)


def test_msc_flags():
    print("\nthe MSc pipeline's para_control (para_control_bez_updated, x_blade_new)")
    import para_control_bez_updated as PB
    import x_blade_new as XB
    pcon = np.array([0.9, 1.0, 0.5, 0.3, 0.3, 0.7])
    ccon = np.array([0.2, 0.6, 0.3, 0.45, 0.2])
    args = (XB.BASE_PITCH, XB.BASE_CHORD, XB.R_VALUES)
    o0 = PB.para_control_bez_updated(*args, 0, pcon, ccon)
    o3 = PB.para_control_bez_updated(*args, 3, pcon, ccon, max_pitch=1.3)
    o2 = PB.para_control_bez_updated(*args, 2, pcon, ccon)
    check("para_control from Y/N (both 7, chord 2, pitch 3, neither 0); 0: the original pitch and chord, no "
          "control points; 3: pitch only, P4 at max_pitch; 2: chord only; 5 refused",
          [PB.para_control_flag(a, b) for a, b in ((1, 1), (0, 1), (1, 0), (0, 0))]
          + [o0[0] is XB.BASE_PITCH, o0[1] is XB.BASE_CHORD, np.abs(o0[2]).max() + np.abs(o0[3]).max(),
             o3[1] is XB.BASE_CHORD, float(o3[0](0.5)), o2[0] is XB.BASE_PITCH, o2[1] is not XB.BASE_CHORD,
             refused(PB.para_control_bez_updated, *args, 5, pcon, ccon)],
          [7, 2, 3, 0, 1, 1, 0, 1, 1.3, 1, 1, 1], 1e-12)
    if not os.path.exists(os.path.join(HERE, "airfoil_data_fixed.csv")):
        print("       X_blade skipped: no airfoil_data_fixed.csv")
        return
    from para import para
    pts = XB.X_blade(np.zeros(6), np.zeros(5), 0, write_dat=False, para_control=0)[0]
    ref = para(XB.BASE_MAX_CAMBER, XB.BASE_PITCH, XB.BASE_CHORD, XB.BASE_MAX_THICKNESS, XB.BASE_SKEW_ANGLE,
               XB.BASE_RAKE, XB.R_VALUES, 0, write_dat=False)
    check("X_blade with para_control 0: the original blade's points (the polynomials'), m", np.abs(pts - ref).max(),
          0.0, 0)


def test_sampling(tmp):
    print("\nInitial_sampling.py and the vary lines (the CAD left out)")
    if importlib.util.find_spec("OCC") is None or not os.path.exists(os.path.join(HERE, "airfoil_data_fixed.csv")):
        print("       skipped: no pythonocc-core (Initial_sampling imports X_CAD_new) or no airfoil_data_fixed.csv")
        return
    import Initial_sampling as IS
    import para as PARA
    files = {"both": vary_file(tmp, "s7.dat", ["pitch Y", "chord Y"]), "none": vary_file(tmp, "s0.dat", ALL_N),
             "pitch": vary_file(tmp, "s3.dat", ["pitch Y 0.9 1.5"]),
             "chord": vary_file(tmp, "s2.dat", ["chord Y 0.2 0.6", "skew Y"])}
    s = {k: IS.sampling_setup(v) for k, v in files.items()}
    s["legacy"], s["missing"] = IS.sampling_setup(msc_file(tmp)), IS.sampling_setup(os.path.join(tmp, "nofile.dat"))
    dev = lambda k: np.abs(s[k]["pitch_bounds"] - IS.pitch_bounds).max() + np.abs(s[k]["chord_bounds"]
                                                                                 - IS.chord_bounds).max()
    check("sampling_setup: pitch and chord Y 7 (the code's bounds); every line N 0; pitch only 3 (its range on "
          "p1y, p7y, max_pitch); chord only 2 (its range on p1y, y4; skew noted); no vary line or no file 7",
          [s["both"]["para_control"], dev("both"), s["none"]["para_control"], "nothing varies" in s["none"]["notes"][0],
           s["pitch"]["para_control"], *s["pitch"]["pitch_bounds"][[0, 5]].ravel(), s["pitch"]["max_pitch"],
           s["chord"]["para_control"], *s["chord"]["chord_bounds"][[0, 3]].ravel(), "skew" in s["chord"]["notes"][-1],
           s["legacy"]["para_control"], dev("legacy"), s["missing"]["para_control"]],
          [7, 0, 0, 1, 3, 0.9, 1.5, 0.9, 1.5, 1.5, 2, 0.2, 0.6, 0.2, 0.6, 1, 7, 0, 7], 0)
    calls = []

    def no_cad(pitch_con, chord_con, case_id, geometry_dir, para_control=7, max_pitch=1.4):
        calls.append((np.asarray(pitch_con, float), np.asarray(chord_con, float), int(case_id), int(para_control),
                      float(max_pitch), Path(geometry_dir).name))
        return True

    keep = {k: getattr(IS, k) for k in ("generate_cad", "BLADE_PARAMS", "INPUT_MODE", "N_GENERATED_BLADES",
                                        "RESET_OUTPUT_FILES", "CONTROL_OUTPUT_PATH", "REJECTED_CONTROL_POINTS_PATH",
                                        "PLOT_OUTPUT_PATH", "GENERATED_CONTROL_POINTS_PATH", "GEOMETRY_DIR",
                                        "ORIGINAL_GEOMETRY_DIR")}
    keep_dir, got = PARA.GEOMETRY_OUTPUT_DIR, {}
    try:
        IS.generate_cad, IS.INPUT_MODE, IS.N_GENERATED_BLADES, IS.RESET_OUTPUT_FILES = no_cad, "test", 3, True
        for k in ("both", "pitch", "chord", "none"):
            d = Path(tmp) / f"sampling_{k}"
            IS.BLADE_PARAMS = Path(files[k])
            IS.CONTROL_OUTPUT_PATH, IS.REJECTED_CONTROL_POINTS_PATH = d / "accepted.txt", d / "rejected.txt"
            IS.PLOT_OUTPUT_PATH, IS.GENERATED_CONTROL_POINTS_PATH = d / "curves.png", d / "generated.txt"
            IS.GEOMETRY_DIR, IS.ORIGINAL_GEOMETRY_DIR = d / "geometry", d / "original"
            n = len(calls)
            with contextlib.redirect_stdout(io.StringIO()):
                IS.main()
            gen = d / "generated.txt"
            got[k] = (calls[n:], np.loadtxt(gen, ndmin=2) if gen.exists() else None)
    finally:
        for k, v in keep.items():
            setattr(IS, k, v)
        PARA.GEOMETRY_OUTPUT_DIR = keep_dir
    flags = {k: sorted({c[3] for c in got[k][0]}) for k in got}
    gp, gc = got["pitch"][1], got["chord"][1]
    check("main() (the CAD left out): both 11 sampled variables, para_control 7; pitch only 6 (p1y, p7y in "
          "0.9 .. 1.5), 3, max_pitch 1.5, chord zeros; chord only 5, 2, pitch zeros; nothing: no sampling, the "
          "original (case 0, para_control 0) only",
          [got["both"][1].shape[1], flags["both"] == [7], gp.shape[1], gp[:, [0, 5]].min() >= 0.9,
           gp[:, [0, 5]].max() <= 1.5, flags["pitch"] == [3], all(c[4] == 1.5 and not c[1].any() for c in got["pitch"][0]),
           gc.shape[1], flags["chord"] == [2], all(not c[0].any() for c in got["chord"][0]), got["none"][1] is None,
           [(c[2], c[3], c[5]) for c in got["none"][0]] == [(0, 0, "original")],
           min(len(got[k][0]) for k in ("both", "pitch", "chord")) > 0],
          [11, 1, 6, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1], 0)


def test_curves(tmp):
    print("\nthe curves")
    u = np.array([0.0, 0.3, 0.55, 0.8, 1.0])
    v = np.array([1.2, 1.4, 0.9, 1.1, 0.6])
    d = BM.BladeDesign(0.7, 0.14, 1.0, 1.0, {c: BM.BladeCurve(u, v) for c in BM.CURVES})
    rr = np.linspace(0.2, 1.0, 41)
    cu, res = BM.fit_curve(rr, d.at("pitch", rr), 0.2, 1.0, 4)
    d2 = BM.BladeDesign(0.7, 0.14, 1.0, 1.0, {c: cu for c in BM.CURVES})
    rf = np.linspace(0.2, 1.0, 2001)
    check("1 segment: fit_curve finds a Bezier of its own order again",
          [d.r0 - 0.2, np.abs(res).max(), np.abs(d2.prop("pitch")(rf) - d.prop("pitch")(rf)).max()], 0.0, 1e-8)
    two = BM.TwoSegmentCurve(1.2, 0.3, 0.5, 0.45, 1.5, 0.6, 0.8, 0.7)
    d3 = BM.BladeDesign(0.7, 0.14, 1.0, 1.0, {c: two for c in BM.CURVES})
    rr = np.linspace(0.2, 1.0, 81)
    t2, res2 = BM.fit_two_segment(rr, d3.at("pitch", rr), 0.2, 1.0)
    d4 = BM.BladeDesign(0.7, 0.14, 1.0, 1.0, {c: t2 for c in BM.CURVES})
    check("2 segments: fit_two_segment finds a 2-segment curve again",
          [np.abs(res2).max(), np.abs(d4.prop("pitch")(rf) - d3.prop("pitch")(rf)).max()], 0.0, 1e-6)
    r4, h = two.radii(0.2, 1.0)[1], 1e-7
    p = two.ctrl(0.2, 1.0)
    check("2 segments: P2 = P3 and P5 = P6 at P4's value, zero slope at P4, ends at P1 and P7",
          [np.abs(p[1] - p[2]).max(), np.abs(p[4] - p[5]).max(), np.ptp(p[1:6, 1]),
           (d3.at("pitch", r4) - d3.at("pitch", r4 - h)) / h, (d3.at("pitch", r4 + h) - d3.at("pitch", r4)) / h,
           d3.at("pitch", 0.2) - 1.2, d3.at("pitch", 1.0) - 0.7], 0.0, 1e-6)
    p = BM.read_params(msc_file(tmp))
    for segs, tol in ((ONE, 0.01), (None, 0.08)):
        design, _ = BM.fit_design(p, segments=segs)
        rep = BM.fit_report(p, design)
        form = "1 segment, 6 control points" if segs else "2 segments"
        for c in BM.CURVES:
            print(f"       {c:9s} {form}: largest difference from the file {rep[c][0]:.2e} {BM.UNITS[c]} "
                  f"({100 * rep[c][1]:.2f} % of its range)")
        check(f"fit_design on the MSc file, {form}: each curve within {100 * tol:.0f} % of its range",
              [rep[c][1] for c in BM.CURVES], 0.0, tol)
    d1, _ = BM.fit_design(p, segments=ONE)
    rs = d1.r0 + rf * (d1.r1 - d1.r0)
    ref = d1.prop("chord")(rs)
    d7 = BM.refit_curve(d1, "chord", 7)
    check("refit_curve to 8 control points keeps the shape (c/D)",
          [d7.curves["chord"].order - 7, np.abs(d7.prop("chord")(rs) - ref).max()], 0.0, 5e-4)
    d2s = BM.refit_curve(d1, "chord", segments=2)
    check("refit_curve from 1 to 2 segments: close to the shape (fraction of the range)",
          [d2s.curves["chord"].segments - 2,
           np.abs(d2s.prop("chord")(rs) - ref).max() / np.ptp(ref)], 0.0, 0.02)
    design, _ = BM.fit_design(p)
    path = BM.write_params(os.path.join(tmp, "design.dat"), BM.design_params(design, p))
    q = BM.read_params(path)
    rr = np.linspace(design.r0, design.r1, 3001)
    check("the design (2 segments) written as a parameter file reproduces it (fraction of each range)",
          [np.abs(q.spline(c)(rr) - design.prop(c)(rr)).max() / np.ptp(design.prop(c)(rr)) for c in BM.CURVES],
          0.0, 2e-3)
    f = design.prop("skew")
    check("a property function: single values and arrays agree, linear beyond the ends",
          [np.abs(np.array([float(f(x)) for x in rr[::50]]) - f(rr[::50])).max(),
           float(f(1.01)) - (float(f(1.0)) + 0.01 * (float(f(1.0)) - float(f(1.0 - 1e-6))) / 1e-6)], 0.0, 1e-4)


def test_space(tmp):
    print("\nthe design space")
    p = BM.read_params(msc_file(tmp))
    d1, _ = BM.fit_design(p, {"pitch": 6}, ONE)
    d2, _ = BM.fit_design(p)
    for design, form, n_all, n_free in ((d1, "1 segment", 3 + 5 * 10 + 12, 12 + 10), (d2, "2 segments", 3 + 6 * 8, 16)):
        space = BM.BladeSpace(p, design, free=("pitch", "chord"))
        x0 = space.to_vector(design)
        dd = space.to_design(x0)
        check(f"{form}: slots (3 globals and the curves'), free pitch and chord",
              [len(space.all_names), len(space), all(n.split(".")[0] in ("pitch", "chord") for n in space.names)],
              [n_all, n_free, 1], 0)
        rr = np.linspace(design.r0, design.r1, 501)
        allv = space.values(design)
        inside = [space.all_bounds[i, 0] <= allv[n] <= space.all_bounds[i, 1] for i, n in enumerate(space.all_names)]
        check(f"{form}: design vector round trip; the fitted design inside its default bounds",
              [max(np.abs(dd.at(c, rr) - design.at(c, rr)).max() for c in BM.CURVES), dd.radius - design.radius,
               all(inside)], [0, 0, 1], 1e-14)
    space = BM.BladeSpace(p, d2, free=("pitch", "radius", "hub_height"))
    vals = space.values(d2)
    vals.update(radius=0.8, hub_height=1.2)
    d3 = space.design_of(vals)
    check("globals in the space: radius and hub height free, the hub radius held",
          [space.names[:2] == ["radius", "hub_height"], d3.radius, d3.hub_height, d3.hub_radius, d3.r0],
          [1, 0.8, 1.2, d2.hub_radius, d2.hub_radius / 0.8], 1e-12)
    rec = BM.space_record(space, d3, p)
    path = os.path.join(tmp, "space.json")
    with open(path, "w") as fh:
        json.dump(rec, fh)
    s3, d4, rec3 = BM.load_blade_space(path)
    check("space_record, then load_blade_space: the same variables, values, bounds, forms",
          [s3.names == space.names, np.abs(s3.to_vector(d4) - space.to_vector(d3)).max(),
           np.abs(s3.bounds - space.bounds).max(), rec3["geometry"] == "blade", s3.forms == space.forms],
          [1, 0, 0, 1, 1], 0)
    old = BM.BladeSpace(p, d1, free=("pitch",)).to_dict(d1)           # a set-up saved before the globals
    old.pop("forms")
    old["r0"] = d1.r0
    old["slots"] = [q for q in old["slots"] if q["name"] not in BM.GLOBAL_SLOTS]
    s5, d5 = BM.BladeSpace.from_dict(p, old)
    rr = np.linspace(d1.r0, d1.r1, 201)
    check("an older set-up (no globals, 1 segment): read, the globals held at the file's",
          [max(np.abs(d5.at(c, rr) - d1.at(c, rr)).max() for c in BM.CURVES), d5.radius - 0.7, d5.r0 - d1.r0,
           "radius" in s5.fixed, len(s5) - 12], [0, 0, 0, 1, 0], 1e-12)


def test_geometry(tmp, table):
    print("\nthe geometry (MSc code: para.py, BladeSurface)")
    if not os.path.exists(table):
        print(f"       skipped: no section table {table}")
        return False
    import x_blade_new as XB
    from para import para
    p = BM.read_params(msc_file(tmp, table))
    st = BM.r_stations(XB.R_VALUES[0])
    check(f"r_stations({XB.R_VALUES[0]:g}) are x_blade_new.R_VALUES",
          [st.shape == XB.R_VALUES.shape, np.abs(st - XB.R_VALUES).max() if st.shape == XB.R_VALUES.shape else 1],
          [1, 0], 0)
    props = tuple(p.spline(c) for c in ("camber", "pitch", "chord", "thickness", "skew", "rake"))
    pts = para(*props, st, 0, write_dat=False, d=p.diameter, airfoil_path=p.section_path())
    ref = para(XB.BASE_MAX_CAMBER, XB.BASE_PITCH, XB.BASE_CHORD, XB.BASE_MAX_THICKNESS, XB.BASE_SKEW_ANGLE,
               XB.BASE_RAKE, XB.R_VALUES, 0, write_dat=False, airfoil_path=p.section_path())
    check("para.py on the file's splines = para.py on the polynomials (m)", np.abs(pts - ref).max(), 0.0, 2e-5)
    from scipy.spatial import cKDTree
    from rot_axis import rot_axis
    p2 = ref @ rot_axis(np.array([1, 0, 0]), np.deg2rad(72))
    gap_xb = min(float(np.min(cKDTree(p2).query(ref, k=1)[0])), float(np.min(cKDTree(ref).query(p2, k=1)[0])))
    check("clearance = X_blade's check (m)", BM.clearance(ref, 5) - gap_xb, 0.0, 1e-15)
    design, _ = BM.fit_design(p)
    bs = BM.blade_surface(design, p)
    eta = np.linspace(bs.eta_min, 1.0, 40)
    xi = np.linspace(0.0, 0.5, 40)
    check("BladeSurface of the design: TE and tip closed (m)",
          [np.abs(bs.b(0.0, eta) - bs.b(1.0, eta)).max(), np.abs(bs.b(xi, 1.0) - bs.b(1.0 - xi, 1.0)).max()], 0.0, 1e-12)
    probs = BM.design_problems(design, p)
    bad = design.copy()
    bad.curves["chord"] = bad.curves["chord"].scaled(-1.0)
    short, wide = design.copy(), design.copy()
    short.hub_height = 0.2
    wide.hub_radius = 0.95 * design.radius
    check("problems: none for the fitted design; a negative chord, a large clearance limit, a short hub, a hub "
          "radius near the tip found",
          [len(probs), "chord not positive" in " ".join(BM.design_problems(bad, p)),
           "apart" in " ".join(BM.design_problems(design, p, clearance_mm=1e4)),
           "does not cover the blade root" in " ".join(BM.design_problems(short, p)),
           "hub radius" in " ".join(BM.design_problems(wide, p)),
           len(BM.design_problems(short, p, hub=False))], [0, 1, 1, 1, 1, 0], 0)
    big = design.copy()
    big.radius *= 1.1
    big.hub_radius *= 1.2
    pb, sb = BM.blade_points(big, p)
    rad = np.hypot(pb[:, 1], pb[:, 2]).reshape(-1, 53)
    frac = np.linspace(0.0, 1.0, 11)
    check("blade and hub radius: the root and tip sections at them (mm), the curves stretched over the span",
          [1000.0 * rad[0].mean() - 1200.0 * design.hub_radius, 1000.0 * rad[-1].mean() - 1100.0 * design.radius,
           max(np.abs(big.at(c, big.r0 + frac * (big.r1 - big.r0)) - design.at(c, design.r0 + frac * (
               design.r1 - design.r0))).max() for c in BM.CURVES)], 0.0, 1e-6)
    res = BM.write_case(tmp, "t", design, p, BM.BladeSpace(p, design, free=("pitch",)), cad=False)
    secs = [b for b in open(res["paths"]["xcad_m.dat"]).read().split("#")[1:]]
    pts_back = np.loadtxt(res["paths"]["xcad_m.dat"], comments="#")
    q = BM.read_params(res["paths"]["params.dat"])
    check(f"write_case: XCAD points ({BM.SECTIONS} sections of 53, as para.py), the parameter file, the JSON, "
          "the plot",
          [len(secs), pts_back.shape[0], q.blades, json.load(open(res["paths"]["design.json"]))["globals"]["blades"],
           os.path.exists(res["paths"]["check"])], [BM.SECTIONS, BM.SECTIONS * 53, 5, 5, 1], 0)
    res_b = BM.write_case(tmp, "big", big, p, cad=False, plot=False)
    qb = BM.read_params(res_b["paths"]["params.dat"])
    check("write_case of a bigger blade and hub: the parameter file has its D, root r/R, hub height",
          [qb.diameter - big.diameter, qb.root_r - big.r0, qb.hub_height - big.hub_height], 0.0, 1e-9)
    s = BM.section_stations(0.17, 57, 0.93, 14)
    s2 = BM.section_stations(0.2, 30, 0.9, 8)
    check("section_stations: 57 from the root to the tip, 14 from r/R 0.93 closer together, even below",
          [len(s), s[0], s[-1], s[43], np.ptp(np.diff(s[:44])), np.all(np.diff(s[43:]) > 0),
           np.all(np.diff(np.diff(s[43:])) < 0), len(s2), s2[0], s2[22], s2[-1]],
          [57, 0.17, 1.0, 0.93, 0.0, 1, 1, 30, 0.2, 0.9, 1.0], 1e-12)
    out = [refused(BM.section_stations, *args)
           for args in ((0.17, 20, 0.1, 5), (0.17, 20, 1.0, 5), (0.17, 10, 0.93, 10), (0.17, 10, 0.93, 1))]
    check("section_stations refuses a band outside the blade, too few sections", out, [1, 1, 1, 1], 0)
    res2 = BM.write_case(tmp, "t30", design, p, cad=False, settings={"sections": 30, "tip_sections": 8}, plot=False)
    secs2 = open(res2["paths"]["xcad_m.dat"]).read().split("#")[1:]
    rec2 = json.load(open(res2["paths"]["design.json"]))
    check("write_case with 30 sections (8 in the tip band): 30 in the file; the clearance as before",
          [len(secs2), res2["sections"], rec2["settings"]["sections"], res2["clearance_mm"] - res["clearance_mm"]],
          [30, 30, 30, 0], 0)
    return True


def test_adapter(tmp, table):
    print("\nthe tool's blade adapter (xgeom_blade.py)")
    import xgeom_blade as XBL
    path = msc_file(tmp, table)
    ad = XBL.BladeAdapter(params=path)
    check("starts with 2-segment curves, pitch and chord free: 16 of 51 variables",
          [ad.n_free()[0], ad.n_free()[1], all(ad.segments(c) == 2 for c in BM.CURVES)], [16, 51, 1], 0)
    v = ad.set("pitch.v4", 99.0)
    check("set clamps to the bounds", v, ad.bounds("pitch.v4")[1], 0)
    ad.set_bounds("chord.w56", 0.1, 1.5)
    w_lo = ad.bounds("chord.w56")[0]
    ad.set_free("skew.v7", True)
    no_order = refused(ad.set_order, "chord", 6)
    msg1 = ad.set_segments("chord", 1)
    msg2 = ad.set_order("chord", 6)
    msg3 = ad.set_segments("pitch", 1)                    # pitch was changed above: its present shape kept
    check("bounds, a free flag, order refused at 2 segments; chord (as fitted) to 1 segment, 7 control points "
          "(free), fitted to the file; pitch (changed) fitted to its shape",
          [w_lo, ad.is_free("skew.v7"), no_order, ad.segments("chord"), ad.order("chord"), ad.n_free("chord")[0],
           msg1.startswith("fitted to the parameter file"), msg2.startswith("fitted to the parameter file"),
           BM.fit_report(ad.params, ad.design)["chord"][1] < 0.01, msg3.startswith("fitted to the curve as it was")],
          [0.1, 1, 1, 1, 6, 12, 1, 1, 1, 1], 0)
    print(f"       {msg1}; {msg2}; pitch: {msg3}")
    pv = ad.preview(skeleton=True)
    check(f"preview: no problems, 6 sections, {BM.SECTIONS} sections in the skeleton",
          [len(pv["problems"]), np.shape(pv["sections"])[0], np.shape(pv["skeleton"])[0]], [0, 6, BM.SECTIONS], 0)
    ad.set("hub_radius", 0.13)
    ad.set("radius", 0.72)
    pv2 = ad.preview()
    ad.set_bounds("hub_height", 0.1, 2.0)
    ad.set("hub_height", 0.2)
    pv3 = ad.preview()
    ad.set("hub_height", 1.0)
    check("globals: hub radius 0.13 m, R 0.72 m (root r/R follows), a 0.2 m hub is a problem",
          [ad.design.r0, ad.row_info("hub_radius") == f"r/R {0.13 / 0.72:.3f}", len(pv2["problems"]),
           "does not cover" in " ".join(pv3["problems"])], [0.13 / 0.72, 1, 0, 1], 1e-12)
    ad.set_sections(40, 0.9, 10)
    no_sec = refused(ad.set_sections, 12, 0.9, 12)
    pv = ad.preview(skeleton=True)
    check("set_sections: 40 sections in the skeleton, a bad set refused",
          [np.shape(pv["skeleton"])[0], no_sec, ad.settings["tip_sections"]], [40, 1, 10], 0)
    sp = ad.save_space(os.path.join(tmp, "setup.json"))
    s2, d2, _ = BM.load_blade_space(sp)
    ad2 = XBL.BladeAdapter(params=path)
    ad2.load_space(sp)
    check("Save set-up: load_blade_space and the adapter read it back (forms, globals, sections)",
          [s2.names == ad.free_space().names, np.abs(s2.to_vector(d2) - ad.free_space().to_vector(ad.design)).max(),
           max(abs(ad2.values[k] - ad.values[k]) for k in ad.values), ad2.n_free() == ad.n_free(),
           ad2.segments("chord"), ad2.design.radius, ad2.settings["sections"] == 40],
          [1, 0, 0, 1, 1, 0.72, 1], 1e-14)
    saved = ad.save_params(os.path.join(tmp, "saved.dat"))
    ad3 = XBL.BladeAdapter(params=saved)
    for c in BM.CURVES:                                    # the same forms (a curve as fitted: from the file)
        ad3.set_segments(c, ad.segments(c))
    rr = np.linspace(ad.design.r0, ad.design.r1, 501)
    check("Save parameters, then Load: the globals, the curves fitted again to the saved file (fraction of range)",
          [ad3.design.radius - 0.72, ad3.design.hub_radius - 0.13]
          + [np.abs(ad3.design.prop(c)(rr) - ad.design.prop(c)(rr)).max() / np.ptp(ad.design.prop(c)(rr))
             for c in BM.CURVES], 0.0, 0.01)
    gap5 = ad.preview()["clearance_mm"]
    no_z = [refused(ad.set_blades, z) for z in (1, 8, 4.5)]
    ad.set_blades(7)
    pv7 = ad.preview()
    sp7 = ad.save_space(os.path.join(tmp, "setup7.json"))
    s7, _, rec7 = BM.load_blade_space(sp7)
    ad7 = XBL.BladeAdapter(params=path)
    ad7.load_space(sp7)
    p7 = BM.read_params(ad.save_params(os.path.join(tmp, "z7.dat")))
    check("blades Z: 1, 8 and 4.5 refused (2 to 7); 7: the blades closer, the set-up and the saved file keep 7, "
          "the file's 5 kept apart", no_z + [pv7["clearance_mm"] < gap5, rec7["globals"]["blades"], s7.params.blades,
                                             ad7.params.blades, ad7.file_blades, p7.blades],
          [1, 1, 1, 1, 7, 7, 7, 5, 7], 0)
    print(f"       blades {gap5:.1f} mm apart with 5, {pv7['clearance_mm']:.1f} mm with 7")
    ad.settings["clearance_mm"] = pv7["clearance_mm"] + 10.0       # closer than the clearance: interfering
    pvw = ad.preview()
    res = BM.write_case(tmp, "warn", ad.design, ad.params, cad=False, settings=ad._case_settings(), plot=False)
    rec = json.load(open(res["paths"]["design.json"]))
    strict = BM.design_problems(ad.design, ad.params, ad.settings["clearance_mm"])
    check("blades closer than the clearance: a warning, not a problem (the tool builds the CAD); design.json "
          "records it; design_problems (the optimiser's check) still rejects the design",
          [len(pvw["problems"]), len(pvw["warnings"]), "interfere" in pvw["warnings"][0], len(res["problems"]),
           len(rec["warnings"]), len(strict)], [0, 1, 1, 0, 1, 1], 0)
    ad.settings["clearance_mm"] = BM.CLEARANCE_MM


def test_adapter_vary(tmp, table):
    print("\nthe tool's blade adapter and the vary lines")
    import xgeom_blade as XBL
    ad = XBL.BladeAdapter(params=vary_file(tmp, "vary_ad.dat", ["pitch Y 0.6 1.6", "chord Y", "radius Y"], table))
    kept = ("thickness", "camber", "skew", "rake")
    check("starts from the vary lines: pitch and chord 2 segments, free (pitch's values in 0.6 .. 1.6), the radius "
          "free, the hub height held; the N curves the file's own ('(file)' in the picker, a pinned row)",
          [ad.n_free()[0], ad.n_free()[1], ad.bounds("pitch.v4") == (0.6, 1.6), ad.is_free("radius"),
           ad.is_free("hub_height"), all(ad.segments(c) == 0 for c in kept), ad.pick_note("skew") == "  (file)",
           ad.pick_note("pitch") == "", [r.kind for r in ad.curve_rows("skew")] == ["pinned"]],
          [17, 19, 1, 1, 0, 1, 1, 1, 1], 0)
    no_order = refused(ad.set_order, "skew", 6)
    msg1 = ad.set_segments("skew", 1)
    n1, rel1 = ad.n_free("skew"), BM.fit_report(ad.params, ad.design)["skew"][1]
    msg0 = ad.set_segments("skew", 0)
    rr = np.linspace(ad.design.r0, ad.design.r1, 501)
    check("a file's own curve: no order to set; 1 segment: fitted to the file, its 10 variables free; 'file' "
          "again: exactly the file's, no variables",
          [no_order, n1[0], n1[1], rel1 < 0.01, msg1.startswith("fitted to the parameter file"),
           msg0.startswith("as the parameter file has it"), ad.n_free("skew")[1],
           np.abs(ad.design.prop("skew")(rr) - ad.params.spline("skew")(rr)).max()], [1, 10, 10, 1, 1, 1, 0, 0], 1e-12)
    for n in ad.space.all_names:
        if n.startswith("chord."):
            ad.set_free(n, False)
    q = BM.read_params(ad.save_params(os.path.join(tmp, "vary_saved.dat")))
    ad2 = XBL.BladeAdapter(params=q.path)
    check("save_params writes the vary lines as the design stands (chord held: N; pitch Y with its range, the "
          "radius Y); loaded again: chord the saved file's own",
          [BM.vary_free(q) == ("pitch", "radius"), q.vary["pitch"].lo, q.vary["pitch"].hi, ad2.segments("chord"),
           ad2.n_free() == (9, 11)], [1, 0.6, 1.6, 0, 1], 0)


def test_control_original(tmp, table):
    print("\nblade_control.py with every vary line N (no CAD, no plot)")
    out = os.path.join(tmp, "control")
    src = open(os.path.join(HERE, "blade_control.py")).read()
    src = src.replace('if __name__ == "__main__":', f"PARAMS, OUT_DIR = {vary_file(tmp, 'control_n.dat', ALL_N, table)!r}"
                      f", {out!r}\nsettings.cad = settings.plot = False\n\nif __name__ == \"__main__\":", 1)
    script = os.path.join(tmp, "blade_control_n.py")
    with open(script, "w") as fh:
        fh.write(src)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE] + [os.environ.get("PYTHONPATH", "")]))
    run = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=tmp, env=env)
    files = sorted(os.listdir(out)) if os.path.isdir(out) else []
    if run.returncode:
        print(run.stdout + run.stderr)
    check("'nothing (every vary line N)': only the case 'original' (parameter file, XCAD points, design, set-up)",
          [run.returncode, "the original blade only" in run.stdout,
           files == [f"original_{f}" for f in ("design.json", "design_space.json", "params.dat", "xcad_m.dat")]],
          [0, 1, 1], 0)


def test_gui(tmp, table):
    print("\nthe window (xgeom_tool.py, blade)")
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:
        print(f"       skipped: {exc}")
        return
    import xgeom_tool as XT
    try:
        app = XT.App(root, "blade", params=msc_file(tmp, table))
        root.update()
        app.curve.set("chord")
        app.on_curve()
        app.set_value("chord.v4", app.ad.get("chord.v4") * 1.1, None)
        rows2 = len(app.rows)
        app.nseg.set(1)
        app.on_segments()
        app.nctrl.set(8)
        app.on_order()
        app.update_views()
        app.book.select(1)
        root.update()
        app.update_views()
        root.update()
        n3 = sum(len(c.get_segments()) for c in app.view3.col.values())
        check("the window: a slider, chord to 1 segment of 8 control points, the propeller tab with the hub",
              [rows2, app.ad.segments("chord"), app.ad.order("chord"), len(app.rows), n3],
              [3 + 8, 1, 7, 3 + 14, BM.SECTIONS * 5 + 14], 0)
        ok_bg = app.status.cget("background")
        app.sec["sections"].set(30)
        app.sec["tip_sections"].set(6)
        app.on_sections()
        app.update_views()
        root.update()
        n30 = sum(len(c.get_segments()) for c in app.view3.col.values())
        app.sec["tip_sections"].set(40)                       # more than fit: refused, the boxes reset
        app.on_sections()
        err_bg = app.status.cget("background")
        kept = app.sec["tip_sections"].get()
        check("Sections: 30 sections (6 in the tip band) in the propeller tab, 40 in the band refused",
              [app.ad.settings["sections"], n30, kept == "6"], [30, 30 * 5 + 14, 1], 0)
        app.ad.set_bounds("hub_height", 0.1, 2.0)
        app.set_value("hub_height", 0.2, None)                # too short for the blade root
        app.update_views()
        prob = app.status.cget("text")
        check("the status bar: green ok, red for an error and for a problem (a short hub: no CAD)",
              [ok_bg == XT.STATUS["ok"][0], err_bg == XT.STATUS["error"][0], prob.startswith("problem (no CAD"),
               "hub height" in prob, app.status.cget("background") == XT.STATUS["error"][0],
               app.status.winfo_ismapped()], [1] * 6, 0)
        app.nblades.set(7)
        app.on_blades()
        app.book.select(1)
        root.update()
        app.update_views()
        root.update()
        n7 = sum(len(c.get_segments()) for c in app.view3.col.values())
        app.nblades.set(8)                                    # refused: the box shows the Z in use
        app.on_blades()
        check("blades Z 7 from the box: 7 blades in the propeller tab; 8 refused (the box shows 7)",
              [app.ad.params.blades, n7, app.nblades.get() == "7"], [7, 30 * 7 + 14, 1], 0)
        app.ad.set("hub_height", 1.0)                          # no problem left
        app.ad.settings["clearance_mm"] = 500.0               # every gap is closer than that
        app.update_views()
        warn = app.status.cget("text")
        check("a warning in orange (blades interfere), the build still allowed",
              [app.status.cget("background") == XT.STATUS["warn"][0], warn.startswith("warning (CAD still possible)"),
               len(app.ad.preview()["problems"])], [1, 1, 0], 0)
        app.ad.settings["clearance_mm"] = BM.CLEARANCE_MM
        app.ad.load_params(vary_file(tmp, "vary_gui.dat", ["pitch Y", "chord Y"], table))   # as Load ... does
        app._title()
        app.after_change(structure=True)
        app.curve.set("skew")
        app.on_curve()
        root.update()
        texts = {n: rb.cget("text") for n, (rb, _) in app.picks.items()}
        seg0, rows0 = app.nseg.get(), len(app.rows)
        app.nseg.set(1)
        app.on_segments()
        rows1 = len(app.rows)
        app.nseg.set(0)
        app.on_segments()
        app.update_views()
        note = app.status.cget("text")
        root.update()
        check("a file with vary lines: '(file)' by the N curves; skew: the 'file' form, no variables; 1 segment: "
              "its 10; 'file' again (the status bar says so)",
              [texts["skew"].endswith("(file)"), texts["pitch"].endswith("(file)"), seg0, rows0, rows1, len(app.rows),
               note.startswith("skew: as the parameter file has it")], [1, 0, 0, 3, 13, 3, 1], 0)
    finally:
        root.destroy()


def test_cad(tmp, table):
    print("\nthe CAD (tip_surfaces_new, X_CAD_new; about two minutes)")
    import importlib.util
    if importlib.util.find_spec("OCC") is None:
        print("       skipped: no pythonocc-core")
        return
    p = BM.read_params(vary_file(tmp, "cad.dat", ["pitch Y", "chord Y"], table))
    design, _ = BM.fit_design(p)                           # pitch and chord 2 segments, the others the file's own
    design.hub_height = 0.8
    out = BM.write_cad(tmp, "cad", design, p)
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    rd = STEPControl_Reader()
    rd.ReadFile(out["cad_paths"]["step"])
    rd.TransferRoots()
    shape = rd.OneShape()
    n, ex = 0, TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        n += 1
        ex.Next()
    try:                                                   # the hub's length along x (mm)
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        box = Bnd_Box()
        brepbndlib.Add(shape, box)
        x0, _, _, x1, _, _ = box.Get()
    except ImportError:
        x0, x1 = -400.0, 400.0
    print(f"       grids {out['grid_seconds']:.0f} s, faces and IGES {out['cad_seconds']:.1f} s; x from {x0:.1f} "
          f"to {x1:.1f} mm")
    check("IGES and STEP (pitch, chord 2 segments, the others the file's; hub 0.8 m): 5 blade faces, hub sector, "
          "2 caps, valid; hub length",
          [os.path.exists(out["cad_paths"]["iges"]), n, BRepCheck_Analyzer(shape).IsValid(),
           not any(f.startswith("x_cad_") for f in os.listdir(tmp)), (x1 - x0) / 800.0], [1, 8, 1, 1, 1], 2e-3)


def main():
    print("blade self-check")
    table = os.path.join(HERE, "airfoil_data_fixed.csv")
    if "--section-table" in sys.argv:
        table = os.path.abspath(sys.argv[sys.argv.index("--section-table") + 1])
    assert RM.Bezier                                       # the curves are rudder_modify's Bezier
    with tempfile.TemporaryDirectory() as tmp:
        test_file(tmp)
        test_vary(tmp)
        test_msc_flags()
        test_sampling(tmp)
        test_curves(tmp)
        test_space(tmp)
        if test_geometry(tmp, table):
            test_adapter(tmp, table)
            test_adapter_vary(tmp, table)
            test_control_original(tmp, table)
            if "--gui" in sys.argv:
                test_gui(tmp, table)
            if "--cad" in sys.argv:
                test_cad(tmp, table)
    print()
    if _FAILED:
        print(f"{len(_FAILED)} check(s) FAILED: " + ", ".join(_FAILED))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
