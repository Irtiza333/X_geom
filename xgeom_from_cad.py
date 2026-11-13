"""
xgeom_from_cad.py

A set-up from a CAD file (13 Nov 2026, the user's request): the design tool's
"From CAD ..." button picks a STEP or IGES file, the extraction code of the
component reads its geometry, the parameters are fitted and a set-up is saved,
and the tool's sliders show where the parameters of the starting geometry lie,
so bounds can be set about them. For the blade, the hull and the rudder, which
have extraction codes (blade_extraction.py, hull_extraction.py,
Rudder_geom_extraction.py); the other components later.

The fit (best_forms): a distribution can be captured by more than one form
(n or n + 1 control points, 1 or 2 segments, a CST curve of order n or n + 1),
so the form with the least error is taken: the candidates are fitted and the
largest difference from the extracted data, as a fraction of its range, is
compared; a candidate whose error is not FIT_TIE (0.1 % of the range) below the
best cheaper one does not earn its extra control points, so the fewest control
points among forms that fit alike win. The user can then change the number of
points or of segments in the tool, which refits the curve and shows the error
(as it did already: "fitted to the parameter file (largest difference ...)").

    blade   blade_extraction.extract writes <case>.txt (the six distributions as
            tables, the reference data) and the section files in out_dir; the
            curves are fitted (2 segments, or 1 segment of 3 to MAX_POINTS
            control points) and the set-up with them is written as
            <case>_setup.txt (blade_setup: the tool's adapter, the variables of
            blade_control.FREE free, the others held)
    hull    hull_extraction's stations of the CAD (hull_extraction.stations_
            from_cad; closed transverse sections) give the station table
            <case>_params.dat (x, the half-height r and half-width r' from the
            sections' depth and breadth, axisymmetric when alike; the nose, the
            middle body and the tail from the sectional area curve's parallel
            middle body, cap 0) and the set-up <case>.txt with each CST curve of
            the order (3 to MAX_ORDER) that fits best (hull_setup: the tool's
            adapter, the lengths and curves free as hull_modify.DEFAULT_FREE)
    rudder  Rudder_geom_extraction.run on the CAD into <out_dir>/<case>_
            extraction (its default run: 200 horizontal sections, the tilted
            tip section, the cap loops), then rudder_modify.fit_section_table on
            every horizontal section (the per-section quartic fits, about 0.4 s
            each), then the original shape fitted with the law orders (2 to
            MAX_LAW_ORDER) that fit best and the set-up <out_dir>/<case>_design_
            space.json (xgeom_rudder.RudderAdapter on those files; the design
            the baseline, which is the rudder itself; rudder_modify.
            load_design_space and the tool's Load ... read it)

    from_cad(kind, cad_path, out_dir, case, progress)   -> dict: "setup", "file", "forms", "report"
    best_blade_forms, best_hull_orders, best_rudder_orders   the chosen forms with their errors
    blade_setup, hull_setup                               the set-up of an extracted table
    best_wing_forms, wing_setup                           the same for a planar rudder, fin or wing's table
                                                          (wing_modify.WingParams; bb2/bb2_aft_setup.py
                                                          extracts BB2's stern plane into one)

Command line (the file a STEP or IGES; for the blade an extracted <case>.txt, for
the hull a <case>_params.dat: the set-up only; written next to the file):

    python xgeom_from_cad.py rudder Wind_Tunnel_Rudder.stp
    python xgeom_from_cad.py blade bb2/outputs/bb2_prop.txt
    python xgeom_from_cad.py hull bb2/outputs/bb2_hull_params.dat --case bb2_hull
"""

from __future__ import annotations

import glob
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MAX_POINTS = 8          # a 1-segment Bezier (blade): up to this many control points
MAX_ORDER = 8           # the hull's CST curves: orders 3 to this
MAX_LAW_ORDER = 6       # the rudder's laws (the original shape's fits): orders 2 to this
FIT_TIE = 1.0e-3        # of the range: a form must beat the best cheaper one by this to earn its points
KINDS = ("blade", "hull", "rudder")


def case_name(cad_path):
    """A case name from a CAD file's name: letters, digits and underscores."""
    name = os.path.splitext(os.path.basename(cad_path))[0]
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "cad"


def pick(candidates):
    """The form with the least error among candidates [(points, form, error)], the fewest points when a
    costlier form is not FIT_TIE better than the best cheaper one: (form, error)."""
    best = None
    for points, form, err in sorted(candidates, key=lambda c: c[0]):
        if best is None or err < best[1] - FIT_TIE:
            best = (form, err)
    return best


# --------------------------------------------------------------------------
# The blade
# --------------------------------------------------------------------------

def best_blade_forms(params, max_points=MAX_POINTS):
    """{curve: ((segments, order), error)} for a BladeParams: 2 segments, or 1 segment of 3 .. max_points
    control points, the least error (pick) as a fraction of the distribution's range."""
    import blade_modify as BM
    r, vals = params.rows()
    out = {}
    for c in BM.CURVES:
        v = vals[c]
        span = max(float(np.ptp(v)), 1e-12)
        cands = []
        for n in range(3, max_points + 1):
            cu, _ = BM.fit_form(r, v, r[0], r[-1], 1, n - 1)
            cands.append((n, (1, n - 1), float(np.abs(BM.curve_at(cu, r[0], r[-1], r) - v).max() / span)))
        cu, _ = BM.fit_form(r, v, r[0], r[-1], 2)
        cands.append((7, (2, None), float(np.abs(BM.curve_at(cu, r[0], r[-1], r) - v).max() / span)))
        out[c] = pick(cands)
    return out


def blade_setup(component, setup_path):
    """The set-up of a blade's component file (an extracted <case>.txt): as the tool starts from the file
    (xgeom_blade.BladeAdapter), each distribution in the form that fits it best (best_blade_forms), the
    variables of blade_control.FREE free; written as a component file. Returns (path, forms)."""
    import xgeom_blade as XB
    ad = XB.BladeAdapter(params=component)
    forms = best_blade_forms(ad.params)
    ad.set_forms({c: f[0] for c, f in forms.items()})
    ad.notes.append(f"the set-up: each distribution in the form that fits {os.path.basename(component)} best "
                    f"(xgeom_from_cad.py); the variables of {', '.join(ad._free0())} free")
    return ad.save_file(setup_path), forms


def blade_from_cad(cad_path, out_dir, case=None, progress=print, **extract_kw):
    """The blade's set-up from its CAD (the module docstring). extract_kw: blade_extraction.extract's options
    (diameter_mm, x0, hub, tip_pitch, stations_n, ...)."""
    import blade_extraction as BX
    case = case or case_name(cad_path)
    t0 = time.time()
    progress(f"extracting the blade from {os.path.basename(cad_path)}")
    res = BX.extract(cad_path, case, out_dir, log=lambda *a, **k: None, **extract_kw)
    progress("fitting the distributions")
    setup, forms = blade_setup(res["component"], os.path.join(out_dir, f"{case}_setup.txt"))
    report = [f"{c}: " + ("2 segments" if f[0][0] == 2 else f"1 segment, {f[0][1] + 1} points")
              + f" ({100 * f[1]:.2f} % of its range)" for c, f in forms.items()]
    report.append(f"the blade rebuilt from the tables lies within {res['deviation'][0, :, 0].max():.2f} mm of the "
                  f"cuts; {len(res['cuts'])} cuts, {res['blades']} blades, R {res['R_mm']:.1f} mm")
    progress(f"done in {time.time() - t0:.0f} s")
    return {"kind": "blade", "setup": setup, "file": res["component"], "forms": forms, "report": report,
            "extraction": res, "seconds": time.time() - t0}


# --------------------------------------------------------------------------
# The hull
# --------------------------------------------------------------------------

def best_hull_orders(params, max_order=MAX_ORDER):
    """{curve: (order, error)} for a HullParams: CST orders 3 .. max_order, the error the largest difference
    from the stations over the largest radius."""
    import hull_modify as HM
    scale = max(float(params.r.max()), 1e-12)
    names = list(HM.fit_report(params, HM.fit_design(params)))
    reports = {}
    for n in range(3, max_order + 1):
        d = HM.fit_design(params, orders={name: n for name in names})
        reports[n] = {k: v / scale for k, v in HM.fit_report(params, d).items()}
    return {name: pick([(n, n, reports[n][name]) for n in reports]) for name in names}


def hull_params_from_cad(cad_path, n=201, case=None):
    """A HullParams from a hull's CAD (hull_extraction: the stations' depth and breadth, the parallel middle
    body of the sectional area curve), and the hydrostatics."""
    import hull_extraction as HX
    import hull_modify as HM
    case = case or case_name(cad_path)
    shape = HX.load_cad(cad_path)
    st = HX.stations_from_cad(shape, n, source=os.path.basename(cad_path))
    try:
        vol, area, c = HX.solid_properties(shape)
        lo = HX._rge().bounding_box(shape)[0] / HX.CAD_SCALE
        hs = HX.hydrostatics(st, exact={"volume": vol, "wetted_surface": area, "lcb": c[0] - lo[0]})
    except Exception:
        hs = HX.hydrostatics(st)
    x = st.x - st.x[0]
    r, rp = 0.5 * st.depth, 0.5 * st.breadth
    axi = bool(np.allclose(r, rp, atol=1e-4 * max(r.max(), 1e-12)))
    nose, end = float(hs["pmb_start"] - st.x[0]), float(hs["pmb_end"] - st.x[0])
    middle = max(end - nose, 0.0)
    tail = float(x[-1] - nose - middle)
    if hs.get("pmb_length", 0.0) <= 0.0:                     # no middle body: the nose ends at the largest section
        k = int(np.argmax(st.area))
        nose, middle, tail = float(x[k]), 0.0, float(x[-1] - x[k])
    keep = np.ones(len(x), dtype=bool)
    keep[1:] = np.diff(x) > 0
    params = HM.HullParams(nose, middle, tail, 0.0, x[keep], r[keep], (r if axi else rp)[keep], axi, "",
                           [f"stations of {os.path.basename(cad_path)} (hull_extraction, {len(x)} cuts): r the "
                            f"half-height, r' the half-width; nose, middle body and tail from the sectional area "
                            f"curve's parallel middle body"])
    return params, hs


def hull_setup(params_path, setup_path, source=None):
    """The set-up of a hull's parameter file: as the tool starts from the file (xgeom_hull.HullAdapter: the
    lengths and curves free, the radii held), each curve of the CST order that fits the stations best
    (best_hull_orders); written as a component file. Returns (path, orders)."""
    import hull_modify as HM
    import xgeom_hull as XH
    orders = best_hull_orders(HM.read_params(params_path))
    ad = XH.HullAdapter(params_path)
    for k, v in orders.items():
        ad.set_order(k, v[0])
    ad.notes.append(f"the set-up from {source or os.path.basename(params_path)}: each curve of the CST order "
                    f"that fits the stations best (xgeom_from_cad.py)")
    return ad.save_file(setup_path), orders


def hull_from_cad(cad_path, out_dir, case=None, progress=print, n=201):
    """The hull's set-up from its CAD (the module docstring)."""
    import hull_modify as HM
    case = case or case_name(cad_path)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    progress(f"cutting {n} stations of {os.path.basename(cad_path)}")
    params, hs = hull_params_from_cad(cad_path, n, case)
    path = HM.write_params(os.path.join(out_dir, f"{case}_params.dat"), params)
    params = HM.read_params(path)
    progress("fitting the curves")
    setup, orders = hull_setup(path, os.path.join(out_dir, f"{case}.txt"), os.path.basename(cad_path))
    scale = float(params.r.max())
    report = [f"{k}: order {v[0]} ({100 * v[1]:.2f} % of r*, {1000 * v[1] * scale:.2f} mm)" for k, v in orders.items()]
    report.append(f"L {hs['length']:.4g} m, nose {params.nose:.4g}, middle body {params.middle:.4g}, tail "
                  f"{params.tail:.4g} m; r* {scale:.4g} m" + ("" if params.axisymmetric else ", elliptical sections"))
    progress(f"done in {time.time() - t0:.0f} s")
    return {"kind": "hull", "setup": setup, "file": path, "forms": orders, "report": report, "hydrostatics": hs,
            "seconds": time.time() - t0}


# --------------------------------------------------------------------------
# A planar rudder, fin or wing (wing_modify)
# --------------------------------------------------------------------------

WING_FLOORS = {"chord": 1.0, "sweep": 1.0, "rake": 1.0, "pitch": 0.01, "thickness": 1.0e-4, "camber": 1.0e-4}
CHANGE_FLOOR = 1.0e-5   # a section change (dt, dc), a fraction of the chord


def best_wing_forms(params, max_points=MAX_POINTS):
    """{curve: ((segments, order), error)} for a wing_modify.WingParams (the six distributions and the section
    changes): 1 segment of 2 .. max_points control points, or 2 segments, the least error (pick) as a fraction
    of the distribution's range along the span, the range at least WING_FLOORS (mm, deg, t/c, f/c) or
    CHANGE_FLOOR (a change), since a distribution that hardly changes has a range of the extraction's scatter."""
    import wing_modify as WM
    eta = WM.fit_stations(params)
    out = {}
    for c in params.names:
        v = params.spline(c)(eta)
        span = max(float(np.ptp(v)), WING_FLOORS.get(c, CHANGE_FLOOR))
        cands = []
        for n in range(2, max_points + 1):
            cu = WM.fit_curve_to(eta, v, c, 1, n - 1)
            cands.append((n, (1, n - 1), float(np.abs(WM.curve_at(cu, 0.0, 1.0, eta) - v).max() / span)))
        cu = WM.fit_curve_to(eta, v, c, 2)
        cands.append((7, (2, None), float(np.abs(WM.curve_at(cu, 0.0, 1.0, eta) - v).max() / span)))
        out[c] = pick(cands)
    return out


def wing_setup(params, setup_path, kind=None, name=None, notes=(), free=None):
    """The set-up of a planar component's table (a WingParams with its section and change columns): each
    distribution in the form that fits it best (best_wing_forms), the variables of free (default
    xgeom_wing.FREE, as the tool starts a file) free, the others held; written as a component file of kind
    (default the table's). Returns (path, forms)."""
    import wing_modify as WM
    forms = best_wing_forms(params)
    design, _ = WM.fit_design(params, orders={c: f[0][1] for c, f in forms.items() if f[0][0] == 1},
                              segments={c: f[0][0] for c, f in forms.items()}, changes=params.changes)
    if free is None:
        import xgeom_wing as XW
        free = XW.FREE
    space = WM.WingSpace(params, design, free=tuple(free))
    notes = list(notes) + [f"the set-up: each distribution in the form that fits the table best (xgeom_from_cad.py); "
                           f"the variables of {', '.join(free)} free"]
    path = WM.write_component(setup_path, params, design, space, {"sections": WM.SECTIONS, "half": False}, name,
                              notes, kind)
    return path, forms


# --------------------------------------------------------------------------
# The rudder (modified below H)
# --------------------------------------------------------------------------

LAW_FLOOR = 0.01        # the rudder's laws: the error over the law's range, the range at least this (a chord
                        # fraction; for the LE this fraction of z_top in mm), since a law that hardly changes
                        # along the span has a range of the fits' scatter only


def best_rudder_orders(corner_points, section_table, degree=None, max_order=MAX_LAW_ORDER):
    """{law: (order, error)} for the original shape's fits (rudder_modify.fit_original): orders 2 .. max_order
    per law, the error the largest residual over the law's range (at least LAW_FLOOR: the section coordinates
    are chord fractions, the LE is in mm and takes LAW_FLOOR of z_top)."""
    import rudder_modify as RM
    degree = RM.BASE_DEGREE if degree is None else int(degree)
    fits = {}
    for n in range(2, max_order + 1):
        orig = RM.fit_original(corner_points, section_table, None, {c: n for c in RM.curve_names(degree)},
                               degree=degree)
        fits[n] = orig
    out = {}
    for c in fits[2].names:
        data = fits[2].data[c][1]
        floor = LAW_FLOOR * (fits[2].z_top if c == "LE" else 1.0)
        span = max(float(np.ptp(data)), floor)
        out[c] = pick([(n, n, float(np.abs(fits[n].residuals[c]).max()) / span) for n in fits])
    return out


def rudder_files(folder):
    """The files of an extraction folder that the rudder adapter takes: corner points, the section table (the
    per-section fits), the XCAD file with the cap, the stack; relative to the X_geom folder when inside it (a
    set-up that names them then holds on another machine), else absolute."""
    def one(pattern, what):
        hits = sorted(glob.glob(os.path.join(folder, pattern)))
        if not hits:
            raise FileNotFoundError(f"{folder}: no {what} ({pattern})")
        path = os.path.abspath(hits[-1])
        inside = os.path.commonpath([path, HERE]) == HERE
        return os.path.relpath(path, HERE).replace(os.sep, "/") if inside else path
    return {"corner_points": one("corner_points.dat", "corner points"),
            "section_table": one("thickness_params_*_info.dat", "section table"),
            "xcad_original": one("sections_xcad_with_cap_*_m.dat", "XCAD file with the cap"),
            "stack": one("sections_stack_*_raw_mm.dat", "section stack")}


def rudder_from_cad(cad_path, out_dir, case=None, progress=print, num_sec=None):
    """The rudder's set-up from its CAD (the module docstring): the extraction into <out_dir>/<case>_extraction,
    the section fits, the original fitted with the best law orders, the set-up JSON."""
    import Rudder_geom_extraction as RGE
    import rudder_modify as RM
    import xgeom_rudder as XR
    case = case or case_name(cad_path)
    folder = os.path.join(out_dir, f"{case}_extraction")
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    progress(f"extracting {os.path.basename(cad_path)}: the sections, the tip and the cap")
    kw = {} if num_sec is None else {"num_sec": int(num_sec)}
    RGE.run(cad_path, out_dir=folder, plot=False, **kw)
    stack = rudder_files_partial(folder, "sections_stack_*_raw_mm.dat")
    secs = [s for s in RM.read_stack(stack) if s.tilt_deg == 0.0]
    top = max(s.y for s in secs)
    tag = re.search(r"sections_stack_(y[\d.]+_y[\d.]+)_raw_mm", os.path.basename(stack)).group(1)
    txt = os.path.join(folder, f"thickness_params_{tag}.txt")
    info = os.path.join(folder, f"thickness_params_{tag}_info.dat")
    t1 = time.time()
    progress(f"fitting the {len(secs)} horizontal sections (about 0.4 s each)")
    RM.fit_section_table(stack, top + 1e-6, txt, info, verbose=False)
    files = rudder_files(folder)
    progress("fitting the original shape's laws")
    orders = best_rudder_orders(files["corner_points"], files["section_table"])
    ad = XR.RudderAdapter(files=files, fit_orders={c: v[0] for c, v in orders.items()}, out_dir=out_dir)
    setup = ad.save_space(os.path.join(out_dir, f"{case}_design_space.json"))
    report = [f"{c}: order {v[0]} ({100 * v[1]:.2f} % of its range)" for c, v in orders.items()]
    report.append(f"{len(secs)} sections to z_top {ad.orig.z_top:.2f} mm, full height {ad.span:.2f} mm; the "
                  f"section fits {time.time() - t1:.0f} s")
    progress(f"done in {time.time() - t0:.0f} s")
    return {"kind": "rudder", "setup": setup, "file": folder, "files": files, "forms": orders, "report": report,
            "seconds": time.time() - t0}


def rudder_files_partial(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    if not hits:
        raise FileNotFoundError(f"{folder}: no {pattern}")
    return hits[-1]


FROM_CAD = {"blade": blade_from_cad, "hull": hull_from_cad, "rudder": rudder_from_cad}


def from_cad(kind, cad_path, out_dir, case=None, progress=print, **kw):
    """The set-up of a component of the kind from its CAD: FROM_CAD[kind]."""
    if kind not in FROM_CAD:
        raise ValueError(f"no extraction for a {kind} yet (one of {', '.join(FROM_CAD)})")
    return FROM_CAD[kind](cad_path, out_dir, case, progress, **kw)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="A set-up from a CAD file, each curve in the form that fits best "
                                             "(the module docstring).")
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("file", help="a STEP or IGES file; for the blade an extracted <case>.txt, for the hull a "
                                 "<case>_params.dat (the set-up only)")
    ap.add_argument("--out", help="the folder to write in (default: the file's)")
    ap.add_argument("--case", help="the case name (default: from the file's name)")
    a = ap.parse_args(argv)
    a.file = os.path.abspath(a.file)
    out = os.path.abspath(a.out or os.path.dirname(a.file))
    os.makedirs(out, exist_ok=True)
    os.chdir(HERE)                                   # the adapters' files are relative to this folder
    table = a.file.lower().endswith((".txt", ".dat"))
    if table and a.kind == "rudder":
        ap.error("the rudder's set-up needs its CAD (the extraction writes the files it is fitted to)")
    if table:
        case = a.case or re.sub(r"_params$", "", case_name(a.file))
        if a.kind == "blade":
            setup, forms = blade_setup(a.file, os.path.join(out, f"{case}_setup.txt"))
            lines = [f"{c}: " + ("2 segments" if f[0][0] == 2 else f"1 segment, {f[0][1] + 1} points")
                     + f" ({100 * f[1]:.2f} % of its range)" for c, f in forms.items()]
        else:
            setup, forms = hull_setup(a.file, os.path.join(out, f"{case}.txt"))
            lines = [f"{k}: order {v[0]} ({100 * v[1]:.2f} % of r*)" for k, v in forms.items()]
    else:
        res = from_cad(a.kind, a.file, out, a.case)
        setup, lines = res["setup"], res["report"]
    print(setup)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
