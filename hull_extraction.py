"""
hull_extraction.py

Stations and hydrostatics of a hull with closed transverse sections (a
submerged body, one loop per section), from its CAD (STEP or IGES) or from its
equations (suboff.py). x runs along the hull axis from the nose to the tail, y
and z across it. Units: metres; the CAD in mm (as STEP files are written). A
CAD hull must lie along its x axis with the nose at the smallest x.

Stations: the hull cut by planes normal to x at n stations, cosine spacing
(dense at the nose and the tail; the end cuts 1e-6 L inside the hull). Each
gives its section loop, area, girth (perimeter), breadth (y extent) and depth
(z extent).

Hydrostatics (hydrostatics(), written by write_hydrostatics):
    length              L, nose to tail
    volume, wetted_surface, lcb
                        V, S and the centre of buoyancy from the nose: from the
                        solid (or the equations) when there is one, else from
                        the stations (V and its moment by the trapezoidal rule,
                        S by frusta between the stations). The stations' values
                        are also written (volume_stations, ...), as a check.
    lcb_mid             (LCB - L/2) / L, + aft of midship
    am, x_am            Am, the largest section area, and its station
    breadth, depth      B and T, the largest breadth and depth
    cp, cm, cb          V / (Am L); Am / (B T) at the largest section (pi/4 for
                        a circle); V / (L B T)
    pmb_start, pmb_end, pmb_length
                        the parallel middle body: the stations whose area is
                        within 1e-6 of Am (at a join as smooth as SUBOFF's,
                        that reaches about 5 mm past it); SUBOFF: its joins
    cp_fore, cp_aft, xc_fore, xc_aft, p_fore, p_aft
                        each half about midship, the quantities Lackenby's
                        variation works with: its prismatic coefficient (its
                        volume over Am L/2), its centroid from midship and its
                        part of the parallel middle body (both over L/2); from
                        the stations

Outputs (outputs/hull/ by default):
    <case>_stations.dat        per station: x, x/L, area, A/Am (the sectional area
                               curve), r = sqrt(A / pi), girth, breadth, depth
    <case>_hydrostatics.dat    the values above (SUBOFF: with the report's)
    <case>_check.png           the profile and the sectional area curve against x
    suboff.step                SUBOFF: the solid of revolution of its equations

    python hull_extraction.py --suboff        SUBOFF from its equations, and its STEP
    python hull_extraction.py hull.step       a hull's CAD (STEP or IGES)
        --stations N (default 401 for SUBOFF, 201 for a CAD), --case NAME, --out DIR

The CAD parts need pythonocc-core (OCC.Core) or cadquery-ocp (OCP); the cuts
use Rudder_geom_extraction's section code.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
from dataclasses import dataclass

import numpy as np

import xcad_loft as XL                                  # the OCC binding shim, STEP files

OUT_DIR = os.path.join("outputs", "hull")
CAD_SCALE = 1000.0                                      # mm per m: the CAD is in mm
PMB_TOL = 1.0e-6                                        # parallel middle body: area within this of Am

HYDRO = (                                               # key, unit, description (the hydrostatics file)
    ("length", "m", "L, nose to tail"),
    ("volume", "m^3", "V"),
    ("wetted_surface", "m^2", "S"),
    ("lcb", "m", "centre of buoyancy from the nose"),
    ("lcb_mid", "-", "(LCB - L/2) / L, + aft of midship"),
    ("am", "m^2", "Am, the largest section area"),
    ("x_am", "m", "its station (the middle of the parallel middle body)"),
    ("breadth", "m", "B, the largest breadth"),
    ("depth", "m", "T, the largest depth"),
    ("cp", "-", "prismatic coefficient V / (Am L)"),
    ("cm", "-", "maximum-section coefficient Am / (B T) at Am"),
    ("cb", "-", "block coefficient V / (L B T)"),
    ("l_over_b", "-", "L / B"),
    ("pmb_start", "m", "parallel middle body, start"),
    ("pmb_end", "m", "parallel middle body, end"),
    ("pmb_length", "m", "parallel middle body, length"),
    ("cp_fore", "-", "fore half (nose to midship): its volume / (Am L/2)"),
    ("cp_aft", "-", "aft half (midship to tail): its volume / (Am L/2)"),
    ("xc_fore", "-", "fore half: its centroid forward of midship / (L/2)"),
    ("xc_aft", "-", "aft half: its centroid aft of midship / (L/2)"),
    ("p_fore", "-", "fore half: its parallel middle body / (L/2)"),
    ("p_aft", "-", "aft half: its parallel middle body / (L/2)"),
    ("volume_stations", "m^3", "V from the stations (trapezoidal rule), a check"),
    ("wetted_surface_stations", "m^2", "S from the stations (frusta), a check"),
    ("lcb_stations", "m", "LCB from the stations, a check"),
    ("n_stations", "-", "number of stations"),
)


# --------------------------------------------------------------------------
# Stations
# --------------------------------------------------------------------------

@dataclass
class Stations:
    x: np.ndarray            # (n,) m from the nose, rising
    area: np.ndarray         # (n,) m^2
    girth: np.ndarray        # (n,) m
    breadth: np.ndarray      # (n,) m
    depth: np.ndarray        # (n,) m
    length: float            # L, m
    loops: list = None       # (k, 2) arrays of (y, z) in m, one per station; None: circles about the axis
    source: str = ""

    @classmethod
    def circles(cls, x, r, source="", length=None):
        """A body of revolution: circles of radius r (m) at x (m)."""
        x, r = np.asarray(x, dtype=float), np.asarray(r, dtype=float)
        return cls(x, np.pi * r ** 2, 2.0 * np.pi * r, 2.0 * r, 2.0 * r,
                   float(x[-1] - x[0]) if length is None else float(length), None, source)

    @classmethod
    def from_loops(cls, x, loops, length, source=""):
        """Sections given as closed loops, (k, 2) arrays of (y, z) in m."""
        rows = []
        for lp in loops:
            y, z = np.asarray(lp, dtype=float).T
            if len(y) < 3:
                rows.append((0.0, 0.0, 0.0, 0.0))
                continue
            area = 0.5 * abs(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))
            girth = float(np.sum(np.hypot(y - np.roll(y, -1), z - np.roll(z, -1))))
            rows.append((area, girth, float(np.ptp(y)), float(np.ptp(z))))
        a = np.array(rows)
        return cls(np.asarray(x, dtype=float), a[:, 0], a[:, 1], a[:, 2], a[:, 3], float(length),
                   [np.asarray(lp, dtype=float) for lp in loops], source)

    @property
    def r(self):
        """Equivalent radius sqrt(A / pi), m."""
        return np.sqrt(self.area / np.pi)


# --------------------------------------------------------------------------
# Hydrostatics
# --------------------------------------------------------------------------

def _integrals(x, a):
    """(volume, moment about x = 0) of the area curve a(x), trapezoidal rule."""
    dx = np.diff(x)
    return (0.5 * float(np.sum(dx * (a[1:] + a[:-1]))),
            0.5 * float(np.sum(dx * (x[1:] * a[1:] + x[:-1] * a[:-1]))))


def hydrostatics(st, exact=None, pmb_tol=PMB_TOL):
    """The hydrostatics of the stations (see the module docstring) as a dict.
    exact: values that replace the stations' own (volume, wetted_surface,
    lcb from the solid or the equations; pmb_start, pmb_end)."""
    exact = exact or {}
    x, a, length = st.x, st.area, st.length
    v_st, m_st = _integrals(x, a)
    s_st = float(np.sum(0.5 * (st.girth[1:] + st.girth[:-1]) * np.hypot(np.diff(x), np.diff(st.r))))
    i_am = int(np.argmax(a))
    am = float(a[i_am])
    lo = hi = i_am                                      # the run of stations at Am around the largest
    while lo > 0 and a[lo - 1] >= (1.0 - pmb_tol) * am:
        lo -= 1
    while hi < len(a) - 1 and a[hi + 1] >= (1.0 - pmb_tol) * am:
        hi += 1
    pmb_start, pmb_end = float(exact.get("pmb_start", x[lo])), float(exact.get("pmb_end", x[hi]))
    vol = float(exact.get("volume", v_st))
    lcb = float(exact.get("lcb", m_st / v_st))
    breadth, depth = float(st.breadth.max()), float(st.depth.max())
    mid, half = 0.5 * length, 0.5 * length
    a_mid = float(np.interp(mid, x, a))
    fore, aft = x < mid, x > mid
    vf, mf = _integrals(np.r_[x[fore], mid], np.r_[a[fore], a_mid])
    va, ma = _integrals(np.r_[mid, x[aft]], np.r_[a_mid, a[aft]])
    return {
        "length": length, "volume": vol, "wetted_surface": float(exact.get("wetted_surface", s_st)),
        "lcb": lcb, "lcb_mid": (lcb - mid) / length, "am": am, "x_am": 0.5 * (pmb_start + pmb_end),
        "breadth": breadth, "depth": depth, "cp": vol / (am * length),
        "cm": am / (st.breadth[i_am] * st.depth[i_am]), "cb": vol / (length * breadth * depth),
        "l_over_b": length / breadth, "pmb_start": pmb_start, "pmb_end": pmb_end,
        "pmb_length": pmb_end - pmb_start,
        "cp_fore": vf / (am * half), "cp_aft": va / (am * half),
        "xc_fore": (mid - mf / vf) / half, "xc_aft": (ma / va - mid) / half,
        "p_fore": max(0.0, min(mid, pmb_end) - pmb_start) / half,
        "p_aft": max(0.0, pmb_end - max(mid, pmb_start)) / half,
        "volume_stations": v_st, "wetted_surface_stations": s_st, "lcb_stations": m_st / v_st,
        "n_stations": len(x),
    }


# --------------------------------------------------------------------------
# CAD (OCC)
# --------------------------------------------------------------------------

def _rge():
    import Rudder_geom_extraction as RGE               # the CAD reader and the section code
    return RGE


def load_cad(path):
    """A STEP or IGES file as a shape (mm)."""
    return _rge().load_cad(path)


def solid_properties(shape, eps=1.0e-10):
    """(volume m^3, surface area m^2, centre of volume (3,) m) of a solid in mm,
    by adaptive integration to the relative error eps (on the SUBOFF solid,
    eps 1e-7 leaves the volume 5e-6 off at its poles; 1e-10 gives 3e-9)."""
    GProp_GProps = XL._occ("GProp", "GProp_GProps")
    mod = importlib.import_module(f"{XL._PKG}.BRepGProp")
    owner = getattr(mod, "brepgprop", None) or getattr(mod, "BRepGProp")
    out = []
    for kind in ("VolumeProperties", "SurfaceProperties"):
        props = GProp_GProps()
        fn = XL._static(owner, kind)
        try:
            fn(shape, props, float(eps))
        except TypeError:                                # a binding without the eps overload
            fn(shape, props)
        out.append(props)
    c = out[0].CentreOfMass()
    return (abs(out[0].Mass()) / CAD_SCALE ** 3, out[1].Mass() / CAD_SCALE ** 2,
            np.array([c.X(), c.Y(), c.Z()]) / CAD_SCALE)


def stations_from_cad(shape, n=201, inset=1.0e-6, loop_points=2000, source=""):
    """Stations cut from a hull's CAD (mm): n stations along x with cosine
    spacing between the ends of its bounding box, the end cuts inset L inside;
    each section one closed loop of about loop_points points."""
    RGE = _rge()
    gp_Pnt, gp_Dir, gp_Pln = XL._occ("gp", "gp_Pnt", "gp_Dir", "gp_Pln")
    lo, hi = RGE.bounding_box(shape)
    length = hi[0] - lo[0]
    t = np.clip(0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n)))), inset, 1.0 - inset)
    xs = lo[0] + length * t
    loops = []
    for x in xs:
        curves = RGE._plane_curves(shape, gp_Pln(gp_Pnt(float(x), 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)),
                                   f"at x = {x:.6f} mm")
        per = max(16, int(loop_points) // len(curves))
        loops.append(RGE._chain([RGE._curve_polyline(c, per) for c in curves])[:, 1:] / CAD_SCALE)
    return Stations.from_loops((xs - lo[0]) / CAD_SCALE, loops, length / CAD_SCALE, source)


def revolve(segments, tangents):
    """The solid of revolution about the x axis of a profile in m: segments
    from the nose (on the axis) to the tail (on the axis), (k, 2) arrays of
    (x, r) whose neighbours share their end point, and the tangents (dx, dr)
    at the two ends of each (suboff.profile gives both). Each segment is one
    edge, a cubic B-spline through its points with those end tangents (two
    points: a line), and so one face of the solid. The solid is in mm."""
    gp_Pnt, gp_Dir, gp_Vec, gp_Ax1 = XL._occ("gp", "gp_Pnt", "gp_Dir", "gp_Vec", "gp_Ax1")
    TColgp_HArray1OfPnt = XL._occ("TColgp", "TColgp_HArray1OfPnt")
    GeomAPI_Interpolate = XL._occ("GeomAPI", "GeomAPI_Interpolate")
    MakeEdge, MakeWire, MakeFace = XL._occ("BRepBuilderAPI", "BRepBuilderAPI_MakeEdge", "BRepBuilderAPI_MakeWire",
                                           "BRepBuilderAPI_MakeFace")
    MakeRevol = XL._occ("BRepPrimAPI", "BRepPrimAPI_MakeRevol")

    def pnt(p):
        return gp_Pnt(float(p[0]) * CAD_SCALE, 0.0, float(p[1]) * CAD_SCALE)

    def vec(t):
        t = np.asarray(t, dtype=float) / np.hypot(*t)
        return gp_Vec(float(t[0]), 0.0, float(t[1]))

    wire = MakeWire()
    for pts, (t0, t1) in zip(segments, tangents):
        pts = np.asarray(pts, dtype=float)
        if len(pts) == 2:
            wire.Add(MakeEdge(pnt(pts[0]), pnt(pts[1])).Edge())
            continue
        arr = TColgp_HArray1OfPnt(1, len(pts))
        for i, p in enumerate(pts):
            arr.SetValue(i + 1, pnt(p))
        it = GeomAPI_Interpolate(arr, False, 1.0e-9)
        it.Load(vec(t0), vec(t1), False)             # unit tangents: the parameter is the chord length
        it.Perform()
        if not it.IsDone():
            raise RuntimeError("the B-spline through a profile segment failed")
        wire.Add(MakeEdge(it.Curve()).Edge())
    wire.Add(MakeEdge(pnt(segments[-1][-1]), pnt(segments[0][0])).Edge())      # back along the axis
    face = MakeFace(wire.Wire(), True).Face()
    return MakeRevol(face, gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 0.0, 0.0)), 2.0 * math.pi).Shape()


def is_valid(shape):
    """BRepCheck_Analyzer's verdict."""
    return bool(XL._occ("BRepCheck", "BRepCheck_Analyzer")(shape).IsValid())


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

def write_stations(path, st, hs, title):
    """The station table: x, x/L, area, A/Am, r, girth, breadth, depth."""
    tab = np.column_stack((st.x, st.x / st.length, st.area, st.area / hs["am"], st.r, st.girth, st.breadth,
                           st.depth))
    head = (f"{title}: stations ({st.source})\n"
            "x from the nose; A/Am against x/L is the sectional area curve; r = sqrt(A / pi)\n"
            "x_m  x/L  area_m2  A/Am  r_m  girth_m  breadth_m  depth_m")
    np.savetxt(path, tab, fmt="%.9f", header=head)
    return path


def write_hydrostatics(path, hs, title, notes=()):
    """The hydrostatics, one value per line: name, value, unit, # description."""
    lines = [f"# {title}: hydrostatics"] + [f"# {n}" for n in notes]
    lines.append(f"# {'name':<24s} {'value':>16s}  {'unit':<5s}  description")
    for key, unit, desc in HYDRO:
        v = hs[key]
        val = f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.10g}"
        lines.append(f"{key:<26s} {val:>16s}  {unit:<5s}  # {desc}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def read_hydrostatics(path):
    """The values of a hydrostatics file as a dict."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                key, val = line.split()[:2]
                out[key] = float(val)
    return out


def plot_check(path, st, hs, title, joins=()):
    """The profile (equivalent radius) and the sectional area curve against x."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, 7))
    FigureCanvasAgg(fig)
    ax1, ax2 = fig.subplots(2, 1)
    length = st.length
    for ax, xs, ys, xl, yl in ((ax1, st.x, st.r, "x (m)", "r = sqrt(A / pi) (m)"),
                               (ax2, st.x / length, st.area / hs["am"], "x / L", "A / Am")):
        f = 1.0 if ax is ax1 else 1.0 / length
        ax.axvspan(hs["pmb_start"] * f, hs["pmb_end"] * f, color="#eeeeea", label="parallel middle body")
        for j in joins[1:-1]:
            ax.axvline(j * f, color="#9a9994", lw=0.8, ls=":")
        ax.plot(xs, ys, "-", color="#2a78d6", lw=1.6)
        ax.axvline(hs["lcb"] * f, color="#d6562a", lw=1.0, label="LCB")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, color="#e4e3df")
    ax1.legend(fontsize=8, loc="lower center")
    fig.suptitle(f"{title}: L {length:.4f} m, V {hs['volume']:.5f} m^3, S {hs['wetted_surface']:.4f} m^2, "
                 f"Cp {hs['cp']:.4f}, LCB {hs['lcb'] / length:.4f} L from the nose ({len(st.x)} stations)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def run_suboff(out_dir=OUT_DIR, n=401, cad=True, plot=True, case="suboff"):
    """SUBOFF from its equations: the files and, with OCC, its STEP. Returns
    (hydrostatics, written files, a line on the CAD)."""
    import suboff
    x, r = suboff.stations(n)
    st = Stations.circles(x, r, suboff.SOURCE, suboff.LENGTH)
    hs = hydrostatics(st, exact=suboff.exact_hydrostatics())
    os.makedirs(out_dir, exist_ok=True)
    title = "DARPA SUBOFF bare hull"
    ref = suboff.REFERENCE
    notes = [f"source: {suboff.SOURCE}",
             f"V, S and LCB by quadrature of the equations; the halves from the {len(x)} stations",
             f"the report: volume {ref['volume']} m^3, wetted surface {ref['wetted_surface']} m^2"]
    files = [write_stations(os.path.join(out_dir, f"{case}_stations.dat"), st, hs, title),
             write_hydrostatics(os.path.join(out_dir, f"{case}_hydrostatics.dat"), hs, title, notes)]
    if plot:
        files.append(plot_check(os.path.join(out_dir, f"{case}_check.png"), st, hs, title, suboff.JOINS))
    note = "no STEP (no OCC binding)" if cad else "no STEP"
    if cad and XL.occ_available():
        solid = revolve(*suboff.profile())
        vol, area, c = solid_properties(solid)
        files.append(XL.write_step(solid, os.path.join(out_dir, f"{case}.step")))
        note = (f"STEP: valid {is_valid(solid)}, volume {vol:.8f} m^3 ({vol / hs['volume'] - 1:+.1e}), "
                f"wetted surface {area:.7f} m^2 ({area / hs['wetted_surface'] - 1:+.1e}), "
                f"LCB {c[0]:.7f} m ({(c[0] - hs['lcb']) * 1e6:+.2f} um)")
    return hs, files, note


def run_cad(path, out_dir=OUT_DIR, n=201, case=None, plot=True):
    """A hull's CAD: stations and hydrostatics (V, S and LCB from the solid).
    Returns (hydrostatics, written files)."""
    case = case or os.path.splitext(os.path.basename(path))[0]
    shape = load_cad(path)
    vol, area, c = solid_properties(shape)
    lo = _rge().bounding_box(shape)[0] / CAD_SCALE
    st = stations_from_cad(shape, n, source=os.path.basename(path))
    hs = hydrostatics(st, exact={"volume": vol, "wetted_surface": area, "lcb": c[0] - lo[0]})
    os.makedirs(out_dir, exist_ok=True)
    notes = [f"source: {path}", f"V, S and LCB from the solid; the rest from {len(st.x)} stations cut from it"]
    files = [write_stations(os.path.join(out_dir, f"{case}_stations.dat"), st, hs, case),
             write_hydrostatics(os.path.join(out_dir, f"{case}_hydrostatics.dat"), hs, case, notes)]
    if plot:
        files.append(plot_check(os.path.join(out_dir, f"{case}_check.png"), st, hs, case))
    return hs, files


def main():
    ap = argparse.ArgumentParser(description="Stations and hydrostatics of a hull.")
    ap.add_argument("cad", nargs="?", help="the hull's CAD (STEP or IGES), along x, nose at the smallest x")
    ap.add_argument("--suboff", action="store_true", help="the DARPA SUBOFF bare hull from its equations")
    ap.add_argument("--stations", type=int, default=None, help="number of stations (401 SUBOFF, 201 CAD)")
    ap.add_argument("--case", default=None, help="case name for the files (default: suboff, or the file's)")
    ap.add_argument("--out", default=OUT_DIR, help=f"output folder (default {OUT_DIR})")
    ap.add_argument("--no-cad", action="store_true", help="SUBOFF: no STEP")
    args = ap.parse_args()
    if args.suboff:
        hs, files, note = run_suboff(args.out, args.stations or 401, not args.no_cad, case=args.case or "suboff")
    elif args.cad:
        hs, files = run_cad(args.cad, args.out, args.stations or 201, args.case)
        note = ""
    else:
        ap.error("give a CAD file or --suboff")
    print(f"L {hs['length']:.6f} m, V {hs['volume']:.8f} m^3, S {hs['wetted_surface']:.7f} m^2, "
          f"LCB {hs['lcb']:.6f} m ({hs['lcb'] / hs['length']:.5f} L), Cp {hs['cp']:.5f}, Cb {hs['cb']:.5f}, "
          f"Cm {hs['cm']:.5f}, parallel middle body {hs['pmb_start']:.4f} .. {hs['pmb_end']:.4f} m")
    if note:
        print(note)
    for f in files:
        print("  " + str(f))


if __name__ == "__main__":
    main()
