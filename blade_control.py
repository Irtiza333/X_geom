"""
blade_control.py

A propeller blade from a parameter file (blade_modify.py), built with the MSc
blade code. Set the file, the settings and the designs below, then run

    python blade_control.py

1. PARAMS is read (the radial table and the globals, see blade_modify) and each
   distribution (pitch, chord, thickness, camber, skew, rake) is fitted with a
   Bezier curve over r/R of 1 or 2 segments (SEGMENTS): the starting design,
   with the blade radius D/2, the hub radius root r/R x D/2 and the file's hub
   height (hub_new.DEFAULT_HUB_HEIGHT when it gives none).
2. Each case in cases() is a design: change the curves (control values,
   positions, weights) or the blade's radius, hub radius and hub height to
   change the blade.
3. For each case, OUT_DIR gets
     <case>_params.dat       the design as a parameter file (read_params and
                             the design tool load it)
     <case>_xcad_m.dat       the blade points for XCAD (para.py's sections, 53
                             points each, at settings.sections stations)
     <case>_design.json      the curves, their difference from the file, the checks
     <case>_check.png        the distributions against the file
   and with cad on: <case>.iges (the DRDC five surfaces and the hub sector,
   X_CAD_new.py) and <case>.step (the same shape).
"""

import importlib.util
from dataclasses import dataclass

import blade_modify as BM

# ---------------- FILES ----------------------------------------------------------
PARAMS = "msc_blade_params.dat"        # the blade's parameter file
OUT_DIR = "outputs/blade"


@dataclass # LEAVE THESE AS DEFAULTS
class BladeSettings:
    clearance_mm: float = 25.0     # designs whose blades come closer than this are rejected, mm (x_blade_new's
                                   # check at its own stations, made for the MSc blade, D = 1.4 m)
    sections: int = 56             # sections in <case>_xcad_m.dat, root to tip
    tip_band: float = 0.93         # the last tip_sections lie from this r/R to the tip, closer together towards
    tip_sections: int = 14         # it; the others evenly spaced from the root (x_blade_new: 56, 0.93, 14)
    hub: bool = True               # the hub sector in the CAD (hub_new.py; its height is the design's)
    cad: bool = True               # write <case>.iges and <case>.step (needs pythonocc-core)
    plot: bool = True              # write <case>_check.png

settings = BladeSettings( # SET PARAMETERS HERE
    clearance_mm = 25.0,
    sections = 56,
    cad = True,
)

# ---------------- DESIGN ---------------------------------------------------------
# segments of each distribution's curve ("pitch", "chord", "thickness", "camber", "skew", "rake"; not named: 2):
#   2  para_control_bez_updated's form (x_blade): rational cubic segments P1-P4 and P4-P7, P2 = P3 and P5 = P6
#      at P4's value (zero slope at P4); variables v1, d1, w23, u4, v4, d2, w56, v7
#   1  one Bezier C0 (root) .. Cm (tip); variables d1 .. d(m-1), v0 .. vm
SEGMENTS = {}
# 1-segment curves: number of control points - 1 (not named: 5)
ORDERS = {}
# the distributions (and globals "radius", "hub_radius", "hub_height") the design tool starts with as design
# variables
FREE = ("pitch", "chord")


def cases(design):
    """name -> BladeDesign. design.radius, .hub_radius and .hub_height (m) are
    the blade's; design.curves[name] its distributions: .scaled(f) multiplies a
    curve's values; a 2-segment curve has the fields v1, d1, w23, u4, v4, d2,
    w56, v7, a 1-segment one the positions u (fractions of the span from the
    root) and values v of C0 .. Cm."""
    base = design.copy()

    pitch = base.copy()
    pitch.curves["pitch"] = pitch.curves["pitch"].scaled(1.05)      # 5 % more pitch everywhere

    return {"baseline": base, "pitch_up5": pitch}


if __name__ == "__main__":
    s = settings
    params = BM.read_params(PARAMS)
    design, _ = BM.fit_design(params, ORDERS, SEGMENTS)
    print(f"{PARAMS}: R = {design.radius:g} m, hub radius {design.hub_radius:g} m (root r/R {design.r0:g}), hub "
          f"height {design.hub_height:g} m, {params.blades} blades, {len(params.r)} rows; section table "
          f"{params.section_path()}")
    for c, (e, rel) in BM.fit_report(params, design).items():
        cu = design.curves[c]
        form = "2 segments" if cu.segments == 2 else f"1 segment, {cu.order + 1} control points"
        print(f"  {c:9s} {form}: largest difference from the file {e:.2e} {BM.UNITS[c]} "
              f"({100 * rel:.2f} % of its range)")
    space = BM.BladeSpace(params, design, free=FREE)
    cad = s.cad and importlib.util.find_spec("OCC") is not None
    if s.cad and not cad:
        print("no pythonocc-core here: the cases are written without the CAD")
    st = {"clearance_mm": s.clearance_mm, "sections": s.sections, "tip_band": s.tip_band,
          "tip_sections": s.tip_sections, "hub": s.hub}
    for name, d in cases(design).items():
        res = BM.write_case(OUT_DIR, name, d, params, space, settings=st, plot=s.plot,
                            cad=cad and not BM.design_problems(d, params, s.clearance_mm, hub=s.hub))
        print(f"\n{name}: {res['sections']} sections, blades {res['clearance_mm']:.1f} mm apart"
              + (f"; problems: {'; '.join(res['problems'])}" if res["problems"] else ""))
        print("  written: " + ", ".join(res["paths"].values()))
