"""
modify_control.py

Modify the rudder below a height H (rudder_modify.py) and write it for XCAD.
Set the files, the settings and the design below, then run

    python modify_control.py

1. The original shape from the root to the top horizontal section (z_top,
   z_full - d) is fitted with Bezier curves: the LE line (corner_points.dat)
   and the control points of the per-section fits (SECTION_TABLE, quartic
   half-sections), raised to DEGREE. With REFIT_SECTIONS on, every horizontal
   section of STACK is fitted first and SECTION_TABLE is written from them.
2. Each case in cases() is a design: the new LE and one spanwise curve per
   section control-point coordinate (P1x, P1z, P2x, ...) below H, pinned to
   the original at H. baseline_design gives the original itself; change its
   control points to change the shape. DEGREE 5 adds a section control point
   (P4) and its two curves; the original shape is the same at any degree.
3. For each case, OUT_DIR gets
     <case>_sections_xcad_m.dat     the whole rudder for XCAD: the modified
                                    sections from the root to H, then the
                                    extracted loops from H + gap up (tip
                                    section and cap included)
     <case>_sections.dat            the modified sections: heights, LE, TE,
                                    chords and control points
     <case>_thickness_params.txt    the same as Section # | X1 | X2 | T1 | T2
                                    | r, when the design is still a quartic
                                    with P1x = 0 and P1z = P2z
     <case>_design.json             the design and the settings
     <case>_check.png               planform, curves and sections
   and, once, original_fit_Zopt<z>.dat / .json, the fits of step 1.
"""

from dataclasses import dataclass

import rudder_modify as RM

# ---------------- FILES (outputs of Rudder_geom_extraction.py) -----------------
CORNER_POINTS = "outputs/corner_points.dat"                     # LE and TE per station; the TE line is kept
XCAD_ORIGINAL = "outputs/sections_xcad_with_cap_y0_y200_m.dat"  # the loops kept above H
STACK = "outputs/sections_stack_y0_y200_raw_mm.dat"             # original sections: check plot, REFIT_SECTIONS
SECTION_TABLE = "outputs/thickness_params_y0_y200_info.dat"    # per-section fits of every horizontal section
REFIT_SECTIONS = False     # True: fit STACK's horizontal sections first (about 0.4 s each) and write SECTION_TABLE
OUT_DIR = "outputs/modified"


@dataclass # LEAVE THESE AS DEFAULTS
class ModifySettings:
    h_max: float = 80.0            # the highest H a design may use (was Z_opt), mm; at most the top horizontal
                                   # section (z_full - d, 199.4987 mm for the default stack)
    h_min: float = 20.0            # the lowest H a design may use, mm; h_min and h_max are the optimiser's bounds
    n_sections: int = 50           # modified sections from the root to H, both included
    num_pts: int = 200             # points per section, laid out as the extraction lays them out
    te_radius_mm: float = 0.75     # TE radius of every modified section, mm (the original sections' radius;
                                   # the circle meets each fitted section tangentially, checked 29 Sep 2026)
    gap_mm: float = None           # original loops kept from H + gap up; None: one modified step, H / (n_sections - 1)
    xcad_units: str = "m"          # units of the XCAD file written ("m" like the extraction's, or "mm")
    fit_free_heights: bool = False # fits to the original shape: False for evenly spaced control heights (least
                                   # squares, exact), True to search the inner heights too (Main_PSO), for
                                   # distributions too curved for a polynomial of the order below
    plot: bool = True              # write <case>_check.png

settings = ModifySettings( # SET PARAMETERS HERE
    h_max = 80.0,
    h_min = 20.0,
    n_sections = 50,
    te_radius_mm = 0.75,
)

# order of the Bezier fitted to each original distribution, root to z_top (curves not named: 3)
FIT_ORDERS = {}

# ---------------- DESIGN ---------------------------------------------------------
DEGREE = 4                        # section Bezier degree: 4 = P0 .. P4 like the section fits; each extra
                                  # degree adds a control point (its x and z curves)
# Bezier order of each spanwise curve below H ("LE", "P1x", "P1z", ...; not named: 3), and how they join
# the original at H: "G1" value and slope (C0 on the original, C1 on its tangent), "G0" value only
ORDERS = {}
JOIN = "G1"
H = 80.0                          # height of the modified region, mm (h_min <= H <= h_max)


def cases(orig):
    """name -> Design. Each curve's control points C0 .. Cn run top first:
    C0 at H (pinned), C1 on the tangent there (pinned with G1; its height is
    free), Cn at the root. design.curves[name].s holds the heights as
    fractions of H and .v the values: x in mm for "LE", fractions of the sharp
    chord for the section coordinates "P1x", "P1z", ... Pinned entries are
    reset from the original whatever is set here."""
    base = RM.baseline_design(orig, H, ORDERS, JOIN)

    fwd = base.copy()
    fwd.curves["LE"].v[-1] -= 20.0      # root LE point 20 mm forward: the LE sweeps forward towards the root

    return {"baseline": base, "le_root_fwd20": fwd}


if __name__ == "__main__":
    s = settings
    if REFIT_SECTIONS:
        top = max(sec.y for sec in RM.read_stack(STACK) if sec.tilt_deg == 0.0)
        print(f"fitting the horizontal sections of {STACK}, up to {top:g} mm")
        RM.fit_section_table(STACK, top + 1e-6, SECTION_TABLE.replace("_info.dat", ".txt"), SECTION_TABLE)
    orig = RM.fit_original(CORNER_POINTS, SECTION_TABLE, None, FIT_ORDERS, free_heights=s.fit_free_heights,
                           degree=DEGREE)
    RM.write_original_fit(OUT_DIR, orig)
    print(f"original shape fitted from the root to z_top = {orig.z_top:g} mm, sections of degree {orig.degree}:")
    for c in orig.names:
        res = orig.residuals[c]
        print(f"  {c:3s} order {orig.curves[c].order}, {len(res)} points, largest residual {abs(res).max():.2e}"
              + (" mm" if c == "LE" else " (fraction of the sharp chord)"))
    stack = RM.read_stack(STACK)
    space = RM.DesignSpace(orig, ORDERS, JOIN, h_min=s.h_min, h_max=s.h_max)
    for name, design in cases(orig).items():
        res = RM.write_case(OUT_DIR, name, design, orig, XCAD_ORIGINAL, s.n_sections, s.num_pts,
                            s.te_radius_mm, s.gap_mm, s.xcad_units, stack=stack, plot=s.plot, space=space,
                            h_min=s.h_min, h_max=s.h_max)
        print()
        print("\n".join(RM.case_report(res, orig, stack)))
        print("  written: " + ", ".join(res["paths"].values()))
