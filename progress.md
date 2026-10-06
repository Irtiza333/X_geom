# Progress

XGeom: rudder parametrization and propeller blades. Updated 6 Oct 2026.

## Achieved

- **Rudder extraction** (`Rudder_geom_extraction.py`): reads the wind-tunnel rudder CAD, reframes it to the root LE
  and writes the horizontal section stack, the tilted tip section, the cap loops (`tip_cap.py`), the spanwise
  parameters, the LE/TE lines and the XCAD point file.
- **Section fit** (`control.py`, `match_rudder_section.py`, `Main_PSO.py`): quartic Bezier half-section with a
  rounded TE, PSO plus a least-squares polish; every horizontal section in `thickness_params_y0_y200.txt`.
- **Rudder modification below H** (`rudder_modify.py`, `modify_control.py`, `bezier_section.py`): the original
  stays above H; below H the LE and the x and z of every section control point follow spanwise Bezier curves
  pinned to the original at H. Output: the whole rudder as an XCAD file, section tables, the design as JSON and a
  flat design vector for the optimiser.
- **Propeller blade** (`blade_modify.py`, `blade_control.py`, `msc_blade_params.dat`): a blade always starts from
  a parameter file (D, blades, root r/R, hub height, section table, and the radial table r/R, P/D, c/D, t/c, f/c,
  skew, rake/D). Design variables: the blade radius, hub radius and hub height (the curves stretch over the span
  between the radii), and each distribution as a Bezier curve over r/R fitted to the file, of 1 segment (C0 .. Cm)
  or 2 segments (the form of `para_control_bez_updated.py`: P1 .. P4 and P4 .. P7, zero slope at P4, the weights
  w23 and w56 as variables; the default for all six). The MSc code builds the geometry (`para.py` for the XCAD
  points, `BladeSurface`, `tip_surfaces_new`, `hub_new` and `X_CAD_new` for the DRDC five-surface IGES with the hub
  sector of the design's height, plus STEP). `msc_blade_params.dat` is the MSc blade as the pipeline builds it:
  its polynomials at 61 stations from `x_blade_new.R_VALUES[0]` (r/R 0.18), hub height 0.625 m (hub_new's).
  XCAD sections (53 points each): their number and how many lie in the tip band, closer together towards the tip
  (default 56, 14 from r/R 0.93, evenly spaced below); the clearance check keeps x_blade_new's stations.
- **Design tool** (`xgeom_tool.py` with `xgeom_rudder.py` or `xgeom_blade.py`, `xgeom_common.py`, `xcad_loft.py`):
  sliders for every variable with bounds and free/held flags, the number of control points per curve, live 2D
  and 3D views, Save set-up for the optimiser, the CAD build, OCC viewer. Blade: Load and Save parameter files,
  blade and hub, the XCAD sections, 1 or 2 segments per curve. Status bar: green ok, amber while a build runs,
  red for a problem or an error. Every 2D plot, in the tool and in the case check plots, has the span (height,
  r, r/R) on the x axis.
- **Checks**: `_rudder_modify_test.py`, `_xgeom_tool_test.py`, `_xgeom_blade_test.py`, `_section_fit_test.py`,
  `_rudder_param_test.py` and `_tip_cap_test.py` pass.

## Current status

- Rudder: the baseline design reproduces the original below 80 mm within 0.14 mm; the lofted solid is valid.
- Blade: the parameter file reproduces the MSc blade (para.py points to 1.3 um, the BladeSurface to 1.3 um). The
  fitted curves differ from the file by (fraction of each range) 2 segments: 0.5 % skew, 1.0 % chord, 1.5 %
  pitch, 2.9 % thickness, 7.6 % camber, 7.7 % rake (the forced zero slope at P4 does not suit their shapes);
  1 segment of 6 control points: 0.06 % to 0.9 %. The CAD build takes about 1.5 min (the DRDC grids).
- Blade: the CAD build and the OCC viewer work on the user's machine (user, 5 Oct 2026).

## Next steps

- Review the blade in the tool: default segments, orders and free variables (`blade_control.py`); camber and
  rake fit poorly as 2 segments.
- Optimisation: sample a saved set-up (`load_design_space`, `load_blade_space`, `Initial_sampling.py`), screen
  designs with the problem checks, couple to CFD.
- Rudder: the tip tilt and cap parameters into the design space; more cap loops if the cap ripples matter.
- Open items: section-fit bounds (X2 on its bound) and an LE-radius constraint; plain-named rudder outputs are
  overwritten by runs at other heights; Fig. 5 of the plan still labels the tip y = 218.07.
