# Progress

Rudder parametrization in XGeom. Updated 30 Sep 2026.

## Achieved

- **Extraction** (`Rudder_geom_extraction.py`): reads the wind-tunnel rudder CAD, reframes it to the root LE and
  writes the horizontal section stack, the tilted tip section, the cap loops (`tip_cap.py`), the spanwise
  parameters, the LE/TE lines and the XCAD point file.
- **Section fit** (`control.py`, `match_rudder_section.py`, `Main_PSO.py`): quartic Bezier half-section with a
  rounded TE, PSO plus a least-squares polish; the 50 sections up to 80 mm are in `thickness_params_H80.txt`.
- **Shape modification below Z_opt** (`rudder_modify.py`, `modify_control.py`, `bezier_section.py`): the original
  stays above H; below H the LE and the x and z of every section control point follow spanwise Bezier curves
  pinned to the original at H (value and slope). Sections take any number of control points. Output: the whole
  rudder as an XCAD file, section tables, the design as JSON and a flat design vector for the optimiser.
- **Design tool** (`xgeom_tool.py`, `xgeom_rudder.py`, `xcad_loft.py`): sliders for every variable with bounds and
  free/held flags, the number of control points per curve and per section, live design and 3D section views,
  XCAD button (XCAD file and STEP solid), OCC viewer.
- **Checks**: `_rudder_modify_test.py`, `_xgeom_tool_test.py`, `_section_fit_test.py`, `_rudder_param_test.py` and
  `_tip_cap_test.py` pass.

## Current status

- All of the above is committed and pushed (`ec4e3db`); the code is identical in X_geom and the Rudder folder.
- The baseline design reproduces the original below 80 mm within 0.14 mm (the quartic fit's error).
- The lofted solid is valid (8 faces); close shading shows small ripples along the cap's edge radius.
- The OCC viewer button is untested (no pythonocc where the tests ran).

## Next steps

- Review the reworked tool: general sections, 3D section view, reduced GUI.
- If the cap ripples matter, rerun the extraction with more cap loops (`--cap-loops`).
- Optimisation: sample the `DesignSpace` (`Initial_sampling.py`), screen designs with `design_problems`, couple to CFD.
- Put the tip tilt and cap parameters (already in `parameters.json`) into the tool's design space.
- Open items: section-fit bounds (X2 sits on its bound) and an LE-radius constraint; plain-named outputs are
  overwritten by runs at other heights; Fig. 5 of the plan still labels the tip y = 218.07.
- Later: blades, wings and hulls as tool adapters, following the build plan's phases.
