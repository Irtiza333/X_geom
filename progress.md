# Progress

XGeom: rudder parametrization, propeller blades and hulls. Updated 8 Oct 2026.

## Achieved

- **Rudder extraction** (`Rudder_geom_extraction.py`): reads the wind-tunnel rudder CAD, reframes it to the root LE
  and writes the horizontal section stack, the tilted tip section, the cap loops (`tip_cap.py`), the spanwise
  parameters, the LE/TE lines and the XCAD point file.
- **Section fit** (`control.py`, `match_rudder_section.py`, `Main_PSO.py`): quartic Bezier half-section with a
  rounded TE, PSO plus a least-squares polish; every horizontal section in `thickness_params_y0_y200.txt`.
- **Rudder modification below H** (`rudder_modify.py`, `modify_control.py`, `bezier_section.py`): the original
  stays above H; below H the LE and the x and z of every section control point follow spanwise Bezier curves
  pinned to the original at H. Output: the whole rudder as an XCAD file, section tables, the design as JSON and a
  flat design vector for the optimiser. The full height (root to the top of the cap) is a setting: a new one
  stretches the whole rudder in height (`OriginalShape.scaled`, `scaled_stack`; H, its bounds and the curves'
  heights in proportion, x, chords and section shapes kept), saved with the set-up as `height_scale`.
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
- **Airfoils** (`airfoils.py`, `airfoils/`): the UIUC Airfoil Coordinates Database offline (1,665 Selig-format
  files, `airfoils/index.dat` lists them) and the NACA 4- and 5-digit equations for NACA names without a file. A
  parameter file names its section by `# airfoil <name>` (for example naca2416, clarky) instead of
  `# section_table`: the airfoil gives the camber-line and thickness shapes, the radial table's t/c and f/c scale
  them. The section table (para.py's 26-row format) is written to `airfoils/sections/` when first used; camber
  the mean of the two surfaces at the same x, half thickness their half difference. Own airfoils: `<name>.dat`
  in `airfoils/`. `para.py` and `BladeSurface` take a symmetric section (camber columns 0).
- **Hull** (`suboff.py`, `hull_extraction.py`): the DARPA SUBOFF bare hull (AFF-1) from the equations of Groves,
  Huang and Chang (1989), in metres, and its STEP (a solid of revolution, one face per part: forebody, parallel
  middle body, afterbody, aft cap). Stations cut from a hull's CAD (STEP or IGES, along x) or taken from the
  equations; hydrostatics: V, S, LCB, Am, B, T, Cp, Cm, Cb, the parallel middle body, and for each half about
  midship its Cp, centroid and share of the parallel middle body (the quantities Lackenby's variation works
  with). Output: `outputs/hull/<case>_stations.dat` (the sectional area curve), `<case>_hydrostatics.dat`,
  `<case>_check.png`, `suboff.step`.
- **Hull curves** (`hull_modify.py`, `suboff_hull_params.dat`): a hull starts from a station table (half-height r
  and half-width r' at stations, the nose, middle-body, tail and cap lengths). Each part of each view is a CST
  curve: a class exponent (the nose's bluntness, the tail end's shape) times a Bernstein polynomial of control
  values, joining the middle body with zero slope. Sections are ellipses (circles where r = r'). Design vector:
  the lengths and the curves (exponents and coefficients) free by default (13 slots for SUBOFF), the radii and
  the cap held. Output: the
  design as a station table and as JSON, the set-up, stations, hydrostatics, section points, check plot, and
  STEP (one NURBS face per part: the profile B-spline times the rational circle, so sections are exact
  ellipses).
- **Design tool** (`xgeom_tool.py` with `xgeom_rudder.py`, `xgeom_blade.py` or `xgeom_hull.py`, `xgeom_common.py`,
  `xcad_loft.py`): sliders for every variable with bounds and free/held flags, the number of control points per
  curve, live 2D and 3D views, Save set-up for the optimiser, the CAD build, OCC viewer. Blade: Load and Save
  parameter files, blade and hub, the number of blades Z (2 to 7, the file's to start with; the clearance check,
  the propeller views, the CAD's hub sector, the parameter file written and the set-up follow it; a setting, not
  in the design vector; blades closer than the clearance give a warning, the CAD is still built), the XCAD
  sections, 1 or 2 segments per curve. Rudder: H and the full height. Hull: Load and Save parameter
  files, the lengths and radii, elliptical sections on or off (r' with radii and curves of its own), each curve's
  exponent and control values. The build button builds its own panel's component (in the vehicle named after it:
  CAD hull, CAD propeller, XCAD rudder) and stays in view under the panel; every build's outcome, and a failed
  one's error, goes to `outputs/xgeom_tool_log.txt`. Status bar: green ok, amber while a build runs, orange a
  warning (the build still runs), red for a problem or an error.
  2D plots, in the tool and in the case check plots: the blade's and the hull's have r/R and x across; the
  rudder's planform and curves are drawn as the rudder stands, x (or the curve's value) across and the height up
  (user, 8 Oct 2026).
- **Vehicle** (`xgeom_vehicle.py`; `python xgeom_tool.py` in X_geom, the rudder alone in the Rudder folder): one
  window switching between the hull, propeller and rudder panels (each as when alone, each keeping its state),
  and a Vehicle tab with the hull, the propeller and the rudder at the stern as their designs stand. Placement:
  the propeller plane on the axis (x - L, default the end) and its D in the vehicle (the blade scaled to it,
  default half the hull's depth); the rudder root's TE (x - L, default the start of the cap, a quarter D clear of
  the blades), the number of rudders (1 to 12, evenly spaced around the axis; default 4, a cross), the first
  one's angle around the axis (default 180 deg, under the stern), a scale (default 1, mm to m). Each rudder sits
  on the hull and follows its slope: the root's TE on the hull at the x set, its LE on the hull one root chord
  ahead along the hull's profile at the rudder's angle, the rudder pitched to that line (again whenever it is
  moved or scaled, or the hull changes); a copy of the root section extruded inwards along the rudder's height
  by the largest gap under the root (its thickness over the curved hull, a concave profile under the chord) plus
  2 % of the root chord closes the joint. The whole vehicle or the stern. Vehicle CAD (`vehicle_cad.py`): each
  component's own case and CAD as its own button writes them, then the parts as solids (the hull's STEP; the
  rudder's loft with the root extrusion sewn on in place of its root face, placed for each rudder; the blade's
  five DRDC faces closed at the root by the hub cylinder inside the root ring, Z copies fused with the hub
  cylinder, scaled and moved), placed as the tab shows them and fused, in a process of its own (the window
  stays responsive): `outputs/vehicle/<case>_vehicle.step` (one solid, mm, the hull's frame),
  `<case>_vehicle_parts.step` (the parts placed, not fused), `<case>_vehicle.json` and `<case>_vehicle_setup.json`
  (`python vehicle_cad.py <setup>` builds the vehicle again from it).
- **Checks**: `_rudder_modify_test.py`, `_xgeom_tool_test.py`, `_xgeom_blade_test.py`, `_xgeom_hull_test.py`,
  `_airfoils_test.py`, `_hull_test.py`, `_hull_modify_test.py`, `_section_fit_test.py`, `_rudder_param_test.py`
  and `_tip_cap_test.py` pass (the window checks with `--gui`).

## Current status

- Rudder: the baseline design reproduces the original below 80 mm within 0.14 mm; the lofted solid is valid.
- Blade: the parameter file reproduces the MSc blade (para.py points to 1.3 um, the BladeSurface to 1.3 um). The
  fitted curves differ from the file by (fraction of each range) 2 segments: 0.5 % skew, 1.0 % chord, 1.5 %
  pitch, 2.9 % thickness, 7.6 % camber, 7.7 % rake (the forced zero slope at P4 does not suit their shapes);
  1 segment of 6 control points: 0.06 % to 0.9 %. The CAD build takes about 1.5 min (the DRDC grids).
- Blade: the CAD build and the OCC viewer work on the user's machine (user, 5 Oct 2026).
- Airfoils: 1,664 of the 1,665 files are read (naca1 is a cowl); 14 whose camber line is negative everywhere are
  refused. A section table gives its airfoil back at the 26 stations (1e-7 c; a nearly symmetric airfoil loses a
  camber below 0.1 % of its thickness). Between the stations a cubic spline in sqrt(x/c) follows the airfoil
  within 3e-4 c for half of the database and 1.2e-3 c for 90 %, the largest differences at the nose
  (x/c < 0.0125) and at cusped trailing edges; para.py fixes the 26 stations.
- `UIUC-propDB/`, the UIUC propeller database (this copy: volume 1 only, 200 of its 1,114 data files), is kept as
  a reference; the code does not read it (user, 6 Oct 2026).
- Hull: SUBOFF's equations give V 0.69921 m^3 and S 5.98826 m^2 (the report: 0.699 and 5.988), LCB 0.4611 L
  from the nose, Cp 0.7919; its STEP matches them within 3e-9, and stations cut from the STEP give the radius
  within 1e-6. Choices (user, 7 Oct 2026): SUBOFF first, built from its equations; form parameters with
  Lackenby's shift as the variation; the whole hull varies; extraction and hydrostatics first.
- Hull curves: order 5 per curve fits SUBOFF within 0.05 mm at the nose (N1 0.477, the equations' 1/2.1) and
  0.12 mm at the tail (N2 1.97); V, S and LCB within 4e-6. Choices (user, 8 Oct 2026): CST control radii with a
  nose exponent (over radii at stations and B-spline control points; SUBOFF needed 8 values per part for 0.6
  to 2 mm with those), elliptical sections (over super-ellipses), in place of Lackenby's shift.
- Vehicle: choices (user, 8 Oct 2026): one window with the hull, propeller and rudder and a Vehicle tab; the
  placement by a few settings, the rudder under the stern by default; any number of rudders, evenly spaced
  around the axis; the number of blades chosen in the propeller panel. The wind-tunnel rudder at scale 1 and the
  MSc propeller at D 0.254 m on SUBOFF: the rudder's root TE 97 mm ahead of the end (x 4.259 m, 29.8 mm from the
  axis), its LE at x 4.081 m (60.1 mm), pitched 9.7 deg; the root extruded 11.6 mm inwards (the largest gap
  8.0 mm: the tail is concave under the root); 61 mm clear of the blades. With the TE 0.45 m from the end the
  pitch is 19.0 deg (the tail convex there: the root dips up to 0.5 mm into the hull). Choices (user, 8 Oct
  2026, later): the gap under the flat root closed by the root section extruded inwards; the rudder pitched to the
  hull's slope, the root's LE and TE both on the hull, following it when moved (it had been kept parallel to
  the axis, touching at one point); the full rudder height not fixed by the example; at most 7 blades, and only
  a warning when blades interfere (the hub radius moves the root section).
- Vehicle CAD (SUBOFF, the wind-tunnel rudder x 4, the MSc propeller at D 0.254 m): one valid solid, V 0.70195
  m^3 (the hull 0.69921); it lies on the Vehicle tab's lines. On the user's machine (pythonocc-core, 8 Oct 2026)
  the same, in 126 s (the fuse 15 s); in the cloud (a pythonocc stand-in over cadquery-ocp) about 3 min. In the
  cloud each panel's build button builds its own component; the user saw only the rudder built (8 Oct 2026), not
  reproduced: the button is now tied to its component, named after it, kept in view, and the log records each
  build.
- Blade count (the MSc blade as fitted): Z 2 to 7 pass the clearance check (25 mm; 29.6 mm apart at 7) and the
  hub sector. Checked without OCC (the hub sector's grids); the CAD with Z other than 5 is still to be built on the
  user's machine.

## Next steps

- Vehicle: the placement is not loaded back yet (the set-up JSON records it); the sail and the fins; blended wing
  bodies later.
- Review the blade in the tool: default segments, orders and free variables (`blade_control.py`); camber and
  rake fit poorly as 2 segments.
- Optimisation: sample a saved set-up (`load_design_space`, `load_blade_space`, `Initial_sampling.py`), screen
  designs with the problem checks, couple to CFD.
- Rudder: the tip tilt and cap parameters into the design space; more cap loops if the cap ripples matter.
- Open items: section-fit bounds (X2 on its bound) and an LE-radius constraint; plain-named rudder outputs are
  overwritten by runs at other heights; Fig. 5 of the plan still labels the tip y = 218.07.
