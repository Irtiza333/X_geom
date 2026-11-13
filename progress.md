# Progress

XGeom: rudder parametrization, propeller blades, hulls, wings and fins. Updated 22 Oct 2026.

## Achieved

- **Rudder extraction** (`Rudder_geom_extraction.py`): reads the wind-tunnel rudder CAD, reframes it to the root LE
  and writes the horizontal section stack, the tilted tip section, the cap loops (`tip_cap.py`), the spanwise
  parameters, the LE/TE lines and the XCAD point file.
- **Section fit** (`control.py`, `match_rudder_section.py`, `Main_PSO.py`): quartic Bezier half-section with a
  rounded TE, PSO plus a least-squares polish; every horizontal section in `thickness_params_y0_y200.txt`.
- **Rudder modification below H** (`rudder_modify.py`, `modify_control.py`, `bezier_section.py`): the original
  stays above H; below H the LE and the x and z of every section control point follow spanwise Bezier curves
  pinned to the original at H. Each modified section is written as the extracted section at its height plus
  the change the design makes to this Bezier model (`original_offsets`, `ModSection.written`), so the baseline
  is the extracted rudder itself and no design steps at H; the tables and the JSON hold the model. Output: the
  whole rudder as an XCAD file, section tables, the design as JSON and a flat design vector for the
  optimiser. The full height (root to the top of the cap) is a setting: a new one
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
- **What varies, from the parameter file**: vary lines `# vary <name> Y|N [low high]` for pitch, chord,
  thickness, camber, skew, rake, radius, hub_radius and hub_height. Y: a design variable (a distribution as a
  fitted Bezier curve, its variables free; low high bound its control values, or the radius or height); N: kept
  as the file has it (a distribution exactly, the cubic spline through its rows, `FileCurve`, no variables); a
  name not listed is N (the reader, `blade_vary.py`, uses the standard library only). Nothing Y: `blade_control.py`
  writes only the case `original` and its CAD. A file with no vary line: every distribution fitted, `FREE` free,
  as before. `msc_blade_params.dat`: pitch and chord Y. The
  same file drives the MSc pipeline: `para_control_bez_updated.py` takes para_control 0 (the original blade: pitch
  and chord unchanged), 2 (chord), 3 (pitch) or 7 (both), and the cap of P4's pitch (max_pitch, 1.4);
  `x_blade.py`, `x_blade_new.py`, `X_CAD_new.X_CAD_from_design` and `cad_worker.py` pass both on.
  `Initial_sampling.py` reads `BLADE_PARAMS` (`msc_blade_params.dat`): pitch and chord Y or N give para_control;
  a range bounds pitch p1y, p7y and max_pitch, or chord p1y and y4 (the positions, fractions and weights keep the
  code's bounds); only the flagged parameters' variables are sampled and written (pitch's 6, then chord's 5);
  nothing Y: the original blade's CAD only (case 0, `geometry/original`).
- **Pipeline folder** (`Prop_hull`, the CFX campaign: `gp_infill_from_training_data.py`, `run_campaign.py`, mesh,
  CFX and training scripts): the files it shares with X_geom are kept identical. `Initial_sampling.py` has
  Prop_hull's campaign changes (`--round r` runs infill mode for round r; case number 100 x round + row,
  `pipeline_config.case_id`; resumable from the accepted file) and X_geom's vary lines; `pipeline_config.py` is
  Prop_hull's (`DATA_DIR` Training_3d). Prop_hull also holds `blade_vary.py` and `msc_blade_params.dat`.
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
- **Wing and fin** (`wing_modify.py`, `wing_params.dat`, `fin_params.dat`): a wing or a fin starts from a parameter
  file (the span, the airfoil by name, the pitch axis, and per span fraction eta the chord, sweep (x of the LE),
  rake (z of the LE: the dihedral), pitch (twist), t/c and f/c; optional change columns). Sections: the named
  airfoil (its thickness with the TE closed, and its camber line; a symmetric airfoil gets the NACA 4-digit camber
  line, its maximum at `camber_xc`) scaled to t/c and f/c, plus Bezier changes of the thickness and the camber line
  (Bernstein polynomials of degree k + 1 along the chord, zero at the LE and the TE; k = 3 change points by
  default, 0 to 8), pitched about the pitch axis (0.25 c by default) and moved by sweep and rake. Design
  variables: the span (the curves stretch over it) and each distribution and change as a Bezier curve over eta
  (1 or 2 segments as the blade's; 4 control points, 3 for the changes), the root's sweep and rake pinned at 0.
  Output (`outputs/wing`, `outputs/fin`): the design as a parameter file and JSON, the set-up, the XCAD file (41
  sections, 61 points per side), the sections table, the check plot, and the loft of the XCAD file as STEP (the
  tip flat for now). Example wing: span 200 mm, chord 120 to 70 mm, LE swept 28 mm, 3 deg dihedral, pitch 2 to
  -1 deg, NACA 2412 thinned to t/c 0.10 and f/c 0.015 at the tip. Example fin: span 220 mm, chord 300 to 220 mm,
  LE swept 50 mm, NACA 0015.
- **Design tool** (`xgeom_tool.py` with `xgeom_rudder.py`, `xgeom_blade.py`, `xgeom_hull.py` or `xgeom_wing.py`,
  `xgeom_common.py`,
  `xcad_loft.py`): sliders for every variable with bounds and free/held flags, the number of control points per
  curve, live 2D and 3D views, Save set-up for the optimiser, the CAD build, OCC viewer. Blade: Load and Save
  parameter files, blade and hub, the number of blades Z (2 to 7, the file's to start with; the clearance check,
  the propeller views, the CAD's hub sector, the parameter file written and the set-up follow it; a setting, not
  in the design vector; blades closer than the clearance give a warning, the CAD is still built), the XCAD
  sections, 1 or 2 segments per curve or the file's own ("file": the vary line N, "(file)" in the picker; Save
  writes vary lines for the design as it stands). Rudder: H and the full height. Hull: Load and Save parameter
  files, the lengths and radii, elliptical sections on or off (r' with radii and curves of its own), each curve's
  exponent and control values. Wing and fin: Load and Save parameter files, the span, the XCAD sections and the
  change points, 1 or 2 segments per curve; planform, front view, the picked curve, the wing in 3D and its sections
  over the chord. The build button builds its own panel's component (in the vehicle named after it: CAD hull, CAD
  propeller, XCAD rudder, CAD fin, CAD wing) and stays in view under the panel; every build's outcome, and a failed
  one's error, goes to `outputs/xgeom_tool_log.txt`. Save CAD ... beside it (and Save vehicle CAD ... in the
  Vehicle tab) writes the CAD of the design as it stands to a file picked in a dialog, STEP or IGES: the last
  build's when nothing has changed since, else it builds first (user, 16 Oct 2026). IGES: the blade's own DRDC
  IGES, the others converted from their STEP (`xcad_loft.save_cad`; faces as trimmed surfaces, mm). Status bar:
  green ok, amber while a build runs, orange a warning (the build still runs), red for a problem or an error.
  2D plots, in the tool and in the case check plots: the blade's and the hull's have r/R and x across; the
  rudder's planform and curves are drawn as the rudder stands, x (or the curve's value) across and the height up
  (user, 8 Oct 2026); the wing's and the fin's the same way, the span up.
- **Vehicle** (`xgeom_vehicle.py`; `python xgeom_tool.py` in X_geom, the rudder alone in the Rudder folder): one
  window switching between the hull, propeller, rudder, fin and wing panels (each as when alone, each keeping its
  state), and a Vehicle tab with them all as their designs stand. Placement:
  the propeller plane on the axis (x - L, default the end) and its D in the vehicle (the blade scaled to it,
  default half the hull's depth); the rudder root's TE (x - L, default the start of the cap, a quarter D clear of
  the blades), the number of rudders (0 to 12, evenly spaced around the axis; default 4, a cross), the first
  one's angle around the axis (default 180 deg, under the stern), a scale (default 1, mm to m). Each rudder sits
  on the hull and follows its slope: the root's TE on the hull at the x set, its LE on the hull one root chord
  ahead along the hull's profile at the rudder's angle, the rudder pitched to that line (again whenever it is
  moved or scaled, or the hull changes); a copy of the root section extruded inwards along the rudder's height
  by the largest gap under the root (its thickness over the curved hull, a concave profile under the chord) plus
  2 % of the root chord closes the joint. With TE rake ticked (user, 22 Oct 2026) the rudder's angle is set
  instead: its TE line at the rake from square to the axis (0: square, as BB2's stern planes; top aft positive),
  meeting the hull at the x set; the rudder rests on the hull (the root's closest point on it) and its root
  section is extruded along the TE line until it meets the hull fully (the largest gap plus 2 % of the chord);
  unticked, the box shows the TE's rake the hull's slope gives. Fins (0 to 12 around the axis; default 1, on top, a tenth of the way
  along the middle body) and the wing pair (wings 2 or 0; the starboard wing's angle 0 to 180 deg, default 90,
  level; a quarter of the way along the middle body) sit the same way, their root's LE on the hull at the x set
  (m from the nose) and the TE one root chord aft along the hull, pitched to the hull's slope (their own pitch on
  top), the root extruded inwards; the port wing is the starboard one's mirror image (y to -y): the same
  dihedral, camber and pitch. A count of 0 leaves that component off the vehicle and out of its CAD. The whole
  vehicle or the stern. Vehicle CAD (`vehicle_cad.py`): each
  component's own case and CAD as its own button writes them, then the parts as solids (the hull's STEP; the
  rudder's, fin's and wing's loft with the root extrusion sewn on in place of its root face, placed for each
  copy and standing 0.2 mm off the hull on it, the port wing mirrored; the blade's
  five DRDC faces closed at the root by the hub cylinder inside the root ring, Z copies fused with the hub
  cylinder, scaled and moved), placed as the tab shows them and fused, in a process of its own (the window
  stays responsive): `outputs/vehicle/<case>_vehicle.step` (one solid, mm, the hull's frame),
  `<case>_vehicle_parts.step` (the parts placed, not fused), `<case>_vehicle.json` and `<case>_vehicle_setup.json`
  (`python vehicle_cad.py <setup>` builds the vehicle again from it).
- **CAD patch topology, ripples and half bodies** (`xcad_loft.py`; 5 Nov 2026, from the user's screenshots of
  the rudder database in the mesher): the loft now builds one bicubic B-spline surface per patch, fitted by
  least squares through the section curves' pole rows (degree 3 both ways, interior knots placed from the row
  spacing), in place of interpolating ThruSections. Patches as the user drew them: the side from the root to a
  collar (`collar_mm` below the highest horizontal section, default 10, 0 for none), the tip band from the
  collar over the tip's side walls onto the tip section, the cap rim from the tip section to the line where
  its edge rounding meets the deck (found from the cap loops' cross profiles and smoothed along the chord),
  and the cap crown from that line to the ridge; seams at the LE and the TE; the rudder is 9 faces (was 8 with
  a ruled sliver at 199.5 mm and a thin closing face at the crest). The ripples are gone: the old cap
  interpolated the extraction's cap loops at spanwise degree 6 and its curvature swung +4200 to -1700 1/m
  across the cap (the bands the user circled); the fitted cap rounds once (one positive hump, the deck under
  10 1/m), and the side's ringing at the H join (+-60 1/m) drops to +-6. The fit stays on the data: largest
  deviation 60 um at the loops' points, volume within 4 mm^3 of the old solid (584 L), booleans and the
  vehicle fuse unchanged. Half bodies: `loft_loops(half=True)` builds the half on the LE-to-TE side of the
  plane of symmetry (one planar face closes it; the loops must be symmetric within 0.2 mm, so a cambered wing
  is refused with a message), and `hull_modify.build_solid(half=True)` the starboard half of the hull (half
  revolutions plus the centre-plane face); both exactly half the volume, valid. The tool: a 'half body' switch
  under Output for the rudder, the fin, the wing and the hull, saved with the set-up (the component files'
  `half`; the rudder's JSON); build and Save CAD then write the half; the Vehicle CAD always builds full
  bodies (`build_task(case, half=False)`).
- **Input files** (`xgeom_files.py`, claude/xgeom-file-formats.md; 26 Oct 2026): one component file layout
  (`.txt`) for the blade, the rudder, the fin, the wing and the hull: the geometry, the tool's settings, each
  curve's form (`table`, `bezier1 <points>`, `bezier2`; the hull's `cst <order>`) and every variable's value,
  bounds and free flag, then the table (blade `r/R P/D c/D t/c f/c skew_deg rake/D`; rudder, fin and wing
  `eta pitch_deg chord_mm t/c f/c sweep_mm rake_mm` and the dt/dc columns; hull `x_m r_m [rp_m]`). The var lines
  define each Bezier curve exactly and its column is the curve at the rows, so a file round-trips (a rewrite is
  the same file); a curve with no var lines is fitted to its column, and a column edited under its var lines is
  refused with a message. `load_component` and `write_component` in `blade_modify.py`, `wing_modify.py` and
  `hull_modify.py` (planar curves can now be `table` too); old `.dat` files still load
  (`python xgeom_files.py convert old.dat new.txt`); `airfoils.py` reads a coordinate file by its path. The
  tool: the File box's Load ... reads a component file and sets the whole panel, Save set-up ... writes one (the
  dialog opens on the file it came from), `--file`; the rudder (modified below H) keeps its set-up JSON (its own
  Load ... since 12 Nov). Case
  outputs: `<case>.txt` in place of `<case>_params.dat` and `<case>_design_space.json`. Examples, now the
  tool's defaults: `msc_blade.txt`, `wing.txt`, `fin.txt`, `suboff_hull.txt`. The assembly file is read and
  written (`read_assembly`, `write_assembly`); the tool and the pipeline do not use it yet.
- **Checks**: `_rudder_modify_test.py`, `_xgeom_tool_test.py`, `_xgeom_blade_test.py`, `_xgeom_hull_test.py`,
  `_xgeom_wing_test.py`, `_xgeom_files_test.py`,
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
- Wing and fin: the example files fit exactly (their distributions are polynomials); an elliptical chord (120 to 60 mm) fits
  within 0.03 mm at 4 control points. The loft is one valid solid of 4 faces (the two sides, the flat root and
  tip), its volume the sections' within 3e-4. Choices (user, 8 Oct 2026): a wing or fin starts from a parameter
  file (example files given); sections a named airfoil with Bezier changes along the chord, each change a curve
  over the span; wings a mirrored pair at a chosen x and angle, fins like the rudders, both on the hull's slope
  with the root extrusion; wing tips after this step.
- Vehicle CAD with 2 fins and the wing pair at 70 deg on SUBOFF (cloud): one valid solid. The fuse came out
  invalid wherever a root lay along the hull (a wing on the middle body: the hull grazes the edge between the
  root extrusion and the loft) until each appendage stood 0.2 mm off the hull on its extrusion; a fin's root LE
  exactly on a join of the hull's faces (the start of the middle body) stayed unfused, so the defaults avoid the
  joins.
- Vehicle, TE rake (user, 22 Oct 2026): `Placement.rudder_rake` (None: pitched to the hull, the default),
  `place_rudder(..., rake, slope)` with the rudder's TE line (`te_slope`, from the original's TE rows, which every
  design keeps: dx/dy -0.142, 8.08 deg for the wind-tunnel rudder), `rest_on_hull`; Vehicle CAD extrudes the root
  along the TE line (`extended_rudder(..., direction)`, the set-up's `slope`). Wind-tunnel rudder on SUBOFF at the
  default x, rake 0: root chord 8.08 deg, root LE 60.4 mm and TE 35.0 mm from the axis, extruded 14.5 mm; the
  vehicle's CAD one valid solid, V 0.701645 m^3. The window checks ran under Xvfb in a Python 3.12 environment
  (the cloud's Python 3.13 has no Tk).
- Blade count (the MSc blade as fitted): Z 2 to 7 pass the clearance check (25 mm; 29.6 mm apart at 7) and the
  hub sector. Checked without OCC (the hub sector's grids); the CAD with Z other than 5 is still to be built on the
  user's machine.
- para_control 0: in this repo only the old `para_control.py` had an explicit 0 branch; `para_control_bez.py` and
  `para_control_bez_updated.py` took 2, 3 and 7 from the first push (14 Sep 2026) and `x_blade.py` fixed 7.
  Choices (user, 14 Oct 2026): 0 back as the original blade; Y/N flags in the parameter file, read by both the
  XGeom blade and the MSc pipeline; a flag's bounds as a value range. Every line N: `Initial_sampling.py` writes
  `geometry/original/sample_blade0.iges` (8 faces; 204 s in the cloud), `blade_control.py` the case `original`.
- Prop_hull synced (user, 14 Oct 2026): it had changed `Initial_sampling.py` and `pipeline_config.py` (11 Oct);
  X_geom had changed the vary lines, para_control 0, `para.py` and `blade_surface_new.py` (airfoil by name). The
  merged `Initial_sampling.py` runs a round in infill mode (cloud, the CAD left out: cases 100 to 102, resumed
  without repeats). The GP script keeps its own copy of the bounds and 11 variables: ranges on the vary lines,
  or pitch or chord alone, are not yet seen by it.
- BB2 (one-off, user, 22 Oct 2026; `bb2/`, the scripts take the input paths): the MARIN stock propeller S7371R cut
  at 34 radii into `bb2/outputs/bb2_blade_params.dat` (blade_modify's format: D 5 m, Z 6, every vary line N), a
  section table per r/R and one at r/R 0.7 (`bb2_sections.py`); one blade on the conic hub and cap of drawing F6
  (the root runs 0.1 m into the hub, so the hub cuts the root sections) and the aft half of the hull rebuilt with
  `hull_modify` (tail order 5, within 0.96 mm of the IGES), fused into one valid solid (`bb2_build.py`). The blade
  takes the section shape per r/R (`SectionsBladeSurface`): within 0.9 mm of the original at 44 cuts; X_CAD_new's
  single table is off by up to 23 mm at the root (the shape changes along the span). The outputs carry MARIN data.
  Then (user, 22 Oct 2026): the six-blade propeller (`bb2_prop6`, `bb2_vehicle6`); quadrant IGES files with
  solids, the quadrant y >= 0, z >= 0 of hull.iges (hull, hub with six blades, vehicle); the wind-tunnel rudder as
  BB2's X stern (`bb2_rudder.py`): scaled 1:20 (the free fit gives 1:20.006) and placed by a fit of its LE, TE and
  their tops onto BB2's stern plane in hull.iges (turned 7.5 deg; LE within 2 mm, TE 16 mm, the tip line 83 mm
  off at its ends: its tip is cut at another angle), the root extruded 510 mm into the hull, four rudders; every
  vehicle one valid solid. No quartered blades (user, 22 Oct 2026): the quadrant files are the hull, the hull with
  the hub, and with the rudder too; the propeller for the periodic set-up is one blade on its 1/6 of the hub, the
  periodic hub strip of hub_new (TM 2013-178 Sec. 10: theta within p_h(x) +- 30 deg, p_h on the mid-line of the
  blade's footprint on the cone), `bb2_blade_sector` as a solid and as its walls; six of them have the six-blade
  propeller's volume to 4e-7.
  Then (user, 22 Oct 2026): the rudder placed by the XGeom tool's rule (`xgeom_vehicle.place_rudder` on the
  rebuilt hull: the root's TE on the hull where BB2's TE meets it, x 5304.6 mm; the LE a root chord ahead on the
  hull, x 8762.6 mm, 80 mm aft of BB2's; pitched 16.46 deg with the hull; root extruded 72.1 mm plus the 0.2 mm
  lift) and compared with BB2's stern plane in hull.iges above the hull (`bb2_rudder_compare.dat`,
  `bb2_rudder_sections.dat`, `bb2_rudder_compare.png`). The previous best fit lay within 4 mm of BB2's surface
  over 95 % of it (57 mm at the tip), area within 0.2 %. The rule turns the rudder 9 deg further aft than BB2's
  (on BB2 the rudder file's root runs at 7.5 deg to the axis, the hull under it at 16.4 deg): the tip LE 722 mm
  aft, the tip TE 626 mm aft and 599 mm lower, area 4.9 % less, BB2's surface up to 650 mm from it. The vehicles
  and the quadrant carry the rule's rudders, fused as their parts outside the hull (a fuse of the whole rudders
  left some apart: the flat root crosses the hull at a shallow angle).
  Then (user, 22 Oct 2026): BB2's rule is the TE line square to the axis. The rudder so placed (XGeom's TE rake
  0: the TE line through BB2's TE on the hull, x 5304.6 mm; resting on the hull at its root LE, x 8874.6 mm, 33 mm
  ahead of BB2's; root chord 8.08 deg; the root extended 616 mm along the TE into the hull) lies within 16 mm of
  95 % of BB2's surface (mean 5.6 mm; up to 123 mm at the tip, whose LE corner sits 180 mm higher: the tip is
  cut at another angle), its LE swept 12.08 deg against 11.46, area 2.2 % more.
  Then (user, 22 Oct 2026; the propeller is optimized in this set-up): the vehicles carry BB2's own stern planes
  (`bb2_sternplanes.py`: hull.iges's stern plane copied to 45 + 90 k deg, cut by the rebuilt hull with the hub
  and fused; the quadrant file likewise). The wind-tunnel rudder cannot line up with BB2 by placement: its tip is
  cut at 94.5 deg to its TE (3.6 deg to its root) against BB2's 90.6, and its LE converges on the TE at 12.1 deg
  against 11.6; XGeom reads the model right but keeps the original's TE line and tip, so it cannot make BB2's tip
  (a tip-angle variable would). Stern plane outside the rebuilt hull 4.799887e9 mm^3, outside hull.iges's own hull
  4.799854e9 (7e-6); vehicles one valid solid each (1.808035e12 and 1.808734e12 mm^3), quadrant 4.51974e11.
  `bb2_rudder.py` keeps the comparison and the TE-square rudder (`bb2_rudder.step`) only.
  Then (user, 22 Oct 2026): the set-up with the propeller periodic, `bb2_quadrant_sector` (.step, .iges with
  solids): the quadrant of the hull without the hub and the stern plane (`bb2_hull_rudder_quadrant`, one solid,
  4.516768e11 mm^3) and the one blade on its 1/6 of the hub (3.376802e8 mm^3) as a second solid, meeting at the
  hub's forward end (x 1929.9 mm). The sector is turned about the axis so that the strip's face at the hull's end
  is centred on the quadrant, 15 to 75 deg (user, 22 Oct 2026: at prop.stp's blade at 32.8 deg it stuck out of
  the hull's quarter face, 35.6 to 95.6 deg): -80.62 deg from prop.stp's blade 1. The strip twists about 100 deg
  along the hub, so the blade then lies at -16.8 to 43.3 deg, its clocking against the stern planes not one of
  prop.stp's.

- Input files (26 Oct 2026; user's choices: the rudder planar on the fin's engine, only the span coordinate
  normalized, optimization runs vary the propeller only for now): steps 1 (the files) and 2 (the tool's Load and
  Save set-up) of 5 done. Checked in the cloud (all checks, the windows with `--gui`, OCC) and here
  (`_xgeom_files_test.py`). Saving the SUBOFF hull puts its CST curves' half-heights into the table (within
  0.12 mm of SUBOFF's equations); a fit at another order then fits to them.

- CAD topology and half bodies (5 Nov 2026): installed in X_geom and the Rudder folder. All checks pass in
  the cloud (the windows with `--gui`, OCC: the new patch counts, the half bodies, the vehicle fuse) and here
  (CAD and window skipped). The collar height and the cap split are `collar_mm` and `crown` in
  `modify_control.ModifySettings`.

- BB2 set-up with the relofted stern plane (5 Nov 2026): `bb2/bb2_sternplane_loft.py` rebuilds the stern
  plane with the mesh patch topology (nine faces; the chopped buried root continued as the straight taper,
  the cap as symmetrized rings resampled by arc into the sections' layout, 60 section poles for the blunt
  TE roll) and writes the set-up files again under the 22 Oct names; `bb2_sternplane.step` stays the
  extracted original, `bb2_sternplane_loft.step/.iges` is the rebuilt plane. Every output one valid solid;
  cuts against the original within 0.7 mm (the exposed span 0.17 to 0.38), the part outside the hull within
  1.5e-4 by volume, the sector turn unchanged. The hull and the propeller as bb2_build.py made them. About
  14 min. In `bb2/` and `bb2/outputs/` here; `claude/bb2.md` has the details.

- `temp_rudder_cad.py` (5 Nov 2026): rebuilt on `xcad_loft.loft_loops`, so its solid carries the new patch
  topology (nine faces); the same files in, the same `.step` out. The full section file gives the same solid
  as the topology work's loft (583,340 mm^3, fit 0.074 mm, valid). The sections may differ in point
  count (resampled by cosine arc into the original file's layout, the point farthest from the LE taken
  as the TE), and a file without '#' marks is split where the height column changes. In X_geom and the
  Rudder folder (byte-identical).

- Five patches (6 Nov 2026): `xcad_loft.loft_loops` rebuilt on the blade's topology (DRDC TM 2013-178): LE and
  TE strips (`strip_frac`, default 0.07 of the root chord), two central panels, and a tensor-product tip patch
  over the top `tip_frac` (0.15) of the span, fitted to wrap curves cut from an interim smooth loft; six faces
  every case, the root flat, the half body a boolean half (0.5 by volume to 8e-7). `ModifySettings` carries
  `strip_frac` and `tip_frac` in place of `collar_mm` and `crown`; set-up files save and restore them. The wind
  tunnel rudder re-exported (`outputs/wind_tunnel_rudder_full.igs`, `_half.igs`, Rudder folder): worst 0.43 mm
  against the extraction data, the sides 0.02. The BB2 set relofted (`bb2/outputs/`, `TIP_FRAC = 0.03`, about
  17 min): cuts against the original within 0.8 mm except the two tip corner blends (to 4.4 mm over the last
  40 mm or so) and just under the cap foot (4.0); the nine-face loft held 0.7 there, the price of one
  four-sided tip patch on the blunt cap corners. Volume outside the hull within 3.2e-4 of the original. All
  four test suites pass. In X_geom and the Rudder folder (byte-identical shared files).

- Input files, LF (6 Nov 2026): `write_component` and `write_assembly` write LF line endings on every
  platform, so a saved component or assembly file is byte-identical across machines (`_xgeom_files_test.py`
  checks both); the file-format spec is in the repo as `claude/xgeom-file-formats.md` (the path
  xgeom_files.py's docstring names), for the web front end building on these files.

- Sample files re-saved (11 Nov 2026): `fin.txt`, `wing.txt` and `suboff_hull.txt` written again through the
  tool's own save, so they carry the half setting they predated (`# half N`; the hull its settings heading)
  and a load and re-save gives the same bytes on any machine (`msc_blade.txt` already did); nothing else in
  them changed. Asked for by the web front end's round-trip test.
- Tip patch at the tip's edge (11 Nov 2026): the five-patch CAD of 6 Nov rippled in three places: the top
  15 % of both sides and the cap (the tip patch could not follow wall, edge radius and crown through one wrap
  parametrization: -0.82 to +0.37 mm against the STEP), and a band round H in modified designs.
  `xcad_loft.loft_loops` rebuilt: the LE strip, the TE strip and the two central panels run from the root to
  the tip's rim (the first loop that is not horizontal: the tilted tip section, a cap's first ring), their
  rows tilting onto it over a band under it; the tip patch covers only the cap above the rim, fitted to a
  mesh of the cap through a mean value map (no interim solid: the loft takes 2.5 s instead of a minute);
  rows are sampled from the smooth curve through each loop, the knots crowd towards the patches' edges.
  `tip_frac` is gone (set-ups that carry it still load); `strip_frac` takes a pair (LE, TE). A step between
  horizontal loops is eased out of them before the fit (`_ease_flats`, argument `ease`): modified sections
  meet the original ones in one, the model's sections lying 0.13 mm off the extracted loops; it is spread
  over bands below and above that tilt the surface by 0.1 deg at most, and named in the tool's status line.
  `Rudder_geom_extraction.py` writes 40 cap loops (`DEFAULT_CAP_LOOPS`; 10 left the tip's 0.7 mm edge radius
  scalloped by 0.1 mm); `outputs/sections_xcad_with_cap_y0_y200_m.dat` and `parameters.json` are the 40-loop
  ones, the 201 sections unchanged. The wind-tunnel rudder: six faces, valid, the sides within 0.005 mm of
  the STEP, the tip within 0.108 (the cap model's own distance), reflection lines straight; checked with
  maps of every face and with the STEP's mesh vertices against the loft. BB2's stern plane relofted
  (`bb2/bb2_sternplane_loft.py`: sections of 801 points, 1 mm apart under the cap's foot, 30 rings on exact
  profiles, strips 0.07 and 0.01 of the root chord): within 0.10 mm of the original outside the synthetic
  root (4.4 mm at the tip's corners with the 6 Nov file), the set-up files in `bb2/outputs` written again.
  Tests: tool (a stepped stack and a strip pair added), hull, wing, files, modify.
- Modified sections written as the extracted ones plus the model's change (12 Nov 2026, the user's word): the
  Bezier model's fit of the original lies up to 0.135 mm (RMS 0.07) off the extracted sections, so sections
  written from the model alone met the original ones at H in a 0.17 mm step that the loft had to ease, and
  the baseline was a replica of the rudder. Now `written = model(design) + (extracted - model(the original's
  fit))`, point by point (both lay their points on the same cosine stations) and for x of the LE.
  `rudder_modify.original_offsets` measures the offsets at the horizontal loops of the original XCAD file
  (`stack_of_xcad`), against the model at the original's TE radius (`OriginalShape.te_radius`, the section
  table's `R_orig_mm`); `Offsets.at` gives them at any height (linear between the loops either side);
  `build_sections`, `modified_sections` and `section_at` take them (`offsets=`), `ModSection.written()` and
  `loop()` add them, `selig` stays the model. `write_case` always writes this form, and the tool draws and
  writes the same sections (`RudderAdapter.offsets`). The baseline writes the extracted rudder itself (a loop
  at an extracted height is the original file's line, digit for digit); the user's set-up of 6 Nov
  (baseline, H = 80) lofts to the original's solid: no step found (0.00003 mm in the loops against 0.168),
  583,361.87 mm^3 against 583,361.9, the STEP's mesh vertices within 0.008 mm (RMS 0.0014) as with the
  original's own loft, the half body half the volume; no step at any H from 20 mm to the top section, nor
  for a changed design, which meets the extracted section at H exactly. The section table, the thickness
  format and the design JSON hold the model (the XCAD loops differ from them by the offsets; the JSON's
  `xcad` entry and the table's notes give the model's distance). A section too thin to take the offsets is
  refused by name (a problem in the tool). Another TE radius still shows as a change, and steps at H;
  `_ease_flats` stays for stacks that step. Tests: modify 137 (29 new, on a bumped synthetic original the
  model cannot hold), tool 40 with the GUI (21 without OCC), hull 59, wing 51, files 53.
- TE strip on the TE's rounding (12 Nov 2026; the user's screenshot from the mesher: a row of teeth along
  the TE from the tip corner down, read as folds; his suggestion: lower-order surfaces). Measured on his
  STEP, the TE strip is smooth there (along the span within 0.0001 mm of its own trend, its normals within
  0.06 deg; the sides within 0.005 mm of the original CAD), so the patches' order is not what the picture
  shows. A viewer or a mesher draws each patch with triangles sized for the patch; the strip was 25 mm wide
  with the 0.75 mm round (2.1 mm of arc) in its middle, and display triangles a few mm long crossing the
  round come out as teeth, one per row (the spacing in the screenshot, about 3.7 mm, fits; emulated here
  with display meshes on the same STEP, where the surface drawn from its own normals is clean). In the
  original CAD the TE round is a face of its own between straight edges, which draws clean.
  `xcad_loft.loft_loops`: with one `strip_frac`, a TE strip more than twice as wide as the TE's rounding is
  narrowed to it (`_edge_rounding`: the arc from the TE to where the curvature has fallen to a quarter of
  its peak; 1.5 times that to each side: 1.74 mm on this rudder instead of 12.62). A sharp TE keeps the
  fraction; a pair (LE, TE) is used as given (BB2's stern plane; `ModifySettings.strip_frac` takes a pair
  too). `LoftResult.strips` and `te_round` record it. Five patches and the root as before, six faces. Rows
  are now sampled from each loop's smooth curve itself (`_SmoothLoop.at`), not by linear interpolation
  along its 1501-point polyline, whose corners a strip that narrow would take up: the loops lie within
  0.00012 mm of the patches (0.0018 before). The wind-tunnel rudder: valid, 583,363.2 mm^3, area
  73,229.80 mm^2 (the STEP's 73,229.84), the half body half; the STEP's mesh vertices within 0.004 mm of
  the sides (0.005 before; rms 0.0006), the tip 0.108 as before; the TE strip along the span within
  0.00003 mm of its trend. `Rudder/outputs/wind_tunnel_rudder_full.igs` and `_half.igs` written again.
  BB2's stern plane relofted here as a check only (its pair of strips as given): valid, its section cuts
  within 0.01 to 0.03 mm (0.03 to 0.05 before); the files in `bb2/outputs` are still those of 11 Nov. Left
  as it is: the tip patch holds its 0.7 mm edge rounding and its end closures inside one patch, which a
  coarse display may blur; the cure would be a band of their own. Tests: tool 43 with the GUI (22 without
  OCC), wing 51, hull 59, files 53.
- Load ... for the rudder's set-up (12 Nov 2026; the user: the rudder's panel showed Save set-up only, no
  way to read a set-up back in). The first tool had Save / Load design space; both went on 29 Sep at the
  user's request, Save set-up came back on 1 Oct for the optimiser, and reading one stayed on the command
  line (`--space`). The Load ... the other components have reads component files, which the rudder modified
  below H does not have. Now the rudder's panel starts with a Set-up box, as the others start with their
  File box: the set-up file loaded (or saved with Save set-up) last, and Load ... (`App.on_load_space`),
  which reads a `<case>_design_space.json` that Save set-up or an XCAD run wrote; the dialog opens on the
  output folder and lists those files. `RudderAdapter.load_space` takes the set-up over whatever the tool
  holds by then (another section degree, full height, orders, bounds, free flags, settings; before, it was
  only called on a fresh adapter) and is all or nothing: a file that is no rudder set-up, or one that fails
  part way, raises a ValueError that names the file and says why, and the design stays as it was. The case
  name is left alone, so the next XCAD or Save set-up does not write over the file loaded unless that is
  the case's own. Save set-up now shows its file in the Set-up box too. Tests: tool 47 with the GUI (24
  without OCC), wing 51, hull 59, files 53, blade 64.
- Propeller extraction and the conic hub (13 Nov 2026; the user: an extraction script for the propeller "so we
  can back out the six curves", and a hub of two radii, the base and the smaller top, a height, and the narrow
  end "capped off with a rounded top" to match the original hub; the CAD to write the whole propeller, half or
  1/Z at the user's choice). The 5 Oct decision against CAD extraction for blades is reversed.
  `blade_extraction.py <cad> [--case --out --stations --diameter --x0 --hub BASE TOP HEIGHT X --hub-height
  --tip-pitch --ref-rr --axis --blade]`: the general form of `bb2/bb2_sections.py`. The axis is found from the
  faces' edges (a root or tip edge is a circle about it; the candidates x, y, z and the principal axes), faces
  of revolution about it are the hub's and left out, the blades are faces that touch along their edges, one is
  cut by cylinders (the root cut moved in until it closes, the tip cut 0.1 % R in), the leading edge is the
  blunter end (upstream: the axis is turned round if not; turning the other way: a left-handed propeller,
  mirrored and said so). Per cut, as before: P/D from the nose-tail line, c' = c / 1.005, the shape on para.py's
  26 stations (`airfoils.shape`), skew and rake of the mid-chord point (x0 the generator line, default the
  root's reference point); the rows at the root (where the faces start) and r/R 1 continued from the nearest
  three cuts. Writes the component file (`<case>.txt`: table curves, the globals held, the hub, the section
  table at --ref-rr and `section_tables` with every cut's), `_cuts.dat`, `_check.png`, and rebuilds the blade
  from the file to report its distance from the cuts. BB2's propeller (`bb2/outputs/bb2_prop.txt`, from the IGES
  the user put there, which is prop.igs of 22 Oct: six blades, no hub): 40 cuts, the same numbers as
  `bb2_blade_params.dat`; rebuilt with the section tables within 0.71 mm of the cuts (rms 0.14), with one table
  23.5 mm (the shape changes along the span). `blade_sections.py` (SectionsBladeSurface, from bb2_build.py)
  now serves `blade_modify.blade_surface` whenever the parameters carry `section_tables`, so the tool's views
  and CAD follow the shapes; one table where the file has none, as before.
  `prop_hub.py`: `HubProfile(base, top, height, x_base)`: the cone from the base radius (upstream, at the base
  plane x_base) to the top radius, closed by the sphere tangent to the cone through the tip (height from the
  base; height tan a - top sin a / (1 + sin a) = base - top fixes the half angle a); the cylinder as before
  (top None, centred on x = 0). F6's numbers (797.22, 138.04, 1568.75 mm, the base 576.13 mm upstream of the
  generator line) give bb2_build.py's hub exactly. `BladeDesign` has `hub_base`, `hub_top`, `hub_x` (m; slots
  `hub_base`, `hub_top`, `hub_x` after the three globals when the hub is a cone, held to start with; the hub
  radius is then the blade's root radius, where the table starts); the component and parameter files carry
  `hub_base_m`, `hub_top_m`, `hub_x_m` (all three or none); the blade is extended below its root into the hub
  (`root_extension`: 2 % of the span under the hub's smallest radius along the root section) and the hub
  trims it; `design_checks` refuses a hub that does not reach under the whole root. The CAD (`write_cad`,
  setting `part`: whole, half, sector; `cad_part`): sector with a cylinder hub the DRDC surfaces as before;
  otherwise solids (`prop_hub.propeller_solid`: the blade closed at its root ring, cut by the hub, Z copies
  fused with the hub, or the k/Z periodic sector of hub_new's wedge about the footprint's mid-line with the k
  blades; half needs an even Z); IGES in BRep mode. The tool: a hub radio (cylinder, cone) in the Blade and hub
  box with the three rows, the hub drawn from its profile in the propeller and Vehicle tabs, "CAD writes" in the
  Output box (the status line says what it writes; the vehicle builds the whole propeller and takes it as the
  solid). BB2 from `bb2_prop.txt` on the F6 hub: the sector 3.37671e8 mm^3 (bb2_blade_sector 3.376802e8), 10
  faces, the footprint 18.8 deg from the sector's edges; half 1.013013e9, 21 faces; whole 2.026021e9
  (bb2_prop6 2.026082e9), 33 faces; all valid, read back from IGES as one solid (`bb2/outputs/bb2_prop_sector`,
  `_half`, `_whole` .step and .iges; `bb2_prop_parts.png`). Tests: blade 78 with the GUI and the CAD (66 here;
  the hub maths, the design and files, the parts, the adapter and window, the cone sector, the extraction of
  the MSc blade's DRDC CAD: its curves within 1.1 % of their ranges below r/R 0.9), tool 47, hull 59, wing 51,
  files 53. `blade_vary.py` (HUB_SLOTS) is in Prop_hull too, as the other shared files.
- From CAD and the set-up files (13 Nov 2026; the user: set-up files for the BB2 propeller, the rudder and the
  hull, and a tool feature that takes a CAD file, extracts the parameters, saves a set-up and shows it on the
  sliders, "so users know where the parameters are on their starting geom and set bounds according to that";
  blade, hull and rudder for now, the others once this works; of the forms that capture a curve, "the
  combination that results in the least error", and the user's own changes refitted with the error shown).
  `xgeom_from_cad.py`: `from_cad(kind, cad, out_dir, case)` runs the component's extraction
  (`blade_extraction.extract`, `hull_extraction` stations, `Rudder_geom_extraction.run` plus
  `fit_section_table`), fits every candidate form and keeps the one with the least error: blade 2 segments or
  1 segment of 3 to 8 points, hull CST orders 3 to 8, rudder law orders 2 to 6; the error is the largest
  difference from the extracted data over its range (hull: over r*; a rudder law's range at least 1 %), and a
  form with more points must beat the best cheaper one by 0.1 % of the range (`FIT_TIE`), so alike fits keep
  the fewer points. The set-up is written through the tool's adapter (`blade_setup`, `hull_setup`: the free
  variables and bounds as the tool starts a file; blade_control.FREE for the blade) as `<case>_setup.txt`
  (blade, next to the extracted `<case>.txt`), `<case>.txt` with `<case>_params.dat` (hull), or
  `<case>_design_space.json` with `<case>_extraction/` (rudder; its laws at the orders found). Command line:
  `python xgeom_from_cad.py blade|hull|rudder <file> [--out --case]`; for the blade an extracted `<case>.txt`
  and for the hull a `<case>_params.dat` give the set-up alone. The tool: "From CAD ..." in the Parameter file
  box (blade, hull) and the rudder's Set-up box; it runs off the GUI thread, then loads the file in the forms
  found (the status line lists them and the error of each); changing points or segments refits and says the
  error, as before. `xgeom_rudder.RudderAdapter(files=, fit_orders=, out_dir=)`, and `load_space` takes a set-up
  of other files (switches to them). `xgeom_blade`: `set_form`, `set_forms`. `hull_modify.default_bounds`: a
  negative coefficient (a high-order fit) is now inside its bounds (+-0.5 about it; before 0 to v + 0.5, which
  left it out). Set-ups: `bb2/outputs/bb2_prop_setup.txt` (F6 hub; pitch 6, chord 8, thickness 6, camber 6,
  skew 6, rake 5 points; 1.33, 0.13, 0.14, 0.62, 0.35, 0.34 % of the ranges; `bb2_prop.txt` and its section
  files extracted again with the axis fix: the rows beyond r/R 0.99 moved, chord up to 2.9 % of its range at
  the tip, the rest within 0.02 %), `bb2/outputs/bb2_hull.txt` (from `bb2_hull_params.dat`: the BB2 hull IGES is a quarter model, no
  closed sections; orders 3 and 3), `outputs/modified/Wind_Tunnel_Rudder_design_space.json` (LE 2, P2x 3, the
  others 2; within 0.73 % but P2x 3.4 %). SUBOFF's STEP: orders 4 and 7 (0.50, 0.38 % of r*). In the window:
  hull 3 s, BB2 propeller 46 s, wind-tunnel rudder 107 s. The 1-segment fit is least squares and the pick by
  the largest difference, so a curve's error need not fall with every added point (BB2 pitch: 6 points 1.33 %,
  8 points 1.52 %). Tests: blade 82 with the GUI and the CAD (69 here), tool 52 (28 here), hull 62 (36 here),
  wing, files.

## Next steps

- Wings and fins: the tips (next; user, 8 Oct 2026), then a check on the user's machine of the wing, the fin and
  the vehicle with them (pythonocc-core).
- Vehicle: the placement is not loaded back yet (the set-up JSON records it); blended wing bodies later.
- Review the blade in the tool: default segments, orders and free variables (`blade_control.py`); camber and
  rake fit poorly as 2 segments.
- Propeller: `Initial_sampling.py` and the pipeline still build the cylinder hub's DRDC files; a conic hub there
  when a campaign needs it. The BB2 assembly (hull, bb2_prop.txt, stern planes) in the Vehicle tab.
- Input files, steps 3 to 5: the planar rudder (the wind-tunnel rudder fitted into a component file; the
  modified-below-H model as `rudder-h`), the assembly file in the Vehicle tab (Load and Save assembly), then
  `Initial_sampling.py` and the infill reading the assembly (claude/xgeom-file-formats.md).
- Optimisation: sample a saved set-up (`load_component`, `Initial_sampling.py`), screen
  designs with the problem checks, couple to CFD. `Initial_sampling.py` varies pitch and chord only (x_blade_new's
  form); other parameters flagged Y are reported, not sampled.
- Rudder: the tip tilt and cap parameters into the design space. If the mesher draws the tip's rounded edge
  the way it drew the TE: a band of its own for it (as the TE strip now is for the TE's round).
- Open items: section-fit bounds (X2 on its bound) and an LE-radius constraint; plain-named rudder outputs are
  overwritten by runs at other heights; Fig. 5 of the plan still labels the tip y = 218.07.
- From CAD for the other components (the wing, the fin, the planar rudder: their extraction codes) once the
  blade, hull and rudder runs are checked in use (the user, 13 Nov 2026).
