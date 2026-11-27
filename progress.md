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
- **Propeller blade** (`blade_modify.py`, `blade_control.py`, `msc_blade.txt`): a blade always starts from
  a component file (D, blades, root r/R, hub height, section table, and the radial table r/R, P/D, c/D, t/c, f/c,
  skew, rake/D). Design variables: the blade radius, hub radius and hub height (the curves stretch over the span
  between the radii), and each distribution as a Bezier curve over r/R fitted to the file, of 1 segment (C0 .. Cm)
  or 2 segments (the form of `para_control_bez_updated.py`: P1 .. P4 and P4 .. P7, zero slope at P4, the weights
  w23 and w56 as variables; the default for all six). The MSc code builds the geometry (`para.py` for the XCAD
  points, `BladeSurface`, `tip_surfaces_new`, `hub_new` and `X_CAD_new` for the DRDC five-surface IGES with the hub
  sector of the design's height, plus STEP). `msc_blade.txt` is the MSc blade as the pipeline builds it:
  its polynomials at 61 stations from `x_blade_new.R_VALUES[0]` (r/R 0.18), hub height 0.625 m (hub_new's).
  XCAD sections (53 points each): their number and how many lie in the tip band, closer together towards the tip
  (default 56, 14 from r/R 0.93, evenly spaced below); the clearance check keeps x_blade_new's stations.
- **What varies, from the file**: var lines (a set-up: each variable free or held); in a table file (no curve
  or var lines) vary lines `# vary <name> Y|N [low high]` for pitch, chord,
  thickness, camber, skew, rake, radius, hub_radius and hub_height. Y: a design variable (a distribution as a
  fitted Bezier curve, its variables free; low high bound its control values, or the radius or height); N: kept
  as the file has it (a distribution exactly, the cubic spline through its rows, `FileCurve`, no variables); a
  name not listed is N (the reader, `blade_vary.py`, uses the standard library only). Nothing Y: `blade_control.py`
  writes only the case `original` and its CAD. A table file with no vary line: every distribution fitted, `FREE`
  free. `msc_blade.txt`: every distribution as its table has it, nothing free. The
  same file drives the MSc pipeline: `para_control_bez_updated.py` takes para_control 0 (the original blade: pitch
  and chord unchanged), 2 (chord), 3 (pitch) or 7 (both), and the cap of P4's pitch (max_pitch, 1.4);
  `x_blade.py`, `x_blade_new.py`, `X_CAD_new.X_CAD_from_design` and `cad_worker.py` pass both on.
  `Initial_sampling.py` reads `BLADE_PARAMS` (`msc_blade.txt`): its vary lines, else its free var lines
  (`blade_vary.read_free`), pitch and chord Y or N give para_control;
  a range bounds pitch p1y, p7y and max_pitch, or chord p1y and y4 (the positions, fractions and weights keep the
  code's bounds); only the flagged parameters' variables are sampled and written (pitch's 6, then chord's 5);
  nothing Y: the original blade's CAD only (case 0, `geometry/original`).
- **Pipeline folder** (`Prop_hull`, the CFX campaign: `gp_infill_from_training_data.py`, `run_campaign.py`, mesh,
  CFX and training scripts): the files it shares with X_geom are kept identical. `Initial_sampling.py` has
  Prop_hull's campaign changes (`--round r` runs infill mode for round r; case number 100 x round + row,
  `pipeline_config.case_id`; resumable from the accepted file) and X_geom's vary lines; `pipeline_config.py` is
  Prop_hull's (`DATA_DIR` Training_3d). Prop_hull also holds `blade_vary.py` and `msc_blade.txt` (until 20 Nov
  2026 `msc_blade_params.dat`).
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
- **Hull curves** (`hull_modify.py`, `suboff_hull.txt`): a hull starts from a station table (half-height r
  and half-width r' at stations, the nose, middle-body, tail and cap lengths). Each part of each view is a CST
  curve: a class exponent (the nose's bluntness, the tail end's shape) times a Bernstein polynomial of control
  values, joining the middle body with zero slope. Sections are ellipses (circles where r = r'). Design vector:
  the lengths and the curves (exponents and coefficients) free by default (13 slots for SUBOFF), the radii and
  the cap held. Output: the
  design as a component file (the set-up) and as JSON, stations, hydrostatics, section points, check plot, and
  STEP (one NURBS face per part: the profile B-spline times the rational circle, so sections are exact
  ellipses).
- **Wing and fin** (`wing_modify.py`, `wing.txt`, `fin.txt`): a wing or a fin starts from a component
  file (the span, the airfoil by name, the pitch axis, and per span fraction eta the chord, sweep (x of the LE),
  rake (z of the LE: the dihedral), pitch (twist), t/c and f/c; optional change columns). Sections: the named
  airfoil (its thickness with the TE closed, and its camber line; a symmetric airfoil gets the NACA 4-digit camber
  line, its maximum at `camber_xc`) scaled to t/c and f/c, plus Bezier changes of the thickness and the camber line
  (Bernstein polynomials of degree k + 1 along the chord, zero at the LE and the TE; k = 3 change points by
  default, 0 to 8), pitched about the pitch axis (0.25 c by default) and moved by sweep and rake. Design
  variables: the span (the curves stretch over it) and each distribution and change as a Bezier curve over eta
  (1 or 2 segments as the blade's; 4 control points, 3 for the changes), the root's sweep and rake pinned at 0.
  Output (`outputs/wing`, `outputs/fin`): the design as a component file (the set-up) and JSON, the XCAD file (41
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
- BB2 (one-off, user, 22 Oct 2026; `bb2/`, the scripts take the input paths): BB2's propeller cut
  at 34 radii into `bb2/outputs/bb2_blade_params.dat` (blade_modify's format, every vary line N), a
  section table per r/R and one at r/R 0.7 (`bb2_sections.py`); one blade on the conic hub and cap of the propeller drawing
  (the root runs 0.1 m into the hub, so the hub cuts the root sections) and the aft half of the hull rebuilt with
  `hull_modify` (tail order 5, within 0.96 mm of the IGES), fused into one valid solid (`bb2_build.py`). The blade
  takes the section shape per r/R (`SectionsBladeSurface`): within 0.9 mm of the original at 44 cuts; X_CAD_new's
  single table is off by up to 23 mm at the root (the shape changes along the span). The outputs carry confidential data.
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
- BB2's aft half: the hull and the stern plane as set-ups (13 Nov 2026; the user: "the hull and the rudder is
  incorrect ... We will only build the aft half of the sub for now, so just the hull aft half, rudder and
  propeller"; the casing's non-round sections and the decks later; the rudder as a planar rudder component,
  the user's choice). `bb2/bb2_aft_setup.py <quarter.iges> [out_dir]` from BB2_H3_DRDC_Q2R_quarter_hull_half_
  upstream.iges (now in bb2/, with BB2_Drawings.pdf; byte for byte the hull.iges of 22 Oct). The IGES's hull is a
  body of revolution (r the same all round each section within 0.002 mm, x 44 to 35078 mm, the hub included).
  The earlier bb2_hull.txt (from bb2_build.py's table: a dummy 3 m nose, L 37.148 m) is replaced:
  `bb2/outputs/bb2_hull.txt` is the aft half alone, no nose (its front the flat cut at mid-length), the middle
  body 12.875 m (r* 4799.81 mm), the tail 20.273 m to the hub's forward end (r 797.22 mm, F6's hub base: the hub
  is the propeller's), L 33.148 m, the tail's CST order 3 within 0.96 mm of the IGES (its stations
  `bb2_hull_aft_params.dat`). hull_modify takes Ln 0: no nose curves and no nose variable, the radius r* at x 0,
  the solid's flat front face (valid, 4 faces; the half body exactly half), S without the cut face; xgeom_hull
  shows no nose row and says "aft half"; fit_cst clips psi to [0, 1] (round-off past the tail's end gave NaN).
  `bb2/outputs/bb2_sternplane.txt` (kind rudder) with `bb2_sternplane_section.dat`: the stern plane as a planar
  rudder, its sections normal to the span in its 45 deg plane, from s 3061.4 (its LE line meets the hull: the
  root section lies outside the hull, the vehicle extends it to the hull along the TE line) to the cap's foot at
  s 6811.5 (the 236 mm rounded cap left off, the planar tip flat): span 3750.07 mm, chord 3535.01 to 2766.12 mm
  and sweep 0 to 760.08 mm (straight, 2 points each), rake, pitch and camber 0 (the sections symmetric to 1e-5
  of the chord), t/c 0.1606 to 0.1628 (4 points); the section the sections' mean thickness shape (NACA
  0016-like, its blunt TE's round kept); no change points (8 of them fit only 0.007 % of the chord better:
  the TE's round is about 35 mm thick at every chord). Rebuilt: the planform within 0.001 mm of the CAD, the
  sections within 0.5 to 1.5 mm (the most at the TE's round, near the root and the tip). xgeom_from_cad:
  best_wing_forms and wing_setup (a planar table's best forms and set-up). The tool: a planar rudder in the
  Rudder slot (xgeom_wing.PlanarRudderAdapter, role rudder): `--file` with a planar rudder's file, or Load ...
  of one in the Set-up box, puts the rudder on the planar model (the wing's panel, "CAD rudder"); Load ... of
  a <case>_design_space.json in its file box puts it back on the model modified below H; its JSON set-up says
  rudder_planar; From CAD stays the rudder-h one. xgeom_vehicle.rudder_part takes a planar rudder (wing_part
  and its TE line's slope). Placed as BB2's (4 rudders at 45 deg, TE rake 0, the TE at x 29.773 m from the cut)
  its root LE rests 2.5 mm from BB2's; the vehicle CAD of the aft hull, the four stern planes and the whole
  propeller on the F6 hub (prop_dx 0.576 m): one valid solid, V 1808.58 m^3, 56 s. _hull_modify_test.py's
  write_case check reads the component file (the JSON set-up went on 26 Oct). Tests: hull 65 with the GUI (38
  here), wing with the GUI (41 here), tool 52 (28 here), blade 79 with the GUI, hull_modify 14, hull 12, files
  35.
- BB2's positions in the Vehicle tab (13 Nov 2026; the user asked where the propeller and the rudders sit on the
  BB2 hull; as x/L, the entry below): Propeller "hub starts at x/L" 1 (the hub's base on the hull's end; the
  default), D 5 m (BB2's own; the default); Rudder "root TE at x/L" 0.89819 (BB2's TE line meets the hull 29.7733 m
  aft of the cut, L 33.1480 m), rudders 4, first at 45 deg (an X), scale 1, TE rake ticked -0.135 deg (BB2's TE
  leans 0.135 deg forward at the top: x = 0.002352 s + 5299.895; rake 0 pitches the rudder 0.13 deg and moves its
  root LE 2.5 mm). Typed so (tick TE rake first: its box is disabled until then; Fin "fins" 0 and Wing "wings" 0:
  the tool's vehicle carries one fin and a wing pair by default, BB2's aft half none), the placed stern planes lie
  within 0.1 mm (x) and 0.03 mm (radius) of BB2's at the root, mid-span and tip, LE and TE, the root chord parallel
  to the axis, and the blades' plane falls on BB2's generator line (576.1 mm aft of the hub's forward end, drawing
  F5's 577). The tool does not keep the placement between sessions yet (the assembly file, step 4).
- Vehicle tab positions as x/L (13 Nov 2026; the user: "we should enter these positions as normalized length of the
  hull ... like hub starts at 1 (if hull aft is counted as 1), then rudder TE sit at 0.85 or smt"; his BB2 build in
  the GUI had the hub slightly off the hull, not one solid): the positions along the hull are fractions of its
  length, 0 the nose (a body without a nose: its front, the cut), 1 the hull's aft end, shown to 5 decimals (0.3 mm
  on BB2's 33.148 m). Propeller "hub starts at x/L" (prop_x, 0 to 2) puts the hub's forward end there on the axis,
  the blades at the hub's own offset scaled with the propeller (the old "plane at x - L" placed the blades' plane,
  so the hub's start moved whenever D scaled the propeller); its default is where the hub sits flush on the hull
  (xgeom_vehicle.hub_seat: the last x where the hull's section is as wide as the hub's forward end; 1 on BB2; on
  SUBOFF, whose tail closes on the axis, where the tail narrows to the hub). D defaults to the design's own
  diameter when it is no larger than the hull's depth or breadth (else half of that, as before). The gap: D had
  defaulted to 4.8 m (half BB2's depth), which scaled the 5 m propeller by 0.96 about its plane and left the hub's
  base 23 mm aft of the hull's end; at x/L 1 the hub now starts on the hull's end whatever D is (D 4.8 typed: one
  valid solid, V 1808.36 m^3). The view's text says where the hub starts (on the hull's aft end, on the hull, so
  many mm behind the hull's end or where the hull has closed on the axis: not joined). Rudder "root TE at x/L"
  (rudder_x), the fins' and wings' "root LE at x/L" (fin_x, wing_x), 0 to 1. The assembly file's placement keys:
  prop_x and rudder_x replace prop_dx and rudder_dx (a file with the old keys is refused), fin_x and wing_x keep
  their names but are x/L now (they were m from the nose): for Eddie's web app. vehicle_cad: a fuse that leaves the
  parts apart is tried again at fuzzy values of 0.01 and 0.05 mm (FUZZY_RETRY_MM; SUBOFF's default rudder TE,
  rounded to 5 decimals, fell 0.018 mm ahead of the tail's join with the cap and the fuse gave 4 solids); parts
  truly apart stay apart, and the message says when the larger value was used. BB2 typed as x/L (above, fins and
  wings 0): the vehicle CAD one valid solid, V 1808.60 m^3, 63 s. Tests: hull 66 with the GUI (39 here), wing 54
  with the GUI (41 here), tool 52 (28 here), blade 79 with the GUI, files 36, hull_modify 14, hull 12.
- Tip patch's corners on the strip lines, the blade's layout again (17 Nov 2026; the user, after discussion with
  his group: for meshing the rudder has to follow the propeller blade's five-patch topology more closely; his
  sketch: the tip patch's corners on the strip lines under the tip, the strips' top edges diagonals up to the
  tops of the LE and TE lines; strip widths computed for the best result, the corners' drop picked from trials;
  BB2's stern plane the test case, its top kept exactly). `xcad_loft.loft_loops(loops, strip_frac=None,
  tip_drop=1.5)`: the tip patch's four corners sit on the strip lines under the tops of the LE and TE lines by
  `tip_drop` times their distance from those tops along the chord (a pair gives the two ends their own; 0.5 to 2
  tried on BB2, all valid without folds, 1.5 and 2 the smoothest); the four patches along the span end on sloping
  top edges, the rows turning onto them over a band under them; the tip patch covers the cap and the sides above
  those edges, fitted to wraps (up the A side to the rim, over the cap, down the B side) through a mean value map,
  the rim on two double-knot lines of the patch. `strip_frac` None (the default) makes each strip as wide as its
  edge's rounding (the wind-tunnel rudder: 11.2 mm to each side of the LE, the TE's 0.75 mm round at the TE;
  BB2 at 0.07: 190 mm and the 15 mm round). Folds found on the wind-tunnel rudder at the tops of both edge lines
  and fixed (wrap stations closing in on the tops, `APEX_STEPS`; the wraps between a pinch and the loops' first
  points scaled from the first whole wrap, `_into_pinch`; the mesh's boundary parametrized on the fitted edges,
  `_on_edges`; no line leaving an edge at less than `EDGE_ANGLE` 15 deg); the result's `tip_angle` is the smallest
  angle between the patch's two families of lines (0 for a fold; 14.9 deg on the rudder, 14.5 on BB2), the note
  and the tool's status line say so (orange when it folds). Very blunt cap ends on made-up wings still fold the
  patch, reported (parked by the user). The patch rippled along the rudder's 0.7 mm edge radius, 1 deg either
  side of it: from the rim the cap runs up a straight wall 0.4 to 1.9 mm long, round the 0.707 mm arc and over
  the crown, two jumps of curvature on each side, neither on a line of the patch. The user allowed a small
  change in a small region: where most wraps jump in curvature, every wrap's cap part is averaged over
  `CAP_BLEND` 0.5 of the cap's tightest radius (0.29 mm here), twice, at most `BLEND_SHARE` 0.08 of the wrap's
  length; the round moves in by 0.013 mm (the cap model itself lies 0.104 mm from the CAD), the patch's normals
  within 0.4 deg of its data over 99 % of the band (0.9 before), its curvature across the round one smooth hump;
  BB2's dome is not touched (the note and `tip_blend` say when a cap was blended). The wind-tunnel rudder: six
  faces, valid, 583,362.83 mm^3, the tip patch 150 x 150 poles within 0.005 mm of its data; against the STEP the
  LE strip within 0.0008 mm, the centrals 0.0029, the TE strip 0.0043, the tip 0.104 (the cap model's own);
  `Rudder/outputs/wind_tunnel_rudder_full.igs` and `_half.igs` written again. BB2's stern plane
  (`bb2/bb2_sternplane_loft.py`: every loop point put back on the original surface by closest point, the rim at
  s 6811.99, rings 0.01 to 1 of the cap's profiles; 24 min): within 0.012 to 0.019 mm of the original over the
  exposed span, 0.036 at the TE-tip corner, the cap cuts within 0.046, the patches point by point within 0.005
  (LE strip) to 0.040 mm (tip patch, rms 0.003); outside the hull within 1.1e-4 of the original by volume; the
  set-up files in `bb2/outputs` written again (vehicle 1.808033e12, vehicle6 1.808732e12 mm^3, the quadrants as
  before). `_xgeom_hull_test.py` measures its root-extrusion volumes with the Gauss-Kronrod rule (the adaptive
  one does not settle on prism faces). The rudder set-up JSON's settings gain `tip_drop`; no component or
  assembly file changed. Tests: tool 61 with the GUI, wing 54, files 51, hull 64, blade 77 (with the GUI),
  modify 136.
- Tip cap for the planar surfaces (18 Nov 2026; the user: "everytime I load in bb2_sternplane.txt it creates a
  flat top instead of creating the actual bb2 rudder"; his choice: fit a cap model and carry it in the set-up).
  The planar rudder had no cap: the set-up written on 13 Nov left the rounded cap off, and a stack of horizontal
  sections is closed flat. BB2's cap is not tip_cap's wind-tunnel family (a wall, an edge radius in mm, a crown):
  every cross-section is a dome 0.25 times the local half-thickness tall with a flat crest, so `tip_cap.py` has
  a second family, `ScaledCap` (the top k(u) w(u), k a cubic Bezier along the chord; each cross-section the
  superellipse |w|/w0 = (1 - (h/H)^q)^(1/p); dimensionless; pinched by itself at the LE and a closed TE) with
  `fit_scaled`. The planar component files (`wing_modify.py`, `xgeom_files.py`) take `cap scaled` (cap_k_le,
  cap_k_te, cap_b1, cap_b2, cap_p, cap_q) or `cap radius` (the wind-tunnel family's seven numbers in mm) and
  the setting `cap_loops` (40); `write_case` appends the cap loops to the XCAD file and xcad_loft's tip patch
  covers them; the 3D sections tab draws them, the Vehicle tab's rudders carry them; the cap's numbers are held.
  `bb2/bb2_aft_setup.py` fits BB2's cap from 48 cuts of the IGES over the planar tip section (its tip at its
  own `S_TIP` 6811.5 now): k 0.2511 to 0.2699, bow -0.0047, -0.0193, p 2.44, q 2.32, the IGES cap within 1.9 mm
  (rms 0.40), and writes it into `bb2/outputs/bb2_sternplane.txt` (`bb2_cap_check.png`; the hull's files
  unchanged, the set-up's other numbers moved by 1e-5 from the better cuts). The rudder built from the set-up:
  one valid solid of six faces, the tip patch within 0.06 mm of the cap loops, no fold, 9 s; against the IGES
  the cap within 1.27 mm (rms 0.31), the sides within 1.50 as before. The lofted stern plane of the set-up
  files stays the exact one. `claude/xgeom-file-formats.md` has the keys (for Eddie: optional keys and one
  optional setting of the rudder, fin and wing files; old files read as before). Tests: tip_cap (the scaled
  family), wing 62 with the GUI (the cap keys, the loops, the loft with both families), files 51, hull 64,
  tool 61.
- The wind-tunnel rudder as a planar set-up (18 Nov 2026; the user: "Can we also build the wind tunnel rudder from
  set up text file?"; his choices: the tilt as a key, the TE round scaled with the chord for now). The planar
  files take `tip_tilt_deg` (`wing_modify.py`: the horizontal sections to half a step under the tip's LE, then the
  tip section tilted about its LE, the cut of the body carried on past the tip by the tilted plane out to the TE
  line, `tilted_tip`; the cap sits on it). `python xgeom_from_cad.py rudder_planar outputs` (numpy only) fits the
  extraction's default run into `outputs/wind_tunnel_rudder.txt` (with `wind_tunnel_rudder_section.dat`): 21 rows
  to the LE top at 200 mm, straight LE and TE lines, the rows' mean section shape, t/c 0.16134 to 0.16271, the
  tilt 3.576626 deg and the extraction's radius cap. Built from it in the tool (CAD rudder, 11 s): the tilted
  tip's TE top and chord as extracted, six faces, valid, no fold, 583,299 mm^3 (the extraction-based loft
  583,363); against the STEP the LE strip within 0.022 mm, the central panels and the TE strip within 0.085 (the
  scaled TE round: about 0.85 mm at the root, 0.65 at the tip, against 0.75), the tip region 0.107 (the cap
  model's own 0.104). Tests: wing 69 with the GUI, files 51, tool 61, hull 64.
- The Vehicle CAD takes the blade panel's part (19 Nov 2026; the user: "although I choose 1/6 blade strip in
  the vehicle CAD the whole propeller showed up"). The Vehicle CAD builds every component with
  `build_task(case, half=False)` (full bodies), and for the blade that had set the part to `whole` whatever
  "CAD writes" said (and wrote `part whole` into the blade's case file). Now the blade keeps its settings' part
  and passes `solids` to `blade_modify.write_cad` only (one blade on 1/Z of a cylinder hub comes as a solid
  too, not the DRDC surfaces; the component file gets no such key); the build's result carries the part and
  its number of blades, `vehicle_cad.run_vehicle` hands it to `vehicle_setup` (`cad_parts`: the propeller
  record's `part` whole, half or sector and `part_blades`), and `build_from_setup` takes the STEP's solid
  whatever the part and fuses it with the hull and the appendages as before; the status line and the report
  name the part (`part_text`). A STEP of DRDC faces only (an older set-up) still gives the whole propeller,
  and the record says so. The blade panel's note says the Vehicle CAD takes its part. The Vehicle tab's lines
  still draw all Z blades. Cloud (the blade's CAD through a pythonocc stand-in over cadquery-ocp): SUBOFF, 4
  rudders and the MSc blade's 1/5 sector, one valid solid, V 0.70169 m^3 (0.70199 with the whole propeller),
  the propeller part to one side of the axis, 99 s. For Eddie: in `<case>_vehicle_setup.json` the propeller
  record's `part` now names the part (it was always `whole`), `part_blades` is new. Tests: hull 66 with the
  window and the vehicle CAD (the set-up's record; the 1/Z through build_task(case, half=False) in the
  vehicle), tool 61, blade 77, wing 69.
- Not tangent at H: two lofts (19 Nov 2026; the user: make the tangency at the joint of the unmodified and the
  modified parts a feature that can be turned off, loft the two parts separately and combine the two solids,
  nine patches). Each curve already had its join (G1, or G0 with C1 a free variable) and a checkbox in its
  box; his choice: a global switch too, the per-curve ones kept. `xgeom_tool.py`: "tangent at H" in the
  rudder's Region box (`App.on_tangent_all`; `RudderAdapter.set_tangent_all`, `tangent_state`): off frees
  every C1 where it is (the shape does not move), on puts every C1 back on the original's tangent; the label
  says how many curves are tangent when they are mixed. The CAD: with any curve G0,
  `xgeom_rudder.xcad_task` lofts with `xcad_loft.loft_split(loops, n_modified - 1)`: the loops up to the H
  section and those from it on, each by loft_loops on its own, both through the H section exactly (loft_loops
  holds its end loops) and with the whole stack's strip widths (`loft_loops(widths=)`, the rule moved to
  `_strip_widths`), so they end on the same four curves and the seams run on across H; their faces (four
  patches below, four and the tip patch above, the flat root; the two flat faces on the joint left out) are
  sewn into one solid, ten faces (the half body cut from it, nine). `LoftResult.joint`; the status line says
  "not tangent at H: lofted below and above H on their own, joined there, nine patches". Tangent designs keep
  the one loft. Why: the wind-tunnel rudder with every curve free and C1 moved off the tangent (LE 10 mm
  forward, P2z +0.01, P1z -0.008), one loft: its rows 0.19 mm off the patches about H and two false steps
  eased out of the loops; two lofts: valid, the rows within 0.0001 mm, no steps, the tip patch as before,
  13 s against 18. Limit: the joint has to stay under the tip patch's corners (about 190 mm on this rudder; H
  185 builds, 192 is refused with the reason: lower H, or tick tangent at H). No file format changed. Tests:
  tool 68 with the window, wing 69, hull 65, modify 136.
- A planar rudder's file loads as the original modified below H (19 Nov 2026; the user: loading BB2's or the
  wind-tunnel rudder's set-up showed a different panel, without the tangency switch; after loading, the panel
  must look the same, with the same features; his choice: the loaded rudder becomes the original). Loading a
  component file of kind rudder (`--file`, or Load ... in the rudder's Set-up box) used to put the Rudder
  slot on the planar model (the wing's panel). Now `xgeom_from_cad.rudder_from_planar` writes the planar
  rudder's sections as an extraction writes a CAD's, into `<name>_modified_below_H/` beside the file: 61
  horizontal sections of 121 points (`PLANAR_SECTIONS`, `PLANAR_POINTS`: the planar builds' own layout; at
  101 points a side the tip patch over BB2's cap folds, in the planar build too), the tilted tip section and
  the cap loops after them in `sections_xcad_with_cap_*_m.dat`, `corner_points.dat`, the stack file (its
  sections at 101 points a side, `PLANAR_FIT_POINTS`: BB2's blunt TE round needs them for the flank the fits
  look for); fits every section (`fit_section_table`) and the laws (`best_rudder_orders`), and writes the
  set-up `<name>_original_design_space.json` there: the rudder unchanged, its sections' TE radius, 121 points
  a section, H between 0.1 and 0.4 of the top horizontal section (`H_FRACTIONS`) and at 0.4. About 25 s the
  first time, off the GUI thread (`App._load_planar_rudder`); while the file and its section file are
  unchanged (their digest in `source.json`) it is read back in a fraction of a second. The tool's cases of it
  go to that folder. `xgeom_rudder.load_space` now also takes `num_pts` from a set-up. `python
  xgeom_from_cad.py rudder <file>.txt` does the same without the window. The planar rudder's model
  (`xgeom_wing.PlanarRudderAdapter`) stays in the code; the tool no longer shows it for rudders. Checked: the
  baseline writes the planar sections themselves (1.4e-5 mm on the wind-tunnel rudder, 7e-4 mm on BB2's
  plane); the window after Load ... of bb2_sternplane.txt shows the panel modified below H (H, tangent at H,
  the section control points); its CAD one valid solid of 6 faces, the planar build's volume to 1e-5, the tip
  patch as the planar build's (no fold), not tangent at H two lofts of 10 faces; the wind-tunnel rudder's
  583.30 cm^3 (the planar build's 583.30). No file format changed (for Eddie: component and assembly files
  as before; the web contract test passes, its 4 notes as before). Tests: wing 71 with the window (the
  conversion, its cache, a changed file made again; the Rudder slot loading the planar file both ways), tool
  68, hull 65.
- The TE strip as wide as the tip patch's edges need (20 Nov 2026; the user, for meshing: the tip patch's
  front edge, at the LE, and its back edge, at the TE, carry the same number of nodes, so the cells spread on
  the long LE edge and crowded on the short TE edge; make the TE strip wide enough that the two are close in
  arc length). The TE strip had been the TE's round itself: on the wind-tunnel rudder the tip patch's LE edge
  was 30.71 mm long, its TE edge 3.20 mm (BB2's plane from its planar set-up: 453.4 against 75.6).
  `xcad_loft._match_te` now widens the TE strip until the TE edge (the TE strip's top row) is as long as the
  LE edge to 0.1 % (`EDGE_RTOL`), never narrower than the round; the LE strip as before. The wind-tunnel
  rudder: the TE strip 10.59 mm to each side of the TE (1.26 before), both edges 30.71 mm (OCCT on the STEP:
  30.711 and 30.709), the tip patch's TE corners 12.4 mm under the TE's top (1.0 before); BB2's plane: 131.7
  mm (25.3), both 453.4 mm. Without a cap the flat top's two ends are matched the same way (the TE strip as
  wide as the LE strip); a pair `strip_frac` (LE, TE) is still used as given. Two lofts (not tangent at H):
  the upper loft matches, the lower takes its widths and knots. With the round inside the wider strip, its
  ends (a jump of curvature) made a fit on the plain knots ring: the strip's rows 0.4 deg off the sections'
  curves on the rudder, 0.7 on BB2's plane (0.04 with the narrow strip). `_strip_knots` halves the knots
  across the strips where a row's fit turns off by more than 0.05 deg (`STRIP_ANGLE`, mirrored about the
  middle, up to `STRIP_POLES` 130): 94 poles across on the rudder (67 before), within 0.042 deg. Results: the
  rudder from the extraction valid, 6 faces, 583,362.85 mm^3 (583,362.83 before), the rows within 6.6e-5 mm
  of the patches, the tip patch within 0.0020 mm (0.0051 before), its lines at 15.6 deg or more (14.9), 12.6
  s (9.3); modified below H: 6 faces tangent, 10 not tangent (joint 50), 9 for the half body, both edges
  30.0 mm; BB2's plane the same volume, the tip patch 0.063 mm, 11 deg. `LoftResult.tip_edges` (the two
  edges' lengths) and `strip_knots`. `Rudder/outputs/wind_tunnel_rudder_full.igs` and `_half.igs` written
  again (6 faces each read back, 583,362.85 and 291,681.41 mm^3). BB2's stern-plane CAD files in
  `bb2/outputs` (17 Nov) still have the narrow TE strip until rebuilt (`bb2/bb2_sternplane_loft.py`, its
  docstring updated). No file format changed. Tests: tool 70 with the window (the CAD's two edges measured,
  a flat top, the ringing on plain and refined knots, one fraction, a pair, two lofts), wing 71, hull 65,
  modify 136; the web contract test passes.
- Bookkeeping (20 Nov 2026; the user: the code base is growing; follow Eddie's interface guideline as far as it
  does not compromise the tool). Eddie's two files are in the repository: `_xgeom_web_contract_test.py` at the
  root (it was the untracked `_xgeom_web_contract_test 1.py`; it has to sit beside the modules) and his note as
  `docs/keeping-interfaces-stable.md`. The contract test runs before every change is handed over (18 s on the
  user's machine). His open requests 1 to 3: the samples `fin.txt`, `wing.txt` and `msc_blade.txt` re-saved as
  the tool writes them (one default line each, `# cap_loops 40` or `# part sector`; the geometry and every
  variable the same, each file a fixed point of open and save; the contract test's notes on them gone), a
  `.gitattributes` (`*.txt text eol=lf`; the repository's .txt files are LF already, so nothing to
  renormalize), and the rudder's set-up JSON without its `"written"` time stamp (`xgeom_rudder.space_record`;
  an unchanged set-up saves to the same bytes; older set-ups still load, the key is ignored). Request 5: the
  `bb2/` scripts' comments and docstrings no longer name the propeller or give BB2's dimensions (positions,
  radii, angles, the drawing and report numbers; the method's spacings and tolerances kept), their plot titles
  and output headers say "BB2's propeller", their code is unchanged (its constants still hold the drawing's
  values); the propeller's name is also gone from `claude/xgeom-file-formats.md` (its examples are now the MSc
  blade and SUBOFF) and from the 22 Oct entry above. Still open: 4 (the blade's root extension in preview()),
  6 (para_control_bez_updated.py imports matplotlib.pyplot at the top, through x_blade_new), 7 (old assembly
  files). New: `docs/module-map.md`, every module with what it does, where it lives (X_geom, the Rudder
  folder), its self-checks and whether the web app imports it; `README.md` points to the docs. Tests: tool 71
  with the window (an unchanged set-up saves to the same bytes), files 51, wing 71, blade 77, hull 65; the
  contract test passes with one note (request 6).
- **Breaking**: component files only (20 Nov 2026; the user: "get rid of all old format set up files. From now
  on it's just the new format"; his choices: old files refused with a clear message, the converter and the old
  set-up loaders removed; Initial_sampling from the free variables; `legacy_kind` kept so the web app can
  recognise and refuse old files). Old files: a parameter file (`# XGeom <kind> parameters`, .dat) and the
  set-up JSONs of the blade, wing, fin, hull and planar rudder (`<case>_design_space.json` with `params_file`,
  `rudder_planar`) are no longer read; `read_component`, `xgeom_files.load` and each module's `read_params` and
  `load_component` say `<path>: an old <kind> parameter file, which XGeom no longer reads (since 20 Nov 2026):
  it reads component files only ('# XGeom component' first)`; the tool still routes an old file to its
  component (`file_kind`), which says so, and `--space` takes the rudder-h model's set-ups only (its JSON
  stays). Deleted, 17 files: `msc_blade_params.dat`, `wing_params.dat`, `fin_params.dat`,
  `suboff_hull_params.dat`, `outputs/hull/suboff_fit_params.dat`, `outputs/hull/suboff_fit_design_space.json`
  (tracked), the tool's old saves `outputs/{blade,fin,wing,hull}/gui_params.dat` and `gui_design_space.json`,
  and BB2's three tables (converted first). Removed from the code (none is in the web contract):
  `xgeom_files.convert` and its command line, `legacy_header`, `Loaded.legacy`; `space_record`,
  `load_blade_space`, `load_wing_space`, `load_hull_space`; `hull_modify.SUBOFF_PARAMS` (SUBOFF from its
  equations in memory: `hull_modify.py --suboff`, and the hull adapter when `suboff_hull.txt` is not there);
  the adapters' `save_params`, `vary_state`, `space_record`, `save_space`, `load_space` and `legacy` (the
  rudder-h adapter's stay). In their place a table file: a component file with no curve or var lines (the
  geometry, vary lines when it has them, the table), which `write_params` of blade_modify, wing_modify and
  hull_modify now writes (`read_params` reads any component file's geometry and table); `load_component` fits
  its curves as the old files were fitted. The component writer writes vary lines when a table has them (a
  set-up has var lines, as before; the samples are unchanged byte for byte). The hull's From CAD writes its
  stations as `<case>_table.txt` (was `<case>_params.dat`). Converted with the old reader before deleting,
  each the same table, design and space to the last digit: `bb2/outputs/bb2_blade.txt` (the frame a note,
  `bb2_frame x0_mm .. theta_ref_deg ..`, which `bb2_sections.read_frame` reads; `bb2_build.blade_surface` gives
  the same blade at 400 points), `bb2_hull_table.txt`, `bb2_hull_aft_table.txt`, written and read by the bb2
  scripts now; and the SUBOFF case's set-up as `outputs/hull/suboff_fit.txt` (its design.json's hull within
  3e-13 m). `Initial_sampling.py` reads `msc_blade.txt`: its vary lines, else its free var lines
  (`blade_vary.read_free`: pitch or chord on when one of its variables is free); `msc_blade.txt` frees nothing
  of pitch or chord, as `msc_blade_params.dat`'s lines (all N) did, so para_control stays 0. Prop_hull needs
  the new `Initial_sampling.py`, `blade_vary.py` and `msc_blade.txt`. For the web app: no key is new or
  changed; the contract test passes (its one note as before). Tests: files 60 with the window (old files
  refused by name, table files, the tool's refusals), blade 82 with the window and the CAD, wing 71, hull 65,
  hull_modify 17, tool 71 with the window; Initial_sampling's checks run with OCC stubbed (the CAD left out).
- `temp_rudder_cad.py` (20 Nov 2026; the user: the new five patches, always two lofts, the modified part and the
  unmodified part, and sections of any number of points). Always two lofts: `xcad_loft.loft_split` at the top
  modified section, H (it had been one `loft_loops` through every loop): the modified sections up to H, and the
  section at H with the original's loops above it (its horizontal sections more than half its section spacing above
  H, the tilted tip section and the cap), each lofted with the five patches, the same strips and the same knots
  across them, sewn with the flat root into one solid of 10 faces; the surface may turn at H as the modified sections
  do. Any number of points: each modified section is put onto the original's layout at its height (`Layout`: the arc
  fractions of each side of the original's horizontal sections, interpolated by height, taken on the section's own
  smooth curve, `xcad_loft._SmoothLoop`), the LE first and the TE, the point farthest from the LE, at the original's
  TE index (99); a section given the other way round is turned round, an open one closed, repeated points dropped,
  and one already on the layout comes back unchanged. Before, a section of another point count was spaced on cosine
  arcs, which is not the original's layout (cosine in x): the wind-tunnel rudder's sections below 80 mm given at 121
  points read as a step of 3.58 mm at H, eased over 178 mm of the span, the solid up to 0.125 mm off the original
  (582,894 mm^3 against 583,363), and loops given the other way round gave a broken solid. Now: valid, 10 faces,
  583,363 mm^3, within 0.017 mm of the original (as closely as 121 points describe the TE's round: 151 points 0.005
  mm, 260 points 0.003 mm); at 200 points the same as one loft (0.0000 mm near H). A design from the tool (the LE
  moved, not tangent at H, 80 mm; the tool lofts it as two parts too): its modified sections through the script give
  the tool's solid (605,404 mm^3, the same distances to the points), and at 121 points within 0.017 mm. Refused with
  a message: a modified file with sections that are not horizontal (how many, and the first), fewer than two sections
  or two at one height, nothing of the original above H (and `loft_split` refuses a joint too close to the tip).
  Tests: tool 73 with the window (`test_temp_cad`: a 121-point section onto the layout within 0.03 mm, its LE and TE
  exact; the original's own section kept; a reversed and an open loop give the same points; a file of 121-point
  sections without '#' lines, some reversed, some open: valid, 10 faces, joined at its 80th section, no step). In
  X_geom and the Rudder folder (byte-identical).
- **Blended wing body** (20 Nov 2026; the user: the last component, mostly the wing's parametrization, mirrored in
  the 3D sections and the CAD; camber, pitch, rake and thickness all in; his choices: a published underwater glider
  as the first example, standalone first (the Vehicle tab later), always smooth across the centre line; then the Hull
  tab offers two bodies, the cylindrical hull or the blended wing body). Kind `bwb` (`xgeom_files`: in PLANAR, its
  forms table, bezier1 and the new `smooth2`, code 3): the wing's file and model over the half-span (`span_mm`), eta
  0 the centre line. Every curve is flat there, so the body and its mirror image meet without a crease
  (`WingDesign.flat`, `WingParams.flat`): a 1-segment curve has C1 at C0's value (its v1 no variable: `slot_keys`,
  `curve_from_slots`), `SmoothTwoSegmentCurve` (smooth2) is two cubics joined C1 at a free joint u3, flat at the root
  by its form (keys v0 d1 d2 v2 u3 v3 d4 d5 v5 v6; `fit_smooth_two`, `fit_flat_curve`); bezier2 is refused; a table
  curve is `even_spline` (PCHIP through the rows and their mirror image: flat at 0, no overshoot). The example
  (`bwb_glider.txt`): Ye, Wang and Pan (2021, J. Northwestern Polytechnical Univ. 39(1) 85-92), Table 1's seven
  sections (NACA 0022, 0017, then 0008; span 1 m, centre chord 350 mm; the paper's 2.64 L is the sections' areas
  taken linearly between them); chord, LE and t/c smooth2 within 2.2 mm, 0.43 mm and 0.0019 of the paper's rows (one
  segment flat at 0 misses t/c by 0.034 with 4 points and 0.009 with 5: it drops from 0.22 to 0.08 by eta 0.18 and
  stays); camber, pitch and rake held at 0 (symmetric sections; a test design with 2 % camber, 3 deg wash-out, 45 mm
  dihedral and a reflexed centre body builds as well). The CAD (`write_cad(mirror=True)`): the half lofted with
  `ease=None` and a row of poles per section, then fused with its mirror image (`mirror_fused`): one valid solid,
  2.590 L (the sections 2.587 L), within 0.022 mm of the design's sections, 12 s; with half the half alone.
  `body_properties`: planform, volume, wetted area and centroid from the sections. Shared `xcad_loft.py` (also in the
  Rudder folder): an edge without a rounding gets a strip of 0.07 of the root chord, at most what the narrowest
  section has room for (24 mm did not fit the glider's 30 mm tip; unchanged wherever it fitted), and
  `loft_loops(ease=None)` (with the easing the blend read as a 3.6 mm step and the CAD sat 1.3 mm off, 2 % short;
  with 30 rows of poles for 41 sections 0.31 mm off). The tool: the Hull slot's Body box (`BODIES`: the cylindrical
  hull or the blended wing body, each keeping its state; Load ... or `--file` of either's file puts the slot on it),
  `xgeom_wing.BwbAdapter` (the wing's panel over the half-span, the smooth rows, C1 tied; planform, front view, 3D
  view and 3D sections with the other half; CAD BWB the whole body or the half; the report gives the volume, wetted
  area and centroid); the Vehicle tab places its parts on the cylindrical hull only and says so (`VehicleView.note`),
  Vehicle CAD refused meanwhile. For the web app: a new kind and form (additive); `BwbAdapter` has the wing's
  methods, with `segment_options` (0, 1, 3), `segment_text(n)`, `name`, `short`, `half_text` and
  `preview()["mirror"]`; `xgeom_vehicle.VehicleView.note`; WingDesign's field `flat` (default False). Tests: wing 78
  with the window (the sample, flatness of random designs, refusals, smooth2 saved and read back, the table form, the
  body's numbers, the CAD whole and half, the Hull slot's Body box and CAD BWB), hull 65, tool 73, blade 79 and files
  60 with the window; the contract test passes. Project doc `claude/bwb.md`.
- Tip patch: thin strips, its straight edges just under the tip (20 Nov 2026; the user, with a picture of the mesh:
  make the tip patch smaller by pushing its straight edges closer to the tip without compromising the CAD's
  robustness and smoothness; then, with a picture of his rebuilt rudder: "This did not go up enough at all, to
  make the angles work you can make LE and TE strips thinner if needed"). The tip patch's corners sit under the
  tops of the LE and TE lines by tip_drop times the strip lines' distance from those tops along the chord, so the
  strips' widths set the height. `xcad_loft.LE_MARGIN` 0.5 (1.5 before): the LE strip half the LE's rounding
  (3.72 mm to each side of the LE on the wind-tunnel rudder instead of 11.16), the TE strip matched to the tip
  patch's LE edge as before (3.08 mm instead of 10.59); `TIP_DROP` 1.5 as before. The corners 2.05 and 3.25 mm
  under the tops (10.67 and 12.39 before), both tip patch edges 8.58 mm (30.71). A flatter slope was tried first
  (`TIP_DROP` 0.75, installed an hour before: 5.6 and 7.1 mm): the angle between the lines at the tip patch's
  corners is about atan(tip_drop) and at the strips' top corners 90 deg less that, so it closed the tip patch's
  corners to 33 deg; and his rebuild still had 1.5, from the planar set-up cached on 19 Nov (a set-up keeps the
  tip_drop it was saved with). Measured with the thin strips at 1.5 (the wind-tunnel rudder extracted and planar,
  a design of the tool, the tool test's tilted wing, BB2's stern plane): every solid valid, 6 faces, no fold; the
  corners' angles kept (the rudder's tip patch 47 to 53 deg, the strips' top corners 42 to 48; 53 to 57 and 40 to
  42 before; BB2's 34 at the TE strip as before); the normal turns less across the tip patch's edges (0.07 deg at
  most on the rudder, 0.97 on BB2's plane; 0.63 and 2.6 before) and by 0.002 to 0.01 across the strip lines (0.24
  at the top of the tilted wing's, 41 loops); the tip patch within 0.0032 mm of the tip, its normals within 0.27
  deg of the tip's over 99 % (0.08 before) and 1.16 at most (1.49); its lines' smallest angle, at the tops of the
  LE and TE lines, 11 deg (16; BB2 9, 12). LE_MARGIN 0.35 to 0.4 gave 1.1 to 2.4 mm but those lines 8 to 10 deg
  apart and the normals up to 6 deg off. BB2's stern plane: the corners 26 and 51 mm under the tops (145 and 183).
  `loft_loops` keeps a guard for slopes under 1.5 given: `_edge_turn` (`LoftResult.tip_turn`, in the tip patch's
  note) and, where such a slope folds the tip patch or turns the normal across its edges by more than `EDGE_TURN`
  (2 deg), the solid built again at `TIP_DROP_SAFE` (1.5) and the smoother kept (at 0.5 to 1.25 BB2's 15 mm TE
  round turned it by 4.5 to 7.8 deg with the wide strips). `_strip_knots`: knots merged within 1e-10 and mirrored
  exactly (rounding to 13 places had split two near-equal knots: one taken twice, its mirror once). The wing, fin
  and BWB samples (flat tops) get the thin strips too. Tests: tool 74 (the LE strip about 2 % of the chord, the
  real rudder's TE strip 2 to 5 mm, the guard forced with EDGE_TURN 0), wing 78, files 60, hull 65 with the
  window, modify 136, tip cap 44; the contract test passes.
- Planar Design tab uncluttered, no variables held (21 Nov 2026; the user, of the blended wing body's tab: the
  planform, the front view and the 3D view are unnecessary there, the 3D sections tab shows the shape much more
  clearly, the Design tab only for plots that inform and help the design; then "just two changes: unclutter the
  design tab and only display plots useful for designing; and do not hold any variables, the user chooses that").
  `xgeom_wing.WingViews` (the wing, the fin and the blended wing body): the picked curve on the left over the full
  height, the sections over the chord on the right in their true shape (the box as tall as they need, the legend
  above them); a design's problems on the curve plot, a failed section build on the sections plot. `preview()`
  keeps the outline entries the web app draws. Every variable starts free: `xgeom_wing.FREE` is all of them
  (chord and pitch before; the blended wing body's own start set gone); new change curves come free unless the
  user held every old one; `bwb_glider.txt` (62 of 62), `wing.txt` and `fin.txt` (59 of 59) with every var line Y,
  nothing else changed. Held only keeps a variable out of the optimiser's space; it moves in the panel either way.
  Tests: wing 79 (the samples all free, a user's holds carried to new change curves), hull 65, files 60 with the
  window; the contract test passes.
- Independent and dependent variables, one structure in every set-up file (24 Nov 2026; the user: the variables
  have been independent numbers so far, but one may have to depend on others, such as p1y <= 0.5*p2y + 2*p3y or
  p2x > p1x + 0.2; a switch per variable between independent and dependent and, for a dependent one, an equation
  like > 0.5p4y; and in the set-up files of every component the same fixed structure for declaring variables
  and defining them with expressions). New module `xgeom_relations.py` (in the Rudder folder too): a relation is
  an operator (< <= > >= =) and an expression of other variables of the component (numbers, names, + - * /,
  brackets; 0.5v4 is 0.5*v4; several relations with commas; a variable of the same curve by its key, another by
  its full name). `parse` and `text` (what is written is read back as the same relation), `order` (the order of
  dependence; a loop is refused, named), `limits`, `nearest`, `apply`, `violations`. A dependent variable's
  limits are its bounds narrowed by its relations at the other variables' present values: it moves inside them,
  and when a variable it depends on moves it follows (each dependent variable into its limits, in the order of
  dependence). With `=` it takes the expression's value and is no design variable. Relations that cannot hold
  inside the bounds leave the variable on a bound, with a warning (the build still runs). Files: the var line
  is `# var <name> <value> <low> <high> <Y|N> [<relation>, ...]` in every kind (`xgeom_files.Var.relation`; a
  wrong grammar refused with the line, a name that is no variable or a loop with the file), and the curves and
  variables block starts with title lines that name its two line forms; the rudder-h set-up JSON has
  `"relation"` in a dependent variable's slot. A file with no dependent variable is written as before but for
  those title lines. A file whose design breaks a relation loads as written (`space.design_violations`); the
  tool then applies the relations and says which variables moved. Design spaces (`BladeSpace`, `WingSpace`,
  `HullSpace` and the rudder's `DesignSpace`, now on `SpaceRelations`): `set_relations`, `relations`,
  `violations(x)`, `project(x)`; `full_vector` is the mixin's one copy and works out the `=` variables, which
  leave the design vector. Adapters (on `AdapterRelations`): `relation`, `set_relation`, `limits`,
  `is_dependent`, `is_defined`, `varies`, `relation_notes`; `set()` keeps a variable inside its limits and lets
  the dependent ones follow; a curve's new form drops the relations of its variables, and a relation elsewhere
  that names a variable now gone (the status bar says so). The tool's rows are free | name | low | slider | high
  | value | where | dep: the name is what a relation calls the variable; dep ticked opens the relation line
  under the row (Return sets it), which shows the limits it gives now, and the slider runs over them; the left
  panel is 640 px wide (610). `blade_vary.read_free` reads var lines with a relation (a pipeline folder's copy
  needs this version for such a file; `Initial_sampling.py` does not apply relations: its variables are
  para_control's). The five samples re-saved with the title lines (`bwb_glider.txt` also gains `# half N` and
  `# cap_loops 40`, which Save set-up writes). For the web app: additive, nothing renamed (docs/keeping-
  interfaces-stable.md). Copy the new module `xgeom_relations.py` (xgeom_files, the four *_modify modules and
  the four adapters import it); a file with a relation needs this version to read; `meta[name]` has `"rel"`
  for a dependent variable only; `n_free()` leaves out a variable defined with `=`; the hull's and the rudder's
  `preview()` gain `"warnings"`. Not done: the rudder-h set-up stays JSON (the same six things per variable,
  not yet the text layout of the component files). Tests: new `_xgeom_relations_test.py`, 69 with the window
  (the grammar, limits and following, every design space, the files, the adapters, the rows; in the Rudder
  folder the module and the rudder), tool 74, wing 79, files 60, hull 65, blade 79 with the window, modify 136,
  hull_modify 17; the contract test passes (its one note as before).
- Dependent bounds (25 Nov 2026; the user: "the bounds should also be able to be dependent, like is the case in
  my msc prop optimization. Look at that as an example"). The MSc pipeline (Prop_hull:
  para_control_bez_updated.py, blade_design.py, gp_infill_from_training_data.py) bounds a control point between
  other points and samples its fraction: P4's pitch is p1y + (max_pitch - p1y) y4_frac, P2 and P5 sit at the
  fractions d1 and d2 between P4 and the ends, and the design variables are those fractions, each 0 to 1 (its
  "design vars" against its "con points"). XGeom now has the same for any variable: low or high may be an
  expression of other variables in place of a number (`pitch.v4` with low `pitch.v1` and high 1.4 is the MSc's
  y4_frac; the self-check compares to_design with the MSc formula). The variable keeps its place between its
  bounds, the fraction of the way from low to high, when the others move, and that fraction is its entry in the
  design vector, with the bounds 0, 1, so a vector inside its bounds is a design inside its bounds. A relation
  (24 Nov) stays what it was: a condition on top of the bounds that an optimiser checks or projects onto (the
  MSc's coupled limits). `xgeom_relations.py`: `Expression`, `parse_bound` (a number, or an expression: one
  word, no blanks), `bounds_at`, `places`, `crossed`; `order`, `limits`, `nearest` and `apply` take the bounds'
  expressions, and `apply(kept=places)` moves a variable whose bounds moved to its place. Bounds can cross (low
  above high) where a number faces an expression: reported like a broken relation (`violations`, the tool's
  warnings); the build still runs. Files: the low and high words of the var line take an expression (`# var
  pitch.v4 1.29 pitch.v1 1.4 Y`; written with full names; `xgeom_files.Var.lo_expr`, `hi_expr`), and the block's
  title lines now say what makes a variable independent or dependent. A bound with blanks or a wrong grammar
  (with its line), a name that is no variable, a loop, or a bound that divides by zero at the file's design is
  refused with the file named. The rudder-h set-up JSON: `lo_expr`, `hi_expr` in a slot beside the numbers,
  `fractions` in its vector; H keeps numbers (h_min, h_max). Design spaces (`SpaceRelations`):
  `set_relations(texts, where, bounds, values)`, `dependent_bounds`, `factors` (the vector's variables that are
  fractions), `bound_text`, `bounds_of(values)`, `vector_of`; `to_vector` is the mixin's one copy; a held
  variable with a dependent bound keeps its fraction; `violations` names crossed bounds too; `project` keeps
  the fractions. `blade_modify.outside_bounds` takes the bounds where the design puts them. Adapters
  (`AdapterRelations`): `bounds(name)` is the mixin's and works a dependent bound out where the design is;
  `set_bounds` takes a number or an expression for each side (the adapters' own checks at the present design);
  new `bound_text`, `set_independent`; `set()` and `follow()` keep the places. The tool: the low and high boxes
  take a number or an expression (a text that is neither is named in the status bar), dep is ticked for a
  variable with a dependent bound too, the line under the row shows its limits now, a bound too long for its
  box is shown in full under that, and a line under the curve's table says what the boxes and dep take; dep
  unticked makes the bounds numbers where they are. The five samples re-saved (their title lines only). For
  the web app: additive (docs/keeping-interfaces-stable.md). `bounds(name)` returns numbers as before and
  `meta[name]["lo"]`, `["hi"]` hold them; `meta[name]` has `"lo_expr"`, `"hi_expr"` for a dependent bound only;
  `set_bounds` given numbers makes the bounds numbers, so a caller that wants to keep a dependent bound passes
  its text back (`bound_text(name)`); `is_dependent` is true for a dependent bound; `preview()["warnings"]` also
  names crossed bounds; a file with a dependent bound needs this version to read. `Initial_sampling.py` is
  unchanged (it samples para_control's own fractions). Tests: `_xgeom_relations_test.py` 131 with the window
  (was 69; new: the grammar of a bound, places and following, the spaces' fractions, the MSc y4_frac, the
  files, the adapters, the rudder's JSON, the boxes; in the Rudder folder 42), tool 74, wing 79, files 60, hull
  65, blade 79 with the window, modify 136, hull_modify 17; the contract test passes (its one note as before).
- Bounding laws, the original as the lower bound, and the Rudder slot's two options (26 Nov 2026; the user:
  rudder optimisation commonly builds on an existing rudder, adding mass to the original, "so it is a valid
  bound law to have"; two options for a rudder, to modify an existing one, where "the original shape in the
  modified zone becomes the lower bounding law to abide by", or to design one "with no original shape").
  Bounding laws (the rudder modified below H): a curve may lie between a low law and a high law, each a number
  or an expression of the curves at the same height (`P3z`, `orig.P2z`, `LE`, `TE`, `chord`, `orig.chord`;
  `xgeom_relations.parse_law`, `sequence`; `min( )` and `max( )`, new in every expression). Its value is
  low + f (high - low), f its own Bezier over the height with control values 0 to 1, so it lies between its laws
  at every height whatever the fractions are. Its variables are `<curve>.f1` .. (bounds 0, 1); C0 is the
  fraction that puts it on the original at H, and it joins there with its value only (two lofts). The low law
  is where the fraction is 0 and may be the larger one; low and high the same set the curve (no variables).
  `chord` is the sharp chord of the section the laws give, found with them (a fixed point; the new
  `bezier_section.te_x_from`, Newton from nearby tangent points, keeps it to a few ms). `rudder_modify`:
  `Design.laws`, `law_names`, `canonical_laws`, `law_state`, `law_bands`, `law_warnings` (laws that cross, the
  original outside its laws at H, a step at H), `with_laws` (a curve put between laws or taken off them, its
  shape kept), `control_points`, `DesignSpace(laws=, te_radius_mm=)`, `sharp_chord`; a law that gives no number
  is a problem; `change_degree` is refused with laws. Set-up JSON: `space.laws`, `space.te_radius_mm`;
  `<case>_design.json`: `design.laws`. The original as the lower bound: `rudder_modify.inside_original`
  measures, on the built sections, the largest depth (mm) of the modified half-thickness under the original's
  at the same x (0 for the original itself); the settings `keep_inside` (on) and `inside_tol_mm` (0.01) are
  saved with the set-up, the preview warns beyond the tolerance and has `"inside"`, the Region box has the
  switch, the tolerance and the depth now. Laws on control points do not settle it: the wind-tunnel rudder with
  every fraction 0 and the root LE 80 mm forward lies 0.34 mm inside the original near mid chord. The tool:
  low law and high law boxes in the Curve box (the curve view draws the two laws); a Rudder box (where
  xgeom_wing.py is) with the slot's two options, modify an existing rudder (rudder-h) or design a new rudder
  (the planar model, from the new sample `rudder.txt`: 200 mm span, chord 180 to 152 mm, NACA 0018), each
  keeping its state; Load ... in the new rudder's File box loads a planar rudder's file there. New module
  `rudder_laws_setup.py` (both folders; the web app does not need it): writes
  `outputs/modified/add_to_original_design_space.json`, the rule of the planar-rudder optimisation (the folder
  Planar_Rudder_Optimization) as laws: 26 variables, every fraction 0 the original. Against that folder's own
  code on 40 random designs, P0 to P3 of every section agree within 0.08 mm (a = b = 2) and 0.23 mm (a = b =
  10). What the original is asked again and again is now kept with it: a change with its preview takes 46 ms
  without laws (62 before) and 81 ms with the rule's laws. For the web app: additive
  (docs/keeping-interfaces-stable.md; claude/xgeom-file-formats.md, Status 26 Nov). New adapter methods
  `laws_of`, `set_laws`, `law_defined`, `law_names`, `set_keep_inside`; `preview()` gains `"bands"` and
  `"inside"`; a curve with laws has variables `<curve>.f<k>`, its `design.curves[c].v` are the fractions
  (`rudder_modify.control_points` gives the control points as a plot shows them) and its join is G0;
  `set_tangent(curve, True)` and `set_degree` raise ValueError with laws; two new settings and `space.laws`,
  `space.te_radius_mm` in every set-up saved. Tests: relations 138 with the window (was 131), rudder_modify 147
  (136), tool 83 with the window (74), wing 80 (79), files 60, hull 65, blade 79, hull_modify 17; the contract
  test passes (its one note as before).
- Every set-up a component file: the rudder modified below H as kind `rudder_h`, the JSON set-up gone (26 Nov
  2026; the user, asked whether the rudder's set-up should move from JSON to the text layout: "yes everything
  should be in the current set up format, delete anything extra or old"). **Breaking** for the web app (below; a
  word to Eddie). The file (`xgeom_files.py`, `MODIFIED`): the blocks every kind has. Geometry: the original's
  four files as an extraction writes them (`corner_points`, `section_table`, `xcad_original`, `stack`, relative
  to the file), `degree`, `span_mm` (the full height), `fit_orders`, `fit_free_heights`. Settings: `sections`,
  `points`, `te_radius_mm`, `half`, `strip_frac`, `tip_drop`, `keep_inside`, `inside_tol_mm`. A curve line is `#
  curve <name> bezier1 <points>` and then `G1` or `G0` (how it meets the original at H) or its low and high law;
  the var lines are every kind's (`H`, `<curve>.d<k>`, `LE.dx<k>`, `<curve>.v<k>`, or `<curve>.f<k>` with laws:
  value, bounds, free flag, relation, dependent bounds); the table is the curves at 21 heights from the root to
  H, written for the reader and not read back. `rudder_modify`: `load_component(path)` (an `xgeom_files.Loaded`:
  `params` a `RudderParams`, `design`, `space`, `settings`; its text shows an optimiser's calls),
  `write_component`, `component`, `component_params`, `component_original`, `component_design`;
  `xgeom_files.load` reads the kind through it. The adapter: `save_file(path)`, `load_params(path)`, `file`,
  `RudderAdapter(params=path)`, as the others; an XCAD build writes `<case>.txt` beside the XCAD file. The tool:
  the rudder's panel has the File box of every component (the set-up file, the original it names, Load ..., From
  CAD ...) in the place of its Set-up box; Save set-up opens the dialog on the file in use, else on the output
  folder and the case; `--file` takes a rudder set-up (the vehicle too); a set-up loaded while the slot holds a
  new rudder puts the slot on the model modified below H. From CAD writes `<case>.txt`; a planar rudder made the
  original writes `<name>_modified_below_H/<name>_original.txt`; `rudder_laws_setup.py` writes
  `outputs/modified/add_to_original.txt`. Removed: `RudderAdapter.space_record`, `save_space`, `load_space`,
  `setup_file`; `rudder_modify.load_design_space`, `DesignSpace.to_dict`, `from_dict`, and the layout of before
  the control-point names (T1, X1, X2, T2) in `design_from_record`; the tool's `--space`, the alias `--params`,
  `App(space=)` and the JSON file types of its dialogs. A JSON file is refused by name (`<path>: a JSON file,
  not a component file: XGeom no longer reads JSON set-ups ...`), before any window is built when it is given at
  the start. The file's figures: a var line's bounds now have 12, as its value (10 before: a variable on a bound
  of more figures read back beside it; files with shorter numbers are unchanged byte for byte, the six samples
  among them); H's upper bound at the top section, and H on it, read back as the top section itself (found when
  the old set-ups were converted: one with h_max at the top would not read back); a pinned value's rounding
  (under 1e-9) is written as 0; `DesignSpace` takes the LE's reference at the H that `pin` keeps; the table is
  the design as the file's own figures give it (found on a full-scale set-up, heights over a metre in mm: 8
  decimals there are more figures than a var line carries, and its table changed in the last decimal on a
  rewrite). A set-up saved, opened and saved again is the same bytes. The Rudder folder gains `xgeom_files.py`,
  and its `xgeom_from_cad.py` is now the shared file (31 modules, all byte-identical). The seven JSON set-ups on
  the machine (`outputs/modified/`: gui, Wind_Tunnel_Rudder, add_to_original;
  `outputs/wind_tunnel_rudder_modified_below_H/`: gui and the original; the same two under `bb2/outputs/`) were
  written again as component files beside them (`<case>.txt`, the same values to 12 figures, each read back
  against its JSON) and the JSON files deleted (the user's word, 26 Nov; none is left on the machine). Still
  JSON, and no set-up: a build's record (`<case>_design.json`), the Vehicle CAD's `<case>_vehicle_setup.json`
  (until the assembly file of step 4), `source.json` (the stamp of a planar rudder made the original). For the
  web app, **Breaking** (`docs/keeping-interfaces-stable.md`; claude/xgeom-file-formats.md, Status 26 Nov,
  later): the bridge's rudder save and load call `save_file(path)` and `RudderAdapter(params=path)` or
  `load_params(path)` as for the other components; `space_record()`, `load_space(path)` and `setup_file` are
  gone, and a set-up the app saved as JSON is not read. `_xgeom_web_contract_test.py` runs the rudder through
  the same file checks as the others. Tests: relations 138 with the window, rudder_modify 150 (147), tool 83
  with the window, wing 80, files 60, hull 65, blade 79, hull_modify 17; the contract test 271 checks (268). On
  the user's machine (no window, no OCC): relations 120, rudder_modify 151, tool 37, wing 62, files 41, hull 39,
  blade 69, hull_modify 14, contract 272; the Rudder folder relations 36, rudder_modify 140, tool 3.
- Which end of the propeller's hub moves when its height changes (26 Nov 2026; the user: the cylinder grew at
  both ends and the cone only aft, there should be "freedom to choose: which end the user wants to extend (fore
  or aft) or both together"). A blade setting `hub_extends`: `fore` (the upstream end, the base plane; the tip
  stays), `aft` (the tip; the base plane stays) or `both` (half each; the middle stays); absent, the hub's own
  (the cylinder both, the cone aft), so every file and sample behaves and saves as before. The cylinder may now
  sit off-centre: `hub_x_m` alone in the geometry is its upstream end, written only when it is not centred
  (`BladeDesign.hub_front` gives the upstream end of either hub; `hub` builds the cylinder from it).
  `blade_modify`: `HUB_EXTENDS`, `EXTEND_SHARE`, `hub_extends_of`; `BladeSpace(..., hub_extends=)` keeps the
  hub's point at the end that does not move, so an optimiser that varies `hub_height` moves the same end as the
  tool (a cone's `hub_x` held follows it; free, the optimiser sets it); `load_component` reads the setting
  (another word refused, the file named) and gives it to the space; `write_component` and `write_params` write
  `hub_x_m` for an off-centre cylinder; `design_checks` checks the cylinder from where it is (its message says
  where); `write_cad` gives the DRDC hub sector its centre (`X_CAD(hub_center=)`, which hub_new had), and the
  solids take the hub's profile as before. `xgeom_blade`: `hub_extends`, `hub_ends`, `set_hub_extends(end)` (the
  hub itself does not move; saved with the set-up); `set("hub_height", v)` moves the chosen end, a cone's
  `hub_x` with it (its bounds widened to the mm when it leaves them); the hub height row says where the hub runs
  (x from .. to ..). The tool: a "hub extends" row of three radio buttons in the Blade and hub box. The Vehicle
  tab places the hub by its upstream end (`prop_x`), so extending it fore moves the blades aft along the hull;
  `vehicle_cad`'s fallback for an older set-up's DRDC faces takes the hub's recorded upstream end.
  `xgeom_files`: `hub_extends` among the blade's settings (the file is shared with the Rudder folder). For the
  web app: additive (claude/xgeom-file-formats.md, Status 26 Nov, later); a file with the setting, or with
  `hub_x_m` alone, needs this version to read. Tests: blade 90 with the window and the CAD (82 before) (an
  off-centre cylinder's DRDC sector and 1/Z solid from -250 to 550 mm, as placed), relations 138, files 60,
  contract 271, tool 83, hull 65, wing 80.
- The CAD kernel, step 1 (27 Nov 2026; Eddie's request, which the user took up in Eddie's order: one module,
  `cad_kernel.py`, the only place that imports OpenCascade, the rest calling it with plain data and keeping the
  shapes it returns as opaque handles, so CAD in the browser or under another binding means reimplementing its
  functions). New module `cad_kernel.py` (both folders): the binding found as before (pythonocc-core or
  cadquery-ocp; conda's DLLs on Windows) and its functions `available`, `binding`, `bspline_edge`, `line_edge`,
  `wire`, `planar_face`, `wire_face`, `surface_face`, `loft`, `sew_solid`, `box`, `common`, `faces`,
  `count_faces`, `mass_properties`, `is_valid`, `bounding_box`, `tessellate`, `write_step`, `read_step`,
  `write_iges` (`brep` for BRep solids), `read_iges`, `view`; for the modules not yet on it, the binding helpers
  `_occ`, `_static`, `_topods`. `xcad_loft.py` calls only the kernel: its OpenCascade code moved there as it was
  (the helpers, the half body's cut, the validity checks, the files, the triangles, the viewer), and its old
  names stay as the kernel's functions (`_bspline_edge`, `_thru`, `_sew_solid`, `write_step`, `occ_available`,
  `_occ`, `_PKG`, ...) for the modules and scripts that call them through it. Nothing changed in what the CAD
  gives: the new tool `_cad_compare.py` built twelve samples' CAD before and after (the rudder modified below H
  with one loft, two lofts and its half body; the wing, the fin, the planar rudder, the wind-tunnel rudder, the
  blended wing body, SUBOFF's hull; the MSc blade's DRDC sector, the whole propeller, a cone hub's sector): the
  same faces, volumes, areas, boxes and validity, and the STEP files' DATA sections and the IGES files' entities
  byte for byte. New self-check `_cad_kernel_test.py` (both folders): each function on shapes whose answers are
  known (a box, a cube sewn from six faces, a loft between two squares, a B-spline patch, STEP and IGES read
  back, the triangles facing out), and that the modules moved over (xcad_loft) call no OpenCascade themselves.
  For the web app: additive (a new module to copy, which xcad_loft imports; the kernel's function list, its
  docstring, is the seam). Tests: kernel 9, tool 83 with the window and OCC, wing 80, hull 65, blade 90 with the
  window and the CAD, hull_eq 15, hull_modify 17, files 60, contract 271, relations 138; the Rudder folder tool
  82, kernel 9.

## Next steps

- The CAD kernel (cad_kernel.py), Eddie's order: step 2, X_CAD_new.py (imports pythonocc directly: the DRDC
  blade CAD runs on pythonocc only; its IGES for the CFD pipeline must stay the same) and prop_hub.py onto
  the kernel; then vehicle_cad, wing_modify and hull_modify; the extraction modules last, if From CAD has
  to run in the browser. `_cad_compare.py` before and after each, and `_cad_kernel_test.MOVED` grows with them.
- Wings and fins: the tips (next; user, 8 Oct 2026), then a check on the user's machine of the wing, the fin and
  the vehicle with them (pythonocc-core).
- Vehicle: the placement is not loaded back yet (the set-up JSON records it); the blended wing body
  in the Vehicle tab (the propeller and fins placed on it) later.
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
- Bounding laws on the curve line of a component file (`# curve <name> <form> [<low law> <high law>]`), for the
  blade, the wing and the hull (26 Nov 2026: built for the rudder modified below H, on the curve line of its
  component file).
- Dependent variables: `Initial_sampling.py` sampling a set-up's own design vector (its fractions and
  relations) when a campaign needs it (it samples para_control's variables today).
- Set-ups: the vehicle's is the last one in JSON (`<case>_vehicle_setup.json`, written by Vehicle CAD and not
  loaded back): the assembly file in the Vehicle tab (step 4 of the input files) replaces it. A planar
  component metres across (kind rudder, fin, wing, bwb; a wing 40 times the sample), and a blade written from a
  fitted design (its skew in deg, on the user's machine): saved, opened and saved again its table changes in the
  last decimal once (12 figures on the var lines against 8 decimals); the rudder modified below H writes its
  table from the design as the file's figures give it, the same for `wing_modify` and `blade_modify` when wanted.
- Rudder: the tip tilt and cap parameters into the design space. If the mesher draws the tip's rounded edge
  the way it drew the TE: a band of its own for it (as the TE strip now is for the TE's round).
- Open items: section-fit bounds (X2 on its bound) and an LE-radius constraint; plain-named rudder outputs are
  overwritten by runs at other heights; Fig. 5 of the plan still labels the tip y = 218.07.
- From CAD for the other components (the wing, the fin, the planar rudder: their extraction codes) once the
  blade, hull and rudder runs are checked in use (the user, 13 Nov 2026).
- BB2's aft half (the user, 13 Nov 2026): the hull's sections that are not round or elliptical (the casing) and the decks;
  then the assembly of bb2_hull.txt, bb2_prop_setup.txt and bb2_sternplane.txt in the Vehicle tab (placement:
  above, BB2's positions in the Vehicle tab, as x/L).
