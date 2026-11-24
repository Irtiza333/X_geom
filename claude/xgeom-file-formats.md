# XGeom input files

The standard input and output files of the XGeom tool: one layout for every component (propeller blade,
rudder, fin, wing, blended wing body, hull) and an assembly file that places the components and defines an optimization run.
The GUI loads the files and writes them back (Save set-up); Initial_sampling.py and the infill read the
assembly. XGeom is the pre-optimization design step.

## Decisions (26 Oct 2026)

- One component file per part, plain text `.txt`: a `#` header and one table (np.loadtxt reads the table).
- Every lifting surface has the same six distributions in one column order; a rudder has 0 in pitch and rake.
- Only the span coordinate is normalized: r/R for the blade, eta = y/span for the others (0 to 1). The values
  keep each kind's usual units: blade P/D, c/D, skew deg, rake/D; rudder, fin, wing deg and mm; hull x and r
  in m.
- The file holds everything the tool sets: geometry, settings, each distribution's form (as the table has it,
  1-segment Bezier with its number of points, 2-segment Bezier) and per variable its value, bounds and free flag.
- The table is the design: the var lines define each Bezier curve and its column is that curve at the rows
  (a table-form curve is its column). Saving a design replaces the reference data it was fitted to (SUBOFF's
  hull: the CST half-heights, within 0.12 mm of its equations); the original file keeps them.
- The rudder is a planar lifting surface on the fin's engine: six distributions over eta, its section shape
  from the extracted rudder, the section's spanwise changes as dt columns. The wind-tunnel rudder is fitted
  into one such file. The TE is closed; the tip is flat unless the file gives it a tip cap (the `cap` keys,
  18 Nov 2026). The modified-below-H model stays as modify_control.py and as its own entry in the tool
  (rudder-h), outside the standard files.
- Optimization runs vary the propeller only for now: the run's variables are the blade file's free variables;
  the hull and the appendages are fixed (their CAD written once); each sample's CAD is the blade with its hub
  sector, as the mesh step takes it. Initial_sampling.py takes the assembly beside the MSc path, so a running
  campaign is unaffected; the infill reads the same variables and bounds from the assembly.
- Component files only (20 Nov 2026, the user: "From now on it's just the new format"; until then the old .dat
  parameter files loaded and `xgeom_files.py convert` converted them): an old parameter file is refused with a
  message that names it, and the JSON set-ups of the blade, wing, fin and hull are not read (the rudder-h
  model keeps its JSON).
- Case outputs: `<case>.txt` (the design as a component file) in place of `<case>_params.dat` and
  `<case>_design_space.json`.
- One structure for declaring variables in every set-up file, whatever the component (24 Nov 2026, the user:
  every set-up file carries the same general structure for defining and declaring things; a variable is
  independent, or dependent on others through an expression): name, value, low, high, free or held, and for a
  dependent variable its relation to other variables (Variables, below). A component file has it on the var
  line; the rudder-h model's set-up JSON has the same six in each slot record.

## Component file

```
# XGeom component
# kind          blade
# name          msc_blade
# note          the MSc blade
# ---- geometry
# diameter_m    1.4
# blades        5
# root_r        0.18
# hub_height_m  0.625
# section       airfoil_data_fixed.csv
# ---- settings
# sections      56
# tip_band      0.93
# tip_sections  14
# clearance_mm  25
# hub           Y
# part          sector
# ---- curves and variables
# ---- curve <name> <form>: table | bezier1 <points> | bezier2
# ---- var <name> <value> <low> <high> <Y free | N held> [<relation>]: independent without a relation; dependent with one,
# ----     an operator (< <= > >= =) and an expression of other variables, such as <= 0.5*chord.v2 + 2*chord.v3
# var    radius     0.7     0.524  0.875  N
# curve  pitch      bezier2
# var    pitch.v1   1.2414  0.60   1.40   Y
# var    pitch.v7   0.62    0.28   1.64   Y  < 0.9*pitch.v1
# curve  skew       table
# ---- distributions
# r/R     P/D     c/D     t/c     f/c     skew_deg  rake/D
  0.1800  1.2414  0.2452  0.2159  0.0304  0.2758    0.0000
```

Every component file has the same blocks in the same order, whatever its kind: the first line; kind, name and
notes; the geometry; the settings; the curves and variables; the table. What differs between kinds is only
which keys, curve forms and columns there are (the table below).

Header lines:
- `# XGeom component` first; `# kind`, `# name`, `# note` (free text, repeatable).
- Geometry and settings keys, per kind (below). An unknown key is an error.
- `# curve <name> <form>`: `table` (the column as it is: the cubic spline through the rows, no variables),
  `bezier1 <points>`, `bezier2` (x_blade's form, P1 to P7), `smooth2` (a blended wing body's: two cubic
  segments joined smoothly, flat at the centre line, P0 to P6), and for the hull `cst <order>`.
- `# var <name> <value> <low> <high> <Y|N> [<relation>, ...]`: one line per variable, the same in every kind: Y
  free, N held (values written to 12 figures); an independent variable ends at the flag, a dependent one has
  its relation to other variables after it (Variables, below).
- `# vary <name> Y|N [low high]` (blade_vary.py): Y free or N held, for the curves and globals without var
  lines; hand-written, or written in a blade's table file that has them (below). A set-up has var lines
  instead.
- `# ---- <title>`: separators, ignored. The last `#` line before the table names the columns. The curves and
  variables block starts with three of them that name its two line forms (above), so a file says how its
  variables are declared.

Keys and columns:

| kind | geometry | settings | columns |
|---|---|---|---|
| blade | diameter_m, blades, root_r, hub_height_m, hub_base_m, hub_top_m, hub_x_m, section, section_tables | sections, tip_band, tip_sections, clearance_mm, hub, part | r/R P/D c/D t/c f/c skew_deg rake/D |
| rudder, fin, wing, bwb | span_mm, section, ref_xc, camber_xc, tip_tilt_deg, cap and its keys (below) | sections, change_points, half, cap_loops | eta pitch_deg chord_mm t/c f/c sweep_mm rake_mm [dt1 .. dtk dc1 .. dck] |
| hull | nose_m, middle_m, tail_m, cap_m, axisymmetric | | x_m r_m [rp_m] |

Rules:
- A table file is a component file with no curve or var lines: the geometry, optional vary lines and the table
  (`write_params` of blade_modify, wing_modify and hull_modify writes one; `read_params` reads the geometry and
  the table of any component file). Loaded, its curves are fitted to its table: the blade's as its vary lines
  say (none: blade_control's forms, `FREE` free), the wing's and the fin's 1 segment (`xgeom_wing.FREE` free),
  the hull's CST curves (the lengths and curves free). It holds reference data: the stations From CAD extracts
  for the hull, the tables the BB2 scripts write.
- The var lines define a Bezier curve exactly; its column is the curve at the rows, so Save and Load
  round-trip (a rewrite gives the same file). A column edited under its curve's var lines is refused with a
  message; delete the curve's var lines and it is fitted to the column (default bounds; free per vary lines,
  else held).
- A geometry value and the var line of the same quantity (diameter_m and radius, span_mm and span, the hull's
  lengths) must agree.
- `section`: a file relative to the component file when one exists (a para.py section table, told apart by its
  comma-separated rows of 4 numbers, or airfoil coordinates, x y per line), else an airfoil by name
  (airfoils.py). A name that is no known airfoil fails the load with the file and the close matches named, so
  a missing table is never papered over silently.
- `section_tables` (blade only): the per-radius section tables, one para.py table per r/R station in one file
  (`# r/R <value>` block headers; blade_extraction.py writes one per cut). The path is read like `section`,
  relative to the component file (a missing file fails the load). Since 13 Nov 2026 the tool builds with them
  (blade_sections.SectionsBladeSurface, cubic in r/R between the tables): the views, the CAD and the checks;
  the XCAD points file (para.py) keeps the one `section`.
- `hub_base_m`, `hub_top_m`, `hub_x_m` (blade only, 13 Nov 2026; all three or none): a conic hub with a rounded
  nose (prop_hub.py): the base radius (the larger end, upstream), the top radius where the cone meets the
  nose (the sphere tangent to the cone through the tip, `hub_height_m` from the base), and the x of the base
  plane in the blade frame (m, downstream positive; the generator line is x = 0). With them `root_r` is the
  blade root, where the table starts; the blade runs on into the hub. Their var lines `hub_base`, `hub_top`,
  `hub_x` (held unless Y). Without them the hub is the cylinder of the root radius, centred on x = 0.
- `part` (blade setting): what the CAD writes, `whole`, `half` (an even Z) or `sector` (one blade on 1/Z of the
  hub, the default). The Vehicle CAD takes the same part, as a solid (19 Nov 2026; it had always taken the
  whole propeller); `outputs/vehicle/<case>_vehicle_setup.json` names it in its propeller record (`part`, and
  `part_blades` the part's number of blades).
- `nose_m 0` (hull, 13 Nov 2026): a body without a nose, an aft half; its front the flat cut at x 0, tail curves
  only, no nose var line (BB2's `bb2/outputs/bb2_hull.txt`).
- kind `rudder` (the planar rudder, wing_modify's engine; 13 Nov 2026: BB2's stern plane,
  `bb2/outputs/bb2_sternplane.txt`): since 19 Nov 2026 the tool loads it in the Rudder slot as the original of
  the rudder-h model (`xgeom_from_cad.rudder_from_planar`, below); the planar model's adapter
  (xgeom_wing.PlanarRudderAdapter, its JSON set-up `rudder_planar`) stays for code.
- `cap` (rudder, fin, wing; 18 Nov 2026, for BB2's stern plane, whose set-up gave a flat top): a tip cap over
  the tip section, tip_cap.py's models. `cap scaled` (tip_cap.ScaledCap): the cap's cross-sections scale with
  the tip section's local half-thickness w; keys `cap_k_le`, `cap_k_te` (the cap's height over w at the LE and
  the TE ends), `cap_b1`, `cap_b2` (the bow of that law along the chord, a cubic Bezier like tip_cap's face;
  default 0), `cap_p`, `cap_q` (each cross-section the superellipse |w|/w0 = (1 - (h/H)^q)^(1/p); default 2, 2,
  a quarter ellipse); the cap shrinks to nothing with the section at the LE and at the closed TE. `cap radius`
  (tip_cap.CapParams, the wind-tunnel rudder's family): `cap_h_le_mm`, `cap_h_te_mm` (the face's height above
  the tip section at the two ends), `cap_b1_mm`, `cap_b2_mm` (its bow; default 0), `cap_crown_radius_mm`
  (`none` or absent: a flat face), `cap_edge_radius_mm` (default 0), `cap_te_radius_mm` (the TE end's corner
  radius in the side view; default 0). `cap none` or no `cap` line: a flat tip; cap keys without a `cap` line,
  or of the other family, are an error. The cap's numbers are held (no var lines yet). The setting `cap_loops`
  (default 40) is how many closed loops over the cap the XCAD file gets after the sections, at the heights
  0.995 (1 - (1 - k/n)^2) of the cap; the loft's tip patch covers them (xcad_loft.py). BB2's
  `bb2/outputs/bb2_sternplane.txt` carries `cap scaled` fitted to the IGES (bb2_aft_setup.py; `claude/bb2.md`).
- kind `bwb` (a blended wing body, 20 Nov 2026; the sample `bwb_glider.txt`, the underwater glider of Ye, Wang
  and Pan 2021): the wing's layout over the half-span, `span_mm` the half-span, eta 0 the centre line. Every
  curve is flat at the centre line: forms `table` (the monotone cubic through the rows and their mirror image),
  `bezier1` (its C1 at C0's value: no var line `<curve>.v1`) and `smooth2` (var lines `v0 d1 d2 v2 u3 v3 d4 d5
  v5 v6`: P1 at P0's value, P3 the joint at u3, P4's value from the slope at P3; each d places a point in the
  stretch left before its segment's end); `bezier2` is refused. `half` Y: the CAD is the half on one side of the
  centre line (else the whole body, the half fused with its mirror image). The tool shows it in the Hull slot,
  beside the cylindrical hull.
- `tip_tilt_deg` (rudder, fin, wing; 18 Nov 2026, for the wind-tunnel rudder): the tip section tilted about the
  thickness axis through its LE, the TE end raised, by 0 (a flat tip; the default) up to 45 deg. The horizontal
  sections then run from the root to half a step under the tip's LE (k / (n - 1/2) of the span, the
  extraction's layout), and the tip section is the cut of the body, carried on past the tip along each
  distribution's tangent there, by the plane through the tip's LE that contains the thickness axis, out to
  where that plane meets the TE line (wing_modify.tilted_tip); a cap sits on that section. The span is then the
  height of the tip's LE. Held (no var line yet).
- Component and assembly files are written with LF line endings on every platform (write_component,
  write_assembly), so the same design gives byte-identical files on any machine, and a rewrite of a loaded
  file gives the same bytes (round-trip tested with filecmp in `_xgeom_files_test.py`).

Code: `xgeom_files.py` (the text, `load`; `legacy_kind` and `file_kind` recognise an old parameter file so it
is refused by name); `read_params`, `write_params` (a table file), `load_component` and `write_component` in
`blade_modify.py`, `wing_modify.py` and `hull_modify.py`. Examples in X_geom (the tool's defaults):
`msc_blade.txt`, `wing.txt`, `fin.txt`, `suboff_hull.txt`, `bwb_glider.txt`.

## Variables: independent and dependent

Every variable of every component is declared the same way (24 Nov 2026), with six things:

| | |
|---|---|
| name | `span`, `radius`, `H`: a global one; `chord.v2`, `LE.dx3`, `nose_r.b1`: `<curve>.<key>` |
| value | where the design is |
| low, high | its bounds: numbers |
| free | Y a design variable, N held |
| relation | none: an independent variable. One or more: a dependent variable |

A relation is an operator and an expression of other variables of the same component:

```
<= 0.5*chord.v2 + 2*chord.v3      at most that            (< reads the same)
>  pitch.u4 + 0.2                 at least that           (>= the same)
=  chord.v0                       that value: the variable follows the others and is no design variable
```

- The expression: numbers, variable names, `+ - * /` and brackets; `0.5v4` is read as `0.5*v4`. Several
  relations are separated by commas (`> 0.5*chord.v0, < chord.v1`); `=` stands alone.
- Names are the var lines' names. A variable of the same curve may be named by its key (`v2` for `chord.v2`);
  the files are written with full names.
- A variable may depend on dependent ones, but not, in a loop, on itself (refused, the loop named). A relation
  cannot use its own variable.
- What it does: the variable's limits are its bounds narrowed by its relations at the other variables' present
  values. In the tool a variable moves inside its limits, and when a variable it depends on moves it follows
  (each dependent variable moved into its limits, in the order of their dependence). Relations that cannot
  hold inside the bounds leave the variable on a bound, and the tool warns (the build still runs).
- For an optimiser (`ld.space` of `load_component`, `rudder_modify.load_design_space`): `space.relations`; a
  free dependent variable stays in the design vector with its bounds, and its relations are constraints:
  `space.violations(x)` lists what a vector breaks (variable, text, amount), `space.project(x)` moves the
  dependent variables into their limits. A variable with `=` leaves the vector; `to_design(x)` works it out.
- Reading: a relation with a wrong grammar is refused with its line (`<path>, line <n>: the relation of
  <name>: ...`), one that names no variable or closes a loop with the file's path. A file whose design breaks
  one of its relations loads as it is written (`space.design_violations(design)` says what); the tool then
  applies the relations and says which variables moved.
- A relation goes with its variable: a curve given another form (points, segments) loses the relations of its
  variables, and a relation elsewhere that names a variable now gone is dropped, which the tool says.

In a component file (every kind):

```
# var    chord.v0                     120                40               150  Y
# var    chord.v1                     108                40               150  Y  < 0.9*chord.v0
# var    chord.v2                      90                40               150  Y  < chord.v1, > 0.25*chord.v0
# var    pitch.v1                       2                -3                 4  Y  = pitch.v0
```

In the rudder-h model's set-up JSON (`<case>_design_space.json`), one record per variable in `space.slots`,
`relation` only in a dependent variable's:

```
{"name": "LE.dx3", "value": -30.0, "lo": -320.0, "hi": 0.0, "free": true, "relation": "< LE.dx2"}
```

A file without relations is written as before: the var lines and the slot records of independent variables
are unchanged. `Initial_sampling.py` does not read relations (the MSc pipeline samples para_control's own
variables, with its own coupled limits). Code: `xgeom_relations.py` (the grammar, `limits`, `apply`,
`violations`; `SpaceRelations` for the design spaces, `AdapterRelations` for the tool's adapters).

## Assembly file

```
# XGeom assembly
# name           suboff
# hull           suboff_hull.txt
# blade          msc_blade.txt
# rudder         outputs/wind_tunnel_rudder.txt
# ---- placement
# prop_x         1
# prop_d         0.3
# rudder_x       0.95
# rudder_angle   45
# rudders        4
# rudder_scale   1
# rudder_rake    0
# fins           0
# wings          0
# view           stern
# ---- optimization run
# samples        30
# batch          10
# seed           20261010
# cad_timeout_s  2000
```

- Component paths are relative to the assembly file.
- The placement keys are the Vehicle tab's (xgeom_vehicle.Placement); rudder_rake is `none` (pitched to the
  hull's slope) or deg. Positions along the hull are fractions of its length, x/L, 0 the nose (a body without
  a nose: its front) and 1 the hull's aft end (13 Nov 2026): `prop_x` where the propeller's hub starts (its
  forward end, 0 to 2; 1 its base on the hull's end), `rudder_x` the rudders' root TE (with rudder_rake: where
  the TE line meets the hull), `fin_x` and `wing_x` the root LE, 0 to 1; `prop_d` stays in m. The keys before
  (prop_dx and rudder_dx, m from the hull's end) are refused, so an old file is not misread; fin_x and wing_x
  kept their names but were m from the nose before. Above: SUBOFF with the MSc blade and the wind-tunnel
  rudder, an illustration (the values are not a worked-out placement).
- The run's design vector is the components' free var lines, named `<kind>.<var>` (for example
  `blade.pitch.v1`), with the file values as the baseline. samples: round-0 designs; batch: designs per infill
  round.

## GUI

- The File box's Load ... reads a component file and sets the whole panel. Save set-up ... writes the component
  file; the dialog opens on the file it came from. `--space` takes the rudder-h model's JSON set-ups only.
- Each variable's row: free | name | low | slider | high | value | where | dep (24 Nov 2026). The name is
  what a relation calls the variable. dep unticked: independent. Ticked: a line opens under the row for its
  relation (`> 0.5*v4`, Return sets it); the row then shows the limits the relation gives now, the slider runs
  over them, and the variable follows the ones it depends on. With `=` the row's slider, value and free box are
  off. Unticked again, or the relation emptied: independent.
- From CAD ... (13 Nov 2026, `xgeom_from_cad.py`): in the File box of the blade and the hull and in the
  rudder's Set-up box. A STEP or IGES file of the component is extracted (blade_extraction.py, hull_extraction's
  stations, Rudder_geom_extraction.py with the per-section fits), every curve is fitted in each candidate form
  and the form with the least error kept, and the set-up is written and loaded, so the sliders start at the
  extracted geometry and the bounds can be set about it. Candidates: blade 2 segments, or 1 segment of 3 to 8
  points; hull CST orders 3 to 8; rudder-h the original's law orders 2 to 6. The error is the largest
  difference from the extracted data over the data's range (hull: over r*; a rudder law's range taken as at
  least 1 %); a form with more points is kept only when it beats the best one with fewer by 0.1 % of the
  range (`FIT_TIE`). The status line lists the forms and their errors. Changing points or segments afterwards
  refits the curve to the extracted file and the status line gives the new error. Files, in the component's
  output folder, the case named after the CAD file: blade `<case>.txt` (the six distributions as tables, the
  reference) and `<case>_setup.txt` (the set-up; blade_control.FREE free); hull `<case>_table.txt` (the
  stations, a table file) and `<case>.txt` (the set-up; lengths and curves free); rudder `<case>_extraction/` and
  `<case>_design_space.json`. The run is off the GUI thread (a hull a few seconds, the BB2 propeller about a
  minute, the wind-tunnel rudder about two minutes). Needs OCC, as the extraction codes do.
- `python xgeom_from_cad.py blade|hull|rudder <file> [--out DIR] [--case NAME]`: the same without the window;
  for the blade an extracted `<case>.txt` and for the hull a table file `<case>_table.txt` give the set-up alone (no
  OCC needed). For a planar table (a WingParams) `best_wing_forms` and `wing_setup` do the same.
- `python xgeom_from_cad.py rudder_planar <extraction folder> [--case NAME]` (18 Nov 2026; no OCC needed): a
  planar rudder's set-up from Rudder_geom_extraction's outputs (`planar_rudder_setup`): 21 rows to the top of
  the LE, the LE and TE lines straight, the section the rows' mean thickness shape
  (`<case>_section.dat`) with the change points that fit best, the tilt and the radius cap of parameters.json.
  The wind-tunnel rudder's is `outputs/wind_tunnel_rudder.txt`.
- The Rudder slot shows the modified-below-H model (rudder-h, its JSON set-ups and From CAD) whatever is
  loaded. Since 19 Nov 2026 (the user: loading BB2's or the wind-tunnel rudder's set-up must give the same
  panel, tangent at H with it) `--file` with a planar rudder's component file, or Load ... of one in the
  Set-up box, makes that rudder the rudder-h model's original (`xgeom_from_cad.rudder_from_planar`; `python
  xgeom_from_cad.py rudder <file>.txt` the same without the window): its sections, tip and cap written as an
  extraction writes a CAD's into `<name>_modified_below_H/` beside the file (61 horizontal sections, 121
  points each), the sections fitted (about 25 s the first time), the set-up
  `<name>_original_design_space.json` there (the rudder unchanged, H 0.1 to 0.4 of the top section, its TE
  radius), read back while the file and its section file are unchanged; the cases of it go to that folder.
  From 13 to 19 Nov such a file put the rudder on the planar model (the wing's panel).
- Vehicle tab (step 4): Load assembly ... and Save assembly ... (each component written to its file; old .dat
  sources as new .txt files beside the assembly).
- `python xgeom_tool.py --file <component file>` (`--params` the same); `--assembly <file>` (step 4).

## Steps

1. File layer and the mapping per kind, round-trip tests; the example files converted. Done 26 Oct 2026.
2. GUI load and save. Done 26 Oct 2026.
3. The planar rudder; the wind-tunnel rudder fitted into a file. Done 13 Nov (BB2's stern plane) and 18 Nov 2026
   (the wind-tunnel rudder, `outputs/wind_tunnel_rudder.txt`, with its tilted tip and cap).
4. The assembly: file, Vehicle tab, SUBOFF and BB2 examples.
5. Initial_sampling.py and the infill read the assembly.

## Status

26 Oct 2026: design agreed (all three questions answered with the recommended option). Steps 1 and 2 done and
installed in X_geom (xgeom_tool.py also in the Rudder folder): checked in the cloud (`_xgeom_files_test.py
--gui` and the blade, wing, hull and tool checks with `--gui` and OCC) and on the user's machine
(`_xgeom_files_test.py`).

6 Nov 2026: LF line endings pinned in the writers, with checks in `_xgeom_files_test.py`, so saved files are
byte-identical across machines; this spec added to the repo as `claude/xgeom-file-formats.md` (the path
xgeom_files.py's docstring names).

13 Nov 2026: the blade's conic hub keys, `part` and `section_tables`; From CAD in the GUI for the blade, the
hull and the rudder (the other components once these are checked in use). Set-ups made with it:
`bb2/outputs/bb2_prop_setup.txt`, `outputs/modified/Wind_Tunnel_Rudder_design_space.json`. Later the same day
(the user: only BB2's aft half for now, the hull and the rudder set-ups were incorrect): `bb2/bb2_aft_setup.py`
writes `bb2/outputs/bb2_hull.txt` (the aft half without a nose) and `bb2/outputs/bb2_sternplane.txt` (BB2's stern
plane as a planar rudder), the planar rudder in the tool (`claude/bb2.md`). Then (the user) the Vehicle tab's
positions as x/L, the propeller placed by where its hub starts: the assembly's placement keys `prop_x`,
`rudder_x`, `fin_x`, `wing_x` follow (Assembly file, above).

18 Nov 2026: the planar surfaces' tip cap (`cap` and its keys, the `cap_loops` setting; above), so that BB2's
stern plane set-up builds with its rounded cap instead of a flat top. Later the same day `tip_tilt_deg` and the
wind-tunnel rudder as a planar set-up (`outputs/wind_tunnel_rudder.txt`, from the extraction's outputs by
`xgeom_from_cad.py rudder_planar`). For the web app: new optional geometry keys of the rudder, fin and wing
files and one optional setting; files without them read as before.

19 Nov 2026: the Vehicle CAD follows the blade's `part` (above). For the web app: in the vehicle set-up JSON the
propeller record's `part` now names the part (it was always `whole`) and `part_blades` is new; the component
files are unchanged.

19 Nov 2026, later: a planar rudder's component file loads in the tool as the original of the rudder-h model
(GUI, above), so the Rudder slot keeps one panel. The files themselves are unchanged (for the web app: nothing
changes in reading or writing them).

20 Nov 2026: the shipped samples re-saved as the tool writes them (`fin.txt` and `wing.txt` gain `# cap_loops
40`, `msc_blade.txt` `# part sector`, the defaults; the geometry the same, each file a fixed point of open and
save), a `.gitattributes` (`*.txt text eol=lf`) so Windows checkouts keep them LF, and the rudder's set-up JSON
without its `"written"` time stamp (an unchanged set-up saves to the same bytes; older set-ups still load). The
examples above are the public ones (the MSc blade, SUBOFF). Eddie's interface rules are in
`docs/keeping-interfaces-stable.md`, and `_xgeom_web_contract_test.py` runs before every change is handed over.

20 Nov 2026, later: component files only (the user). The old parameter files and the old JSON set-ups of the
blade, wing, fin and hull are gone from the repository and the outputs (`msc_blade_params.dat`,
`wing_params.dat`, `fin_params.dat`, `suboff_hull_params.dat`, the tool's old `gui_params.dat` and
`gui_design_space.json`); reading one gives `<path>: an old <kind> parameter file, which XGeom no longer reads
(since 20 Nov 2026): it reads component files only`. Removed: `xgeom_files.convert` and its command line,
`legacy_header`, `Loaded.legacy`; the old set-up JSON's functions `space_record`, `load_blade_space`,
`load_wing_space`, `load_hull_space`; the adapters' `save_params`, `space_record`, `save_space`, `load_space`
(the blade's, the wing's, the fin's, the hull's and the planar rudder's `rudder_planar` JSON; the rudder-h
model's stay) and their `legacy` attribute. `write_params` now writes a table file (above). Kept for the web app: `legacy_kind` and `file_kind`. The hull's
From CAD writes its stations as `<case>_table.txt`. `Initial_sampling.py` reads `msc_blade.txt`: its vary lines,
else its free var lines (`blade_vary.read_free`: pitch or chord free when one of its variables is). For the web
app: no key is new or changed; a table file may carry vary lines, which every XGeom's component reader takes.

20 Nov 2026, later: the blended wing body, kind `bwb` (above), with its curve form `smooth2`; the tool's Hull slot
holds the cylindrical hull or the blended wing body (its Body box). For the web app: a new kind and a new form,
both additive (other files read and write as before); its adapter `xgeom_wing.BwbAdapter` has the wing's
methods, with `segment_options` (0, 1, 3) (3: smooth2) and `segment_text(n)` for the buttons' text, and
`preview()["mirror"]` (True: draw the body with its mirror image, y to -y).

24 Nov 2026: independent and dependent variables (Variables, above): the var line takes a relation after the
flag, the same in every kind; the rudder-h model's set-up JSON takes `relation` in a slot; the curves and
variables block starts with title lines that name its line forms (the five samples re-saved with them;
`bwb_glider.txt` also gains `# half N` and `# cap_loops 40`, the defaults Save set-up writes). For the web
app: additive. A file with no dependent variable differs only in those separator lines, which every reader
skips; one with a relation needs this version to read (an older reader refuses the longer var line). New
module `xgeom_relations.py`, which `xgeom_files`, the four `*_modify` modules and the four adapters import (add
it to the copy list). New adapter methods `relation(name)`, `set_relation(name, text)`, `limits(name)`,
`is_dependent(name)`, `is_defined(name)`, `varies(name)`, `relation_notes()`; `meta[name]` has `"rel"` for a
dependent variable only; `set(name, value)` keeps a variable inside `limits(name)` and returns the value used,
as before; `n_free()` leaves out a variable defined with `=`; the hull's and the rudder's `preview()` gain
`"warnings"` (relations that cannot hold), which the blade's and the wing's already had.
