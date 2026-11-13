# XGeom input files

The standard input and output files of the XGeom tool: one layout for every component (propeller blade,
rudder, fin, wing, hull) and an assembly file that places the components and defines an optimization run.
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
  into one such file. Tip flat and TE closed for now. The modified-below-H model stays as modify_control.py
  and as its own entry in the tool (rudder-h), outside the standard files.
- Optimization runs vary the propeller only for now: the run's variables are the blade file's free variables;
  the hull and the appendages are fixed (their CAD written once); each sample's CAD is the blade with its hub
  sector, as the mesh step takes it. Initial_sampling.py takes the assembly beside the MSc path, so a running
  campaign is unaffected; the infill reads the same variables and bounds from the assembly.
- Old .dat parameter files still load; the JSON set-ups are replaced by the component files (the rudder-h
  model keeps its JSON).
- Case outputs: `<case>.txt` (the design as a component file) in place of `<case>_params.dat` and
  `<case>_design_space.json`.

## Component file

```
# XGeom component
# kind          blade
# name          bb2_propeller
# note          MARIN stock propeller S7371R of BB2
# ---- geometry
# diameter_m    5
# blades        6
# root_r        0.2
# hub_height_m  1
# section       bb2_section_table.dat
# ---- settings
# sections      56
# tip_band      0.93
# tip_sections  14
# clearance_mm  25
# hub           Y
# ---- curves and variables
# var    radius     2.5     2.25  2.75  N
# curve  pitch      bezier2
# var    pitch.v1   1.2413  0.60  1.40  Y
# curve  skew       table
# ---- distributions
# r/R     P/D     c/D     t/c     f/c     skew_deg  rake/D
  0.2000  1.2413  0.2452  0.2159  0.0304  0.2758    0.0000
```

Header lines:
- `# XGeom component` first; `# kind`, `# name`, `# note` (free text, repeatable).
- Geometry and settings keys, per kind (below). An unknown key is an error.
- `# curve <name> <form>`: `table` (the column as it is: the cubic spline through the rows, no variables),
  `bezier1 <points>`, `bezier2` (x_blade's form, P1 to P7), and for the hull `cst <order>`.
- `# var <name> <value> <low> <high> <Y|N>`: one line per variable, Y free (values written to 12 figures).
- `# vary <name> Y|N [low high]`: accepted in hand-written files (blade_vary.py), not written.
- `# ---- <title>`: separators, ignored. The last `#` line before the table names the columns.

Keys and columns:

| kind | geometry | settings | columns |
|---|---|---|---|
| blade | diameter_m, blades, root_r, hub_height_m, hub_base_m, hub_top_m, hub_x_m, section, section_tables | sections, tip_band, tip_sections, clearance_mm, hub, part | r/R P/D c/D t/c f/c skew_deg rake/D |
| rudder, fin, wing | span_mm, section, ref_xc, camber_xc | sections, change_points | eta pitch_deg chord_mm t/c f/c sweep_mm rake_mm [dt1 .. dtk dc1 .. dck] |
| hull | nose_m, middle_m, tail_m, cap_m, axisymmetric | | x_m r_m [rp_m] |

Rules:
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
  hub, the default).
- `nose_m 0` (hull, 13 Nov 2026): a body without a nose, an aft half; its front the flat cut at x 0, tail curves
  only, no nose var line (BB2's `bb2/outputs/bb2_hull.txt`).
- kind `rudder` (the planar rudder, wing_modify's engine; 13 Nov 2026: BB2's stern plane,
  `bb2/outputs/bb2_sternplane.txt`): the tool opens it in the Rudder slot on the planar model (the wing's panel;
  xgeom_wing.PlanarRudderAdapter); its JSON set-up says `rudder_planar`, the rudder-h model's keep `rudder`.
- Component and assembly files are written with LF line endings on every platform (write_component,
  write_assembly), so the same design gives byte-identical files on any machine, and a rewrite of a loaded
  file gives the same bytes (round-trip tested with filecmp in `_xgeom_files_test.py`).

Code: `xgeom_files.py` (the text, `load` and `convert`); `load_component` and `write_component` in
`blade_modify.py`, `wing_modify.py` and `hull_modify.py`. Old files: `python xgeom_files.py convert old.dat
new.txt`. Examples in X_geom (the tool's defaults): `msc_blade.txt`, `wing.txt`, `fin.txt`, `suboff_hull.txt`.

## Assembly file

```
# XGeom assembly
# name           bb2
# hull           bb2_hull.txt
# blade          bb2_blade.txt
# rudder         bb2_sternplane.txt
# ---- placement
# prop_dx        0
# prop_d         5
# rudder_dx      -0.36
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
  hull's slope) or deg.
- The run's design vector is the components' free var lines, named `<kind>.<var>` (for example
  `blade.pitch.v1`), with the file values as the baseline. samples: round-0 designs; batch: designs per infill
  round.

## GUI

- The File box's Load ... reads a component file (or an old .dat) and sets the whole panel. Save set-up ...
  writes the component file; the dialog opens on the file it came from (an old .dat gives a .txt beside it).
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
  reference) and `<case>_setup.txt` (the set-up; blade_control.FREE free); hull `<case>_params.dat` (the
  stations) and `<case>.txt` (the set-up; lengths and curves free); rudder `<case>_extraction/` and
  `<case>_design_space.json`. The run is off the GUI thread (a hull a few seconds, the BB2 propeller about a
  minute, the wind-tunnel rudder about two minutes). Needs OCC, as the extraction codes do.
- `python xgeom_from_cad.py blade|hull|rudder <file> [--out DIR] [--case NAME]`: the same without the window;
  for the blade an extracted `<case>.txt` and for the hull a `<case>_params.dat` give the set-up alone (no
  OCC needed). For a planar table (a WingParams) `best_wing_forms` and `wing_setup` do the same.
- The Rudder slot has two models: the modified-below-H one (rudder-h, its JSON set-ups and From CAD) and, since
  13 Nov 2026, the planar rudder (a component file of kind rudder). `--file` with a planar rudder's file, or
  Load ... of one in the rudder-h Set-up box, puts the rudder on the planar model; Load ... of a
  `<case>_design_space.json` in the planar panel's File box puts it back on rudder-h. The Vehicle tab places a
  planar rudder as the rudder-h one (the TE rake rule takes its TE line's slope from the design).
- Vehicle tab (step 4): Load assembly ... and Save assembly ... (each component written to its file; old .dat
  sources as new .txt files beside the assembly).
- `python xgeom_tool.py --file <component file>` (`--params` the same); `--assembly <file>` (step 4).

## Steps

1. File layer and the mapping per kind, round-trip tests; the example files converted. Done 26 Oct 2026.
2. GUI load and save. Done 26 Oct 2026.
3. The planar rudder; the wind-tunnel rudder fitted into a file.
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
plane as a planar rudder), the planar rudder in the tool (`claude/bb2.md`).
