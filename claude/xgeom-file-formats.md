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
  into one such file. The TE is closed; the tip is flat unless the file gives it a tip cap (the `cap` keys,
  18 Nov 2026). The modified-below-H model stays as modify_control.py and as its own entry in the tool
  (rudder-h), outside the standard files.
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
| rudder, fin, wing | span_mm, section, ref_xc, camber_xc, tip_tilt_deg, cap and its keys (below) | sections, change_points, half, cap_loops | eta pitch_deg chord_mm t/c f/c sweep_mm rake_mm [dt1 .. dtk dc1 .. dck] |
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
# prop_x         1
# prop_d         5
# rudder_x       0.89819
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
  kept their names but were m from the nose before. Above: BB2's aft half.
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
- `python xgeom_from_cad.py rudder_planar <extraction folder> [--case NAME]` (18 Nov 2026; no OCC needed): a
  planar rudder's set-up from Rudder_geom_extraction's outputs (`planar_rudder_setup`): 21 rows to the top of
  the LE, the LE and TE lines straight, the section the rows' mean thickness shape
  (`<case>_section.dat`) with the change points that fit best, the tilt and the radius cap of parameters.json.
  The wind-tunnel rudder's is `outputs/wind_tunnel_rudder.txt`.
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
