# XGeom input files

The standard input and output files of the XGeom tool: one layout for every component (propeller blade,
rudder, the rudder modified below H, fin, wing, blended wing body, hull) and an assembly file that places the components and defines an optimization run.
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
  (rudder-h); until 26 Nov 2026 it was outside the standard files, since then its set-up is a component file
  too (kind `rudder_h`, below).
- Optimization runs vary the propeller only for now: the run's variables are the blade file's free variables;
  the hull and the appendages are fixed (their CAD written once); each sample's CAD is the blade with its hub
  sector, as the mesh step takes it. Initial_sampling.py takes the assembly beside the MSc path, so a running
  campaign is unaffected; the infill reads the same variables and bounds from the assembly.
- Component files only (20 Nov 2026, the user: "From now on it's just the new format"; until then the old .dat
  parameter files loaded and `xgeom_files.py convert` converted them): an old parameter file is refused with a
  message that names it, and the JSON set-ups of the blade, wing, fin and hull are not read (the rudder-h
  model kept its JSON until 26 Nov 2026).
- Case outputs: `<case>.txt` (the design as a component file) in place of `<case>_params.dat` and
  `<case>_design_space.json`.
- One structure for declaring variables in every set-up file, whatever the component (24 Nov 2026, the user:
  every set-up file carries the same general structure for defining and declaring things; a variable is
  independent, or dependent on others through an expression): name, value, low, high, free or held, and for a
  dependent variable its relation to other variables (Variables, below). A component file has it on the var
  line (the rudder-h model's set-up too, a component file since 26 Nov 2026).
- A bound may itself depend on other variables (25 Nov 2026, the user: "the bounds should also be able to be
  dependent, like is the case in my msc prop optimization"): low or high is then an expression in place of a
  number, the variable keeps its place between its bounds, and its entry in an optimiser's design vector is
  that place, the fraction 0 to 1, as the MSc pipeline's y4_frac, d1 and d2 are.
- Bounds between curves, and the original as a lower bound (26 Nov 2026, the user: rudder optimisation commonly
  builds on an existing rudder, adding mass to the original structure, "so it is a valid bound law to have";
  the tool stays general: two options for a rudder, to modify an existing one, where "the original shape in
  the modified zone becomes the lower bounding law to abide by", or "to let people design a rudder for a
  vehicle (with no original shape)"). A bound now has three levels: a number; an expression of other
  variables (a dependent bound); a law between curves (a bounding law: the curve lies between a low law and a
  high law, each an expression of other curves at the same span position, and its variables are the fraction
  of the way). Bounding laws are built for the rudder modified below H (Bounding laws, below), on the curve
  line of its component file; for the blade, the wing and the hull they are the next step.
- Every set-up is a component file (26 Nov 2026, the user, asked whether the rudder's set-up should move from
  JSON to the text layout: "yes everything should be in the current set up format, delete anything extra or
  old"). The rudder modified below H is kind `rudder_h` (below); its JSON set-up
  (`<case>_design_space.json`), the tool's `--space`, and the functions that read and wrote it are gone, and
  a JSON file is refused by name. What stays JSON is no set-up: a build's record (`<case>_design.json`) and
  the Vehicle CAD's (`<case>_vehicle_setup.json`, until the assembly file of step 4).

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
# ---- var <name> <value> <low> <high> <Y free | N held> [<relation>]
# ----     independent: numbers for low and high, no relation
# ----     dependent:   low or high an expression of other variables (one word, such as 0.9*chord.v0),
# ----                  or a relation after the flag: an operator (< <= > >= =) and an expression,
# ----                  such as <= 0.5*chord.v2 + 2*chord.v3
# var    radius     0.7     0.524     0.875  N
# curve  pitch      bezier2
# var    pitch.v1   1.2414  0.60      1.25   Y
# var    pitch.v4   1.2912  pitch.v1  1.40   Y
# var    pitch.v7   0.62    0.28      1.64   Y  < 0.9*pitch.v1
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
  segments joined smoothly, flat at the centre line, P0 to P6), and for the hull `cst <order>`. For the rudder
  modified below H, `bezier1 <points>` and then how the curve meets the original at H, `G1` or `G0`, or its two
  bounding laws (kind `rudder_h`, below).
- `# var <name> <value> <low> <high> <Y|N> [<relation>, ...]`: one line per variable, the same in every kind: Y
  free, N held (the value and the bounds written to 12 figures; the bounds had 10 until 26 Nov 2026, when a
  variable on a bound of more figures could read back beside it). low and high are each a number, or an
  expression of other variables written as one word (a dependent bound). An independent variable has numbers and ends at the flag;
  a dependent one has an expression as a bound, a relation to other variables after the flag, or both
  (Variables, below).
- `# vary <name> Y|N [low high]` (blade_vary.py): Y free or N held, for the curves and globals without var
  lines; hand-written, or written in a blade's table file that has them (below). A set-up has var lines
  instead.
- `# ---- <title>`: separators, ignored. The last `#` line before the table names the columns. The curves and
  variables block starts with title lines that name its two line forms and what makes a variable independent
  or dependent (above), so a file says how its variables are declared.

Keys and columns:

| kind | geometry | settings | columns |
|---|---|---|---|
| blade | diameter_m, blades, root_r, hub_height_m, hub_base_m, hub_top_m, hub_x_m, section, section_tables | sections, tip_band, tip_sections, clearance_mm, hub, part | r/R P/D c/D t/c f/c skew_deg rake/D |
| rudder, fin, wing, bwb | span_mm, section, ref_xc, camber_xc, tip_tilt_deg, cap and its keys (below) | sections, change_points, half, cap_loops | eta pitch_deg chord_mm t/c f/c sweep_mm rake_mm [dt1 .. dtk dc1 .. dck] |
| rudder_h | corner_points, section_table, xcad_original, stack, degree, span_mm, fit_orders, fit_free_heights | sections, points, te_radius_mm, half, strip_frac, tip_drop, keep_inside, inside_tol_mm | y_mm LE_mm P1x P1z .. P(n-1)x P(n-1)z |
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
  the rudder-h model (`xgeom_from_cad.rudder_from_planar`, below); the planar model itself
  (xgeom_wing.PlanarRudderAdapter) is the Rudder slot's other option since 26 Nov 2026, design a new rudder.
- kind `rudder_h` (the rudder modified below H, rudder_modify's model; 26 Nov 2026, in place of its JSON
  set-up): an existing rudder kept above the height H and changed below it.

  ```
  # XGeom component
  # kind          rudder_h
  # name          gui
  # ---- geometry
  # corner_points ../corner_points.dat
  # section_table ../thickness_params_y0_y200_info.dat
  # xcad_original ../sections_xcad_with_cap_y0_y200_m.dat
  # stack         ../sections_stack_y0_y200_raw_mm.dat
  # degree        4
  # span_mm       211.27677
  # ---- settings
  # sections      50
  # points        200
  # te_radius_mm  0.75
  # half          N
  # strip_frac    none
  # tip_drop      1.5
  # keep_inside   Y
  # inside_tol_mm 0.01
  # ---- curves and variables
  # var    H            80       20       80  Y
  # curve  LE       bezier1 4 G1
  # var    LE.d1    0.333333333333  0.05  0.95  Y
  # var    LE.d2    0.5      0.05     0.95  Y
  # var    LE.dx2   -12      -320     0     Y
  # var    LE.dx3   -30      LE.dx2   0     Y
  # curve  P1z      bezier1 4 P2z P2z
  # curve  P2z      bezier1 4 orig.P2z*orig.chord/chord 2*orig.P2z*orig.chord/chord
  # var    P2z.d1   0.333333333333  0.05  0.95  Y
  # var    P2z.d2   0.5      0.05     0.95  Y
  # var    P2z.f1   0        0        1     Y
  # ...
  # ---- the curves from the root to H (what the var lines give; not read back)
  #        y_mm       LE_mm         P1x         P1z ..
     0.00000000 -30.00000000  0.00000000  0.07336669 ..
  ```

  - Geometry: the original rudder, as an extraction writes it and `rudder_modify.fit_section_table` fits it,
    by four files relative to the component file: `corner_points` (the LE and TE points), `section_table` (the
    per-section fits), `xcad_original` (the XCAD loops with the cap) and `stack` (the section stack). `degree`
    of the sections' Bezier (4, the table's quartics, or more); `span_mm` the full height, the root to the top
    of the cap (absent: the extracted rudder's; another value stretches the rudder in height, H and its bounds
    with it); `fit_orders` the orders the original's laws are fitted at, as pairs (`LE 3 P1x 2 ..`; absent:
    cubics); `fit_free_heights Y` when the fit searched its control heights.
  - Settings: `sections` (from the root to H) and `points` (per section) of the XCAD file, `te_radius_mm`,
    `half` (the CAD's half body), `strip_frac` and `tip_drop` of the loft (`none`, one number, or `LE TE` as two),
    `keep_inside` and `inside_tol_mm` (the original as the lower bound, below).
  - The curves are the LE (x in mm) and the section control points' coordinates P1x, P1z .. (fractions of the
    sharp chord), each a Bezier law over the height from H (C0, on the original) down to the root. A curve line
    is `# curve <name> bezier1 <points>` and then `G1` (the curve leaves the original along its tangent: C1's
    value follows, no var line for it) or `G0` (its value only: C1 free), or a low law and a high law, one word
    each (Bounding laws, below; such a curve joins G0).
  - Variables: `H` (its bounds are h_min and h_max, numbers); per curve the control heights `<curve>.d<k>` (each
    the share of the stretch left, as the blade's) and the control values: `LE.dx<k>` (mm from the original LE
    at that height), `<curve>.v<k>` (the fraction of the chord), or for a curve with laws `<curve>.f<k>` (the
    fraction of the way between them). A curve with no var lines is the original's (its variables free, P1x's
    held, the default bounds); var lines for some of a curve's variables only are refused.
  - The table is written for the reader: the curves at 21 heights from the root to H, as the var lines'
    figures give them. It is not read back (the var lines and the original's files give the design), so
    editing it changes nothing.
  - Read by `rudder_modify.load_component(path)` (an optimiser: `ld.space`, `ld.design`, `ld.params`,
    `ld.settings`; its text shows the calls), `xgeom_files.load`, the tool (`--file`, Load ...) and
    `xgeom_rudder.RudderAdapter(params=path)`; written by Save set-up, an XCAD build (`<case>.txt` beside the
    XCAD file), From CAD and `rudder_modify.write_component`. A set-up with its files beside it can be moved
    as a folder.
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
is refused by name, and a JSON file is named as one); `read_params`, `write_params` (a table file),
`load_component` and `write_component` in `blade_modify.py`, `wing_modify.py` and `hull_modify.py`;
`load_component` and `write_component` in `rudder_modify.py` (kind `rudder_h`). Examples in X_geom (the tool's defaults):
`msc_blade.txt`, `wing.txt`, `fin.txt`, `suboff_hull.txt`, `bwb_glider.txt`.

## Variables: independent and dependent

Every variable of every component is declared the same way (24 Nov 2026; dependent bounds 25 Nov 2026), with
six things:

| | |
|---|---|
| name | `span`, `radius`, `H`: a global one; `chord.v2`, `LE.dx3`, `nose_r.b1`: `<curve>.<key>` |
| value | where the design is |
| low, high | its bounds: each a number, or an expression of other variables (a dependent bound) |
| free | Y a design variable, N held |
| relation | optional: a condition on top of the bounds |

A variable is independent when its bounds are numbers and it has no relation. It is dependent when it depends
on other variables of the same component, in one or both of two ways.

A dependent bound is an expression in place of the number for low or for high:

```
low   pitch.v1           P4's pitch not under P1's
high  0.9*chord.v0       at most 0.9 of the root chord
```

The variable then lies between its bounds wherever the other variables are. When one of them moves it keeps its
place between its bounds, the fraction of the way from low to high, and that fraction, 0 to 1, is the
variable's entry in an optimiser's design vector: a vector inside its bounds gives a design inside its bounds,
with nothing to check or repair. This is how the MSc propeller optimisation bounds its control points
(para_control_bez_updated.py): P4's pitch is `p1y + (max_pitch - p1y) * y4_frac`, so `pitch.v4` with low
`pitch.v1` and high `1.4` has the MSc's `y4_frac` as its vector entry (the self-check compares the two). P2 and
P5 need nothing: the 2-segment form's `d1` and `d2` are those fractions already.

A relation is an operator and an expression, after the flag:

```
<= 0.5*chord.v2 + 2*chord.v3      at most that            (< reads the same)
>  pitch.u4 + 0.2                 at least that           (>= the same)
=  chord.v0                       that value: the variable follows the others and is no design variable
```

A relation is a condition on top of the bounds. The variable stays in the design vector as its own value, and
an optimiser checks the relation or projects the vector onto it (the MSc pipeline's coupled limits, chord_sum
and pitch_diff, with reject or project). In the tool the variable stays where it is until the relation would
break, and is then moved onto it.

Which one: a dependent bound when the variable's range itself moves with another variable and every sample
should lie inside it (the sampling stays a box); a relation for a limit that binds only sometimes, or to tie a
variable to others exactly (`=`).

- The expression: numbers, variable names, `+ - * /` and brackets; `0.5v4` is read as `0.5*v4`. A bound is one
  word, written without blanks (`0.9*chord.v0`), because blanks separate the var line's columns; a relation,
  the last thing on the line, may have blanks. Several relations are separated by commas (`> 0.5*chord.v0, <
  chord.v1`); `=` stands alone.
- Names are the var lines' names. A variable of the same curve may be named by its key (`v2` for `chord.v2`);
  the files are written with full names.
- A variable may depend on dependent ones, but not, in a loop, on itself, through bounds, relations or both
  (refused, the loop named). An expression cannot use its own variable.
- What it does: the variable's limits are its bounds (a dependent one where the other variables put it now)
  narrowed by its relations. In the tool a variable moves inside its limits, and when a variable it depends on
  moves it follows, in the order of their dependence: one with a dependent bound to its place between its
  bounds, one whose relation would break onto it. Relations that cannot hold inside the bounds leave the
  variable on a bound. Bounds can cross, low above high, where a number faces an expression (low 40, high
  `chord.v1`, and `chord.v1` under 40); bounds such as `0.5*chord.v0` .. `chord.v0` (a chord is positive), or
  ranges that do not overlap (the MSc's: P1's pitch up to 1.25, P4's from it to 1.4), never do. The tool warns
  of both (the build still runs).
- For an optimiser (`ld.space` of `load_component`, every kind's): `space.factors` names
  the free variables with a dependent bound; their entries of the design vector are fractions and their rows of
  `space.bounds` are 0, 1. `to_design(x)` places them, in the order of their dependence, `to_vector(design)`
  gives them back, and `space.bounds_of(values)` has every variable's bounds at a design. A held variable with
  a dependent bound keeps the fraction it has in the set-up's design. (`to_design` and `to_vector` are what the
  MSc pipeline's `designvars_to_conpoints` and `conpoints_to_designvars` are.) A free variable with relations stays in
  the vector as its value, and its relations are constraints: `space.violations(x)` lists what a vector breaks
  (a relation that does not hold, bounds that cross: variable, text, amount) and `space.project(x)` moves the
  variables onto their relations (fractions kept). A variable with `=` leaves the vector; `to_design(x)` works
  it out.
- Reading: a bound or a relation with a wrong grammar is refused with its line (`<path>, line <n>: the low
  bound of <name>: ...`, `... the relation of <name>: ...`), one that names no variable or closes a loop with
  the file's path. A dependent bound's number is its expression's value at the file's design. A file whose
  design breaks one of its relations loads as it is written (`space.design_violations(design)` says what); the
  tool then applies the relations and says which variables moved.
- A dependent bound or a relation goes with its variable: a curve given another form (points, segments) loses
  those of its variables, and one elsewhere that names a variable now gone is dropped (the relation gone, the
  bound the number it last had), which the tool says.
- The rudder-h model's H keeps numbers for its bounds (h_min, h_max).

In a component file (every kind):

```
# var    chord.v0                     120                40               150  Y
# var    chord.v1                     100      0.5*chord.v0          chord.v0  Y
# var    chord.v2                      90                40               150  Y  < chord.v1, > 0.25*chord.v0
# var    pitch.v1                       2                -3                 4  Y  = pitch.v0
```

The rudder modified below H the same (26 Nov 2026; until then a JSON record per variable):

```
# var    LE.dx2                      -200              -320                 0  Y  > -2.5*H
# var    LE.dx3                       -50            LE.dx2                 0  Y
```

A build's record of that rudder, `<case>_design.json`, has `vector` (names, values, bounds: what an optimiser's
design vector is), a variable with a dependent bound there as its fraction, with the bounds 0, 1, and named in
`fractions`.

A file without dependent variables is written as before: the var lines of independent variables are
unchanged. `Initial_sampling.py` reads neither dependent bounds nor relations (the MSc pipeline
samples para_control's own variables, which are these fractions already, with its own coupled limits). Code:
`xgeom_relations.py` (the grammar: `parse`, `parse_bound`; `bounds_at`, `places`, `limits`, `apply`,
`violations`, `crossed`; `SpaceRelations` for the design spaces, `AdapterRelations` for the tool's adapters).
The self-check is `_xgeom_relations_test.py`.

## Bounding laws: a curve between two laws

A curve of the rudder modified below H may be given a low law and a high law (26 Nov 2026). Its value at every
height y is then

    value(y) = low(y) + f(y) [high(y) - low(y)]

where f is the curve's own Bezier law over the height, its control values the fractions 0 to 1. A Bezier curve
lies within its control values, so the curve lies between its two laws at every height whatever the fractions
are. The fractions are the curve's variables: `P2z.f1`, `P2z.f2`, .. (C1 to the root's Cn; bounds 0 to 1), with
the control heights `P2z.d1` .. as before. C0, at H, is the fraction that puts the curve on the original there
(0 when the two laws meet at H), so such a curve joins the original with its value only (no tangent switch),
and the CAD is two lofts joined at H.

A law is a number or an expression in the grammar of the relations (+ - * / and brackets), with `min( )` and
`max( )` of two or more expressions (new on 26 Nov, in relations and dependent bounds too), written as one
word. The names it may use, each meaning the value at the same height:

| name | |
|---|---|
| `LE`, `P1x`, `P1z`, `P2x`, .. | the design's own curves (LE: x of the leading edge, mm; the others: fractions of the sharp chord) |
| `orig.LE`, `orig.P2z`, .. | the original's |
| `TE` | x of the TE line, mm |
| `chord` | the section's sharp chord, mm: the length its coordinates are fractions of |
| `orig.chord` | the original's sharp chord at that height |

So `orig.P2z*orig.chord/chord` is the original's P2 height in mm, as a fraction of the new chord: as the low law
of P2z it keeps P2 at least as high as the original's wherever the LE goes. The curves are worked out in the
order of their dependence; a curve cannot use itself, a loop is refused (the chord counts as the LE: it follows
from it, so the LE's own laws cannot use it). The chord a law names is the sharp chord of the section the laws
give, found with them (a fixed point of a few steps; it equals the built sections' to 1e-12 mm).

- The low law is where the fraction is 0; it may be the larger of the two (the fraction then runs down).
- Low and high the same expression set the curve: no variables (P1z with `P2z` and `P2z` is as high as P2).
- Warnings (the preview's, in the status bar): laws that cross along the span; the original at H outside its
  laws (the curve then leaves them near H); laws that meet at H away from the original (a step at H). A law
  that gives no number (a division by zero) is a problem: no build.
- A design with laws keeps its section degree (the laws name its curves).

In the set-up file (kind `rudder_h`) the two laws follow the points on the curve line, in the place of the
join:

```
# curve  P2z       bezier1 4 orig.P2z*orig.chord/chord 10*orig.P2z*orig.chord/chord
# var    P2z.d1        0.333333333333              0.05              0.95  Y
# var    P2z.d2                   0.5              0.05              0.95  Y
# var    P2z.f1                     0                 0                 1  Y
# var    P2z.f2                     0                 0                 1  Y
# var    P2z.f3                     0                 0                 1  Y
# curve  P1z       bezier1 4 P2z P2z
```

The var lines of such a curve are its `d` and `f` (none for a curve its laws define), and the TE radius the
chord is found with is the setting `te_radius_mm`. Reading: a law with a wrong grammar is refused with its line
(`<path>, line <n>: the high law of P2z: ...`), one that uses a name no law can use with the file's path.
`<case>_design.json` has `design.laws`, and for a curve with laws its `low` and `high` and its fractions as
`value`.

### The original as the lower bound (keep the original inside)

Laws and bounds act on control points. They make most designs enclose the original, not all: with every
fraction 0 and the root LE 80 mm forward the wind-tunnel rudder's modified section lies 0.34 mm inside the
original near mid chord. What tells is a measure on the built sections: `rudder_modify.inside_original(design,
orig, n_sections, te_radius_mm)` gives the largest depth (mm) of the modified half-thickness under the
original's at the same x, over the sections from the root to H (`depth`, with its height `y` and station
`x_c`; `depths` per section; 0 for the original itself). The set-up's settings carry `keep_inside` (default
on) and `inside_tol_mm` (default 0.01): the tool warns when the depth is beyond the tolerance, and an
optimiser keeps `depth <= inside_tol_mm` as a constraint (`rudder_modify.load_component`'s text shows how).

### The set-up that builds on the original (rudder_laws_setup.py)

`python rudder_laws_setup.py [--a 10] [--b 10]` writes `outputs/modified/add_to_original.txt` (a component
file of kind `rudder_h`; `python xgeom_tool.py rudder --file outputs/modified/add_to_original.txt` opens it)
for the wind-tunnel rudder: the rule of the planar-rudder optimisation (the folder `Planar_Rudder_Optimization`)
as bounding laws. Every curve a cubic, no curve tangent at H, 26 free variables (H; 5 for the LE; 5 for each of
P2x, P2z, P3x, P3z); P1x held at 0. With X2 = `orig.LE+orig.P2x*orig.chord` (x of the original P2, mm), TE0 =
`orig.LE+orig.chord` (the original's sharp TE), y2 = `orig.P2z*orig.chord` and y3 = `orig.P3z*orig.chord` (the
original P2 and P3 heights, mm):

| curve | low law (fraction 0: the original) | high law (fraction 1) |
|---|---|---|
| P2z | `y2/chord` | `a*y2/chord` |
| P1z | `P2z` | `P2z` |
| P3z | `y3/chord` | `b*y3/chord` |
| P2x | `(X2-LE)/chord` | `0` |
| P3x | `min(TE0-LE, X2-LE+(orig.P3x-orig.P2x)*orig.chord*(P3z*chord-y2)/(y3-y2))/chord` | `max(X2-LE, TE0-LE+(orig.P3x-1)*orig.chord*P3z*chord/y3)/chord` |

The LE is a free cubic with dependent bounds: `LE.dx3` -160 to 0; `LE.d1` 0.05 to 0.95; `LE.d2` 0 to
`1-0.05/(1-d1)`; `LE.dx1` from `dx3-0.07*(1-d1)*H` to 0 and `LE.dx2` from `dx3-0.07*(1-d1)*(1-d2)*H` to 0 (the
control points between the original LE at their heights and the root's x; 0.07 the original LE's slope).
Checked against the folder's own code on 40 random designs (the same 26 numbers, the original's P2 and P3
taken from each section's own LE): P0 to P3 of every section agree within 0.08 mm with a = b = 2 and within
0.23 mm with a = b = 10 (their table read linearly between sections and their laws on 100 samples, against
fits and exact curves here).

Code: `xgeom_relations.py` (`parse_law`, `sequence`, `min`/`max`), `rudder_modify.py` (`Design.laws`,
`law_names`, `canonical_laws`, `law_state`, `law_bands`, `law_warnings`, `with_laws`, `control_points`,
`inside_original`; `DesignSpace(laws=, te_radius_mm=)`; `load_component`, `write_component`), `xgeom_files.py`
(the curve line's laws), `xgeom_rudder.py` (`set_laws`, `laws_of`, `law_defined`, `law_names`,
`set_keep_inside`). Self-checks: `_xgeom_relations_test.py`,
`_rudder_modify_test.py`, `_xgeom_tool_test.py`.

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
  file; the dialog opens on the file it came from. Every component has it, the rudder modified below H too
  since 26 Nov 2026 (its File box in the place of the Set-up box: the set-up file, the original it names, Load
  ... and From CAD ...; Save set-up's dialog opens on the output folder and the case when the design came from
  no file). `--file` takes any component's file; `--space` is gone, and a JSON file is refused by name.
- Each variable's row: free | name | low | slider | high | value | where | dep (24 Nov 2026). The name is
  what an expression calls the variable. The low and high boxes take a number or an expression of other
  variables (`v1`, `0.9*chord.v0`; 25 Nov 2026): a dependent bound, shown in the box as the tool writes it, and
  in full on a line under the row when the box cuts it. dep unticked: independent. Ticked (by hand, or because
  a bound is an expression): a line opens under the row for a relation (`> 0.5*v4`, Return sets it) and shows
  the limits the variable has now; the slider runs over them, and the variable follows the ones it depends on.
  With `=` the row's slider, value and free box are off. Unticked again: independent, the relation dropped and
  the dependent bounds numbers where they are.
- The rudder modified below H (26 Nov 2026): the Curve box has a low law and a high law box (both given: the
  curve lies between them and its rows are the fractions; one alone waits for the other; both emptied: no
  laws; the curve view draws the two laws). The Region box has keep the original inside with its tolerance
  (mm) and, beside it, how far inside the rudder is now.
- The Rudder slot has two options (26 Nov 2026; a Rudder box at the top of its panel, where the planar model
  is): modify an existing rudder (rudder-h: the panel above) or design a new rudder (the planar model, the
  wing's panel, starting from the new sample `rudder.txt`: 200 mm span, chord 180 to 152 mm, LE swept 14 mm,
  NACA 0018). Each keeps its state while the other is shown. In the new rudder's File box Load ... of a planar
  rudder's file loads it there, and of a rudder-h set-up puts the slot on that option; a new rudder saved with
  Save set-up loads in the other option as the original to modify (a planar rudder's file there, below).
- From CAD ... (13 Nov 2026, `xgeom_from_cad.py`): in the File box of the blade, the hull and the rudder. A
  STEP or IGES file of the component is extracted (blade_extraction.py, hull_extraction's
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
  `<case>.txt` (the set-up, kind `rudder_h`, naming the extraction's files). The run is off the GUI thread (a hull a few seconds, the BB2 propeller about a
  minute, the wind-tunnel rudder about two minutes). Needs OCC, as the extraction codes do.
- `python xgeom_from_cad.py blade|hull|rudder <file> [--out DIR] [--case NAME]`: the same without the window;
  for the blade an extracted `<case>.txt` and for the hull a table file `<case>_table.txt` give the set-up alone (no
  OCC needed). For a planar table (a WingParams) `best_wing_forms` and `wing_setup` do the same.
- `python xgeom_from_cad.py rudder_planar <extraction folder> [--case NAME]` (18 Nov 2026; no OCC needed): a
  planar rudder's set-up from Rudder_geom_extraction's outputs (`planar_rudder_setup`): 21 rows to the top of
  the LE, the LE and TE lines straight, the section the rows' mean thickness shape
  (`<case>_section.dat`) with the change points that fit best, the tilt and the radius cap of parameters.json.
  The wind-tunnel rudder's is `outputs/wind_tunnel_rudder.txt`.
- The Rudder slot's modify option shows the modified-below-H model (rudder-h, its set-ups and From CAD)
  whatever is loaded there. Since 19 Nov 2026 (the user: loading BB2's or the wind-tunnel rudder's set-up must give the same
  panel, tangent at H with it) `--file` with a planar rudder's component file, or Load ... of one in the
  File box, makes that rudder the rudder-h model's original (`xgeom_from_cad.rudder_from_planar`; `python
  xgeom_from_cad.py rudder <file>.txt` the same without the window): its sections, tip and cap written as an
  extraction writes a CAD's into `<name>_modified_below_H/` beside the file (61 horizontal sections, 121
  points each), the sections fitted (about 25 s the first time), the set-up
  `<name>_original.txt` there (the rudder unchanged, H 0.1 to 0.4 of the top section, its TE
  radius), read back while the file and its section file are unchanged; the cases of it go to that folder.
  From 13 to 19 Nov such a file put the rudder on the planar model (the wing's panel); since 26 Nov that
  model is the slot's other option, design a new rudder.
- Vehicle tab (step 4): Load assembly ... and Save assembly ... (each component written to its file; old .dat
  sources as new .txt files beside the assembly).
- `python xgeom_tool.py --file <component file>`; `--assembly <file>` (step 4).

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

25 Nov 2026: dependent bounds (Variables, above): the low or high word of a var line may be an expression of
other variables; the rudder-h model's set-up JSON takes `lo_expr` and `hi_expr` in a slot and `fractions` in
its vector; the title lines of the curves and variables block say what makes a variable independent or
dependent (the five samples re-saved with them, nothing else in them changed). For the web app: additive. A
file with no dependent bound differs only in the separator lines; one with a dependent bound needs this
version to read (an older reader takes low and high as numbers only). `bounds(name)` returns numbers as before,
a dependent bound where the design puts it now, and `meta[name]["lo"]`, `["hi"]` hold the same numbers;
`meta[name]` has `"lo_expr"`, `"hi_expr"` for a dependent bound only. `set_bounds(name, low, high)` takes
numbers as before, or an expression's text for either; given numbers, it makes the bounds numbers, so a
caller that wants to keep a dependent bound passes its text back, from the new `bound_text(name)` ((low, high)
texts, None for a number; `short=False` for full names). New `set_independent(name)` (the relation dropped,
the bounds numbers where they are); `is_dependent(name)` is true for a dependent bound too; `set(name, value)`
moves the variables whose bounds use `name` with it; `preview()["warnings"]` also names bounds that cross. The
spaces: `factors`, `dependent_bounds`, `bound_text(name)`, `bounds_of(values)`; `to_vector` and `to_design`
carry the fractions; `xgeom_files.Var` has `lo_expr` and `hi_expr` (`''` for a number).

26 Nov 2026: bounding laws, the original as the lower bound, and the Rudder slot's two options (Bounding laws,
above; GUI). Nothing changes for a set-up without laws except two new settings (`keep_inside` true,
`inside_tol_mm` 0.01) and, in `space`, `laws` (empty) and `te_radius_mm`. For the web app: additive. The
rudder adapter gains `laws_of(curve)`, `set_laws(curve, low, high)`, `law_defined(curve)`, `law_names()`,
`set_keep_inside(on, tol_mm)`; its `preview()` gains `"bands"` ({curve: (low, high)} at `yy`, for the curves
with laws) and `"inside"` (`inside_original`'s result, None when not measured), and its `"warnings"` may name
the laws and the depth inside the original. A curve with laws has variables `<curve>.f<k>` in place of
`<curve>.v<k>` (rows "C1 fraction" ..), its `design.curves[c].v` are those fractions and its join is G0:
`rudder_modify.control_points(design, orig, curve, te_radius_mm)` gives its control points as a plot of the
curve shows them, and `curve_values(design, ys, orig, te_radius_mm)` every curve's values (a design with laws
needs the original there). `set_tangent(curve, True)` and `set_degree` raise ValueError for a curve, or a
design, with laws. `xgeom_relations`: `parse_law`, `sequence`, `min( )`/`max( )` in every expression;
`bezier_section.te_x_from`; `rudder_modify.sharp_chord` (bezier_section's, faster). New files: `rudder.txt`
(the new rudder's start) and `rudder_laws_setup.py` (a script; the app does not need it). The self-checks:
relations 138 with the window, rudder_modify 147, the tool 83 with the window, wing 80.

26 Nov 2026, later: every set-up a component file (the user: "everything should be in the current set up
format, delete anything extra or old"). The rudder modified below H is kind `rudder_h` (Component file, above):
its curve line carries the join at H or the two laws, its var lines are every kind's. The JSON set-up is gone:
`<case>_design_space.json` is no longer written or read, and a JSON file given to the tool or a reader is
refused (`<path>: a JSON file, not a component file: XGeom no longer reads JSON set-ups ...`). An XCAD build
writes `<case>.txt` beside the XCAD file, From CAD `<case>.txt`, a planar rudder made the original
`<name>_modified_below_H/<name>_original.txt`, `rudder_laws_setup.py` `outputs/modified/add_to_original.txt`.
The seven old set-ups on the user's machine were converted (the same values to 12 figures, each read back
against its JSON) and then deleted. A var line's bounds now have 12 figures, as its value has (files with
shorter numbers are unchanged byte for byte). H's upper bound at the top section, and H on it, read back as
the top section itself. The table of a `rudder_h` file is the design as the file's own figures give it, so a
file read and written again is the same bytes whatever the rudder's size (a height over a metre in mm has
more figures at 8 decimals than a var line carries; a planar component that large, kind rudder, fin, wing or
bwb, still changes its table's last decimal on the first rewrite: open).

**Breaking** for the web app (a change in its bridge, `docs/keeping-interfaces-stable.md`): the rudder adapter's
`space_record()`, `save_space(path)`, `load_space(path)` and `setup_file` are gone. In their place, as every
other adapter has: `save_file(path)` (returns the path), `load_params(path)`, `file` (the set-up in use, None
before one is saved or loaded) and `RudderAdapter(params=path)`; opening the saved file and saving again gives
the same bytes, LF only (`_xgeom_web_contract_test.py` now runs the rudder through the same checks as the
others: 271 checks). `rudder_modify.load_design_space(path)` -> `rudder_modify.load_component(path)` (an
`xgeom_files.Loaded`: `params`, `design`, `space`, `settings`, `name`, `notes`); `DesignSpace.to_dict` and
`from_dict` are gone (`write_component`, `component_design`); `design_from_record` no longer reads the layout
of before the control-point names (T1, X1, X2, T2). An XCAD build's result has `["case"]["paths"]["component"]`
in the place of `["space"]`. The tool: `--space` and the alias `--params` are gone (`--file`), `App(root,
geometry, params)` has no `space`. `xgeom_files`: `MODIFIED` (`"rudder_h"`), in `KINDS`; `Component.joins` and
`.laws`; `rudder_modify` and `xgeom_rudder` now import `xgeom_files` (already in the copy list). A set-up the
web app saved as JSON is not read; it is saved again from the app. The self-checks: relations 138 with the
window, rudder_modify 150, the tool 83 with the window, wing 80, files 60, hull 65, blade 79.
