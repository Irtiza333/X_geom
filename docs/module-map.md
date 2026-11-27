# XGeom module map

Every Python module in X_geom: what it does, where it lives and which self-checks cover it. Written 20 Nov
2026; add a line when a module is added, and move it when it changes role.

- **Where.** `X_geom` is this repository. `both` means the module is also in the Rudder folder
  (`C:\Users\ikhan1\Claude\Projects\Rudder`), where the two copies must stay byte-identical. That folder has
  no planar code, so it holds only the rudder modules.
- **web.** The web app imports the module by name. Its names, methods, keys and `preview()` entries are the
  web app's interface: change them only as `docs/keeping-interfaces-stable.md` says, and run
  `python _xgeom_web_contract_test.py` before handing a change over.
- **Tests.** The self-checks that import the module (short names, listed under Self-checks below). A module
  with none is exercised through the modules that import it, or not at all (old code kept for reference).

## The design tool

`python xgeom_tool.py`: the window with a panel and views per component, the build and Save CAD buttons, the
Vehicle tab. Each component is an adapter class that the window drives through the same methods.

| Module | What it does | Where | Tests |
| --- | --- | --- | --- |
| `xgeom_tool.py` | The window (Tk + matplotlib): component switcher, panels, sliders, views, build, Save CAD, From CAD, the Vehicle tab; the Hull slot's two bodies (the cylindrical hull, the blended wing body); the Rudder slot's two options (modify an existing rudder, design a new one); a curve's low and high law boxes | both | tool, wing, hull, blade, files |
| `xgeom_common.py` | What the adapters share: the panel's row spec, the colours, the fit of a 3D box to its panel (no Tk) | both | through the adapters |
| `xgeom_rudder.py` | The rudder modified below H: rudder_modify's design space as variables, views, its set-up as a component file (`save_file`, `load_params`, `RudderAdapter(params=)`), the XCAD build (one loft; two when not tangent at H); a curve between bounding laws (`set_laws`), the original kept inside by measure (`set_keep_inside`) | both, web | tool, wing, hull, contract |
| `xgeom_blade.py` | The propeller blade: a blade file as variables, views, the CAD build (whole, half, 1/Z sector); the hub a cylinder or a cone, and which end of it moves with its height (`set_hub_extends`) | X_geom, web | blade, hull, contract |
| `xgeom_hull.py` | The hull: a hull file as variables, views, the CAD build | X_geom, web | hull, wing, contract |
| `xgeom_wing.py` | The wing and the fin (and the planar rudder's own model), and the blended wing body (`BwbAdapter`, the Hull slot's other body: mirrored views, the whole or half CAD): a wing file as variables, views, the CAD build | X_geom, web | wing, contract |
| `xgeom_vehicle.py` | The Vehicle tab: the propeller, rudders, fins and wings placed on the hull (positions as x/L), `assemble()` | X_geom, web | hull, wing, files, contract |
| `xgeom_files.py` | The component and assembly files: read and write, every kind (kind `rudder_h`, the rudder modified below H: a curve line's join at H or its two laws); an old `.dat` parameter file or a JSON set-up is refused, named as one | both, web | files, wing, contract, modify, tool, relations |
| `xgeom_relations.py` | Independent and dependent variables: the grammar of a relation (`> 0.5*v4`) and of a dependent bound (low or high an expression, `0.9*v0`), a variable's limits and its place between its bounds, the dependent ones following the others; `min( )` and `max( )`; laws between curves (`parse_law`, `sequence`: bounding laws); `SpaceRelations` (every design space: relations, dependent bounds as the design vector's fractions, violations, project) and `AdapterRelations` (every adapter: relation, set_relation, bounds, bound_text, set_independent, limits); numpy only | both, web | relations, files, contract |
| `xgeom_from_cad.py` | From CAD: a set-up (a component file) from a STEP or IGES file (extraction and fits per component); a planar rudder's file as the original modified below H | both | tool, wing, hull, blade |
| `vehicle_cad.py` | The vehicle as one CAD solid (Vehicle CAD): each component's CAD placed and fused; blade solids, booleans (through cad_kernel) | X_geom | hull, wing, blade, kernel |

## Geometry models

| Module | What it does | Where | Tests |
| --- | --- | --- | --- |
| `rudder_modify.py` | The rudder below H: the original's fits, the Bezier curves and designs, the written sections (extracted plus the model's change), the XCAD writer, its set-up as a component file of kind `rudder_h` (`load_component`, `write_component`); bounding laws (a curve between a low law and a high law, its variables the fraction of the way: `Design.laws`, `with_laws`, `law_state`); `inside_original` (how far the modified rudder goes inside the original) | both, web | modify, tool, wing, blade, contract |
| `modify_control.py` | The rudder model's settings (input files, H bounds, strips, tip drop) and example cases; run it to write cases without the window | both, web | tool, wing |
| `rudder_laws_setup.py` | Writes the set-up that builds on the original (`outputs/modified/add_to_original.txt`, a component file): the planar-rudder optimisation's rule as bounding laws, 26 variables, the original kept inside by measure | both | tool |
| `bezier_section.py` | The Bezier half-section of any degree with its rounded TE (numpy only); `te_x_from`: the rounded TE by Newton's method from nearby tangent points | both, web | modify |
| `tip_cap.py` | The rudder's tip cap: its model, fit and loops (numpy and scipy only) | both, web | tip_cap, wing |
| `wing_modify.py` | Wings, fins, the planar rudder and the blended wing body (kind bwb: every curve flat at the centre line, the smooth 2-segment form, the whole body mirrored and fused, through cad_kernel): the six distributions over the span, sections, tilted tip, cap, design space, component files | X_geom, web | wing, files, contract, kernel |
| `blade_modify.py` | The propeller blade's design space: the distributions as Bezier curves over r/R, the blade surface, its CAD | X_geom, web | blade, files, contract |
| `blade_control.py` | A blade from a component file, built with the MSc blade code | X_geom, web | through blade_modify |
| `blade_sections.py` | Blades whose section shape changes along the span (one section table per radius) | X_geom, web | blade |
| `blade_vary.py` | What changes in a blade file: its vary lines (and their ranges), or its free var lines (`read_free`); standard library only, for `Initial_sampling.py` | X_geom | blade |
| `prop_hub.py` | The propeller's hub: the DRDC cylinder, or a cone with a rounded nose (its CAD through cad_kernel); the propeller's solids and their IGES face by face (each face its own trimmed B-spline surface, as the DRDC IGES) | X_geom, web | blade, kernel |
| `hull_modify.py` | Hulls with elliptical sections from a few CST curves: fit, design space, sections, CAD (through cad_kernel) | X_geom, web | hull_modify, hull, files, wing, contract, kernel |
| `suboff.py` | The DARPA SUBOFF bare hull from its equations (the public example) | X_geom | hull_modify, hull_eq |
| `airfoils.py` | Airfoil sections by name from the offline UIUC database in `airfoils/`, and the NACA 4- and 5-digit equations | X_geom, web | airfoils, wing, contract |

## CAD

| Module | What it does | Where | Tests |
| --- | --- | --- | --- |
| `cad_kernel.py` | The CAD kernel (27 Nov 2026, Eddie's request): the one module that calls OpenCascade, pythonocc-core or cadquery-ocp. Plain data in (points, pole grids, knots, numbers, paths), shapes out as opaque handles: edges (straight, arcs, B-spline), a curve through points, wires, faces (planar, on a cylinder), B-spline surfaces (rational too) and their faces, lofts, sewing (a shell, or a solid), compounds, box, cylinder, revolve, prism, moves and mirror images, common, cut and fuse (two shapes, or many in one boolean), merging same-domain faces, measures (sub-shapes, free boundaries, points along an edge or on a surface, the kind of a face's surface, volume, area and centroid, validity, bounding box, triangles), STEP and IGES. Its function list is the seam a browser build reimplements. xcad_loft, X_CAD_new, prop_hub, vehicle_cad, wing_modify and hull_modify are on it; the extraction modules (and bb2's scripts, some tests) still reach OpenCascade through its binding helpers (`_occ`, `_static`, `_topods`) until they move | both, web | kernel, tool, wing, hull, blade, hull_modify, hull_eq, tip_new |
| `xcad_loft.py` | The solid through section loops (XCAD files): four patches along the span, the tip patch over the cap, the flat root; two lofts joined at a section (`loft_split`); half bodies; STEP and IGES (through cad_kernel, whose functions it keeps under their old names) | both | tool, wing, hull, hull_modify, hull_eq |
| `temp_rudder_cad.py` | A stopgap: the CAD of a file of modified sections with the original above them, as two lofts joined at H (the modified part, the unmodified part); sections of any number of points put onto the original's layout | both | tool |
| `X_CAD_new.py` | The MSc blade's CAD by the DRDC five-surface method (supersedes `X_CAD.py`); OpenCascade through cad_kernel, so either binding | X_geom | tip_new, blade, kernel |
| `X_CAD.py` | The original loft-and-cap blade CAD, kept | X_geom | none |
| `cad_worker.py` | One CAD case in a subprocess, so a pythonocc hang stops only that case | X_geom | none |
| `iges_writer_new.py` | A minimal IGES 5.3 writer for B-spline surfaces | X_geom | none |

The DRDC blade pipeline (DRDC TM 2013-177 and -178), which the blade builds on:

| Module | What it does | Where | Tests |
| --- | --- | --- | --- |
| `x_blade_new.py` | The MSc blade's geometry on the DRDC blade surface (replaces `x_blade.py`'s CAD path) | X_geom, web | blade, hub_center, tip_new |
| `blade_surface_new.py` | The smooth global blade surface b(xi, eta) | X_geom | through blade_modify |
| `tip_surfaces_new.py` | The blade split into its five surfaces | X_geom | hub_center, tip_new |
| `blade_cuts_new.py` | Blade cuts by planes through three points (Sec. 6) | X_geom | through tip_surfaces_new |
| `tip_smoothing_new.py` | Sampling and smoothing of the cuts (Sec. 7) | X_geom | through tip_surfaces_new |
| `hub_new.py` | The per-blade hub sector (Sec. 10) | X_geom | hub_center |
| `para.py` | The MSc blade's sections from its distributions (para.py's section tables) | X_geom | airfoils, blade |
| `para_control_bez_updated.py` | The MSc blade's pitch and chord as Bezier curves with coupled constraints; imports matplotlib.pyplot at the top (Eddie's open request 6) | X_geom | blade |
| `coupled_constraint_config.py` | The coupled constraints' bounds for the module above | X_geom | none |
| `x_blade.py` | The original MSc blade geometry, kept | X_geom | none |
| `para_control.py`, `para_control_bez.py` | Earlier versions of the distributions' control (polynomials ported from MATLAB; rational Bezier) | X_geom | none |
| `fnval1.py`, `spmak1.py` | B-spline helpers ported from MATLAB, for `para_control.py` | X_geom | none |

## Extraction from CAD

| Module | What it does | Where | Tests |
| --- | --- | --- | --- |
| `Rudder_geom_extraction.py` | The rudder's sections, tilted tip section, cap and spanwise parameters from its STEP or IGES | both | rudder_param |
| `rot_axis.py`, `skew_symmetric_matrix.py` | Rotation about an axis (the user's helpers) | both | rudder_param, blade |
| `blade_extraction.py` | A propeller blade's six distributions, section shapes and diameter from its CAD | X_geom | blade |
| `hull_extraction.py` | A hull's stations and hydrostatics from its CAD, or from SUBOFF's equations | X_geom | hull_modify, hull_eq, hull |

## Rudder section fit

The Bezier replica of a rudder section, used to fit the original rudder's sections (28 Sep 2026).

| Module | What it does | Where | Tests |
| --- | --- | --- | --- |
| `control.py` | The fit's driver: settings, bounds, plot | both | none |
| `match_rudder_section.py` | Fits the Bezier half-section with Main_PSO, then rounds the TE like the original's | both | section_fit |
| `replica_funcs.py` | The Bezier half-section and rounded sections from its five numbers | both | modify, section_fit, tool |
| `retruncate.py` | The original's TE circle and the replica's TE rounded like it | both | section_fit |
| `temp_funcs.py` | Reading Selig sections and sharpening the TE (a stand-in until Evan's code) | both | section_fit |
| `Main_PSO.py` | The user's particle swarm optimiser | both | section_fit |
| `bound_check.py`, `penalty.py`, `repair.py` | Main_PSO's sampling within the bounds, constraint violation and repair | both | section_fit |
| `ESPSOLS.py` | Speciation PSO with local search, for surrogate infill (not used by the fit) | X_geom | none |

## Optimization pipeline and figures

| Module | What it does | Where | Tests |
| --- | --- | --- | --- |
| `Initial_sampling.py` | Round-0 designs for the surrogate (Latin hypercube over what `msc_blade.txt` frees: its vary lines, else its free var lines) and their CAD | X_geom | blade |
| `pipeline_config.py`, `pipeline_io.py` | The pipeline's paths and table helpers | X_geom | none |
| `plot_fig1_parameter_space.py`, `plot_fig2_tip_closure.py`, `plot_fig3_five_surfaces.py` | The DRDC pipeline's explainer figures | X_geom | none |

## BB2 (`bb2/`)

The BB2 case study. Its inputs and `bb2/outputs` are confidential and not in the repository; the scripts take
the input paths and hold no geometry files (their comments were cleared of the propeller's name and BB2's
dimensions on 20 Nov 2026).

| Module | What it does |
| --- | --- |
| `bb2_sections.py` | BB2's propeller cut into XGeom blade input (distributions, section tables) |
| `bb2_build.py` | The blade on its conic hub, the aft hull, the vehicles, the 1/6 sector, the quadrant files |
| `bb2_aft_setup.py` | BB2's aft half as XGeom set-up files (the hull, the stern plane as a planar rudder with its cap) |
| `bb2_rudder.py` | The wind-tunnel rudder placed on BB2 by XGeom's rules, compared with BB2's stern plane |
| `bb2_sternplanes.py` | BB2's own stern planes on the vehicles; the quadrant set-ups |
| `bb2_sternplane_loft.py` | BB2's stern plane lofted again with xcad_loft's patches, and the set-up files with it |

## Self-checks

Run from the folder they are in: `python <test>.py`, and `--gui` where noted to drive the window too (needs Tk
and a display). Checks that need OCC or Tk are skipped where those are missing (the user's machine runs the
suites without OCC; the cloud runs them with cadquery-ocp under Xvfb).

| Short name | File | Covers | Where |
| --- | --- | --- | --- |
| tool | `_xgeom_tool_test.py` (`--gui`) | xcad_loft, the rudder adapter, temp_rudder_cad, the window; bounding laws and the original kept inside in the adapter and the window, rudder_laws_setup, the Rudder slot's two options | both |
| wing | `_xgeom_wing_test.py` (`--gui`) | wing_modify, xgeom_wing, the planar rudder and its conversion, fins and wings on the vehicle, the blended wing body (and the Hull slot's Body box) | X_geom |
| hull | `_xgeom_hull_test.py` (`--gui`) | xgeom_hull, the vehicle, the Vehicle CAD | X_geom |
| blade | `_xgeom_blade_test.py` (`--gui`) | blade_modify and its files, extraction, sections, hub, Initial_sampling | X_geom |
| files | `_xgeom_files_test.py` (`--gui`) | the component and assembly files | X_geom |
| contract | `_xgeom_web_contract_test.py` | what the web app relies on (Eddie's; run before handing a change over) | X_geom |
| relations | `_xgeom_relations_test.py` (`--gui`) | xgeom_relations: the grammar, limits and following; relations and dependent bounds in every design space (the MSc optimisation's y4_frac), in the component files (the rudder's set-up among them), in the adapters and the window's rows; `min( )`, `max( )` and laws between curves | both (the Rudder folder: the module and the rudder) |
| modify | `_rudder_modify_test.py` | rudder_modify, bezier_section, the replica's pieces; bounding laws and `inside_original`; the set-up as a component file (kind `rudder_h`) and what the reader refuses | both |
| section_fit | `_section_fit_test.py` | Main_PSO, the replica, retruncate, temp_funcs | both |
| rudder_param | `_rudder_param_test.py` | Rudder_geom_extraction's spanwise parameters | both |
| tip_cap | `_tip_cap_test.py` | tip_cap | both |
| kernel | `_cad_kernel_test.py` | cad_kernel: each function on shapes whose answers are known; the modules moved over to it call no OpenCascade themselves | both |
| hull_modify | `_hull_modify_test.py` | hull_modify | X_geom |
| hull_eq | `_hull_test.py` | suboff, hull_extraction | X_geom |
| airfoils | `_airfoils_test.py` | airfoils, para's section tables | X_geom |
| hub_center | `_hub_center_test.py` | the DRDC hub sector's centring | X_geom |
| tip_new | `_tip_test_new.py` | the DRDC tip pipeline | X_geom |

Tools, not checks: `_cad_compare.py` (the samples' CAD recorded before and after a change to the CAD code,
and the two records compared: faces, volume, area, validity, bounding box, the STEP and IGES data),
`_diag_new.py` (one-run diagnostic of the DRDC tip pipeline), `_iges_de.py` and `_iges_faces.py` (what an IGES
file holds), `_tmp_summarize_chats.py` (a scratch script for chat transcripts, unrelated to XGeom).

## Folders and data

- `airfoils/`: the UIUC airfoil database (about 1,650 sections) that `airfoils.py` reads.
- `outputs/`: the wind-tunnel rudder's extraction and its fits, the planar set-up `wind_tunnel_rudder.txt`, the
  modified cases (`outputs/modified/`).
- `claude/xgeom-file-formats.md`: the component and assembly files. `docs/`: this map and
  `keeping-interfaces-stable.md`. `progress.md`: the work log.
- `Bezier code/` and `Alternate truncation/`: the user's earlier Bezier and truncation work, kept for reference;
  the tool imports none of it.
- At the root: the component samples `fin.txt`, `wing.txt`, `rudder.txt` (a planar rudder: what the Rudder
  slot's new rudder starts from), `suboff_hull.txt`, `msc_blade.txt`, `bwb_glider.txt`
  (the blended wing body: Ye, Wang and Pan's underwater glider) (what the tool writes, LF; the only file format
  since 20 Nov 2026), `airfoil_data_fixed.csv` (the MSc blade's sections),
  the wind-tunnel rudder's CAD (`Wind_Tunnel_Rudder.stp`, `.igs`, `.stl`), `XGeom_plan.pdf`.

On the user's machine but not in the repository: copies named `*-1.py` (`blade_modify-1.py`,
`vehicle_cad-1.py`, `xgeom_blade-1.py`, `xgeom_tool-1.py`, `_xgeom_hull_test-1.py`) and a root copy of
`bb2_sternplane_loft.py` (older than `bb2/bb2_sternplane_loft.py`), and the folders `UIUC-propDB/`,
`geometry/`, `airfoils/sections/`.

## The Rudder folder

Its 32 modules are all copies of X_geom's: `cad_kernel.py`, `xgeom_tool.py`, `xgeom_common.py`, `xgeom_relations.py`,
`xgeom_files.py` (since 26 Nov 2026: the rudder's set-up is a component file), `xgeom_from_cad.py`,
`xgeom_rudder.py`, `rudder_modify.py`, `modify_control.py`, `rudder_laws_setup.py`, `bezier_section.py`, `tip_cap.py`, `xcad_loft.py`,
`temp_rudder_cad.py`, `Rudder_geom_extraction.py`, `rot_axis.py`, `skew_symmetric_matrix.py`, the section fit's
nine (`control.py`, `match_rudder_section.py`, `replica_funcs.py`, `retruncate.py`, `temp_funcs.py`,
`Main_PSO.py`, `bound_check.py`, `penalty.py`, `repair.py`) and seven self-checks (`_xgeom_tool_test.py`,
`_xgeom_relations_test.py`, `_rudder_modify_test.py`, `_section_fit_test.py`, `_rudder_param_test.py`,
`_tip_cap_test.py`, `_cad_kernel_test.py`), all byte-identical.

## Before handing a change over

1. The self-checks of every module changed (the tables above), with `--gui` where they have it.
2. `python _xgeom_web_contract_test.py`: every check passes, and the notes are read.
3. The shared modules byte-identical in X_geom and the Rudder folder.
4. A `progress.md` entry. If a key, a method, a `preview()` entry or a module name changed: a **Breaking**
   entry and a word to Eddie (`docs/keeping-interfaces-stable.md`).
5. A new module: a line in this map, and a mention in `progress.md` so the web app copies it.
