# Keeping XGeom's interfaces stable

## Why this matters

The web app runs X\_geom's own Python unchanged, so every name and file key in X\_geom is also part of the web app. Adding things is cheap there; renaming or changing the meaning of things is not. This note says which parts to keep stable, and how to grow the tool without breaking them.

The 19 Nov vehicle change (positions in metres became x/L) is a good example of the cost. It renamed `prop_dx` and `rudder_dx` to `prop_x` and `rudder_x`, and kept the names `fin_x` and `wing_x` but changed their unit. On the web side that means changes to:

- the bridge (`python/xgeom_web/vehicle.py`) and the vehicle inspector's labels, units and slider ranges;
- the tests that place components and save assembly files;
- the contract test, which reported the rename as a failure;
- every assembly file and saved revision written before the change: they no longer open, or, for `fin_x` and `wing_x`, open with the fins in the wrong place and no error.

The web app has no active users yet, so this time it is a one-off clean switch. Once people keep designs in it, a change like this would need a converter on both sides.


## What the web app depends on

These are X\_geom's public interface: change them only with a plan (section 5). Everything else, including how each one works inside, can change freely. `docs/_xgeom_web_contract_test.py` in the web repo checks all of it; run it from X\_geom's folder before pushing.

| Area | What the web app uses | Examples |
| --- | --- | --- |
| Module names | the modules each component needs, imported by name | `xgeom_hull`, `hull_modify`, `blade_modify`, `blade_control`, `x_blade_new`, `prop_hub`, `blade_sections`, `xgeom_rudder`, `rudder_modify`, `bezier_section`, `modify_control`, `xgeom_wing`, `wing_modify`, `xgeom_vehicle`, `xgeom_files`, `airfoils` |
| Component files | the first line, every key's name, unit and meaning, the defaults, LF line endings, and the same bytes when an unchanged file is saved again | `# XGeom component`, `# kind fin`, `span_mm`, `hub_base_m`, `section`, `part` |
| Assembly files | the first line, the component and placement keys, the run keys | `# XGeom assembly`, `PLACEMENT_KEYS`, `RUN_KEYS` |
| Adapters | the methods and what they return | `meta`, `names`, `bounds(name)`, `is_free(name)`, `set(name, value)`, `n_free()`, `global_rows()`, `curve_rows(curve)`, `summary()`, `describe(curve)`, `save_file(path)`, `load_params(path)` |
| Form changes | the methods behind the Settings controls | `set_order`, `set_sections`, `set_segments`, `set_blades`, `set_elliptic`, `set_degree`, `set_span`, `set_tangent`, `set_changes`, `set_hub_shape` |
| `preview()` | the entries the views draw | hull `r`, `rp`, `x`; blade `curves`, `rows`, `rr`, `file`, `hub_outline`; fin and wing `le`, `te`, `sections`, `shapes`; `skeleton` (the CAD sections) |
| Internals (for now) | a few attributes the 2D views still read | `design.curves`, `design.ctrl(curve)`, `adapter.settings["sections"]`, `rudder_modify.original_section_at` |
| Vehicle | the parts and placement API | `PARTS`, `OWNS`, `COLLECTIONS`, `Placement` fields, `default_placement()`, `placement_problem()`, `assemble()` and its `info` keys |
| Messages | how errors start, so the web app can explain them | `airfoil '<name>': not in …`, `; close: a, b, c`, `<path>, line <n>: …` |

## Rules for changing things without breaking them

1. **Never give an existing key a new meaning or unit.** Use a new name instead. `fin_x` changing from metres to x/L is the riskiest kind of change, because an old file still opens, just wrong. Had it become `fin_xl`, an old file would have failed loudly, or could have been converted.
2. **Put the unit in the key's name**, as most keys already do (`span_mm`, `hub_height_m`, `tip_tilt_deg`, `cap_h_le_mm`). For fractions, a suffix such as `_xl` (of hull length) or `_rR` (of the radius) makes a later unit change a new key by itself.
3. **Add, don't rename.** A new key or setting must be optional, with a default that reproduces the old geometry exactly. `tip_tilt_deg` (default 0, a flat tip), `cap` (default none) and the conic hub's `hub_base_m`, `hub_top_m`, `hub_x_m` (absent means a cylinder) all did this well.
4. **If a name must change, read the old one too.** Write only the new name, but keep accepting the old one and convert it, so old files open unchanged. A file with `prop_dx` is certainly from before the x/L change, so it could have been converted using the hull's length instead of being refused.
5. **Keep adapter methods and their return shapes.** Add methods and add optional arguments at the end; don't change what an existing method returns. A tuple stays a tuple, a list of rows keeps its fields.
6. **`preview()` only grows.** Add entries freely. Removing one (as `le`, `te`, `mid` and `tip` were from the blade) is a breaking change: say so in `progress.md`, even if you believe nothing uses it.
7. **Keep module names and the folder layout.** The web app imports modules by name and copies a list of them. New modules are fine; just mention them, so the copy list grows. The `.dat` to `.txt` switch earlier showed how far a file-level rename reaches.
8. **Same design, same bytes.** Saving an unchanged design should give a byte-identical file: no times, dates or random values in it. The rudder's `"written"` timestamp breaks this (section 7).
9. **Keep identifiers apart from labels.** Labels, help text, menu wording and window layout can change any time. Key names, `kind` values, curve and variable names, and method names are the interface.
10. **Keep the start of error messages.** The web app recognises some by how they start (section 2). Rewording the rest of a message is fine.

## Changes that are always safe

These need nothing from the web side beyond the next sync:

- **New optional keys and settings** with a default that keeps old designs as they were (rule 3).
- **New design variables** on an adapter: they appear in `meta` and the rows, and the web app shows them with sliders automatically.
- **New form changes** (`set_*` methods). The web app adds a control for each one it is told about, as it did for `set_hub_shape`.
- **New `preview()` entries**, new `info` keys from `assemble()`, new collections in `COLLECTIONS`.
- **New modules**, and new components with a new `kind`, as long as their adapter offers the same methods as the existing ones.
- **Changes inside**: better fitting, faster code, new CAD steps. If the same inputs give the same outputs and files, the web app doesn't notice.
- **Labels, help text and messages** after their first words.

The tip cap and the tilted tip (17–18 Nov) were almost entirely of this kind. Their one break was small: `cap_loops` is now written into every fin and wing file, and older copies of X\_geom refuse a key they don't know. Section 5 covers that case.

## When a breaking change is unavoidable

Some changes are worth a break, like x/L positions. Make them planned rather than surprising:

1. **Give the files a format number.** Add a line after the first, `# format 2` (a file without one counts as format 1). Raise it whenever a file's keys change: added, renamed, or with a new meaning.
2. **Read older formats, and convert them.** The reader knows exactly what changed between versions, so `prop_dx` becomes `prop_x` and the metres become x/L in one place. Old files then open everywhere, desktop and web alike.
3. **Refuse newer formats with a clear message**, for example "made by a newer XGeom (format 3); this one reads up to format 2". Today an older copy says `'cap_loops' is not a key`, which is true but doesn't tell anyone to update. Keep refusing unknown keys within a format: ignoring one could silently change the geometry (an ignored `cap` drops the tip cap).
4. **Batch the breaks.** Several renames in one format change are cheaper for the web side than one per week. Each break means a sync, a bridge change and new tests.
5. **Announce it before pushing.** Add a `progress.md` entry headed **Breaking**, with old name → new name, units, and what happens to old files. Tell Eddie, so the web sync is ready when it lands.
6. **Update the samples in the same commit.** Re-save `fin.txt`, `wing.txt`, `suboff_hull.txt` and `msc_blade.txt` with Save set-up, so the shipped files are what the tool writes. The contract test checks this.

With a format number in place, the web app's next sync can become automatic: a scheduled job pulls X\_geom, runs the contract test and the full suite, and opens a merge request. It can only do that safely if breaks are rare and announced.

## Checklist before pushing

- [ ] The contract test passes: `python _xgeom_web_contract_test.py` from X\_geom's folder, with numpy, scipy, pandas and matplotlib installed (without pandas, the blade's checks fail for that reason alone).
- [ ] No key was renamed or given a new meaning or unit. If one was: a new format number, old files converted, a **Breaking** entry in `progress.md`, and Eddie told.
- [ ] Every new key or setting has a default that keeps old designs unchanged.
- [ ] No adapter method, `preview()` entry or module the web app uses was removed or renamed (the table in section 2).
- [ ] Saving an unchanged design gives the same bytes: no times, dates or random values in files.
- [ ] If the file format changed, the four samples are re-saved in the same commit.
- [ ] New modules are mentioned in `progress.md`, so the web app copies them.

## Open requests

None of these block anything; the first three take minutes. Items 1–5 are from the 18 Nov message and still open as of commit `a5eba52`; 6 and 7 are new.

| # | Request | Why | Where |
| --- | --- | --- | --- |
| 1 | Re-save the samples with Save set-up | The tool now writes `# part` (blade) and `# cap_loops` (fin, wing), so the shipped files are no longer what it writes. The contract test notes it for all three | `msc_blade.txt`, `fin.txt`, `wing.txt` |
| 2 | Add a `.gitattributes` with `*.txt text eol=lf` | Git on Windows turns the samples to CRLF on checkout, so they stop matching the repository. The contract test notes it on every Windows run | the repo's root |
| 3 | Drop the rudder's `"written"` timestamp, or keep it out of the file | It changes on every save, so an unchanged rudder never saves to the same bytes (rule 8). The web app now ignores it when telling revisions apart, but other tools won't | `xgeom_rudder.py`, `space_record()` (line 419) |
| 4 | Question: could `preview()` draw the blade's root extension? | With a conic hub, the preview's blade starts at the root ring, so the 3D view shows a gap to a narrower cone. The CAD's `root_extension` fills it, so the solid is fine; the view just doesn't show it. A preview entry, or extra skeleton sections, would let both tools draw it | `xgeom_blade.py` and `blade_modify.root_extension` |
| 5 | Check that the BB2 details in the `bb2/` scripts are fine in a public repo | BB2 is MARIN-confidential. The scripts hold no geometry (they read `prop.stp` and `hull.iges`, which aren't in the repo), but their comments name the propeller (S7371R) and some dimensions (a hub radius of 797.22 mm). The web app never copies `bb2/` | `bb2/*.py` |
| 6 | Import `matplotlib.pyplot` inside the functions that plot | The blade imports it on the way, which costs the browser 9 MB of download and start-up time; nothing is drawn with it there | the blade's modules (the contract test names them) |
| 7 | Consider reading old assembly files (rule 4) | A file with `prop_dx` is refused, and an old `fin_x` or `wing_x` in metres is read as x/L. The web app has no users yet, so it switches cleanly; desktop users' older files are worth converting | `xgeom_files.read_assembly` |
