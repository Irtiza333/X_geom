# initial sampling for surrogate model.
# The script supports two input paths:
# 1) read existing control points from file
# 2) generate N new blade designs using optimized Latin hypercube sampling
# What changes (pitch, chord, both, or nothing: the original blade's CAD only) comes from the '# vary' lines of
# the blade parameter file BLADE_PARAMS (see below).

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

import para_control_bez_updated as bezier_constraints
import para
from blade_vary import read_vary
from pipeline_config import INFILL_ROUND, case_id, infill_paths, initial_sampling_paths
from pipeline_io import append_design_row, clear_file, load_table, save_table, split_case_ids
from X_CAD_new import X_CAD_from_design
from x_blade_new import X_blade


# python Initial_sampling.py --round r   -> CAD for round r's proposed designs (infill mode)
parser = argparse.ArgumentParser(description="CAD for proposed designs.")
parser.add_argument("--round", type=int, default=None, help="run infill mode for this round")
ROUND = parser.parse_known_args()[0].round

ROOT = Path(__file__).resolve().parent
PATHS = initial_sampling_paths()
INFILL_PATHS = infill_paths(ROUND)

# Input mode:
# - "existing": read design/control points from EXISTING_CONTROL_POINTS_PATH
# - "lhs": generate N_GENERATED_BLADES control points with optimized Latin hypercube
# - "infill": generate CAD for the current infill round's control points
# - "test": short LHS smoke test of the DRDC CAD path (x_blade_new / X_CAD_new)
INPUT_MODE = "test"
if ROUND is not None:
    INPUT_MODE = "infill"

# In "test" mode, case ids start at CASE_ID_START. In "infill" mode they follow
# pipeline_config.case_id (100 * round + row), so rounds never overwrite each other.
CASE_ID_START = 1

EXISTING_CONTROL_POINTS_PATH = PATHS["existing_input"]
GENERATED_CONTROL_POINTS_PATH = PATHS["generated"]
INFILL_CONTROL_POINTS_PATH = INFILL_PATHS["control_points"]

# Per-case CAD timeout (seconds). pythonOCC hangs on some degenerate geometries
# and cannot be interrupted in-process, so each X_CAD runs in a subprocess that
# is killed if it exceeds this budget. Set to None to disable the guard.
CAD_TIMEOUT_S = 2000

if INPUT_MODE == "infill":
    CONTROL_OUTPUT_PATH = INFILL_PATHS["cad_accepted"]
    REJECTED_CONTROL_POINTS_PATH = INFILL_PATHS["cad_rejected"]
    PLOT_OUTPUT_PATH = INFILL_PATHS["cad_plot"]
    GEOMETRY_DIR = INFILL_PATHS["geometry_dir"]
    CASE_ID_OFFSET = case_id(INFILL_ROUND if ROUND is None else ROUND, 0)   # 100 * round
    # Keep prior outputs so a killed/partial batch can be resumed (cases whose
    # IGES already exists are skipped below).
    RESET_OUTPUT_FILES_DEFAULT = False
elif INPUT_MODE == "test":
    CONTROL_OUTPUT_PATH = ROOT / "New_training" / "test_control_points.txt"
    REJECTED_CONTROL_POINTS_PATH = ROOT / "New_training" / "test_rejected_control_points.txt"
    PLOT_OUTPUT_PATH = ROOT / "New_training" / "test_pitch_chord_curves.png"
    GEOMETRY_DIR = ROOT / "geometry" / "test_drdc"
    GENERATED_CONTROL_POINTS_PATH = ROOT / "New_training" / "test_generated_control_points.txt"
    CASE_ID_OFFSET = CASE_ID_START
    RESET_OUTPUT_FILES_DEFAULT = True
else:
    CONTROL_OUTPUT_PATH = PATHS["accepted"]
    REJECTED_CONTROL_POINTS_PATH = PATHS["rejected"]
    PLOT_OUTPUT_PATH = PATHS["plot"]
    GEOMETRY_DIR = PATHS["geometry_dir"]
    CASE_ID_OFFSET = 0
    RESET_OUTPUT_FILES_DEFAULT = False

# When False, skip chord_sum / pitch_diff coupled limits only (no reject or project).
# Variable bounds and blade-clearance rejection always apply.
USE_BEZIER_VIOLATION_FILTER = False
BEZIER_CONSTRAINT_MODE = "reject"
BEZIER_CHORD_SUM_MAX_NORM = 1.49
BEZIER_PITCH_DIFF_MAX_NORM = 0.73
WRITE_CASE_ID_TO_OUTPUT_FILES = True
CONTROL_POINT_FORMAT = "%.6f"

N_GENERATED_BLADES = 1
LHS_SEED = 62955
LHS_OPTIMIZATION = "random-cd"
WRITE_GENERATED_CONTROL_POINTS = True
RESET_OUTPUT_FILES = RESET_OUTPUT_FILES_DEFAULT

# Keep in sync with gp_infill_from_training_data.py (expanded after infill5 edge-saturation).
pitch_bounds = np.array([
    [0.60, 1.25],  # p1y
    [0.00, 1.00],  # y4 factor
    [0.35, 0.75],  # p4x
    [0.05, 0.50],  # d1
    [0.05, 0.50],  # d2
    [0.40, 0.85],  # p7y
])

chord_bounds = np.array([
    [0.15, 0.30],  # p1y
    [0.45, 0.85],  # p4x
    [0.05, 0.50],  # d1
    [0.25, 0.70],  # y4
    [0.05, 0.30],  # w56
])

from x_blade_new import R_VALUES as _R_VALUES
# Plot range starts at the first design station, so the curves never show
# the clamped extrapolation below it as if it were design.
R_FINE = np.linspace(float(_R_VALUES[0]), 0.998, 100)

# What changes comes from the blade parameter file's '# vary <name> Y|N [low high]' lines (blade_vary.read_vary),
# the file the XGeom tools read. pitch and chord Y or N give para_control (both 7, chord only 2, pitch only 3;
# neither 0: no sampling, only the original blade's CAD, case ORIGINAL_CASE_ID in ORIGINAL_GEOMETRY_DIR). A range
# on their line ('# vary pitch Y 0.6 1.3') bounds their values: pitch p1y and p7y and the cap of P4's pitch
# (max_pitch, else 1.4); chord p1y and y4. The positions, fractions and weights keep the bounds above. Only the
# flagged parameters' variables are sampled and written (pitch's 6, then chord's 5). The blade is x_blade_new's
# (the MSc polynomials, which msc_blade_params.dat tabulates); other parameters flagged Y are not varied here. A
# file with no vary line: both, with the bounds above, as before.
BLADE_PARAMS = ROOT / "msc_blade_params.dat"
ORIGINAL_CASE_ID = 0
ORIGINAL_GEOMETRY_DIR = ROOT / "geometry" / "original"
PITCH_VALUE_ROWS = (0, 5)        # pitch_bounds rows that are P/D values: p1y, p7y
CHORD_VALUE_ROWS = (0, 3)        # chord_bounds rows that are c/D values: p1y, y4


def sampling_setup(params_path=BLADE_PARAMS):
    """What the parameter file's vary lines ask for: a dict with para_control, the pitch and chord bounds (their
    value rows set by the file's ranges), max_pitch, which parameters are sampled, and notes to print."""
    p_bounds, c_bounds = pitch_bounds.copy(), chord_bounds.copy()
    out = {"pitch_bounds": p_bounds, "chord_bounds": c_bounds, "max_pitch": bezier_constraints.MAX_PITCH}
    vary = {}
    if Path(params_path).is_file():
        vary = read_vary(params_path)
    if not vary:
        out.update(para_control=7, pitch=True, chord=True,
                   notes=[f"{Path(params_path).name}: " + ("no vary line" if Path(params_path).is_file() else
                                                          "not found") + "; pitch and chord both vary (para_control 7)"])
        return out
    pitch, chord = vary["pitch"], vary["chord"]
    out.update(para_control=bezier_constraints.para_control_flag(pitch.on, chord.on), pitch=pitch.on,
               chord=chord.on, notes=[])
    if pitch.on and pitch.lo is not None:
        p_bounds[list(PITCH_VALUE_ROWS)] = (pitch.lo, pitch.hi)
        out["max_pitch"] = pitch.hi
    if chord.on and chord.lo is not None:
        c_bounds[list(CHORD_VALUE_ROWS)] = (chord.lo, chord.hi)
    on = [n for n in ("pitch", "chord") if vary[n].on]
    out["notes"].append(f"{Path(params_path).name}: " + (" and ".join(on) + (" vary" if len(on) > 1 else " varies")
                                                          if on else "nothing varies")
                        + f" (para_control {out['para_control']})"
                        + ("" if on else ": only the original blade's CAD"))
    others = [n for n, v in vary.items() if v.on and n not in ("pitch", "chord")]
    if others:
        out["notes"].append(f"flagged Y but not varied here (pitch and chord only): {', '.join(others)}")
    return out


def load_or_generate_control_points(bounds, d_total):
    if INPUT_MODE == "existing":
        X_data = load_table(EXISTING_CONTROL_POINTS_PATH)
        case_ids, X_phys = split_case_ids(X_data, d_total)
        print(f"Loaded {X_phys.shape[0]} designs from {EXISTING_CONTROL_POINTS_PATH.name}")
        if X_data.shape[1] == d_total + 1:
            print("Using leading case_id column from the input control-point file")
        else:
            print("No case_id column found; falling back to row index for output naming")
        return case_ids, X_phys

    if INPUT_MODE == "infill":
        X_data = load_table(INFILL_CONTROL_POINTS_PATH)
        case_ids, X_phys = split_case_ids(X_data, d_total)
        case_ids = case_ids.astype(int) + CASE_ID_OFFSET
        print(
            f"Loaded {X_phys.shape[0]} infill designs from "
            f"{INFILL_CONTROL_POINTS_PATH.name}; case ids {int(case_ids.min())}.."
            f"{int(case_ids.max())}"
        )
        return case_ids, X_phys

    if INPUT_MODE in ("lhs", "test"):
        sampler = qmc.LatinHypercube(
            d=d_total,
            seed=LHS_SEED,
            optimization=LHS_OPTIMIZATION,
        )
        unit_samples = sampler.random(N_GENERATED_BLADES)
        X_phys = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])

        if WRITE_GENERATED_CONTROL_POINTS:
            save_table(GENERATED_CONTROL_POINTS_PATH, X_phys, fmt=CONTROL_POINT_FORMAT)
            print(f"Wrote generated control points to {GENERATED_CONTROL_POINTS_PATH.name}")

        case_ids = CASE_ID_OFFSET + np.arange(X_phys.shape[0], dtype=int)
        mode_label = "test LHS" if INPUT_MODE == "test" else "optimized Latin hypercube"
        print(
            f"Generated {X_phys.shape[0]} designs using {mode_label} "
            f"({LHS_OPTIMIZATION}, seed={LHS_SEED}); "
            f"case ids {int(case_ids.min())}..{int(case_ids.max())}"
        )
        return case_ids, X_phys

    raise ValueError("INPUT_MODE must be 'existing', 'lhs', 'infill', or 'test'")


def initialize_output_files():
    if not RESET_OUTPUT_FILES:
        return
    CONTROL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REJECTED_CONTROL_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clear_file(CONTROL_OUTPUT_PATH)
    clear_file(REJECTED_CONTROL_POINTS_PATH)


def generate_cad(pitch_con, chord_con, case_id, geometry_dir, para_control=7, max_pitch=1.4):
    """Generate one DRDC blade CAD, guarded by a subprocess timeout if configured.
    para_control and max_pitch go to X_blade (0: the original blade).

    Returns True on success, False on timeout/failure so the caller can reject
    the case and continue with the rest of the batch.
    """
    pitch_con = np.asarray(pitch_con, dtype=float).ravel()
    chord_con = np.asarray(chord_con, dtype=float).ravel()
    if not CAD_TIMEOUT_S:
        X_CAD_from_design(pitch_con, chord_con, case_id, output_dir=geometry_dir, hub=True, hub_height=None, n_blades=5,
                          para_control=para_control, max_pitch=max_pitch)
        return True

    tmp_design = Path(tempfile.gettempdir()) / f"cad_design_{case_id}.npy"
    np.save(tmp_design, np.concatenate([pitch_con, chord_con]))
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "cad_worker.py"),
             str(int(case_id)), str(geometry_dir), str(tmp_design), str(int(para_control)), repr(float(max_pitch))],
            check=True,
            timeout=CAD_TIMEOUT_S,
        )
        return True
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] X_CAD exceeded {CAD_TIMEOUT_S}s for case {case_id}; skipping")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] X_CAD worker failed for case {case_id}: rc={exc.returncode}")
        return False
    finally:
        try:
            tmp_design.unlink()
        except FileNotFoundError:
            pass


def append_control_points_row(path, values, case_id=None):
    append_design_row(
        path,
        values,
        case_id=case_id,
        fmt=CONTROL_POINT_FORMAT,
        write_case_id=WRITE_CASE_ID_TO_OUTPUT_FILES,
    )


def build_original():
    """Nothing flagged Y: the original blade's CAD only (para_control 0)."""
    ORIGINAL_GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    para.GEOMETRY_OUTPUT_DIR = ORIGINAL_GEOMETRY_DIR
    print(f"The original blade: case {ORIGINAL_CASE_ID} in {ORIGINAL_GEOMETRY_DIR}")
    ok = generate_cad(np.zeros(pitch_bounds.shape[0]), np.zeros(chord_bounds.shape[0]), ORIGINAL_CASE_ID,
                      ORIGINAL_GEOMETRY_DIR, para_control=0)
    print("CAD written" if ok else "CAD failed")


def main():
    setup = sampling_setup(BLADE_PARAMS)
    for note in setup["notes"]:
        print(note)
    para_control, max_pitch = setup["para_control"], setup["max_pitch"]
    if para_control == 0:
        build_original()
        return
    d_pitch = pitch_bounds.shape[0]
    d_chord = chord_bounds.shape[0]
    blocks = ([setup["pitch_bounds"]] if setup["pitch"] else []) + ([setup["chord_bounds"]] if setup["chord"] else [])
    bounds = np.vstack(blocks)                      # the sampled variables: pitch's, then chord's
    d_total = bounds.shape[0]

    initialize_output_files()

    geometry_dir = GEOMETRY_DIR
    geometry_dir.mkdir(parents=True, exist_ok=True)
    para.GEOMETRY_OUTPUT_DIR = geometry_dir
    print(f"Geometry output directory: {geometry_dir}")

    bezier_constraints.APPLY_COUPLED_CONSTRAINTS = USE_BEZIER_VIOLATION_FILTER
    if USE_BEZIER_VIOLATION_FILTER:
        bezier_constraints.CHORD_SUM_MAX_NORM = BEZIER_CHORD_SUM_MAX_NORM
        bezier_constraints.PITCH_DIFF_MAX_NORM = BEZIER_PITCH_DIFF_MAX_NORM
        print(
            "Using Bezier coupled-constraint thresholds: "
            f"b1={bezier_constraints.CHORD_SUM_MAX_NORM:.2f}, "
            f"b2={bezier_constraints.PITCH_DIFF_MAX_NORM:.2f}"
        )
        print(f"Bezier coupled-constraint filter enabled ({BEZIER_CONSTRAINT_MODE})")
    else:
        print(
            "Coupled Bezier limits disabled; "
            "variable bounds and blade-clearance rejection still apply"
        )

    case_ids, X_phys = load_or_generate_control_points(bounds, d_total)
    # Resumable: cases already in the accepted file are skipped. A CAD file alone is
    # not enough (it may be incomplete if a run was interrupted), so such cases are redone.
    done_ids = set()
    if CONTROL_OUTPUT_PATH.exists() and CONTROL_OUTPUT_PATH.stat().st_size > 0:
        done_ids = {int(c) for c in split_case_ids(load_table(CONTROL_OUTPUT_PATH), d_total)[0]}
    n_samples = X_phys.shape[0]
    d_p = d_pitch if setup["pitch"] else 0               # an unflagged parameter: zeros, not used (para_control)
    pitch_phys = X_phys[:, :d_pitch] if setup["pitch"] else np.zeros((n_samples, d_pitch))
    chord_phys = X_phys[:, d_p:d_p + d_chord] if setup["chord"] else np.zeros((n_samples, d_chord))

    colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
    fig, (ax_pitch, ax_chord) = plt.subplots(1, 2, figsize=(12, 5))
    ax_pitch.set_title("Pitch (all samples)")
    ax_chord.set_title("Chord (all samples)")
    ax_pitch.set_xlabel("radius (R)")
    ax_pitch.set_ylabel("pitch")
    ax_chord.set_xlabel("radius (R)")
    ax_chord.set_ylabel("chord")
    ax_pitch.grid(True)
    ax_chord.grid(True)

    accepted_count = 0
    rejected_count = 0

    for i in range(n_samples):
        case_id = int(case_ids[i])
        pitch_con = pitch_phys[i, :]
        chord_con = chord_phys[i, :]
        blade_count = case_id
        raw_input_values = np.asarray(X_phys[i, :], dtype=float).ravel()       # the sampled variables

        if case_id in done_ids:
            print(f"Case {case_id} already has CAD; skipping")
            accepted_count += 1
            continue

        (
            points,
            min_dis,
            constraint_violation,
            _chord_con_points,
            _pitch_con_points,
            Pitch,
            ChordLength,
            bezier_info,
        ) = X_blade(
            pitch_con,
            chord_con,
            blade_count,
            return_bezier_info=True,
            bezier_constraint_mode=BEZIER_CONSTRAINT_MODE,
            apply_coupled_constraints=USE_BEZIER_VIOLATION_FILTER,
            para_control=para_control,
            max_pitch=max_pitch,
        )

        if USE_BEZIER_VIOLATION_FILTER and bezier_info["rejected"]:
            reasons = ", ".join(bezier_info["reasons"]) or "unknown coupled-constraint violation"
            print(f"Bezier rejection for case {case_id}: {reasons}")
            append_control_points_row(REJECTED_CONTROL_POINTS_PATH, raw_input_values, case_id=case_id)
            rejected_count += 1
            continue

        if constraint_violation == 1:
            print(f"Blade clearance violation for case {case_id}")
            print(min_dis)
            append_control_points_row(REJECTED_CONTROL_POINTS_PATH, raw_input_values, case_id=case_id)
            rejected_count += 1
            continue

        if USE_BEZIER_VIOLATION_FILTER and bezier_info["violated"]:
            reasons = ", ".join(bezier_info["reasons"])
            print(f"Projected case {case_id} to satisfy Bezier constraints: {reasons}")

        print(f"Case {case_id} accepted for CAD and data export")

        try:
            cad_ok = generate_cad(pitch_con, chord_con, blade_count, geometry_dir, para_control, max_pitch)
        except Exception as e:
            print(f"[WARN] X_CAD failed for case {case_id}: {e}")
            cad_ok = False
        if not cad_ok:
            append_control_points_row(REJECTED_CONTROL_POINTS_PATH, raw_input_values, case_id=case_id)
            rejected_count += 1
            continue

        # Save the raw design variables, optionally prefixed by case_id.
        append_control_points_row(CONTROL_OUTPUT_PATH, raw_input_values, case_id=case_id)

        c = colors[i % len(colors)]
        ax_pitch.plot(R_FINE, Pitch(R_FINE), color=c, linewidth=1.3, label=f"c{case_id}")
        ax_chord.plot(R_FINE, ChordLength(R_FINE), color=c, linewidth=1.3, label=f"c{case_id}")
        accepted_count += 1

    handles1, labels1 = ax_pitch.get_legend_handles_labels()
    if len(labels1) <= 15:
        ax_pitch.legend(ncol=2, fontsize=8)

    handles2, labels2 = ax_chord.get_legend_handles_labels()
    if len(labels2) <= 15:
        ax_chord.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Accepted {accepted_count} of {n_samples} samples")
    print(f"Rejected {rejected_count} of {n_samples} samples")


if __name__ == "__main__":
    main()


      