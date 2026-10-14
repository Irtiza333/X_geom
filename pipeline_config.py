"""
Shared paths for the Prop_hull surrogate pipeline.

Set DATA_DIR for this campaign and INFILL_ROUND for each iteration.
Round 0 creates initial proposals; rounds 1+ use earlier training tables.
All scripts import paths from here instead of hard-coding filenames.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Training_3d"

# --- change these between iterations ---
INFILL_ROUND = 0
INITIAL_TAG = "initial_sampling_new"
TRAINING_EXTENSION = ".txt"  # Existing .dat training tables are also accepted.

# Current tables store design variables; list legacy con-point filenames here only if needed.
CONPOINT_TRAINING_NAMES = frozenset()


def case_id(round_num, row):
    """Case number for CAD, mesh, CFX and Nibi files: row in round 0, 100*round + row after."""
    if not 0 <= int(row) < 100:
        raise ValueError("At most 100 designs per round")
    return 100 * int(round_num) + int(row)


def training_path(tag, data_dir=None):
    """Choose the existing text table, rejecting ambiguous duplicate extensions."""
    folder = Path(DATA_DIR if data_dir is None else data_dir)
    candidates = [folder / f"training_data_{tag}{ext}" for ext in (".txt", ".dat")]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(f"Both .txt and .dat training tables exist for {tag}; keep one authoritative table")
    return existing[0] if existing else folder / f"training_data_{tag}{TRAINING_EXTENSION}"


def training_data_files(before_round=None, data_dir=None):
    """Training tables in numeric order; optionally exclude the target and later rounds."""
    import re
    folder = Path(DATA_DIR if data_dir is None else data_dir)
    rounds = set()
    for path in folder.glob("training_data_*.*"):
        match = re.fullmatch(r"training_data_(initial|infill([1-9]\d*))\.(txt|dat)", path.name)
        if match:
            rounds.add(0 if match[1] == "initial" else int(match[2]))
    return [training_path("initial" if r == 0 else f"infill{r}", folder)
            for r in sorted(rounds) if before_round is None or r < before_round]


def initial_sampling_paths():
    """Output paths for Initial_sampling.py."""
    geometry_dir = ROOT / "geometry" / INITIAL_TAG
    return {
        "existing_input": ROOT / f"rejected_{INITIAL_TAG}_control_points.txt",
        "generated": ROOT / f"generated_{INITIAL_TAG}_control_points.txt",
        "accepted": ROOT / f"input__var_values_{INITIAL_TAG}.dat",
        "rejected": ROOT / f"rejected_{INITIAL_TAG}_control_points.txt",
        "plot": ROOT / f"{INITIAL_TAG}_samples.png",
        "geometry_dir": geometry_dir,
    }


def cad_output_paths(case_id, geometry_dir=None):
    """Per-case geometry files (ORCA coordinates, IGES, optional STL)."""
    case_id = int(case_id)
    folder = Path(geometry_dir) if geometry_dir is not None else ROOT / "geometry" / INITIAL_TAG
    return {
        "dir": folder,
        "orca_dat": folder / f"ORCA{case_id}.dat",
        "iges": folder / f"sample_blade{case_id}.iges",
        "stl": folder / f"sample_blade{case_id}.stl",
    }


def infill_paths(round_num=None, data_dir=None):
    """Standard filenames for one round (suggest → CFD → training).

    Round 0 is the initial sampling round and is tagged 'initial' so it is not
    confused with the actual infill rounds (1, 2, ...) that follow.
    """
    r = INFILL_ROUND if round_num is None else int(round_num)
    if r < 0:
        raise ValueError("INFILL_ROUND must be nonnegative")
    folder = Path(DATA_DIR if data_dir is None else data_dir)
    tag = "initial" if r == 0 else f"infill{r}"
    return {
        "control_points": folder / f"{tag}_control_points.txt",
        "control_points_con": folder / f"{tag}_control_points_con_points.txt",
        "predictions": folder / f"{tag}_predictions.txt",
        "results": folder / f"Results_{tag}.txt",
        "poly_fit": folder / f"poly_fit_eq_{tag}.txt",
        "training_data": training_path(tag, folder),
        "plot_dir": folder / f"intersection_plots_{tag}",
        "geometry_dir": ROOT / "geometry" / tag,
        "cad_accepted": folder / f"{tag}_cad_accepted.txt",
        "cad_rejected": folder / f"{tag}_cad_rejected.txt",
        "cad_plot": folder / f"{tag}_cad_samples.png",
        # Control points to map results back to design vars for training.
        # Initial round: row index == case id (no case-id column).
        # Infill rounds: cad_accepted carries an explicit case-id column
        # (case ids are offset, e.g. 50..67), so use that mapping.
        "training_control_points": (
            folder / f"{tag}_control_points.txt"
            if r == 0 and not (folder / f"{tag}_cad_accepted.txt").exists()
            else folder / f"{tag}_cad_accepted.txt"
        ),
    }

