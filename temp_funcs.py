"""
temp_funcs.py

Temporary functions until this is implemented in Evan's code: read a Selig
.dat section, take its upper surface, and extend the straight part of its
flank to a sharp trailing edge.
"""

import numpy as np


def get_info_temp(REF_PATH, slope_tol):
    """The section's upper surface, rounded (as read) and sharpened, plus the
    sharpened chord."""
    rud_x_rnd, rud_y_rnd, _ = get_reference_half_raw(REF_PATH)
    rud_x_shrp, rud_y_shrp, rud_shrp_chord = untruncated_profile(rud_x_rnd, rud_y_rnd, slope_tol)
    return rud_x_rnd, rud_y_rnd, rud_x_shrp, rud_y_shrp, rud_shrp_chord


def load_dat_coords(ref_path):
    """(x, y) rows of a .dat file; skips the title line and anything that is
    not two numbers."""
    coords = []
    with open(ref_path) as f:
        next(f, None)
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return np.array(coords)


def get_reference_half_raw(ref_path):
    """Load a Selig-style .dat file (TE -> upper surface -> LE -> lower surface
    -> TE) and pull out one surface as an ascending-x curve, translated so its
    leading edge sits at (0, 0) and its trailing edge lands at x = chord.

    Returns (x, y, chord).
    """
    return reference_half(load_dat_coords(ref_path))


def reference_half(coords):
    """The same as get_reference_half_raw for Selig-ordered points already in
    memory, (n, 2)."""
    coords = np.asarray(coords, dtype=float)
    le = int(np.argmin(coords[:, 0]))
    le_x, le_y = coords[le]

    def to_ascending(b):
        return b[::-1] if b[0, 0] > b[-1, 0] else b

    branch1 = to_ascending(coords[:le + 1])
    branch2 = to_ascending(coords[le:])
    chosen = branch1 if branch1[:, 1].mean() >= branch2[:, 1].mean() else branch2

    xu, iu = np.unique(chosen[:, 0], return_index=True)
    yu = chosen[iu, 1]
    return xu - le_x, yu - le_y, xu[-1] - xu[0]


def untruncated_profile(X, Y, slope_tol):
    """Sharpen a rounded trailing edge.

    X, Y: one surface, LE to TE (ascending x). Walking upstream from the TE,
    the first point i where the segments either side of point i-1 have slopes
    within slope_tol of each other is taken to be on the straight flank. The
    line through points i-1 and i is extended to y = 0, and that sharp TE
    replaces point i and everything downstream of it.

    Returns (X_new, Y_new, chord): the points before i plus the sharp TE, and
    the chord from X[0] to the sharp TE.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    s = np.diff(Y) / np.diff(X)                 # s[k]: slope from point k to k + 1
    straight = np.nonzero(np.abs(s[1:] - s[:-1]) < slope_tol)[0]
    if straight.size == 0:
        raise ValueError(f"no straight flank found near the TE (slope_tol = {slope_tol})")
    i = straight[-1] + 2
    x_te = X[i] - Y[i] / s[i - 1]
    return np.append(X[:i], x_te), np.append(Y[:i], 0.0), x_te - X[0]
