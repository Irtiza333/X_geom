import numpy as np
from dataclasses import dataclass

# temporary functions until its implemented in Evans code
def get_info_temp(REF_PATH, slope_tol):

  rud_x_rnd, rud_y_rnd, chord_ref_original = get_reference_half_raw(REF_PATH)
  rud_x_shrp, rud_y_shrp, rud_shrp_chord   = untruncated_profile(rud_x_rnd, rud_y_rnd, slope_tol)

  return rud_x_rnd, rud_y_rnd, rud_x_shrp, rud_y_shrp, rud_shrp_chord  

@dataclass
class HalfAirfoilConfig:
    chord: float
    n_bez_samples: int      # samples cross_section_points draws along the Bezier
    xq: np.ndarray          # target x-stations
    yq: np.ndarray          # target y at those stations
    ConsOn: int = 0

def load_dat_coords(ref_path):
    with open(ref_path, 'r') as f:
        lines = f.readlines()
    coords = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                coords.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    return np.array(coords)


def get_reference_half_raw(ref_path):
    """Load a Selig-style .dat file (TE -> upper surface -> LE -> lower surface
    -> TE) and pull out one surface as an ascending-x curve, translated so its
    leading edge sits at (0, 0) and its trailing edge lands at x = chord.

    Returns (x, y, chord).
    """
    coords = load_dat_coords(ref_path)
    x = coords[:, 0]
    le = int(np.argmin(x))
    le_x, le_y = coords[le, 0], coords[le, 1]

    branch1 = coords[:le + 1]
    branch2 = coords[le:]

    def to_ascending(b):
        return b[::-1] if b[0, 0] > b[-1, 0] else b

    branch1 = to_ascending(branch1)
    branch2 = to_ascending(branch2)

    chosen = branch1 if np.mean(branch1[:, 1]) >= np.mean(branch2[:, 1]) else branch2

    xu, iu = np.unique(chosen[:, 0], return_index=True)
    yu = chosen[iu, 1]
    chord = xu[-1] - xu[0]

    # Translate so the leading edge (originally at (le_x, le_y)) lands exactly
    # at the origin; x stays ascending LE -> TE.
    xu = xu - le_x
    yu = yu - le_y
    return xu, yu, chord

def segment_properties(P1, P2):
    dx = P2[0] - P1[0]
    dy = P2[1] - P1[1]
    slope = dy/dx
    return(slope)

def untruncated_profile(X, Y, slope_tol):
    i = len(X) - 1
    while i > 0:
        # Comparing the unit vector at each
        s_1 = segment_properties([X[i-1], Y[i-1]], [X[i], Y[i]])
        s_2 = segment_properties([X[i-2], Y[i-2]], [X[i-1], Y[i-1]])
        if abs(s_1 - s_2) < slope_tol:
            print(X[i], Y[i], s_1)
            untrunc_x = (-Y[i]) /  s_1 + X[i]
            c = untrunc_x - X[0]
            break
        i = i - 1

    j = 0
    X_new = [0] * (i + 1)
    print(len(X_new))
    Y_new = [0] * (i + 1)
    while j < i:
        X_new[j] = X[j]
        Y_new[j] = Y[j]
        j = j + 1
    X_new[len(X_new) - 1] = untrunc_x

    return(X_new, Y_new, c)




# OLD AND UNUSED

# def build_target_stations(x_t, y_t, n_stations):
#     """Cosine-spaced x-stations (clustered near both ends) over the target's
#     x-range, with the target y interpolated onto them."""
#     beta = np.linspace(0.0, np.pi, n_stations)
#     xq = x_t[0] + 0.5 * (1.0 - np.cos(beta)) * (x_t[-1] - x_t[0])
#     yq = np.interp(xq, x_t, y_t)
#     return xq, yq




# def make_config_from_target(x_t, y_t, chord, n_bez_samples,
#                              n_target_stations, ConsOn=0):
#     """Build a HalfAirfoilConfig directly from an already-in-hand (x, y, chord)
#     target curve -- used by make_config, and directly by anything that needs to
#     match against a synthesized/modified target (e.g. test_radius_recovery.py)
#     without round-tripping through a .dat file."""
#     xq, yq = build_target_stations(x_t, y_t, n_target_stations)

#     return HalfAirfoilConfig(
#         chord=chord,
#         n_bez_samples=n_bez_samples,
#         xq=xq,
#         yq=yq,
#         ConsOn=ConsOn,
#     )


# def _to_cross_section_params(xi, chord):
#     """Map the PSO position vector [Y1, X2, X3, Y3, r] (fractions of chord) onto cross_section_points'
#     arguments (x1, y1, x2, y2, r_frac), all as fractions of the chord.
#     """
#     Y1, X2, X3, Y3 = xi
#     y1 = Y1 / chord
#     x2 = X2 / chord
#     x3 = X3 / chord
#     y3 = Y3 / chord
#     return y1, x2, x3, y3
