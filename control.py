import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from match_rudder_section import match_rudder_section, mirror_selig
from retruncate import find_te_fillet
from temp_funcs import get_info_temp

# ---------------- TARGET DATA (open for a different sample-points file later) --
REF_PATH = 'section_cut_y80_selig.dat'

@dataclass # LEAVE THESE AS DEFAULTS
class FoilSettings:
    slope_tol: float = 0.001
    n_bez_samples: int = 200     # number of points to calculate along the Bezier curve
    n_target_stations: int = 0   # 0 to fit at the section's own points, N to resample it at N cosine-spaced stations
    fillet_method: int = 0       # 0 for a circle tangent to the replica where the original's fillet starts (normals meet on the chord line),
                                 # 1 for a circle through those cut points and the original TE point (TE matches exactly, not tangent),
                                 # 2 for a circle of the original's radius, tangent to the replica (the cut moves to where it fits)

foil_settings = FoilSettings( # SET PARAMETERS HERE
    slope_tol = 0.001,
    n_bez_samples = 200,
    n_target_stations = 0,
    fillet_method = 0,
)

# ---------------- PSO SETTINGS (Main_PSO) --------------------------------------
@dataclass # LEAVE THESE AS DEFAULTS
class PSOparams:
    swarm_size: int = 50
    num_iter: int = 100
    C1: float = 2.05               # Pbest coefficient
    C2: float = 2.05               # Gbest coefficient
    PSO_variant: int = 1           # 1 constriction factor, 2 inertia (w_max down to w_min)
    w_max: float = 0.9
    w_min: float = 0.5
    vmax_frac: float = 0.2         # Vmax as a fraction of each variable's range
    constraint_handling: int = 2   # 0 none, 1 repair, 2 penalty, 3 clip to the bounds
    penalty_factor: float = 5.0
    craziness: int = 0
    stall_iter: int = 15           # stop after this many iterations without improvement, 0 never
    seed: int = 42                 # same seed, same answer
    ConsOn: int = 0                # 1 to enforce X2 <= X3, 0 for none
    Obj: int = 0                   # 0 to use RMS, 1 to use the largest deviation
    polish: bool = True            # finish with a bounded least-squares step (Obj 0 only)
    verbose: bool = True           # print what the PSO and the polish reached

pso_params = PSOparams( # SET PARAMETERS HERE
    swarm_size = 50,
    num_iter = 100,
    PSO_variant = 1,
    constraint_handling = 3,       # the best fit sits on a bound (X3), which clipping keeps exactly
    stall_iter = 15,
    seed = 42,
    ConsOn = 0,
    Obj = 0,
    polish = True,
)

# ---------------- FREE-VARIABLE BOUNDS -----------------------------------------
# [low, high] for each of [Y1, X2, X3, Y3]  (fractions of the sharp chord)
bounds = np.array([
    [0.00, 0.15],   # Y1
    [0.05, 0.50],   # X2
    [0.30, 0.95],   # X3
    [0.00, 0.15],   # Y3
])

# this part is temporary
rud_x_rnd, rud_y_rnd, rud_x_shrp, rud_y_shrp, rud_shrp_chord = get_info_temp(REF_PATH, foil_settings.slope_tol)

BestPt, BestVal, r, repl_coords, repl_shrp_coords, repl_ctrl_pts = match_rudder_section(rud_x_rnd, rud_y_rnd, rud_x_shrp, rud_y_shrp, rud_shrp_chord, pso_params, foil_settings, bounds)

te = find_te_fillet(mirror_selig(rud_x_rnd, rud_y_rnd))
print(f"Y1, X2, X3, Y3 = {np.array2string(BestPt, precision=5)}   (sharp chord {rud_shrp_chord:.5f})")
print(f"fit error {BestVal:.4e} of the sharp chord")
print(f"TE radius {r:.5f}, original {te['R']:.5f}; fillet from x = {te['x_start']:.5f}")

# plotting results
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_aspect('equal')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.75)

# Original airfoil surface, as lines through the actual points.
ax.plot(rud_x_rnd, rud_y_rnd, 'k.-', ms=4, lw=1, label="original")
ax.plot(rud_x_shrp, rud_y_shrp, 'r.-', ms=4, lw=1, label="sharpened")
ax.plot(repl_shrp_coords[:,0], repl_shrp_coords[:,1], 'b.-', ms=4, lw=1, label="matched sharpened")
ax.plot(repl_ctrl_pts[:,0], repl_ctrl_pts[:,1], 'go--')
ax.plot(repl_coords[:,0], repl_coords[:,1], label = "fillet for matched")
ax.legend(loc = 'lower center')
ax.plot()
plt.show()
