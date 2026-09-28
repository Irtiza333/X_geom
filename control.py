import numpy as np
import matplotlib.pyplot as plt
from match_rudder_section import *

from temp_funcs import *

# ---------------- TARGET DATA (open for a different sample-points file later) --
REF_PATH = 'section_cut_y80_selig.dat'

@dataclass # LEAVE THESE AS DEFAULTS
class FoilSettings:
    slope_tol: float = 0.001
    n_bez_samples: int = 200   # number of points to calculate along the Bezier curve
    n_target_stations: int = 2000   # number of points to interpolate w/ cosine spacing along the original data (building smoother foil)
    fillet_method: int = 0 # 0 to find center and radius from ⟂ slope at TE (perfectly tangent, radius slightly off),
                          # 1 to fit a circle to points laying along the trailing edge (TE point(s) near TE match exactly, not tangent)

foil_settings = FoilSettings( # SET PARAMETERS HERE
    slope_tol = 0.001,
    n_bez_samples=200,   # number of points to calculate along the Bezier curve
    n_target_stations = 2000,   # number of points to interpolate w/ cosine spacing along the original data (building smoother foil)
    fillet_method = 0, # 0 to find center and radius from ⟂ slope at TE (perfectly tangent, radius slightly off),
                      # 1 to fit a circle to points laying along the trailing edge (TE point(s) near TE match exactly, not tangent) 
)

# ---------------- PSO SETTINGS -------------------------------------------------
@dataclass # LEAVE THESE AS DEFAULTS
class PSOparams:
    nPop: int = 500
    pers_fac: float = 0.25
    soc_fac: float = 0.25
    t_max: int = 200
    VmaxPerc: float = 15
    BoundHandling: int = 1   # 0 fly over, 1 repair, 2 bounce
    ConsHandling: int = 0   # 0 fly over, 1 penalty, 2 repair
    ConsOn: int = 0             # 1 to enforce X2 <= X3, 0 for none
    Obj: int = 0                # 0 to use RMS, 1 to use min max Root-Square error


pso_params = PSOparams( # SET PARAMETERS HERE
    nPop = 100,
    pers_fac = 0.25,
    soc_fac = 0.25,
    t_max = 200,
    VmaxPerc = 15,
    BoundHandling = 1 ,    # 0 fly over, 1 repair, 2 bounce
    ConsHandling = 0 ,     # 0 fly over, 1 penalty, 2 repair
    ConsOn = 0 ,            # 1 to enforce X2 <= X3, 0 for none
    Obj = 0  ,              # 0 to use RMS, 1 to use min max Root-Square error
)



# ---------------- FREE-VARIABLE BOUNDS -----------------------------------------
# [low, high] for each of [Y1, X2, X3, Y3, r]  (fractions of chord)
bounds = np.array([
    [0.00, 0.15],   # Y1
    [0.05, 0.50],   # X2
    [0.30, 0.95],   # X3
    [0.00, 0.15],   # Y3
])

# this part is temporary
rud_x_rnd, rud_y_rnd, rud_x_shrp, rud_y_shrp, rud_shrp_chord = get_info_temp(REF_PATH, foil_settings.slope_tol)



BestPt, BestVal, r, repl_coords, repl_shrp_coords, repl_ctrl_pts = match_rudder_section(rud_x_rnd, rud_y_rnd, rud_x_shrp, rud_y_shrp, rud_shrp_chord, pso_params, foil_settings, bounds)

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
