from pso import *
from retruncate import find_te_fillet, perp_method, make_fillet


def match_rudder_section(rud_x_rnd, rud_y_rnd, rud_x_shrp, rud_y_shrp, rud_shrp_chord, pso_params, foil_settings, bounds):

  # normalize sharpened rudder
  rud_x_shrp_norm = rud_x_shrp / rud_shrp_chord
  rud_y_shrp_norm = rud_y_shrp / rud_shrp_chord

  # solve for normalized control points
  BestPt, BestVal, pos_history, velo_history = PSO_ctrl_pts(rud_x_shrp_norm, rud_y_shrp_norm, pso_params, foil_settings, bounds)
  best_xi = BestPt
  # create replica airfoil from control points
  repl_x_shrp_norm, repl_y_shrp_norm = build_half_airfoil(best_xi, foil_settings.n_bez_samples)

  # scale back to same chord length as sharpened rudder
  repl_x_shrp = repl_x_shrp_norm * rud_shrp_chord
  repl_y_shrp = repl_y_shrp_norm * rud_shrp_chord

  # make array of all points
  repl_y_shrp_low = -repl_y_shrp
  repl_shrp_upper = np.flip(np.column_stack((repl_x_shrp, repl_y_shrp)), axis=0)
  repl_shrp_lower = np.column_stack((repl_x_shrp, repl_y_shrp_low))
  repl_shrp_coords = np.vstack((repl_shrp_upper, repl_shrp_lower))
  repl_ctrl_pts = expand_ctrl_pts(best_xi, rud_shrp_chord) 
  rud_rnd_upper = np.flip(np.column_stack((rud_x_rnd, rud_y_rnd)), axis=0)
  rud_rnd_lower = np.column_stack((rud_x_rnd, -rud_y_rnd))
  rud_rnd_coords = np.vstack((rud_rnd_upper, rud_rnd_lower))

  # find original fillet starting location in x
  fillet = find_te_fillet(rud_rnd_coords)

  # cut off sharpened replica at that x, draw perpendicular from here, distance of perp to x-axis is radius, intersect is center
  center, r, slope_up, keep_coords = perp_method(fillet, repl_shrp_coords)

  # generate points along fillet
  fillet_points = make_fillet(center, r, slope_up)

  # combine cutoff-sharpened replica with its fillet to get final coords
  half = len(fillet_points) // 2
  te_pt = center + np.array([r, 0.0])
  fillet_upper = fillet_points[half:-1]   # TE side -> upper cut pt
  fillet_lower = fillet_points[1:half]    # lower cut pt -> TE side
  repl_coords = np.vstack((te_pt, fillet_upper, keep_coords, fillet_lower, te_pt))


  return BestPt, BestVal, r, repl_coords, repl_shrp_coords, repl_ctrl_pts