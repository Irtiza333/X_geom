"""
match_rudder_section.py

Fit the Bezier half-section (replica_funcs) to a sharpened rudder section with
Main_PSO, then round its trailing edge like the original's (retruncate).
"""

import numpy as np
from scipy.optimize import brentq, least_squares

from Main_PSO import PSO_main
from replica_funcs import (build_half_airfoil, expand_ctrl_pts, half_airfoil_jac, half_airfoil_y,
                           is_valid, y_and_slope)
from retruncate import circle_method, find_te_fillet, make_fillet, perp_method, trim_at

BAD_FIT = 10.0      # fitness of a candidate whose curve folds back on itself


def fit_error(xi, x_target, y_target, obj=0):
    """Vertical gap between each candidate's half-section and the target at
    the target's stations, in sharp-chord units: RMS (obj 0) or largest
    (obj 1). xi is one candidate (4,) or a swarm (n, 4)."""
    xi = np.atleast_2d(xi)
    dev = half_airfoil_y(xi, x_target) - y_target
    err = np.sqrt(np.mean(dev**2, axis=1)) if obj == 0 else np.abs(dev).max(axis=1)
    return np.where(is_valid(xi), err, BAD_FIT)


def polish(xi, x_target, y_target, bounds):
    """Least-squares finish from xi inside the bounds (trust-region reflective,
    analytic Jacobian): the RMS fit error at its minimum in that basin."""
    lo, hi = np.asarray(bounds, dtype=float).T
    res = least_squares(lambda p: half_airfoil_y(p, x_target)[0] - y_target, np.clip(xi, lo, hi),
                        jac=lambda p: half_airfoil_jac(p, x_target), bounds=(lo, hi),
                        x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return res.x


def cut_for_radius(xi, radius, x_from, x_to=1.0 - 1e-12):
    """Where on the replica (sharp-chord units) the circle tangent to it has
    the given radius: y sqrt(1 + slope^2) = radius, searched between x_from
    and x_to. That radius falls to 0 at the sharp TE."""
    def gap(x):
        y, s = y_and_slope(xi, x)
        return y[0] * np.sqrt(1.0 + s[0]**2) - radius
    return brentq(gap, x_from, x_to, xtol=1e-15)


def target_stations(x, y, n_stations):
    """The target resampled at n_stations cosine-spaced stations (clustered at
    both ends). y is interpolated linearly in sqrt(x), which follows the round
    leading edge (y ~ sqrt(x) there) far better than linear in x."""
    beta = np.linspace(0.0, np.pi, n_stations)
    xq = x[0] + 0.5 * (1.0 - np.cos(beta)) * (x[-1] - x[0])
    return xq, np.interp(np.sqrt(xq - x[0]), np.sqrt(x - x[0]), y)


def mirror_selig(x, y):
    """Selig-ordered points (TE -> upper -> LE -> lower -> TE) of a symmetric
    section from its upper surface given LE to TE; the LE point appears once."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.vstack((np.column_stack((x, y))[::-1], np.column_stack((x[1:], -y[1:]))))


def match_rudder_section(rud_x_rnd, rud_y_rnd, rud_x_shrp, rud_y_shrp, rud_shrp_chord,
                         pso_params, foil_settings, bounds):
    """Fit the replica to the sharpened section and round its TE.

    Returns (BestPt, BestVal, r, repl_coords, repl_shrp_coords, repl_ctrl_pts):
    the free variables [Y1, X2, X3, Y3] and the fit error, both in sharp-chord
    units; the replica's TE radius; the rounded and the sharp replica (Selig
    order); and the control polygon. r and the coordinates are in the units of
    the input section.
    """
    c = rud_shrp_chord
    x_t = np.asarray(rud_x_shrp, dtype=float) / c
    y_t = np.asarray(rud_y_shrp, dtype=float) / c
    if foil_settings.n_target_stations:
        x_t, y_t = target_stations(x_t, y_t, foil_settings.n_target_stations)

    # fit [Y1, X2, X3, Y3] with Main_PSO, the whole swarm per objective call
    p = pso_params
    x2_le_x3 = (lambda X: X[:, 1] - X[:, 2]) if p.ConsOn else None
    res = PSO_main(lambda X: fit_error(X, x_t, y_t, p.Obj), len(bounds), bounds,
                   x2_le_x3, 0.0, "minimization",
                   swarm_size=p.swarm_size, num_iter=p.num_iter, C1=p.C1, C2=p.C2,
                   PSO_variant=p.PSO_variant, w_max=p.w_max, w_min=p.w_min,
                   vmax_frac=p.vmax_frac, constraint_handling=p.constraint_handling,
                   penalty_factor=p.penalty_factor, craziness=p.craziness,
                   stall_iter=p.stall_iter, seed=p.seed, verbose=p.verbose)
    BestPt, BestVal = res.best_pos, res.best_val
    if p.polish and p.Obj == 0:
        x = polish(BestPt, x_t, y_t, bounds)
        err = float(fit_error(x, x_t, y_t)[0])
        if err <= BestVal and (not p.ConsOn or x[1] <= x[2]):
            BestPt, BestVal = x, err
    if p.verbose:
        print(f"Main_PSO: fit error {res.best_val:.4e} after {res.n_iter} iterations"
              + (f", {BestVal:.4e} after the least-squares polish" if BestVal != res.best_val else ""))

    # replica, back at the section's scale
    bx, by = build_half_airfoil(BestPt, foil_settings.n_bez_samples)
    repl_shrp_coords = mirror_selig(bx * c, by * c)
    repl_ctrl_pts = expand_ctrl_pts(BestPt, c)

    # cut the replica where the original's TE circle starts and round it there
    fillet = find_te_fillet(mirror_selig(rud_x_rnd, rud_y_rnd))
    method = foil_settings.fillet_method
    x_cut = fillet["x_start"]
    if method == 2:         # move the cut to where the tangent circle has the original's radius
        x_cut = c * cut_for_radius(BestPt, fillet["R"] / c, x_cut / c - 0.05)
    y_cut, slope = y_and_slope(BestPt, x_cut / c)
    cut_pt = np.array([x_cut, y_cut[0] * c])
    if method in (0, 2):
        center, r = perp_method(cut_pt, slope[0])
    elif method == 1:
        center, r = circle_method(cut_pt, (rud_x_rnd[-1], rud_y_rnd[-1]))
    else:
        raise ValueError("fillet_method must be 0, 1 or 2")

    # TE point, upper fillet, replica upstream of the cut, lower fillet, TE point
    te_pt = center + np.array([r, 0.0])
    arc = make_fillet(center, cut_pt)
    repl_coords = np.vstack((te_pt, arc, trim_at(repl_shrp_coords, cut_pt),
                             arc[::-1] * [1.0, -1.0], te_pt))

    return BestPt, BestVal, r, repl_coords, repl_shrp_coords, repl_ctrl_pts
