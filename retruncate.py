import numpy as np
import math

def perp_method(fillet, coords):

    n_coords = len(coords)
    # create search window for inteprolation
    n_quart = int(round(n_coords/4))
    # find where cutoff x lies on lower surface (lower for easier interpolation)
    cut_y = np.interp(fillet["x_start"], coords[n_coords-n_quart:,0], coords[n_coords-n_quart:,1])
    cut_pt = [fillet["x_start"], cut_y]
    keep_inds = np.where(coords[:,0]<cut_pt[0])[0]
    keep_coords = coords[keep_inds,:]
    keep_coords = np.vstack(([cut_pt[0], -cut_pt[1]], keep_coords,cut_pt ))

    slope_up = np.arctan2(keep_coords[0, 1] - keep_coords[1, 1], keep_coords[0, 0] - keep_coords[1, 0])
    # slope_lo = -slope_up
    radline_up = -1/math.tan(slope_up)
    radline_lo = -radline_up
    upper_point = keep_coords[0,:]   # point where fillet meets straight section
    lower_point = keep_coords[-1,:]   # point where fillet meets straight section
    center = find_center(radline_up, upper_point, radline_lo, lower_point)
    r = np.linalg.norm(center - upper_point)


    return center, r, slope_up, keep_coords

def find_center(radline1, p1, radline2, p2):
    if np.isclose(radline1, radline2):
        raise ValueError("Lines are parallel; no intersection.")
    A = np.array([[-radline1, 1], 
                   [-radline2, 1]])
    B = np.array([p1[1] - radline1 * p1[0],
                  p2[1] - radline2 * p2[0]])
    intersection = np.linalg.solve(A, B)
    return intersection

def make_fillet(center, r, slope_up, n_points=50):

  angle = 90 - abs(math.degrees(slope_up))
  # find intersection points on UNIT circle
  unit_int_x = math.cos(math.radians(angle))
  unit_int_y = math.sin(math.radians(angle))
  unit_int = np.array([unit_int_x, unit_int_y])
  # find location on actual fillet
  int_real = unit_int * r + center
  ang1 = -angle
  ang2 = angle
  fillet_points = []
  for i in range(n_points):
    # Calculate intermediate fraction (0.0 to 1.0)
    t = i / (n_points - 1)
    # Interpolate angle (degrees), then convert to radians for cos/sin
    theta = math.radians(ang1 + t * (ang2 - ang1))

    # Convert polar back to Cartesian coordinates
    x = center[0] + r * math.cos(theta)
    y = center[1] + r * math.sin(theta)
    fillet_points.append((x, y))
  fillet_points  = np.asarray(fillet_points)
  return fillet_points
  

def find_te_fillet(coords, tol=0.0001, max_iter=5):
    """Find the rounded-TE fillet radius/center, without having to guess a
    window size up front.

    Seeds a circle from the TE + the 2nd point out on each surface (an
    unbiased 3-point fit -- see `te_radius_3pt`), classifies which points sit
    on that circle (`_n_on_circle`), refits the circle using only those
    points, and repeats until the point count stops changing.
    """
    upper, lower = split_surfaces(coords)
    # find radius for a circle through te point and the adjacent points k=1
    seed_xc, seed_R = te_radius_3pt(coords, 1)
    xc, R = seed_xc, seed_R

    n_up = n_lo = None

    for _ in range(max_iter):
        # find how many more points on each surface sit within tol of the current circle
        n_up = _n_on_circle(upper, xc, R, tol)
        n_lo = _n_on_circle(lower, xc, R, tol)
        if n_up < 2 or n_lo < 2:
            raise ValueError(
                "fewer than 2 points landed on the fitted circle -- this "
                "section may not have a rounded TE, or `tol` is too tight"
            )
        # new circle fit using the furthest out points that are still on circle
        new_xc, new_R = fit_circle(np.vstack([upper[:n_up], lower[:n_lo]]))
        converged = np.allclose([new_xc, new_R], [xc, R])
        xc, R = new_xc,  new_R
        if converged:
            break
    x_start = upper[n_up,0]
    return {
        "xc": xc,  "R": R,
        "seed_xc": seed_xc,  "seed_R": seed_R,
        "n_upper": n_up, "n_lower": n_lo,
        "x_start":x_start
    }

def split_surfaces(coords):
    """Selig order is TE -> upper surface -> LE -> lower surface -> TE.
    Split at the LE (min x) and return both branches ordered TE-first."""
    x = coords[:, 0]
    le = int(np.argmin(x))
    upper = coords[:le + 1]        # already TE -> LE
    lower = coords[le:][::-1]      # LE -> TE, reverse to TE -> LE
    return upper, lower

def _n_on_circle(branch, xc, R, tol):
    """How many points, walking out from branch[0] (the TE), stay within `tol`
    (relative) of radius R from (xc, yc). Fillet points sit essentially exactly
    on the circle; once the surface peels away the deviation jumps by 1-2
    orders of magnitude, so this index is a sharp cutoff, not a fuzzy one."""
    dev = np.abs(np.hypot(branch[:, 0] - xc, branch[:, 1]) - R) / R
    off = np.nonzero(dev > tol)[0]
    return int(off[0]) if len(off) else len(branch)


def fit_circle(points):
    """Least-squares circle fit with the center constrained to y=0 -- the TE
    fillet is symmetric top/bottom about the chord line, so xc is the only
    free center coordinate. Returns (xc, 0.0, R)."""
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([2 * x, np.ones_like(x)])
    b = x**2 + y**2
    (xc, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    R = np.sqrt(c + xc**2)
    return xc, R


def te_radius_3pt(coords, k):
    """y=0-constrained circle fit through the TE point and the k-th point out
    on each surface (k=1 is the point adjacent to the TE)."""
    upper, lower = split_surfaces(coords)
    te = upper[0]
    return fit_circle(np.array([upper[k], te, lower[k]]))