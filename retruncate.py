"""
retruncate.py

Round the replica's sharp trailing edge like the original section's.

    find_te_fillet  the original's TE circle (centre on y = 0) and the x where
                    it meets the flank
    perp_method     fillet_method 0: circle tangent to the replica at the cut
    circle_method   fillet_method 1: circle through the two cut points and the
                    original TE point (TE matched exactly, joins not tangent)
    make_fillet     points along the upper half of the fillet arc
    trim_at         the replica upstream of the cut, closed by the cut points

Coordinates are Selig-ordered (TE -> upper -> LE -> lower -> TE) sections of a
symmetric foil with the chord line on y = 0.
"""

import numpy as np


def find_te_fillet(coords, tol=1e-4, max_iter=5):
    """The rounded-TE circle of a section and where it starts.

    Seeds a circle (centre on y = 0) through the TE point and the next point
    on each surface (`te_radius_3pt`), counts the points that stay on it
    walking out from the TE (`_n_on_circle`, relative tolerance tol), refits
    with only those (`fit_circle`), and repeats until the circle stops
    changing.

    The fillet starts where it is tangent to the flank. With the flank slope
    taken from the first two points off the circle, at angle phi to the chord
    line, that point is at x_start = xc + R sin|phi|.

    Returns a dict: xc, R, seed_xc, seed_R, n_upper, n_lower (points on the
    arc per surface, TE point included), flank_slope, x_start, and x_sample
    (the first point off the arc).
    """
    upper, lower = split_surfaces(coords)
    seed_xc, seed_R = te_radius_3pt(coords, 1)
    xc, R = seed_xc, seed_R
    for _ in range(max_iter):
        n_up = _n_on_circle(upper, xc, R, tol)
        n_lo = _n_on_circle(lower, xc, R, tol)
        if n_up < 2 or n_lo < 2:
            raise ValueError(
                "fewer than 2 points landed on the fitted circle: this "
                "section may not have a rounded TE, or `tol` is too tight")
        if n_up + 1 >= len(upper):
            raise ValueError("no flank points left after the TE circle")
        new_xc, new_R = fit_circle(np.vstack([upper[:n_up], lower[:n_lo]]))
        converged = np.allclose([new_xc, new_R], [xc, R])
        xc, R = new_xc, new_R
        if converged:
            break
    p, q = upper[n_up], upper[n_up + 1]
    flank_slope = (q[1] - p[1]) / (q[0] - p[0])
    return {
        "xc": xc, "R": R,
        "seed_xc": seed_xc, "seed_R": seed_R,
        "n_upper": n_up, "n_lower": n_lo,
        "flank_slope": flank_slope,
        "x_start": xc + R * np.sin(np.arctan(abs(flank_slope))),
        "x_sample": p[0],
    }


def perp_method(cut_pt, slope):
    """fillet_method 0: the circle tangent to the replica at the cut.

    The normal to the upper surface at cut_pt = (x, y), where the slope is
    dy/dx, meets its mirror image on y = 0 at x + slope * y; that is the
    centre, and its distance to cut_pt the radius. Returns (center, r)."""
    x, y = cut_pt
    center = np.array([x + slope * y, 0.0])
    return center, float(np.hypot(x - center[0], y))


def circle_method(cut_pt, te_pt):
    """fillet_method 1: the circle (centre on y = 0) through the two cut
    points and the original section's TE point. The TE is matched exactly;
    the joins at the cut points are not tangent. Returns (center, r)."""
    x, y = cut_pt
    xc, r = fit_circle(np.array([[x, y], te_pt, [x, -y]], dtype=float))
    return np.array([xc, 0.0]), float(r)


def make_fillet(center, cut_pt, n_side=24):
    """n_side points on the upper half of the fillet, strictly between the TE
    point (angle 0 about center) and cut_pt, evenly spaced in angle and
    ordered from the TE towards the cut. The lower half is its mirror image."""
    dx, dy = cut_pt[0] - center[0], cut_pt[1] - center[1]
    r, phi = np.hypot(dx, dy), np.arctan2(dy, dx)
    theta = phi * np.arange(1, n_side + 1) / (n_side + 1)
    return np.column_stack((center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)))


def trim_at(coords, cut_pt):
    """The points of a Selig-ordered section upstream of the cut (x < cut x),
    between the upper cut point and its mirror image on the lower surface."""
    x, y = cut_pt
    keep = coords[coords[:, 0] < x]
    return np.vstack(([x, y], keep, [x, -y]))


def split_surfaces(coords):
    """Selig order is TE -> upper surface -> LE -> lower surface -> TE.
    Split at the LE (min x) and return both branches ordered TE-first."""
    le = int(np.argmin(coords[:, 0]))
    upper = coords[:le + 1]        # already TE -> LE
    lower = coords[le:][::-1]      # LE -> TE, reversed to TE -> LE
    return upper, lower


def _n_on_circle(branch, xc, R, tol):
    """How many points, walking out from branch[0] (the TE), stay within `tol`
    (relative) of radius R from (xc, 0). Fillet points sit essentially exactly
    on the circle; once the surface peels away the deviation jumps by 1-2
    orders of magnitude, so this index is a sharp cutoff, not a fuzzy one."""
    dev = np.abs(np.hypot(branch[:, 0] - xc, branch[:, 1]) - R) / R
    off = np.nonzero(dev > tol)[0]
    return int(off[0]) if len(off) else len(branch)


def fit_circle(points):
    """Least-squares circle fit with the centre constrained to y = 0 (the TE
    fillet is symmetric about the chord line), so xc is the only free centre
    coordinate. Returns (xc, R)."""
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([2 * x, np.ones_like(x)])
    b = x**2 + y**2
    (xc, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return xc, np.sqrt(c + xc**2)


def te_radius_3pt(coords, k):
    """y=0-constrained circle fit through the TE point and the k-th point out
    on each surface (k=1 is the point adjacent to the TE)."""
    upper, lower = split_surfaces(coords)
    return fit_circle(np.array([upper[k], upper[0], lower[k]]))
