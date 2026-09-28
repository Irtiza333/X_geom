# import math
from dataclasses import dataclass
import numpy as np
# from Cross_sections import cross_section_points
# from untrucate import untruncated_profile



pasc_tri = [   [1],            # n=0
              [1, 1],           # n=1
            [1, 2, 1],          # n=2
           [1, 3, 3, 1],         # n=3
         [1, 4, 6, 4, 1],        # n=4
      [1, 5, 10, 10, 5, 1],     # n=5
    [1, 6, 15, 20, 15, 6, 1],]  # n=6

def binom(n, k):
  while n >= len(pasc_tri):
    s = len(pasc_tri)
    prev = pasc_tri[-1]
    nextRow = [1] + [prev[i - 1] + prev[i] for i in range(1,s)] + [1]
    pasc_tri.append(nextRow)
  return pasc_tri[n][k]

def Bezierxy_t(ctrl_pts,n_points, t_order):
  bez_mat_t = np.zeros((n_points+1,t_order+1))
  bi_coeffs_t = np.array([binom(t_order, j) for j in np.arange(t_order + 1)])
  t_bez = np.atleast_1d(np.linspace(0,1,n_points+1)[:,None])
  pt_t_ind = np.arange((t_order+1), dtype=float)
  bez_mat_t = bi_coeffs_t * (1 - t_bez) ** (t_order - pt_t_ind) * t_bez ** pt_t_ind
  t_distr = bez_mat_t @ ctrl_pts
  return t_distr


# --------------------------------------------------------------------------
# Candidate curve
# --------------------------------------------------------------------------

def build_half_airfoil(xi, n_bez_samples):
    """free vars -> cross_section_points -> rounded half-airfoil curve.

    Returns the rounded profile as (x, y) arrays. cross_section_points can raise
    or return NaNs for degenerate candidates (curve never reaches y = r, no valid
    tangency segment, zero-length segments); those come back as empty arrays so
    EvalObj can score them as a heavy penalty instead of the PSO crashing.
    """
    # normalizing the free variables to fractions of chord
    # look at if what is coming in is normalized or not, can probably get rid of this
    # call or the multiplication by chord in the cross_section_points function
    # y1, x2, x3, y3 = _to_cross_section_params(xi)
    y1, x2, x3, y3 = xi   
    _Bx, _By = cross_section_points(y1, x2, x3, y3, n_bez_samples)

    _Bx = np.asarray(_Bx, dtype=float)
    _By = np.asarray(_By, dtype=float)

    return _Bx, _By




# Initial Points
def cross_section_points(y1, x2, x3, y3, bezier_sample):
    X0 = 0 # Fixed at 0
    Y0 = 0 # Fixed at 0
    X1 = X0 # Fixed at 0
    Y1 = y1
    X2 = x2
    Y2 = Y1
    X3 = x3
    Y3 = y3
    X4 = 1 # Fixed at c
    Y4 = 0 # Fixed at 0
  
    points = np.array([[X0, X1, X2, X3, X4], [Y0, Y1, Y2, Y3, Y4]])
    points4bez = np.transpose(points) # Flipping the points so that the leading edge is on the right and the trailing edge is on the left
    # get thickness distribution (surface coords) from bezier curve
    t_distr = Bezierxy_t(points4bez, n_points=bezier_sample, t_order=4)
    # Bx, By = bezier(points[0, :], points[1, :], samples=bezier_sample)

    Bx = t_distr[:, 0]
    By = t_distr[:, 1]
   
    return Bx, By

#--------------------------------------------------------------------------
# Control points / parameter mapping
# --------------------------------------------------------------------------

def expand_ctrl_pts(xi, chord):
    """Full 5x2 control-point array (LE -> TE, x = chord down to x = 0), for
    plotting the control polygon. Same polygon cross_section_points builds,
    just listed from the LE end."""
    Y1, X2, X3, Y3 = xi
    return np.array([
        [0.0,      0.0],   # P0: leading edge, fixed at (0, 0)
        [0.0,      Y1],    # P1: X1 = X0 = chord (locked), Y1 free
        [X2, Y1],    # P2: X2 free (distance from LE), Y2 = Y1 (locked)
        [X3, Y3],    # P3: both free (X3 measured as distance from LE)
        [chord,        0.0],   # P4: trailing edge, fixed at the origin
    ])

