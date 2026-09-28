"""
This code returns the values by how much the particle violates the constraints.
                                                Irtiza Khan
                                                Date started: September 26, 2025
"""

import numpy as np


def penalty_check(pop, search_bounds):
    """
    Return the values by how much the particle violates the constraints: for
    each particle, the summed distance by which its variables lie outside
    their [low, high] bounds (0 for a particle inside the bounds).
    """
    pop = np.atleast_2d(np.asarray(pop, dtype=float))
    search_bounds = np.asarray(search_bounds, dtype=float)
    below = np.maximum(search_bounds[:, 0] - pop, 0.0)
    above = np.maximum(pop - search_bounds[:, 1], 0.0)
    return (below + above).sum(axis=1)
