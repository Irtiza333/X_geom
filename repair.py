"""
This code repairs the particle if it violates the constraints.
                                                Irtiza Khan
                                                Date started: September 26, 2025
"""

import numpy as np


def repair(pop, search_bounds, rng=None):
    """
    Replace the given particles with fresh uniform draws inside the bounds.
    rng: optional numpy Generator (np.random is used when it is None).
    """
    search_bounds = np.asarray(search_bounds, dtype=float)
    shape = (np.shape(pop)[0], search_bounds.shape[0])
    u = np.random.rand(*shape) if rng is None else rng.random(shape)
    return search_bounds[:, 0] + (search_bounds[:, 1] - search_bounds[:, 0]) * u
