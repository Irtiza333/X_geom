import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from Bezier import bezier

rng = np.random.default_rng()

# Future work: 
# Pull thickness at max height from blending database

def thickness_curves(T1_Y3, T2_Y3, X1_Y3, X2_Y3, sample_t):
    # --- Thickness 1 ---
    T1_Y0 = rng.uniform(0, 1)
    T1_Y1 = rng.uniform(0, 1)
    T1_Y2 = rng.uniform(0, 1)
    T1_Y = np.array([T1_Y0, T1_Y1, T1_Y2, T1_Y3])

    T1_X0 = 0
    T1_X1 = rng.uniform(0, 1)
    T1_X2 = rng.uniform(0, 1)
    T1_X3 = 1
    T1_X = np.array([T1_X0, T1_X1, T1_X2, T1_X3])

    # --- Thickness 2 ---
    T2_Y0 = rng.uniform(0, 1)
    T2_Y1 = rng.uniform(0, 1)
    T2_Y2 = rng.uniform(0, 1)
    T2_Y = np.array([T2_Y0, T2_Y1, T2_Y2, T2_Y3])

    T2_X0 = 0
    T2_X1 = rng.uniform(0, 1)
    T2_X2 = rng.uniform(0, 1)
    T2_X3 = 1
    T2_X = np.array([T2_X0, T2_X1, T2_X2, T2_X3])

    # --- X Pos. 1 ---
    X1_Y0 = rng.uniform(0, 1)
    X1_Y1 = rng.uniform(0, 1)
    X1_Y2 = rng.uniform(0, 1)
    X1_Y = np.array([X1_Y0, X1_Y1, X1_Y2, X1_Y3])

    X1_X0 = 0
    X1_X1 = rng.uniform(0, 1)
    X1_X2 = rng.uniform(0, 1)
    X1_X3 = 1
    X1_X = np.array([X1_X0, X1_X1, X1_X2, X1_X3])

    # --- X Pos. 2 ---
    X2_Y0 = rng.uniform(0, 1)
    X2_Y1 = rng.uniform(0, 1)
    X2_Y2 = rng.uniform(0, 1)
    X2_Y = np.array([X2_Y0, X2_Y1, X2_Y2, X2_Y3])

    X2_X0 = 0
    X2_X1 = rng.uniform(0, 1)
    X2_X2 = rng.uniform(0, 1)
    X2_X3 = 1
    X2_X = np.array([X2_X0, X2_X1, X2_X2, X2_X3])

    BxT1, ByT1 = bezier(T1_X, T1_Y, sample_t)
    BxX1, ByX1 = bezier(X1_X, X1_Y, sample_t)
    BxT2, ByT2 = bezier(T2_X, T2_Y, sample_t)
    BxX2, ByX2 = bezier(X2_X, X2_Y, sample_t)

    return BxT1, ByT1, T1_X, T1_Y, BxX1, ByX1, X1_X, X1_Y, BxT2, ByT2, T2_X, T2_Y, BxX2, ByX2, X2_X, X2_Y 