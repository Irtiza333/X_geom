import numpy as np
from bspline_basis import bspline_basis

def fnval1(S, u):
    """
    Evaluate a B-form spline at parameters u 
    
    Parameters:
    S: spline structure from spmak1
    u: parameter values
    
    Returns:
    vals: dim×numel(u) array of spline values
    """
    # Extract fields
    knots = S['knots']
    coefs = S['coefs']
    k = S['order']
    m = S['number']
    dim = S['dim']

    u = np.array(u).flatten()  # row vector
    num_u = len(u)

    # Compute Cox–de Boor basis functions
    N = np.zeros((m, num_u))
    for i in range(m):
        N[i, :] = bspline_basis(i, k - 1, u, knots, m)

    # Weighted sum of control points
    vals = coefs @ N
    
    return vals 