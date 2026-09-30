import numpy as np

def bspline_basis(i, d, u, knots, m):
    """
    Cox–de Boor recursion for i-th B-spline basis of degree d
    
    Parameters:
    i: zero-based index of the basis function (0..m-1)
    d: degree (0..order-1)
    u: array of parameter values
    knots: knot vector of length m + degree + 1
    m: number of control points
    """
    u = np.array(u)
    
    if d == 0:
        # Zero-degree basis: support on [knots(i+1), knots(i+2))
        if i < m - 1:
            N = ((u >= knots[i]) & (u < knots[i + 1])).astype(float)
        else:
            # Last interval: include the endpoint
            N = ((u >= knots[i]) & (u <= knots[i + 1])).astype(float)
    else:
        # Recursive Cox–de Boor
        denom1 = knots[i + d] - knots[i]
        if denom1 > 0:
            N1 = bspline_basis(i, d - 1, u, knots, m)
            term1 = ((u - knots[i]) / denom1) * N1
        else:
            term1 = np.zeros_like(u, dtype=float)

        denom2 = knots[i + d + 1] - knots[i + 1]
        if denom2 > 0:
            N2 = bspline_basis(i + 1, d - 1, u, knots, m)
            term2 = ((knots[i + d + 1] - u) / denom2) * N2
        else:
            term2 = np.zeros_like(u, dtype=float)

        N = term1 + term2
    
    return N 