import numpy as np

def spmak1(knots, coefs):
    """
    Construct a B-form spline structure
    
    Parameters:
    knots – 1×(n+k) vector of knot positions (non-decreasing).
    coefs – dim×n matrix of control-point coordinates.
    
    Returns:
    S – dict with fields:
        form   = 'B-';
        knots  = knots;
        coefs  = coefs;
        order  = k (spline order);
        number = n (number of control points);
        dim    = dim (coordinate dimension).
    """
    # Validate inputs
    knots = np.array(knots).flatten()
    coefs = np.array(coefs)
    
    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)
    
    dim, n = coefs.shape

    # Determine spline order 'k' from multiplicity of the first knot value
    tol = max(abs(knots)) * np.finfo(float).eps
    idx_first_diff = np.where(knots > knots[0] + tol)[0]
    
    if len(idx_first_diff) == 0:
        raise ValueError('Cannot determine spline order: all knots equal.')
    
    k = idx_first_diff[0]

    # Check consistency: #knots = n + k
    if len(knots) != n + k:
        raise ValueError('Invalid inputs: length(knots) must equal n + order.')

    # Build the B-form struct
    S = {
        'form': 'B-',
        'knots': knots,
        'coefs': coefs,
        'order': k,
        'number': n,
        'dim': dim
    }
    
    return S 