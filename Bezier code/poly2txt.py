import numpy as np

def poly2txt(p):
    """
    Convert coefficient vector p into a string like "1*x.^2 - 4*x + 7"
    OCTAVE-FRIENDLY version
    """
    n = len(p)
    parts = [None] * n
    
    for k in range(n):
        c = p[k]
        e = n - k - 1  # exponent for this term (adjusted for 0-based indexing)
        
        if abs(c) < np.finfo(float).eps:
            continue  # skip zero coefficients
            
        # Choose format based on exponent
        if e > 1:
            fmt = '{}*x.^{}'
        elif e == 1:
            fmt = '{}*x'
        else:
            fmt = '{}'
            
        if e > 1:
            term = fmt.format(abs(c), e)
        elif e == 1:
            term = fmt.format(abs(c))
        else:
            term = fmt.format(abs(c))
            
        if k == 0:
            # first term: include sign only if negative
            if c < 0:
                parts[k] = '-' + term
            else:
                parts[k] = term
        else:
            # subsequent terms: always show "+"/"-"
            if c < 0:
                parts[k] = ' - ' + term
            else:
                parts[k] = ' + ' + term
    
    # Concatenate all nonempty parts
    s = ''
    for k in range(n):
        if parts[k] is not None:
            s += parts[k]
            
    return s 