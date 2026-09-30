import numpy as np
import matplotlib.pyplot as plt 
from numpy.linalg import inv  # matrix inverse; returns an (n,n) array given an (n,n) array

# ---------------------------------------------------------------------
# Kernel: exponential/RBF-like covariance
# params: a 2-element list/array
#   params[0] := amplitude (interpreted here as variance σ^2 at zero distance)
#   params[1] := length-scale factor (larger => faster decay in correlation)
# np.subtract.outer(x, y) builds all pairwise (xi - yj) differences
#   If x has shape (m,) and y has shape (n,), result is (m, n)
#   If both x and y are scalars, result is a 0-d scalar
# Output shape:
#   (m, n) for 1D arrays; scalar for scalars
# ---------------------------------------------------------------------
def exponential_cov(x, y, params):
    return params[0] * np.exp( -0.5 * params[1] * (np.subtract.outer(x, y)**2) )
    #               ^^^^^^
    #   variance at zero lag (k(z,z) = params[0] since exp(0)=1)
    # NOTE: the "/10" just rescales the distance; keep or tune as intended.


# ---------------------------------------------------------------------
# GP conditional (posterior) at x_new given training (x, y)
# x_new: (m,) array of test points OR scalar
# x    : (n,) array/list of training inputs
# y    : (n,) array/list of training targets (treated as (n,) here)
# params: kernel hyperparameters as above
#
# Returns:
#   mu   : (m,) posterior mean (squeezed to 1D, or scalar if m=1)
#   sigma: (m,m) posterior covariance (squeezed; scalar if m=1)
# ---------------------------------------------------------------------
def conditional(x_new, x, y, params):
    B = exponential_cov(x_new, x, params)      # K_*n : shape (m, n)
    C = exponential_cov(x, x, params)          # K_nn : shape (n, n)
    A = exponential_cov(x_new, x_new, params)  # K_** : shape (m, m)

    # Posterior mean: μ* = K_*n K_nn^{-1} y
    # Shapes: (m,n) @ (n,n) -> (m,n); then (m,n) @ (n,) -> (m,)
    mu = inv(C).dot(B.T).T.dot(y)

    # Posterior covariance: Σ* = K_** - K_*n K_nn^{-1} K_n*
    # Shapes: (m,n) @ (n,n) @ (n,m) -> (m,m)
    sigma = A - B.dot(inv(C).dot(B.T))

    return (mu.squeeze(), sigma.squeeze())
    # .squeeze() makes (m,) from (m,1) and scalar if m==1


# ---------------------------------------------------------------------
# Single-point GP prediction written in a more "manual" style
# x     : scalar test point x*
# data  : (n,) training inputs
# kernel: callable k(·,·,params) returning scalar for scalars or (m,n) for arrays
# params: kernel hyperparameters
# sigma : (n,n) training covariance matrix (ideally K_nn + noise*I)
# t     : (n,) training targets
#
# Returns:
#   y_pred    : scalar posterior mean at x*
#   sigma_new : scalar posterior variance at x*
# ---------------------------------------------------------------------
def predict(x, data, kernel, params, sigma, t):
    # Cross-covariance vector k_*n: length n
    # For scalar x and scalar yi, kernel returns a scalar; the list becomes a 1D vector of length n.
    k = [kernel(x, y, params) for y in data]  # list of n scalars -> behaves like (n,) in dot products

    # Inverse of the (n,n) covariance (expensive: O(n^3))
    Sinv = np.linalg.inv(sigma)               # shape (n, n)

    # Posterior mean: k^T K^{-1} t
    # Shapes: (n,) @ (n,n) -> (n,); then (n,) @ (n,) -> scalar
    y_pred = np.dot(k, Sinv).dot(t)           # scalar

    # Posterior variance: k(x,x) - k^T K^{-1} k
    # Shapes: scalar - (n,) @ (n,n) @ (n,) -> scalar
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)  # scalar

    return y_pred, sigma_new


# ===========================
# Data & hyperparameters
# ===========================

θ = [2, 1]                     # θ[0]=variance at zero lag; θ[1]=length-scale factor
σ_0 = exponential_cov(0, 0, θ)  # k(0,0): scalar ~ θ[0]; shape: () (0-d scalar)

x = [-1, 2, 3]                  # training inputs; length n=3
# WARNING: np.random.normal(..., scale=σ) expects **standard deviation**.
# σ_0 here is variance if θ[0] is variance. If so, using scale=σ_0 is too large.
# Prefer: np.sqrt(σ_0)
y = [
    np.random.normal(scale=σ_0),
    np.random.normal(scale=σ_0),
    np.random.normal(scale=σ_0)
]                               # training targets; length 3 (list of scalars)
# If σ_0 is variance, use: np.random.normal(scale=np.sqrt(σ_0))


# Training covariance K_nn (add tiny jitter for numerical stability)
σ_1 = exponential_cov(x, x, θ) + 1e-8 * np.eye(len(x))  # shape (3,3)
print(σ_1)                                              # prints a 3×3 matrix


# Grid of test points
x_pred = np.linspace(-3, 3, 300)                         # shape (30,)

# Predict at each test point; list of 30 tuples (mean, variance), both scalars
predictions = [predict(i, x, exponential_cov, θ, σ_1, y) for i in x_pred]
# Cross-covariance rows (optional diagnostic): each entry is a vector of length 3
k = [exponential_cov(i, x, θ) for i in x_pred]          # list of 30 arrays, each shape (3,)

# Precompute inverse (not used below by predict, but shown here; predict recomputes it per call)
Sinv = np.linalg.inv(σ_1)                                # shape (3,3)


# Unpack the list of pairs into two arrays of length 30:
# np.transpose(predictions) -> shape (2, 30); left is means, right is variances
y_pred, sigmas = np.transpose(predictions)              # y_pred: (30,), sigmas: (30,)
# NOTE: 'sigmas' here are **variances**. For error bars, matplotlib expects **std** (σ).
# Use np.sqrt(sigmas) if you want 1-σ bands.

# Plot posterior mean with vertical error bars
plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)    # If you want std bars, use yerr=np.sqrt(sigmas)
plt.plot(x, y, "ro")                                    # training points (3 of them)
plt.show()
