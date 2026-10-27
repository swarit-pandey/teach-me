"""Workbook 09 — Expectation, variance, Gaussians & the CLT.

Rules of the game
  * Part A: plain Python only (lists of floats, loops, the math module). No numpy.
    You are building mean / variance / covariance by hand so you know exactly
    what np.mean, np.var and np.cov do (and which `ddof` they use).
  * Part B: NumPy, vectorized. You may use np.mean / np.sum / matrix products,
    but NOT np.cov, np.corrcoef, np.var, np.std, np.linalg.cholesky or
    rng.multivariate_normal — those are what you're re-building.
  * Part C: stretch — experiments that make the theory visible.
  * Run   ./course check 09   to grade. Stuck > 20 min? Read the Hint, then
    peek at solutions.py, close it, and re-write from memory.
"""
import math
import numpy as np


# ----------------------------------------------------------------------------
# Part A — from scratch (lists)
# ----------------------------------------------------------------------------

def expectation(values, probs):
    """E[X] for a discrete random variable that takes values[i] with probability probs[i].

    expectation([1, 2, 3, 4, 5, 6], [1/6]*6) -> 3.5
    Raise ValueError if the lengths differ or the probabilities don't sum to 1
    (tolerance 1e-9) or any probability is negative.
    """
    raise NotImplementedError


def discrete_variance(values, probs):
    """Var[X] = E[(X - E[X])^2] for a discrete random variable (same inputs as above).

    A fair die has variance 35/12.
    """
    raise NotImplementedError


def mean(xs):
    """Sample mean of a non-empty list. Raise ValueError on an empty list."""
    raise NotImplementedError


def variance(xs, ddof=0):
    """Sample variance:  sum (x_i - mean)^2 / (n - ddof).

    ddof=0 is the "plug-in" (biased, maximum-likelihood) estimate; ddof=1 is the
    unbiased estimate (Bessel's correction, lesson §4). Raise ValueError if
    n - ddof <= 0.
    """
    raise NotImplementedError


def std(xs, ddof=0):
    """Square root of variance(xs, ddof)."""
    raise NotImplementedError


def covariance(xs, ys, ddof=0):
    """Sample covariance:  sum (x_i - mean_x)(y_i - mean_y) / (n - ddof).

    Raise ValueError if the lengths differ. Note covariance(xs, xs) == variance(xs).
    """
    raise NotImplementedError


def correlation(xs, ys):
    """Pearson correlation  cov(x, y) / (std(x) std(y)),  always in [-1, 1].

    Hint: ddof cancels between numerator and denominator, so any consistent
    choice works. Clamp to [-1, 1] to absorb floating-point error.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — vectorized (NumPy)
# ----------------------------------------------------------------------------

def covariance_matrix(X, ddof=1):
    """Covariance matrix of data X with shape (n, d): n samples, d features.

    Returns a (d, d) array whose (j, k) entry is the covariance of column j and
    column k. Must equal np.cov(X, rowvar=False, ddof=ddof) — but don't call it.
    Hint: center the columns (Xc = X - X.mean(axis=0)), then one matrix product:
    Xc.T @ Xc / (n - ddof).
    """
    raise NotImplementedError


def standardize(X):
    """Z-score every column of X (shape (n, d)): subtract the column mean, divide by
    the column standard deviation (ddof=0). Columns with zero spread become all 0
    (don't divide by zero). Returns a new array of the same shape.
    """
    raise NotImplementedError


def gaussian_pdf(x, mu=0.0, sigma=1.0):
    """Density of N(mu, sigma^2) at x (x may be a float or a numpy array).

        p(x) = exp(-(x - mu)^2 / (2 sigma^2)) / (sigma sqrt(2 pi))

    Raise ValueError if sigma <= 0.
    """
    raise NotImplementedError


def normal_cdf(x, mu=0.0, sigma=1.0):
    """P(X <= x) for X ~ N(mu, sigma^2), for a float x.

    Hint: math.erf.  Phi(z) = (1 + erf(z / sqrt 2)) / 2  with  z = (x - mu) / sigma.
    """
    raise NotImplementedError


def cholesky(A):
    """Cholesky factor of a symmetric positive-definite matrix A (shape (d, d)):
    the lower-triangular L with positive diagonal and L @ L.T == A.

    Raise ValueError if A is not positive definite (a pivot under the square
    root is <= 0). Loops over rows/columns are fine here (d is small).

    Hint: work row by row. For j < i:
        L[i, j] = (A[i, j] - sum_{k<j} L[i, k] L[j, k]) / L[j, j]
    and on the diagonal
        L[i, i] = sqrt(A[i, i] - sum_{k<i} L[i, k]^2).
    """
    raise NotImplementedError


def sample_mvn(mu, Sigma, n, rng):
    """Draw n samples from the multivariate normal N(mu, Sigma). Returns shape (n, d).

    Use YOUR cholesky: if z ~ N(0, I) then x = mu + L z has covariance L L^T = Sigma.
    rng is a np.random.Generator; use rng.standard_normal only.
    Hint: with Z of shape (n, d), the rows of  Z @ L.T  are the L z's.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — experiments
# ----------------------------------------------------------------------------

def clt_standardized_sums(n, trials, rng):
    """CLT experiment. For each of `trials` trials, add up n independent
    Uniform(0, 1) draws, then standardize the sum using its EXACT mean and
    standard deviation (each uniform has mean 1/2 and variance 1/12).

    Returns an array of shape (trials,). By the CLT it should look like N(0, 1)
    for moderate n (even n = 12 is already very close).
    """
    raise NotImplementedError


def variance_bias_experiment(n, trials, sigma, rng):
    """Why n - 1? Draw `trials` independent samples of size n from N(0, sigma^2).
    For each sample compute the ddof=0 and ddof=1 variance estimates.

    Returns (avg_biased, avg_unbiased): the averages over trials. Theory says
    avg_biased ≈ (n-1)/n sigma^2 and avg_unbiased ≈ sigma^2.
    Vectorize: one (trials, n) array, reductions along axis=1.
    """
    raise NotImplementedError


def standard_error_curve(ns, trials, rng):
    """For each n in ns, draw `trials` samples of size n of a fair die
    (rng.integers(1, 7)), compute each sample's mean, and record the standard
    deviation (ddof=0) of those `trials` means.

    Returns an array of len(ns). Theory: sqrt(35/12) / sqrt(n).
    """
    raise NotImplementedError
