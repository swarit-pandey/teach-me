"""Workbook 04 — Eigenvectors, SVD & PCA.

Rules of the game
  * Part A: plain Python only (lists, loops, math). 2x2 eigen-analysis by hand,
    exactly as in lesson 04 section 2.
  * Part B: NumPy. You may use np.linalg.eigh / np.linalg.svd where a docstring
    says so — the point is to understand what they return and how the pieces fit,
    not to re-implement LAPACK.
  * Part C: stretch.
  * Run   ./course check 04   to grade. Each function says what is tested.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.

Conventions used throughout
  * Data matrices X have shape (n, d): n samples as ROWS, d features as columns.
  * "Components" are returned as the ROWS of a (k, d) array, unit length,
    ordered by decreasing variance.
  * Covariance uses the unbiased 1/(n-1) normalization (like np.cov).
"""
import math
import numpy as np


# ----------------------------------------------------------------------------
# Part A — 2x2 by hand (plain Python)
# ----------------------------------------------------------------------------

def char_poly_2x2(A):
    """Coefficients of the characteristic polynomial det(A - λI) = λ² + bλ + c
    for a 2x2 list-of-lists A. Return the tuple (b, c).

    Hint: expand (a-λ)(d-λ) - bc. You should find b = -trace(A), c = det(A).
    """
    raise NotImplementedError


def eig_2x2(A):
    """Real eigenvalues and unit eigenvectors of a 2x2 matrix (list of lists).

    Returns (vals, vecs) with vals = [l1, l2] sorted so l1 >= l2, and
    vecs = [v1, v2], each a unit-length list with A v_i = l_i v_i.
    Raise ValueError if the eigenvalues are complex (e.g. a rotation).

    Hint: solve the quadratic from char_poly_2x2. For an eigenvector, (A - λI)
    is singular, so any non-zero row (p, q) of it is orthogonal to v: take
    v = (-q, p). If both rows are zero, A = λI and every vector works.
    Repeated eigenvalue with a defective matrix (a shear) — return the same
    eigenvector twice.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — NumPy
# ----------------------------------------------------------------------------

def power_iteration(A, num_iters=1000, tol=1e-12, seed=0):
    """Dominant eigenpair of a square array A by power iteration.

    Start from a random unit vector (np.random.default_rng(seed)), repeat
    v <- A v / ||A v|| until v stops changing (up to sign) by more than tol, or
    num_iters is reached. Return (lam, v) where lam = v·Av (the Rayleigh
    quotient; v is unit length) and v is a unit vector.

    Hint: "stops changing up to sign" — compare min(||v_new - v||, ||v_new + v||)
    to tol, since a negative dominant eigenvalue flips v every step.
    """
    raise NotImplementedError


def covariance(X):
    """Sample covariance (d, d) of data X (n, d): center the columns, then
    Xc.T @ Xc / (n - 1). Do not call np.cov (but the test compares to it)."""
    raise NotImplementedError


def pca_eig(X, k):
    """PCA via eigendecomposition of the covariance matrix.

    Return (components, variances, mean):
      components: (k, d), rows = top-k unit eigenvectors of covariance(X),
                  ordered by decreasing eigenvalue;
      variances:  (k,), the matching eigenvalues (variance along each component);
      mean:       (d,), the column means of X.
    You may use np.linalg.eigh — but note it returns eigenvalues in ASCENDING
    order and eigenvectors as COLUMNS.
    """
    raise NotImplementedError


def pca_svd(X, k):
    """PCA via the SVD of the centered data — same return values as pca_eig.

    If Xc = U S V^T then Xc^T Xc / (n-1) = V (S²/(n-1)) V^T, so the components are
    the top rows of V^T and the variances are S²/(n-1).
    Use np.linalg.svd(Xc, full_matrices=False) (it returns S descending).
    The two PCA functions must agree up to the sign of each component.
    """
    raise NotImplementedError


def explained_variance_ratio(X):
    """Fraction of total variance along every principal component, (d,) array,
    descending, summing to 1."""
    raise NotImplementedError


def project(X, components, mean):
    """Coordinates of each sample in the component basis: (X - mean) @ components.T,
    shape (n, k)."""
    raise NotImplementedError


def reconstruct(Z, components, mean):
    """Map component coordinates Z (n, k) back to data space (n, d)."""
    raise NotImplementedError


def low_rank_approx(A, k):
    """Best rank-k approximation of a (m, n) array A: keep the top k singular
    triplets, A_k = sum_{i<k} s_i u_i v_i^T. Use np.linalg.svd.

    Hint: U[:, :k] * s[:k] @ Vt[:k] does it in one line (broadcasting scales
    each column of U by its singular value).
    """
    raise NotImplementedError


def eckart_young_error(A, k):
    """The Frobenius error ||A - A_k||_F predicted by Eckart–Young, computed from
    the singular values alone: sqrt(s_{k+1}² + s_{k+2}² + ...)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch
# ----------------------------------------------------------------------------

def top_k_eig_power(S, k, num_iters=2000, tol=1e-12, seed=0):
    """Top-k eigenpairs of a SYMMETRIC matrix S using only power_iteration plus
    deflation: after finding (lam, v), replace S with S - lam * outer(v, v) and
    repeat. Return (vals (k,), vecs (k, d) rows), ordered as found.

    Assumes S is positive semi-definite (like a covariance), so "dominant" means
    "largest". Why does deflation work? (Hint: spectral theorem, section 4.)
    """
    raise NotImplementedError
