"""Workbook 02 — Matrices as transformations.

Rules of the game
  * Part A: plain Python only. A matrix is a list of rows (``[[a, b], [c, d]]``),
    a vector is a list. No numpy. Feel where every number goes.
  * Part B: NumPy, vectorized — no Python loops over points.
  * Part C: stretch. Uses the "columns = where the basis vectors land" idea.
  * Run   ./course check 02   to grade. Each function names the idea it practises.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import math
import numpy as np


# ----------------------------------------------------------------------------
# Part A — from scratch (lists)
# ----------------------------------------------------------------------------

def shape(A):
    """(rows, cols) of a list-of-rows matrix. Raise ValueError if the rows have
    different lengths (a ragged "matrix" is a bug, not a matrix)."""
    raise NotImplementedError


def matvec_cols(A, x):
    """A @ x computed the COLUMN way: a linear combination of A's columns,
    weighted by the entries of x:   A x = x[0]*col_0 + x[1]*col_1 + ...

    A is m x n (list of m rows), x has length n; return a list of length m.
    Raise ValueError if len(x) != n.
    Hint: start from [0.0]*m and add x[j] * A[i][j] into slot i, looping j outside.
    """
    raise NotImplementedError


def matvec_rows(A, x):
    """A @ x computed the ROW way: output entry i is (row i of A) · x.
    Same contract as matvec_cols (and must give the same answer)."""
    raise NotImplementedError


def det2(A):
    """Determinant of a 2x2 matrix [[a, b], [c, d]]:  ad - bc.
    Raise ValueError if A is not 2x2."""
    raise NotImplementedError


def rotation(theta):
    """2x2 matrix rotating the plane counter-clockwise by `theta` radians.
    Hint: where do (1,0) and (0,1) land? Those are the columns."""
    raise NotImplementedError


def scaling(sx, sy):
    """2x2 matrix stretching x by sx and y by sy."""
    raise NotImplementedError


def shear(k):
    """Horizontal shear: (x, y) -> (x + k*y, y). (1,0) stays put, (0,1) leans to (k,1)."""
    raise NotImplementedError


def reflection(theta):
    """Reflection across the line through the origin at angle `theta` (radians).

    Hint: the result is [[cos 2θ, sin 2θ], [sin 2θ, -cos 2θ]] — but derive it:
    reflect(v) = 2*proj(v) - v, where proj is projection onto the unit vector
    u = (cos θ, sin θ) (lesson 01). Apply that to (1,0) and (0,1).
    """
    raise NotImplementedError


def projection(theta):
    """Orthogonal projection onto the line through the origin at angle `theta`.
    With u = (cos θ, sin θ):  proj(v) = (u·v) u, so the matrix is u uᵀ."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — vectorized (NumPy)
# ----------------------------------------------------------------------------

def transform_points(A, P):
    """Apply the 2x2 (or m x n) matrix A to every point in P.

    P has shape (N, n): one point per ROW (the usual data-matrix layout).
    Return shape (N, m). No loops.
    Hint: for a single column vector it's A @ p. Stacking points as rows means
    you need P @ A.T  (transpose both sides of  A pᵀ ).
    """
    raise NotImplementedError


def is_linear(f, n, trials=20, seed=0, tol=1e-8):
    """Numerically test whether f: R^n -> R^m (takes and returns 1-D numpy arrays)
    is linear: f(a x + b y) == a f(x) + b f(y) for random x, y, a, b.

    Use np.random.default_rng(seed). Return True/False.
    Compare with np.allclose(..., atol=tol, rtol=1e-6).
    """
    raise NotImplementedError


def shoelace_area(P):
    """SIGNED area of the polygon with vertices P (shape (N, 2), in order).
    Positive for counter-clockwise order, negative for clockwise.

    Shoelace formula: 1/2 * sum_i (x_i y_{i+1} - x_{i+1} y_i), indices wrap around.
    Hint: np.roll(P, -1, axis=0) gives the "next" vertex for every i.
    """
    raise NotImplementedError


def unit_square_image_area(A):
    """Transform the unit square (0,0),(1,0),(1,1),(0,1) by the 2x2 matrix A and
    return the SIGNED area of the image (via shoelace_area).

    It should equal det(A) exactly — that is the whole point of the determinant.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch
# ----------------------------------------------------------------------------

def matrix_of(f, n):
    """Recover the matrix of a linear function f: R^n -> R^m.

    Column j of the matrix is f(e_j), where e_j is the j-th standard basis vector.
    Return an (m, n) numpy array. (This works because x = sum_j x_j e_j and f is
    linear, so f(x) = sum_j x_j f(e_j).)
    """
    raise NotImplementedError


def orientation(A, tol=1e-12):
    """Classify what a 2x2 matrix does to orientation:
    return "preserved" if det > tol, "flipped" if det < -tol, else "collapsed"."""
    raise NotImplementedError
