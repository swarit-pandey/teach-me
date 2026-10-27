"""Workbook 03 — Matmul, systems, inverse & rank.

Rules of the game
  * Part A: plain Python only (lists of lists, loops). No numpy. Matrices are
    lists of rows:  A = [[1, 2], [3, 4]]  is the 2x2 matrix with rows (1,2), (3,4).
    Never modify the caller's input lists — copy first.
  * Part B: elimination-based algorithms. You may use numpy arrays for storage
    and row operations (A[i] -= f * A[k]), but NOT np.linalg (no solve, inv,
    matrix_rank, svd, ...). The tests compare your answers against np.linalg.
  * Run   ./course check 03   to grade. Each function says what is expected.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Part A — matrix multiplication from scratch (lists)
# ----------------------------------------------------------------------------

def shape(A):
    """(rows, cols) of a list-of-rows matrix.  shape([[1,2,3],[4,5,6]]) -> (2, 3).

    Raise ValueError if the rows have different lengths (a ragged "matrix").
    """
    raise NotImplementedError


def matmul(A, B):
    """Product C = A B with the classic triple loop: C[i][j] = sum_k A[i][k] * B[k][j].

    A is (m, n), B is (n, p), result is (m, p) as a list of lists.
    Raise ValueError if the inner dimensions don't match (lesson §2: shape rules).
    """
    raise NotImplementedError


def outer(u, v):
    """Outer product u v^T of two lists: the (len(u), len(v)) matrix with entries u[i]*v[j]."""
    raise NotImplementedError


def matmul_outer(A, B):
    """Same product A B, computed as a SUM OF OUTER PRODUCTS (view 4 in the lesson):

        A B = sum_k  (column k of A) (row k of B)^T

    Build each rank-1 matrix with `outer` and add them up. Must agree with `matmul`.
    Raise ValueError on mismatched inner dimensions.
    """
    raise NotImplementedError


def identity(n):
    """n x n identity matrix as a list of lists (floats or ints)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — elimination (numpy storage allowed, np.linalg forbidden)
# ----------------------------------------------------------------------------

def forward_eliminate(A, b, tol=1e-12):
    """Reduce the square system A x = b to upper-triangular form using Gaussian
    elimination WITH PARTIAL PIVOTING.

    For each column k: swap into row k the row (at or below k) whose |entry| in
    column k is largest, then subtract multiples of row k to zero out every entry
    below the pivot.

    Returns (U, c): float arrays with U upper triangular and U x = c having the
    same solution as A x = b. Do not modify the inputs.
    Raise np.linalg.LinAlgError if a pivot is smaller than tol in absolute value
    (the matrix is singular).

    Hint: np.argmax(np.abs(U[k:, k])) + k finds the pivot row; swap with
    U[[k, p]] = U[[p, k]] (same for c).
    """
    raise NotImplementedError


def back_substitute(U, c):
    """Solve U x = c for upper-triangular U, from the last row up:

        x[i] = (c[i] - sum_{j>i} U[i, j] x[j]) / U[i, i]

    Returns a float array.
    """
    raise NotImplementedError


def solve(A, b):
    """Solve the square system A x = b: forward_eliminate then back_substitute.
    Raise np.linalg.LinAlgError for singular A.
    """
    raise NotImplementedError


def inverse(A, tol=1e-12):
    """Inverse of a square matrix by GAUSS–JORDAN elimination.

    Form the augmented matrix [A | I]. Row-reduce (with partial pivoting) until the
    left half is I; the right half is then A^{-1}. Raise np.linalg.LinAlgError if A
    is singular.

    Hint: process each column k: pivot, divide row k by its pivot, then eliminate
    column k from EVERY other row (above and below) — that's the "Jordan" part.
    """
    raise NotImplementedError


def rref(A, tol=1e-10):
    """Reduced row echelon form of any (m, n) matrix.

    Returns (R, pivots): R is the RREF as a float array (each pivot is 1, the only
    nonzero in its column) and pivots is the list of pivot column indices, in order.
    Entries with |value| < tol count as zero. Use partial pivoting.

    Hint: keep a row counter r. Walk the columns j = 0..n-1; if column j has no
    usable pivot at or below row r, skip it (it's a FREE column); otherwise pivot,
    normalize, eliminate above and below, and advance r.
    """
    raise NotImplementedError


def rank(A, tol=1e-10):
    """Rank = number of pivots in the RREF."""
    raise NotImplementedError


def null_space(A, tol=1e-10):
    """A basis for the null space {x : A x = 0}, as an (n, n - rank) array whose
    COLUMNS are the basis vectors. Return shape (n, 0) when only x = 0 works.

    Recipe (lesson §7): from the RREF, each free column f gives one basis vector:
    set x[f] = 1, the other free variables to 0, and solve for each pivot
    variable:  x[pivot_col_i] = -R[i, f].
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch
# ----------------------------------------------------------------------------

def solve_general(A, b, tol=1e-10):
    """For any (m, n) A, return ONE solution x of A x = b, or None if the system
    is inconsistent (b not in the column space of A).

    Hint: row-reduce the augmented matrix [A | b]. If a pivot lands in the last
    column, there's a row saying 0 = nonzero: no solution. Otherwise set the free
    variables to 0 and read the pivot variables off the last column.
    """
    raise NotImplementedError
