"""Workbook 01 — Vectors & the dot product.

Rules of the game
  * Part A: plain Python only (lists of floats, loops). No numpy. The point is to
    feel every multiply-add.
  * Part B: NumPy, vectorized — no Python loops over vector components.
  * Run   ./course check 01   to grade. Each function says which test covers it.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import math
import numpy as np


# ----------------------------------------------------------------------------
# Part A — from scratch (lists)
# ----------------------------------------------------------------------------

def dot(a, b):
    """Dot product of two equal-length lists.  dot([1,2,3],[4,5,6]) -> 32.

    Raise ValueError if the lengths differ (silent broadcasting bugs are the
    #1 source of wrong ML code — be strict).
    """
    raise NotImplementedError


def norm(v, p=2):
    """L-p norm of a list. Support p=1, p=2, any p>=1, and p=math.inf (max |v_i|).

    Hint: the general formula is (sum |v_i|^p)^(1/p); handle inf separately.
    """
    raise NotImplementedError


def normalize(v):
    """Unit vector in the direction of v. Raise ValueError for the zero vector."""
    raise NotImplementedError


def cosine_similarity(a, b):
    """cos of the angle between a and b, in [-1, 1].

    Hint: floating-point error can produce 1.0000000002 — clamp the result so that
    math.acos never blows up downstream.
    """
    raise NotImplementedError


def angle_degrees(a, b):
    """Angle between a and b in degrees, in [0, 180]."""
    raise NotImplementedError


def project(b, a):
    """Vector projection of b onto a:  (a·b / a·a) a.   Returns a list."""
    raise NotImplementedError


def orthogonal_decomposition(b, a):
    """Split b into (parallel, perpendicular) parts w.r.t. a.

    Returns a tuple (b_par, b_perp) of lists with b_par + b_perp == b and
    b_perp · a == 0 (up to float error).
    """
    raise NotImplementedError


def coords_in_orthogonal_basis(x, basis):
    """Coefficients c such that x = sum_i c_i * basis[i], for an ORTHOGONAL basis.

    No linear solve allowed — use projections (lesson §9, problem 7).
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — vectorized (NumPy)
# ----------------------------------------------------------------------------

def cosine_matrix(W):
    """Given W of shape (n, d) — n vectors as rows — return the (n, n) matrix of
    pairwise cosine similarities.

    No Python loops: normalize the rows (W / norms[:, None]) then one matrix multiply.
    Hint: np.linalg.norm(W, axis=1, keepdims=True).
    """
    raise NotImplementedError


def nearest_words(query, vocab, E, k=3):
    """Return the k words most cosine-similar to `query`, best first, excluding
    the query itself.

    vocab: list of n words; E: (n, d) array, row i is the embedding of vocab[i].
    Hint: np.argsort on the negated similarity row.
    """
    raise NotImplementedError


def analogy(a, b, c, vocab, E):
    """Solve  a : b :: c : ?   with vector arithmetic:  target = E[b] - E[a] + E[c].

    Return the word (not a, b or c) whose embedding is most cosine-similar to the
    target. Classic example: man : king :: woman : queen.
    """
    raise NotImplementedError
