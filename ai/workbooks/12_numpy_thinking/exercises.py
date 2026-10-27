"""Workbook 12 — NumPy thinking: vectorization & broadcasting.

Rules of the game
  * NumPy only. No Python loops over array elements, rows or columns unless a
    docstring explicitly allows one. (A loop over a handful of *parameters* is fine.)
    The tests compare you against slow loop references, so correctness is all that's
    graded — but the point is to think in whole arrays.
  * Before writing a line, write the shapes down. Most bugs here are shape bugs.
  * Run   ./course check 12   to grade. Each function says which test covers it.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Part A — broadcasting & reductions
# ----------------------------------------------------------------------------

def pairwise_sq_dists(X, Y):
    """Squared Euclidean distances between every row of X and every row of Y.

    X: (n, d), Y: (m, d)  ->  D: (n, m) with D[i, j] = ||X[i] - Y[j]||^2.
    Use broadcasting: X[:, None, :] - Y[None, :, :] has shape (n, m, d).
    (test_pairwise_sq_dists)
    """
    raise NotImplementedError


def pairwise_sq_dists_gram(X, Y):
    """Same result as pairwise_sq_dists, but WITHOUT materializing the (n, m, d)
    tensor:  ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y.

    One matrix multiply plus broadcasting of two norm vectors. Floating-point
    cancellation can make tiny distances slightly negative: clip them at 0.
    (test_pairwise_gram)
    """
    raise NotImplementedError


def softmax(Z, axis=-1):
    """Numerically stable softmax along `axis` (default: last axis, i.e. per row).

    Must not overflow for logits like 1000. Hint: softmax(z) == softmax(z - max(z));
    use keepdims=True so the max broadcasts back against Z.
    (test_softmax_*)
    """
    raise NotImplementedError


def standardize(X):
    """Standardize each COLUMN of X (shape (n, d)) to mean 0 and std 1.

    Returns (Z, mean, std) where mean and std have shape (d,). Columns with zero
    standard deviation must not produce NaN: leave them at 0 (treat std as 1).
    (test_standardize)
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — indexing
# ----------------------------------------------------------------------------

def one_hot(y, num_classes):
    """Integer labels y (shape (n,)) -> float one-hot matrix (n, num_classes).

    Hint: fancy indexing.  out[np.arange(n), y] = 1.
    (test_one_hot)
    """
    raise NotImplementedError


def top_k(X, k):
    """Indices of the k largest entries in each row of X, largest first.

    X: (n, m) -> (n, k) int array. Ties may be broken any way.
    Hint: np.argsort(-X, axis=1)[:, :k] works. Stretch: np.argpartition is O(m)
    per row instead of O(m log m) — then sort only the k survivors with
    np.take_along_axis.
    (test_top_k)
    """
    raise NotImplementedError


def clip_outliers(X, n_std=3.0):
    """Return a COPY of X (n, d) where, per column, every value further than
    n_std standard deviations from the column mean is replaced by the column mean.

    Use a boolean mask and np.where — don't modify the input.
    (test_clip_outliers)
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — einsum & strides (stretch)
# ----------------------------------------------------------------------------

def batched_matmul(A, B):
    """A: (b, n, k), B: (b, k, m) -> (b, n, m), one matmul per batch element.

    Write it with np.einsum (one call). Indices: 'bnk,bkm->bnm'.
    (test_batched_matmul)
    """
    raise NotImplementedError


def bilinear_scores(Q, W, K):
    """Scores S[i, j] = Q[i] @ W @ K[j] for Q: (n, d), W: (d, e), K: (m, e).

    One np.einsum call. This is attention with a learned bilinear form (lesson 22).
    (test_bilinear)
    """
    raise NotImplementedError


def sliding_windows(x, w):
    """All length-w windows of the 1-D array x, as a 2-D array of shape (n-w+1, w).

    Row i is x[i:i+w]. It must be a VIEW (no copy): the result shares memory
    with x. Hint: np.lib.stride_tricks.sliding_window_view — or build it yourself
    with np.lib.stride_tricks.as_strided(x, shape=..., strides=(s, s)) where
    s = x.strides[0]. Raise ValueError if w < 1 or w > len(x).
    (test_sliding_windows)
    """
    raise NotImplementedError


def moving_average(x, w):
    """Moving average of 1-D x with window w: shape (n-w+1,), entry i is mean(x[i:i+w]).

    Use sliding_windows (or a cumulative sum — both are loop-free).
    (test_moving_average)
    """
    raise NotImplementedError
