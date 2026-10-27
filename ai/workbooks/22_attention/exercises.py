"""Workbook 22 — Attention.

Rules of the game
  * Conventions (same as the lesson): tokens are ROWS. A sequence is X of shape
    (n, d), or (B, n, d) for a batch. Projections are right-multiplied:
    Q = X @ Wq with Wq of shape (d, d_k).
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.
  * Masks are boolean arrays with True = "may attend", broadcastable to the
    (..., n_q, n_k) score matrix.
  * Every function must accept arbitrary leading batch dimensions ("...").
    Use  @  (np.matmul), .swapaxes(-1, -2)  and  axis=-1 , never hard-coded axes.
  * Part A: NumPy attention. Part B: multi-head attention via reshape.
    Part C: the same in PyTorch, checked against NumPy and torch's own ops.
  * Run   ./course check 22   to grade.
"""
import math
import numpy as np


# ----------------------------------------------------------------------------
# Part A — scaled dot-product attention (NumPy)
# ----------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Numerically stable softmax along `axis`.

    Subtract the max along the axis before exponentiating (softmax is
    shift-invariant, lesson 14), so softmax([1000, 1001]) works.
    Entries equal to -inf must get probability exactly 0.
    """
    raise NotImplementedError


def causal_mask(n):
    """(n, n) boolean matrix, True where key j <= query i (lower triangle
    including the diagonal). Hint: np.tril."""
    raise NotImplementedError


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Q: (..., n_q, d_k), K: (..., n_k, d_k), V: (..., n_k, d_v)
    mask: optional boolean, broadcastable to (..., n_q, n_k); True = allowed.
    Returns (out, weights): out (..., n_q, d_v), weights (..., n_q, n_k), with
    every row of weights summing to 1 and weights == 0 wherever mask is False.

    Hint: scores = Q @ K.swapaxes(-1, -2) / sqrt(d_k);
          np.where(mask, scores, -np.inf); then your softmax.
    """
    raise NotImplementedError


def self_attention(X, Wq, Wk, Wv, causal=False):
    """Single-head self-attention: Q, K, V are all projections of the same X.
    X: (..., n, d); Wq, Wk: (d, d_k); Wv: (d, d_v). Returns (out, weights)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — multi-head attention by reshaping
# ----------------------------------------------------------------------------

def split_heads(X, n_heads):
    """(..., n, d) -> (..., n_heads, n, d // n_heads).

    Head h gets columns [h*dh, (h+1)*dh) of the last axis.
    Raise ValueError if d is not divisible by n_heads.
    Hint: reshape to (..., n, n_heads, dh), then swap the axes -3 and -2.
    """
    raise NotImplementedError


def merge_heads(H):
    """Inverse of split_heads: (..., n_heads, n, dh) -> (..., n, n_heads * dh)."""
    raise NotImplementedError


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    """Multi-head self-attention.

    X: (..., n, d); Wq, Wk, Wv, Wo: (d, d). Project once with the full
    matrices, split into heads, attend in every head at once (the head axis is
    just another batch axis for `attention`), merge, then apply Wo.
    Returns (out, weights) with out (..., n, d), weights (..., n_heads, n, n).
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — PyTorch
# ----------------------------------------------------------------------------

def attention_torch(Q, K, V, mask=None):
    """Same contract as `attention`, for torch tensors (any float dtype).
    Must be differentiable (no .numpy(), no in-place ops on inputs).
    Hint: scores.masked_fill(~mask, float("-inf")) and torch.softmax.
    """
    raise NotImplementedError


def multi_head_attention_torch(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    """Same contract as `multi_head_attention`, for torch tensors.
    Hint: x.view(..., n, h, dh).transpose(-3, -2), and .transpose(-3, -2)
    followed by .reshape to merge (a transposed tensor is not contiguous, so
    use reshape rather than view).
    """
    raise NotImplementedError
