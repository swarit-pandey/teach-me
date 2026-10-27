"""Workbook 22 — reference solutions. Try first; read after."""
import math
import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)      # a row of all -inf stays nan-free until exp
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))


def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    w = softmax(scores, axis=-1)
    return w @ V, w


def self_attention(X, Wq, Wk, Wv, causal=False):
    mask = causal_mask(X.shape[-2]) if causal else None
    return attention(X @ Wq, X @ Wk, X @ Wv, mask)


def split_heads(X, n_heads):
    *lead, n, d = X.shape
    if d % n_heads:
        raise ValueError(f"d={d} not divisible by n_heads={n_heads}")
    return X.reshape(*lead, n, n_heads, d // n_heads).swapaxes(-3, -2)


def merge_heads(H):
    *lead, h, n, dh = H.shape
    return H.swapaxes(-3, -2).reshape(*lead, n, h * dh)


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    q, k, v = (split_heads(X @ W, n_heads) for W in (Wq, Wk, Wv))
    mask = causal_mask(X.shape[-2]) if causal else None
    out, w = attention(q, k, v, mask)          # head axis acts as a batch axis
    return merge_heads(out) @ Wo, w


def attention_torch(Q, K, V, mask=None):
    import torch
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return w @ V, w


def multi_head_attention_torch(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    import torch
    *lead, n, d = X.shape
    dh = d // n_heads

    def split(t):
        return t.view(*lead, n, n_heads, dh).transpose(-3, -2)

    q, k, v = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=X.device)) if causal else None
    out, w = attention_torch(q, k, v, mask)
    out = out.transpose(-3, -2).reshape(*lead, n, d)
    return out @ Wo, w
