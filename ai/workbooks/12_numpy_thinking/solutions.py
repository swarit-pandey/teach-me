"""Workbook 12 — reference solutions. Try first; read after."""
import numpy as np


def pairwise_sq_dists(X, Y):
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    diff = X[:, None, :] - Y[None, :, :]          # (n, 1, d) - (1, m, d) -> (n, m, d)
    return (diff ** 2).sum(axis=-1)               # reduce over d -> (n, m)


def pairwise_sq_dists_gram(X, Y):
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    xx = (X ** 2).sum(axis=1)                     # (n,)
    yy = (Y ** 2).sum(axis=1)                     # (m,)
    D = xx[:, None] + yy[None, :] - 2.0 * X @ Y.T  # (n,1) + (1,m) - (n,m)
    return np.maximum(D, 0.0)                     # cancellation can give -1e-15


def softmax(Z, axis=-1):
    Z = np.asarray(Z, float)
    Z = Z - Z.max(axis=axis, keepdims=True)       # largest logit becomes 0 -> exp <= 1
    E = np.exp(Z)
    return E / E.sum(axis=axis, keepdims=True)


def standardize(X):
    X = np.asarray(X, float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    safe = np.where(std > 0, std, 1.0)            # constant column -> divide by 1
    return (X - mean) / safe, mean, std


def one_hot(y, num_classes):
    y = np.asarray(y)
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1.0           # pairs (row i, column y[i])
    return out


def top_k(X, k):
    X = np.asarray(X)
    part = np.argpartition(-X, k - 1, axis=1)[:, :k]          # k largest, unordered
    vals = np.take_along_axis(X, part, axis=1)
    order = np.argsort(-vals, axis=1)                         # sort just those k
    return np.take_along_axis(part, order, axis=1)


def clip_outliers(X, n_std=3.0):
    X = np.asarray(X, float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    mask = np.abs(X - mu) > n_std * sd            # (n, d) boolean, broadcast per column
    return np.where(mask, mu, X)                  # np.where builds a new array


def batched_matmul(A, B):
    return np.einsum("bnk,bkm->bnm", A, B)


def bilinear_scores(Q, W, K):
    return np.einsum("id,de,je->ij", Q, W, K)


def sliding_windows(x, w):
    x = np.asarray(x)
    n = x.shape[0]
    if w < 1 or w > n:
        raise ValueError(f"window {w} invalid for length {n}")
    s = x.strides[0]
    # Row i starts i elements (i*s bytes) later; inside a row, step one element.
    return np.lib.stride_tricks.as_strided(x, shape=(n - w + 1, w), strides=(s, s), writeable=False)


def moving_average(x, w):
    return sliding_windows(np.asarray(x, float), w).mean(axis=1)
