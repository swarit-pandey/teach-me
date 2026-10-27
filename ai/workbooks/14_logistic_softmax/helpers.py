"""Synthetic datasets for workbook 14 (provided — not an exercise)."""
import numpy as np


def make_blobs(n_per_class, centers, std=1.0, seed=0):
    """Gaussian blobs. centers: (K, d). Returns X (K*n, d) float, y (K*n,) int in 0..K-1, shuffled."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(centers, float)
    K, d = centers.shape
    X = np.concatenate([c + std * rng.normal(size=(n_per_class, d)) for c in centers])
    y = np.repeat(np.arange(K), n_per_class)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def two_blobs(n_per_class=100, seed=0):
    return make_blobs(n_per_class, [[-1.5, -1.0], [1.5, 1.0]], std=1.0, seed=seed)


def three_blobs(n_per_class=100, seed=0):
    return make_blobs(n_per_class, [[0.0, 2.5], [-2.5, -1.5], [2.5, -1.5]], std=1.0, seed=seed)
