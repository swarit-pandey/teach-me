"""Provided helpers for workbook 19 (not exercises). Same spirals as workbook 16."""
import numpy as np


def make_spirals(n_per_class=100, n_classes=3, noise=0.2, seed=0):
    """The classic interleaved-spirals toy dataset (CS231n style).

    Returns X of shape (n_per_class * n_classes, 2) and integer labels y of shape (N,).
    Class j is a spiral arm: radius grows linearly 0 -> 1 while the angle sweeps
    4 radians, offset by 4j, plus Gaussian angle noise. No straight line separates
    the arms, so a linear/softmax classifier (lesson 14) tops out around 50%.
    """
    rng = np.random.default_rng(seed)
    N = n_per_class * n_classes
    X = np.zeros((N, 2))
    y = np.zeros(N, dtype=int)
    for j in range(n_classes):
        ix = slice(n_per_class * j, n_per_class * (j + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(j * 4.0, (j + 1) * 4.0, n_per_class) + rng.normal(0, noise, n_per_class)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


def make_blobs(n_per_class=100, centers=((-2.0, 0.0), (2.0, 0.0), (0.0, 2.5)), std=0.6, seed=0):
    """Gaussian blobs, one per class. Returns float32 X (N, 2) and int64 y (N,)."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.normal(c, std, size=(n_per_class, 2)) for c in centers]).astype(np.float32)
    y = np.repeat(np.arange(len(centers)), n_per_class).astype(np.int64)
    return X, y
