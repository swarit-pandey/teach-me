"""Provided helpers for workbook 16 (not exercises)."""
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


def numerical_grad(f, x, h=1e-5):
    """Central-difference gradient of scalar f() w.r.t. array x (perturbed in place)."""
    g = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = f()
        x[i] = old - h
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g
