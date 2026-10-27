"""Provided helpers for workbook 18 (not exercises): test functions to optimize."""
import numpy as np


def rosenbrock(p, a=1.0, b=100.0):
    """The banana valley: f(x, y) = (a - x)^2 + b (y - x^2)^2, minimum 0 at (a, a^2)."""
    x, y = p
    return (a - x) ** 2 + b * (y - x * x) ** 2


def rosenbrock_grad(p, a=1.0, b=100.0):
    x, y = p
    return np.array([-2 * (a - x) - 4 * b * x * (y - x * x), 2 * b * (y - x * x)])


def quadratic(curvatures):
    """f(p) = 0.5 * sum_i c_i p_i^2 — an axis-aligned bowl/ravine. Returns (f, grad_f).
    The condition number max(c)/min(c) is what makes plain SGD slow (lesson 06)."""
    c = np.asarray(curvatures, dtype=float)
    return (lambda p: 0.5 * float(np.sum(c * np.asarray(p) ** 2)),
            lambda p: c * np.asarray(p, dtype=float))


def numerical_grad(f, x, h=1e-5):
    """Central-difference gradient of scalar f() w.r.t. array x (perturbed in place)."""
    g = np.zeros_like(x, dtype=float)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + h
        fp = f()
        x[i] = old - h
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g
