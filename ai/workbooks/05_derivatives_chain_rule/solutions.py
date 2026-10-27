"""Workbook 05 — reference solutions. Try first; read after."""
import math
import numpy as np


def forward_diff(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h


def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def error_study(f, df, x, hs, method="central"):
    if method == "forward":
        est = forward_diff
    elif method == "central":
        est = central_diff
    else:
        raise ValueError(f"unknown method {method!r}")
    truth = df(x)
    return [abs(est(f, x, h) - truth) for h in hs]


def best_h(f, df, x, method="central"):
    hs = [10 ** (k / 10) for k in range(-160, 1)]
    errs = error_study(f, df, x, hs, method)
    i = min(range(len(hs)), key=lambda j: errs[j])
    return hs[i]


def compose(outer, inner):
    f, df = outer
    g, dg = inner
    def h(x):
        return f(g(x))
    def dh(x):
        u = g(x)                 # forward: compute the inner value once...
        return df(u) * dg(x)     # ...then multiply the two local rates
    return h, dh


def compose_all(pairs):
    acc = (lambda x: x, lambda x: 1.0)
    for p in pairs:              # each new pair wraps around everything so far
        acc = compose(p, acc)
    return acc


def newton(f, df, x0, tol=1e-12, max_iter=50):
    x = x0
    hist = [x]
    for _ in range(max_iter + 1):
        fx = f(x)
        if abs(fx) <= tol:
            return x, hist
        d = df(x)
        if d == 0:
            raise ValueError(f"zero derivative at x={x}")
        x = x - fx / d
        hist.append(x)
    raise RuntimeError(f"Newton did not converge in {max_iter} steps (last x={x})")


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))                       # always <= 1: never overflows
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh_prime(z):
    return 1 - np.tanh(np.asarray(z, dtype=float)) ** 2


def relu(z):
    return np.maximum(0.0, np.asarray(z, dtype=float))


def relu_prime(z):
    return (np.asarray(z) > 0).astype(float)


def second_diff(f, x, h=1e-4):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def newton_minimize(df, d2f, x0, tol=1e-10, max_iter=50):
    return newton(df, d2f, x0, tol=tol, max_iter=max_iter)
