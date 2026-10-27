"""Workbook 07 — reference solutions. Try first; read after."""
import numpy as np


def numerical_jacobian(f, x, h=1e-5):
    x = np.array(x, dtype=float)
    cols = []
    for j in range(x.size):
        e = np.zeros_like(x); e[j] = h
        cols.append((np.asarray(f(x + e), dtype=float) - np.asarray(f(x - e), dtype=float)) / (2 * h))
    return np.stack(cols, axis=1)


def jac_affine(A, b, x):
    return np.array(A, dtype=float)


def jac_elementwise(fprime, x):
    return np.diag(fprime(np.asarray(x, dtype=float)))


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def jac_softmax(z):
    p = softmax(z)
    return np.diag(p) - np.outer(p, p)


def chain_jacobian(jac_f, jac_g, f, x):
    return jac_g(f(x)) @ jac_f(x)


def grad_least_squares(A, x, b):
    return 2 * A.T @ (A @ x - b)


def grad_linear_layer(W, x, y):
    delta = W @ x - y
    return np.outer(delta, x)


def init_params(d_in, d_hidden, d_out, seed=0):
    r = np.random.default_rng(seed)
    return {
        "W1": r.normal(size=(d_in, d_hidden)) / np.sqrt(d_in),
        "b1": np.zeros(d_hidden),
        "W2": r.normal(size=(d_hidden, d_out)) / np.sqrt(d_hidden),
        "b2": np.zeros(d_out),
    }


def forward(params, X):
    Z1 = X @ params["W1"] + params["b1"]
    H = np.tanh(Z1)
    Yh = H @ params["W2"] + params["b2"]
    return Yh, {"X": X, "H": H}


def mse_loss(Yh, Y):
    return 0.5 * np.sum((Yh - Y) ** 2) / Yh.shape[0]


def backward(params, cache, Yh, Y):
    X, H = cache["X"], cache["H"]
    N = X.shape[0]
    dYh = (Yh - Y) / N                      # dL/dYh, shape (N, d_out)
    dW2 = H.T @ dYh                         # (d_hidden, d_out) — shape of W2
    db2 = dYh.sum(axis=0)                   # bias was broadcast over the batch -> sum
    dH = dYh @ params["W2"].T               # vector-Jacobian product through H @ W2
    dZ1 = dH * (1 - H ** 2)                 # elementwise Jacobian is diagonal
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def grad_check(loss_fn, params, grads, h=1e-6):
    worst = 0.0
    for k, P in params.items():
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + h; lp = loss_fn(params)
            P[idx] = old - h; lm = loss_fn(params)
            P[idx] = old
            n = (lp - lm) / (2 * h)
            a = grads[k][idx]
            worst = max(worst, abs(a - n) / max(1e-12, abs(a) + abs(n)))
    return worst


def jacobian_product(Js, mode):
    if mode == "forward":
        acc, cost = Js[0], 0
        for J in Js[1:]:
            cost += J.shape[0] * J.shape[1] * acc.shape[1]
            acc = J @ acc
    elif mode == "reverse":
        acc, cost = Js[-1], 0
        for J in reversed(Js[:-1]):
            cost += acc.shape[0] * acc.shape[1] * J.shape[1]
            acc = acc @ J
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return acc, cost
