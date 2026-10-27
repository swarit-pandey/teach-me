"""Workbook 16 — reference solutions. Try first; read after."""
import numpy as np

from helpers import make_spirals  # noqa: F401


# ---------------------------------------------------------------- Part A

def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (np.asarray(z) > 0).astype(float)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))                    # in (0, 1]: never overflows
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def tanh_grad(z):
    t = np.tanh(z)
    return 1.0 - t * t


def collapse_linear(layers):
    W_eff, b_eff = layers[0][0].copy(), layers[0][1].copy()
    for W, b in layers[1:]:
        # (X W_eff + b_eff) W + b = X (W_eff W) + (b_eff W + b)
        W_eff, b_eff = W_eff @ W, b_eff @ W + b
    return W_eff, b_eff


def count_params(sizes):
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def init_mlp(sizes, seed=0, scheme="he"):
    rng = np.random.default_rng(seed)
    params = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        if scheme == "he":
            std = np.sqrt(2.0 / d_in)
        elif scheme == "small":
            std = 0.01
        else:
            raise ValueError(f"unknown init scheme {scheme!r}")
        params.append((rng.normal(0.0, std, (d_in, d_out)), np.zeros(d_out)))
    return params


def mlp_forward(params, X):
    h, hiddens = X, []
    for i, (W, b) in enumerate(params):
        z = h @ W + b
        if i < len(params) - 1:
            h = relu(z)
            hiddens.append(h)
        else:
            return z, hiddens


# ---------------------------------------------------------------- Part B

def softmax_cross_entropy(logits, y):
    N = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))   # log-softmax
    loss = -logp[np.arange(N), y].mean()
    d = np.exp(logp)
    d[np.arange(N), y] -= 1.0
    return loss, d / N


def init_2layer(D, H, C, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(2.0 / D), (D, H))
    W2 = rng.normal(0.0, np.sqrt(2.0 / H), (H, C))
    return {"W1": W1, "b1": np.zeros(H), "W2": W2, "b2": np.zeros(C)}


def forward_2layer(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    logits = a1 @ params["W2"] + params["b2"]
    return logits, {"X": X, "z1": z1, "a1": a1}


def backward_2layer(params, cache, dlogits):
    X, z1, a1 = cache["X"], cache["z1"], cache["a1"]
    dW2 = a1.T @ dlogits
    db2 = dlogits.sum(axis=0)          # b2 was broadcast over rows -> sum over rows
    da1 = dlogits @ params["W2"].T
    dz1 = da1 * relu_grad(z1)          # ReLU's Jacobian is diagonal: a mask
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train_2layer(X, y, H=100, lr=1.0, steps=3000, reg=1e-3, seed=0):
    C = int(y.max()) + 1
    params = init_2layer(X.shape[1], H, C, seed)
    losses = []
    for _ in range(steps):
        logits, cache = forward_2layer(params, X)
        ce, dlogits = softmax_cross_entropy(logits, y)
        losses.append(ce + 0.5 * reg * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2)))
        g = backward_2layer(params, cache, dlogits)
        g["W1"] += reg * params["W1"]
        g["W2"] += reg * params["W2"]
        for k in params:
            params[k] -= lr * g[k]
    return params, losses


def accuracy(params, X, y):
    logits, _ = forward_2layer(params, X)
    return float(np.mean(logits.argmax(axis=1) == y))


# ---------------------------------------------------------------- Part C

def relu_interpolant(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    slopes = np.diff(ys) / np.diff(xs)                 # s_0 .. s_{n-1}
    a = np.concatenate([[slopes[0]], np.diff(slopes)])  # how much each hinge bends
    w = np.ones(len(slopes))
    c = -xs[:-1]                                        # hinge k sits at x_k
    return w, c, a, float(ys[0])


def relu_net_eval(w, c, a, d, x):
    x = np.asarray(x, float)
    return relu(x[..., None] * w + c) @ a + d
