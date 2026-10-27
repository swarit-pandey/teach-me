"""Workbook 18 — reference solutions. Try first; read after."""
import math

import numpy as np

from helpers import rosenbrock, rosenbrock_grad, quadratic  # noqa: F401


# ---------------------------------------------------------------- Part A: optimizers

class SGD:
    def __init__(self, params, lr=0.01):
        self.params, self.lr = params, lr

    def step(self, grads):
        for p, g in zip(self.params, grads):
            p -= self.lr * g                       # in place: the caller's arrays move


class Momentum:
    def __init__(self, params, lr=0.01, beta=0.9):
        self.params, self.lr, self.beta = params, lr, beta
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for p, g, v in zip(self.params, grads, self.v):
            v *= self.beta
            v += g                                 # v = beta v + g  (a decaying sum of past grads)
            p -= self.lr * v


class RMSProp:
    def __init__(self, params, lr=0.01, beta=0.99, eps=1e-8):
        self.params, self.lr, self.beta, self.eps = params, lr, beta, eps
        self.s = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for p, g, s in zip(self.params, grads, self.s):
            s *= self.beta
            s += (1 - self.beta) * g * g           # running mean of g^2, per coordinate
            p -= self.lr * g / (np.sqrt(s) + self.eps)


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, beta1, beta2, eps
        self.bias_correction = bias_correction
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            if self.bias_correction:
                # m and v start at 0, so early on they underestimate E[g], E[g^2]
                # by exactly the factor (1 - beta^t). Undo it.
                mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            else:
                mh, vh = m, v
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def optimize(opt_cls, grad_fn, x0, steps, **kw):
    x = np.array(x0, dtype=float)
    opt = opt_cls([x], **kw)
    traj = [x.copy()]
    for _ in range(steps):
        opt.step([grad_fn(x)])
        traj.append(x.copy())
    return np.array(traj)


def warmup_cosine(step, total_steps, warmup_steps, lr_max, lr_min=0.0):
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------- Part B: init

def xavier_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, math.sqrt(2.0 / (fan_in + fan_out)), (fan_in, fan_out))


def he_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, math.sqrt(2.0 / fan_in), (fan_in, fan_out))


def _act(name):
    return {"tanh": np.tanh, "relu": lambda z: np.maximum(z, 0.0), "linear": lambda z: z}[name]


def _init(name, n, rng):
    if name == "small":
        return rng.normal(0.0, 0.01, (n, n))
    if name == "large":
        return rng.normal(0.0, 1.0, (n, n))
    if name == "xavier":
        return xavier_normal(n, n, rng)
    if name == "he":
        return he_normal(n, n, rng)
    raise ValueError(name)


def activation_stds(depth, width, init, act, n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    f = _act(act)
    h = rng.normal(size=(n_samples, width))
    stds = []
    for _ in range(depth):
        h = f(h @ _init(init, width, rng))
        stds.append(float(h.std()))
    return stds


# ---------------------------------------------------------------- Part B: normalization

def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)            # biased (divide by D), like PyTorch
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return gamma * xhat + beta, (xhat, rstd, gamma)


def layernorm_backward(dy, cache):
    xhat, rstd, gamma = cache
    lead = tuple(range(dy.ndim - 1))
    dgamma = (dy * xhat).sum(axis=lead)
    dbeta = dy.sum(axis=lead)
    g = dy * gamma                                   # dL/dxhat
    # xhat = (x - mean) * rstd. Differentiating through mean and var (each row
    # independently) gives the compact form
    #     dx = rstd * (g - mean(g) - xhat * mean(g * xhat))
    dx = rstd * (g - g.mean(axis=-1, keepdims=True) - xhat * (g * xhat).mean(axis=-1, keepdims=True))
    return dx, dgamma, dbeta


def batchnorm_forward_train(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


# ---------------------------------------------------------------- Part C: residual highways

def input_grad_norm(depth, width, residual, seed=0, n_samples=64, scale=1.0):
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0.0, scale / math.sqrt(width), (width, width)) for _ in range(depth)]
    x = rng.normal(size=(n_samples, width))
    hs, h = [], x
    for W in Ws:                                     # forward, remembering inputs
        hs.append(h)
        f = np.tanh(h @ W)
        h = h + f if residual else f
    g = np.ones_like(h)                              # dL/dh for L = sum(h)
    out_norm = np.linalg.norm(g)
    for W, hin in zip(reversed(Ws), reversed(hs)):
        t = np.tanh(hin @ W)
        through_f = ((1 - t * t) * g) @ W.T          # Jacobian-vector product of tanh(h W)
        g = g + through_f if residual else through_f # the residual adds an identity path
    return float(np.linalg.norm(g) / out_norm)
