"""Workbook 18 — Optimizers, initialization & normalization.

Rules of the game
  * NumPy only. Optimizers hold a list of numpy arrays (the parameters) and update
    them IN PLACE (p -= ..., never p = ...), exactly like torch.optim does, so the
    model that owns those arrays sees the change.
  * Part A: SGD, Momentum, RMSProp, Adam (with bias correction), a driver, and the
    warmup + cosine learning-rate schedule.
  * Part B: Xavier/He init and an experiment measuring activation scale through a
    deep net; LayerNorm forward AND backward; BatchNorm forward (to compare).
  * Part C (stretch): residual connections as gradient highways.
  * helpers.py provides rosenbrock, rosenbrock_grad, quadratic and numerical_grad.
  * Run   ./course check 18   to grade.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import math

import numpy as np

from helpers import rosenbrock, rosenbrock_grad, quadratic  # noqa: F401


# ----------------------------------------------------------------------------
# Part A — optimizers.  Each: __init__(params, ...hyperparameters) and step(grads)
# where grads is a list of arrays matching params.
# ----------------------------------------------------------------------------

class SGD:
    """p <- p - lr * g"""

    def __init__(self, params, lr=0.01):
        raise NotImplementedError

    def step(self, grads):
        raise NotImplementedError


class Momentum:
    """Heavy-ball momentum (PyTorch's convention):
        v <- beta * v + g
        p <- p - lr * v
    v starts at zeros, one buffer per parameter."""

    def __init__(self, params, lr=0.01, beta=0.9):
        raise NotImplementedError

    def step(self, grads):
        raise NotImplementedError


class RMSProp:
    """Per-coordinate step size from a running mean of squared gradients:
        s <- beta * s + (1 - beta) * g^2
        p <- p - lr * g / (sqrt(s) + eps)"""

    def __init__(self, params, lr=0.01, beta=0.99, eps=1e-8):
        raise NotImplementedError

    def step(self, grads):
        raise NotImplementedError


class Adam:
    """Momentum + RMSProp + bias correction. With t = number of steps taken so far
    (t = 1 on the first call):
        m <- b1 m + (1 - b1) g            v <- b2 v + (1 - b2) g^2
        m_hat = m / (1 - b1^t)            v_hat = v / (1 - b2^t)
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    If bias_correction=False, use m and v directly (to see why correction matters).
    """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True):
        raise NotImplementedError

    def step(self, grads):
        raise NotImplementedError


def optimize(opt_cls, grad_fn, x0, steps, **kw):
    """Run an optimizer on a function of one parameter vector.

    Make x = a float copy of x0, opt = opt_cls([x], **kw), then `steps` times:
    opt.step([grad_fn(x)]).  Return the trajectory as an array of shape
    (steps + 1, len(x0)) whose first row is x0.
    """
    raise NotImplementedError


def warmup_cosine(step, total_steps, warmup_steps, lr_max, lr_min=0.0):
    """Learning rate at integer `step` in [0, total_steps).

    Warmup (step < warmup_steps): linear ramp  lr_max * (step + 1) / warmup_steps.
    Then cosine decay from lr_max to lr_min:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — initialization
# ----------------------------------------------------------------------------

def xavier_normal(fan_in, fan_out, rng):
    """(fan_in, fan_out) matrix ~ Normal(0, std = sqrt(2 / (fan_in + fan_out)))."""
    raise NotImplementedError


def he_normal(fan_in, fan_out, rng):
    """(fan_in, fan_out) matrix ~ Normal(0, std = sqrt(2 / fan_in))."""
    raise NotImplementedError


def activation_stds(depth, width, init, act, n_samples=500, seed=0):
    """Push a batch of unit Gaussians through `depth` layers  h <- act(h @ W)
    (no biases, all layers width x width) and return the list of h.std() after
    each layer (length depth).

    init in {"small": std 0.01, "large": std 1.0, "xavier", "he"};
    act in {"tanh", "relu", "linear"}.
    Use rng = np.random.default_rng(seed): draw the input batch
    rng.normal(size=(n_samples, width)) first, then each layer's W in order.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — normalization
# ----------------------------------------------------------------------------

def layernorm_forward(x, gamma, beta, eps=1e-5):
    """LayerNorm over the LAST axis (each example / token normalized on its own).

    x: (..., D); gamma, beta: (D,).
        mu = mean over last axis; var = mean of (x - mu)^2 (biased, like PyTorch)
        xhat = (x - mu) / sqrt(var + eps);   y = gamma * xhat + beta
    Return (y, cache).
    """
    raise NotImplementedError


def layernorm_backward(dy, cache):
    """Given dy = dL/dy (same shape as x), return (dx, dgamma, dbeta).

    dgamma, dbeta have shape (D,) — sum over every leading axis.
    Hint: with g = dy * gamma (the gradient w.r.t. xhat), working through the
    mean and variance (problem 6 in the lesson) gives, per row,
        dx = (g - mean(g) - xhat * mean(g * xhat)) / sqrt(var + eps).
    Check yourself against helpers.numerical_grad.
    """
    raise NotImplementedError


def batchnorm_forward_train(x, gamma, beta, eps=1e-5):
    """BatchNorm (training mode) for x of shape (N, D): normalize each FEATURE over
    the BATCH axis 0, then scale/shift. Return y only."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch: residual connections as gradient highways
# ----------------------------------------------------------------------------

def input_grad_norm(depth, width, residual, seed=0, n_samples=64, scale=1.0):
    """Gradient size at the INPUT of a deep tanh net, relative to the output.

    Layers l = 1..depth, each W_l ~ Normal(0, std = scale / sqrt(width)), drawn in
    order from rng = np.random.default_rng(seed) BEFORE the input
    x = rng.normal(size=(n_samples, width)).
        plain:     h <- tanh(h @ W)
        residual:  h <- h + tanh(h @ W)
    With L = sum(h_out), backpropagate by hand (no autograd) and return
    ||dL/dx|| / ||dL/dh_out||   (Frobenius norms).
    Hint: backward through tanh(h W) maps g to ((1 - t^2) * g) @ W.T; the residual
    branch adds g itself (the identity's Jacobian is I).
    """
    raise NotImplementedError
