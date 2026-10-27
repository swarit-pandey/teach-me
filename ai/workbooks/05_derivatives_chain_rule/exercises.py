"""Workbook 05 — Derivatives & the chain rule.

Rules of the game
  * Part A: plain Python + the `math` module only. A "function" is any Python
    callable float -> float. A "differentiable function" is a PAIR (f, df) of
    callables: the function and its derivative.
  * Part B: NumPy. Activation functions must work elementwise on arrays of any
    shape (and on plain floats), with no Python loops.
  * Part C (stretch): second derivatives and Newton's method for minimization —
    a preview of the Hessian in lesson 06.
  * Run   ./course check 05   to grade. Stuck > 20 min? Read the Hint, then
    peek at solutions.py, close it, and re-write from memory.
"""
import math
import numpy as np


# ----------------------------------------------------------------------------
# Part A — numerical derivatives (plain Python)
# ----------------------------------------------------------------------------

def forward_diff(f, x, h=1e-6):
    """Forward-difference estimate of f'(x):   (f(x+h) - f(x)) / h.

    Truncation error is O(h) (lesson §7: Taylor-expand f(x+h)).
    """
    raise NotImplementedError


def central_diff(f, x, h=1e-5):
    """Central-difference estimate of f'(x):   (f(x+h) - f(x-h)) / (2h).

    Truncation error is O(h^2): the f'' terms of the two Taylor expansions cancel.
    """
    raise NotImplementedError


def error_study(f, df, x, hs, method="central"):
    """Absolute error |estimate - df(x)| for every step size in `hs`.

    method is "forward" or "central" (raise ValueError otherwise).
    Returns a list of floats, same length and order as hs.
    """
    raise NotImplementedError


def best_h(f, df, x, method="central"):
    """The step size h = 10**k, for k in -16.0, -15.9, ..., -0.1, 0.0 (161 values),
    that gives the smallest absolute error for the chosen method.

    You should find ~1e-8 for forward and ~1e-5 for central (lesson §7 explains
    why: truncation error falls with h, floating-point rounding error grows like 1/h).
    Hint: build the grid as [10 ** (k / 10) for k in range(-160, 1)].
    """
    raise NotImplementedError


def compose(outer, inner):
    """Chain rule as code. outer = (f, df), inner = (g, dg).

    Return the pair (h, dh) where h(x) = f(g(x)) and dh(x) = df(g(x)) * dg(x).
    Both returned items must be callables.
    """
    raise NotImplementedError


def compose_all(pairs):
    """Compose a list of (f, df) pairs, applied LEFT TO RIGHT:
    pairs = [p1, p2, p3] means x -> p3(p2(p1(x))).

    Return the (h, dh) pair. An empty list is the identity (x, 1).
    Hint: fold with compose(). Watch the order — compose(outer, inner).
    """
    raise NotImplementedError


def newton(f, df, x0, tol=1e-12, max_iter=50):
    """Newton's method for a root of f:   x_{n+1} = x_n - f(x_n) / df(x_n).

    Stop when |f(x_n)| <= tol (check BEFORE taking a step, so a starting point that
    is already a root returns immediately). Returns (root, history) where history
    is the list of iterates [x0, x1, ..., root] (so history[-1] == root).

    Raise ValueError if df(x_n) == 0 (the tangent is flat: no intersection).
    Raise RuntimeError if not converged after max_iter steps.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — activation functions and their derivatives (NumPy, elementwise)
# ----------------------------------------------------------------------------

def sigmoid(z):
    """sigma(z) = 1 / (1 + e^{-z}), elementwise.

    Must be numerically stable: sigmoid(-1000) == 0.0 and sigmoid(1000) == 1.0
    WITHOUT any overflow warning. Hint: for z < 0 use the equivalent form
    e^{z} / (1 + e^{z}); np.where evaluates both branches, so compute
    e = np.exp(-np.abs(z)) once and build both forms from it.
    """
    raise NotImplementedError


def sigmoid_prime(z):
    """d sigma / dz = sigma(z) * (1 - sigma(z)), elementwise."""
    raise NotImplementedError


def tanh_prime(z):
    """d tanh / dz = 1 - tanh(z)^2, elementwise (np.tanh is fine)."""
    raise NotImplementedError


def relu(z):
    """max(0, z), elementwise. Return floats."""
    raise NotImplementedError


def relu_prime(z):
    """1.0 where z > 0, else 0.0 (convention: the derivative at exactly 0 is 0)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch: second derivatives and Newton for minimization
# ----------------------------------------------------------------------------

def second_diff(f, x, h=1e-4):
    """Central estimate of f''(x):  (f(x+h) - 2 f(x) + f(x-h)) / h^2.

    Hint: add the Taylor expansions of f(x+h) and f(x-h); the odd terms cancel.
    """
    raise NotImplementedError


def newton_minimize(df, d2f, x0, tol=1e-10, max_iter=50):
    """Find a stationary point of f by running Newton's root-finder on f'.

    The update is x <- x - f'(x) / f''(x): jump to the minimum of the local
    quadratic (parabola) approximation. Returns (x_star, history) like newton().
    """
    raise NotImplementedError
