"""Workbook 17 — Backprop & building micro-autograd.

Rules of the game
  * Part A: a scalar autograd engine. `Value` wraps one float, remembers which
    Values produced it and how, and can backpropagate. Plain Python + math only.
  * Part B: a tiny neural-network library (Neuron, Layer, MLP) on top of Value,
    trained on a small dataset — every gradient flows through YOUR engine.
  * Part C (stretch): the same idea for whole arrays — a numpy `Tensor` with
    matmul, add (with broadcasting), relu, sum. This is what PyTorch is (lesson 19).
  * Run   ./course check 17   to grade.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.

The contract for every operation (lesson §4): when you build out = f(a, b), give
`out` a `_backward` closure that ADDS (+=, never =) the local-gradient-times-
upstream-gradient into a.grad and b.grad:

        a.grad += (d out / d a) * out.grad

"+=" is what makes a node that's used twice get the sum of both paths (§5).
"""
import math
import random

import numpy as np


# ----------------------------------------------------------------------------
# Part A — Value: a scalar that knows its own history
# ----------------------------------------------------------------------------

class Value:
    """A scalar node in a computation graph.

    Attributes (already set up for you):
      data      the float value
      grad      d(final output)/d(this node); 0.0 until backward() runs
      _prev     tuple of the Values this one was computed from (its children)
      _op       a short string naming the op, for debugging ('+', '*', 'tanh', ...)
      _backward a function of no arguments that pushes self.grad to the children
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # --- the core ops: each returns a NEW Value with a _backward closure -------
    def __add__(self, other):
        """self + other. `other` may be a plain number: wrap it with Value(other).
        Local gradients: d/da (a+b) = 1, d/db (a+b) = 1."""
        raise NotImplementedError

    def __mul__(self, other):
        """self * other. Local gradients: d/da (ab) = b, d/db (ab) = a."""
        raise NotImplementedError

    def __pow__(self, k):
        """self ** k for a constant int/float k (not a Value).
        Local gradient: k * a**(k-1)."""
        raise NotImplementedError

    def exp(self):
        """e**self. Local gradient: e**a (= the output itself)."""
        raise NotImplementedError

    def log(self):
        """Natural log (self.data > 0). Local gradient: 1/a."""
        raise NotImplementedError

    def tanh(self):
        """tanh(self). Local gradient: 1 - tanh(a)**2."""
        raise NotImplementedError

    def relu(self):
        """max(0, self). Local gradient: 1 if a > 0 else 0."""
        raise NotImplementedError

    # --- derived ops: build these from the core ops, no new _backward needed ---
    def __neg__(self):
        """-self.   Hint: self * -1."""
        raise NotImplementedError

    def __sub__(self, other):
        """self - other.   Hint: self + (-other)."""
        raise NotImplementedError

    def __truediv__(self, other):
        """self / other.   Hint: self * other**-1."""
        raise NotImplementedError

    # Python calls these when the LEFT operand is a plain number: 2 + v, 2 * v, ...
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        """other - self, where other is a number."""
        raise NotImplementedError

    def __rtruediv__(self, other):
        """other / self, where other is a number."""
        raise NotImplementedError

    # --- backprop ---------------------------------------------------------------
    def backward(self):
        """Fill .grad of every node that self depends on with d self / d node.

        1. Build a topological order of the graph (topo_order(self)).
        2. Set self.grad = 1.0 (d self / d self).
        3. Walk the order in REVERSE, calling each node's _backward().
        Note: grads accumulate, so call zero_grad on parameters between
        training steps (Part B) — exactly like PyTorch.
        """
        raise NotImplementedError


def topo_order(root):
    """Return a list of every Value reachable from root (including root), ordered so
    that each node appears AFTER all of its children (_prev).  Each node exactly once.

    Hint: depth-first search with a `visited` set; append a node after visiting
    all of its children (post-order).  Use an explicit stack if you want to avoid
    Python's recursion limit on deep graphs (a test builds a 5000-node chain).
    """
    raise NotImplementedError


def numerical_grad(f, xs, h=1e-6):
    """Central-difference gradient of f (a function of a list of floats returning
    a float) at the point xs. Return a list of partial derivatives.
    This is your ground truth for testing the engine (lesson 06)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — a neural-net library on Value
# ----------------------------------------------------------------------------

class Neuron:
    """n_in weights + 1 bias, all Values initialized uniform(-1, 1) from `rng`
    (a random.Random). Calling it on a list of n_in inputs (Values or floats)
    returns  act(w·x + b)  where act is 'tanh', 'relu' or 'linear'."""

    def __init__(self, n_in, rng, act="tanh"):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        """List of the Value parameters: the weights followed by the bias."""
        raise NotImplementedError


class Layer:
    """n_out Neurons that all see the same input. __call__ returns a list of n_out
    Values (or a single Value when n_out == 1)."""

    def __init__(self, n_in, n_out, rng, act="tanh"):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


class MLP:
    """MLP(n_in, [h1, h2, ..., n_out], seed): tanh on hidden layers, 'linear' on the
    last. Build every Neuron from one random.Random(seed), layer by layer."""

    def __init__(self, n_in, n_outs, seed=0):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def zero_grad(self):
        """Set .grad = 0.0 on every parameter."""
        raise NotImplementedError


def train(model, xs, ys, steps=100, lr=0.05):
    """Fit model to (xs, ys) by gradient descent on the mean squared error
    loss = mean_i (model(xs[i]) - ys[i])**2.

    Each step: forward over all examples, build the loss Value, zero_grad,
    loss.backward(), then p.data -= lr * p.grad for every parameter.
    Return the list of loss floats, one per step (measured before the update).
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch: a numpy Tensor with autograd
# ----------------------------------------------------------------------------

def unbroadcast(grad, shape):
    """Reduce `grad` to `shape` by summing over the axes that broadcasting expanded.

    If a (3,) bias was broadcast to (N, 3) in the forward pass, its gradient is the
    (N, 3) upstream gradient summed over axis 0. In general: sum away leading extra
    axes, then sum (keepdims) every axis where shape has size 1.
    """
    raise NotImplementedError


class Tensor:
    """Like Value but `data` is a float64 numpy array and `grad` an array of the same
    shape (zeros until backward). Support:
        a + b    (numpy broadcasting; use unbroadcast in backward)
        a * b    (element-wise, broadcasting)
        a @ b    (2-D matmul: dA = dOut @ B^T, dB = A^T @ dOut)
        a.relu()
        a.sum()  (to a 0-d scalar Tensor)
    backward() requires self to be a scalar (0-d) and seeds its grad with 1.0.
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def relu(self):
        raise NotImplementedError

    def sum(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
