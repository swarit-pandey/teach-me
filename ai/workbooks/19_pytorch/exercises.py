"""Workbook 19 — Moving to PyTorch.

Rules of the game
  * From here on: PyTorch. You may use torch, torch.nn.functional (F) and torch.optim,
    but NOT nn.Linear or nn.MLP-style shortcuts where the exercise asks you to build
    the thing (MyLinear, MLP). Everything else in torch is fair game.
  * Part A: shapes and autograd basics — broadcasting rules by hand, gradients via
    .backward().
  * Part B: nn.Module / nn.Parameter, a Dataset, the canonical training loop, and
    lesson 16's spiral classifier rebuilt in ~15 lines.
  * Part C: a custom torch.autograd.Function with a hand-written backward, verified
    by torch.autograd.gradcheck (exactly what workbook 17 did with numerical grads).
  * helpers.py provides make_spirals and make_blobs (numpy).
  * Run   ./course check 19   to grade.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from helpers import make_spirals, make_blobs  # noqa: F401


# ----------------------------------------------------------------------------
# Part A — shapes & autograd
# ----------------------------------------------------------------------------

def broadcast_shape(a, b):
    """Shape that broadcasting two tensors of shapes a and b produces (tuples of ints).

    Rules (NumPy and PyTorch share them, lesson 12): align the shapes on the RIGHT,
    padding the shorter one with 1s on the left; each pair of sizes must be equal or
    contain a 1 (which stretches). Raise ValueError if some pair is incompatible.
        broadcast_shape((8, 1, 6), (7, 1)) == (8, 7, 6)
    Don't call torch/numpy broadcasting helpers — the tests compare against them.
    """
    raise NotImplementedError


def grad_of(f, x):
    """Gradient of a scalar function f at the point x, using torch autograd.

    f takes a torch tensor and returns a 0-dim tensor; x is a numpy array.
    Return df/dx as a numpy array of x's shape (float64).
    Hint: torch.tensor(x, dtype=torch.float64, requires_grad=True), call f,
    .backward(), read .grad, then .detach().numpy().
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — modules, data, the training loop
# ----------------------------------------------------------------------------

class MyLinear(nn.Module):
    """y = x @ W^T + b  — the same convention (and init) as nn.Linear.

    weight: nn.Parameter of shape (out_features, in_features)
    bias:   nn.Parameter of shape (out_features,), or None if bias=False
    Both initialized Uniform(-k, k) with k = 1/sqrt(in_features).
    Hint: torch.empty(shape).uniform_(-k, k) wrapped in nn.Parameter.
    Remember super().__init__() first.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class MLP(nn.Module):
    """MLP(sizes=[d0, d1, ..., dL]) built from MyLinear layers, ReLU between layers
    (not after the last). Store the layers in an nn.ModuleList so that
    .parameters() finds them (a plain list would not!)."""

    def __init__(self, sizes):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def count_parameters(model):
    """Total number of trainable scalars. Hint: p.numel() over model.parameters()."""
    raise NotImplementedError


class BlobDataset(Dataset):
    """A torch Dataset over helpers.make_blobs(n_per_class, seed=seed).

    __len__ returns the number of examples; __getitem__(i) returns (x_i, y_i) as
    tensors: x_i float32 of shape (2,), y_i an int64 scalar. A DataLoader then
    stacks these into batches for you.
    """

    def __init__(self, n_per_class=100, seed=0):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


def train(model, loader, epochs=10, lr=1e-2):
    """The canonical training loop with torch.optim.Adam and F.cross_entropy.

    For each epoch, for each (xb, yb) batch:  forward -> loss -> zero_grad ->
    backward -> step.  Call model.train() at the start of each epoch.
    Return a list with the mean training loss of each epoch
    (average over examples, i.e. weight each batch's loss by its size).
    """
    raise NotImplementedError


def accuracy(model, X, y):
    """Fraction of rows of X whose argmax logit equals y, as a Python float.
    Use model.eval() and torch.no_grad() — no graph is needed to evaluate."""
    raise NotImplementedError


def train_spiral_classifier(seed=0, steps=1500):
    """Lesson 16's spiral classifier in PyTorch.

    torch.manual_seed(seed); data = make_spirals(100, 3, seed=0) as tensors
    (float32 X, int64 y); model = MLP([2, 100, 3]); Adam(lr=1e-2); `steps`
    full-batch steps of cross-entropy. Return (model, training accuracy).
    Compare with the ~40 lines of hand-written backward in workbook 16.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — a custom autograd.Function
# ----------------------------------------------------------------------------

class ScaledTanh(torch.autograd.Function):
    """y = tanh(a * x) for a tensor x and a 0-dim tensor a (a learnable "gain").

    forward(ctx, x, a): compute y, ctx.save_for_backward(...) what you need.
    backward(ctx, grad_out): return (grad_x, grad_a) — one gradient per forward
    input, each with the SAME SHAPE as that input.
        d y / d x = a (1 - y^2)          d y / d a = x (1 - y^2)
    Hint: a was broadcast to every element of x, so its gradient is a SUM over
    all elements (lesson 17, "broadcast forward = sum backward").
    """

    @staticmethod
    def forward(ctx, x, a):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


def scaled_tanh(x, a):
    """Convenience wrapper: ScaledTanh.apply(x, a)."""
    raise NotImplementedError
