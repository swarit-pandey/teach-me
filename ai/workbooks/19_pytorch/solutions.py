"""Workbook 19 — reference solutions. Try first; read after."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from helpers import make_spirals, make_blobs  # noqa: F401


# ---------------------------------------------------------------- Part A

def broadcast_shape(a, b):
    a, b = tuple(a), tuple(b)
    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + a                # align on the RIGHT: pad with 1s on the left
    b = (1,) * (n - len(b)) + b
    out = []
    for x, y in zip(a, b):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"cannot broadcast {a} with {b}")
    return tuple(out)


def grad_of(f, x):
    t = torch.tensor(x, dtype=torch.float64, requires_grad=True)   # a leaf that records ops
    y = f(t)
    y.backward()
    return t.grad.detach().numpy()


# ---------------------------------------------------------------- Part B

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        k = 1.0 / math.sqrt(in_features)
        # nn.Parameter = a tensor with requires_grad=True that the Module registers,
        # so it shows up in .parameters() and gets moved by the optimizer.
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-k, k))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-k, k)) if bias else None

    def forward(self, x):
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        # A plain Python list would hide the layers from .parameters(); ModuleList registers them.
        self.layers = nn.ModuleList(MyLinear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BlobDataset(Dataset):
    def __init__(self, n_per_class=100, seed=0):
        X, y = make_blobs(n_per_class, seed=seed)
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def train(model, loader, epochs=10, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            logits = model(xb)                     # 1. forward
            loss = F.cross_entropy(logits, yb)     # 2. loss
            opt.zero_grad()                        # 3. clear old grads (they accumulate!)
            loss.backward()                        # 4. backprop
            opt.step()                             # 5. update
            total += loss.item() * len(yb)
            n += len(yb)
        history.append(total / n)
    return history


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    return (model(X).argmax(dim=1) == y).float().mean().item()


def train_spiral_classifier(seed=0, steps=1500):
    torch.manual_seed(seed)
    X, y = make_spirals(100, 3, seed=0)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    model = MLP([2, 100, 3])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, accuracy(model, X, y)


# ---------------------------------------------------------------- Part C

class ScaledTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        t = torch.tanh(a * x)
        ctx.save_for_backward(x, a, t)          # whatever backward needs
        return t

    @staticmethod
    def backward(ctx, grad_out):
        x, a, t = ctx.saved_tensors
        local = 1 - t * t                       # tanh'(a x)
        grad_x = grad_out * local * a if ctx.needs_input_grad[0] else None
        # a is a scalar that was broadcast over every element of x: sum (lesson 17 §5)
        grad_a = (grad_out * local * x).sum() if ctx.needs_input_grad[1] else None
        return grad_x, grad_a


def scaled_tanh(x, a):
    return ScaledTanh.apply(x, a)
