"""Workbook 17 — reference solutions. Try first; read after."""
import math
import random

import numpy as np


# ---------------------------------------------------------------- Part A

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad          # local gradient 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) - self

    def __rtruediv__(self, other):
        return Value(other) / self

    def backward(self):
        order = topo_order(self)
        self.grad = 1.0
        for node in reversed(order):
            node._backward()


def topo_order(root):
    order, visited = [], set()
    stack = [(root, False)]           # (node, children_done?)
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)        # post-order: after all children
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for child in node._prev:
            if id(child) not in visited:
                stack.append((child, False))
    return order


def numerical_grad(f, xs, h=1e-6):
    g = []
    for i in range(len(xs)):
        xp, xm = list(xs), list(xs)
        xp[i] += h
        xm[i] -= h
        g.append((f(xp) - f(xm)) / (2 * h))
    return g


# ---------------------------------------------------------------- Part B

class Neuron:
    def __init__(self, n_in, rng, act="tanh"):
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(rng.uniform(-1, 1))
        self.act = act

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.act == "tanh":
            return z.tanh()
        if self.act == "relu":
            return z.relu()
        return z

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, rng, act="tanh"):
        self.neurons = [Neuron(n_in, rng, act) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, n_outs, seed=0):
        rng = random.Random(seed)
        sizes = [n_in] + list(n_outs)
        self.layers = [Layer(sizes[i], sizes[i + 1], rng, act="tanh" if i < len(n_outs) - 1 else "linear")
                       for i in range(len(n_outs))]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(x, Value) and i < len(self.layers) - 1:
                x = [x]                      # a width-1 hidden layer still feeds a list
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


def train(model, xs, ys, steps=100, lr=0.05):
    losses = []
    for _ in range(steps):
        preds = [model(x) for x in xs]
        loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), Value(0.0)) / len(xs)
        losses.append(loss.data)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
    return losses


# ---------------------------------------------------------------- Part C

def unbroadcast(grad, shape):
    while grad.ndim > len(shape):            # leading axes that were added
        grad = grad.sum(axis=0)
    for ax, n in enumerate(shape):           # axes that were stretched from 1
        if n == 1 and grad.shape[ax] != 1:
            grad = grad.sum(axis=ax, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T     # lesson 07: dL/dA = dL/dC B^T
            other.grad += self.data.T @ out.grad     #            dL/dB = A^T dL/dC
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad   # broadcast the scalar back
        out._backward = _backward
        return out

    def backward(self):
        assert self.data.ndim == 0, "backward() needs a scalar output"
        order = topo_order(self)
        self.grad = np.array(1.0)
        for node in reversed(order):
            node._backward()
