import math
import random

import numpy as np
import pytest
from _impl import load

ex = load(__file__)
V = ex.Value


def ref_numgrad(f, xs, h=1e-6):
    out = []
    for i in range(len(xs)):
        xp, xm = list(xs), list(xs)
        xp[i] += h
        xm[i] -= h
        out.append((f(xp) - f(xm)) / (2 * h))
    return out


def grads_of(build, xs):
    """Run build() on Values, backprop, return (output float, list of input grads)."""
    vs = [V(x) for x in xs]
    out = build(vs)
    out.backward()
    return out.data, [v.grad for v in vs]


# ---- Part A ----
def test_add_mul_forward_and_backward():
    a, b = V(2.0), V(-3.0)
    c = a * b + a          # dc/da = b + 1 = -2, dc/db = a = 2
    assert c.data == -4.0
    c.backward()
    assert a.grad == pytest.approx(-2.0) and b.grad == pytest.approx(2.0)


def test_mixed_with_plain_numbers():
    a = V(3.0)
    y = 2 * a + 1 - a / 4 + 10 / a - 1     # uses radd/rmul/rsub/rtruediv
    assert y.data == pytest.approx(6 + 1 - 0.75 + 10 / 3 - 1)
    y.backward()
    assert a.grad == pytest.approx(2 - 0.25 - 10 / 9)
    z = 5 - a                             # __rsub__
    assert z.data == 2.0


@pytest.mark.parametrize("name,fv,ff,x0", [
    ("pow", lambda v: v[0] ** 3 + v[0] ** -0.5, lambda x: x[0] ** 3 + x[0] ** -0.5, [1.7]),
    ("exp", lambda v: (v[0] * 0.5).exp(), lambda x: math.exp(x[0] * 0.5), [0.8]),
    ("log", lambda v: (v[0] * v[0] + 1).log(), lambda x: math.log(x[0] ** 2 + 1), [-1.3]),
    ("tanh", lambda v: v[0].tanh(), lambda x: math.tanh(x[0]), [0.4]),
    ("relu+", lambda v: (v[0] * 2).relu(), lambda x: max(0, 2 * x[0]), [0.7]),
    ("relu-", lambda v: (v[0] * 2).relu(), lambda x: max(0, 2 * x[0]), [-0.7]),
    ("div", lambda v: v[0] / v[1], lambda x: x[0] / x[1], [1.5, -2.5]),
    ("sub", lambda v: v[0] - v[1] * v[1], lambda x: x[0] - x[1] ** 2, [1.5, -2.5]),
])
def test_each_op_matches_numerical_gradient(name, fv, ff, x0):
    out, g = grads_of(fv, x0)
    assert out == pytest.approx(ff(x0))
    assert g == pytest.approx(ref_numgrad(ff, x0), abs=1e-5)


def test_reused_node_accumulates():
    # b is used three times: y = b*b + b  -> dy/db = 2b + 1
    a = V(3.0)
    b = a * 2
    y = b * b + b
    y.backward()
    assert b.grad == pytest.approx(2 * 6 + 1)
    assert a.grad == pytest.approx(2 * (2 * 6 + 1))
    # x + x must give gradient 2, not 1 (the classic "=" instead of "+=" bug)
    x = V(1.0)
    s = x + x
    s.backward()
    assert x.grad == 2.0


def test_complicated_expression():
    def f(v):
        x, y, z = v
        h = (x * y + z).tanh()
        return (h * h + (x - z).exp() / (y * y + 1)).log() + (x * 3).relu()

    def fr(x):
        h = math.tanh(x[0] * x[1] + x[2])
        return math.log(h * h + math.exp(x[0] - x[2]) / (x[1] ** 2 + 1)) + max(0, 3 * x[0])

    x0 = [0.3, -1.2, 0.5]
    out, g = grads_of(f, x0)
    assert out == pytest.approx(fr(x0))
    assert g == pytest.approx(ref_numgrad(fr, x0), abs=1e-5)


def test_topo_order():
    a, b = V(1.0), V(2.0)
    c = a * b
    d = c + a
    e = d * c
    order = ex.topo_order(e)
    assert len(order) == len({id(n) for n in order}) == 5
    pos = {id(n): i for i, n in enumerate(order)}
    for n in order:
        for ch in n._prev:
            assert pos[id(ch)] < pos[id(n)]
    assert order[-1] is e


def test_deep_chain_no_recursion_error():
    x = V(0.5)
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.backward()
    assert x.grad == pytest.approx(1.0001 ** 5000, rel=1e-9)


def test_numerical_grad_helper():
    g = ex.numerical_grad(lambda v: v[0] ** 2 * v[1], [3.0, 2.0])
    assert g == pytest.approx([12.0, 9.0], abs=1e-5)


# ---- Part B ----
def test_neuron_and_parameters():
    n = ex.Neuron(3, random.Random(0), act="linear")
    ps = n.parameters()
    assert len(ps) == 4 and all(isinstance(p, V) for p in ps)
    assert all(-1 <= p.data <= 1 for p in ps)
    out = n([1.0, 2.0, 3.0])
    want = sum(w.data * x for w, x in zip(ps[:3], [1, 2, 3])) + ps[3].data
    assert out.data == pytest.approx(want)
    out.backward()
    assert [p.grad for p in ps] == pytest.approx([1, 2, 3, 1])


def test_mlp_shapes_and_param_count():
    m = ex.MLP(3, [4, 4, 1], seed=0)
    assert len(m.parameters()) == (3 * 4 + 4) + (4 * 4 + 4) + (4 + 1)
    y = m([1.0, -2.0, 0.5])
    assert isinstance(y, V)
    m2 = ex.MLP(2, [3, 2], seed=0)
    out = m2([0.1, 0.2])
    assert isinstance(out, list) and len(out) == 2
    # deterministic given the seed
    assert ex.MLP(3, [4, 1], seed=5)([1, 1, 1]).data == ex.MLP(3, [4, 1], seed=5)([1, 1, 1]).data


def test_mlp_gradient_matches_numerical():
    m = ex.MLP(2, [3, 1], seed=1)
    ps = m.parameters()
    x = [0.5, -1.5]

    def loss_at(vals):
        for p, v in zip(ps, vals):
            p.data = v
        return (m(x) - 0.3) ** 2

    base = [p.data for p in ps]
    L = loss_at(base)
    m.zero_grad()
    L.backward()
    got = [p.grad for p in ps]
    num = ref_numgrad(lambda vals: loss_at(vals).data, base)
    loss_at(base)
    assert got == pytest.approx(num, abs=1e-5)
    m.zero_grad()
    assert all(p.grad == 0.0 for p in m.parameters())


def test_train_tiny_dataset():
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    m = ex.MLP(3, [4, 4, 1], seed=0)
    losses = ex.train(m, xs, ys, steps=200, lr=0.05)
    assert len(losses) == 200
    assert losses[-1] < 0.01 and losses[-1] < losses[0]
    preds = [m(x).data for x in xs]
    assert all(np.sign(p) == y for p, y in zip(preds, ys))


# ---- Part C ----
def test_unbroadcast():
    g = np.ones((4, 3))
    assert np.array_equal(ex.unbroadcast(g, (3,)), [4, 4, 4])
    assert ex.unbroadcast(g, (4, 1)).shape == (4, 1)
    assert ex.unbroadcast(g, (1, 3)).tolist() == [[4, 4, 4]]
    assert ex.unbroadcast(g, (4, 3)).shape == (4, 3)
    assert ex.unbroadcast(np.ones((2, 4, 3)), ()).shape == ()


def test_tensor_mlp_gradients():
    rng = np.random.default_rng(0)
    Xd, W1d, b1d, W2d = rng.normal(size=(5, 3)), rng.normal(size=(3, 4)), rng.normal(size=4), rng.normal(size=(4, 2))
    cd = rng.normal(size=(5, 2))

    def forward(W1v, b1v, W2v):
        X, W1, b1, W2, c = ex.Tensor(Xd), ex.Tensor(W1v), ex.Tensor(b1v), ex.Tensor(W2v), ex.Tensor(cd)
        out = (((X @ W1 + b1).relu() @ W2) * c).sum()
        return out, (W1, b1, W2)

    out, (W1, b1, W2) = forward(W1d, b1d, W2d)
    assert out.data.shape == ()
    out.backward()
    for t, arr, k in [(W1, W1d, 0), (b1, b1d, 1), (W2, W2d, 2)]:
        num = np.zeros_like(arr)
        for i in np.ndindex(arr.shape):
            args = [W1d.copy(), b1d.copy(), W2d.copy()]
            args[k][i] += 1e-6
            fp = forward(*args)[0].data
            args[k][i] -= 2e-6
            fm = forward(*args)[0].data
            num[i] = (fp - fm) / 2e-6
        assert t.grad.shape == arr.shape
        assert np.allclose(t.grad, num, atol=1e-5)


def test_tensor_matches_value_engine():
    # the same tiny computation done with scalars and with tensors must agree
    a, b = V(1.5), V(-0.5)
    s = (a * b + a).relu() + a * a
    s.backward()
    A, B = ex.Tensor([1.5]), ex.Tensor([-0.5])
    t = ((A * B + A).relu() + A * A).sum()
    t.backward()
    assert t.data == pytest.approx(s.data)
    assert A.grad[0] == pytest.approx(a.grad) and B.grad[0] == pytest.approx(b.grad)
