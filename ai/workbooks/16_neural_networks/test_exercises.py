import numpy as np
import pytest
from _impl import load

ex = load(__file__)
from helpers import make_spirals, numerical_grad  # noqa: E402


# ---- Part A ----
def test_relu_and_grad():
    z = np.array([[-2.0, 0.0, 3.0], [1e-9, -1e-9, 5.0]])
    assert np.array_equal(ex.relu(z), [[0, 0, 3], [1e-9, 0, 5]])
    assert np.array_equal(ex.relu_grad(z), [[0, 0, 1], [1, 0, 1]])
    assert ex.relu(z).shape == z.shape


def test_sigmoid_stable_and_symmetric():
    z = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
    s = ex.sigmoid(z)
    assert np.all(np.isfinite(s))
    assert np.allclose(s, [0, 1 / (1 + np.exp(5)), 0.5, 1 / (1 + np.exp(-5)), 1])
    assert np.allclose(ex.sigmoid(z) + ex.sigmoid(-z), 1)   # sigma(-z) = 1 - sigma(z)


def test_tanh_grad_matches_numeric():
    z = np.linspace(-3, 3, 13)
    num = (np.tanh(z + 1e-6) - np.tanh(z - 1e-6)) / 2e-6
    assert np.allclose(ex.tanh_grad(z), num, atol=1e-8)
    assert ex.tanh_grad(np.array([0.0]))[0] == pytest.approx(1.0)


def test_collapse_linear():
    rng = np.random.default_rng(1)
    sizes = [4, 7, 5, 3]
    layers = [(rng.normal(size=(a, b)), rng.normal(size=b)) for a, b in zip(sizes[:-1], sizes[1:])]
    X = rng.normal(size=(10, 4))
    h = X
    for W, b in layers:
        h = h @ W + b
    W_eff, b_eff = ex.collapse_linear(layers)
    assert W_eff.shape == (4, 3) and b_eff.shape == (3,)
    assert np.allclose(X @ W_eff + b_eff, h)
    # with one layer it's the identity operation
    W1, b1 = ex.collapse_linear(layers[:1])
    assert np.allclose(W1, layers[0][0]) and np.allclose(b1, layers[0][1])


def test_count_params():
    assert ex.count_params([784, 128, 10]) == 101770
    assert ex.count_params([2, 1]) == 3
    assert ex.count_params([3, 4, 4, 2]) == (3 * 4 + 4) + (4 * 4 + 4) + (4 * 2 + 2)


def test_init_mlp():
    p = ex.init_mlp([400, 300, 10], seed=0)
    assert [W.shape for W, _ in p] == [(400, 300), (300, 10)]
    assert all(np.all(b == 0) for _, b in p)
    assert abs(p[0][0].std() - np.sqrt(2 / 400)) < 0.005
    assert abs(p[1][0].std() - np.sqrt(2 / 300)) < 0.01
    small = ex.init_mlp([400, 300], seed=0, scheme="small")
    assert abs(small[0][0].std() - 0.01) < 0.001
    assert np.array_equal(ex.init_mlp([5, 3], seed=3)[0][0], ex.init_mlp([5, 3], seed=3)[0][0])
    with pytest.raises(ValueError):
        ex.init_mlp([2, 2], scheme="nope")


def test_mlp_forward_shapes_and_batching():
    p = ex.init_mlp([3, 8, 6, 4], seed=2)
    X = np.random.default_rng(0).normal(size=(5, 3))
    logits, hiddens = ex.mlp_forward(p, X)
    assert logits.shape == (5, 4)
    assert [h.shape for h in hiddens] == [(5, 8), (5, 6)]
    assert all(np.all(h >= 0) for h in hiddens)
    # a batch is just many independent examples: row i of the batch output equals
    # the output for X[i] alone
    for i in range(5):
        li, _ = ex.mlp_forward(p, X[i:i + 1])
        assert np.allclose(li[0], logits[i])
    # the last layer is linear: logits may be negative
    assert (logits < 0).any()


# ---- Part B ----
def test_softmax_cross_entropy():
    logits = np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0], [1000.0, 0.0, -1000.0]])
    y = np.array([0, 2, 0])
    loss, d = ex.softmax_cross_entropy(logits, y)
    p0 = np.exp([2.0, 1.0, 0.1]); p0 /= p0.sum()
    want = (-np.log(p0[0]) + np.log(3) + 0.0) / 3
    assert loss == pytest.approx(want, rel=1e-9)
    assert np.allclose(d.sum(axis=1), 0)        # each row of softmax - onehot sums to 0
    L = logits[:2].copy()
    num = numerical_grad(lambda: ex.softmax_cross_entropy(L, y[:2])[0], L)
    assert np.allclose(ex.softmax_cross_entropy(L, y[:2])[1], num, atol=1e-7)


def test_backward_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3))
    y = rng.integers(0, 4, size=6)
    params = ex.init_2layer(3, 5, 4, seed=1)
    params["b1"] = rng.normal(size=5) * 0.1      # avoid kinks sitting exactly at 0

    def loss():
        return ex.softmax_cross_entropy(ex.forward_2layer(params, X)[0], y)[0]

    logits, cache = ex.forward_2layer(params, X)
    grads = ex.backward_2layer(params, cache, ex.softmax_cross_entropy(logits, y)[1])
    assert set(grads) == {"W1", "b1", "W2", "b2"}
    for k in params:
        assert grads[k].shape == params[k].shape
        num = numerical_grad(loss, params[k])
        assert np.allclose(grads[k], num, atol=1e-6), k


def test_train_spirals():
    X, y = make_spirals(100, 3, seed=0)
    params, losses = ex.train_2layer(X, y, H=100, lr=1.0, steps=3000, seed=0)
    assert len(losses) == 3000
    assert losses[-1] < 0.5 * losses[0]
    assert ex.accuracy(params, X, y) > 0.9


def test_accuracy():
    params = {"W1": np.eye(2), "b1": np.zeros(2), "W2": np.eye(2), "b2": np.zeros(2)}
    X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.5], [0.1, 0.0]])
    assert ex.accuracy(params, X, np.array([0, 1, 0, 1])) == 0.75


# ---- Part C ----
def test_relu_interpolant():
    xs = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    ys = np.array([1.0, -1.0, 0.5, 0.5, 2.0])
    w, c, a, d = ex.relu_interpolant(xs, ys)
    assert len(w) == len(c) == len(a) == 4
    grid = np.linspace(-2, 3, 201)
    assert np.allclose(ex.relu_net_eval(w, c, a, d, grid), np.interp(grid, xs, ys))
    # approximating a smooth curve gets better with more units
    f = np.sin
    errs = []
    for n in (4, 16, 64):
        k = np.linspace(-3, 3, n + 1)
        net = ex.relu_interpolant(k, f(k))
        g = np.linspace(-3, 3, 1001)
        errs.append(np.abs(ex.relu_net_eval(*net, g) - f(g)).max())
    assert errs[0] > errs[1] > errs[2] and errs[2] < 2e-3
