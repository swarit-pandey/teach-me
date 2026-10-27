import math

import numpy as np
import pytest
from _impl import load

ex = load(__file__)
from helpers import rosenbrock, rosenbrock_grad, quadratic, numerical_grad  # noqa: E402


# ---- Part A: optimizers ----
def test_sgd_updates_in_place():
    p = np.array([1.0, -2.0])
    q = np.array([[0.5]])
    opt = ex.SGD([p, q], lr=0.1)
    opt.step([np.array([2.0, 4.0]), np.array([[-1.0]])])
    assert np.allclose(p, [0.8, -2.4]) and np.allclose(q, [[0.6]])


def test_momentum_constant_gradient():
    # with a constant gradient g, v_k = g (1 + beta + ... + beta^(k-1))
    p = np.array([0.0])
    opt = ex.Momentum([p], lr=1.0, beta=0.5)
    xs = []
    for _ in range(3):
        opt.step([np.array([1.0])])
        xs.append(p[0])
    assert np.allclose(xs, [-1.0, -2.5, -4.25])


def test_momentum_beats_sgd_on_ravine():
    f, g = quadratic([1.0, 50.0])

    def steps_to(traj, tol=1e-3):
        n = np.linalg.norm(traj, axis=1)
        return int(np.argmax(n < tol)) if (n < tol).any() else 10 ** 9

    sgd = ex.optimize(ex.SGD, g, [1.0, 1.0], 1500, lr=0.01)
    mom = ex.optimize(ex.Momentum, g, [1.0, 1.0], 1500, lr=0.01, beta=0.9)
    assert sgd.shape == (1501, 2) and np.allclose(sgd[0], [1, 1])
    assert steps_to(mom) * 3 < steps_to(sgd) < 10 ** 9


def test_rmsprop_first_step_and_scale_invariance():
    p = np.array([1.0, 1.0])
    opt = ex.RMSProp([p], lr=0.01, beta=0.99)
    opt.step([np.array([1e-3, 1e3])])
    # s = 0.01 g^2  ->  step = lr * g / (0.1 |g|) = 10 lr, whatever the size of g
    assert np.allclose(p, [1 - 0.1, 1 - 0.1], atol=1e-6)
    # rescaling the loss (hence every gradient) by 1000 leaves the path unchanged
    _, g = quadratic([1.0, 20.0])
    a = ex.optimize(ex.RMSProp, g, [1.0, -1.0], 50, lr=0.01)
    b = ex.optimize(ex.RMSProp, lambda x: 1000 * g(x), [1.0, -1.0], 50, lr=0.01)
    assert np.allclose(a, b, atol=1e-6)


def test_adam_bias_correction():
    for gval in (1e-3, 1.0, 1e3):
        p = np.array([0.0])
        ex.Adam([p], lr=0.1).step([np.array([gval])])
        assert p[0] == pytest.approx(-0.1, rel=1e-4)     # first step has size exactly lr
        q = np.array([0.0])
        ex.Adam([q], lr=0.1, bias_correction=False).step([np.array([gval])])
        # uncorrected: 0.1 g / sqrt(0.001 g^2) = 0.1/sqrt(0.001) ~ 3.16 lr
        assert q[0] == pytest.approx(-0.1 * 0.1 / math.sqrt(0.001), rel=1e-3)


def test_adam_matches_formula_over_steps():
    rng = np.random.default_rng(0)
    gs = rng.normal(size=(5, 3))
    p = np.zeros(3)
    opt = ex.Adam([p], lr=0.01, beta1=0.8, beta2=0.9)
    m = v = np.zeros(3)
    ref = np.zeros(3)
    for t, g in enumerate(gs, 1):
        opt.step([g])
        m = 0.8 * m + 0.2 * g
        v = 0.9 * v + 0.1 * g * g
        ref = ref - 0.01 * (m / (1 - 0.8 ** t)) / (np.sqrt(v / (1 - 0.9 ** t)) + 1e-8)
    assert np.allclose(p, ref)


def test_rosenbrock_race():
    adam = ex.optimize(ex.Adam, rosenbrock_grad, [-1.5, 2.0], 3000, lr=0.02)
    sgd = ex.optimize(ex.SGD, rosenbrock_grad, [-1.5, 2.0], 3000, lr=0.001)
    assert np.linalg.norm(adam[-1] - [1, 1]) < 1e-2
    assert np.linalg.norm(sgd[-1] - [1, 1]) > 0.1          # still crawling along the valley
    assert rosenbrock(sgd[-1]) < rosenbrock(sgd[0])        # ... but it did go downhill


def test_warmup_cosine():
    lrs = [ex.warmup_cosine(s, 100, 10, 1.0, 0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0) and lrs[10] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(lrs[:9], lrs[1:10]))           # rising
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))          # falling
    assert lrs[55] == pytest.approx(0.55)                            # halfway: mean of max & min
    assert 0.1 < lrs[-1] < 0.101
    assert ex.warmup_cosine(0, 50, 0, 3e-4) == pytest.approx(3e-4)   # no warmup


# ---- Part B: init ----
def test_xavier_he_variance():
    rng = np.random.default_rng(0)
    X = ex.xavier_normal(300, 500, rng)
    H = ex.he_normal(300, 500, rng)
    assert X.shape == H.shape == (300, 500)
    assert X.var() == pytest.approx(2 / 800, rel=0.03)
    assert H.var() == pytest.approx(2 / 300, rel=0.03)
    assert abs(H.mean()) < 0.002


def test_activation_stds():
    he = ex.activation_stds(30, 256, "he", "relu")
    assert len(he) == 30
    assert 0.5 < he[-1] / he[0] < 2                     # He keeps ReLU nets at constant scale
    xr = ex.activation_stds(30, 256, "xavier", "relu")
    assert xr[-1] < 1e-3                                 # Xavier halves the variance per ReLU layer
    assert ex.activation_stds(10, 128, "small", "tanh")[-1] < 1e-8   # vanishes
    assert ex.activation_stds(10, 128, "large", "relu")[-1] > 1e8     # explodes
    lin = ex.activation_stds(20, 256, "xavier", "linear")
    assert 0.6 < lin[-1] < 1.6                            # fan_in = fan_out: Xavier preserves variance
    with pytest.raises((ValueError, KeyError)):
        ex.activation_stds(2, 8, "nope", "relu")


# ---- Part B: normalization ----
def test_layernorm_forward():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 5.0, size=(4, 6, 8))
    y, _ = ex.layernorm_forward(x, np.ones(8), np.zeros(8))
    assert y.shape == x.shape
    assert np.allclose(y.mean(axis=-1), 0, atol=1e-9)
    assert np.allclose(y.var(axis=-1), 1, atol=1e-3)
    g, b = rng.normal(size=8), rng.normal(size=8)
    y2, _ = ex.layernorm_forward(x, g, b)
    assert np.allclose(y2, g * y + b)
    # each row is normalized on its own: changing other rows changes nothing
    x2 = x.copy()
    x2[0] *= 100
    assert np.allclose(ex.layernorm_forward(x2, g, b)[0][1:], y2[1:])


def test_layernorm_backward_gradcheck():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 5))
    gamma, beta = rng.normal(size=5), rng.normal(size=5)
    w = rng.normal(size=(3, 5))                           # L = sum(w * y): an arbitrary upstream grad

    def L():
        return float(np.sum(w * ex.layernorm_forward(x, gamma, beta)[0]))

    _, cache = ex.layernorm_forward(x, gamma, beta)
    dx, dg, db = ex.layernorm_backward(w, cache)
    assert dx.shape == x.shape and dg.shape == db.shape == (5,)
    assert np.allclose(dx, numerical_grad(L, x), atol=1e-6)
    assert np.allclose(dg, numerical_grad(L, gamma), atol=1e-6)
    assert np.allclose(db, numerical_grad(L, beta), atol=1e-6)
    assert np.allclose(dx.sum(axis=-1), 0, atol=1e-9)    # shifting a row doesn't change its output


def test_batchnorm_vs_layernorm():
    rng = np.random.default_rng(2)
    x = rng.normal(2.0, 3.0, size=(16, 4))
    y = ex.batchnorm_forward_train(x, np.ones(4), np.zeros(4))
    assert np.allclose(y.mean(axis=0), 0, atol=1e-9) and np.allclose(y.var(axis=0), 1, atol=1e-3)
    # BatchNorm couples examples: changing example 0 changes example 1's output
    x2 = x.copy()
    x2[0] += 10
    y2 = ex.batchnorm_forward_train(x2, np.ones(4), np.zeros(4))
    assert not np.allclose(y2[1], y[1])


# ---- Part C ----
def test_residual_gradient_highway():
    plain = ex.input_grad_norm(30, 64, residual=False, scale=0.5)
    res = ex.input_grad_norm(30, 64, residual=True, scale=0.5)
    assert plain < 1e-6
    assert res > 1.0
    # shallow sanity: 1 layer, plain == explicit computation
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.5 / math.sqrt(8), (8, 8))
    x = rng.normal(size=(64, 8))
    t = np.tanh(x @ W)
    want = np.linalg.norm((1 - t * t) @ W.T) / np.linalg.norm(np.ones((64, 8)))
    assert ex.input_grad_norm(1, 8, residual=False, scale=0.5) == pytest.approx(want)
