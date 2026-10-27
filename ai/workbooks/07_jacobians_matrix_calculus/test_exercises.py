import numpy as np
import pytest
from _impl import load

ex = load(__file__)


def num_jac(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    return np.stack([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(x.size)], axis=1)


# ---- Part A ----
def test_numerical_jacobian_polar():
    f = lambda v: np.array([v[0] * np.cos(v[1]), v[0] * np.sin(v[1]), v[0] ** 2])
    x = np.array([2.0, 0.5])
    J = ex.numerical_jacobian(f, x)
    assert J.shape == (3, 2)
    want = np.array([[np.cos(.5), -2 * np.sin(.5)], [np.sin(.5), 2 * np.cos(.5)], [4.0, 0.0]])
    assert np.allclose(J, want, atol=1e-8)
    assert np.array_equal(x, [2.0, 0.5])


def test_jac_affine_and_elementwise():
    r = np.random.default_rng(1)
    A, b, x = r.normal(size=(3, 4)), r.normal(size=3), r.normal(size=4)
    J = ex.jac_affine(A, b, x)
    assert J.shape == (3, 4) and np.allclose(J, num_jac(lambda v: A @ v + b, x))
    D = ex.jac_elementwise(lambda v: 1 - np.tanh(v) ** 2, x)
    assert D.shape == (4, 4)
    assert np.allclose(D, num_jac(np.tanh, x), atol=1e-8)
    assert np.count_nonzero(D - np.diag(np.diag(D))) == 0      # diagonal


def test_softmax():
    p = ex.softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(p.sum(), 1) and np.all(np.diff(p) > 0)
    assert np.allclose(ex.softmax(np.array([1000.0, 1001.0])), ex.softmax(np.array([0.0, 1.0])))
    assert np.allclose(ex.softmax(np.array([2.0, 2.0, 2.0, 2.0])), 0.25)


def test_jac_softmax():
    z = np.array([0.5, -1.0, 2.0, 0.0])
    J = ex.jac_softmax(z)
    assert J.shape == (4, 4)
    assert np.allclose(J, num_jac(ex.softmax, z), atol=1e-8)
    assert np.allclose(J, J.T)                       # symmetric
    assert np.allclose(J.sum(axis=0), 0) and np.allclose(J.sum(axis=1), 0)   # probabilities sum to 1
    assert np.allclose(J @ np.ones(4), 0)            # adding a constant to z changes nothing


def test_chain_jacobian():
    A = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0]])
    f = lambda x: A @ x
    g = ex.softmax
    x = np.array([0.3, -0.2])
    J = ex.chain_jacobian(lambda x: A, ex.jac_softmax, f, x)
    assert J.shape == (3, 2)
    assert np.allclose(J, num_jac(lambda v: g(f(v)), x), atol=1e-8)


# ---- Part B ----
def test_grad_least_squares():
    r = np.random.default_rng(2)
    A, b, x = r.normal(size=(6, 3)), r.normal(size=6), r.normal(size=3)
    g = ex.grad_least_squares(A, x, b)
    assert g.shape == (3,)
    L = lambda v: np.sum((A @ v - b) ** 2)
    assert np.allclose(g, num_jac(lambda v: np.array([L(v)]), x)[0], atol=1e-6)
    xs = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.allclose(ex.grad_least_squares(A, xs, b), 0, atol=1e-10)   # normal equations


def test_grad_linear_layer_shape_rule():
    r = np.random.default_rng(3)
    W, x, y = r.normal(size=(2, 5)), r.normal(size=5), r.normal(size=2)
    G = ex.grad_linear_layer(W, x, y)
    assert G.shape == W.shape
    L = lambda w: 0.5 * np.sum((w.reshape(2, 5) @ x - y) ** 2)
    num = num_jac(lambda w: np.array([L(w)]), W.ravel())[0].reshape(2, 5)
    assert np.allclose(G, num, atol=1e-6)
    assert np.linalg.matrix_rank(G) == 1             # an outer product


def make_data(N=16, d_in=3, d_out=2, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(N, d_in))
    Y = np.stack([np.sin(X[:, 0]) + 0.5 * X[:, 1], X[:, 2] * X[:, 0]], axis=1)[:, :d_out]
    return X, Y


def test_init_and_forward_shapes():
    P = ex.init_params(3, 5, 2, seed=0)
    assert P["W1"].shape == (3, 5) and P["b1"].shape == (5,) and P["W2"].shape == (5, 2) and P["b2"].shape == (2,)
    assert np.all(P["b1"] == 0) and np.all(P["b2"] == 0)
    big = ex.init_params(400, 300, 1, seed=1)
    assert abs(big["W1"].std() - 1 / np.sqrt(400)) < 0.005
    X, Y = make_data()
    Yh, cache = ex.forward(P, X)
    assert Yh.shape == (16, 2)
    H = np.tanh(X @ P["W1"] + P["b1"])
    assert np.allclose(Yh, H @ P["W2"] + P["b2"])
    assert np.isclose(ex.mse_loss(Yh, Y), 0.5 * np.mean(np.sum((Yh - Y) ** 2, axis=1)))


def test_backward_matches_numerical():
    X, Y = make_data()
    P = ex.init_params(3, 5, 2, seed=4)
    P["b1"] += 0.1; P["b2"] -= 0.2                       # non-trivial biases
    Yh, cache = ex.forward(P, X)
    G = ex.backward(P, cache, Yh, Y)
    assert set(G) == {"W1", "b1", "W2", "b2"}
    for k in P:
        assert G[k].shape == P[k].shape, k               # the shape rule
    lossf = lambda Q: ex.mse_loss(ex.forward(Q, X)[0], Y)
    # independent check, not using the learner's grad_check
    for k in P:
        num = np.zeros_like(P[k])
        for idx in np.ndindex(P[k].shape):
            old = P[k][idx]
            P[k][idx] = old + 1e-6; lp = lossf(P)
            P[k][idx] = old - 1e-6; lm = lossf(P)
            P[k][idx] = old
            num[idx] = (lp - lm) / 2e-6
        assert np.allclose(G[k], num, atol=1e-7, rtol=1e-5), k


def test_grad_check_function():
    X, Y = make_data()
    P = ex.init_params(3, 4, 2, seed=5)
    Yh, cache = ex.forward(P, X)
    G = ex.backward(P, cache, Yh, Y)
    before = {k: v.copy() for k, v in P.items()}
    lossf = lambda Q: ex.mse_loss(ex.forward(Q, X)[0], Y)
    assert ex.grad_check(lossf, P, G) < 1e-6
    for k in P:
        assert np.array_equal(P[k], before[k])           # restored
    bad = {k: v.copy() for k, v in G.items()}
    bad["W1"][0, 0] += 0.05                              # plant a bug
    assert ex.grad_check(lossf, P, bad) > 1e-3


def test_training_reduces_loss():
    X, Y = make_data(N=64, seed=7)
    P = ex.init_params(3, 16, 2, seed=8)
    l0 = ex.mse_loss(ex.forward(P, X)[0], Y)
    for _ in range(300):
        Yh, c = ex.forward(P, X)
        G = ex.backward(P, c, Yh, Y)
        for k in P:
            P[k] -= 0.2 * G[k]
    l1 = ex.mse_loss(ex.forward(P, X)[0], Y)
    assert l1 < 0.3 * l0


# ---- Part C ----
def test_jacobian_product_modes():
    r = np.random.default_rng(9)
    dims = [50, 40, 30, 1]                     # many inputs, scalar output
    Js = [r.normal(size=(dims[k + 1], dims[k])) for k in range(3)]
    Pf, cf = ex.jacobian_product(Js, "forward")
    Pr, cr = ex.jacobian_product(Js, "reverse")
    want = Js[2] @ Js[1] @ Js[0]
    assert Pf.shape == (1, 50) and np.allclose(Pf, want) and np.allclose(Pr, want)
    assert cf == 30 * 40 * 50 + 1 * 30 * 50
    assert cr == 1 * 30 * 40 + 1 * 40 * 50
    assert cr < cf / 10                          # reverse mode wins for scalar losses
    # ...and forward mode wins when there's one input and many outputs
    dims2 = [1, 30, 40, 50]
    Js2 = [r.normal(size=(dims2[k + 1], dims2[k])) for k in range(3)]
    assert ex.jacobian_product(Js2, "forward")[1] < ex.jacobian_product(Js2, "reverse")[1]
    with pytest.raises(ValueError):
        ex.jacobian_product(Js, "sideways")
