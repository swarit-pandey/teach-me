import math
import numpy as np
import pytest
from _impl import load

ex = load(__file__)
rng = np.random.default_rng(0)


def spd(d, kappa, seed=0):
    """Random symmetric positive-definite matrix with eigenvalues in [1, kappa]."""
    r = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(r.normal(size=(d, d)))
    lam = np.linspace(1, kappa, d)
    return Q @ np.diag(lam) @ Q.T


# ---- Part A ----
def test_numerical_gradient_simple():
    f = lambda v: v[0] ** 2 * v[1] + np.sin(v[1])
    x = np.array([1.5, -0.4])
    g = ex.numerical_gradient(f, x)
    assert g.shape == (2,)
    assert np.allclose(g, [2 * 1.5 * -0.4, 1.5 ** 2 + np.cos(-0.4)], atol=1e-8)
    assert np.array_equal(x, [1.5, -0.4])           # not mutated


def test_numerical_gradient_matrix_shaped():
    W = rng.normal(size=(3, 4))
    f = lambda M: float(np.sum(M ** 3))
    g = ex.numerical_gradient(f, W)
    assert g.shape == (3, 4)
    assert np.allclose(g, 3 * W ** 2, atol=1e-7)


def test_directional_derivative():
    f = lambda v: v[0] ** 2 + 3 * v[0] * v[1]
    x = np.array([1.0, 2.0])                         # grad = (2x+3y, 3x) = (8, 3)
    assert math.isclose(ex.directional_derivative(f, x, [1, 0]), 8, rel_tol=1e-7)
    assert math.isclose(ex.directional_derivative(f, x, [0, 5]), 3, rel_tol=1e-7)   # normalized
    g = np.array([8.0, 3.0])
    best = ex.directional_derivative(f, x, g)
    assert math.isclose(best, np.linalg.norm(g), rel_tol=1e-7)      # steepest = |grad|
    assert abs(ex.directional_derivative(f, x, [-3, 8])) < 1e-6      # along a contour: 0
    with pytest.raises(ValueError):
        ex.directional_derivative(f, x, [0, 0])


def test_quad_form_grad():
    A = rng.normal(size=(4, 4))                       # NOT symmetric
    x = rng.normal(size=4)
    q = lambda v: v @ A @ v
    g = ex.quad_form_grad(A, x)
    assert g.shape == (4,)
    assert np.allclose(g, ex_num_grad(q, x), atol=1e-6)
    S = A + A.T
    assert np.allclose(ex.quad_form_grad(S, x), 2 * S @ x)


def ex_num_grad(f, x, h=1e-6):
    return np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))])


def test_rosenbrock():
    assert ex.rosenbrock(np.array([1.0, 1.0])) == 0
    assert math.isclose(ex.rosenbrock(np.array([0.0, 0.0])), 1)
    assert np.allclose(ex.rosenbrock_grad(np.array([1.0, 1.0])), 0)
    for p in ([-1.2, 1.0], [0.3, -0.7], [2.0, 2.5]):
        p = np.array(p)
        assert np.allclose(ex.rosenbrock_grad(p), ex_num_grad(ex.rosenbrock, p), rtol=1e-6, atol=1e-5)


# ---- Part B ----
def test_gradient_descent_history():
    A = np.diag([1.0, 4.0]); b = np.array([1.0, 2.0])
    H = ex.gradient_descent(lambda x: A @ x - b, np.array([3.0, -1.0]), 0.1, 100)
    assert H.shape == (101, 2)
    assert np.array_equal(H[0], [3.0, -1.0])
    assert np.allclose(H[1], [3.0, -1.0] - 0.1 * (A @ [3.0, -1.0] - b))
    f = lambda x: 0.5 * x @ A @ x - b @ x
    vals = [f(x) for x in H]
    assert all(v2 <= v1 + 1e-12 for v1, v2 in zip(vals, vals[1:]))   # small lr: monotone
    assert np.allclose(H[-1], [1.0, 0.5], atol=1e-2)


def test_gradient_descent_stops_on_divergence():
    H = ex.gradient_descent(lambda x: 2 * x, np.array([1.0]), 1.5, 10_000)   # factor -2 per step
    assert len(H) < 100
    assert np.max(np.abs(H[-1])) > 1e12


def test_gd_rosenbrock():
    H = ex.gradient_descent(ex.rosenbrock_grad, np.array([-1.2, 1.0]), 1e-3, 20000)
    assert np.linalg.norm(H[-1] - [1, 1]) < 1e-2
    # too big a step and the banana valley throws you out
    H2 = ex.gradient_descent(ex.rosenbrock_grad, np.array([-1.2, 1.0]), 1e-2, 2000)
    assert not np.all(np.isfinite(H2[-1])) or np.max(np.abs(H2[-1])) > 1e3


def test_max_stable_lr():
    A = np.array([[3.0, 1.0], [1.0, 3.0]])           # eigenvalues 2, 4
    assert math.isclose(ex.max_stable_lr(A), 0.5)
    A = spd(5, 50.0, seed=3)
    assert math.isclose(ex.max_stable_lr(A), 2 / 50.0)


def test_lr_bound_empirically():
    for seed, kappa in ((1, 10.0), (2, 30.0)):
        A = spd(4, kappa, seed=seed); b = np.random.default_rng(seed).normal(size=4)
        lr_max = ex.max_stable_lr(A)
        assert ex.gd_converges(A, b, 0.9 * lr_max, steps=3000)
        assert not ex.gd_converges(A, b, 1.05 * lr_max, steps=3000)
        assert not ex.gd_converges(A, b, 0.02 * lr_max, steps=50)     # too timid: not there yet


def test_optimal_rate():
    A = spd(3, 9.0, seed=4); b = np.ones(3)
    lr, rate = ex.optimal_lr_and_rate(A)
    assert math.isclose(lr, 2 / 10.0) and math.isclose(rate, 0.8)
    # the measured error ratio approaches the predicted rate
    xs = np.linalg.solve(A, b)
    H = ex.gradient_descent(lambda x: A @ x - b, np.zeros(3), lr, 60)
    e = np.linalg.norm(H - xs, axis=1)
    assert e[-1] / e[-2] <= rate + 1e-6
    assert (e[-1] / e[0]) ** (1 / 60) > 0.7           # and not (much) faster


# ---- Part C ----
def test_hessian_fd():
    A = spd(3, 5.0, seed=5); b = np.arange(3.0)
    f = lambda x: 0.5 * x @ A @ x - b @ x
    H = ex.hessian_fd(f, np.array([0.3, -1.0, 2.0]))
    assert H.shape == (3, 3) and np.allclose(H, H.T)
    assert np.allclose(H, A, atol=1e-5)
    Hr = ex.hessian_fd(ex.rosenbrock, np.array([1.0, 1.0]))
    assert np.allclose(Hr, [[802, -400], [-400, 200]], rtol=1e-4)


def test_classify():
    assert ex.classify_critical_point(np.diag([1.0, 3.0])) == "minimum"
    assert ex.classify_critical_point(np.diag([-1.0, -0.5])) == "maximum"
    assert ex.classify_critical_point(np.array([[2.0, 0.0], [0.0, -2.0]])) == "saddle"
    assert ex.classify_critical_point(np.array([[1.0, 2.0], [2.0, 1.0]])) == "saddle"   # eig 3, -1
    assert ex.classify_critical_point(np.diag([1.0, 0.0])) == "degenerate"
    # f = x^2/2 + y^4/4 - y^2/2 has a saddle at the origin (lesson §7)
    f = lambda v: v[0] ** 2 / 2 + v[1] ** 4 / 4 - v[1] ** 2 / 2
    assert ex.classify_critical_point(ex.hessian_fd(f, np.zeros(2)), tol=1e-4) == "saddle"
    assert ex.classify_critical_point(ex.hessian_fd(f, np.array([0.0, 1.0])), tol=1e-4) == "minimum"
