import math
import warnings
import numpy as np
import pytest
from _impl import load

ex = load(__file__)

EXP = (math.exp, math.exp)
SIN = (math.sin, math.cos)
SQ = (lambda x: x * x, lambda x: 2 * x)


# ---- Part A: finite differences ----
def test_forward_diff():
    assert math.isclose(ex.forward_diff(math.exp, 0.0, 0.1), (math.exp(0.1) - 1) / 0.1)
    assert abs(ex.forward_diff(math.sin, 1.0) - math.cos(1.0)) < 1e-5
    # default h must be used (and be small)
    assert abs(ex.forward_diff(lambda x: x ** 3, 2.0) - 12) < 1e-4


def test_central_diff():
    assert math.isclose(ex.central_diff(math.exp, 0.0, 0.1), (math.exp(0.1) - math.exp(-0.1)) / 0.2)
    assert abs(ex.central_diff(math.sin, 1.0) - math.cos(1.0)) < 1e-9
    # exact for quadratics (the h^2 error term involves f''' = 0)
    assert abs(ex.central_diff(lambda x: 3 * x * x - x, 2.0, 0.5) - 11) < 1e-12


def test_truncation_orders():
    """Halving h halves the forward error (O(h)) and quarters the central error (O(h^2))."""
    ef = ex.error_study(math.exp, math.exp, 0.0, [1e-2, 5e-3], "forward")
    ec = ex.error_study(math.exp, math.exp, 0.0, [1e-2, 5e-3], "central")
    assert len(ef) == 2 and len(ec) == 2
    assert 1.9 < ef[0] / ef[1] < 2.1
    assert 3.9 < ec[0] / ec[1] < 4.1
    with pytest.raises(ValueError):
        ex.error_study(math.exp, math.exp, 0.0, [1e-3], "backward-ish")


def test_best_h():
    for f, df in (EXP, SIN):
        hf = ex.best_h(f, df, 1.0, "forward")
        hc = ex.best_h(f, df, 1.0, "central")
        assert 1e-10 <= hf <= 1e-6, hf
        assert 1e-7 <= hc <= 1e-3, hc
        # central at its best beats forward at its best
        assert ex.error_study(f, df, 1.0, [hc], "central")[0] < ex.error_study(f, df, 1.0, [hf], "forward")[0]
    # tiny h is a disaster: rounding error dominates
    assert ex.error_study(math.exp, math.exp, 1.0, [1e-15], "forward")[0] > 1e-3


def test_compose():
    h, dh = ex.compose(SIN, SQ)           # sin(x^2)
    for x in (-1.3, 0.0, 0.7, 2.0):
        assert math.isclose(h(x), math.sin(x * x))
        assert math.isclose(dh(x), math.cos(x * x) * 2 * x, abs_tol=1e-12)
    h2, dh2 = ex.compose(SQ, SIN)         # sin(x)^2 — order matters
    assert math.isclose(dh2(0.7), 2 * math.sin(0.7) * math.cos(0.7))
    assert not math.isclose(dh(0.7), dh2(0.7))


def test_compose_all():
    pairs = [SQ, (lambda u: -u / 2, lambda u: -0.5), EXP]     # x -> exp(-x^2/2)
    h, dh = ex.compose_all(pairs)
    for x in (-2.0, -0.5, 0.3, 1.7):
        assert math.isclose(h(x), math.exp(-x * x / 2))
        assert abs(dh(x) - ex_central(h, x)) < 1e-7
    i, di = ex.compose_all([])
    assert i(3.5) == 3.5 and di(3.5) == 1


def ex_central(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_newton_sqrt2():
    root, hist = ex.newton(lambda x: x * x - 2, lambda x: 2 * x, 1.0)
    assert abs(root - math.sqrt(2)) < 1e-12
    assert hist[0] == 1.0 and hist[-1] == root
    assert math.isclose(hist[1], 1.5) and math.isclose(hist[2], 17 / 12)
    # quadratic convergence: e_{n+1} ~ C e_n^2
    errs = [abs(x - math.sqrt(2)) for x in hist[:4]]
    for a, b in zip(errs, errs[1:]):
        assert b <= 0.5 * a * a + 1e-15


def test_newton_edge_cases():
    root, hist = ex.newton(lambda x: x - 3, lambda x: 1.0, 3.0)
    assert root == 3.0 and hist == [3.0]
    with pytest.raises(ValueError):
        ex.newton(lambda x: x * x + 1, lambda x: 2 * x, 0.0)
    with pytest.raises(RuntimeError):     # no real root: wanders forever
        ex.newton(lambda x: x * x + 1, lambda x: 2 * x, 0.5, max_iter=30)
    r, _ = ex.newton(lambda x: math.cos(x) - x, lambda x: -math.sin(x) - 1, 1.0)
    assert abs(math.cos(r) - r) < 1e-12


# ---- Part B: activations ----
Z = np.linspace(-6, 6, 41).reshape(1, 41) + np.zeros((3, 1))


def test_sigmoid_values_and_shape():
    s = ex.sigmoid(Z)
    assert s.shape == (3, 41)
    assert np.allclose(s, 1 / (1 + np.exp(-Z)))
    assert np.allclose(ex.sigmoid(Z) + ex.sigmoid(-Z), 1)          # symmetry
    assert math.isclose(float(ex.sigmoid(0.0)), 0.5)


def test_sigmoid_stable():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = ex.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert out[0] == 0.0 and out[2] == 1.0 and not np.isnan(out).any()


def test_activation_derivatives_match_numerical():
    z = np.array([-3.1, -0.7, 0.4, 1.9, 5.0])
    h = 1e-5
    num = lambda f: (f(z + h) - f(z - h)) / (2 * h)
    assert np.allclose(ex.sigmoid_prime(z), num(ex.sigmoid), atol=1e-8)
    assert np.allclose(ex.tanh_prime(z), num(np.tanh), atol=1e-8)
    assert np.allclose(ex.relu_prime(z), num(ex.relu), atol=1e-8)


def test_activation_derivative_facts():
    zz = np.linspace(-8, 8, 1601)
    sp = ex.sigmoid_prime(zz)
    assert math.isclose(sp.max(), 0.25) and zz[sp.argmax()] == 0     # vanishing-gradient bound
    tp = ex.tanh_prime(zz)
    assert math.isclose(tp.max(), 1.0)
    # tanh(z) = 2 sigma(2z) - 1   =>   tanh'(z) = 4 sigma'(2z)
    assert np.allclose(tp, 4 * ex.sigmoid_prime(2 * zz))


def test_relu():
    z = np.array([[-2.0, 0.0, 3.0]])
    assert np.array_equal(ex.relu(z), np.array([[0.0, 0.0, 3.0]]))
    assert np.array_equal(ex.relu_prime(z), np.array([[0.0, 0.0, 1.0]]))
    assert ex.relu_prime(z).dtype.kind == "f"


# ---- Part C ----
def test_second_diff():
    assert abs(ex.second_diff(math.sin, 1.0) + math.sin(1.0)) < 1e-6
    assert abs(ex.second_diff(lambda x: x ** 3, 2.0, 0.1) - 12) < 1e-9   # exact for cubics


def test_newton_minimize():
    # f(x) = x^4 - 3x^2 + x : f' = 4x^3 - 6x + 1, f'' = 12x^2 - 6
    df = lambda x: 4 * x ** 3 - 6 * x + 1
    d2f = lambda x: 12 * x * x - 6
    xs, hist = ex.newton_minimize(df, d2f, 1.5)
    assert abs(df(xs)) < 1e-10 and d2f(xs) > 0          # a genuine local minimum
    assert len(hist) < 10
    # started near the hump, Newton happily converges to the local MAXIMUM:
    xm, _ = ex.newton_minimize(df, d2f, 0.0)
    assert abs(df(xm)) < 1e-10 and d2f(xm) < 0
