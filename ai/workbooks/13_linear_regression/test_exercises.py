import numpy as np
import pytest
from _impl import load

ex = load(__file__)


def make_data(n=200, d=3, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.arange(1, d + 1, dtype=float) * np.array([1, -1, 0.5][:d] + [1] * max(0, d - 3))
    y = X @ w_true + 2.0 + noise * rng.normal(size=n)
    return X, y, w_true


# ---- Part A ----
def test_predict():
    X = np.array([[1.0, 2.0], [3.0, -1.0]])
    assert np.allclose(ex.predict(X, np.array([2.0, 1.0]), 0.5), [4.5, 5.5])
    assert ex.predict(X, np.zeros(2), 0.0).shape == (2,)


def test_mse():
    assert ex.mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 3.0])) == pytest.approx(4 / 3)
    with pytest.raises(ValueError):
        ex.mse(np.zeros(5), np.zeros((5, 1)))


def test_mse_grad_matches_numerical():
    X, y, _ = make_data(n=30, d=3, seed=1)
    rng = np.random.default_rng(2)
    w, b = rng.normal(size=3), 0.7
    dw, db = ex.mse_grad(X, y, w, b)
    f = lambda w_, b_: np.mean((X @ w_ + b_ - y) ** 2)
    h = 1e-6
    num_w = np.array([(f(w + h * e, b) - f(w - h * e, b)) / (2 * h) for e in np.eye(3)])
    num_b = (f(w, b + h) - f(w, b - h)) / (2 * h)
    assert np.allclose(dw, num_w, rtol=1e-5, atol=1e-7)
    assert db == pytest.approx(num_b, rel=1e-5)
    # at the least-squares solution the gradient vanishes
    A = np.hstack([np.ones((30, 1)), X]); th = np.linalg.lstsq(A, y, rcond=None)[0]
    dw0, db0 = ex.mse_grad(X, y, th[1:], th[0])
    assert np.allclose(dw0, 0, atol=1e-10) and abs(db0) < 1e-10


def test_gd_converges_to_least_squares():
    X, y, _ = make_data()
    w, b, hist = ex.gd_fit(X, y, lr=0.1, steps=500)
    A = np.hstack([np.ones((len(y), 1)), X]); th = np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.allclose(w, th[1:], atol=1e-6) and b == pytest.approx(th[0], abs=1e-6)
    assert len(hist) == 501
    assert hist[0] == pytest.approx(np.mean(y ** 2))
    assert all(h1 <= h0 + 1e-12 for h0, h1 in zip(hist, hist[1:])), "loss must not increase at a small lr"


def test_gd_diverges_with_huge_lr():
    X, y, _ = make_data(n=50)
    with np.errstate(all="ignore"):
        _, _, hist = ex.gd_fit(X, y, lr=5.0, steps=30)
    assert not (hist[-1] < hist[0]), "lr=5 is far above 2/lambda_max: GD should blow up"


# ---- Part B ----
def test_normal_equations():
    X, y, w_true = make_data(noise=0.0)
    A = ex.add_bias_column(X)
    assert A.shape == (200, 4) and np.allclose(A[:, 0], 1)
    w, b = ex.fit_normal_equations(X, y)
    assert np.allclose(w, w_true) and b == pytest.approx(2.0)
    # noisy case: residual is orthogonal to every column of A (the projection picture)
    X, y, _ = make_data(noise=1.0, seed=3)
    w, b = ex.fit_normal_equations(X, y)
    r = y - (X @ w + b)
    assert np.allclose(ex.add_bias_column(X).T @ r, 0, atol=1e-8)


def test_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert ex.r2_score(y, y) == pytest.approx(1.0)
    assert ex.r2_score(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert ex.r2_score(y, y[::-1]) < 0
    assert ex.r2_score(y, np.array([1.0, 2.0, 3.0, 5.0])) == pytest.approx(1 - 1 / 5)


def test_standardizer():
    rng = np.random.default_rng(4)
    Xtr = np.hstack([rng.normal(50, 10, (100, 1)), rng.normal(0, 0.01, (100, 1)), np.full((100, 1), 3.0)])
    s = ex.Standardizer().fit(Xtr)
    Z = s.transform(Xtr)
    assert np.allclose(Z[:, :2].mean(0), 0, atol=1e-10) and np.allclose(Z[:, :2].std(0), 1)
    assert np.allclose(Z[:, 2], 0)
    Xte = np.array([[60.0, 0.0, 3.0]])
    assert np.allclose(s.transform(Xte)[0, 0], (60 - Xtr[:, 0].mean()) / Xtr[:, 0].std()), "use TRAIN stats"


def test_standardizer_unscale():
    rng = np.random.default_rng(5)
    X = rng.normal([100, -3], [20, 0.5], size=(40, 2))
    s = ex.Standardizer().fit(X)
    Z = s.transform(X)
    w, b = np.array([1.5, -2.0]), 0.3
    wo, bo = s.unscale(w, b)
    assert np.allclose(X @ wo + bo, Z @ w + b)


def test_polynomial_features():
    x = np.array([1.0, 2.0, -3.0])
    P = ex.polynomial_features(x, 3)
    assert P.shape == (3, 3)
    assert np.allclose(P, [[1, 1, 1], [2, 4, 8], [-3, 9, -27]])
    # a cubic is fitted exactly by a degree-3 linear regression
    x = np.linspace(-1, 1, 20); y = 1 - 2 * x + 0.5 * x ** 3
    w, b = ex.fit_normal_equations(ex.polynomial_features(x, 3), y)
    assert np.allclose(w, [-2, 0, 0.5], atol=1e-8) and b == pytest.approx(1)


def test_condition_number():
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 10, size=(100, 1))
    raw = ex.condition_number(x)
    std = ex.condition_number((x - x.mean()) / x.std())
    assert std == pytest.approx(1.0)
    assert raw > 20 * std, "un-centered features give an elongated loss bowl"


# ---- Part C ----
def test_back_substitution():
    rng = np.random.default_rng(7)
    R = np.triu(rng.normal(size=(5, 5))) + 3 * np.eye(5)
    z = rng.normal(size=5)
    x = ex.back_substitution(R, z)
    assert np.allclose(R @ x, z)


def test_fit_qr():
    X, y, _ = make_data(noise=0.5, seed=8)
    w, b = ex.fit_qr(X, y)
    w2, b2 = ex.fit_normal_equations(X, y)
    assert np.allclose(w, w2) and b == pytest.approx(b2)
    # ill-conditioned polynomial basis: QR stays accurate
    x = np.linspace(0, 1, 50); P = x[:, None] ** np.arange(1, 9)
    y = P @ np.ones(8) + 1.0
    w, b = ex.fit_qr(P, y)
    assert np.allclose(P @ w + b, y, atol=1e-6)
