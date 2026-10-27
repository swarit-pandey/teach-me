import numpy as np
import pytest
from _impl import load

ex = load(__file__)


def sparse_data(n=100, d=10, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d); w[:3] = [3.0, -2.0, 1.5]
    return X, X @ w + 1.0 + noise * rng.normal(size=n), w


# ---- Part A ----
def test_split():
    tr, va, te = ex.train_val_test_split(103, 0.2, 0.1, seed=1)
    assert (len(va), len(te), len(tr)) == (21, 10, 72)
    allidx = np.concatenate([tr, va, te])
    assert np.array_equal(np.sort(allidx), np.arange(103))
    tr2, _, _ = ex.train_val_test_split(103, 0.2, 0.1, seed=1)
    assert np.array_equal(tr, tr2), "same seed -> same split"
    assert not np.array_equal(np.sort(tr[:20]), np.arange(20)), "indices must be shuffled"


def test_kfold():
    folds = ex.kfold_indices(23, 5, seed=0)
    assert len(folds) == 5
    vals = [va for _, va in folds]
    assert sorted(len(v) for v in vals) == [4, 4, 5, 5, 5]
    assert np.array_equal(np.sort(np.concatenate(vals)), np.arange(23))
    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == 23


# ---- Part B ----
def test_ridge_zero_is_ols():
    X, y, _ = sparse_data()
    w, b = ex.ridge_fit(X, y, 0.0)
    A = np.hstack([np.ones((len(y), 1)), X]); th = np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.allclose(w, th[1:]) and b == pytest.approx(th[0])


def test_ridge_shrinks_and_is_optimal():
    X, y, _ = sparse_data(seed=1)
    norms = [np.linalg.norm(ex.ridge_fit(X, y, lam)[0]) for lam in [0, 1, 10, 100, 1e4]]
    assert all(a > b for a, b in zip(norms, norms[1:]))
    assert norms[-1] < 0.1 * norms[0]
    lam = 5.0
    w, b = ex.ridge_fit(X, y, lam)
    # stationarity of ||y - Xw - b||^2 + lam ||w||^2
    r = y - X @ w - b
    assert np.allclose(-2 * X.T @ r + 2 * lam * w, 0, atol=1e-8) and abs(r.sum()) < 1e-8


def test_soft_threshold():
    z = np.array([-3.0, -0.5, 0.0, 0.2, 2.0])
    assert np.allclose(ex.soft_threshold(z, 1.0), [-2, 0, 0, 0, 1])
    assert np.allclose(ex.soft_threshold(z, 0.0), z)


def test_lasso_kkt():
    X, y, _ = sparse_data(seed=2)
    lam = 0.1
    w, b = ex.lasso_cd(X, y, lam)
    n = len(y)
    r = y - X @ w - b
    g = X.T @ r / n                                  # = X_c^T r / n since sum(r) = 0
    assert abs(r.sum()) < 1e-8
    nz = w != 0
    assert np.allclose(g[nz], lam * np.sign(w[nz]), atol=1e-6)   # active: gradient balanced
    assert np.all(np.abs(g[~nz]) <= lam + 1e-6)                 # inactive: subgradient fits


def test_lasso_sparsity_and_lambda_max():
    X, y, w_true = sparse_data(seed=3)
    w, _ = ex.lasso_cd(X, y, 0.3)
    assert np.all(w[:3] != 0), "true features survive"
    assert np.sum(w[3:] != 0) <= 2, "most noise features are exactly zero"
    lmax = ex.lasso_lambda_max(X, y)
    assert np.all(ex.lasso_cd(X, y, lmax * 1.0001)[0] == 0)
    assert np.any(ex.lasso_cd(X, y, lmax * 0.95)[0] != 0)
    # tiny lam ~ least squares
    w0, b0 = ex.lasso_cd(X, y, 1e-8, n_iter=5000, tol=1e-14)
    wr, br = ex.ridge_fit(X, y, 0.0)
    assert np.allclose(w0, wr, atol=1e-5)


# ---- Part C ----
def test_cross_val():
    X, y, _ = sparse_data(seed=4)
    fit = lambda A, t: ex.ridge_fit(A, t, 1.0)
    s = ex.cross_val_mse(fit, X, y, k=4, seed=7)
    manual = []
    for tr, va in ex.kfold_indices(len(y), 4, seed=7):
        w, b = fit(X[tr], y[tr]); manual.append(np.mean((X[va] @ w + b - y[va]) ** 2))
    assert s == pytest.approx(np.mean(manual))
    assert 0.1 < s < 1.0, "noise variance is 0.25; CV MSE should be near it"


def test_choose_lambda():
    rng = np.random.default_rng(5)
    n, d = 40, 30                                   # few examples, many features -> needs shrinkage
    X = rng.normal(size=(n, d)); w = rng.normal(size=d) * 0.3
    y = X @ w + rng.normal(size=n)
    lams = np.array([1e-3, 0.1, 1, 10, 100, 1e4])
    best, scores = ex.choose_lambda(X, y, lams, "ridge", k=5)
    assert scores.shape == (6,) and best == lams[np.argmin(scores)]
    assert 0.1 <= best <= 100 and scores[0] > scores.min()
    best_l, scores_l = ex.choose_lambda(*sparse_data(seed=6)[:2], np.array([1e-3, 0.05, 0.3, 3.0]), "lasso")
    assert best_l in (1e-3, 0.05, 0.3) and scores_l[-1] == scores_l.max()


def test_learning_curve():
    X, y, _ = sparse_data(n=300, seed=8)
    fit = lambda A, t: ex.ridge_fit(A, t, 1e-6)
    tr, va = ex.learning_curve(X[:200], y[:200], X[200:], y[200:], [12, 25, 50, 100, 200], fit)
    assert tr.shape == va.shape == (5,)
    assert np.all(va[:3] > tr[:3])                  # with little data, validation error sits far above training
    assert va[0] > va[-1] and tr[0] < tr[-1]        # the two curves converge
    assert abs(va[-1] - 0.25) < 0.15


def test_bias_variance():
    f = lambda x: np.sin(np.pi * x)
    b1, v1 = ex.bias_variance(f, 1, trials=200, seed=0)
    b9, v9 = ex.bias_variance(f, 9, trials=200, seed=0)
    b3, v3 = ex.bias_variance(f, 3, trials=200, seed=0)
    assert b1 > 10 * b3, "a line can't bend: high bias"
    assert v9 > 3 * v3, "a degree-9 fit on 20 points: high variance"
    assert b1 > 0.1 and v3 < 0.05
