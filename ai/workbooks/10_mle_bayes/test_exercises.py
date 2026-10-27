from _impl import load
import math
import numpy as np
import pytest

ex = load(__file__)


# ---- Part A ----
def test_bernoulli_log_likelihood():
    flips = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]           # 7 heads, 3 tails
    ll = ex.bernoulli_log_likelihood(0.7, flips)
    assert math.isclose(ll, 7 * math.log(0.7) + 3 * math.log(0.3))
    assert ex.bernoulli_log_likelihood(0.0, [1, 0]) == -math.inf
    assert ex.bernoulli_log_likelihood(1.0, [1, 1]) == 0.0
    with pytest.raises(ValueError):
        ex.bernoulli_log_likelihood(1.5, flips)


def test_log_avoids_underflow():
    flips = [1, 0] * 1000
    assert ex.bernoulli_likelihood_naive(0.5, flips) == 0.0     # underflows!
    assert math.isclose(ex.bernoulli_likelihood_naive(0.5, [1, 0, 1]), 0.125)
    ll = ex.bernoulli_log_likelihood(0.5, flips)
    assert math.isfinite(ll) and math.isclose(ll, 2000 * math.log(0.5))


def test_bernoulli_mle_is_argmax():
    flips = [1, 0, 0, 1, 1, 1, 0, 1]
    th = ex.bernoulli_mle(flips)
    assert math.isclose(th, 5 / 8)
    grid = [i / 1000 for i in range(1, 1000)]
    best = max(grid, key=lambda t: ex.bernoulli_log_likelihood(t, flips))
    assert abs(best - th) < 1e-3
    with pytest.raises(ValueError):
        ex.bernoulli_mle([])


def test_gaussian_nll_and_mle():
    xs = [2, 4, 4, 4, 5, 5, 7, 9]
    mu, sigma = ex.gaussian_mle(xs)
    assert math.isclose(mu, 5) and math.isclose(sigma, 2)
    assert math.isclose(ex.gaussian_nll(5, 2, xs), 4 * math.log(8 * math.pi) + 4)
    best = ex.gaussian_nll(mu, sigma, xs)
    for dm, ds in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1), (0.05, -0.05)]:
        assert ex.gaussian_nll(mu + dm, sigma + ds, xs) > best      # it really is the minimum
    with pytest.raises(ValueError):
        ex.gaussian_nll(0, 0, xs)


def test_categorical():
    data = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
    p = ex.categorical_mle(data, 4)
    assert len(p) == 4 and np.allclose(p, [0.5, 0.3, 0.2, 0.0])
    assert math.isclose(ex.categorical_nll(p[:3] + [0.0], data),
                        -(5 * math.log(.5) + 3 * math.log(.3) + 2 * math.log(.2)))
    assert ex.categorical_nll([0.5, 0.5, 0.0], [2]) == math.inf
    # the MLE beats any other distribution on its own data
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.dirichlet(np.ones(4)).tolist()
        assert ex.categorical_nll(q, data) >= ex.categorical_nll(p, data) - 1e-12


# ---- Part B ----
def test_gaussian_nll_grad_matches_finite_differences():
    rng = np.random.default_rng(1)
    xs = rng.normal(3, 2, size=20).tolist()
    for mu, sigma in [(0.0, 1.0), (2.5, 3.0), (4.0, 0.7)]:
        g = ex.gaussian_nll_grad(mu, sigma, xs)
        h = 1e-6
        f = lambda m, s: ex.gaussian_nll(m, s, xs)
        num = ((f(mu + h, sigma) - f(mu - h, sigma)) / (2 * h),
               (f(mu, sigma + h) - f(mu, sigma - h)) / (2 * h))
        assert np.allclose(g, num, rtol=1e-5, atol=1e-5)
    m, s = ex.gaussian_mle(xs)
    assert np.allclose(ex.gaussian_nll_grad(m, s, xs), 0, atol=1e-8)   # zero gradient at the MLE


def test_fit_bernoulli_gd_matches_closed_form():
    rng = np.random.default_rng(2)
    flips = (rng.random(200) < 0.3).astype(int).tolist()
    th = ex.fit_bernoulli_gd(flips)
    assert abs(th - ex.bernoulli_mle(flips)) < 1e-6
    assert 0 < ex.fit_bernoulli_gd([1, 1, 1], steps=200) < 1    # sigmoid keeps it inside (0,1)


def test_fit_gaussian_gd_matches_closed_form():
    rng = np.random.default_rng(3)
    xs = rng.normal(1.5, 0.8, size=100).tolist()
    mu, sigma = ex.fit_gaussian_gd(xs)
    m, s = ex.gaussian_mle(xs)
    assert abs(mu - m) < 1e-4 and abs(sigma - s) < 1e-4


# ---- Part C ----
def test_beta_pdf():
    assert math.isclose(ex.beta_pdf(0.3, 1, 1), 1.0)                    # uniform
    assert math.isclose(ex.beta_pdf(0.5, 2, 2), 1.5)                    # 6 t (1-t)
    assert ex.beta_pdf(1.2, 2, 2) == 0.0
    # integrates to 1 (midpoint rule)
    N = 20000
    tot = sum(ex.beta_pdf((i + .5) / N, 3.5, 2.2) for i in range(N)) / N
    assert abs(tot - 1) < 1e-4


def test_beta_updates():
    assert ex.beta_bernoulli_update(2, 2, [1, 1, 0, 1]) == (5, 3)
    flips = [1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    seq = ex.beta_posterior_sequence(1, 1, flips)
    assert len(seq) == 11 and seq[0] == (1, 1)
    assert seq[-1] == ex.beta_bernoulli_update(1, 1, flips)
    for (a0, b0), (a1, b1) in zip(seq, seq[1:]):
        assert (a1 - a0) + (b1 - b0) == 1                                # one pseudo-count per flip
    assert math.isclose(ex.beta_mean(5, 3), 0.625)


def test_map_bernoulli():
    flips = [1, 1, 1]
    assert ex.map_bernoulli(flips, 1, 1) == ex.bernoulli_mle(flips)       # flat prior: MAP = MLE
    assert math.isclose(ex.map_bernoulli(flips, 2, 2), 0.8)
    with pytest.raises(ValueError):
        ex.map_bernoulli([], 1, 1)
    # with lots of data the prior washes out
    rng = np.random.default_rng(4)
    big = (rng.random(100000) < 0.62).astype(int).tolist()
    assert abs(ex.map_bernoulli(big, 20, 20) - ex.bernoulli_mle(big)) < 1e-3
    # MAP is the argmax of the posterior density
    a, b = ex.beta_bernoulli_update(3, 5, [1, 0, 1, 1])
    grid = [i / 2000 for i in range(1, 2000)]
    best = max(grid, key=lambda t: ex.beta_pdf(t, a, b))
    assert abs(best - ex.map_bernoulli([1, 0, 1, 1], 3, 5)) < 1e-3


def test_map_is_ridge():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 4))
    w_true = np.array([2.0, -1.0, 0.5, 3.0])
    y = X @ w_true + rng.normal(0, 0.5, size=15)
    w_mle = ex.mle_linear_regression(X, y)
    assert np.allclose(X.T @ (X @ w_mle - y), 0, atol=1e-8)               # normal equations
    sigma, tau = 0.5, 0.3
    w_map = ex.map_linear_regression(X, y, sigma, tau)
    # gradient of the MAP objective vanishes at w_map
    g = X.T @ (X @ w_map - y) / sigma ** 2 + w_map / tau ** 2
    assert np.allclose(g, 0, atol=1e-8)
    assert np.linalg.norm(w_map) < np.linalg.norm(w_mle)                  # prior shrinks toward 0
    assert np.allclose(ex.map_linear_regression(X, y, sigma, 1e6), w_mle, atol=1e-6)   # vague prior -> MLE
