import math
import numpy as np
import pytest
from _impl import load

ex = load(__file__)

DIE = [1, 2, 3, 4, 5, 6]


# ---- Part A ----
def test_expectation_die_and_linearity():
    assert math.isclose(ex.expectation(DIE, [1 / 6] * 6), 3.5)
    # linearity: E[2X + 3] = 2 E[X] + 3
    assert math.isclose(ex.expectation([2 * v + 3 for v in DIE], [1 / 6] * 6), 10)
    # a loaded die
    assert math.isclose(ex.expectation(DIE, [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]), 2.5)


def test_expectation_validates():
    with pytest.raises(ValueError):
        ex.expectation([1, 2], [0.5])
    with pytest.raises(ValueError):
        ex.expectation([1, 2], [0.5, 0.6])
    with pytest.raises(ValueError):
        ex.expectation([1, 2], [1.5, -0.5])


def test_discrete_variance():
    assert math.isclose(ex.discrete_variance(DIE, [1 / 6] * 6), 35 / 12)
    # Bernoulli(p): p(1-p)
    assert math.isclose(ex.discrete_variance([0, 1], [0.7, 0.3]), 0.21)
    # shift doesn't change variance, scaling by a multiplies it by a^2
    assert math.isclose(ex.discrete_variance([v + 100 for v in DIE], [1 / 6] * 6), 35 / 12)
    assert math.isclose(ex.discrete_variance([-3 * v for v in DIE], [1 / 6] * 6), 9 * 35 / 12)


def test_mean_variance_std():
    xs = [2, 4, 4, 4, 5, 5, 7, 9]
    assert math.isclose(ex.mean(xs), 5)
    assert math.isclose(ex.variance(xs), 4)
    assert math.isclose(ex.std(xs), 2)
    assert math.isclose(ex.variance(xs, ddof=1), 32 / 7)
    with pytest.raises(ValueError):
        ex.mean([])
    with pytest.raises(ValueError):
        ex.variance([1.0], ddof=1)


def test_variance_is_numerically_stable():
    # large offset, tiny spread: the E[X^2] - E[X]^2 shortcut fails here
    xs = [1e9 + v for v in [0.1, 0.2, 0.3, 0.4]]
    assert math.isclose(ex.variance(xs), np.var([0.1, 0.2, 0.3, 0.4]), rel_tol=1e-6)


def test_covariance_and_correlation():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 6, 8, 10]
    assert math.isclose(ex.covariance(xs, xs), ex.variance(xs))
    assert math.isclose(ex.covariance(xs, ys), 4)
    assert math.isclose(ex.covariance(xs, ys, ddof=1), 5)
    assert math.isclose(ex.correlation(xs, ys), 1)
    assert math.isclose(ex.correlation(xs, [-y + 7 for y in ys]), -1)
    # Y = X^2 on a symmetric X: uncorrelated yet completely dependent
    assert abs(ex.correlation([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4])) < 1e-12
    with pytest.raises(ValueError):
        ex.covariance([1, 2], [1, 2, 3])


def test_correlation_scale_invariant_and_matches_numpy():
    r = np.random.default_rng(0)
    x = r.standard_normal(200)
    y = 0.6 * x + 0.8 * r.standard_normal(200)
    c = ex.correlation(list(x), list(y))
    assert math.isclose(c, np.corrcoef(x, y)[0, 1], rel_tol=1e-9)
    assert math.isclose(ex.correlation(list(5 * x + 1), list(0.1 * y - 3)), c, rel_tol=1e-9)
    assert -1 <= c <= 1


# ---- Part B ----
def test_covariance_matrix():
    r = np.random.default_rng(1)
    X = r.standard_normal((50, 3)) @ np.array([[1, 0.5, 0], [0, 1, 0.3], [0, 0, 2]])
    C = ex.covariance_matrix(X)
    assert C.shape == (3, 3)
    assert np.allclose(C, np.cov(X, rowvar=False))
    assert np.allclose(ex.covariance_matrix(X, ddof=0), np.cov(X, rowvar=False, ddof=0))
    assert np.allclose(C, C.T)
    assert np.all(np.linalg.eigvalsh(C) > -1e-12)  # PSD


def test_standardize():
    r = np.random.default_rng(2)
    X = np.column_stack([r.normal(10, 3, 100), r.normal(-5, 0.1, 100), np.full(100, 7.0)])
    X0 = X.copy()
    Z = ex.standardize(X)
    assert Z.shape == X.shape
    assert np.allclose(Z.mean(axis=0), 0)
    assert np.allclose(Z[:, :2].std(axis=0), 1)
    assert np.all(Z[:, 2] == 0) and not np.any(np.isnan(Z))
    assert np.array_equal(X, X0)  # input not modified in place


def test_gaussian_pdf_and_cdf():
    assert math.isclose(ex.gaussian_pdf(0.0), 1 / math.sqrt(2 * math.pi))
    assert math.isclose(ex.gaussian_pdf(3.0, mu=1.0, sigma=2.0), math.exp(-0.5) / (2 * math.sqrt(2 * math.pi)))
    xs = np.linspace(-10, 10, 20001)
    p = ex.gaussian_pdf(xs, mu=0.5, sigma=1.5)
    assert p.shape == xs.shape
    assert abs(p.sum() * (xs[1] - xs[0]) - 1) < 1e-6  # integrates to 1
    with pytest.raises(ValueError):
        ex.gaussian_pdf(0.0, sigma=0)
    assert math.isclose(ex.normal_cdf(0.0), 0.5)
    assert math.isclose(ex.normal_cdf(1) - ex.normal_cdf(-1), 0.682689492, rel_tol=1e-6)
    assert math.isclose(ex.normal_cdf(190, 170, 10), 0.977249868, rel_tol=1e-6)


def test_cholesky():
    A = np.array([[4.0, 2, 0.4], [2, 5, 1], [0.4, 1, 3]])
    L = ex.cholesky(A)
    assert np.allclose(L, np.tril(L))
    assert np.all(np.diag(L) > 0)
    assert np.allclose(L @ L.T, A)
    assert np.allclose(L, np.linalg.cholesky(A))
    S = np.array([[2.0, 1], [1, 2]])
    assert np.allclose(ex.cholesky(S), [[math.sqrt(2), 0], [1 / math.sqrt(2), math.sqrt(1.5)]])
    with pytest.raises(ValueError):
        ex.cholesky(np.array([[1.0, 2], [2, 1]]))  # eigenvalues 3 and -1


def test_sample_mvn():
    mu = np.array([1.0, -2.0])
    Sigma = np.array([[2.0, 1.2], [1.2, 1.0]])
    X = ex.sample_mvn(mu, Sigma, 200_000, np.random.default_rng(3))
    assert X.shape == (200_000, 2)
    assert np.allclose(X.mean(axis=0), mu, atol=0.02)
    assert np.allclose(np.cov(X, rowvar=False), Sigma, atol=0.03)
    # reproducible given the generator seed
    a = ex.sample_mvn(mu, Sigma, 5, np.random.default_rng(9))
    b = ex.sample_mvn(mu, Sigma, 5, np.random.default_rng(9))
    assert np.array_equal(a, b)


# ---- Part C ----
def test_clt_standardized_sums():
    z = ex.clt_standardized_sums(12, 100_000, np.random.default_rng(4))
    assert z.shape == (100_000,)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02
    # fraction within one sigma ~ 68.3%, within two ~ 95.4%
    assert abs(np.mean(np.abs(z) < 1) - 0.6827) < 0.01
    assert abs(np.mean(np.abs(z) < 2) - 0.9545) < 0.006
    # n = 1 is just a (standardized) uniform: flat, nothing beyond sqrt(3)
    z1 = ex.clt_standardized_sums(1, 10_000, np.random.default_rng(5))
    assert np.max(np.abs(z1)) <= math.sqrt(3) + 1e-9


def test_variance_bias_experiment():
    b, u = ex.variance_bias_experiment(5, 200_000, 2.0, np.random.default_rng(6))
    assert abs(u - 4.0) < 0.05
    assert abs(b - 4.0 * 4 / 5) < 0.05
    assert math.isclose(u / b, 5 / 4)


def test_standard_error_curve():
    ns = [1, 4, 16, 64]
    se = ex.standard_error_curve(ns, 20_000, np.random.default_rng(7))
    assert len(se) == 4
    theory = math.sqrt(35 / 12) / np.sqrt(ns)
    assert np.allclose(se, theory, rtol=0.03)
    # quadrupling n halves the standard error
    assert np.allclose(se[1:] / se[:-1], 0.5, atol=0.03)
