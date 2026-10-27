"""Workbook 09 — reference solutions. Try first; read after."""
import math
import numpy as np


# ---- Part A ----

def expectation(values, probs):
    if len(values) != len(probs):
        raise ValueError("values and probs differ in length")
    if any(p < 0 for p in probs) or abs(sum(probs) - 1) > 1e-9:
        raise ValueError("probs must be non-negative and sum to 1")
    return sum(v * p for v, p in zip(values, probs))


def discrete_variance(values, probs):
    m = expectation(values, probs)
    return sum(p * (v - m) ** 2 for v, p in zip(values, probs))


def mean(xs):
    if not xs:
        raise ValueError("mean of empty list")
    return sum(xs) / len(xs)


def variance(xs, ddof=0):
    n = len(xs)
    if n - ddof <= 0:
        raise ValueError("need n - ddof > 0")
    m = mean(xs)
    # two-pass: subtract the mean first. The "E[X^2] - E[X]^2" shortcut is exact
    # algebra but loses precision badly when the mean is large relative to the spread.
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def std(xs, ddof=0):
    return math.sqrt(variance(xs, ddof))


def covariance(xs, ys, ddof=0):
    if len(xs) != len(ys):
        raise ValueError("length mismatch")
    n = len(xs)
    if n - ddof <= 0:
        raise ValueError("need n - ddof > 0")
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - ddof)


def correlation(xs, ys):
    r = covariance(xs, ys) / (std(xs) * std(ys))
    return max(-1.0, min(1.0, r))


# ---- Part B ----

def covariance_matrix(X, ddof=1):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (X.shape[0] - ddof)


def standardize(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = np.sqrt(((X - mu) ** 2).mean(axis=0))
    sd_safe = np.where(sd > 0, sd, 1.0)  # constant column -> (x - mu) = 0 anyway
    return (X - mu) / sd_safe


def gaussian_pdf(x, mu=0.0, sigma=1.0):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    x = np.asarray(x, dtype=float)
    out = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    return float(out) if out.ndim == 0 else out


def normal_cdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def cholesky(A):
    A = np.asarray(A, dtype=float)
    d = A.shape[0]
    L = np.zeros_like(A)
    for i in range(d):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("matrix is not positive definite")
                L[i, i] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def sample_mvn(mu, Sigma, n, rng):
    mu = np.asarray(mu, dtype=float)
    L = cholesky(Sigma)
    Z = rng.standard_normal((n, mu.shape[0]))
    return mu + Z @ L.T


# ---- Part C ----

def clt_standardized_sums(n, trials, rng):
    S = rng.random((trials, n)).sum(axis=1)
    return (S - n * 0.5) / math.sqrt(n / 12)


def variance_bias_experiment(n, trials, sigma, rng):
    X = rng.standard_normal((trials, n)) * sigma
    ss = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return float((ss / n).mean()), float((ss / (n - 1)).mean())


def standard_error_curve(ns, trials, rng):
    out = []
    for n in ns:
        means = rng.integers(1, 7, size=(trials, n)).mean(axis=1)
        out.append(means.std())
    return np.array(out)
