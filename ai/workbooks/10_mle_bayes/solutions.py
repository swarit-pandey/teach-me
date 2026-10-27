"""Workbook 10 — reference solutions. Try first; read after."""
import math
import numpy as np


# ---- Part A ----

def bernoulli_log_likelihood(theta, flips):
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    total = 0.0
    for x in flips:
        p = theta if x == 1 else 1.0 - theta
        if p == 0.0:
            return -math.inf          # an observed event had probability 0
        total += math.log(p)
    return total


def bernoulli_likelihood_naive(theta, flips):
    L = 1.0
    for x in flips:
        L *= theta if x == 1 else 1.0 - theta   # underflows to 0.0 for long sequences
    return L


def bernoulli_mle(flips):
    if not flips:
        raise ValueError("need at least one flip")
    return sum(flips) / len(flips)


def gaussian_nll(mu, sigma, xs):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    n = len(xs)
    sse = sum((x - mu) ** 2 for x in xs)
    return 0.5 * n * math.log(2 * math.pi * sigma ** 2) + sse / (2 * sigma ** 2)


def gaussian_mle(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n      # divide by n: the MLE, slightly biased
    return mu, math.sqrt(var)


def categorical_nll(probs, data):
    total = 0.0
    for k in data:
        if probs[k] == 0.0:
            return math.inf
        total -= math.log(probs[k])
    return total


def categorical_mle(data, K):
    counts = [0] * K
    for k in data:
        counts[k] += 1
    n = len(data)
    return [c / n for c in counts]


# ---- Part B ----

def gaussian_nll_grad(mu, sigma, xs):
    x = np.asarray(xs, dtype=float)
    r = x - mu
    d_mu = -r.sum() / sigma ** 2
    d_sigma = len(x) / sigma - (r ** 2).sum() / sigma ** 3
    return float(d_mu), float(d_sigma)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_bernoulli_gd(flips, lr=0.5, steps=2000):
    xbar = float(np.mean(flips))
    z = 0.0
    for _ in range(steps):
        # d/dz (1/n) sum [x log s(z) + (1-x) log(1-s(z))] = xbar - s(z)
        z += lr * (xbar - _sigmoid(z))
    return float(_sigmoid(z))


def fit_gaussian_gd(xs, lr=0.1, steps=3000):
    x = np.asarray(xs, dtype=float)
    mu, s = 0.0, 0.0
    for _ in range(steps):
        sigma2 = math.exp(2 * s)
        r = x - mu
        g_mu = r.mean() / sigma2
        g_s = (r ** 2).mean() / sigma2 - 1.0
        mu += lr * g_mu
        s += lr * g_s
    return float(mu), float(math.exp(s))


# ---- Part C ----

def beta_pdf(theta, a, b):
    if theta < 0 or theta > 1:
        return 0.0
    if (theta == 0 and a < 1) or (theta == 1 and b < 1):
        return math.inf
    if (theta == 0 and a > 1) or (theta == 1 and b > 1):
        return 0.0
    log_B = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    t1 = (a - 1) * math.log(theta) if a != 1 else 0.0
    t2 = (b - 1) * math.log(1 - theta) if b != 1 else 0.0
    return math.exp(t1 + t2 - log_B)


def beta_bernoulli_update(a, b, flips):
    h = sum(flips)
    return a + h, b + len(flips) - h


def beta_posterior_sequence(a, b, flips):
    seq = [(a, b)]
    for x in flips:
        a, b = (a + 1, b) if x == 1 else (a, b + 1)
        seq.append((a, b))
    return seq


def beta_mean(a, b):
    return a / (a + b)


def map_bernoulli(flips, a, b):
    n, h = len(flips), sum(flips)
    den = n + a + b - 2
    if den == 0:
        raise ValueError("posterior is flat: no unique mode")
    return (h + a - 1) / den


def mle_linear_regression(X, y):
    return np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)[0]


def map_linear_regression(X, y, sigma, tau):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    lam = sigma ** 2 / tau ** 2      # prior strength = noise variance / prior variance
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
