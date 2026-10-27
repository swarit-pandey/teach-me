"""Workbook 08 — reference solutions. Try first; read after."""
import itertools
import math
import random

import numpy as np


# ---- Part A ----

def sample_space(n_dice=2, sides=6):
    return list(itertools.product(range(1, sides + 1), repeat=n_dice))


def exact_prob(event, n_dice=2, sides=6):
    omega = sample_space(n_dice, sides)
    return sum(1 for o in omega if event(o)) / len(omega)


def simulate_prob(event, n_trials, rng, n_dice=2, sides=6):
    hits = 0
    for _ in range(n_trials):
        o = tuple(rng.randint(1, sides) for _ in range(n_dice))
        hits += bool(event(o))
    return hits / n_trials


def conditional_prob(event_a, event_b, n_dice=2, sides=6):
    # Shrink the world to B, then count the share of it where A holds.
    omega = sample_space(n_dice, sides)
    in_b = [o for o in omega if event_b(o)]
    if not in_b:
        raise ValueError("P(B) = 0: conditional probability is undefined")
    return sum(1 for o in in_b if event_a(o)) / len(in_b)


def are_independent(event_a, event_b, n_dice=2, sides=6, tol=1e-12):
    pa = exact_prob(event_a, n_dice, sides)
    pb = exact_prob(event_b, n_dice, sides)
    pab = exact_prob(lambda o: event_a(o) and event_b(o), n_dice, sides)
    return abs(pab - pa * pb) <= tol


def bayes_posterior(prior, sensitivity, specificity):
    true_pos = sensitivity * prior
    false_pos = (1 - specificity) * (1 - prior)
    return true_pos / (true_pos + false_pos)


def bayes_update(prior, likelihood):
    if len(prior) != len(likelihood):
        raise ValueError("prior and likelihood must have the same length")
    joint = [l * p for l, p in zip(likelihood, prior)]
    evidence = sum(joint)
    if evidence == 0:
        raise ValueError("evidence has probability 0 under every hypothesis")
    return [j / evidence for j in joint]


def binomial_pmf(k, n, p):
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binomial_cdf(k, n, p):
    return sum(binomial_pmf(j, n, p) for j in range(0, min(k, n) + 1)) if k >= 0 else 0.0


def cdf_from_pmf(probs):
    if any(q < 0 for q in probs):
        raise ValueError("negative probability")
    out, s = [], 0.0
    for q in probs:
        s += q
        out.append(s)
    if abs(s - 1) > 1e-9:
        raise ValueError(f"probabilities sum to {s}, not 1")
    return out


def sample_categorical(probs, u):
    F = cdf_from_pmf(probs)
    for i, f in enumerate(F):
        if u < f:          # strict: a zero-width segment [F_{i-1}, F_i) can never contain u
            return i
    # round-off: u >= F[-1] (which is 1 - tiny). Fall back to the last real category.
    return max(i for i, q in enumerate(probs) if q > 0)


# ---- Part B ----

def sample_categorical_np(probs, n, rng):
    probs = np.asarray(probs, dtype=float)
    F = np.cumsum(probs)
    u = rng.random(n)
    # side='right' returns the first index with F[i] > u, i.e. u < F[i]: same rule as above.
    idx = np.searchsorted(F, u, side="right")
    return np.minimum(idx, len(probs) - 1)


def empirical_pmf(samples, k):
    samples = np.asarray(samples)
    return np.bincount(samples, minlength=k) / len(samples)


def running_mean(x):
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def monte_carlo_pi(n, rng):
    pts = rng.random((n, 2))
    inside = (pts ** 2).sum(axis=1) < 1
    return 4 * inside.mean()


def binomial_pmf_np(n, p):
    k = np.arange(n + 1)
    if p == 0 or p == 1:
        out = np.zeros(n + 1)
        out[0 if p == 0 else n] = 1.0
        return out
    # log C(n,k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1): no huge factorials.
    lg = np.vectorize(math.lgamma)
    logc = lg(n + 1) - lg(k + 1) - lg(n - k + 1)
    return np.exp(logc + k * np.log(p) + (n - k) * np.log1p(-p))


# ---- Part C ----

def birthday_collision(n, days=365):
    if n > days:
        return 1.0
    p_distinct = 1.0
    for i in range(n):
        p_distinct *= (days - i) / days
    return 1 - p_distinct


def sample_exponential(lam, n, rng):
    u = rng.random(n)
    return -np.log1p(-u) / lam
