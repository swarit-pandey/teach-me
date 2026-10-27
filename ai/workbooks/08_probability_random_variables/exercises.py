"""Workbook 08 — Probability & random variables.

Rules of the game
  * Part A: plain Python only (the `math`, `itertools`, `random` modules are fine).
    No numpy. Enumerate sample spaces, multiply and add probabilities by hand.
  * Part B: NumPy, vectorized — no Python loops over samples.
  * Part C: stretch.
  * Run   ./course check 08   to grade. Each function says which test covers it.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.

An "event" below is a predicate: a function that takes one outcome (a tuple of die
faces, e.g. (3, 4)) and returns True/False. Example:  lambda o: sum(o) == 7.
"""
import itertools
import math
import random

import numpy as np


# ----------------------------------------------------------------------------
# Part A — from scratch (plain Python)
# ----------------------------------------------------------------------------

def sample_space(n_dice=2, sides=6):
    """List of all outcomes of rolling `n_dice` dice with faces 1..sides, as tuples.

    sample_space(2) -> [(1,1), (1,2), ..., (6,6)]  (36 tuples).   [test_sample_space]
    Hint: itertools.product.
    """
    raise NotImplementedError


def exact_prob(event, n_dice=2, sides=6):
    """Exact P(event) for fair dice: |{outcomes where event is True}| / |Omega|.

    exact_prob(lambda o: sum(o) == 7) -> 1/6.   [test_exact_prob]
    """
    raise NotImplementedError


def simulate_prob(event, n_trials, rng, n_dice=2, sides=6):
    """Monte Carlo estimate of P(event): roll the dice n_trials times using
    rng.randint(1, sides) for each die, and return the fraction of trials where
    event(outcome) is True.

    rng is a random.Random instance (seeded by the caller).   [test_simulate_prob]
    """
    raise NotImplementedError


def conditional_prob(event_a, event_b, n_dice=2, sides=6):
    """Exact P(A | B) = P(A and B) / P(B) for fair dice.

    Raise ValueError if P(B) == 0 (conditioning on the impossible is undefined).
    [test_conditional_prob]
    """
    raise NotImplementedError


def are_independent(event_a, event_b, n_dice=2, sides=6, tol=1e-12):
    """True iff P(A and B) == P(A) * P(B) (within tol).   [test_independence]"""
    raise NotImplementedError


def bayes_posterior(prior, sensitivity, specificity):
    """P(disease | positive test).

    prior       = P(disease)
    sensitivity = P(positive | disease)
    specificity = P(negative | healthy)
    Hint: the evidence P(positive) comes from the law of total probability.
    [test_bayes_posterior]
    """
    raise NotImplementedError


def bayes_update(prior, likelihood):
    """General discrete Bayes rule.

    prior:      list of P(H_i), summing to 1.
    likelihood: list of P(E | H_i) (need NOT sum to 1).
    Returns the posterior list P(H_i | E) = likelihood_i * prior_i / evidence.
    Raise ValueError if the evidence is 0.   [test_bayes_update]
    """
    raise NotImplementedError


def binomial_pmf(k, n, p):
    """P(X = k) for X ~ Binomial(n, p). Return 0.0 for k < 0 or k > n.

    Hint: math.comb(n, k) * p**k * (1-p)**(n-k).   [test_binomial]
    """
    raise NotImplementedError


def binomial_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p). (k may be outside 0..n.)   [test_binomial]"""
    raise NotImplementedError


def cdf_from_pmf(probs):
    """Running sums of probs: [p0, p0+p1, ..., 1].

    Raise ValueError if any prob is negative or they don't sum to 1 (tol 1e-9).
    [test_cdf_from_pmf]
    """
    raise NotImplementedError


def sample_categorical(probs, u):
    """Inverse-CDF sampling: map ONE uniform u in [0, 1) to a category index.

    Return the smallest i with u < F_i, where F = cdf_from_pmf(probs).
    A category with probability 0 must never be returned. If floating-point
    round-off makes F[-1] slightly less than 1 and u lands above it, return the
    last index with nonzero probability.

    sample_categorical([0.5, 0.2, 0.2, 0.1], 0.55) -> 1.   [test_sample_categorical]
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — vectorized (NumPy)
# ----------------------------------------------------------------------------

def sample_categorical_np(probs, n, rng):
    """Draw n category indices from `probs` at once. rng is np.random.Generator.

    No Python loops: draw u = rng.random(n), then np.searchsorted on the CDF.
    Think about side='right' vs 'left' so zero-probability categories are never
    returned. Clip the result to len(probs)-1 to survive round-off.
    Returns an int array of shape (n,).   [test_sample_categorical_np]
    """
    raise NotImplementedError


def empirical_pmf(samples, k):
    """Fraction of `samples` (ints in 0..k-1) equal to each value: shape (k,).

    Hint: np.bincount(..., minlength=k).   [test_sample_categorical_np]
    """
    raise NotImplementedError


def running_mean(x):
    """Running averages of a 1-D array: out[i] = mean(x[:i+1]).

    No loops: np.cumsum and np.arange.   [test_running_mean]
    """
    raise NotImplementedError


def monte_carlo_pi(n, rng):
    """Estimate pi: draw n points uniform in the unit square (rng.random((n, 2))),
    return 4 * (fraction with x^2 + y^2 < 1).   [test_monte_carlo_pi]
    """
    raise NotImplementedError


def binomial_pmf_np(n, p):
    """Whole Binomial(n, p) PMF as an array of shape (n+1,), for k = 0..n.

    Must sum to 1. You may reuse binomial_pmf, but try computing it with
    log-space arithmetic (math.lgamma) so that n = 2000 doesn't overflow.
    [test_binomial_np]
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch
# ----------------------------------------------------------------------------

def birthday_collision(n, days=365):
    """Exact probability that among n people at least two share a birthday
    (uniform over `days`, independent).

    Hint: complement. P(all distinct) = (days/days) * ((days-1)/days) * ...
    [test_birthday]
    """
    raise NotImplementedError


def sample_exponential(lam, n, rng):
    """n samples from Exponential(lam) by inverse-CDF: F(x) = 1 - exp(-lam x),
    so X = -log(1 - U) / lam with U ~ Uniform[0,1). rng is np.random.Generator.
    Returns shape (n,).   [test_sample_exponential]
    """
    raise NotImplementedError
