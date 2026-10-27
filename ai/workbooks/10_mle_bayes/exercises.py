"""Workbook 10 — Maximum likelihood & Bayes.

Rules of the game
  * Part A: plain Python only (lists, loops, the math module). No numpy. Write every
    log-likelihood as an explicit sum so you see where each term comes from.
  * Part B: NumPy. Fit the same models numerically by gradient ascent and check that
    the optimizer lands exactly where your pencil-and-paper formula says it should.
  * Part C: Bayes. Beta–Bernoulli updates, MAP estimates, and MAP = L2-regularized
    least squares.
  * Run   ./course check 10   to grade. Stuck > 20 min? Read the Hint, then peek at
    solutions.py, close it, and re-write from memory.

Conventions: a coin flip is 1 (heads) or 0 (tails); theta = P(heads).
"Log" always means the natural log.
"""
import math
import numpy as np


# ----------------------------------------------------------------------------
# Part A — likelihoods and closed-form MLEs (plain Python)
# ----------------------------------------------------------------------------

def bernoulli_log_likelihood(theta, flips):
    """log P(flips | theta) = sum_i log( theta if x_i == 1 else 1 - theta ).

    theta must be in [0, 1] (raise ValueError otherwise). If some observed flip has
    probability 0 under theta (e.g. theta = 0 and a head was seen), return -math.inf
    rather than crashing in math.log(0).
    """
    raise NotImplementedError


def bernoulli_likelihood_naive(theta, flips):
    """The likelihood as a raw PRODUCT of per-flip probabilities (no logs).

    Exists so you can watch it underflow: try 2000 flips at theta = 0.5.
    """
    raise NotImplementedError


def bernoulli_mle(flips):
    """Closed-form MLE of theta: (number of heads) / n.  ValueError on empty data."""
    raise NotImplementedError


def gaussian_nll(mu, sigma, xs):
    """Negative log-likelihood of xs under N(mu, sigma^2):

        NLL = (n/2) log(2 pi sigma^2) + sum_i (x_i - mu)^2 / (2 sigma^2)

    Raise ValueError if sigma <= 0.
    """
    raise NotImplementedError


def gaussian_mle(xs):
    """Closed-form MLE (mu_hat, sigma_hat) for N(mu, sigma^2).

    mu_hat is the sample mean; sigma_hat^2 is the mean squared deviation — note the
    divisor is n, NOT n - 1 (lesson 09 explains why the MLE is biased).
    """
    raise NotImplementedError


def categorical_nll(probs, data):
    """NLL of a sequence of category indices under a categorical distribution.

    probs: list of K probabilities; data: list of ints in range(K).
    NLL = -sum_i log probs[data[i]].  This is exactly the language-model loss
    (summed instead of averaged) when probs comes from a softmax.
    Return math.inf if an observed category has probability 0.
    """
    raise NotImplementedError


def categorical_mle(data, K):
    """MLE of a categorical over K classes: the empirical frequencies counts / n.

    Returns a list of length K (classes never seen get 0.0).
    Hint: the derivation needs a Lagrange multiplier for sum(p) = 1 (lesson §5).
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — numerical MLE by gradient ascent (NumPy)
# ----------------------------------------------------------------------------

def gaussian_nll_grad(mu, sigma, xs):
    """Gradient of gaussian_nll with respect to (mu, sigma). Returns (d_mu, d_sigma).

        d/dmu    = -sum(x - mu) / sigma^2
        d/dsigma =  n / sigma - sum((x - mu)^2) / sigma^3

    Derive these yourself before typing them; the test checks against finite
    differences.
    """
    raise NotImplementedError


def fit_bernoulli_gd(flips, lr=0.5, steps=2000):
    """Maximize the AVERAGE Bernoulli log-likelihood by gradient ascent on the logit.

    Parameterize theta = sigmoid(z) so theta can never leave (0, 1). Start at z = 0.
    Show (on paper) that d/dz [ (1/n) log L ] = mean(flips) - sigmoid(z),
    then iterate  z += lr * grad.  Return theta = sigmoid(z) as a float.
    """
    raise NotImplementedError


def fit_gaussian_gd(xs, lr=0.1, steps=3000):
    """Maximize the AVERAGE Gaussian log-likelihood over (mu, s) where sigma = exp(s).

    Start at mu = 0, s = 0. With r = x - mu:
        d/dmu [avg LL] = mean(r) / sigma^2
        d/ds  [avg LL] = mean(r^2) / sigma^2 - 1
    Return (mu, sigma) as floats. It should match gaussian_mle(xs).
    Hint: optimizing log sigma instead of sigma keeps sigma positive for free.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — Bayes: Beta–Bernoulli, MAP, and MAP = regularization
# ----------------------------------------------------------------------------

def beta_pdf(theta, a, b):
    """Density of Beta(a, b) at theta in [0, 1]:

        theta^(a-1) (1-theta)^(b-1) / B(a, b),   B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b)

    Hint: compute in log space with math.lgamma, then exponentiate.
    Return 0.0 outside [0, 1].
    """
    raise NotImplementedError


def beta_bernoulli_update(a, b, flips):
    """Posterior parameters after observing flips with a Beta(a, b) prior:
    Beta(a + heads, b + tails).  Returns a tuple (a', b')."""
    raise NotImplementedError


def beta_posterior_sequence(a, b, flips):
    """List of (a, b) after 0, 1, 2, ..., n flips — length n + 1, starting with the
    prior. Feeding flips one at a time must agree with feeding them all at once."""
    raise NotImplementedError


def beta_mean(a, b):
    """Mean of Beta(a, b): a / (a + b)."""
    raise NotImplementedError


def map_bernoulli(flips, a, b):
    """MAP estimate of theta under a Beta(a, b) prior (a, b >= 1):
    the mode of the posterior, (heads + a - 1) / (n + a + b - 2).
    Raise ValueError if the denominator is 0 (e.g. no data and a = b = 1:
    a flat posterior has no unique mode)."""
    raise NotImplementedError


def mle_linear_regression(X, y):
    """Least-squares weights w minimizing ||X w - y||^2 (= the Gaussian-noise MLE).
    X: (n, d), y: (n,).  Hint: np.linalg.lstsq(X, y, rcond=None)[0]."""
    raise NotImplementedError


def map_linear_regression(X, y, sigma, tau):
    """MAP weights for y = X w + noise, noise ~ N(0, sigma^2), prior w ~ N(0, tau^2 I).

    Minimizing  ||Xw - y||^2 / (2 sigma^2) + ||w||^2 / (2 tau^2)  gives the ridge
    solution  w = (X^T X + lam I)^{-1} X^T y  with  lam = sigma^2 / tau^2.
    Use np.linalg.solve, not an explicit inverse.
    """
    raise NotImplementedError
