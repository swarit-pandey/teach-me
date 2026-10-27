import math
import random

import numpy as np
import pytest
from _impl import load

ex = load(__file__)

SUM7 = lambda o: sum(o) == 7
SUM9 = lambda o: sum(o) >= 9
FIRST_EVEN = lambda o: o[0] % 2 == 0
FIRST6 = lambda o: o[0] == 6
DOUBLES = lambda o: o[0] == o[1]


# ---- Part A ----
def test_sample_space():
    om = ex.sample_space()
    assert len(om) == 36 and len(set(om)) == 36
    assert (1, 1) in om and (6, 6) in om and (3, 4) in om
    assert len(ex.sample_space(3, 4)) == 64
    assert all(len(o) == 3 for o in ex.sample_space(3, 4))


def test_exact_prob():
    assert math.isclose(ex.exact_prob(SUM7), 1 / 6)
    assert math.isclose(ex.exact_prob(lambda o: sum(o) == 8), 5 / 36)
    assert math.isclose(ex.exact_prob(SUM9), 10 / 36)
    assert ex.exact_prob(lambda o: True) == 1 and ex.exact_prob(lambda o: False) == 0
    # complement rule and inclusion-exclusion fall out of counting
    assert math.isclose(ex.exact_prob(lambda o: not SUM7(o)), 1 - ex.exact_prob(SUM7))
    union = ex.exact_prob(lambda o: SUM9(o) or FIRST6(o))
    inter = ex.exact_prob(lambda o: SUM9(o) and FIRST6(o))
    assert math.isclose(union, ex.exact_prob(SUM9) + ex.exact_prob(FIRST6) - inter)
    # at least one six in 4 rolls (de Mere)
    assert math.isclose(ex.exact_prob(lambda o: 6 in o, n_dice=4), 1 - (5 / 6) ** 4)


def test_simulate_prob():
    rng = random.Random(0)
    est = ex.simulate_prob(SUM7, 20000, rng)
    assert abs(est - 1 / 6) < 0.015
    rng = random.Random(1)
    est = ex.simulate_prob(lambda o: 6 in o, 20000, rng, n_dice=4)
    assert abs(est - (1 - (5 / 6) ** 4)) < 0.02
    # reproducible given the seed
    assert ex.simulate_prob(SUM9, 500, random.Random(3)) == ex.simulate_prob(SUM9, 500, random.Random(3))


def test_conditional_prob():
    assert math.isclose(ex.conditional_prob(SUM9, FIRST6), 4 / 6)
    assert math.isclose(ex.conditional_prob(SUM7, lambda o: o[0] == 3), 1 / 6)
    assert math.isclose(ex.conditional_prob(FIRST6, SUM9), 4 / 10)  # not symmetric!
    assert ex.conditional_prob(SUM7, DOUBLES) == 0
    with pytest.raises(ValueError):
        ex.conditional_prob(SUM7, lambda o: sum(o) == 13)


def test_independence():
    assert ex.are_independent(FIRST_EVEN, SUM7)
    assert ex.are_independent(lambda o: o[0] <= 2, lambda o: o[1] >= 5)
    assert not ex.are_independent(SUM9, FIRST6)
    # disjoint events with positive probability are DEPENDENT
    assert not ex.are_independent(SUM7, DOUBLES)


def test_bayes_posterior():
    assert math.isclose(ex.bayes_posterior(0.01, 0.9, 0.91), 0.009 / (0.009 + 0.0891))
    assert math.isclose(ex.bayes_posterior(0.02, 0.95, 0.90), 19 / 117)
    # a perfect-specificity test makes a positive certain
    assert math.isclose(ex.bayes_posterior(0.001, 0.5, 1.0), 1.0)
    # an uninformative test (sens = 1 - spec) leaves the prior unchanged
    assert math.isclose(ex.bayes_posterior(0.3, 0.6, 0.4), 0.3)


def test_bayes_update():
    post = ex.bayes_update([0.99, 0.01], [0.09, 0.9])  # [healthy, sick], evidence = positive test
    assert math.isclose(sum(post), 1)
    assert math.isclose(post[1], ex.bayes_posterior(0.01, 0.9, 0.91))
    # sequential updating == one update with the product of likelihoods
    two = ex.bayes_update(post, [0.09, 0.9])
    once = ex.bayes_update([0.99, 0.01], [0.09 ** 2, 0.9 ** 2])
    assert np.allclose(two, once) and math.isclose(two[1], 0.5025125628, rel_tol=1e-8)
    # three hypotheses: which die? fair, or loaded toward 6 (P(6)=1/2), observed a 6
    post = ex.bayes_update([0.5, 0.5], [1 / 6, 1 / 2])
    assert np.allclose(post, [0.25, 0.75])
    with pytest.raises(ValueError):
        ex.bayes_update([0.5, 0.5], [0, 0])


def test_binomial():
    assert math.isclose(ex.binomial_pmf(2, 4, 0.5), 0.375)
    assert math.isclose(ex.binomial_pmf(3, 10, 0.3), 0.266827932)
    assert ex.binomial_pmf(-1, 5, 0.3) == 0 and ex.binomial_pmf(6, 5, 0.3) == 0
    for n, p in [(1, 0.2), (7, 0.5), (20, 0.9)]:
        assert math.isclose(sum(ex.binomial_pmf(k, n, p) for k in range(n + 1)), 1)
    assert math.isclose(ex.binomial_cdf(1, 4, 0.5), 5 / 16)
    assert math.isclose(ex.binomial_cdf(10, 10, 0.3), 1)
    assert ex.binomial_cdf(-1, 10, 0.3) == 0
    assert math.isclose(ex.binomial_cdf(99, 10, 0.3), 1)


def test_cdf_from_pmf():
    F = ex.cdf_from_pmf([0.5, 0.2, 0.2, 0.1])
    assert np.allclose(F, [0.5, 0.7, 0.9, 1.0])
    assert all(b >= a for a, b in zip(F, F[1:]))
    with pytest.raises(ValueError):
        ex.cdf_from_pmf([0.5, 0.6])
    with pytest.raises(ValueError):
        ex.cdf_from_pmf([1.2, -0.2])


def test_sample_categorical():
    pi = [0.5, 0.2, 0.2, 0.1]
    assert [ex.sample_categorical(pi, u) for u in (0.12, 0.55, 0.71, 0.95)] == [0, 1, 2, 3]
    assert ex.sample_categorical(pi, 0.0) == 0
    assert ex.sample_categorical(pi, 0.5) == 1      # boundaries belong to the right segment
    # zero-probability categories are never chosen, even at boundaries
    pz = [0.0, 0.5, 0.0, 0.5, 0.0]
    for u in (0.0, 0.25, 0.5, 0.75, 0.999999):
        assert ex.sample_categorical(pz, u) in (1, 3)
    # round-off: thirds sum to 0.999..., u just below 1 must still map to a real category
    assert ex.sample_categorical([1 / 3] * 3, 1 - 1e-17) == 2
    # frequencies match probabilities over a fine grid of u
    counts = [0] * 4
    for i in range(10000):
        counts[ex.sample_categorical(pi, (i + 0.5) / 10000)] += 1
    assert np.allclose(np.array(counts) / 10000, pi, atol=1e-3)


# ---- Part B ----
def test_sample_categorical_np():
    rng = np.random.default_rng(0)
    pi = np.array([0.477, 0.215, 0.144, 0.107, 0.043, 0.014])
    s = ex.sample_categorical_np(pi, 200000, rng)
    assert s.shape == (200000,) and s.min() >= 0 and s.max() <= 5
    f = ex.empirical_pmf(s, 6)
    assert f.shape == (6,) and math.isclose(f.sum(), 1)
    assert np.allclose(f, pi, atol=0.005)
    s0 = ex.sample_categorical_np([0.0, 0.3, 0.0, 0.7], 50000, np.random.default_rng(1))
    assert set(np.unique(s0)) <= {1, 3}
    # consistent with the scalar version for the same uniforms
    u = np.random.default_rng(2).random(1000)
    rng_a = np.random.default_rng(2)
    vec = ex.sample_categorical_np(pi / pi.sum(), 1000, rng_a)
    loop = [ex.sample_categorical(list(pi / pi.sum()), x) for x in u]
    assert list(vec) == loop
    assert np.allclose(ex.empirical_pmf([0, 0, 2], 4), [2 / 3, 0, 1 / 3, 0])


def test_running_mean():
    assert np.allclose(ex.running_mean([2, 4, 6, 8]), [2, 3, 4, 5])
    rng = np.random.default_rng(5)
    rolls = rng.integers(1, 7, size=100000)
    m = ex.running_mean(rolls)
    assert m.shape == (100000,)
    assert math.isclose(m[-1], rolls.mean())
    assert abs(m[-1] - 3.5) < 0.03          # law of large numbers
    # later averages wander far less than early ones
    assert np.abs(m[-1000:] - 3.5).max() < np.abs(m[:10] - 3.5).max()


def test_monte_carlo_pi():
    est = ex.monte_carlo_pi(200000, np.random.default_rng(0))
    assert abs(est - math.pi) < 0.02
    # error shrinks with n (averaged over seeds to avoid flukes)
    err = lambda n: np.mean([abs(ex.monte_carlo_pi(n, np.random.default_rng(s)) - math.pi) for s in range(20)])
    assert err(40000) < err(400)


def test_binomial_np():
    pmf = ex.binomial_pmf_np(10, 0.3)
    assert pmf.shape == (11,)
    assert np.allclose(pmf, [ex.binomial_pmf(k, 10, 0.3) for k in range(11)])
    big = ex.binomial_pmf_np(2000, 0.3)
    assert np.all(np.isfinite(big)) and math.isclose(big.sum(), 1, rel_tol=1e-9)
    assert int(np.argmax(big)) == 600            # mode at about n*p
    assert np.allclose(ex.binomial_pmf_np(4, 0.0), [1, 0, 0, 0, 0])


# ---- Part C ----
def test_birthday():
    assert ex.birthday_collision(1) == 0
    assert math.isclose(ex.birthday_collision(23), 0.5072972343, rel_tol=1e-8)
    assert ex.birthday_collision(366) == 1
    assert math.isclose(ex.birthday_collision(2, days=6), 1 / 6)  # two dice match
    vals = [ex.birthday_collision(n) for n in range(1, 60)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_sample_exponential():
    x = ex.sample_exponential(2.0, 200000, np.random.default_rng(0))
    assert x.shape == (200000,) and x.min() >= 0
    assert abs(x.mean() - 0.5) < 0.01
    # P(1 <= X <= 2) for lam=1 equals e^-1 - e^-2
    y = ex.sample_exponential(1.0, 200000, np.random.default_rng(1))
    assert abs(((y >= 1) & (y <= 2)).mean() - (math.exp(-1) - math.exp(-2))) < 0.005
