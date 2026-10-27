import math
import numpy as np
import pytest
from _impl import load

ex = load(__file__)


def rng(seed=0):
    return np.random.default_rng(seed)


# ---- Part A ----
def test_surprise():
    assert math.isclose(ex.surprise(0.5, base=2), 1.0)
    assert math.isclose(ex.surprise(1 / 8, base=2), 3.0)
    assert ex.surprise(1.0) == 0
    assert math.isclose(ex.surprise(0.2), -math.log(0.2))
    assert ex.surprise(0.0) == math.inf
    # independent events: surprises add
    assert math.isclose(ex.surprise(0.3 * 0.6), ex.surprise(0.3) + ex.surprise(0.6))


def test_entropy_basics():
    assert math.isclose(ex.entropy([0.5, 0.5], base=2), 1.0)
    assert math.isclose(ex.entropy([0.5, 0.25, 0.125, 0.125], base=2), 1.75)
    assert ex.entropy([1.0, 0.0, 0.0]) == 0                   # certainty, and 0 log 0 = 0
    assert math.isclose(ex.entropy([1 / 6] * 6, base=2), math.log2(6))
    # base change is a constant factor: bits = nats / ln 2
    p = [0.1, 0.2, 0.7]
    assert math.isclose(ex.entropy(p, base=2), ex.entropy(p) / math.log(2))


def test_entropy_max_is_uniform():
    r = rng(1)
    for k in (2, 5, 20):
        for _ in range(50):
            p = list(r.dirichlet(np.ones(k)))
            assert ex.entropy(p) <= math.log(k) + 1e-12


def test_cross_entropy_and_kl():
    p, q = [0.5, 0.5], [0.9, 0.1]
    assert math.isclose(ex.kl(p, q), 0.5108256237659907)
    assert math.isclose(ex.kl(q, p), 0.3680642071684971)        # asymmetric!
    assert math.isclose(ex.cross_entropy(p, q), ex.entropy(p) + ex.kl(p, q))
    assert math.isclose(ex.cross_entropy(p, p), ex.entropy(p))
    assert ex.kl(p, p) == 0
    # one-hot target: CE is just -log of the probability on the right answer
    assert math.isclose(ex.cross_entropy([0, 1, 0], [0.3, 0.2, 0.5]), -math.log(0.2))
    assert math.isclose(ex.kl([0.25, 0.75], [0.5, 0.5], base=2), ex.kl([0.25, 0.75], [0.5, 0.5]) / math.log(2))


def test_infinities_and_errors():
    assert ex.cross_entropy([0.5, 0.5], [1.0, 0.0]) == math.inf
    assert ex.kl([0.5, 0.5], [1.0, 0.0]) == math.inf
    assert math.isfinite(ex.kl([1.0, 0.0], [0.5, 0.5]))          # zeros in p are fine
    with pytest.raises(ValueError):
        ex.kl([0.5, 0.5], [0.2, 0.3, 0.5])
    with pytest.raises(ValueError):
        ex.cross_entropy([1.0], [0.5, 0.5])


def test_perplexity():
    assert math.isclose(ex.perplexity([0.25] * 10), 4.0)
    assert math.isclose(ex.perplexity([0.5, 0.25, 0.125, 0.5]), 2 ** 1.75)
    assert math.isclose(ex.perplexity([1.0, 1.0]), 1.0)


# ---- Part B ----
def test_logsumexp_matches_naive_when_safe():
    x = rng(2).normal(size=(4, 7))
    assert np.allclose(ex.logsumexp(x, axis=-1), np.log(np.exp(x).sum(-1)))
    assert np.allclose(ex.logsumexp(x, axis=0), np.log(np.exp(x).sum(0)))
    assert ex.logsumexp(x, axis=-1).shape == (4,)


def test_logsumexp_huge_and_tiny():
    assert math.isclose(float(ex.logsumexp(np.array([1000.0, 1000.0]))), 1000 + math.log(2))
    assert math.isclose(float(ex.logsumexp(np.array([-1000.0, -1000.0]))), -1000 + math.log(2))
    out = ex.logsumexp(np.array([[1e4, 0.0], [-1e4, -1e4]]), axis=1)
    assert np.all(np.isfinite(out))
    assert math.isclose(out[0], 1e4)


def test_log_softmax():
    z = np.array([[1000.0, 1001.0, 1002.0], [-5.0, 0.0, 5.0]])
    lp = ex.log_softmax(z)
    assert lp.shape == z.shape
    assert np.all(np.isfinite(lp))
    assert np.allclose(np.exp(lp).sum(-1), 1)
    assert math.isclose(lp[0, 2], -math.log(1 + math.exp(-1) + math.exp(-2)))
    # shift invariance: adding a constant to every logit changes nothing
    assert np.allclose(ex.log_softmax(z + 123.0), lp)
    # a hopeless class still gets a finite (very negative) log-prob
    assert ex.log_softmax(np.array([0.0, 2000.0]))[0] == pytest.approx(-2000.0)


def test_cross_entropy_from_logits():
    z = np.zeros((3, 50257))                                       # uniform guesser over GPT-2's vocab
    assert math.isclose(ex.cross_entropy_from_logits(z, np.array([0, 7, 99])), math.log(50257))
    z = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    t = np.array([0, 1])
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    assert math.isclose(ex.cross_entropy_from_logits(z, t), -np.mean(np.log(p[[0, 1], t])))
    big = np.array([[1000.0, -1000.0]])
    assert math.isfinite(ex.cross_entropy_from_logits(big, np.array([1])))
    assert ex.cross_entropy_from_logits(big, np.array([1])) == pytest.approx(2000.0)


def test_cross_entropy_gradient_numerically():
    r = rng(3)
    z = r.normal(size=(4, 5))
    t = np.array([0, 3, 3, 1])
    g = ex.cross_entropy_grad_logits(z, t)
    assert g.shape == z.shape
    num = np.zeros_like(z)
    h = 1e-6
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            zp, zm = z.copy(), z.copy()
            zp[i, j] += h; zm[i, j] -= h
            num[i, j] = (ex.cross_entropy_from_logits(zp, t) - ex.cross_entropy_from_logits(zm, t)) / (2 * h)
    assert np.allclose(g, num, atol=1e-7)
    assert np.allclose(g.sum(1), 0)              # softmax - onehot sums to zero per row


def test_random_distribution_and_gibbs():
    r = rng(4)
    for k in (2, 3, 10, 50):
        for _ in range(100):
            p, q = ex.random_distribution(k, r), ex.random_distribution(k, r)
            assert p.shape == (k,) and np.all(p > 0) and math.isclose(p.sum(), 1)
            d = ex.kl(list(p), list(q))
            assert d >= 0                                          # Gibbs' inequality
            assert ex.cross_entropy(list(p), list(q)) >= ex.entropy(list(p)) - 1e-12
    p = ex.random_distribution(6, r)
    assert abs(ex.kl(list(p), list(p))) < 1e-12                   # equality iff p == q


# ---- Part C ----
def test_huffman():
    assert ex.huffman_code_lengths([0.5, 0.25, 0.125, 0.125]) == [1, 2, 3, 3]
    assert ex.huffman_code_lengths([1.0]) == [1]
    r = rng(5)
    for k in (2, 3, 7, 26):
        for _ in range(20):
            p = r.dirichlet(np.ones(k))
            L = ex.huffman_code_lengths(list(p))
            assert len(L) == k
            assert sum(2.0 ** -l for l in L) <= 1 + 1e-12            # Kraft: a valid prefix code
            avg = float(np.dot(p, L))
            H = ex.entropy(list(p), base=2)
            assert H - 1e-12 <= avg < H + 1                       # Shannon's source coding bounds


XS = np.linspace(-6, 6, 241)


def bimodal():
    p = np.exp(-0.5 * ((XS + 2) / 0.6) ** 2) + np.exp(-0.5 * ((XS - 2) / 0.6) ** 2)
    return p / p.sum()


def test_discretized_gaussian():
    q = ex.discretized_gaussian(XS, 1.0, 0.8)
    assert q.shape == XS.shape and math.isclose(q.sum(), 1)
    assert XS[np.argmax(q)] == pytest.approx(1.0)
    assert np.all(np.isfinite(ex.discretized_gaussian(XS, 0.0, 0.01)))


def test_forward_kl_is_mode_covering():
    p = bimodal()
    mu, sigma = ex.fit_gaussian_kl(XS, p, "forward", mu0=1.0)
    mean = float(XS @ p)
    std = float(np.sqrt(((XS - mean) ** 2) @ p))
    # forward KL to a Gaussian = moment matching: q sits between the modes, wide
    assert abs(mu - mean) < 0.05
    assert abs(sigma - std) < 0.05
    assert sigma > 1.8


def test_reverse_kl_is_mode_seeking():
    p = bimodal()
    mu, sigma = ex.fit_gaussian_kl(XS, p, "reverse", mu0=1.0)
    assert abs(mu - 2.0) < 0.1                       # locks onto the nearer mode
    assert abs(sigma - 0.6) < 0.1                    # and matches its width
    mu2, _ = ex.fit_gaussian_kl(XS, p, "reverse", mu0=-1.0)
    assert abs(mu2 + 2.0) < 0.1                      # the other start finds the other mode
    # the fitted q covers one of two equal modes, so KL(q||p) is about ln 2
    q = ex.discretized_gaussian(XS, mu, sigma)
    assert abs(float(np.sum(q * np.log(q / p))) - math.log(2)) < 0.02
