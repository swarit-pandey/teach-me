"""Workbook 11 — reference solutions. Try first; read after."""
import heapq
import math
import numpy as np


# ---------------------------------------------------------------- Part A
def surprise(p, base=math.e):
    if p <= 0:
        return math.inf
    return -math.log(p) / math.log(base)


def entropy(p, base=math.e):
    # 0 log 0 = 0: impossible outcomes contribute nothing.
    return sum(pi * surprise(pi, base) for pi in p if pi > 0)


def cross_entropy(p, q, base=math.e):
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0:
            continue                      # outcomes p never produces cost nothing
        if qi == 0:
            return math.inf               # p produces something q calls impossible
        total += pi * surprise(qi, base)
    return total


def kl(p, q, base=math.e):
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0:
            continue
        if qi == 0:
            return math.inf
        total += pi * math.log(pi / qi) / math.log(base)   # direct form: no CE - H cancellation
    return total


def perplexity(token_probs):
    nll = [-math.log(t) for t in token_probs]
    return math.exp(sum(nll) / len(nll))


# ---------------------------------------------------------------- Part B
def logsumexp(x, axis=-1):
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)        # all -inf row: leave it -inf
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def log_softmax(logits, axis=-1):
    z = np.asarray(logits, dtype=float)
    return z - np.expand_dims(logsumexp(z, axis=axis), axis)


def cross_entropy_from_logits(logits, targets):
    lp = log_softmax(logits, axis=-1)
    targets = np.asarray(targets)
    return float(-np.mean(lp[np.arange(len(targets)), targets]))


def cross_entropy_grad_logits(logits, targets):
    z = np.asarray(logits, dtype=float)
    n = z.shape[0]
    g = np.exp(log_softmax(z))                  # softmax
    g[np.arange(n), targets] -= 1.0             # minus one-hot
    return g / n                                # because the loss is a mean


def random_distribution(k, rng):
    e = rng.exponential(size=k)                 # normalized exponentials = uniform on the simplex
    return e / e.sum()


# ---------------------------------------------------------------- Part C
def huffman_code_lengths(probs):
    # heap of (probability, tiebreak, [symbols in this subtree])
    if len(probs) == 1:
        return [1]
    lengths = [0] * len(probs)
    heap = [(p, i, [i]) for i, p in enumerate(probs)]
    heapq.heapify(heap)
    counter = len(probs)
    while len(heap) > 1:
        p1, _, s1 = heapq.heappop(heap)
        p2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:                      # every symbol under the merge gets one bit deeper
            lengths[s] += 1
        heapq.heappush(heap, (p1 + p2, counter, s1 + s2))
        counter += 1
    return lengths


def discretized_gaussian(xs, mu, sigma):
    xs = np.asarray(xs, dtype=float)
    logq = -0.5 * ((xs - mu) / sigma) ** 2
    return np.exp(log_softmax(logq))


def fit_gaussian_kl(xs, p, direction, mu0=1.0, log_sigma0=math.log(0.5), lr=0.05, steps=3000):
    xs = np.asarray(xs, dtype=float)
    p = np.asarray(p, dtype=float)
    eps = 1e-300

    def loss(theta):
        q = discretized_gaussian(xs, theta[0], math.exp(theta[1]))
        if direction == "forward":           # KL(p || q): average over where p lives
            m = p > 0
            return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m] + eps))))
        else:                                # KL(q || p): average over where q lives
            return float(np.sum(q * (np.log(q + eps) - np.log(p + eps))))

    theta = np.array([mu0, log_sigma0], dtype=float)
    h = 1e-5
    for _ in range(steps):
        g = np.zeros(2)
        for i in range(2):
            e = np.zeros(2); e[i] = h
            g[i] = (loss(theta + e) - loss(theta - e)) / (2 * h)
        theta -= lr * g
    return float(theta[0]), float(math.exp(theta[1]))
