"""Workbook 11 — Entropy, cross-entropy & KL.

Rules of the game
  * Part A: plain Python only (lists of floats, the math module). No numpy.
    Natural log (nats) by default; every function takes an optional `base`
    (base=2 gives bits).
  * Part B: NumPy, vectorized. This is the numerically careful version that real
    training code uses: logsumexp, log_softmax, the cross-entropy loss from logits
    and its gradient.
  * Part C: stretch — Huffman codes (entropy as code length) and fitting a Gaussian
    to a bimodal target by minimizing KL in each direction.
  * Convention everywhere: 0 · log 0 = 0 (an outcome that never happens costs
    nothing). If p puts mass where q puts none, CE and KL are +infinity (math.inf).
  * Run   ./course check 11   to grade. Stuck > 20 min? Read the Hint, then peek
    at solutions.py, close it, and re-write from memory.
"""
import heapq
import math
import numpy as np


# ----------------------------------------------------------------------------
# Part A — from scratch (lists)
# ----------------------------------------------------------------------------

def surprise(p, base=math.e):
    """Surprise (self-information) of an event with probability p:  -log_base(p).

    surprise(0.5, base=2) -> 1.0 bit.  surprise(1) -> 0.  surprise(0) -> math.inf.
    Hint: log_b(x) = ln(x) / ln(b).
    """
    raise NotImplementedError


def entropy(p, base=math.e):
    """Entropy H(p) = sum_i p_i * (-log p_i) of a discrete distribution (a list).

    Use 0 log 0 = 0, i.e. skip zero entries.  entropy([0.5, 0.5], base=2) -> 1.0.
    """
    raise NotImplementedError


def cross_entropy(p, q, base=math.e):
    """Cross-entropy H(p, q) = sum_i p_i * (-log q_i).

    The average surprise when outcomes come from p but you score them with q.
    Raise ValueError on length mismatch. Skip terms with p_i == 0. Return
    math.inf if some p_i > 0 has q_i == 0.
    """
    raise NotImplementedError


def kl(p, q, base=math.e):
    """KL divergence  KL(p || q) = sum_i p_i * log(p_i / q_i)  (= H(p,q) - H(p)).

    Same conventions as cross_entropy (ValueError, skip p_i == 0, inf if q_i == 0
    where p_i > 0). Compute it directly with log(p_i / q_i), not as CE - H: the
    subtraction of two large, nearly-equal numbers loses precision.
    """
    raise NotImplementedError


def perplexity(token_probs):
    """Perplexity of a model on a text, given the probability the model assigned
    to each actual next token:  exp(mean of -ln p_t).

    perplexity([0.25] * 10) -> 4.0  ("as unsure as a fair 4-way choice").
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — vectorized & numerically stable (NumPy)
# ----------------------------------------------------------------------------

def logsumexp(x, axis=-1):
    """log(sum(exp(x))) along `axis`, without overflow or underflow.

    logsumexp([1000, 1000]) must be 1000 + log 2 (the naive formula gives inf),
    and logsumexp([-1000, -1000]) must be -1000 + log 2 (the naive one gives -inf).
    The result has `axis` removed (like np.sum).
    Hint: log sum exp(x) = m + log sum exp(x - m) for any m; choose m = max(x).
    """
    raise NotImplementedError


def log_softmax(logits, axis=-1):
    """log of softmax(logits) along `axis`: z - logsumexp(z). Same shape as logits.

    Never compute np.log(softmax(z)): when a probability underflows to 0 you get
    -inf, and its gradient is NaN. The direct formula stays finite.
    """
    raise NotImplementedError


def cross_entropy_from_logits(logits, targets):
    """Mean cross-entropy loss of a batch — the loss of every classifier and LM.

    logits: (n, K) array of unnormalized scores; targets: length-n int array of
    correct classes. Returns the float  -mean_i log_softmax(logits)[i, targets[i]].
    Hint: fancy indexing  lp[np.arange(n), targets].
    """
    raise NotImplementedError


def cross_entropy_grad_logits(logits, targets):
    """Gradient of cross_entropy_from_logits w.r.t. the logits, shape (n, K).

    Derive it first (lesson 11, problem 9): for one example it is
    softmax(z) - onehot(target). Don't forget the 1/n from the mean.
    """
    raise NotImplementedError


def random_distribution(k, rng):
    """A random probability vector of length k (all entries > 0, sum 1), drawn
    using the numpy Generator `rng` (e.g. rng.exponential or rng.random).
    Used by the tests to check Gibbs' inequality KL(p||q) >= 0 on many pairs.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch
# ----------------------------------------------------------------------------

def huffman_code_lengths(probs):
    """Code length (in bits) of each symbol in a Huffman code for `probs`.

    Algorithm: put every symbol in a min-heap keyed by probability. Repeatedly pop
    the two least likely subtrees, merge them (prob = sum) and push the merge
    back; every symbol inside a merged subtree gets one bit longer. Stop at one
    tree. A single symbol gets length 1.
    Example: [0.5, 0.25, 0.125, 0.125] -> [1, 2, 3, 3].
    Hint: heap entries (prob, tiebreak_counter, list_of_symbols) avoid comparing lists.
    """
    raise NotImplementedError


def discretized_gaussian(xs, mu, sigma):
    """A Gaussian bump restricted to the grid `xs` and renormalized to sum to 1:
    q_j proportional to exp(-(x_j - mu)^2 / (2 sigma^2)).  Returns a numpy array.
    Hint: that is softmax of the log-weights — reuse log_softmax for stability.
    """
    raise NotImplementedError


def fit_gaussian_kl(xs, p, direction, mu0=1.0, log_sigma0=math.log(0.5), lr=0.05, steps=3000):
    """Fit discretized_gaussian(xs, mu, sigma) to a target distribution p on xs by
    gradient descent on
        direction == "forward":  KL(p || q)
        direction == "reverse":  KL(q || p)
    over theta = (mu, log sigma) (log sigma keeps sigma positive). Start from
    (mu0, log_sigma0), take `steps` steps of size `lr`. Return (mu, sigma).

    Hint: central-difference numerical gradients (lesson 06) are fine here: two
    parameters, 4 loss evaluations per step. In the reverse direction p may be
    tiny; add 1e-300 inside the logs so log(0) never happens.
    Curiosity: reverse KL is not convex here. Start it at mu0=0.0, log_sigma0=log(2)
    and it gets stuck straddling both modes (lesson 11, section 6, landscape plot).
    """
    raise NotImplementedError
