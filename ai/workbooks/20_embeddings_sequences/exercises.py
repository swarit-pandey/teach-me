"""Workbook 20 — Embeddings & sequence models.

Rules of the game
  * Part A (numpy): a character vocabulary, one-hot vs lookup, and the counting
    bigram language model — counts, probabilities, NLL, sampling.
  * Part B (torch): the same bigram model learned by gradient descent, then a
    Bengio-2003-style MLP language model with learned embeddings that beats it.
  * Part C (torch): a vanilla RNN cell, and a measurement of how gradients vanish
    (or explode) as they flow back through time.
  * helpers.py provides load_corpus(max_chars) and split(seq, frac).
  * Run   ./course check 20   to grade.  (Torch tests take ~10-20 s.)
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import load_corpus, split  # noqa: F401


# ----------------------------------------------------------------------------
# Part A — characters, one-hot, and the counting bigram model (numpy)
# ----------------------------------------------------------------------------

def build_vocab(text):
    """Return (stoi, itos): stoi maps each distinct character to an id 0..V-1 in
    SORTED character order; itos is the inverse dict (id -> char)."""
    raise NotImplementedError


def encode(s, stoi):
    """String -> list of int ids."""
    raise NotImplementedError


def decode(ids, itos):
    """Iterable of ids (ints or 0-d tensors) -> string."""
    raise NotImplementedError


def one_hot(ids, V):
    """(N, V) float array with a single 1 per row at column ids[i].
    Notice: one_hot(ids, V) @ E == E[ids] for any (V, d) matrix E — an embedding
    lookup IS a matrix product with a one-hot vector."""
    raise NotImplementedError


def bigram_counts(ids, V):
    """(V, V) int array: C[a, b] = number of positions t with ids[t] == a and
    ids[t+1] == b.  Hint: np.add.at(C, (ids[:-1], ids[1:]), 1) handles repeated
    pairs correctly (plain fancy-index += does NOT)."""
    raise NotImplementedError


def bigram_probs(counts, alpha=1.0):
    """Row-normalized probabilities with add-alpha smoothing:
    P[a, b] = (C[a, b] + alpha) / sum_b' (C[a, b'] + alpha).
    Each row is the distribution of the next char given the current one."""
    raise NotImplementedError


def bigram_nll(P, ids):
    """Average negative log-likelihood per predicted character:
    -(1 / (len(ids) - 1)) * sum_t log P[ids[t], ids[t+1]]   (a float, in nats)."""
    raise NotImplementedError


def sample_bigram(P, start, n, seed=0):
    """Generate a list of n ids that begins with `start`: repeatedly draw the next
    id from row P[current] using rng = np.random.default_rng(seed) and
    rng.choice(V, p=P[current])."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — neural language models (torch)
# ----------------------------------------------------------------------------

def build_dataset(ids, block_size):
    """Sliding-window training pairs for next-character prediction.

    Return (X, Y), int64 tensors: X has shape (N, block_size) with
    X[t] = ids[t : t + block_size], and Y[t] = ids[t + block_size], where
    N = len(ids) - block_size.
    Hint: no Python loop needed — build an index matrix with broadcasting:
    torch.arange(N)[:, None] + torch.arange(block_size)[None, :].
    """
    raise NotImplementedError


class NeuralBigram(nn.Module):
    """The bigram model as a neural net: a single (V, V) weight matrix W
    (an nn.Parameter initialized to zeros) whose row x holds the logits of the
    next character.  forward(x) for an int64 tensor x of shape (N,) returns (N, V)
    logits.  (Equivalently one_hot(x) @ W — but index, don't multiply by zeros.)"""

    def __init__(self, V):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def train_neural_bigram(ids, V, steps=300, lr=0.1):
    """Fit NeuralBigram on all (ids[t], ids[t+1]) pairs with full-batch Adam on
    F.cross_entropy. Return (model, final_loss) where final_loss is the training
    cross-entropy (a float) after the last step, computed under torch.no_grad()."""
    raise NotImplementedError


class MLPLanguageModel(nn.Module):
    """Bengio et al. (2003): embed each of the block_size previous characters,
    concatenate, one tanh hidden layer, logits over the vocabulary.

        self.emb = nn.Embedding(V, emb_dim)
        self.fc1 = nn.Linear(block_size * emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, V)
        forward(X: (N, block_size) int64) -> (N, V) logits

    Also store self.block_size (generate() needs it).
    Hint: self.emb(X) has shape (N, block_size, emb_dim); .view(N, -1) concatenates.
    """

    def __init__(self, V, block_size, emb_dim=10, hidden=100):
        super().__init__()
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError


def train_lm(model, X, Y, steps=3000, batch_size=64, lr=3e-3, seed=0):
    """Minibatch training with torch.optim.Adam(lr) and F.cross_entropy.
    Each step draws batch_size random rows with
    torch.randint(0, len(X), (batch_size,), generator=g), where
    g = torch.Generator().manual_seed(seed).  Return the list of per-step losses."""
    raise NotImplementedError


def evaluate_loss(model, X, Y):
    """Mean cross-entropy of model on (X, Y) as a float — eval mode, no_grad."""
    raise NotImplementedError


def generate(model, context, n, seed=0):
    """Sample n new ids from the MLP LM. context: list of at least block_size ids.
    Each step feeds the last block_size ids, softmaxes the logits and draws with
    torch.multinomial(p, 1, generator=g), g = torch.Generator().manual_seed(seed).
    Return just the n new ids (a list of ints)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — RNNs and gradients through time (torch)
# ----------------------------------------------------------------------------

def rnn_forward(xs, h0, Wxh, Whh, b):
    """Vanilla RNN over a sequence.

    xs: (T, B, D), h0: (B, H), Wxh: (D, H), Whh: (H, H), b: (H,).
        h_t = tanh(x_t @ Wxh + h_{t-1} @ Whh + b)
    Return all hidden states stacked: (T, B, H).  The same weights are used at
    every time step — the unrolled graph is a T-layer net with tied weights.
    """
    raise NotImplementedError


def grad_norms_through_time(T=30, H=32, D=8, B=16, scale=0.5, seed=0):
    """How big is dL/dh_t for L = sum(h_{T-1} * v)?  Return a list of T floats,
    element t being the Frobenius norm of dL/dh_t.

    Setup (in this order, after torch.manual_seed(seed)):
        Wxh = randn(D, H) / sqrt(D)            (requires_grad)
        Whh = randn(H, H) * scale / sqrt(H)    (requires_grad)
        b = zeros(H);  xs = randn(T, B, D);  v = randn(B, H);  h_{-1} = zeros(B, H)
    Run the recurrence, call .retain_grad() on every h_t (they are non-leaf
    tensors, whose .grad PyTorch normally discards), backprop L, read h_t.grad.
    Hint: why does Whh need requires_grad? Without any input that requires grad,
    there is no graph at all and retain_grad() fails.
    """
    raise NotImplementedError
