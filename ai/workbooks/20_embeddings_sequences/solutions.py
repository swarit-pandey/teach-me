"""Workbook 20 — reference solutions. Try first; read after."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import load_corpus, split  # noqa: F401


# ---------------------------------------------------------------- Part A (numpy)

def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(s, stoi):
    return [stoi[c] for c in s]


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


def one_hot(ids, V):
    ids = np.asarray(ids)
    out = np.zeros((len(ids), V))
    out[np.arange(len(ids)), ids] = 1.0
    return out


def bigram_counts(ids, V):
    ids = np.asarray(ids)
    C = np.zeros((V, V), dtype=np.int64)
    np.add.at(C, (ids[:-1], ids[1:]), 1)      # += 1 at every (current, next) pair, repeats included
    return C


def bigram_probs(counts, alpha=1.0):
    c = counts + alpha
    return c / c.sum(axis=1, keepdims=True)


def bigram_nll(P, ids):
    ids = np.asarray(ids)
    return float(-np.log(P[ids[:-1], ids[1:]]).mean())


def sample_bigram(P, start, n, seed=0):
    rng = np.random.default_rng(seed)
    out = [int(start)]
    for _ in range(n - 1):
        out.append(int(rng.choice(P.shape[1], p=P[out[-1]])))
    return out


# ---------------------------------------------------------------- Part B (torch)

def build_dataset(ids, block_size):
    ids = torch.as_tensor(np.asarray(ids), dtype=torch.long)
    N = len(ids) - block_size
    # row t of X is ids[t : t+block]; the target is the character right after it
    idx = torch.arange(N)[:, None] + torch.arange(block_size)[None, :]
    return ids[idx], ids[block_size:]


class NeuralBigram(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(V, V))

    def forward(self, x):
        # one_hot(x) @ W  ==  W[x]: the row lookup is the same matrix product, minus the zeros
        return self.W[x]


def train_neural_bigram(ids, V, steps=300, lr=0.1):
    ids = torch.as_tensor(np.asarray(ids), dtype=torch.long)
    x, y = ids[:-1], ids[1:]
    model = NeuralBigram(V)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = F.cross_entropy(model(x), y).item()
    return model, final


class MLPLanguageModel(nn.Module):
    def __init__(self, V, block_size, emb_dim=10, hidden=100):
        super().__init__()
        self.block_size = block_size
        self.emb = nn.Embedding(V, emb_dim)                  # the lookup table C of Bengio et al.
        self.fc1 = nn.Linear(block_size * emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, V)

    def forward(self, X):
        e = self.emb(X)                                      # (N, block, emb)
        h = torch.tanh(self.fc1(e.view(X.shape[0], -1)))     # concatenate the block's embeddings
        return self.fc2(h)                                   # (N, V) logits


def train_lm(model, X, Y, steps=3000, batch_size=64, lr=3e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch_size,), generator=g)
        loss = F.cross_entropy(model(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def evaluate_loss(model, X, Y):
    model.eval()
    return F.cross_entropy(model(X), Y).item()


@torch.no_grad()
def generate(model, context, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    ctx, out = list(context), []
    for _ in range(n):
        x = torch.tensor([ctx[-model.block_size:]])
        p = F.softmax(model(x), dim=-1)[0]
        nxt = int(torch.multinomial(p, 1, generator=g))
        out.append(nxt)
        ctx.append(nxt)
    return out


# ---------------------------------------------------------------- Part C (torch): RNN

def rnn_forward(xs, h0, Wxh, Whh, b):
    h, hs = h0, []
    for x in xs:                                  # iterate over time (dim 0)
        h = torch.tanh(x @ Wxh + h @ Whh + b)     # the SAME weights at every step
        hs.append(h)
    return torch.stack(hs)


def grad_norms_through_time(T=30, H=32, D=8, B=16, scale=0.5, seed=0):
    torch.manual_seed(seed)
    Wxh = (torch.randn(D, H) / math.sqrt(D)).requires_grad_()    # parameters: so h's need grads
    Whh = (torch.randn(H, H) * scale / math.sqrt(H)).requires_grad_()
    b = torch.zeros(H)
    xs = torch.randn(T, B, D)
    v = torch.randn(B, H)
    h, hs = torch.zeros(B, H), []
    for t in range(T):
        h = torch.tanh(xs[t] @ Wxh + h @ Whh + b)
        h.retain_grad()                           # keep .grad on this non-leaf tensor
        hs.append(h)
    (hs[-1] * v).sum().backward()
    return [h.grad.norm().item() for h in hs]
