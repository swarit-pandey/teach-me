"""Workbook 01 — reference solutions. Try first; read after."""
import math
import numpy as np


def dot(a, b):
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def norm(v, p=2):
    if p == math.inf:
        return max(abs(x) for x in v)
    if p < 1:
        raise ValueError("p must be >= 1 for a norm")
    return sum(abs(x) ** p for x in v) ** (1.0 / p)


def normalize(v):
    n = norm(v)
    if n == 0:
        raise ValueError("cannot normalize the zero vector")
    return [x / n for x in v]


def cosine_similarity(a, b):
    c = dot(a, b) / (norm(a) * norm(b))
    return max(-1.0, min(1.0, c))


def angle_degrees(a, b):
    return math.degrees(math.acos(cosine_similarity(a, b)))


def project(b, a):
    k = dot(a, b) / dot(a, a)
    return [k * x for x in a]


def orthogonal_decomposition(b, a):
    par = project(b, a)
    perp = [x - y for x, y in zip(b, par)]
    return par, perp


def coords_in_orthogonal_basis(x, basis):
    return [dot(x, q) / dot(q, q) for q in basis]


def cosine_matrix(W):
    W = np.asarray(W, dtype=float)
    U = W / np.linalg.norm(W, axis=1, keepdims=True)
    return U @ U.T


def nearest_words(query, vocab, E, k=3):
    C = cosine_matrix(E)
    i = vocab.index(query)
    order = [j for j in np.argsort(-C[i]) if j != i]
    return [vocab[j] for j in order[:k]]


def analogy(a, b, c, vocab, E):
    E = np.asarray(E, dtype=float)
    ix = {w: i for i, w in enumerate(vocab)}
    t = E[ix[b]] - E[ix[a]] + E[ix[c]]
    sims = (E @ t) / (np.linalg.norm(E, axis=1) * np.linalg.norm(t))
    for j in np.argsort(-sims):
        if vocab[j] not in (a, b, c):
            return vocab[j]
