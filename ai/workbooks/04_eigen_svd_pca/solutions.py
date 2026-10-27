"""Workbook 04 — reference solutions. Try first; read after."""
import math
import numpy as np


# ---- Part A ----
def char_poly_2x2(A):
    (a, b), (c, d) = A
    return -(a + d), a * d - b * c


def eig_2x2(A):
    b, c = char_poly_2x2(A)
    disc = b * b - 4 * c
    if disc < -1e-12:
        raise ValueError("complex eigenvalues")
    r = math.sqrt(max(disc, 0.0))
    vals = [(-b + r) / 2, (-b - r) / 2]
    vecs = []
    for lam in vals:
        M = [[A[0][0] - lam, A[0][1]], [A[1][0], A[1][1] - lam]]
        # use the larger row of the singular matrix M for numerical robustness
        row = max(M, key=lambda rw: abs(rw[0]) + abs(rw[1]))
        p, q = row
        if abs(p) + abs(q) < 1e-12:  # A = λI: every vector is an eigenvector
            v = [1.0, 0.0] if not vecs else [-vecs[0][1], vecs[0][0]]
        else:
            v = [-q, p]
        n = math.hypot(*v)
        vecs.append([v[0] / n, v[1] / n])
    return vals, vecs


# ---- Part B ----
def power_iteration(A, num_iters=1000, tol=1e-12, seed=0):
    A = np.asarray(A, dtype=float)
    v = np.random.default_rng(seed).normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(num_iters):
        w = A @ v
        w /= np.linalg.norm(w)
        done = min(np.linalg.norm(w - v), np.linalg.norm(w + v)) < tol
        v = w
        if done:
            break
    return float(v @ A @ v), v


def covariance(X):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (len(X) - 1)


def pca_eig(X, k):
    X = np.asarray(X, dtype=float)
    vals, vecs = np.linalg.eigh(covariance(X))
    order = np.argsort(vals)[::-1][:k]      # eigh is ascending; flip
    return vecs[:, order].T, vals[order], X.mean(axis=0)


def pca_svd(X, k):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return Vt[:k], s[:k] ** 2 / (len(X) - 1), mean


def explained_variance_ratio(X):
    X = np.asarray(X, dtype=float)
    _, var, _ = pca_svd(X, X.shape[1])
    return var / var.sum()


def project(X, components, mean):
    return (np.asarray(X, dtype=float) - mean) @ components.T


def reconstruct(Z, components, mean):
    return Z @ components + mean


def low_rank_approx(A, k):
    U, s, Vt = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=False)
    return U[:, :k] * s[:k] @ Vt[:k]


def eckart_young_error(A, k):
    s = np.linalg.svd(np.asarray(A, dtype=float), compute_uv=False)
    return float(np.sqrt(np.sum(s[k:] ** 2)))


# ---- Part C ----
def top_k_eig_power(S, k, num_iters=2000, tol=1e-12, seed=0):
    S = np.array(S, dtype=float)
    vals, vecs = [], []
    for i in range(k):
        lam, v = power_iteration(S, num_iters, tol, seed + i)
        vals.append(lam)
        vecs.append(v)
        # S = sum λ_j q_j q_j^T, so subtracting this term removes q_i from the
        # spectrum (its eigenvalue becomes 0) and leaves the others untouched.
        S = S - lam * np.outer(v, v)
    return np.array(vals), np.array(vecs)
