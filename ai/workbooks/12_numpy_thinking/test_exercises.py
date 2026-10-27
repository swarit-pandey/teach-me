import numpy as np
import pytest
from _impl import load

ex = load(__file__)
rng = np.random.default_rng(12)


# ---- loop references (slow but obviously correct) ----
def ref_sq_dists(X, Y):
    D = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            D[i, j] = sum((a - b) ** 2 for a, b in zip(X[i], Y[j]))
    return D


def ref_softmax_rows(Z):
    out = np.zeros_like(Z, dtype=float)
    for i, row in enumerate(Z):
        m = max(row)
        e = [np.exp(v - m) for v in row]
        s = sum(e)
        out[i] = [v / s for v in e]
    return out


# ---- Part A ----
def test_pairwise_sq_dists():
    X, Y = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
    D = ex.pairwise_sq_dists(X, Y)
    assert D.shape == (7, 5)
    assert np.allclose(D, ref_sq_dists(X, Y))
    assert np.allclose(np.diag(ex.pairwise_sq_dists(X, X)), 0)


def test_pairwise_gram():
    X, Y = rng.normal(size=(30, 4)), rng.normal(size=(20, 4))
    D = ex.pairwise_sq_dists_gram(X, Y)
    assert D.shape == (30, 20)
    assert np.allclose(D, ref_sq_dists(X, Y))
    big = 1e4 + rng.normal(size=(10, 3))          # cancellation stress test
    assert (ex.pairwise_sq_dists_gram(big, big) >= 0).all()


def test_softmax_rows():
    Z = rng.normal(size=(6, 4)) * 3
    P = ex.softmax(Z)
    assert P.shape == Z.shape
    assert np.allclose(P, ref_softmax_rows(Z))
    assert np.allclose(P.sum(axis=1), 1)


def test_softmax_stable_and_axis():
    Z = np.array([[1000.0, 1000.0, 990.0], [-1000.0, 0.0, 1000.0]])
    with np.errstate(over="raise", invalid="raise"):
        P = ex.softmax(Z)
    assert np.isfinite(P).all()
    assert np.allclose(P[0, :2], 1 / (2 + np.exp(-10)))
    assert np.allclose(P[1], [0, 0, 1])
    Z2 = rng.normal(size=(3, 5))
    assert np.allclose(ex.softmax(Z2, axis=0), ref_softmax_rows(Z2.T).T)
    # shift invariance
    assert np.allclose(ex.softmax(Z2 + 7.0), ex.softmax(Z2))


def test_standardize():
    X = rng.normal(loc=[1, -5, 100], scale=[1, 0.1, 30], size=(200, 3))
    X = np.hstack([X, np.full((200, 1), 4.0)])     # constant column
    Z, mu, sd = ex.standardize(X)
    assert Z.shape == X.shape and mu.shape == (4,) and sd.shape == (4,)
    assert np.isfinite(Z).all()
    assert np.allclose(Z[:, :3].mean(axis=0), 0, atol=1e-10)
    assert np.allclose(Z[:, :3].std(axis=0), 1)
    assert np.allclose(Z[:, 3], 0)
    for j in range(3):                             # matches a per-column loop
        col = X[:, j]
        assert np.allclose(Z[:, j], (col - col.mean()) / col.std())


# ---- Part B ----
def test_one_hot():
    y = np.array([2, 0, 1, 2])
    O = ex.one_hot(y, 4)
    assert O.shape == (4, 4)
    ref = np.zeros((4, 4))
    for i, c in enumerate(y):
        ref[i, c] = 1
    assert np.array_equal(O, ref)
    assert np.array_equal(O.argmax(axis=1), y)


def test_top_k():
    X = rng.permutation(60).reshape(6, 10).astype(float)   # distinct values, no ties
    idx = ex.top_k(X, 3)
    assert idx.shape == (6, 3)
    for i in range(6):
        ref = sorted(range(10), key=lambda j: -X[i, j])[:3]
        assert list(idx[i]) == ref
    assert np.array_equal(ex.top_k(X, 1)[:, 0], X.argmax(axis=1))


def test_clip_outliers():
    X = rng.normal(size=(500, 2))
    X[0, 0], X[1, 1] = 50.0, -40.0
    X0 = X.copy()
    C = ex.clip_outliers(X, 3.0)
    assert np.array_equal(X, X0), "input must not be modified"
    mu, sd = X.mean(axis=0), X.std(axis=0)
    for j in range(2):
        for i in range(500):
            want = mu[j] if abs(X[i, j] - mu[j]) > 3 * sd[j] else X[i, j]
            assert C[i, j] == pytest.approx(want)
    assert C[0, 0] == pytest.approx(mu[0])


# ---- Part C ----
def test_batched_matmul():
    A, B = rng.normal(size=(4, 3, 5)), rng.normal(size=(4, 5, 2))
    C = ex.batched_matmul(A, B)
    assert C.shape == (4, 3, 2)
    for b in range(4):
        assert np.allclose(C[b], A[b] @ B[b])


def test_bilinear():
    Q, W, K = rng.normal(size=(3, 4)), rng.normal(size=(4, 6)), rng.normal(size=(5, 6))
    S = ex.bilinear_scores(Q, W, K)
    assert S.shape == (3, 5)
    for i in range(3):
        for j in range(5):
            assert S[i, j] == pytest.approx(Q[i] @ W @ K[j])


def test_sliding_windows():
    x = np.arange(10.0)
    S = ex.sliding_windows(x, 4)
    assert S.shape == (7, 4)
    for i in range(7):
        assert np.array_equal(S[i], x[i:i + 4])
    assert np.shares_memory(S, x), "must be a view, not a copy"
    assert ex.sliding_windows(x, 10).shape == (1, 10)
    with pytest.raises(ValueError):
        ex.sliding_windows(x, 11)
    with pytest.raises(ValueError):
        ex.sliding_windows(x, 0)


def test_moving_average():
    x = rng.normal(size=50)
    m = ex.moving_average(x, 5)
    assert m.shape == (46,)
    assert np.allclose(m, [x[i:i + 5].mean() for i in range(46)])
