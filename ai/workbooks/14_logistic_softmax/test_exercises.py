import numpy as np
import pytest
from _impl import load

ex = load(__file__)  # also puts this folder on sys.path
from helpers import two_blobs, three_blobs  # noqa: E402


# ---- Part A ----
def test_sigmoid():
    z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    assert np.allclose(ex.sigmoid(z), 1 / (1 + np.exp(-z)))
    assert np.allclose(ex.sigmoid(-z), 1 - ex.sigmoid(z))          # symmetry
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        s = ex.sigmoid(np.array([-1000.0, 1000.0]))
    assert np.allclose(s, [0.0, 1.0])


def test_log_loss():
    z, y = np.array([2.0, -1.0, 0.3]), np.array([1, 0, 0])
    p = 1 / (1 + np.exp(-z))
    naive = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert ex.log_loss(z, y) == pytest.approx(naive)
    assert ex.log_loss(np.zeros(4), np.array([0, 1, 0, 1])) == pytest.approx(np.log(2))
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        big = ex.log_loss(np.array([1000.0, -1000.0]), np.array([0, 1]))
    assert big == pytest.approx(1000.0)                              # confidently wrong costs |z|


def test_numerical_gradient():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    f = lambda t: 0.5 * t @ A @ t + np.sin(t[0])
    t = np.array([0.4, -1.2]); t0 = t.copy()
    g = ex.numerical_gradient(f, t)
    assert np.allclose(g, A @ t + np.array([np.cos(0.4), 0]), atol=1e-6)
    assert np.array_equal(t, t0), "theta must not be modified"


def test_log_loss_grad():
    X, y = two_blobs(30, seed=1)
    rng = np.random.default_rng(1)
    w, b = rng.normal(size=2), 0.3
    dw, db = ex.log_loss_grad(X, y, w, b)
    f = lambda th: ex.log_loss(X @ th[:2] + th[2], y)
    num = ex.numerical_gradient(f, np.r_[w, b])
    assert np.allclose(np.r_[dw, db], num, atol=1e-7)
    # reference formula, independently
    p = 1 / (1 + np.exp(-(X @ w + b)))
    assert np.allclose(dw, X.T @ (p - y) / len(y))


def test_train_logistic():
    X, y = two_blobs(150, seed=2)
    w, b, hist = ex.train_logistic(X, y, lr=0.5, steps=300)
    assert len(hist) == 301 and hist[0] == pytest.approx(np.log(2))
    assert hist[-1] < 0.3 and all(h1 <= h0 + 1e-12 for h0, h1 in zip(hist, hist[1:]))
    p = ex.predict_proba(X, w, b)
    assert p.shape == (300,) and ((p >= 0) & (p <= 1)).all()
    assert ex.accuracy(y, (p > 0.5).astype(int)) > 0.9
    # optimality: gradient is (nearly) zero after enough steps
    w2, b2, _ = ex.train_logistic(X, y, lr=1.0, steps=3000)
    dw, db = ex.log_loss_grad(X, y, w2, b2)
    assert np.linalg.norm(dw) < 1e-4 and abs(db) < 1e-4
    # L2 shrinks the weights
    w3, _, _ = ex.train_logistic(X, y, lr=0.5, steps=1000, l2=0.5)
    assert np.linalg.norm(w3) < np.linalg.norm(w2)


def test_accuracy():
    assert ex.accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])) == pytest.approx(0.75)


# ---- Part B ----
def test_softmax():
    Z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, -1000.0]])
    P = ex.softmax(Z)
    assert np.allclose(P.sum(axis=1), 1) and np.isfinite(P).all()
    e = np.exp([1.0, 2.0, 3.0]); assert np.allclose(P[0], e / e.sum())
    assert np.allclose(P[1], [0.5, 0.5, 0])


def test_cross_entropy():
    Z = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]]); y = np.array([0, 2])
    P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    assert ex.cross_entropy(Z, y) == pytest.approx(-np.mean(np.log(P[[0, 1], y])))
    assert ex.cross_entropy(np.zeros((5, 4)), np.arange(5) % 4) == pytest.approx(np.log(4))
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        v = ex.cross_entropy(np.array([[1000.0, 0.0]]), np.array([1]))
    assert v == pytest.approx(1000.0)


def test_softmax_grad():
    X, y = three_blobs(10, seed=3)
    rng = np.random.default_rng(3)
    W, b = rng.normal(size=(2, 3)), rng.normal(size=3)
    dW, db = ex.softmax_grad(X, y, W, b)
    assert dW.shape == (2, 3) and db.shape == (3,)
    f = lambda th: ex.cross_entropy(X @ th[:6].reshape(2, 3) + th[6:], y)
    num = ex.numerical_gradient(f, np.r_[W.ravel(), b])
    assert np.allclose(np.r_[dW.ravel(), db], num, atol=1e-7)
    assert np.allclose(db.sum(), 0, atol=1e-12)                      # rows of P - Y sum to 0


def test_train_softmax():
    X, y = three_blobs(100, seed=4)
    W, b, hist = ex.train_softmax(X, y, 3, lr=0.5, steps=400)
    assert hist[0] == pytest.approx(np.log(3)) and hist[-1] < 0.3
    pred = (X @ W + b).argmax(axis=1)
    assert ex.accuracy(y, pred) > 0.9
    Xt, yt = three_blobs(100, seed=5)                                 # fresh data from same blobs
    assert ex.accuracy(yt, (Xt @ W + b).argmax(axis=1)) > 0.85


def test_confusion_matrix():
    rng = np.random.default_rng(6)
    yt, yp = rng.integers(0, 4, 200), rng.integers(0, 4, 200)
    C = ex.confusion_matrix(yt, yp, 4)
    ref = np.zeros((4, 4), int)
    for a, b in zip(yt, yp):
        ref[a, b] += 1
    assert np.array_equal(C, ref) and C.sum() == 200
    assert np.trace(C) == np.sum(yt == yp)


# ---- Part C ----
def test_precision_recall():
    C = np.array([[5, 1, 0], [2, 3, 0], [0, 0, 0]])
    p, r = ex.precision_recall(C)
    assert np.allclose(p, [5 / 7, 3 / 4, 0]) and np.allclose(r, [5 / 6, 3 / 5, 0])
