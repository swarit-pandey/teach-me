"""Workbook 14 — reference solutions. Try first; read after."""
import numpy as np


def sigmoid(z):
    z = np.asarray(z, float)
    e = np.exp(-np.abs(z))                       # in (0, 1]: never overflows
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def log_loss(z, y):
    z, y = np.asarray(z, float), np.asarray(y, float)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def log_loss_grad(X, y, w, b):
    X = np.asarray(X, float)
    p = sigmoid(X @ w + b)
    r = (p - y) / X.shape[0]                      # dL/dz, (n,)
    return X.T @ r, float(r.sum())


def numerical_gradient(f, theta, eps=1e-6):
    theta = np.asarray(theta, float)
    g = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta); e.flat[i] = eps
        g.flat[i] = (f(theta + e) - f(theta - e)) / (2 * eps)
    return g


def train_logistic(X, y, lr=0.5, steps=500, l2=0.0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    w, b = np.zeros(X.shape[1]), 0.0
    hist = [log_loss(X @ w + b, y)]
    for _ in range(steps):
        dw, db = log_loss_grad(X, y, w, b)
        w = w - lr * (dw + l2 * w)
        b = b - lr * db
        hist.append(log_loss(X @ w + b, y))
    return w, b, hist


def predict_proba(X, w, b):
    return sigmoid(np.asarray(X, float) @ w + b)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def softmax(Z):
    Z = np.asarray(Z, float)
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(Z, y):
    Z = np.asarray(Z, float)
    m = Z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(Z - m).sum(axis=1))
    return float(np.mean(lse - Z[np.arange(Z.shape[0]), y]))


def softmax_grad(X, y, W, b):
    X = np.asarray(X, float)
    n = X.shape[0]
    dZ = softmax(X @ W + b)
    dZ[np.arange(n), y] -= 1.0                    # P - Y without building Y
    dZ /= n
    return X.T @ dZ, dZ.sum(axis=0)


def train_softmax(X, y, K, lr=0.5, steps=500):
    X = np.asarray(X, float)
    W, b = np.zeros((X.shape[1], K)), np.zeros(K)
    hist = [cross_entropy(X @ W + b, y)]
    for _ in range(steps):
        dW, db = softmax_grad(X, y, W, b)
        W, b = W - lr * dW, b - lr * db
        hist.append(cross_entropy(X @ W + b, y))
    return W, b, hist


def confusion_matrix(y_true, y_pred, K):
    C = np.zeros((K, K), dtype=int)
    np.add.at(C, (np.asarray(y_true), np.asarray(y_pred)), 1)   # handles repeated pairs
    return C


def precision_recall(C):
    C = np.asarray(C, float)
    tp = np.diag(C)
    col, row = C.sum(axis=0), C.sum(axis=1)
    prec = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    return prec, rec
