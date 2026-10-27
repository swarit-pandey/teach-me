"""Workbook 14 — Logistic & softmax classification.

Rules of the game
  * NumPy only, vectorized (the only loops allowed: over training steps, and over
    parameters inside numerical_gradient).
  * Conventions:  X (n, d) float;  binary y (n,) in {0, 1};  multiclass y (n,) int in
    0..K-1.  Binary model: logits z = X @ w + b, p = sigmoid(z).  Multiclass model:
    logits Z = X @ W + b with W (d, K), b (K,), P = softmax(Z) row-wise.
  * Work with LOGITS, not probabilities, whenever you compute a loss: it's the
    numerically stable way (lesson 14 §3, lesson 12 problem 6).
  * Datasets come from helpers.py (two_blobs, three_blobs, make_blobs).
  * Run   ./course check 14   to grade. Each function says which test covers it.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Part A — binary logistic regression
# ----------------------------------------------------------------------------

def sigmoid(z):
    """1 / (1 + exp(-z)), element-wise, with NO overflow warnings for z = ±1000.

    Hint: for z >= 0 use 1/(1+exp(-z)); for z < 0 use exp(z)/(1+exp(z)). Both only
    ever exponentiate a non-positive number. np.exp(-np.abs(z)) gives you that
    quantity for every element at once; combine with np.where.
    (test_sigmoid)
    """
    raise NotImplementedError


def log_loss(z, y):
    """Mean binary cross-entropy computed from LOGITS z (not probabilities):

        -[y log sigmoid(z) + (1-y) log(1 - sigmoid(z))]  ==  log(1 + e^z) - y z

    Use np.logaddexp(0, z) for log(1 + e^z). Must be finite for |z| = 1000.
    (test_log_loss)
    """
    raise NotImplementedError


def log_loss_grad(X, y, w, b):
    """Gradient of log_loss(X @ w + b, y) w.r.t. (w, b).  Returns (dw (d,), db float).

    The famous result (lesson 14 §4):  dL/dz_i = (p_i - y_i) / n, so
        dw = X^T (p - y) / n,    db = mean(p - y).
    (test_log_loss_grad)
    """
    raise NotImplementedError


def numerical_gradient(f, theta, eps=1e-6):
    """Central-difference gradient of scalar function f at the 1-D array theta:
        g_i = (f(theta + eps e_i) - f(theta - eps e_i)) / (2 eps).
    Don't modify theta in place (or restore it if you do). A loop over i is fine.
    (test_numerical_gradient, test_log_loss_grad)
    """
    raise NotImplementedError


def train_logistic(X, y, lr=0.5, steps=500, l2=0.0):
    """Gradient descent on  log_loss + (l2/2) ||w||^2   (no penalty on b), from w=0, b=0.
    Returns (w, b, history) with history the list of steps+1 (unpenalized) losses.
    (test_train_logistic)
    """
    raise NotImplementedError


def predict_proba(X, w, b):
    """P(y = 1 | x) for each row.   (test_train_logistic)"""
    raise NotImplementedError


def accuracy(y_true, y_pred):
    """Fraction of positions where y_true == y_pred.   (test_accuracy)"""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — softmax regression (K classes)
# ----------------------------------------------------------------------------

def softmax(Z):
    """Row-wise stable softmax of Z (n, K).   (test_softmax)"""
    raise NotImplementedError


def cross_entropy(Z, y):
    """Mean cross-entropy from LOGITS Z (n, K) and integer labels y (n,):
        mean_i [ logsumexp(Z_i) - Z_i[y_i] ]
    Stable for huge logits. Hint: subtract the row max before exponentiating, and
    pick Z_i[y_i] with fancy indexing Z[np.arange(n), y].
    (test_cross_entropy)
    """
    raise NotImplementedError


def softmax_grad(X, y, W, b):
    """Gradient of cross_entropy(X @ W + b, y) w.r.t. W (d, K) and b (K,).

    Same shape as the binary case:  dZ = (P - Y) / n with Y the one-hot labels, then
        dW = X^T dZ,   db = dZ.sum(axis=0).
    (test_softmax_grad)
    """
    raise NotImplementedError


def train_softmax(X, y, K, lr=0.5, steps=500):
    """Gradient descent from W = 0, b = 0. Returns (W, b, history), history of
    length steps+1.   (test_train_softmax)
    """
    raise NotImplementedError


def confusion_matrix(y_true, y_pred, K):
    """C (K, K) int array, C[i, j] = number of examples with true class i predicted as j.
    No Python loop over examples. Hint: np.add.at(C, (y_true, y_pred), 1) or
    np.bincount(K * y_true + y_pred, minlength=K*K).reshape(K, K).
    (test_confusion_matrix)
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch
# ----------------------------------------------------------------------------

def precision_recall(C):
    """Per-class precision and recall from a confusion matrix C (rows = true).
        precision_k = C[k,k] / (column k sum)   — of everything predicted k, how much was k
        recall_k    = C[k,k] / (row k sum)      — of all true k, how much did we find
    Return (precision (K,), recall (K,)); define 0/0 as 0.
    (test_precision_recall)
    """
    raise NotImplementedError
