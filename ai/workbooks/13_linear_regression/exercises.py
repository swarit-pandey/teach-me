"""Workbook 13 — Linear regression from scratch.

Rules of the game
  * NumPy only, vectorized: no Python loops over examples or features. The only
    loop allowed is the gradient-descent loop over *steps*.
  * Conventions used everywhere below:
        X : (n, d) float array, one example per row
        y : (n,)   targets          (a 1-D vector — NOT (n, 1); see lesson 12 §4)
        w : (d,)   weights,  b : float bias,   prediction  y_hat = X @ w + b
        loss  L(w, b) = mean((y_hat - y)**2)        (MSE, no 1/2 factor)
  * Never call np.linalg.inv. Solve linear systems with np.linalg.solve /
    np.linalg.lstsq (or your own triangular solve in Part C).
  * Run   ./course check 13   to grade. Each function says which test covers it.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Part A — the model, the loss, its gradient, gradient descent
# ----------------------------------------------------------------------------

def predict(X, w, b):
    """y_hat = X @ w + b, shape (n,).   (test_predict)"""
    raise NotImplementedError


def mse(y_hat, y):
    """Mean squared error. Raise ValueError if the shapes differ — this is the guard
    against the (n,) vs (n,1) broadcasting bug from lesson 12.   (test_mse)
    """
    raise NotImplementedError


def mse_grad(X, y, w, b):
    """Gradient of L(w,b) = mean((X@w + b - y)**2). Returns (dw, db) with dw: (d,), db: float.

    Derivation (lesson 13 §4): with residual r = X@w + b - y,
        dL/dw = (2/n) X^T r,     dL/db = (2/n) sum(r).
    (test_mse_grad_matches_numerical)
    """
    raise NotImplementedError


def gd_fit(X, y, lr=0.1, steps=1000):
    """Batch gradient descent from w = 0, b = 0.

    Returns (w, b, history) where history is a list of steps+1 floats:
    history[0] is the loss at the start, history[t] the loss after t updates.
    (test_gd_*)
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — closed form, evaluation, preprocessing
# ----------------------------------------------------------------------------

def add_bias_column(X):
    """Prepend a column of ones: (n, d) -> (n, d+1). Then theta = [b, w].   (test_normal_equations)"""
    raise NotImplementedError


def fit_normal_equations(X, y):
    """Closed-form least squares via the normal equations  A^T A theta = A^T y,
    with A = add_bias_column(X). Returns (w, b).

    Use np.linalg.solve — NOT inv. (Why: solve is faster and more accurate; inv
    computes n columns you never need.)
    (test_normal_equations)
    """
    raise NotImplementedError


def r2_score(y, y_hat):
    """Coefficient of determination  R^2 = 1 - SS_res / SS_tot, where
    SS_res = sum((y - y_hat)^2) and SS_tot = sum((y - mean(y))^2).
    1 = perfect, 0 = no better than predicting the mean, negative = worse.
    (test_r2)
    """
    raise NotImplementedError


class Standardizer:
    """Per-feature standardization z = (x - mean) / std, with statistics learned on
    TRAINING data only (fit), then applied to any data (transform).

    Constant features (std == 0) must map to 0, not NaN.
    (test_standardizer)
    """

    def fit(self, X):
        """Store self.mean_ and self.std_ (both shape (d,)); return self."""
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def unscale(self, w, b):
        """You trained (w, b) on standardized features. Return (w_orig, b_orig) that
        give the SAME predictions on raw X:   X @ w_orig + b_orig == Z @ w + b.

        Hint: expand  sum_j w_j (x_j - mu_j) / s_j + b.
        (test_standardizer_unscale)
        """
        raise NotImplementedError


def polynomial_features(x, degree):
    """1-D x (n,) -> (n, degree) matrix with columns x, x^2, ..., x^degree.
    (No constant column: the bias b plays that role.)
    Hint: broadcasting  x[:, None] ** np.arange(1, degree + 1).
    (test_polynomial_features)
    """
    raise NotImplementedError


def condition_number(X):
    """Condition number of the MSE Hessian for features X (with a bias column):
    lambda_max / lambda_min of A^T A, A = add_bias_column(X).
    Large = elongated loss valley = slow gradient descent (lesson 06).
    Hint: np.linalg.eigvalsh for symmetric matrices.
    (test_condition_number)
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — stretch: least squares via QR, the numerically stable way
# ----------------------------------------------------------------------------

def back_substitution(R, z):
    """Solve R x = z for upper-triangular R (k, k). A loop over rows (from the
    bottom up) is allowed here; each row's update should be one dot product.
    (test_back_substitution)
    """
    raise NotImplementedError


def fit_qr(X, y):
    """Least squares via QR: A = QR (np.linalg.qr, reduced), then R theta = Q^T y.
    Returns (w, b). Never forms A^T A, so it squares nothing: condition number
    kappa(A) instead of kappa(A)^2.
    (test_fit_qr)
    """
    raise NotImplementedError
