"""Workbook 15 — Generalization, overfitting & regularization.

Rules of the game
  * NumPy only. Loops are fine over folds, lambdas, trials and (in coordinate
    descent) over coordinates — but not over examples.
  * Conventions: X (n, d), y (n,), models return (w, b) with prediction X @ w + b.
    The bias b is NEVER penalized. The trick: center X and y, solve for w on the
    centered data, then recover b = mean(y) - mean(X, axis=0) @ w.
  * Objectives (note the different scalings — they match the usual libraries):
        ridge:  ||y - Xw - b||^2 + lam * ||w||_2^2
        lasso:  (1 / (2n)) ||y - Xw - b||^2 + lam * ||w||_1
  * Use np.random.default_rng(seed) for every shuffle so results are reproducible.
  * Run   ./course check 15   to grade. Each function says which test covers it.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Part A — splitting data
# ----------------------------------------------------------------------------

def train_val_test_split(n, val_frac=0.2, test_frac=0.2, seed=0):
    """Shuffle indices 0..n-1 and cut them into (train_idx, val_idx, test_idx).

    Sizes: n_val = round(n * val_frac), n_test = round(n * test_frac), the rest train.
    The three arrays are disjoint and together contain every index exactly once.
    (test_split)
    """
    raise NotImplementedError


def kfold_indices(n, k, seed=0):
    """K-fold cross-validation: return a list of k pairs (train_idx, val_idx).

    Shuffle once, then cut into k folds whose sizes differ by at most 1
    (np.array_split does exactly this). Fold i is the validation set of pair i;
    the other k-1 folds, concatenated, are its training set.
    (test_kfold)
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — ridge and lasso
# ----------------------------------------------------------------------------

def ridge_fit(X, y, lam):
    """Ridge regression in closed form (bias unpenalized):
        center:  Xc = X - mean(X),  yc = y - mean(y)
        solve:   (Xc^T Xc + lam I) w = Xc^T yc         (np.linalg.solve, no inv)
        bias:    b = mean(y) - mean(X) @ w
    Returns (w, b).   (test_ridge_*)
    """
    raise NotImplementedError


def soft_threshold(z, t):
    """S(z, t) = sign(z) * max(|z| - t, 0), element-wise. The proximal operator of
    t*|.| — it shrinks toward 0 and snaps small values to exactly 0.
    (test_soft_threshold)
    """
    raise NotImplementedError


def lasso_lambda_max(X, y):
    """Smallest lam for which the lasso solution is w = 0:
        max_j |Xc[:, j] @ yc| / n      (centered data).
    (test_lasso_*)
    """
    raise NotImplementedError


def lasso_cd(X, y, lam, n_iter=500, tol=1e-10):
    """Lasso by cyclic coordinate descent on centered data. Start at w = 0, keep the
    residual r = yc - Xc @ w up to date, and for each j:

        z_j   = ||Xc[:, j]||^2 / n
        rho_j = Xc[:, j] @ (r + Xc[:, j] * w_j) / n     # correlation with partial residual
        new   = soft_threshold(rho_j, lam) / z_j
        r    -= Xc[:, j] * (new - w_j);  w_j = new

    Stop after n_iter sweeps or when the largest change in a sweep is < tol.
    Columns with z_j == 0 stay at 0. Returns (w, b).
    (test_lasso_*)
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — model selection
# ----------------------------------------------------------------------------

def cross_val_mse(fit_fn, X, y, k=5, seed=0):
    """Mean validation MSE over the k folds of kfold_indices(len(y), k, seed).
    fit_fn(X_train, y_train) -> (w, b).   (test_cross_val)
    """
    raise NotImplementedError


def choose_lambda(X, y, lams, method="ridge", k=5, seed=0):
    """For each lam in lams, compute cross_val_mse of ridge_fit (method="ridge") or
    lasso_cd (method="lasso"). Return (best_lam, scores) with scores an array
    aligned with lams.   (test_choose_lambda)
    """
    raise NotImplementedError


def learning_curve(X_tr, y_tr, X_val, y_val, sizes, fit_fn):
    """For each m in sizes: fit on the first m training examples, record the MSE on
    those m examples and on the full validation set.
    Returns (train_mse, val_mse), two arrays of len(sizes).   (test_learning_curve)
    """
    raise NotImplementedError


def bias_variance(f, degree, n=20, noise=0.3, trials=200, x_test=None, seed=0):
    """Monte-Carlo bias-variance decomposition for polynomial regression on [-1, 1].

    For each trial: draw x ~ Uniform(-1, 1) (n points), y = f(x) + noise * N(0,1),
    fit a degree-`degree` polynomial by least squares (np.polynomial.polynomial.polyfit
    is fine), and predict at x_test (default np.linspace(-1, 1, 50)).
    With preds of shape (trials, len(x_test)) and mean prediction m = preds.mean(0):
        bias2    = mean over x_test of (m - f(x_test))^2
        variance = mean over x_test of preds.var(axis=0)
    Return (bias2, variance).   (test_bias_variance)
    """
    raise NotImplementedError
