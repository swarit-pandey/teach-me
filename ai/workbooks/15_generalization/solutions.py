"""Workbook 15 — reference solutions. Try first; read after."""
import numpy as np


def train_val_test_split(n, val_frac=0.2, test_frac=0.2, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    n_val, n_test = int(round(n * val_frac)), int(round(n * test_frac))
    return perm[n_val + n_test:], perm[:n_val], perm[n_val:n_val + n_test]


def kfold_indices(n, k, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    return [(np.concatenate(folds[:i] + folds[i + 1:]), folds[i]) for i in range(k)]


def _center(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    mx, my = X.mean(axis=0), y.mean()
    return X - mx, y - my, mx, my


def ridge_fit(X, y, lam):
    Xc, yc, mx, my = _center(X, y)
    d = Xc.shape[1]
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)
    return w, float(my - mx @ w)


def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso_lambda_max(X, y):
    Xc, yc, _, _ = _center(X, y)
    return float(np.max(np.abs(Xc.T @ yc)) / len(yc))


def lasso_cd(X, y, lam, n_iter=500, tol=1e-10):
    Xc, yc, mx, my = _center(X, y)
    n, d = Xc.shape
    w = np.zeros(d)
    r = yc.copy()                                  # residual for w = 0
    z = (Xc ** 2).sum(axis=0) / n
    for _ in range(n_iter):
        biggest = 0.0
        for j in range(d):
            if z[j] == 0:
                continue
            xj = Xc[:, j]
            rho = xj @ (r + xj * w[j]) / n         # what w_j alone would explain
            new = soft_threshold(rho, lam) / z[j]
            if new != w[j]:
                r -= xj * (new - w[j])
                biggest = max(biggest, abs(new - w[j]))
                w[j] = new
        if biggest < tol:
            break
    return w, float(my - mx @ w)


def cross_val_mse(fit_fn, X, y, k=5, seed=0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    errs = []
    for tr, va in kfold_indices(len(y), k, seed):
        w, b = fit_fn(X[tr], y[tr])
        errs.append(np.mean((X[va] @ w + b - y[va]) ** 2))
    return float(np.mean(errs))


def choose_lambda(X, y, lams, method="ridge", k=5, seed=0):
    fit = ridge_fit if method == "ridge" else lasso_cd
    scores = np.array([cross_val_mse(lambda A, t, lam=lam: fit(A, t, lam), X, y, k, seed) for lam in lams])
    return lams[int(np.argmin(scores))], scores


def learning_curve(X_tr, y_tr, X_val, y_val, sizes, fit_fn):
    tr, va = [], []
    for m in sizes:
        w, b = fit_fn(X_tr[:m], y_tr[:m])
        tr.append(np.mean((X_tr[:m] @ w + b - y_tr[:m]) ** 2))
        va.append(np.mean((X_val @ w + b - y_val) ** 2))
    return np.array(tr), np.array(va)


def bias_variance(f, degree, n=20, noise=0.3, trials=200, x_test=None, seed=0):
    rng = np.random.default_rng(seed)
    xt = np.linspace(-1, 1, 50) if x_test is None else np.asarray(x_test, float)
    P = np.polynomial.polynomial
    preds = np.empty((trials, len(xt)))
    for t in range(trials):
        x = rng.uniform(-1, 1, n)
        y = f(x) + noise * rng.normal(size=n)
        preds[t] = P.polyval(xt, P.polyfit(x, y, degree))
    m = preds.mean(axis=0)
    return float(np.mean((m - f(xt)) ** 2)), float(np.mean(preds.var(axis=0)))
