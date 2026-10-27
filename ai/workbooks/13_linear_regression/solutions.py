"""Workbook 13 — reference solutions. Try first; read after."""
import numpy as np


def predict(X, w, b):
    return np.asarray(X, float) @ np.asarray(w, float) + b


def mse(y_hat, y):
    y_hat, y = np.asarray(y_hat, float), np.asarray(y, float)
    if y_hat.shape != y.shape:
        raise ValueError(f"shape mismatch {y_hat.shape} vs {y.shape}")
    return float(np.mean((y_hat - y) ** 2))


def mse_grad(X, y, w, b):
    X = np.asarray(X, float)
    r = predict(X, w, b) - y                      # residuals, (n,)
    n = X.shape[0]
    return (2.0 / n) * (X.T @ r), (2.0 / n) * float(r.sum())


def gd_fit(X, y, lr=0.1, steps=1000):
    X, y = np.asarray(X, float), np.asarray(y, float)
    w, b = np.zeros(X.shape[1]), 0.0
    history = [mse(predict(X, w, b), y)]
    for _ in range(steps):
        dw, db = mse_grad(X, y, w, b)
        w, b = w - lr * dw, b - lr * db
        history.append(mse(predict(X, w, b), y))
    return w, b, history


def add_bias_column(X):
    X = np.asarray(X, float)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_normal_equations(X, y):
    A = add_bias_column(X)
    theta = np.linalg.solve(A.T @ A, A.T @ np.asarray(y, float))
    return theta[1:], float(theta[0])


def r2_score(y, y_hat):
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


class Standardizer:
    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def _safe_std(self):
        return np.where(self.std_ > 0, self.std_, 1.0)

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self._safe_std()

    def unscale(self, w, b):
        # sum_j w_j (x_j - mu_j)/s_j + b  =  sum_j (w_j/s_j) x_j + (b - sum_j w_j mu_j / s_j)
        s = self._safe_std()
        w_orig = np.asarray(w, float) / s
        return w_orig, float(b - w_orig @ self.mean_)


def polynomial_features(x, degree):
    x = np.asarray(x, float)
    return x[:, None] ** np.arange(1, degree + 1)


def condition_number(X):
    A = add_bias_column(X)
    ev = np.linalg.eigvalsh(A.T @ A)              # ascending
    return float(ev[-1] / ev[0])


def back_substitution(R, z):
    k = R.shape[0]
    x = np.zeros(k)
    for i in range(k - 1, -1, -1):
        x[i] = (z[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    return x


def fit_qr(X, y):
    A = add_bias_column(X)
    Q, R = np.linalg.qr(A)                        # A = QR, Q: (n, d+1) orthonormal columns
    theta = back_substitution(R, Q.T @ np.asarray(y, float))
    return theta[1:], float(theta[0])
