"""Workbook 06 — reference solutions. Try first; read after."""
import numpy as np


def numerical_gradient(f, x, h=1e-5):
    x = np.array(x, dtype=float)          # private copy
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + h; fp = f(x)
        x[i] = old - h; fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g


def directional_derivative(f, x, u, h=1e-5):
    u = np.asarray(u, dtype=float)
    n = np.linalg.norm(u)
    if n == 0:
        raise ValueError("direction must be non-zero")
    return float(numerical_gradient(f, x, h) @ (u / n))


def quad_form_grad(A, x):
    # d/dx_k sum_ij A_ij x_i x_j = sum_j A_kj x_j + sum_i A_ik x_i
    return (A + A.T) @ x


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    r = x[1] - x[0] ** 2
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * r, 200 * r])


def gradient_descent(grad, x0, lr, steps):
    x = np.array(x0, dtype=float)
    hist = [x.copy()]
    for _ in range(steps):
        x = x - lr * grad(x)
        hist.append(x.copy())
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e12:
            break
    return np.array(hist)


def max_stable_lr(A):
    return 2.0 / np.linalg.eigvalsh(A)[-1]


def optimal_lr_and_rate(A):
    lam = np.linalg.eigvalsh(A)
    lo, hi = lam[0], lam[-1]
    return 2.0 / (lo + hi), (hi - lo) / (hi + lo)


def gd_converges(A, b, lr, steps=2000, tol=1e-6):
    xstar = np.linalg.solve(A, b)
    hist = gradient_descent(lambda x: A @ x - b, np.zeros(len(b)), lr, steps)
    return bool(np.all(np.isfinite(hist[-1])) and np.linalg.norm(hist[-1] - xstar) < tol)


def hessian_fd(f, x, h=1e-4):
    x = np.array(x, dtype=float)
    d = x.size
    H = np.zeros((d, d))
    E = np.eye(d) * h
    for i in range(d):
        for j in range(d):
            H[i, j] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j])
                       - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) / (4 * h * h)
    return (H + H.T) / 2


def classify_critical_point(H, tol=1e-8):
    lam = np.linalg.eigvalsh((H + H.T) / 2)
    if np.all(lam > tol):
        return "minimum"
    if np.all(lam < -tol):
        return "maximum"
    if np.any(lam > tol) and np.any(lam < -tol):
        return "saddle"
    return "degenerate"
