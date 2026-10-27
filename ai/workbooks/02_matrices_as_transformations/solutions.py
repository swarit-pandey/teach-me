"""Workbook 02 — reference solutions. Try first; read after."""
import math
import numpy as np


# ---- Part A ----
def shape(A):
    m = len(A)
    n = len(A[0]) if m else 0
    if any(len(r) != n for r in A):
        raise ValueError("ragged matrix")
    return (m, n)


def matvec_cols(A, x):
    m, n = shape(A)
    if len(x) != n:
        raise ValueError(f"A is {m}x{n} but x has length {len(x)}")
    out = [0.0] * m
    for j in range(n):          # walk the columns...
        for i in range(m):      # ...adding x[j] * column j into the output
            out[i] += x[j] * A[i][j]
    return out


def matvec_rows(A, x):
    m, n = shape(A)
    if len(x) != n:
        raise ValueError(f"A is {m}x{n} but x has length {len(x)}")
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def det2(A):
    if shape(A) != (2, 2):
        raise ValueError("det2 needs a 2x2 matrix")
    (a, b), (c, d) = A
    return a * d - b * c


def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    # columns: (1,0) -> (c, s);  (0,1) -> (-s, c)
    return [[c, -s], [s, c]]


def scaling(sx, sy):
    return [[sx, 0.0], [0.0, sy]]


def shear(k):
    return [[1.0, k], [0.0, 1.0]]


def reflection(theta):
    c2, s2 = math.cos(2 * theta), math.sin(2 * theta)
    return [[c2, s2], [s2, -c2]]


def projection(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c * c, c * s], [c * s, s * s]]   # u uᵀ


# ---- Part B ----
def transform_points(A, P):
    return np.asarray(P, dtype=float) @ np.asarray(A, dtype=float).T


def is_linear(f, n, trials=20, seed=0, tol=1e-8):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        x, y = rng.normal(size=n), rng.normal(size=n)
        a, b = rng.normal(size=2)
        if not np.allclose(f(a * x + b * y), a * f(x) + b * f(y), atol=tol, rtol=1e-6):
            return False
    return True


def shoelace_area(P):
    P = np.asarray(P, dtype=float)
    Q = np.roll(P, -1, axis=0)
    return 0.5 * float(np.sum(P[:, 0] * Q[:, 1] - Q[:, 0] * P[:, 1]))


def unit_square_image_area(A):
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return shoelace_area(transform_points(A, sq))


# ---- Part C ----
def matrix_of(f, n):
    cols = [np.asarray(f(np.eye(n)[j]), dtype=float) for j in range(n)]
    return np.stack(cols, axis=1)


def orientation(A, tol=1e-12):
    d = det2([list(r) for r in np.asarray(A, dtype=float)])
    return "preserved" if d > tol else "flipped" if d < -tol else "collapsed"
