"""Workbook 03 — reference solutions. Try first; read after."""
import numpy as np


# ---- Part A ----

def shape(A):
    m = len(A)
    n = len(A[0]) if m else 0
    if any(len(r) != n for r in A):
        raise ValueError("ragged matrix")
    return (m, n)


def matmul(A, B):
    m, n = shape(A)
    n2, p = shape(B)
    if n != n2:
        raise ValueError(f"inner dimensions differ: ({m},{n}) @ ({n2},{p})")
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def outer(u, v):
    return [[a * b for b in v] for a in u]


def matmul_outer(A, B):
    m, n = shape(A)
    n2, p = shape(B)
    if n != n2:
        raise ValueError(f"inner dimensions differ: ({m},{n}) @ ({n2},{p})")
    C = [[0.0] * p for _ in range(m)]
    for k in range(n):
        col = [A[i][k] for i in range(m)]
        P = outer(col, B[k])          # rank-1 slice: column k of A times row k of B
        for i in range(m):
            for j in range(p):
                C[i][j] += P[i][j]
    return C


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# ---- Part B ----

def forward_eliminate(A, b, tol=1e-12):
    U = np.array(A, dtype=float)
    c = np.array(b, dtype=float)
    n = U.shape[0]
    for k in range(n):
        p = int(np.argmax(np.abs(U[k:, k]))) + k     # partial pivoting: biggest |entry|
        if abs(U[p, k]) < tol:
            raise np.linalg.LinAlgError("matrix is singular")
        if p != k:
            U[[k, p]] = U[[p, k]]
            c[[k, p]] = c[[p, k]]
        for i in range(k + 1, n):
            f = U[i, k] / U[k, k]                     # the multiplier l_ik
            U[i, k:] -= f * U[k, k:]
            c[i] -= f * c[k]
    return U, c


def back_substitute(U, c):
    U = np.asarray(U, dtype=float)
    c = np.asarray(c, dtype=float)
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def solve(A, b):
    U, c = forward_eliminate(A, b)
    return back_substitute(U, c)


def inverse(A, tol=1e-12):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    M = np.hstack([A, np.eye(n)])                     # [A | I]
    for k in range(n):
        p = int(np.argmax(np.abs(M[k:, k]))) + k
        if abs(M[p, k]) < tol:
            raise np.linalg.LinAlgError("matrix is singular")
        M[[k, p]] = M[[p, k]]
        M[k] /= M[k, k]
        for i in range(n):
            if i != k:
                M[i] -= M[i, k] * M[k]
    return M[:, n:]


def rref(A, tol=1e-10):
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivots, r = [], 0
    for j in range(n):
        if r == m:
            break
        p = int(np.argmax(np.abs(R[r:, j]))) + r
        if abs(R[p, j]) < tol:
            R[r:, j] = 0.0                            # clean numerical dust; free column
            continue
        R[[r, p]] = R[[p, r]]
        R[r] /= R[r, j]
        for i in range(m):
            if i != r:
                R[i] -= R[i, j] * R[r]
        pivots.append(j)
        r += 1
    R[np.abs(R) < tol] = 0.0
    return R, pivots


def rank(A, tol=1e-10):
    return len(rref(A, tol)[1])


def null_space(A, tol=1e-10):
    R, pivots = rref(A, tol)
    n = R.shape[1]
    free = [j for j in range(n) if j not in pivots]
    N = np.zeros((n, len(free)))
    for col, f in enumerate(free):
        N[f, col] = 1.0
        for i, pc in enumerate(pivots):
            N[pc, col] = -R[i, f]
    return N


# ---- Part C ----

def solve_general(A, b, tol=1e-10):
    A = np.asarray(A, dtype=float)
    n = A.shape[1]
    R, pivots = rref(np.hstack([A, np.asarray(b, dtype=float).reshape(-1, 1)]), tol)
    if n in pivots:                                   # a row reads 0 = nonzero
        return None
    x = np.zeros(n)
    for i, pc in enumerate(pivots):
        x[pc] = R[i, n]
    return x
