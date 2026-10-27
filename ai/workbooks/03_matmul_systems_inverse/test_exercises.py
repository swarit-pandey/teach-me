import numpy as np
import pytest
from _impl import load

ex = load(__file__)
rng = np.random.default_rng(3)


def arr(M):
    return np.array(M, dtype=float)


# ---- Part A ----
def test_shape():
    assert ex.shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
    assert ex.shape([[7]]) == (1, 1)
    with pytest.raises(ValueError):
        ex.shape([[1, 2], [3]])


def test_matmul_small_and_order_matters():
    A, B = [[1, 2], [3, 4]], [[0, 1], [1, 0]]
    assert np.allclose(ex.matmul(A, B), [[2, 1], [4, 3]])   # B on the right swaps columns
    assert np.allclose(ex.matmul(B, A), [[3, 4], [1, 2]])   # B on the left swaps rows
    with pytest.raises(ValueError):
        ex.matmul([[1, 2, 3]], [[1, 2, 3]])


def test_matmul_random_shapes():
    for m, n, p in [(1, 1, 1), (2, 3, 4), (5, 1, 3), (4, 4, 4), (3, 6, 1)]:
        A, B = rng.normal(size=(m, n)), rng.normal(size=(n, p))
        C = ex.matmul(A.tolist(), B.tolist())
        assert arr(C).shape == (m, p)
        assert np.allclose(C, A @ B)


def test_matmul_does_not_mutate():
    A, B = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
    ex.matmul(A, B)
    assert A == [[1, 2], [3, 4]] and B == [[5, 6], [7, 8]]


def test_outer_is_rank_one():
    P = arr(ex.outer([1, 2, 3], [4, 5]))
    assert P.shape == (3, 2)
    assert np.allclose(P, np.outer([1, 2, 3], [4, 5]))
    assert np.linalg.matrix_rank(P) == 1


def test_matmul_outer_agrees():
    for m, n, p in [(2, 3, 4), (3, 1, 3), (4, 5, 2)]:
        A, B = rng.normal(size=(m, n)), rng.normal(size=(n, p))
        assert np.allclose(ex.matmul_outer(A.tolist(), B.tolist()), A @ B)
    with pytest.raises(ValueError):
        ex.matmul_outer([[1, 2]], [[1, 2]])


def test_identity_is_neutral():
    I = ex.identity(3)
    assert np.allclose(I, np.eye(3))
    A = rng.normal(size=(3, 3)).tolist()
    assert np.allclose(ex.matmul(A, I), A) and np.allclose(ex.matmul(I, A), A)


def test_associative_not_commutative():
    A, B, C = (rng.normal(size=(3, 3)).tolist() for _ in range(3))
    assert np.allclose(ex.matmul(ex.matmul(A, B), C), ex.matmul(A, ex.matmul(B, C)))
    assert not np.allclose(ex.matmul(A, B), ex.matmul(B, A))


# ---- Part B ----
def test_forward_eliminate_upper_triangular():
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    U, c = ex.forward_eliminate(A, b)
    U, c = arr(U), arr(c)
    assert np.allclose(np.tril(U, -1), 0)
    assert np.allclose(np.linalg.solve(U, c), [2, 3, -1])
    # partial pivoting: first pivot is the largest |entry| of column 0
    assert abs(U[0, 0]) == 3
    assert A == [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]   # input untouched


def test_back_substitute():
    U = [[2, 1, 1], [0, 3, -1], [0, 0, 4]]
    x = arr(ex.back_substitute(U, [9, 5, 8]))
    assert np.allclose(x, np.linalg.solve(arr(U), [9, 5, 8]))


def test_solve_classic():
    x = ex.solve([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3])
    assert np.allclose(x, [2, 3, -1])


def test_solve_needs_pivoting():
    # zero in the top-left: naive elimination divides by 0
    A, b = [[0, 1], [1, 1]], [2, 3]
    assert np.allclose(ex.solve(A, b), [1, 2])
    # tiny pivot: without pivoting this loses all accuracy
    A, b = [[1e-20, 1], [1, 1]], [1, 2]
    assert np.allclose(ex.solve(A, b), np.linalg.solve(arr(A), b))


def test_solve_random_vs_numpy():
    for n in [1, 2, 5, 10]:
        A, b = rng.normal(size=(n, n)), rng.normal(size=n)
        x = arr(ex.solve(A, b))
        assert x.shape == (n,)
        assert np.allclose(x, np.linalg.solve(A, b))
        assert np.allclose(A @ x, b)


def test_solve_singular_raises():
    with pytest.raises(np.linalg.LinAlgError):
        ex.solve([[1, 2], [2, 4]], [3, 6])


def test_inverse():
    A = [[4, 7], [2, 6]]
    assert np.allclose(ex.inverse(A), np.linalg.inv(arr(A)))
    for n in [1, 3, 6]:
        M = rng.normal(size=(n, n))
        Mi = arr(ex.inverse(M))
        assert np.allclose(Mi, np.linalg.inv(M))
        assert np.allclose(M @ Mi, np.eye(n)) and np.allclose(Mi @ M, np.eye(n))
    with pytest.raises(np.linalg.LinAlgError):
        ex.inverse([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_inverse_of_product_reverses_order():
    A, B = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    assert np.allclose(ex.inverse(A @ B), arr(ex.inverse(B)) @ arr(ex.inverse(A)))


def test_rref_known():
    A = [[1, 2, 1, 1], [2, 4, 0, 6], [1, 2, 2, -1]]
    R, piv = ex.rref(A)
    assert list(piv) == [0, 2]
    assert np.allclose(R, [[1, 2, 0, 3], [0, 0, 1, -2], [0, 0, 0, 0]])


def test_rank_vs_numpy():
    cases = [np.eye(3), np.zeros((2, 3)), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.]]),
             np.outer([1, 2, 3], [4, 5]), rng.normal(size=(4, 6))]
    # a 5x5 of rank 2, built as a product of thin matrices
    cases.append(rng.normal(size=(5, 2)) @ rng.normal(size=(2, 5)))
    for M in cases:
        assert ex.rank(M) == np.linalg.matrix_rank(M)


def test_null_space():
    A = np.array([[1, 2, 1, 1], [2, 4, 0, 6], [1, 2, 2, -1.]])
    N = arr(ex.null_space(A))
    assert N.shape == (4, 2)
    assert np.allclose(A @ N, 0)
    assert np.linalg.matrix_rank(N) == 2              # independent
    # rank–nullity on random wide/low-rank matrices
    for M in [rng.normal(size=(3, 7)), rng.normal(size=(6, 2)) @ rng.normal(size=(2, 5))]:
        N = arr(ex.null_space(M))
        assert N.shape == (M.shape[1], M.shape[1] - np.linalg.matrix_rank(M))
        assert np.allclose(M @ N, 0, atol=1e-8)
        if N.shape[1]:
            assert np.linalg.matrix_rank(N) == N.shape[1]


def test_null_space_trivial():
    N = arr(ex.null_space(np.eye(3)))
    assert N.shape == (3, 0)


# ---- Part C ----
def test_solve_general():
    A = np.array([[1, 2, 1, 1], [2, 4, 0, 6], [1, 2, 2, -1.]])
    b = A @ np.array([1., -1, 2, 0.5])
    x = ex.solve_general(A, b)
    assert x is not None and np.allclose(A @ x, b)
    assert ex.solve_general(A, [1, 0, 0]) is None      # not in the column space
    assert ex.solve_general([[1, 1], [1, 1]], [1, 2]) is None
    x = ex.solve_general([[2, 0], [0, 4]], [2, 8])
    assert np.allclose(x, [1, 2])
