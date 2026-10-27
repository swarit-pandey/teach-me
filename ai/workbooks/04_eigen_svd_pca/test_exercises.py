import math
import numpy as np
import pytest
from _impl import load

ex = load(__file__)


def same_dir(u, v, tol=1e-6):
    """u and v are parallel unit vectors (either sign)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    return abs(abs(u @ v) - 1) < tol


def gaussian_cloud(n=500, seed=0):
    rng = np.random.default_rng(seed)
    # anisotropic 3-D Gaussian with known covariance, rotated and shifted
    th = 0.6
    R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    L = R @ np.diag([3.0, 1.0, 0.3])
    return rng.normal(size=(n, 3)) @ L.T + np.array([5.0, -2.0, 1.0])


# ---- Part A ----
def test_char_poly():
    assert ex.char_poly_2x2([[2, 1], [1, 2]]) == (-4, 3)
    b, c = ex.char_poly_2x2([[1, 2], [3, 4]])
    assert math.isclose(b, -5) and math.isclose(c, -2)


@pytest.mark.parametrize("A", [[[2, 1], [1, 2]], [[4, 1], [2, 3]], [[3, 0], [4, 5]],
                               [[1, 2], [3, 4]], [[0, 1], [1, 0]], [[-2, 0.5], [0.3, 1]]])
def test_eig_2x2_matches_definition(A):
    vals, vecs = ex.eig_2x2(A)
    assert vals[0] >= vals[1]
    ref = sorted(np.linalg.eigvals(np.array(A, float)).real, reverse=True)
    assert np.allclose(vals, ref)
    M = np.array(A, float)
    for lam, v in zip(vals, vecs):
        v = np.array(v)
        assert math.isclose(np.linalg.norm(v), 1, rel_tol=1e-9)
        assert np.allclose(M @ v, lam * v, atol=1e-9)        # A v = λ v
    # sum = trace, product = determinant
    assert math.isclose(sum(vals), M[0, 0] + M[1, 1], abs_tol=1e-9)
    assert math.isclose(vals[0] * vals[1], np.linalg.det(M), abs_tol=1e-9)


def test_eig_2x2_edge_cases():
    with pytest.raises(ValueError):
        ex.eig_2x2([[0, -1], [1, 0]])                       # 90° rotation
    vals, vecs = ex.eig_2x2([[3, 0], [0, 3]])               # scalar matrix
    assert np.allclose(vals, [3, 3])
    vals, vecs = ex.eig_2x2([[1, 1], [0, 1]])               # shear: defective
    assert np.allclose(vals, [1, 1])
    assert same_dir(vecs[0], [1, 0]) and same_dir(vecs[1], [1, 0])


# ---- Part B ----
def test_power_iteration():
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    lams = np.array([7.0, 3.0, 2.0, -1.0, 0.5])
    S = Q @ np.diag(lams) @ Q.T
    lam, v = ex.power_iteration(S)
    assert math.isclose(lam, 7.0, rel_tol=1e-8)
    assert same_dir(v, Q[:, 0])
    assert math.isclose(np.linalg.norm(v), 1, rel_tol=1e-9)
    # non-symmetric with a negative dominant eigenvalue (v flips sign every step)
    A = np.array([[-5.0, 1.0], [0.0, 2.0]])
    lam, v = ex.power_iteration(A)
    assert math.isclose(lam, -5.0, rel_tol=1e-6)
    assert same_dir(v, [1, 0])


def test_covariance():
    X = gaussian_cloud()
    C = ex.covariance(X)
    assert C.shape == (3, 3)
    assert np.allclose(C, np.cov(X.T))
    assert np.allclose(C, C.T)
    assert np.all(np.linalg.eigvalsh(C) > 0)                 # PSD (here: PD)


def test_pca_eig_and_svd_agree():
    X = gaussian_cloud()
    Ce, ve, me = ex.pca_eig(X, 2)
    Cs, vs, ms = ex.pca_svd(X, 2)
    assert Ce.shape == Cs.shape == (2, 3)
    assert np.allclose(me, X.mean(axis=0)) and np.allclose(ms, me)
    assert np.allclose(ve, vs)
    assert ve[0] >= ve[1]
    for a, b in zip(Ce, Cs):
        assert same_dir(a, b)                                # equal up to sign
    assert np.allclose(Ce @ Ce.T, np.eye(2), atol=1e-9)       # orthonormal rows
    # true axes: variances near 9 and 1, first axis at angle 0.6 in the xy-plane
    assert same_dir(Ce[0], [math.cos(0.6), math.sin(0.6), 0], tol=1e-2)
    assert 7.5 < ve[0] < 10.5 and 0.8 < ve[1] < 1.2


def test_pca_components_are_eigenvectors():
    X = gaussian_cloud(seed=3)
    C = ex.covariance(X)
    comps, var, _ = ex.pca_svd(X, 3)
    for c, lam in zip(comps, var):
        assert np.allclose(C @ c, lam * c)


def test_explained_variance_ratio():
    X = gaussian_cloud()
    r = ex.explained_variance_ratio(X)
    assert r.shape == (3,)
    assert math.isclose(r.sum(), 1)
    assert np.all(np.diff(r) <= 1e-12)
    assert np.allclose(r * np.trace(np.cov(X.T)), np.sort(np.linalg.eigvalsh(np.cov(X.T)))[::-1])


def test_project_and_reconstruct():
    X = gaussian_cloud()
    comps, var, mean = ex.pca_eig(X, 2)
    Z = ex.project(X, comps, mean)
    assert Z.shape == (500, 2)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)          # centered scores
    assert np.allclose(np.var(Z, axis=0, ddof=1), var)       # variance along each PC
    Xr = ex.reconstruct(Z, comps, mean)
    assert Xr.shape == X.shape
    # residual is orthogonal to the kept components, and its energy is the dropped variance
    R = X - Xr
    assert np.allclose(R @ comps.T, 0, atol=1e-9)
    full = np.sort(np.linalg.eigvalsh(np.cov(X.T)))[::-1]
    assert math.isclose((R ** 2).sum() / (len(X) - 1), full[2], rel_tol=1e-8)
    # keeping every component reconstructs exactly
    c3, _, m3 = ex.pca_svd(X, 3)
    assert np.allclose(ex.reconstruct(ex.project(X, c3, m3), c3, m3), X)


def test_low_rank_approx():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(30, 20))
    for k in (1, 5, 19):
        Ak = ex.low_rank_approx(A, k)
        assert Ak.shape == A.shape
        assert np.linalg.matrix_rank(Ak) == k
    assert np.allclose(ex.low_rank_approx(A, 20), A)
    # an exactly rank-3 matrix is recovered by k = 3
    B = rng.normal(size=(15, 3)) @ rng.normal(size=(3, 12))
    assert np.allclose(ex.low_rank_approx(B, 3), B)


def test_eckart_young():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(25, 18))
    for k in (1, 4, 10):
        err = np.linalg.norm(A - ex.low_rank_approx(A, k))
        pred = ex.eckart_young_error(A, k)
        assert math.isclose(err, pred, rel_tol=1e-9)
        # no other rank-k matrix does better (spot-check random competitors)
        for t in range(20):
            B = rng.normal(size=(25, k)) @ rng.normal(size=(k, 18))
            P = B @ np.linalg.pinv(B)                       # project onto B's column space
            assert np.linalg.norm(A - P @ A) >= pred - 1e-9
    assert ex.eckart_young_error(A, 18) < 1e-9


# ---- Part C ----
def test_top_k_eig_power():
    X = gaussian_cloud(seed=5)
    C = np.cov(X.T)
    vals, vecs = ex.top_k_eig_power(C, 3)
    ref_vals, ref_vecs = np.linalg.eigh(C)
    ref_vals, ref_vecs = ref_vals[::-1], ref_vecs[:, ::-1]
    assert np.allclose(vals, ref_vals, rtol=1e-6)
    for v, r in zip(vecs, ref_vecs.T):
        assert same_dir(v, r, tol=1e-6)
