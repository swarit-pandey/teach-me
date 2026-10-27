import math
import numpy as np
import pytest
from _impl import load

ex = load(__file__)
RNG = np.random.default_rng(2)


def close(a, b, tol=1e-9):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b, atol=tol)


# ---- Part A ----
def test_shape():
    assert ex.shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
    with pytest.raises(ValueError):
        ex.shape([[1, 2], [3]])


def test_matvec_both_views_agree_with_numpy():
    for m, n in [(2, 2), (3, 2), (2, 4), (1, 5)]:
        A = RNG.normal(size=(m, n)).tolist()
        x = RNG.normal(size=n).tolist()
        want = np.array(A) @ np.array(x)
        assert close(ex.matvec_cols(A, x), want)
        assert close(ex.matvec_rows(A, x), want)
    assert close(ex.matvec_cols([[2, 1], [1, 3]], [1, 2]), [4, 7])


def test_matvec_columns_are_images_of_basis():
    A = [[2, -1], [1, 3]]
    assert close(ex.matvec_cols(A, [1, 0]), [2, 1])
    assert close(ex.matvec_rows(A, [0, 1]), [-1, 3])


def test_matvec_rejects_mismatch():
    with pytest.raises(ValueError):
        ex.matvec_cols([[1, 2], [3, 4]], [1, 2, 3])
    with pytest.raises(ValueError):
        ex.matvec_rows([[1, 2], [3, 4]], [1])


def test_det2():
    assert ex.det2([[3, 1], [1, 2]]) == 5
    assert ex.det2([[1, 2], [2, 4]]) == 0
    assert ex.det2([[0, 1], [1, 0]]) == -1
    with pytest.raises(ValueError):
        ex.det2([[1, 2, 3], [4, 5, 6]])


def test_rotation():
    R = ex.rotation(math.pi / 2)
    assert close(ex.matvec_cols(R, [1, 0]), [0, 1])
    assert close(ex.matvec_cols(R, [0, 1]), [-1, 0])
    t = 0.7
    R = np.array(ex.rotation(t))
    assert close(R.T @ R, np.eye(2))                 # preserves lengths and angles
    assert math.isclose(ex.det2(R.tolist()), 1)
    assert close(np.array(ex.rotation(0.3)) @ np.array(ex.rotation(0.4)), R)  # angles add


def test_scale_and_shear():
    assert close(ex.matvec_cols(ex.scaling(2, -3), [1, 1]), [2, -3])
    S = ex.shear(1.5)
    assert close(ex.matvec_cols(S, [1, 0]), [1, 0])
    assert close(ex.matvec_cols(S, [2, 2]), [5, 2])
    assert math.isclose(ex.det2(S), 1)


def test_reflection():
    F = np.array(ex.reflection(math.pi / 4))         # across y = x: swaps coordinates
    assert close(F, [[0, 1], [1, 0]])
    t = 1.1
    F = np.array(ex.reflection(t))
    u = np.array([math.cos(t), math.sin(t)])
    assert close(F @ u, u)                            # the mirror line is fixed
    assert close(F @ np.array([-u[1], u[0]]), -np.array([-u[1], u[0]]))  # its normal flips
    assert close(F @ F, np.eye(2))                    # reflecting twice = identity
    assert math.isclose(ex.det2(F.tolist()), -1)


def test_projection():
    P = np.array(ex.projection(math.pi / 4))
    assert close(P @ [3, 1], [2, 2])
    t = -0.4
    P = np.array(ex.projection(t))
    assert close(P @ P, P)                            # projecting twice changes nothing
    assert abs(ex.det2(P.tolist())) < 1e-12           # space collapses to a line
    v = np.array([0.3, -2.0])
    assert abs((v - P @ v) @ np.array([math.cos(t), math.sin(t)])) < 1e-12  # residual ⟂ line


# ---- Part B ----
def test_transform_points():
    P = RNG.normal(size=(50, 2))
    A = np.array([[2.0, -1.0], [0.5, 3.0]])
    out = ex.transform_points(A, P)
    assert out.shape == (50, 2)
    assert close(out, np.array([A @ p for p in P]))
    B = RNG.normal(size=(3, 2))                       # R^2 -> R^3 works too
    assert ex.transform_points(B, P).shape == (50, 3)


def test_is_linear():
    M = RNG.normal(size=(3, 4))
    assert ex.is_linear(lambda x: M @ x, 4)
    assert not ex.is_linear(lambda x: M @ x + 1.0, 4)        # affine: moves the origin
    assert not ex.is_linear(lambda x: x ** 2, 3)
    assert not ex.is_linear(lambda x: np.abs(x), 3)
    assert not ex.is_linear(lambda x: np.maximum(x, 0), 3)    # ReLU is NOT linear


def test_shoelace():
    sq = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert math.isclose(ex.shoelace_area(sq), 1)
    assert math.isclose(ex.shoelace_area(sq[::-1]), -1)
    assert math.isclose(ex.shoelace_area([[0, 0], [4, 0], [0, 3]]), 6)


def test_area_scaling_is_det():
    for _ in range(10):
        A = RNG.normal(size=(2, 2))
        assert math.isclose(ex.unit_square_image_area(A), np.linalg.det(A), abs_tol=1e-12)
    assert math.isclose(ex.unit_square_image_area([[3, 1], [1, 2]]), 5)
    assert math.isclose(ex.unit_square_image_area(ex.reflection(0.3)), -1)
    # any shape scales by the same factor: a triangle under A
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    tri = np.array([[0, 0], [2, 0], [0, 1.0]])
    assert math.isclose(ex.shoelace_area(ex.transform_points(A, tri)), 5 * ex.shoelace_area(tri))


# ---- Part C ----
def test_matrix_of():
    M = RNG.normal(size=(3, 2))
    assert close(ex.matrix_of(lambda x: M @ x, 2), M)
    swap = lambda x: np.array([x[1], x[0], x[0] + x[1]])
    assert close(ex.matrix_of(swap, 2), [[0, 1], [1, 0], [1, 1]])


def test_orientation():
    assert ex.orientation([[2, 0], [0, 3]]) == "preserved"
    assert ex.orientation(ex.reflection(0.2)) == "flipped"
    assert ex.orientation([[1, 2], [2, 4]]) == "collapsed"
    assert ex.orientation(ex.rotation(2.5)) == "preserved"
