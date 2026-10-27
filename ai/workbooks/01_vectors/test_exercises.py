import math
import numpy as np
import pytest
from _impl import load

ex = load(__file__)


def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b)) and len(a) == len(b)


# ---- Part A ----
def test_dot():
    assert ex.dot([1, 2, 3], [4, 5, 6]) == 32
    assert ex.dot([3, 4], [4, -3]) == 0
    assert ex.dot([], []) == 0


def test_dot_rejects_mismatch():
    with pytest.raises(ValueError):
        ex.dot([1, 2], [1, 2, 3])


def test_norms():
    assert math.isclose(ex.norm([3, 4]), 5)
    assert math.isclose(ex.norm([3, -4], p=1), 7)
    assert math.isclose(ex.norm([3, -4], p=math.inf), 4)
    assert math.isclose(ex.norm([1, 1, 1, 1], p=3), 4 ** (1 / 3))


def test_normalize():
    u = ex.normalize([0, 3, 4])
    assert close(u, [0, 0.6, 0.8])
    with pytest.raises(ValueError):
        ex.normalize([0, 0])


def test_cosine_and_angle():
    assert math.isclose(ex.cosine_similarity([1, 0], [0, 2]), 0, abs_tol=1e-12)
    assert math.isclose(ex.cosine_similarity([1, 2], [-2, -4]), -1)
    assert math.isclose(ex.angle_degrees([1, 0], [1, math.sqrt(3)]), 60)
    # clamping: nearly-parallel vectors must not crash acos
    v = [0.1 * i + 1e-3 for i in range(50)]
    assert ex.angle_degrees(v, [3 * x for x in v]) < 1e-3


def test_project_and_decomposition():
    assert close(ex.project([2, 3], [1, 0]), [2, 0])
    par, perp = ex.orthogonal_decomposition([1, 2, 3], [1, 1, 0])
    assert close([p + q for p, q in zip(par, perp)], [1, 2, 3])
    assert abs(ex.dot(perp, [1, 1, 0])) < 1e-12
    # projection is invariant to rescaling a
    assert close(ex.project([1, 2, 3], [5, 5, 0]), par)


def test_orthogonal_basis_coords():
    c = ex.coords_in_orthogonal_basis([3, 1, 2], [[1, 1, 0], [1, -1, 0], [0, 0, 1]])
    assert close(c, [2, 1, 2])


# ---- Part B ----
VOCAB = ["king", "queen", "man", "woman", "apple", "banana", "car", "truck"]
# 4-d toy embeddings: dims ~ [royalty, gender(+male), fruit, vehicle]
E = np.array([
    [0.9, 0.8, 0.0, 0.05],
    [0.9, -0.8, 0.05, 0.0],
    [0.1, 0.8, 0.0, 0.1],
    [0.1, -0.8, 0.1, 0.0],
    [0.0, 0.05, 0.9, 0.0],
    [0.05, -0.05, 0.85, 0.1],
    [0.0, 0.1, 0.0, 0.9],
    [0.05, 0.2, 0.05, 0.95],
])


def test_cosine_matrix():
    C = ex.cosine_matrix(E)
    assert C.shape == (8, 8)
    assert np.allclose(np.diag(C), 1)
    assert np.allclose(C, C.T)
    ref = np.array([[ex_cos(a, b) for b in E] for a in E])
    assert np.allclose(C, ref)


def ex_cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def test_nearest_words():
    assert ex.nearest_words("apple", VOCAB, E, k=1) == ["banana"]
    assert ex.nearest_words("car", VOCAB, E, k=1) == ["truck"]
    out = ex.nearest_words("king", VOCAB, E, k=3)
    assert "king" not in out and len(out) == 3


def test_analogy():
    assert ex.analogy("man", "king", "woman", VOCAB, E) == "queen"
    assert ex.analogy("woman", "queen", "man", VOCAB, E) == "king"
