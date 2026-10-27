import math
import numpy as np
import pytest
from _impl import load

ex = load(__file__)


def rng(seed=0):
    return np.random.default_rng(seed)


def ref_attention(Q, K, V, mask=None):
    """Slow, obviously-correct loop version (single batch element)."""
    n_q, n_k = Q.shape[0], K.shape[0]
    out = np.zeros((n_q, V.shape[1]))
    W = np.zeros((n_q, n_k))
    for i in range(n_q):
        s = np.array([Q[i] @ K[j] / math.sqrt(Q.shape[1]) for j in range(n_k)])
        allowed = np.ones(n_k, bool) if mask is None else mask[i]
        e = np.where(allowed, np.exp(s - s[allowed].max()), 0.0)
        W[i] = e / e.sum()
        out[i] = W[i] @ V
    return out, W


# ---- Part A ----
def test_softmax_basic():
    p = ex.softmax(np.array([2.0, 1.0, 0.0]))
    assert np.allclose(p, np.exp([2, 1, 0]) / np.exp([2, 1, 0]).sum())
    assert np.isclose(p[0], 0.6652409557748219)


def test_softmax_stable_and_shift_invariant():
    x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    p = ex.softmax(x)
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(-1), 1)
    assert np.allclose(p[0], ex.softmax(np.array([0.0, 1.0, 2.0])))
    z = rng(1).normal(size=(4, 5))
    assert np.allclose(ex.softmax(z + 7.5), ex.softmax(z))


def test_softmax_axis_and_neg_inf():
    z = rng(2).normal(size=(3, 4))
    assert np.allclose(ex.softmax(z, axis=0).sum(0), 1)
    p = ex.softmax(np.array([0.0, -np.inf, 1.0]))
    assert p[1] == 0.0 and np.isclose(p.sum(), 1)


def test_causal_mask():
    m = ex.causal_mask(4)
    assert m.dtype == bool and m.shape == (4, 4)
    assert m[3, 0] and m[2, 2] and not m[0, 1] and m.sum() == 10


def test_attention_matches_reference():
    r = rng(3)
    Q, K, V = r.normal(size=(5, 8)), r.normal(size=(7, 8)), r.normal(size=(7, 3))
    out, W = ex.attention(Q, K, V)
    rout, rW = ref_attention(Q, K, V)
    assert out.shape == (5, 3) and W.shape == (5, 7)
    assert np.allclose(W, rW) and np.allclose(out, rout)
    assert np.allclose(W.sum(-1), 1)


def test_attention_special_cases():
    V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # identical scores -> plain average of the values
    out, W = ex.attention(np.zeros((1, 2)), np.ones((3, 2)), V)
    assert np.allclose(W, 1 / 3) and np.allclose(out, V.mean(0))
    # one key dominates -> output is (almost) that key's value
    K = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 0.0]])
    out, _ = ex.attention(np.array([[1.0, 0.0]]), K, V)
    assert np.allclose(out, V[1], atol=1e-6)
    # the lesson's worked example
    out, W = ex.attention(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), V)
    assert np.allclose(W, [[0.4011, 0.1978, 0.4011]], atol=1e-4)
    assert np.allclose(out, [[3.0, 4.0]], atol=1e-9)


def test_attention_batched_and_mask():
    r = rng(4)
    Q, K, V = r.normal(size=(2, 3, 6, 4)), r.normal(size=(2, 3, 6, 4)), r.normal(size=(2, 3, 6, 5))
    m = ex.causal_mask(6)
    out, W = ex.attention(Q, K, V, m)
    assert out.shape == (2, 3, 6, 5) and W.shape == (2, 3, 6, 6)
    assert np.all(W[..., ~m] == 0)
    for b in range(2):
        for h in range(3):
            ro, rw = ref_attention(Q[b, h], K[b, h], V[b, h], m)
            assert np.allclose(out[b, h], ro) and np.allclose(W[b, h], rw)


def test_scaling_keeps_unit_variance():
    # with unit-variance q, k components, raw scores have variance d; scaled ones 1
    r = rng(5)
    d = 256
    Q, K = r.normal(size=(400, d)), r.normal(size=(400, d))
    raw = (Q @ K.T).ravel()
    assert 200 < raw.var() < 312
    _, W = ex.attention(Q, K, np.eye(400))
    # scaled scores are moderate, so no row is anywhere near one-hot
    assert W.max() < 0.2


def test_causal_future_does_not_leak():
    r = rng(6)
    n, d = 7, 8
    X = r.normal(size=(n, d))
    Wq, Wk, Wv = (r.normal(size=(d, d)) / math.sqrt(d) for _ in range(3))
    out1, W = ex.self_attention(X, Wq, Wk, Wv, causal=True)
    X2 = X.copy()
    X2[4:] = r.normal(size=(3, d))          # rewrite the "future"
    out2, _ = ex.self_attention(X2, Wq, Wk, Wv, causal=True)
    assert np.allclose(out1[:4], out2[:4])
    assert not np.allclose(out1[4:], out2[4:])
    assert np.allclose(np.triu(W, 1), 0)
    # first token can only see itself
    assert np.allclose(out1[0], X[0] @ Wv)


def test_permutation_equivariance():
    r = rng(7)
    n, d = 6, 4
    X = r.normal(size=(n, d))
    Wq, Wk, Wv = (r.normal(size=(d, d)) for _ in range(3))
    perm = r.permutation(n)
    out, W = ex.self_attention(X, Wq, Wk, Wv)
    outp, Wp = ex.self_attention(X[perm], Wq, Wk, Wv)
    assert np.allclose(outp, out[perm])
    assert np.allclose(Wp, W[perm][:, perm])


# ---- Part B ----
def test_split_merge_heads():
    X = np.arange(2 * 3 * 8).reshape(2, 3, 8).astype(float)
    H = ex.split_heads(X, 4)
    assert H.shape == (2, 4, 3, 2)
    assert np.array_equal(H[1, 2, 0], X[1, 0, 4:6])     # head 2 = columns 4..5
    assert np.array_equal(ex.merge_heads(H), X)
    assert ex.split_heads(X[0], 2).shape == (2, 3, 4)
    with pytest.raises(ValueError):
        ex.split_heads(X, 3)


def mha_ref(X, Wq, Wk, Wv, Wo, h, causal):
    n, d = X.shape
    dh = d // h
    m = np.tril(np.ones((n, n), bool)) if causal else None
    outs = []
    for i in range(h):
        sl = slice(i * dh, (i + 1) * dh)
        o, _ = ref_attention(X @ Wq[:, sl], X @ Wk[:, sl], X @ Wv[:, sl], m)
        outs.append(o)
    return np.concatenate(outs, axis=1) @ Wo


@pytest.mark.parametrize("causal", [False, True])
def test_multi_head_matches_loop(causal):
    r = rng(8)
    B, n, d, h = 2, 5, 12, 3
    X = r.normal(size=(B, n, d))
    Ws = [r.normal(size=(d, d)) / math.sqrt(d) for _ in range(4)]
    out, W = ex.multi_head_attention(X, *Ws, n_heads=h, causal=causal)
    assert out.shape == (B, n, d) and W.shape == (B, h, n, n)
    for b in range(B):
        assert np.allclose(out[b], mha_ref(X[b], *Ws, h, causal))


def test_one_head_equals_single_head():
    r = rng(9)
    n, d = 5, 6
    X = r.normal(size=(n, d))
    Wq, Wk, Wv, Wo = (r.normal(size=(d, d)) for _ in range(4))
    out, _ = ex.multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=1)
    o1, _ = ex.self_attention(X, Wq, Wk, Wv)
    assert np.allclose(out, o1 @ Wo)


# ---- Part C ----
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="PyTorch not installed (./course setup)")


@needs_torch
def test_torch_attention_matches_numpy_and_grad():
    r = rng(10)
    Q, K, V = r.normal(size=(2, 5, 4)), r.normal(size=(2, 5, 4)), r.normal(size=(2, 5, 3))
    m = np.tril(np.ones((5, 5), bool))
    T = [torch.tensor(a, requires_grad=True) for a in (Q, K, V)]
    out, W = ex.attention_torch(*T, mask=torch.tensor(m))
    ref, refW = ex_np_ref_batched(Q, K, V, m)
    assert torch.allclose(out, torch.tensor(ref)) and torch.allclose(W, torch.tensor(refW))
    out.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in T)
    # softmax rows sum to 1, so sum(out) doesn't depend on the scores... except via V
    assert torch.allclose(T[2].grad.sum(), torch.tensor(2.0 * 5 * 3, dtype=torch.float64))


def ex_np_ref_batched(Q, K, V, m):
    outs, Ws = zip(*(ref_attention(Q[b], K[b], V[b], m) for b in range(Q.shape[0])))
    return np.stack(outs), np.stack(Ws)


@needs_torch
def test_torch_attention_matches_sdpa():
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(0)
    Q, K, V = (torch.randn(3, 4, 6, 8, generator=g) for _ in range(3))
    m = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    out, _ = ex.attention_torch(Q, K, V, mask=m)
    assert torch.allclose(out, F.scaled_dot_product_attention(Q, K, V, attn_mask=m), atol=1e-5)
    out, _ = ex.attention_torch(Q, K, V)
    assert torch.allclose(out, F.scaled_dot_product_attention(Q, K, V), atol=1e-5)


@needs_torch
@pytest.mark.parametrize("causal", [False, True])
def test_torch_mha_matches_numpy_and_nn(causal):
    r = rng(11)
    B, n, d, h = 2, 5, 8, 2
    X = r.normal(size=(B, n, d))
    Ws = [r.normal(size=(d, d)) / math.sqrt(d) for _ in range(4)]
    out_np, _ = ex_np_mha(X, Ws, h, causal)
    out, W = ex.multi_head_attention_torch(torch.tensor(X), *[torch.tensor(w) for w in Ws], n_heads=h, causal=causal)
    assert out.shape == (B, n, d) and W.shape == (B, h, n, n)
    assert torch.allclose(out, torch.tensor(out_np))
    # torch's own module, with our weights copied in (it computes x @ W.T)
    mha = torch.nn.MultiheadAttention(d, h, bias=False, batch_first=True).double()
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.tensor(np.concatenate([w.T for w in Ws[:3]])))
        mha.out_proj.weight.copy_(torch.tensor(Ws[3].T))
    am = torch.triu(torch.ones(n, n, dtype=torch.bool), 1) if causal else None   # True = BLOCK
    ref, _ = mha(torch.tensor(X), torch.tensor(X), torch.tensor(X), attn_mask=am, need_weights=False)
    assert torch.allclose(out, ref, atol=1e-10)


def ex_np_mha(X, Ws, h, causal):
    return np.stack([mha_ref(X[b], *Ws, h, causal) for b in range(X.shape[0])]), None
