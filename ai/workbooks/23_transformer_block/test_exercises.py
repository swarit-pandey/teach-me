import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _impl import load

ex = load(__file__)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ---- Part A: LayerNorm ----
def test_layernorm_matches_torch():
    torch.manual_seed(0)
    ln = ex.LayerNorm(16).double()
    assert ln.weight.shape == (16,) and ln.bias.shape == (16,)
    assert torch.equal(ln.weight.detach(), torch.ones(16, dtype=torch.float64))
    with torch.no_grad():
        ln.weight.normal_()
        ln.bias.normal_()
    x = torch.randn(3, 5, 16, dtype=torch.float64, generator=gen(1)) * 4 + 2
    ref = F.layer_norm(x, (16,), ln.weight, ln.bias, eps=1e-5)
    assert torch.allclose(ln(x), ref, atol=1e-10)


def test_layernorm_normalizes_and_backprops():
    ln = ex.LayerNorm(32)
    x = (torch.randn(4, 32, generator=gen(2)) * 10 + 5).requires_grad_()
    y = ln(x)
    assert torch.allclose(y.mean(-1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(((y - y.mean(-1, keepdim=True)) ** 2).mean(-1), torch.ones(4), atol=1e-3)
    (y[:, 0] * 3).sum().backward()
    assert x.grad is not None and ln.weight.grad is not None and ln.bias.grad is not None
    # LayerNorm output is invariant to shifting the input, so grads sum to ~0 per row
    assert torch.allclose(x.grad.sum(-1), torch.zeros(4), atol=1e-4)


# ---- MLP ----
def test_mlp_shapes_and_params():
    m = ex.MLP(8)
    assert isinstance(m.fc, nn.Linear) and m.fc.out_features == 32
    assert isinstance(m.proj, nn.Linear) and m.proj.out_features == 8
    assert sum(p.numel() for p in m.parameters()) == 8 * 32 + 32 + 32 * 8 + 8
    x = torch.randn(2, 3, 8, generator=gen(3))
    assert torch.allclose(m(x), m.proj(F.gelu(m.fc(x))))
    assert ex.MLP(8, hidden=10).fc.out_features == 10


def test_mlp_is_position_wise():
    m = ex.MLP(8)
    x = torch.randn(1, 5, 8, generator=gen(4))
    full = m(x)
    for t in range(5):
        assert torch.allclose(full[:, t], m(x[:, t:t + 1])[:, 0], atol=1e-6)


# ---- CausalSelfAttention ----
@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_attention_matches_nn_multihead(n_heads):
    torch.manual_seed(5)
    d, T = 16, 7
    att = ex.CausalSelfAttention(d, n_heads).double()
    assert att.qkv.weight.shape == (3 * d, d) and att.proj.weight.shape == (d, d)
    ref = nn.MultiheadAttention(d, n_heads, batch_first=True).double()
    with torch.no_grad():
        ref.in_proj_weight.copy_(att.qkv.weight)
        ref.in_proj_bias.copy_(att.qkv.bias)
        ref.out_proj.weight.copy_(att.proj.weight)
        ref.out_proj.bias.copy_(att.proj.bias)
    x = torch.randn(2, T, d, dtype=torch.float64, generator=gen(6))
    block_future = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    want, _ = ref(x, x, x, attn_mask=block_future, need_weights=False)
    assert torch.allclose(att(x), want, atol=1e-10)


def test_attention_is_causal():
    torch.manual_seed(7)
    att = ex.CausalSelfAttention(12, 3)
    x = torch.randn(2, 9, 12, generator=gen(8))
    x2 = x.clone()
    x2[:, 5:] = torch.randn(2, 4, 12, generator=gen(9))
    y, y2 = att(x), att(x2)
    assert torch.allclose(y[:, :5], y2[:, :5], atol=1e-6)
    assert not torch.allclose(y[:, 5:], y2[:, 5:])


def test_attention_bad_heads():
    with pytest.raises(ValueError):
        ex.CausalSelfAttention(10, 3)


# ---- Block ----
def test_block_is_prenorm_residual():
    torch.manual_seed(10)
    blk = ex.Block(16, 4)
    for name in ("ln1", "attn", "ln2", "mlp"):
        assert hasattr(blk, name), f"Block needs attribute {name}"
    x = torch.randn(2, 6, 16, generator=gen(11))
    h = x + blk.attn(blk.ln1(x))
    want = h + blk.mlp(blk.ln2(h))
    assert torch.allclose(blk(x), want, atol=1e-6)


def test_block_identity_when_branches_zeroed():
    blk = ex.Block(8, 2)
    with torch.no_grad():
        for lin in (blk.attn.proj, blk.mlp.proj):
            lin.weight.zero_()
            lin.bias.zero_()
    x = torch.randn(3, 5, 8, generator=gen(12))
    assert torch.allclose(blk(x), x)


# ---- GPT ----
def small_gpt(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=11, block_size=8, d=16, n_heads=2, n_layers=2)
    cfg.update(kw)
    return ex.GPT(**cfg), cfg


def test_gpt_forward_and_loss():
    m, cfg = small_gpt()
    idx = torch.randint(0, 11, (3, 8), generator=gen(13))
    tgt = torch.randint(0, 11, (3, 8), generator=gen(14))
    logits, loss = m(idx)
    assert logits.shape == (3, 8, 11) and loss is None
    logits, loss = m(idx, tgt)
    assert torch.allclose(loss, F.cross_entropy(logits.view(-1, 11), tgt.view(-1)))
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    # shorter sequences are fine; longer ones are not
    assert m(idx[:, :3])[0].shape == (3, 3, 11)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 9, dtype=torch.long))


def test_gpt_causality():
    m, _ = small_gpt(seed=1)
    m.eval()
    idx = torch.randint(0, 11, (2, 8), generator=gen(15))
    idx2 = idx.clone()
    idx2[:, 4:] = (idx2[:, 4:] + 1) % 11
    a, _ = m(idx)
    b, _ = m(idx2)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6), "changing future tokens changed past logits"
    assert not torch.allclose(a[:, 4:], b[:, 4:])


def test_gpt_uses_positions():
    # Without positions attention is permutation-equivariant (lesson 22): for a
    # sequence of identical tokens, every position would get the same logits.
    m, _ = small_gpt(seed=2)
    assert m.pos_emb.weight.shape == (8, 16)
    idx = torch.tensor([[3, 3, 3, 3, 3]])
    logits, _ = m(idx)
    assert not torch.allclose(logits[0, 0], logits[0, 4], atol=1e-4)
    with torch.no_grad():
        m.pos_emb.weight.zero_()
    logits, _ = m(idx)
    assert torch.allclose(logits[0, 0], logits[0, 4], atol=1e-5)


# ---- Part B ----
@pytest.mark.parametrize("d,h", [(8, 2), (16, 4), (48, 3)])
def test_block_param_formula(d, h):
    assert ex.block_param_count(d) == sum(p.numel() for p in ex.Block(d, h).parameters())


@pytest.mark.parametrize("V,bs,d,h,L", [(11, 8, 16, 2, 2), (65, 64, 48, 4, 3), (100, 32, 24, 3, 1)])
def test_gpt_param_formula(V, bs, d, h, L):
    m = ex.GPT(V, bs, d, h, L)
    assert ex.gpt_param_count(V, bs, d, L) == sum(p.numel() for p in m.parameters())


def test_param_formula_gpt2_small():
    # GPT-2 small with an untied head: 124M + 38.6M
    assert ex.gpt_param_count(50257, 1024, 768, 12) == 163037184


# ---- Part C ----
def test_sinusoidal_positions():
    pe = ex.sinusoidal_positions(50, 16)
    assert pe.shape == (50, 16) and pe.dtype == torch.float32
    assert torch.allclose(pe[0], torch.tensor([0.0, 1.0] * 8))
    assert math.isclose(pe[3, 0].item(), math.sin(3), abs_tol=1e-6)
    assert math.isclose(pe[3, 3].item(), math.cos(3 / 10000 ** (2 / 16)), abs_tol=1e-6)
    # every row has squared norm d/2
    assert torch.allclose((pe ** 2).sum(-1), torch.full((50,), 8.0), atol=1e-4)
    # PE[p] . PE[p+k] depends only on the offset k
    dots = [(pe[p] @ pe[p + 5]).item() for p in range(0, 40, 7)]
    assert max(dots) - min(dots) < 1e-4


def test_rope_rotates():
    x = torch.randn(2, 10, 8, dtype=torch.float64, generator=gen(16))
    y = ex.apply_rope(x)
    assert y.shape == x.shape
    assert torch.allclose(y[:, 0], x[:, 0])                          # position 0: no rotation
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1))            # rotations keep length
    assert not torch.allclose(y[:, 1:], x[:, 1:])


def test_rope_scores_depend_on_offset_only():
    g = gen(17)
    q, k = torch.randn(8, dtype=torch.float64, generator=g), torch.randn(8, dtype=torch.float64, generator=g)
    T = 20
    rq = ex.apply_rope(q.expand(T, 8).clone())
    rk = ex.apply_rope(k.expand(T, 8).clone())
    s = lambda m, n: (rq[m] @ rk[n]).item()
    assert math.isclose(s(5, 2), s(15, 12), rel_tol=1e-9)
    assert math.isclose(s(3, 3), (q @ k).item(), rel_tol=1e-9)
    assert not math.isclose(s(5, 2), s(5, 4), rel_tol=1e-6)
