import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _impl import load

ex = load(__file__)
import helpers  # noqa: E402  (on sys.path after load)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def tiny_gpt(seed=0, V=11, block_size=24, d=32, n_heads=4, n_layers=2, dtype=torch.float64):
    torch.manual_seed(seed)
    m = helpers.GPT(V, block_size, d, n_heads, n_layers).to(dtype)
    with torch.no_grad():   # larger random weights than the 0.02 init, so logits differ a lot
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return m.eval()


# ---- Part A: samplers ----
def test_temperature():
    z = torch.tensor([[2.0, 1.0, 0.0]])
    assert torch.allclose(ex.apply_temperature(z, 2.0), z / 2)
    p = torch.softmax(ex.apply_temperature(z, 0.5), -1)
    assert torch.allclose(p, torch.tensor([[0.8668, 0.1173, 0.0159]]), atol=1e-4)
    with pytest.raises(ValueError):
        ex.apply_temperature(z, 0.0)


def test_top_k():
    z = torch.tensor([[0.1, 3.0, -1.0, 2.0, 0.5], [5.0, 4.0, 3.0, 2.0, 1.0]])
    f = ex.top_k_filter(z, 2)
    assert torch.equal(torch.isfinite(f), torch.tensor([[False, True, False, True, False], [True, True, False, False, False]]))
    assert torch.equal(f[torch.isfinite(f)], z[torch.isfinite(f)])
    assert torch.equal(ex.top_k_filter(z, 10), z)
    assert torch.isfinite(ex.top_k_filter(z, 1)).sum() == 2


def test_top_p():
    probs = torch.tensor([[0.15, 0.5, 0.05, 0.3]])
    z = probs.log()
    keep = lambda p: torch.isfinite(ex.top_p_filter(z, p))
    assert torch.equal(keep(0.75), torch.tensor([[False, True, False, True]]))
    assert torch.equal(keep(0.85), torch.tensor([[True, True, False, True]]))
    assert torch.equal(keep(0.01), torch.tensor([[False, True, False, False]]))
    assert torch.isfinite(ex.top_p_filter(z, 1.0)).all()
    # rows are independent
    z2 = torch.cat([z, torch.zeros(1, 4)])
    assert torch.isfinite(ex.top_p_filter(z2, 0.6))[1].sum() == 3     # uniform row: any 3 of 4


def test_sample_next_distribution():
    probs = torch.tensor([0.1, 0.4, 0.2, 0.3])
    z = probs.log().expand(20000, 4)
    s = ex.sample_next(z, top_k=2, generator=gen(1))
    assert s.shape == (20000,)
    freq = torch.bincount(s, minlength=4).double() / 20000
    assert torch.allclose(freq, torch.tensor([0.0, 4 / 7, 0.0, 3 / 7], dtype=torch.float64), atol=0.02)
    s = ex.sample_next(z, temperature=0.5, generator=gen(2))
    freq = torch.bincount(s, minlength=4).double() / 20000
    want = probs.double() ** 2 / (probs.double() ** 2).sum()
    assert torch.allclose(freq, want, atol=0.02)
    assert torch.equal(ex.sample_next(z[:3], temperature=0), torch.tensor([1, 1, 1]))
    a = ex.sample_next(z[:50], top_p=0.65, generator=gen(3))
    assert set(a.tolist()) <= {1, 3}


# ---- Part B: KV cache ----
def test_forward_cached_prefill_matches_full():
    m = tiny_gpt()
    idx = torch.randint(0, 11, (2, 9), generator=gen(4))
    full, _ = m(idx)
    logits, cache = ex.forward_cached(m, idx)
    assert torch.allclose(logits, full, atol=1e-10)
    assert len(cache) == 2 and cache[0][0].shape == (2, 4, 9, 8) and cache[0][1].shape == (2, 4, 9, 8)


def test_forward_cached_incremental():
    m = tiny_gpt(seed=1)
    idx = torch.randint(0, 11, (2, 12), generator=gen(5))
    full, _ = m(idx)
    logits, cache = ex.forward_cached(m, idx[:, :5])
    outs = [logits]
    for t in range(5, 12):
        logits, cache = ex.forward_cached(m, idx[:, t:t + 1], cache)
        assert logits.shape == (2, 1, 11)
        outs.append(logits)
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-10)
    assert cache[1][0].shape[2] == 12
    # chunks of several new tokens at once must also respect causality
    l1, c1 = ex.forward_cached(m, idx[:, :4])
    l2, _ = ex.forward_cached(m, idx[:, 4:10], c1)
    assert torch.allclose(torch.cat([l1, l2], 1), full[:, :10], atol=1e-10)
    with pytest.raises(ValueError):
        ex.forward_cached(m, torch.zeros(1, 25, dtype=torch.long))


@pytest.mark.parametrize("temperature", [0, 1.0, 0.7])
def test_generate_cached_matches_nocache(temperature):
    m = tiny_gpt(seed=2)
    prompt = torch.randint(0, 11, (3, 4), generator=gen(6))
    a = helpers.generate_nocache(m, prompt, 15, temperature, gen(7))
    b = ex.generate_cached(m, prompt, 15, temperature, gen(7))
    assert b.shape == (3, 19)
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        ex.generate_cached(m, prompt, 30)


# ---- Part C: LoRA ----
def test_lora_linear():
    torch.manual_seed(8)
    base = nn.Linear(6, 5)
    lo = ex.LoRALinear(base, r=2, alpha=4.0)
    x = torch.randn(3, 6)
    assert torch.allclose(lo(x), base(x)), "B starts at zero, so LoRA starts as the identity edit"
    trainable = [p for p in lo.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == 2 * (6 + 5)
    assert lo.A.shape == (2, 6) and lo.B.shape == (5, 2) and lo.scale == 2.0
    with torch.no_grad():
        lo.B.normal_()
    dW = lo.delta_weight()
    assert dW.shape == (5, 6) and torch.linalg.matrix_rank(dW) <= 2
    merged = lo.merge()
    assert isinstance(merged, nn.Linear)
    assert torch.allclose(merged(x), lo(x), atol=1e-6)
    assert torch.allclose(lo(x), base(x) + x @ dW.T, atol=1e-6)


def test_add_lora_freezes_base():
    m = tiny_gpt(seed=9, dtype=torch.float32)
    ex.add_lora(m, r=2, alpha=2.0)
    assert isinstance(m.blocks[0].attn.qkv, ex.LoRALinear) and isinstance(m.blocks[1].attn.proj, ex.LoRALinear)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    d = 32
    assert trainable == 2 * (2 * (d + 3 * d) + 2 * (d + d))
    assert m(torch.zeros(1, 5, dtype=torch.long))[0].shape == (1, 5, 11)


# ---- Part C: SFT on a toy "reverse the string" format ----
LETTERS = "abcdef"
VOCAB = LETTERS + "=."
STOI = {c: i for i, c in enumerate(VOCAB)}


def task(n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        s = "".join(LETTERS[int(i)] for i in torch.randint(0, 6, (4,), generator=g))
        out.append((s + "=", s[::-1] + "."))
    return out


def test_make_sft_example():
    x, y = ex.make_sft_example("ab=", "ba.", STOI)
    assert x.tolist() == [STOI[c] for c in "ab=ba"]
    assert y.tolist() == [-100, -100, STOI["b"], STOI["a"], STOI["."]]
    assert x.dtype == torch.long and y.dtype == torch.long


def test_sft_loss_ignores_prompt():
    m = tiny_gpt(seed=10, V=8, dtype=torch.float32)
    x, y = ex.make_sft_example("abcd=", "dcba.", STOI)
    x, y = x[None], y[None]
    logits, _ = m(x)
    want = F.cross_entropy(logits[0, 4:], y[0, 4:])
    assert torch.allclose(ex.sft_loss(m, x, y), want, atol=1e-6)


def batch(examples):
    xs, ys = zip(*(ex.make_sft_example(p, r, STOI) for p, r in examples))
    return torch.stack(xs), torch.stack(ys)


def exact_match(m, examples):
    prompts = torch.tensor([[STOI[c] for c in p] for p, _ in examples])
    out = helpers.generate_nocache(m, prompts, 5, temperature=0)
    got = ["".join(VOCAB[i] for i in row[5:].tolist()) for row in out]
    return sum(g == r for g, (_, r) in zip(got, examples)) / len(examples)


def train(m, params, steps, lr, seed):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    data = task(2000, seed)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(data), (64,), generator=g)
        x, y = batch([data[i] for i in idx])
        loss = ex.sft_loss(m, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_sft_learns_the_format():
    torch.manual_seed(11)
    m = helpers.GPT(len(VOCAB), 16, 64, 4, 2)
    held_out = task(200, 999)
    xh, yh = batch(held_out)
    before_loss, before_acc = ex.sft_loss(m, xh, yh).item(), exact_match(m, held_out)
    m.train()
    train(m, m.parameters(), 400, 3e-3, 12)
    m.eval()
    after_loss, after_acc = ex.sft_loss(m, xh, yh).item(), exact_match(m, held_out)
    assert after_loss < 0.3 * before_loss, (before_loss, after_loss)
    assert before_acc < 0.05 and after_acc > 0.8, (before_acc, after_acc)


def test_lora_sft_improves_and_keeps_base():
    torch.manual_seed(13)
    m = helpers.GPT(len(VOCAB), 16, 64, 4, 2)
    base_state = {k: v.clone() for k, v in m.state_dict().items()}
    ex.add_lora(m, r=8, alpha=16.0)
    held_out = task(200, 998)
    xh, yh = batch(held_out)
    before = ex.sft_loss(m, xh, yh).item()
    train(m, [p for p in m.parameters() if p.requires_grad], 300, 1e-2, 14)
    after = ex.sft_loss(m, xh, yh).item()
    assert after < 0.75 * before, (before, after)
    # the frozen base weights did not move
    for blk_i, blk in enumerate(m.blocks):
        assert torch.equal(blk.attn.qkv.base.weight, base_state[f"blocks.{blk_i}.attn.qkv.weight"])
    assert torch.equal(m.tok_emb.weight, base_state["tok_emb.weight"])


# ---- Part D: scaling laws ----
def test_chinchilla_loss():
    assert math.isclose(ex.chinchilla_loss(70e9, 1.4e12), 1.9366454705587173, rel_tol=1e-9)
    # more parameters or more data always helps, and the loss never goes below E
    assert ex.chinchilla_loss(1e9, 1e11) > ex.chinchilla_loss(2e9, 1e11) > ex.chinchilla_loss(2e9, 2e11) > 1.69


def test_compute_optimal():
    for C in (1e19, 1e21, 5.76e23):
        N, D = ex.compute_optimal(C)
        assert math.isclose(6 * N * D, C, rel_tol=1e-9)
        best = ex.chinchilla_loss(N, D)
        for f in (0.5, 0.8, 1.25, 2.0):     # moving along the budget line only hurts
            assert ex.chinchilla_loss(N * f, C / (6 * N * f)) > best
    N, D = ex.compute_optimal(5.76e23)
    assert math.isclose(N, 3.219e10, rel_tol=1e-3) and math.isclose(D, 2.982e12, rel_tol=1e-3)
    # N grows like C^(beta/(alpha+beta))
    N1, _ = ex.compute_optimal(1e20)
    N2, _ = ex.compute_optimal(1e22)
    assert math.isclose(math.log10(N2 / N1) / 2, 0.28 / 0.62, rel_tol=1e-9)
