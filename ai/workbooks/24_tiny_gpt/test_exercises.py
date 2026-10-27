import math
import pytest
import torch
import torch.nn.functional as F
from _impl import load

ex = load(__file__)
import helpers  # noqa: E402  (on sys.path after load)

TEXT = helpers.load_text()[:200_000]


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.fixture(scope="module")
def data():
    stoi, itos = ex.build_vocab(TEXT)
    return torch.tensor(ex.encode(TEXT, stoi), dtype=torch.long), stoi, itos


# ---- Part A ----
def test_vocab_and_roundtrip():
    stoi, itos = ex.build_vocab("hello world")
    assert itos == sorted(set("hello world"))
    assert all(stoi[c] == i for i, c in enumerate(itos))
    ids = ex.encode("hello world", stoi)
    assert len(ids) == 11 and ex.decode(ids, itos) == "hello world"
    assert ex.decode(torch.tensor(ids), itos) == "hello world"


def test_split():
    d = torch.arange(1000)
    tr, va = ex.train_val_split(d, 0.1)
    assert len(tr) == 900 and len(va) == 100
    assert torch.equal(tr, d[:900]) and torch.equal(va, d[900:])


def test_get_batch(data):
    d, _, _ = data
    x, y = ex.get_batch(d, 16, 8, gen(1))
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1]), "y must be x shifted by one"
    # every row really is a window of the data
    small = torch.arange(50)
    x, y = ex.get_batch(small, 10, 64, gen(2))
    assert torch.equal(y, x + 1)
    assert int(x[:, 0].max()) <= 50 - 10 - 1 and int(x.min()) >= 0
    # reproducible with a generator
    a = ex.get_batch(d, 8, 4, gen(3))[0]
    b = ex.get_batch(d, 8, 4, gen(3))[0]
    assert torch.equal(a, b)


# ---- Part B ----
def make_model(V, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(block_size=32, d=64, n_heads=4, n_layers=2)
    cfg.update(kw)
    return ex.TinyGPT(V, **cfg)


def test_model_shapes_and_loss(data):
    d, stoi, _ = data
    V = len(stoi)
    m = make_model(V)
    x, y = ex.get_batch(d, 32, 4, gen(4))
    logits, loss = m(x, y)
    assert logits.shape == (4, 32, V)
    assert torch.allclose(loss, F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)))
    assert m(x)[1] is None
    assert m(x[:, :5])[0].shape == (4, 5, V)


def test_init_is_gpt2_style():
    m = make_model(65)
    assert abs(m.head.weight.std().item() - 0.02) < 0.003
    assert abs(m.tok_emb.weight.std().item() - 0.02) < 0.003
    assert m.blocks[0].mlp[0].bias.abs().max().item() == 0.0


def test_initial_loss_is_log_vocab(data):
    d, stoi, _ = data
    V = len(stoi)
    assert math.isclose(ex.expected_initial_loss(V), math.log(V))
    assert math.isclose(ex.expected_initial_loss(2), math.log(2))
    m = make_model(V, seed=5)
    x, y = ex.get_batch(d, 32, 16, gen(5))
    with torch.no_grad():
        loss = m(x, y)[1].item()
    assert abs(loss - math.log(V)) < 0.1, f"initial loss {loss:.3f} vs ln V = {math.log(V):.3f}"


# ---- Part C ----
def test_optimizer_groups():
    m = make_model(65)
    opt = ex.make_optimizer(m, lr=3e-4, weight_decay=0.1)
    assert isinstance(opt, torch.optim.AdamW)
    g0, g1 = opt.param_groups
    assert g0["weight_decay"] == 0.1 and g1["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in g0["params"]) and all(p.dim() < 2 for p in g1["params"])
    assert len(g0["params"]) + len(g1["params"]) == len(list(m.parameters()))
    assert g0["lr"] == 3e-4 and g0["betas"] == (0.9, 0.99)


def test_overfits_single_batch_fast(data):
    d, stoi, _ = data
    m = make_model(len(stoi), seed=6)
    opt = ex.make_optimizer(m, lr=3e-3, weight_decay=0.0)
    x, y = ex.get_batch(d, 32, 4, gen(6))
    first = ex.train_step(m, opt, x, y)
    assert isinstance(first, float)
    for _ in range(150):
        last = ex.train_step(m, opt, x, y)
    assert last < 0.3, f"could not memorize one batch: loss {last:.3f} (started {first:.3f})"


def test_estimate_loss(data):
    d, stoi, _ = data
    m = make_model(len(stoi), seed=7)
    m.train()
    a = ex.estimate_loss(m, d, 32, 8, 3, gen(8))
    assert isinstance(a, float) and m.training, "restore train mode"
    b = ex.estimate_loss(m, d, 32, 8, 3, gen(8))
    assert a == b
    g = gen(8)
    want = sum(m(*ex.get_batch(d, 32, 8, g))[1].item() for _ in range(3)) / 3
    assert math.isclose(a, want, rel_tol=1e-5)
    m.eval()
    ex.estimate_loss(m, d, 32, 8, 1, gen(9))
    assert not m.training


def test_generate(data):
    d, stoi, itos = data
    m = make_model(len(stoi), seed=8, block_size=16)
    prompt = torch.tensor([ex.encode("ROMEO:", stoi)])
    out = ex.generate(m, prompt, 40, generator=gen(10))
    assert out.shape == (1, 46)
    assert torch.equal(out[:, :6], prompt)
    assert int(out.min()) >= 0 and int(out.max()) < len(stoi)
    # longer than block_size: must crop, not crash; same seed -> same text
    again = ex.generate(m, prompt, 40, generator=gen(10))
    assert torch.equal(out, again)
    # greedy decoding follows the argmax chain
    g = ex.generate(m, prompt, 5, temperature=0)
    ctx = prompt
    for _ in range(5):
        nxt = m(ctx[:, -16:])[0][:, -1].argmax(-1, keepdim=True)
        ctx = torch.cat([ctx, nxt], 1)
    assert torch.equal(g, ctx)
    assert not out.requires_grad
