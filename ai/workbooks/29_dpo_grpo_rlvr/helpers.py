"""Provided (non-exercise) code for workbook 29. You don't need to edit this file.

Two toy worlds, both with a tiny causal transformer (TinyGPT, same as workbook 28):

1. LETTERS (for DPO) — identical to workbook 28. Tokens 0 = BOS, 1..6 = a..f;
   completions are BOS + 8 letters from a fixed Markov chain; labellers' hidden
   utility true_score(y) = 3 − |#a − 3|; preference_pairs() samples Bradley–Terry labels.

2. ARITHMETIC (for GRPO / RL with verifiable rewards). Tokens 0..9 are digits,
   10 = '+', 11 = '=', 12 = BOS. A prompt is  BOS a + b =  (a, b single digits) and
   the answer is two digits: the tens and ones of a + b (e.g. 7+8= -> 1 5).
   The SFT model imitates a SLOPPY teacher who is right only 40% of the time
   (otherwise off by 1 or 2), so sampling from it is right only ~40% of the time.
   A verifier can check any answer exactly — no reward model needed.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BOS, A = 0, 1
LETTERS = "abcdef"
VOCAB_SIZE = 7
GEN_LEN = 8
BLOCK_SIZE = GEN_LEN + 1

# transition probabilities between letters (rows: from a..f, cols: to a..f)
REF_T = torch.tensor([
    [0.10, 0.35, 0.15, 0.10, 0.20, 0.10],
    [0.25, 0.05, 0.30, 0.15, 0.10, 0.15],
    [0.20, 0.15, 0.05, 0.35, 0.15, 0.10],
    [0.15, 0.20, 0.15, 0.05, 0.30, 0.15],
    [0.20, 0.10, 0.20, 0.15, 0.05, 0.30],
    [0.30, 0.15, 0.10, 0.20, 0.20, 0.05],
])


# ----------------------------------------------------------------------------
# A tiny GPT (same architecture as lessons 23–24, just small)
# ----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.qkv, self.proj, self.n_head = nn.Linear(d, 3 * d), nn.Linear(d, d), n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Decoder-only transformer.  forward(idx) -> logits (B, T, vocab);
    hidden(idx) -> final hidden states (B, T, d_model) (after the last LayerNorm)."""

    def __init__(self, vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, d_model=32, n_head=2, n_layer=2):
        super().__init__()
        self.d_model = d_model
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def hidden(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x)

    def forward(self, idx):
        return self.lm_head(self.hidden(idx))


# ----------------------------------------------------------------------------
# Data, sampling, pretraining
# ----------------------------------------------------------------------------

def sample_reference(n, generator=None, length=GEN_LEN):
    """n sequences from the reference Markov chain: LongTensor (n, 1 + length), BOS first."""
    cur = torch.multinomial(torch.full((n, 6), 1 / 6), 1, generator=generator).squeeze(1)
    out = [torch.full((n,), BOS), cur + 1]
    for _ in range(length - 1):
        cur = torch.multinomial(REF_T[cur], 1, generator=generator).squeeze(1)
        out.append(cur + 1)
    return torch.stack(out, 1)


def count_a(tokens):
    """Number of 'a' tokens in each sequence -> float tensor (B,)."""
    return (tokens == A).sum(1).float()


def true_score(tokens):
    """The synthetic labellers' hidden utility: 3 − |#a − 3| (best with exactly three 'a's)."""
    return 3.0 - (count_a(tokens) - 3.0).abs()


def preference_pairs(n, seed=0, score_fn=true_score):
    """n preference pairs (chosen, rejected), each LongTensor (n, 9).
    Both completions come from the reference distribution; the labeller prefers
    the first with probability sigmoid(score(y1) − score(y2))  (Bradley–Terry)."""
    g = torch.Generator().manual_seed(seed)
    y1, y2 = sample_reference(n, g), sample_reference(n, g)
    first = torch.rand(n, generator=g) < torch.sigmoid(score_fn(y1) - score_fn(y2))
    chosen = torch.where(first[:, None], y1, y2)
    rejected = torch.where(first[:, None], y2, y1)
    return chosen, rejected


@torch.no_grad()
def generate(model, n, length=GEN_LEN, temperature=1.0, generator=None, prompts=None):
    """Sample n completions of `length` tokens after BOS (or after the given prompts)."""
    idx = prompts.clone() if prompts is not None else torch.full((n, 1), BOS, dtype=torch.long)
    for _ in range(length):
        logits = model(idx)[:, -1, :] / temperature
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)
        idx = torch.cat([idx, nxt], 1)
    return idx


def pretrain(model, data, steps=400, batch=64, lr=3e-3, seed=0):
    """Next-token cross-entropy training (this is the SFT step of the pipeline)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        x = data[torch.randint(len(data), (batch,), generator=g)]
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


_CACHE = {}


def make_sft_model(seed=0, steps=400):
    """A TinyGPT pretrained on the reference corpus (cached per process; returns a fresh copy)."""
    key = (seed, steps)
    if key not in _CACHE:
        torch.manual_seed(seed)
        data = sample_reference(4000, torch.Generator().manual_seed(seed + 100))
        _CACHE[key] = pretrain(TinyGPT(), data, steps=steps, seed=seed).state_dict()
    m = TinyGPT()
    m.load_state_dict(_CACHE[key])
    return m


def decode(tokens):
    """LongTensor of token ids -> string, e.g. 'abcab...' ('^' for BOS)."""
    return "".join("^" if int(t) == BOS else LETTERS[int(t) - 1] for t in tokens)


# ----------------------------------------------------------------------------
# The arithmetic world
# ----------------------------------------------------------------------------

PLUS, EQ, ABOS = 10, 11, 12
ARITH_VOCAB = 13
PROMPT_LEN = 5            # BOS a + b =
ANSWER_LEN = 2            # tens, ones


def arith_prompts(n, generator=None):
    """n random prompts  [BOS, a, +, b, =]  as a LongTensor (n, 5)."""
    ab = torch.randint(0, 10, (n, 2), generator=generator)
    return torch.stack([torch.full((n,), ABOS), ab[:, 0], torch.full((n,), PLUS), ab[:, 1], torch.full((n,), EQ)], 1)


def sloppy_teacher_data(n, p_correct=0.4, seed=0):
    """SFT data: prompt + the teacher's answer. Correct with prob p_correct, otherwise
    off by ±1 or ±2 (kept within 0..18)."""
    g = torch.Generator().manual_seed(seed)
    pr = arith_prompts(n, g)
    s = pr[:, 1] + pr[:, 3]
    off = torch.tensor([-2, -1, 1, 2])[torch.randint(0, 4, (n,), generator=g)]
    wrong = s + off
    wrong = torch.where((wrong < 0) | (wrong > 18), s - off, wrong)
    ans = torch.where(torch.rand(n, generator=g) < p_correct, s, wrong)
    return torch.cat([pr, (ans // 10)[:, None], (ans % 10)[:, None]], 1)


def pretrain_answers(model, data, steps=600, batch=128, lr=3e-3, seed=0):
    """Like pretrain(), but the loss is only on the two answer tokens (prompts are random)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        x = data[torch.randint(len(data), (batch,), generator=g)]
        logits = model(x[:, :-1])[:, PROMPT_LEN - 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, PROMPT_LEN:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def new_arith_model():
    return TinyGPT(vocab_size=ARITH_VOCAB, block_size=PROMPT_LEN + ANSWER_LEN, d_model=48, n_head=4, n_layer=2)


def make_arith_sft_model(seed=0, steps=600):
    """A TinyGPT fine-tuned on the sloppy teacher (cached per process; returns a fresh copy)."""
    key = ("arith", seed, steps)
    if key not in _CACHE:
        torch.manual_seed(seed)
        data = sloppy_teacher_data(20000, seed=seed + 100)
        _CACHE[key] = pretrain_answers(new_arith_model(), data, steps=steps, seed=seed).state_dict()
    m = new_arith_model()
    m.load_state_dict(_CACHE[key])
    return m


@torch.no_grad()
def sampled_accuracy(model, verifier, n=1000, seed=123, temperature=1.0):
    """Fraction of n sampled answers (fresh random prompts) that `verifier` accepts."""
    g = torch.Generator().manual_seed(seed)
    toks = generate(model, n, length=ANSWER_LEN, temperature=temperature, generator=g, prompts=arith_prompts(n, g))
    return verifier(toks).mean().item()


def arith_decode(tokens):
    """[12, 7, 10, 8, 11, 1, 5] -> '7+8=15'."""
    m = {PLUS: "+", EQ: "=", ABOS: ""}
    return "".join(m.get(int(t), str(int(t))) for t in tokens)
