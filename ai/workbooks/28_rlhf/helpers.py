"""Provided (non-exercise) code for workbook 28: a tiny causal transformer, a toy
"language", sampling, pretraining, and synthetic human preferences.
You don't need to edit this file — but skim it: everything RLHF does is on top of it.

The toy world
  * Tokens: 0 = BOS, 1..6 = the letters a..f.  A "completion" is BOS + 8 letters.
  * The "pretraining corpus" comes from a fixed Markov chain over letters (REF_T).
    A TinyGPT trained on it (make_sft_model) is our SFT / reference policy π_ref.
  * Synthetic labellers prefer completions with a few 'a's, but find too many
    spammy:  true_score(y) = 3 − |#a(y) − 3|   (peaks at exactly three 'a's).
    Preference labels are sampled from the Bradley–Terry model on that score.
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


class ValueNet(nn.Module):
    """The PPO critic: a TinyGPT body with a scalar head at EVERY position.
    forward(tokens) -> (B, T): values[:, t] estimates the return of the completion
    given the prefix tokens[:, :t+1]."""

    def __init__(self, **kw):
        super().__init__()
        self.body = TinyGPT(**kw)
        self.head = nn.Linear(self.body.d_model, 1)

    def forward(self, tokens):
        return self.head(self.body.hidden(tokens)).squeeze(-1)


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
