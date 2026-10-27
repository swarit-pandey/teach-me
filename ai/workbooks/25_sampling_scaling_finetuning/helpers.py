"""Provided code for workbook 25 (not an exercise).

`GPT` has exactly the architecture of workbooks/24_tiny_gpt/train.py, so a
checkpoint saved by train.py (data/tinygpt.pt) loads into it with
`load_tinygpt()`. `generate_nocache` is the slow reference generator your
KV-cache version must match token for token.
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n_heads = n_heads

    def forward(self, x):
        B, T, d = x.shape
        h = self.n_heads
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, T, h, d // h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, d, n_heads, n_layers):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T), ignore_index=-100)
        return logits, loss


def next_token(logits, temperature=1.0, generator=None):
    """The ONE sampling call both generators use: (B, V) logits -> (B, 1) ids."""
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    return torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=generator)


@torch.no_grad()
def generate_nocache(model, idx, max_new_tokens, temperature=1.0, generator=None):
    """Reference generation: re-run the whole sequence for every new token.
    Requires idx.shape[1] + max_new_tokens <= model.block_size."""
    if idx.shape[1] + max_new_tokens > model.block_size:
        raise ValueError("sequence would exceed block_size")
    for _ in range(max_new_tokens):
        logits = model(idx)[0][:, -1]
        idx = torch.cat([idx, next_token(logits, temperature, generator)], dim=1)
    return idx


def load_tinygpt(path=None):
    """Load the checkpoint written by workbooks/24_tiny_gpt/train.py.
    Returns (model, chars) or raises FileNotFoundError."""
    ck = torch.load(path or os.path.join(DATA, "tinygpt.pt"), map_location="cpu")
    c = ck["config"]
    m = GPT(len(ck["vocab"]), c["block_size"], c["d"], c["n_heads"], c["n_layers"])
    m.load_state_dict(ck["model"])
    return m.eval(), ck["vocab"]
