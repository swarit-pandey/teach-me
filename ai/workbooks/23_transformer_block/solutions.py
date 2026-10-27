"""Workbook 23 — reference solutions. Try first; read after."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)     # biased variance
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


class MLP(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or 4 * d
        self.fc = nn.Linear(d, hidden)
        self.proj = nn.Linear(hidden, d)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d={d} not divisible by n_heads={n_heads}")
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n_heads = n_heads

    def forward(self, x):
        B, T, d = x.shape
        h, dh = self.n_heads, d // self.n_heads
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v))   # (B, h, T, dh)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)                      # (B, h, T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        w = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        y = (w @ v).transpose(1, 2).reshape(B, T, d)                          # merge heads
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = LayerNorm(d)
        self.mlp = MLP(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # communicate: tokens read from each other
        x = x + self.mlp(self.ln2(x))      # compute: each token on its own
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, d, n_heads, n_layers):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln_f = LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.block_size}")
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss


def block_param_count(d):
    # ln1, ln2: 2d each | attn: qkv 3d^2+3d, proj d^2+d | mlp: fc 4d^2+4d, proj 4d^2+d
    return 12 * d * d + 13 * d


def gpt_param_count(vocab_size, block_size, d, n_layers):
    return vocab_size * d + block_size * d + n_layers * block_param_count(d) + 2 * d + d * vocab_size


def sinusoidal_positions(T, d):
    p = torch.arange(T, dtype=torch.float64)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float64)[None, :]
    ang = p / (10000.0 ** (i / d))
    pe = torch.zeros(T, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe.float()


def apply_rope(x, base=10000.0):
    T, d = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float64, device=x.device) / d)   # (d/2,)
    ang = torch.arange(T, dtype=torch.float64, device=x.device)[:, None] * theta          # (T, d/2)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
