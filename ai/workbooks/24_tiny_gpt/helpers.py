"""Provided code for workbook 24 (not an exercise).

`Block` is the pre-norm transformer block you built in workbook 23, written
with torch's built-in LayerNorm so this workbook doesn't depend on yours.
`load_text` reads Tiny Shakespeare from data/.
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def load_text(name="tinyshakespeare.txt"):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return f.read()


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n_heads = n_heads

    def forward(self, x):
        B, T, d = x.shape
        h = self.n_heads
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, T, h, d // h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # lesson 22, fused
        return self.proj(y.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    """x = x + attn(ln1(x)); x = x + mlp(ln2(x))   (lesson 23)."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
