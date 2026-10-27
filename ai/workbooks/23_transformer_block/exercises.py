"""Workbook 23 — The transformer block, module by module (PyTorch).

Rules of the game
  * Shapes: activations are (B, T, d): batch, time (sequence position), width.
  * Build every module from nn.Linear / nn.Embedding / nn.Parameter and plain
    tensor ops. Do NOT use nn.LayerNorm, F.layer_norm, nn.MultiheadAttention or
    F.scaled_dot_product_attention inside your modules: the tests use them as
    the reference you are checked against.
  * Attribute names matter (the tests copy weights in and count parameters):
    use exactly the names given in each docstring.
  * Part A: the four modules and the GPT. Part B: parameter counting.
    Part C: positional encodings (sinusoidal, RoPE).
  * Run   ./course check 23   to grade.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Part A — modules
# ----------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm over the last dimension (lesson 18).

    Parameters: self.weight (ones, shape (d,)), self.bias (zeros, shape (d,)).
    forward: (x - mean) / sqrt(var + eps) * weight + bias, where mean and the
    BIASED variance (divide by d, not d - 1) are over the last dim.
    """

    def __init__(self, d, eps=1e-5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class MLP(nn.Module):
    """Position-wise feed-forward network: Linear(d, 4d) -> GELU -> Linear(4d, d).

    Attributes: self.fc = nn.Linear(d, hidden), self.proj = nn.Linear(hidden, d),
    with hidden = 4 * d unless given. Use F.gelu (the exact erf version).
    """

    def __init__(self, d, hidden=None):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Attributes: self.qkv = nn.Linear(d, 3 * d)  (its output columns are
    [q | k | v], each d wide), self.proj = nn.Linear(d, d), self.n_heads.
    forward(x): (B, T, d) -> (B, T, d).
      1. q, k, v = self.qkv(x).split(d, dim=-1)
      2. reshape each to (B, n_heads, T, d // n_heads)
      3. scores = q @ k^T / sqrt(dh), mask the future with -inf, softmax
      4. weights @ v, merge heads back to (B, T, d), apply self.proj
    Raise ValueError in __init__ if d % n_heads != 0.
    Hint: register the causal mask once with
      self.register_buffer("mask", torch.tril(torch.ones(T_max, T_max, dtype=torch.bool)))
    or just build torch.tril(...) of size T in forward (simpler; fine here).
    """

    def __init__(self, d, n_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Block(nn.Module):
    """Pre-norm transformer block.

    Attributes: self.ln1, self.attn, self.ln2, self.mlp  (your classes above).
    forward:  x = x + attn(ln1(x));  x = x + mlp(ln2(x));  return x
    """

    def __init__(self, d, n_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class GPT(nn.Module):
    """A GPT: embeddings -> n_layers Blocks -> final LayerNorm -> logits.

    Attributes:
      self.tok_emb = nn.Embedding(vocab_size, d)
      self.pos_emb = nn.Embedding(block_size, d)        (learned positions)
      self.blocks  = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
      self.ln_f    = LayerNorm(d)
      self.head    = nn.Linear(d, vocab_size, bias=False)
      self.block_size = block_size
    forward(idx, targets=None):
      idx (B, T) int64 with T <= block_size (else raise ValueError).
      x = tok_emb(idx) + pos_emb(arange(T)); blocks; ln_f; logits = head(x).
      If targets (B, T) is given, loss = F.cross_entropy over all B*T positions,
      else loss = None. Return (logits, loss) with logits (B, T, vocab_size).
    """

    def __init__(self, vocab_size, block_size, d, n_heads, n_layers):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx, targets=None):
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — counting parameters
# ----------------------------------------------------------------------------

def block_param_count(d):
    """Number of parameters in one Block of width d (biases included), as a
    closed-form expression in d. Derive it on paper first (lesson section 7)."""
    raise NotImplementedError


def gpt_param_count(vocab_size, block_size, d, n_layers):
    """Closed-form parameter count of your GPT (untied head, no head bias)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — positional encodings
# ----------------------------------------------------------------------------

def sinusoidal_positions(T, d):
    """The (T, d) sinusoidal table from "Attention Is All You Need":
        PE[p, 2i]   = sin(p / 10000^(2i/d))
        PE[p, 2i+1] = cos(p / 10000^(2i/d))
    for p = 0..T-1, i = 0..d/2-1 (d even). Return a float32 tensor.
    """
    raise NotImplementedError


def apply_rope(x, base=10000.0):
    """Rotary position embedding on x of shape (..., T, d), d even.

    Pair up dimensions (2i, 2i+1) and rotate each pair at position p by the
    angle p * theta_i with theta_i = base^(-2i/d):
        y[2i]   = x[2i] cos(a) - x[2i+1] sin(a)
        y[2i+1] = x[2i] sin(a) + x[2i+1] cos(a)
    Hint: x[..., 0::2] and x[..., 1::2] give the pair halves; torch.stack(...,
    dim=-1).flatten(-2) interleaves them back.
    """
    raise NotImplementedError
