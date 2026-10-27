"""Workbook 25 — Sampling, KV cache, LoRA/SFT and scaling laws.

Rules of the game
  * PyTorch. `helpers.py` provides `GPT` (the lesson-24 architecture),
    `next_token` (the single sampling call shared by both generators) and the
    slow reference `generate_nocache`.
  * Logits are (B, V) tensors unless stated otherwise. Filters return logits
    with removed entries set to -inf (so a following softmax gives them 0).
  * Part A: samplers. Part B: KV-cache generation. Part C: LoRA and supervised
    fine-tuning. Part D: Chinchilla scaling-law arithmetic (plain Python).
  * Run   ./course check 25   to grade.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import next_token


# ----------------------------------------------------------------------------
# Part A — samplers
# ----------------------------------------------------------------------------

def apply_temperature(logits, temperature):
    """Divide logits by temperature. Raise ValueError if temperature <= 0."""
    raise NotImplementedError


def top_k_filter(logits, k):
    """Keep the k largest logits in each row, set the rest to -inf.
    (Ties at the k-th value may all be kept.) k larger than V keeps everything.
    Hint: torch.topk(logits, k).values[..., -1:] is the k-th largest per row."""
    raise NotImplementedError


def top_p_filter(logits, p):
    """Nucleus filter: keep the SMALLEST set of most-likely tokens whose total
    probability is >= p; set the rest to -inf. The single most likely token is
    always kept.
    Hint: sort descending, softmax, compute the probability mass strictly
    BEFORE each token (cumsum - probs); drop tokens where that mass >= p; then
    scatter the mask back to the original order with mask.scatter(-1, order, mask).
    """
    raise NotImplementedError


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """Sample one token per row: temperature, then top-k (if given), then
    top-p (if given), then softmax and torch.multinomial(probs, 1,
    generator=generator). Return shape (B,). temperature == 0 means argmax."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — KV cache
# ----------------------------------------------------------------------------

def forward_cached(model, idx, cache=None):
    """Run `model` (a helpers.GPT) on NEW tokens idx (B, n), reusing cached
    keys/values for the P tokens processed before.

    cache: None, or a list with one (K, V) pair per block, each of shape
           (B, n_heads, P, d // n_heads).
    Returns (logits (B, n, vocab), new_cache) where new_cache holds the keys
    and values of all P + n tokens.

    Use the model's own submodules; per block (pre-norm, lesson 23):
        h = blk.ln1(x);  q, k, v = blk.attn.qkv(h).split(d, -1); split heads;
        append k, v to the cache;  attention of the n new queries over all
        P + n keys, where new query i (absolute position P + i) may only see
        keys at positions <= P + i;  x = x + blk.attn.proj(merged);
        x = x + blk.mlp(blk.ln2(x))
    Positions: the new tokens are at absolute positions P .. P+n-1.
    Raise ValueError if P + n > model.block_size.
    """
    raise NotImplementedError


def generate_cached(model, idx, max_new_tokens, temperature=1.0, generator=None):
    """Same contract and SAME OUTPUT as helpers.generate_nocache, but:
    run the prompt once ("prefill"), then feed only the newest token each step.
    Pick tokens with helpers' next_token(logits[:, -1], temperature, generator)
    so the random numbers are consumed in the same order as the reference.
    Raise ValueError if the result would exceed model.block_size.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — LoRA and supervised fine-tuning
# ----------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = base(x) + scale * x A^T B^T,  scale = alpha / r.

    Attributes: self.base (the wrapped nn.Linear, all its params frozen with
    requires_grad_(False)), self.A (r × in_features, init randn / sqrt(in)),
    self.B (out_features × r, init ZEROS, so the wrapped layer starts exactly
    equal to the base), self.r, self.scale. Match the base layer's dtype.
    Methods:
      delta_weight() -> scale * B @ A, shape (out, in)
      merge() -> a NEW nn.Linear whose weight is base.weight + delta_weight()
                 (and the same bias) — same outputs, no extra cost at inference.
    """

    def __init__(self, base, r=4, alpha=8.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def delta_weight(self):
        raise NotImplementedError

    def merge(self):
        raise NotImplementedError


def add_lora(model, r=4, alpha=8.0):
    """Freeze every parameter of `model`, then replace blk.attn.qkv and
    blk.attn.proj in every block by LoRALinear wrappers. Return the model."""
    raise NotImplementedError


def make_sft_example(prompt, response, stoi):
    """Build one training pair for SFT on the string prompt + response.

    x = ids[:-1], y = ids[1:] (next-token targets, int64 tensors).
    Positions whose target is still part of the PROMPT get y = -100 (PyTorch's
    ignore_index), so the loss only teaches the model to produce the response.
    E.g. prompt "ab=", response "ba." -> x = "ab=ba", y = [-100, -100, b, a, .]
    """
    raise NotImplementedError


def sft_loss(model, x, y):
    """Mean cross-entropy over non-ignored targets. x, y: (B, T).
    Hint: F.cross_entropy(..., ignore_index=-100)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part D — scaling laws (plain Python floats)
# ----------------------------------------------------------------------------

def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """L(N, D) = E + A / N^alpha + B / D^beta  (N parameters, D training tokens)."""
    raise NotImplementedError


def compute_optimal(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """Minimize L(N, D) subject to the compute budget C = 6 N D.
    Return (N_opt, D_opt) in closed form (derive it: substitute D = C / (6N),
    differentiate in N, set to zero)."""
    raise NotImplementedError
