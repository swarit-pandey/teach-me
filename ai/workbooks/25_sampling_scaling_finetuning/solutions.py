"""Workbook 25 — reference solutions. Try first; read after."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import next_token


# ---- Part A: samplers ----
def apply_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use argmax for greedy)")
    return logits / temperature


def top_k_filter(logits, k):
    k = min(k, logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]          # k-th largest per row
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    before = torch.cumsum(probs, dim=-1) - probs                   # mass strictly before each token
    drop_sorted = before >= p                                      # first token always kept (before = 0)
    drop = drop_sorted.scatter(-1, order, drop_sorted)             # back to the original order
    return logits.masked_fill(drop, float("-inf"))


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0:
        return logits.argmax(-1)
    z = apply_temperature(logits, temperature)
    if top_k is not None:
        z = top_k_filter(z, top_k)
    if top_p is not None:
        z = top_p_filter(z, top_p)
    return torch.multinomial(torch.softmax(z, dim=-1), 1, generator=generator).squeeze(-1)


# ---- Part B: KV cache ----
def forward_cached(model, idx, cache=None):
    B, n = idx.shape
    P = 0 if cache is None else cache[0][0].shape[2]               # tokens already cached
    if P + n > model.block_size:
        raise ValueError("sequence would exceed block_size")
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(P, P + n, device=idx.device))
    # new query i sits at absolute position P+i and may see keys 0..P+i
    allowed = torch.arange(P + n, device=idx.device)[None, :] <= (P + torch.arange(n, device=idx.device))[:, None]
    new_cache = []
    for li, blk in enumerate(model.blocks):
        att = blk.attn
        d = x.shape[-1]
        h = att.n_heads
        q, k, v = att.qkv(blk.ln1(x)).split(d, dim=-1)
        q, k, v = (t.view(B, n, h, d // h).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            k = torch.cat([cache[li][0], k], dim=2)
            v = torch.cat([cache[li][1], v], dim=2)
        new_cache.append((k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // h)        # (B, h, n, P+n)
        w = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        y = (w @ v).transpose(1, 2).reshape(B, n, d)
        x = x + att.proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return model.head(model.ln_f(x)), new_cache


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, temperature=1.0, generator=None):
    if idx.shape[1] + max_new_tokens > model.block_size:
        raise ValueError("sequence would exceed block_size")
    logits, cache = forward_cached(model, idx)                      # prefill: the whole prompt once
    for i in range(max_new_tokens):
        nxt = next_token(logits[:, -1], temperature, generator)
        idx = torch.cat([idx, nxt], dim=1)
        if i + 1 < max_new_tokens:
            logits, cache = forward_cached(model, nxt, cache)       # decode: one token at a time
    return idx


# ---- Part C: LoRA and SFT ----
class LoRALinear(nn.Module):
    def __init__(self, base, r=4, alpha=8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, dtype=base.weight.dtype) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def delta_weight(self):
        return self.scale * self.B @ self.A                         # (out, in), rank <= r

    @torch.no_grad()
    def merge(self):
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                        dtype=self.base.weight.dtype)
        lin.weight.copy_(self.base.weight + self.delta_weight())
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin


def add_lora(model, r=4, alpha=8.0):
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r, alpha)
        blk.attn.proj = LoRALinear(blk.attn.proj, r, alpha)
    return model


def make_sft_example(prompt, response, stoi):
    ids = [stoi[c] for c in prompt + response]
    x = torch.tensor(ids[:-1])
    y = torch.tensor(ids[1:])
    y[: len(prompt) - 1] = -100            # targets that are still prompt characters: ignored
    return x, y


def sft_loss(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)


# ---- Part D: scaling laws ----
CHINCHILLA = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)


def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    return E + A / N ** alpha + B / D ** beta


def compute_optimal(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    # minimize A N^-a + B D^-b subject to 6ND = C: substitute D = C/(6N), set d/dN = 0
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    N = G * (C / 6.0) ** (beta / (alpha + beta))
    return N, C / (6.0 * N)
