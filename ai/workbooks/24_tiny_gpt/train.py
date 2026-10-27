"""Train a tiny character-level GPT on Tiny Shakespeare (provided code, lesson 24).

    .venv/bin/python workbooks/24_tiny_gpt/train.py            # full run, a few minutes on a laptop CPU
    .venv/bin/python workbooks/24_tiny_gpt/train.py --steps 500  # quick smoke test

Everything is in this one file on purpose: data -> model -> AdamW loop ->
evaluation -> sampling -> checkpoint. Read it top to bottom; every piece is
something you built in workbooks 22-24.

Outputs (in data/):
  tinygpt.pt         checkpoint: model weights + config + vocabulary
  tinygpt_vocab.json the vocabulary (list of characters, index = token id)
  tinygpt_log.json   loss curve and the samples printed during training
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA = os.path.join(ROOT, "data")

# ---------------------------------------------------------------- config
CFG = dict(
    block_size=128,     # context length T
    d=128,              # width
    n_heads=4,
    n_layers=4,
    batch_size=32,
    steps=3000,
    lr=2e-3,            # peak learning rate (small models tolerate large lr)
    min_lr=2e-4,
    warmup=100,
    weight_decay=0.1,
    eval_every=250,
    eval_batches=20,
    seed=1337,
)


# ---------------------------------------------------------------- model (workbooks 22-23)
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
        q, k, v = (t.view(B, T, h, d // h).transpose(1, 2) for t in (q, k, v))   # (B, h, T, dh)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)            # softmax(QK^T/sqrt(dh) + mask) V
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
        # GPT-2 trick: shrink the layers that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layers))

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
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, generator=None):
        for _ in range(n):
            logits = self(idx[:, -self.block_size:])[0][:, -1] / temperature
            idx = torch.cat([idx, torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)], 1)
        return idx


# ---------------------------------------------------------------- helpers
def get_batch(data, T, B, g):
    ix = torch.randint(0, len(data) - T, (B,), generator=g)
    return torch.stack([data[i:i + T] for i in ix]), torch.stack([data[i + 1:i + 1 + T] for i in ix])


def lr_at(step, c):
    """Linear warm-up, then cosine decay from lr to min_lr."""
    if step < c["warmup"]:
        return c["lr"] * (step + 1) / c["warmup"]
    t = (step - c["warmup"]) / max(1, c["steps"] - c["warmup"])
    return c["min_lr"] + 0.5 * (c["lr"] - c["min_lr"]) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, data, c, g):
    model.eval()
    out = sum(model(*get_batch(data, c["block_size"], c["batch_size"], g))[1].item() for _ in range(c["eval_batches"]))
    model.train()
    return out / c["eval_batches"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=CFG["steps"])
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    args = ap.parse_args()
    c = dict(CFG, steps=args.steps)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(c["seed"])

    # ---- data: characters -> ids, 90/10 contiguous split
    with open(os.path.join(DATA, "tinyshakespeare.txt"), encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]
    V = len(chars)
    print(f"text: {len(text):,} chars, vocab V = {V}, train {len(train):,} / val {len(val):,} tokens")

    model = GPT(V, c["block_size"], c["d"], c["n_heads"], c["n_layers"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters  (d={c['d']}, L={c['n_layers']}, h={c['n_heads']}, T={c['block_size']})")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": c["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=c["lr"], betas=(0.9, 0.99))

    g_train, g_eval, g_sample = (torch.Generator().manual_seed(c["seed"] + k) for k in range(3))
    prompt = torch.tensor([[stoi["\n"]]])
    log = {"config": c, "n_params": n_params, "vocab_size": V, "ln_vocab": math.log(V),
           "steps": [], "train_loss": [], "val_loss": [], "batch_loss": [], "samples": {}, "seconds": []}

    def checkpoint(step):
        tr = evaluate(model, train, c, torch.Generator().manual_seed(99))
        va = evaluate(model, val, c, torch.Generator().manual_seed(98))
        model.eval()
        sample = "".join(chars[i] for i in model.generate(prompt, 200, generator=g_sample.manual_seed(7))[0, 1:].tolist())
        model.train()
        el = time.time() - t0
        log["steps"].append(step); log["train_loss"].append(round(tr, 4)); log["val_loss"].append(round(va, 4))
        log["samples"][str(step)] = sample; log["seconds"].append(round(el, 1))
        print(f"\nstep {step:5d} | train {tr:.4f} | val {va:.4f} | {el:6.1f}s\n{'-' * 60}\n{sample}\n{'-' * 60}", flush=True)

    t0 = time.time()
    model.train()
    for step in range(c["steps"]):
        if step % c["eval_every"] == 0 or step in (50, 100):
            checkpoint(step)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, c)
        x, y = get_batch(train, c["block_size"], c["batch_size"], g_train)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            log["batch_loss"].append(round(loss.item(), 4))
        if step % 100 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {lr_at(step, c):.2e}", flush=True)
    checkpoint(c["steps"])

    # ---- save
    os.makedirs(DATA, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {k: c[k] for k in ("block_size", "d", "n_heads", "n_layers")},
                "vocab": chars}, os.path.join(DATA, "tinygpt.pt"))
    with open(os.path.join(DATA, "tinygpt_vocab.json"), "w") as f:
        json.dump(chars, f)
    with open(os.path.join(DATA, "tinygpt_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(f"\nsaved data/tinygpt.pt, data/tinygpt_vocab.json, data/tinygpt_log.json   ({time.time() - t0:.0f}s total)")

    # ---- a longer sample from the final model
    model.eval()
    out = model.generate(prompt, 600, temperature=0.8, generator=torch.Generator().manual_seed(0))
    print("\nFinal sample (temperature 0.8):\n" + "".join(chars[i] for i in out[0, 1:].tolist()))


if __name__ == "__main__":
    main()
