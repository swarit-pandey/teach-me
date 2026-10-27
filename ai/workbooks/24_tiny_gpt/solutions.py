"""Workbook 24 — reference solutions. Try first; read after."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import Block


# ---- Part A: data ----
def build_vocab(text):
    itos = sorted(set(text))
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos


def encode(s, stoi):
    return [stoi[c] for c in s]


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


def train_val_split(data, val_frac=0.1):
    n = int(len(data) * (1 - val_frac))
    return data[:n], data[n:]


def get_batch(data, block_size, batch_size, generator=None):
    # the last valid start leaves room for block_size inputs AND one more target
    ix = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


# ---- Part B: model ----
class TinyGPT(nn.Module):
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
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
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


def expected_initial_loss(vocab_size):
    # near-zero logits -> uniform prediction -> cross-entropy = -log(1/V)
    return math.log(vocab_size)


# ---- Part C: training & sampling ----
def make_optimizer(model, lr=1e-3, weight_decay=0.1):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.99))


def train_step(model, optimizer, x, y):
    model.train()
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, n_batches, generator=None):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, generator)
        losses.append(model(x, y)[1].item())
    model.train(was_training)
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, generator=None):
    for _ in range(max_new_tokens):
        ctx = idx[:, -model.block_size:]               # never feed more than block_size
        logits = model(ctx)[0][:, -1, :]               # only the last position predicts the next token
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
    return idx
