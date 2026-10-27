"""Workbook 24 — Train a tiny GPT.

Rules of the game
  * PyTorch. `helpers.py` gives you a ready-made `Block` (the one you built in
    workbook 23) and `load_text()`. The full, runnable training script is
    `train.py`; read it after you've written the pieces here.
  * Data are 1-D int64 tensors of character ids. Batches are (B, T) tensors.
  * Pass `generator=` through to every torch random call you make
    (torch.randint, torch.multinomial) so the tests are reproducible.
  * Part A: the data pipeline. Part B: the model and its initial loss.
    Part C: optimizer, training step, evaluation, sampling.
  * Run   ./course check 24   to grade.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import Block


# ----------------------------------------------------------------------------
# Part A — data
# ----------------------------------------------------------------------------

def build_vocab(text):
    """Character vocabulary: return (stoi, itos) where itos is the SORTED list
    of distinct characters and stoi maps each character to its index."""
    raise NotImplementedError


def encode(s, stoi):
    """String -> list of ids."""
    raise NotImplementedError


def decode(ids, itos):
    """Iterable of ids (ints or 0-d tensors) -> string."""
    raise NotImplementedError


def train_val_split(data, val_frac=0.1):
    """Split a 1-D tensor into (train, val): the first (1 - val_frac) fraction
    and the rest, with n_train = int(len(data) * (1 - val_frac)).
    Do NOT shuffle: shuffled characters would leak validation text into
    training through overlapping windows (see the lesson)."""
    raise NotImplementedError


def get_batch(data, block_size, batch_size, generator=None):
    """Sample `batch_size` random windows.

    Returns (x, y), both (batch_size, block_size) int64, where for each row
    x = data[i : i + block_size] and y = data[i + 1 : i + 1 + block_size]
    (y is x shifted left by one: y[t] is the token that follows x[..t]).
    Draw the starts with torch.randint(0, len(data) - block_size, (batch_size,),
    generator=generator) so every window fits.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — the model
# ----------------------------------------------------------------------------

class TinyGPT(nn.Module):
    """GPT assembled from helpers.Block.

    Attributes: block_size, tok_emb (Embedding V×d), pos_emb (Embedding
    block_size×d), blocks (ModuleList of n_layers Blocks), ln_f (nn.LayerNorm),
    head (Linear d→V, bias=False).
    Initialization (GPT-2 style): every nn.Linear and nn.Embedding weight
    ~ Normal(0, 0.02), every Linear bias = 0.  Hint: self.apply(fn) calls fn on
    every submodule.
    forward(idx, targets=None) -> (logits (B, T, V), loss or None) with loss
    the mean cross-entropy over all B*T positions.
    """

    def __init__(self, vocab_size, block_size, d, n_heads, n_layers):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx, targets=None):
        raise NotImplementedError


def expected_initial_loss(vocab_size):
    """The cross-entropy you should see at step 0, before any training.
    Derive it (lesson section 4); one line of code."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — training and sampling
# ----------------------------------------------------------------------------

def make_optimizer(model, lr=1e-3, weight_decay=0.1):
    """AdamW (lesson 18) with two parameter groups:
      group 0: every parameter with p.dim() >= 2 (weight matrices, embeddings),
               with weight_decay=weight_decay
      group 1: every other parameter (biases, LayerNorm gains), weight_decay=0.
    Use betas=(0.9, 0.99).
    """
    raise NotImplementedError


def train_step(model, optimizer, x, y):
    """One optimization step: forward, zero grads, backward, clip the global
    gradient norm to 1.0 (torch.nn.utils.clip_grad_norm_), step.
    Return the loss as a Python float."""
    raise NotImplementedError


def estimate_loss(model, data, block_size, batch_size, n_batches, generator=None):
    """Average loss over n_batches random batches, without building a graph
    (torch.no_grad) and with the model in eval mode. Restore the model's
    previous train/eval mode before returning. Return a float."""
    raise NotImplementedError


def generate(model, idx, max_new_tokens, temperature=1.0, generator=None):
    """Autoregressive sampling.

    idx: (B, T0) int64 prompt. Repeat max_new_tokens times:
      crop the context to the last model.block_size tokens, run the model,
      take the logits at the LAST position, divide by temperature, softmax,
      sample one token per row with torch.multinomial(..., generator=generator),
      append it.
    temperature == 0 means greedy (argmax) decoding.
    Return the (B, T0 + max_new_tokens) tensor. No gradients needed.
    """
    raise NotImplementedError
