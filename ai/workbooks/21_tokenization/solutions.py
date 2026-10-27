"""Workbook 21 — reference solutions. Try first; read after."""
import re

SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"


def get_pair_counts(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    out, i, n = [], 0, len(ids)
    while i < n:
        if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2          # skip both halves: matches never overlap
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text, vocab_size):
    if vocab_size < 256:
        raise ValueError("vocab_size must be >= 256 (the byte alphabet)")
    ids = list(text.encode("utf-8"))
    merges = {}
    for new_id in range(256, vocab_size):
        counts = get_pair_counts(ids)
        if not counts:
            break
        best = max(counts, key=counts.get)   # first maximal key = first appearance
        if counts[best] < 2:
            break
        merges[best] = new_id
        ids = merge(ids, best, new_id)
    return merges


def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), c in merges.items():      # insertion order = learning order
        vocab[c] = vocab[a] + vocab[b]
    return vocab


def decode(ids, merges):
    vocab = build_vocab(merges)
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        # the merge learned earliest must be applied first: later merges were
        # learned on a sequence in which the earlier ones had already happened
        best = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if best not in merges:
            break
        ids = merge(ids, best, merges[best])
    return ids


def pretokenize(text):
    return re.findall(SPLIT_PATTERN, text)


def train_bpe_chunked(text, vocab_size):
    if vocab_size < 256:
        raise ValueError("vocab_size must be >= 256 (the byte alphabet)")
    chunks = [list(c.encode("utf-8")) for c in pretokenize(text)]
    merges = {}
    for new_id in range(256, vocab_size):
        counts = {}
        for ch in chunks:
            get_pair_counts(ch, counts)
        if not counts:
            break
        best = max(counts, key=counts.get)
        if counts[best] < 2:
            break
        merges[best] = new_id
        chunks = [merge(ch, best, new_id) for ch in chunks]
    return merges


def encode_chunked(text, merges):
    out = []
    for c in pretokenize(text):
        out.extend(encode(c, merges))
    return out


def compression_ratio(text, ids):
    if not ids:
        raise ValueError("no tokens")
    return len(text.encode("utf-8")) / len(ids)


def token_lengths(ids, merges):
    vocab = build_vocab(merges)
    out = {}
    for i in ids:
        L = len(vocab[i])
        out[L] = out.get(L, 0) + 1
    return out
