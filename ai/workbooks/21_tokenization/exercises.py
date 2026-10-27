"""Workbook 21 — Tokenization: byte-level BPE from scratch.

Rules of the game
  * Pure Python only (no numpy needed, no libraries). A tokenizer is string and
    list manipulation; you should be able to write one in an interview.
  * Token ids 0..255 are the raw UTF-8 bytes. Every merge creates a new id,
    256, 257, ... in the order the merges were learned.
  * A "merges" table is a dict  {(left_id, right_id): new_id}. Python dicts keep
    insertion order, so iterating it replays the merges in the order learned.
  * Part A: the BPE core (count, merge, train, encode, decode).
    Part B: GPT-style pre-tokenization (merges never cross word boundaries).
    Part C: measuring a tokenizer.
  * Run   ./course check 21   to grade.
"""
import re


# ----------------------------------------------------------------------------
# Part A — the BPE core
# ----------------------------------------------------------------------------

def get_pair_counts(ids, counts=None):
    """Count adjacent pairs in a list of ids.

    get_pair_counts([1, 2, 3, 1, 2]) -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}

    If `counts` (a dict) is given, add into it and return it; this lets you
    accumulate counts over many chunks (Part B). Overlapping pairs are all
    counted: [5, 5, 5] -> {(5, 5): 2}.
    Keys must be inserted in order of first appearance (a plain dict does that
    for free if you scan left to right).
    """
    raise NotImplementedError


def merge(ids, pair, new_id):
    """Return a NEW list where every occurrence of `pair` (scanning left to
    right, non-overlapping) is replaced by `new_id`.

    merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
    merge([5, 5, 5], (5, 5), 99)       -> [99, 5]      (left-most match wins)
    Do not modify the input list.
    """
    raise NotImplementedError


def train_bpe(text, vocab_size):
    """Learn byte-level BPE merges from `text` until the vocabulary has
    `vocab_size` ids (so vocab_size - 256 merges).

    Algorithm: ids = list(text.encode("utf-8")); repeat: count pairs, take the
    most frequent pair, give it the next id, merge it everywhere.
    Tie-break: among equally frequent pairs, choose the one that appears FIRST
    in the current sequence. (max(counts, key=counts.get) does exactly that,
    because max returns the first maximal key and dicts keep insertion order.)
    Stop early if no pair occurs at least twice (merging a pair that occurs
    once compresses nothing).

    Returns the merges dict {(a, b): new_id}.
    Raise ValueError if vocab_size < 256.
    """
    raise NotImplementedError


def build_vocab(merges):
    """Map every id to the bytes it stands for.

    Start with {i: bytes([i]) for i in range(256)}; each merge (a, b) -> c
    gives vocab[c] = vocab[a] + vocab[b]. Returns the dict.
    """
    raise NotImplementedError


def decode(ids, merges):
    """Ids -> string. Concatenate the bytes of every token, then decode UTF-8
    with errors="replace" (an arbitrary id sequence may cut a multi-byte
    character in half; never crash on it).
    """
    raise NotImplementedError


def encode(text, merges):
    """String -> ids, applying the learned merges.

    Start from the UTF-8 bytes. Repeatedly: among all adjacent pairs present,
    find the one that was learned EARLIEST (smallest new id); if none of the
    present pairs is in `merges`, stop; otherwise merge it everywhere.
    Hint: min(pairs, key=lambda p: merges.get(p, float("inf"))).
    Empty string -> [].
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — pre-tokenization (what GPT-2 actually does)
# ----------------------------------------------------------------------------

# A simplified GPT-2 split pattern: contractions, an optional leading space plus
# a run of letters, an optional space plus a run of digits, an optional space
# plus a run of punctuation, and runs of whitespace.
SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"


def pretokenize(text):
    """Split text into chunks with re.findall(SPLIT_PATTERN, text).

    pretokenize("Hello world!!") -> ["Hello", " world", "!!"]
    The chunks must concatenate back to the original text exactly.
    """
    raise NotImplementedError


def train_bpe_chunked(text, vocab_size):
    """Like train_bpe, but pairs are counted and merged INSIDE each pre-token
    chunk only, so no token ever spans two chunks (e.g. "d" + " w").

    Same tie-break rule (first appearance, scanning the chunks in order) and
    the same early stop. Returns the merges dict.
    Hint: keep a list of id-lists, one per chunk; accumulate counts with
    get_pair_counts(chunk_ids, counts).
    """
    raise NotImplementedError


def encode_chunked(text, merges):
    """Pre-tokenize, encode each chunk with `encode`, concatenate the ids."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — measuring a tokenizer
# ----------------------------------------------------------------------------

def compression_ratio(text, ids):
    """UTF-8 bytes per token: len(text.encode("utf-8")) / len(ids).
    1.0 means no compression (one token per byte). Raise ValueError if ids is empty.
    """
    raise NotImplementedError


def token_lengths(ids, merges):
    """Return a dict {byte_length: number_of_tokens_in_ids_with_that_length}.
    E.g. if ids decode to tokens b"th", b"e", b" " the answer is {2: 1, 1: 2}.
    """
    raise NotImplementedError
