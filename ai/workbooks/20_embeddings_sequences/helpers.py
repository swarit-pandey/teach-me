"""Provided helpers for workbook 20 (not exercises): loading the character corpus."""
import os

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "tinyshakespeare.txt")


def load_corpus(max_chars=None):
    """The tiny-Shakespeare text as one string (optionally only the first max_chars).
    Works with any size; the course ships at least ~20 KB of it."""
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    return text[:max_chars] if max_chars else text


def split(seq, frac=0.9):
    """First frac of a sequence for training, the rest for validation (no shuffling:
    the validation text is later in the play, like real held-out data)."""
    n = int(len(seq) * frac)
    return seq[:n], seq[n:]
