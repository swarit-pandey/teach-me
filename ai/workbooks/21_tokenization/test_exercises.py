import os
import random
import pytest
from _impl import load

ex = load(__file__)

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tinyshakespeare.txt")


def shakespeare(n=20000):
    with open(DATA, encoding="utf-8") as f:
        return f.read()[:n]


def random_strings(seed, n=40):
    """Random strings mixing ASCII, accents, CJK, emoji and whitespace."""
    rng = random.Random(seed)
    alphabet = list("abcdefgh ABC 0123456789 .,!?'\n\t") + list("éüñßçø") + list("你好世界") + ["🙂", "🚀", "👍🏽"]
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30))) for _ in range(n)]


# ---- Part A ----
def test_pair_counts():
    assert ex.get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert ex.get_pair_counts([5, 5, 5]) == {(5, 5): 2}
    assert ex.get_pair_counts([7]) == {}
    # first-appearance order of keys
    assert list(ex.get_pair_counts([3, 4, 1, 2, 3, 4])) == [(3, 4), (4, 1), (1, 2), (2, 3)]
    # accumulation into an existing dict
    acc = ex.get_pair_counts([1, 2])
    assert ex.get_pair_counts([1, 2, 1], acc) == {(1, 2): 2, (2, 1): 1}


def test_merge():
    src = [1, 2, 3, 1, 2]
    assert ex.merge(src, (1, 2), 99) == [99, 3, 99]
    assert src == [1, 2, 3, 1, 2], "merge must not modify its input"
    assert ex.merge([5, 5, 5], (5, 5), 99) == [99, 5]
    assert ex.merge([5, 5, 5, 5], (5, 5), 99) == [99, 99]
    assert ex.merge([1, 2], (2, 1), 99) == [1, 2]
    assert ex.merge([], (1, 2), 99) == []


def test_merge_shrinks_by_occurrences():
    rng = random.Random(0)
    for _ in range(200):
        ids = [rng.randint(0, 3) for _ in range(rng.randint(0, 40))]
        pair = (rng.randint(0, 3), rng.randint(0, 3))
        out = ex.merge(ids, pair, 100)
        assert len(ids) - len(out) == out.count(100)


def test_train_tiny_example():
    # the lesson's worked example
    merges = ex.train_bpe("aaabdaaabac", 259)
    assert merges == {(97, 97): 256, (256, 97): 257, (257, 98): 258}
    assert list(merges.values()) == [256, 257, 258]


def test_train_edge_cases():
    assert ex.train_bpe("hello", 256) == {}
    with pytest.raises(ValueError):
        ex.train_bpe("hello", 100)
    # no pair repeats -> early stop, no merges
    assert ex.train_bpe("abcdef", 300) == {}
    assert ex.train_bpe("", 300) == {}


def test_build_vocab_and_decode():
    merges = ex.train_bpe("aaabdaaabac", 259)
    vocab = ex.build_vocab(merges)
    assert vocab[97] == b"a" and vocab[256] == b"aa" and vocab[258] == b"aaab"
    assert len(vocab) == 259
    assert ex.decode([258, 100, 258, 97, 99], merges) == "aaabdaaabac"
    # half of a 2-byte character must not crash
    assert ex.decode([0xC3], {}) == "�"


def test_encode_matches_training():
    merges = ex.train_bpe("aaabdaaabac", 259)
    assert ex.encode("aaabdaaabac", merges) == [258, 100, 258, 97, 99]
    assert ex.encode("", merges) == []
    assert ex.encode("a", merges) == [97]
    assert ex.encode("hé", {}) == [104, 0xC3, 0xA9]


def test_encode_order_matters():
    # (b,c) learned before (a,b): "abc" must become a,[bc]; not [ab],c
    merges = {(98, 99): 256, (97, 98): 257}
    assert ex.encode("abc", merges) == [97, 256]


def test_roundtrip_property():
    text = shakespeare(8000)
    merges = ex.train_bpe(text, 320)
    assert len(merges) == 64
    for s in random_strings(1) + [text[:500], "", " ", "\n\n"]:
        assert ex.decode(ex.encode(s, merges), merges) == s


def test_ids_in_range_and_compress():
    text = shakespeare(8000)
    merges = ex.train_bpe(text, 320)
    ids = ex.encode(text, merges)
    assert all(0 <= i < 320 for i in ids)
    assert len(ids) < 0.75 * len(text.encode("utf-8"))


# ---- Part B ----
def test_pretokenize():
    assert ex.pretokenize("Hello world!!") == ["Hello", " world", "!!"]
    assert ex.pretokenize("I'm 2024") == ["I", "'m", " 2024"]
    for s in random_strings(2) + [shakespeare(3000)]:
        assert "".join(ex.pretokenize(s)) == s


def test_chunked_tokens_never_cross_words():
    text = shakespeare(8000)
    merges = ex.train_bpe_chunked(text, 320)
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), c in merges.items():
        vocab[c] = vocab[a] + vocab[b]
        tok = vocab[c].decode("utf-8", errors="replace")
        # a learned token is itself a single pre-token chunk
        assert ex.pretokenize(tok) == [tok], f"token {tok!r} crosses a chunk boundary"
    for s in random_strings(3) + [text[:300]]:
        assert ex.decode(ex.encode_chunked(s, merges), merges) == s


def test_chunked_learns_leading_space_words():
    merges = ex.train_bpe_chunked(shakespeare(8000), 320)
    vocab = ex.build_vocab(merges)
    learned = {vocab[i] for i in range(256, 256 + len(merges))}
    assert b" the" in learned


# ---- Part C ----
def test_compression_ratio():
    assert ex.compression_ratio("hello", [1, 2, 3, 4, 5]) == 1.0
    assert ex.compression_ratio("hé", [1]) == 3.0
    with pytest.raises(ValueError):
        ex.compression_ratio("x", [])


def test_compression_grows_with_vocab():
    text = shakespeare(8000)
    ratios = []
    for V in (256, 280, 320, 400):
        merges = ex.train_bpe(text, V)
        ratios.append(ex.compression_ratio(text, ex.encode(text, merges)))
    assert ratios[0] == 1.0
    assert all(b > a for a, b in zip(ratios, ratios[1:]))


def test_token_lengths():
    merges = ex.train_bpe("aaabdaaabac", 259)
    ids = ex.encode("aaabdaaabac", merges)
    assert ex.token_lengths(ids, merges) == {4: 2, 1: 3}
    assert sum(k * v for k, v in ex.token_lengths(ids, merges).items()) == 11
