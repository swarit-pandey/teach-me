import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _impl import load

ex = load(__file__)
from helpers import load_corpus, split  # noqa: E402

TEXT = load_corpus(max_chars=200_000)      # works with any corpus size >= ~20 KB


# ---- Part A ----
def test_vocab_encode_decode():
    stoi, itos = ex.build_vocab("hello world")
    assert [stoi[c] for c in sorted(set("hello world"))] == list(range(8))
    assert all(itos[i] == c for c, i in stoi.items())
    ids = ex.encode("hello", stoi)
    assert ids == [stoi[c] for c in "hello"]
    assert ex.decode(ids, itos) == "hello"
    assert ex.decode(torch.tensor(ids), itos) == "hello"
    s2, i2 = ex.build_vocab(TEXT)
    assert ex.decode(ex.encode(TEXT[:500], s2), i2) == TEXT[:500]


def test_one_hot_is_embedding_lookup():
    oh = ex.one_hot([2, 0, 2], 4)
    assert oh.shape == (3, 4) and np.array_equal(oh.sum(axis=1), [1, 1, 1])
    assert oh[0, 2] == 1 and oh[1, 0] == 1
    E = np.random.default_rng(0).normal(size=(4, 3))
    assert np.allclose(oh @ E, E[[2, 0, 2]])


def test_bigram_counts():
    C = ex.bigram_counts([0, 1, 1, 1, 0, 1], 3)
    assert C.shape == (3, 3)
    assert C[0, 1] == 2 and C[1, 1] == 2 and C[1, 0] == 1 and C.sum() == 5
    stoi, _ = ex.build_vocab(TEXT)
    ids = ex.encode(TEXT, stoi)
    Cf = ex.bigram_counts(ids, len(stoi))
    assert Cf.sum() == len(ids) - 1
    # row a counts how often a appears (except as the very last character)
    assert Cf[stoi["e"]].sum() == TEXT[:-1].count("e")


def test_bigram_probs_and_nll():
    C = np.array([[0, 3], [1, 1]])
    P = ex.bigram_probs(C, alpha=1.0)
    assert np.allclose(P, [[1 / 5, 4 / 5], [0.5, 0.5]])
    assert np.allclose(ex.bigram_probs(C, alpha=0.0)[0], [0, 1])
    assert ex.bigram_nll(P, [0, 1, 1]) == pytest.approx(-(math.log(0.8) + math.log(0.5)) / 2)
    # uniform model: NLL = log V
    assert ex.bigram_nll(np.full((5, 5), 0.2), [0, 3, 1, 4]) == pytest.approx(math.log(5))


def test_bigram_on_shakespeare():
    stoi, itos = ex.build_vocab(TEXT)
    V = len(stoi)
    tr, va = split(ex.encode(TEXT, stoi))
    P = ex.bigram_probs(ex.bigram_counts(tr, V), alpha=0.1)
    tr_nll, va_nll = ex.bigram_nll(P, tr), ex.bigram_nll(P, va)
    assert tr_nll < math.log(V) - 1.0           # far better than uniform guessing
    assert 2.2 < tr_nll < 2.8 and va_nll < 2.9  # ~2.45 nats: the famous bigram ballpark


def test_sample_bigram():
    C = np.array([[0, 5, 0], [0, 0, 7], [3, 0, 0]])     # deterministic cycle 0 -> 1 -> 2 -> 0
    s = ex.sample_bigram(ex.bigram_probs(C, alpha=0.0), 1, 7, seed=0)
    assert s == [1, 2, 0, 1, 2, 0, 1]
    P = np.array([[0.25, 0.75], [0.25, 0.75]])
    s = ex.sample_bigram(P, 0, 4001, seed=3)
    assert len(s) == 4001 and set(s) <= {0, 1}
    assert abs(np.mean(s[1:]) - 0.75) < 0.03
    assert ex.sample_bigram(P, 0, 50, seed=3) == ex.sample_bigram(P, 0, 50, seed=3)


# ---- Part B ----
def test_build_dataset():
    X, Y = ex.build_dataset([5, 6, 7, 8, 9, 10], 3)
    assert X.dtype == torch.long and Y.dtype == torch.long
    assert X.tolist() == [[5, 6, 7], [6, 7, 8], [7, 8, 9]]
    assert Y.tolist() == [8, 9, 10]
    X2, Y2 = ex.build_dataset(list(range(100)), 8)
    assert X2.shape == (92, 8) and Y2.shape == (92,)
    assert torch.equal(X2[:, -1] + 1, Y2)


def test_neural_bigram_matches_counts():
    text = TEXT[:50_000]
    stoi, _ = ex.build_vocab(TEXT)
    V = len(stoi)
    ids = ex.encode(text, stoi)
    m = ex.NeuralBigram(V)
    assert [p.shape for p in m.parameters()] == [(V, V)]
    assert torch.all(m.W == 0)                  # zeros: every next char equally likely, loss = log V
    out = m(torch.tensor([0, 3]))
    assert out.shape == (2, V)
    torch.manual_seed(0)
    model, final = ex.train_neural_bigram(ids, V, steps=300, lr=0.1)
    ref = ex.bigram_nll(ex.bigram_probs(ex.bigram_counts(ids, V), alpha=0.01), ids)
    assert isinstance(final, float)
    # gradient descent rediscovers the counting solution (count model = the MLE, lesson 10)
    assert abs(final - ref) < 0.05
    P_learned = F.softmax(model.W.detach(), dim=1).numpy()
    e = stoi["e"]
    P_count = ex.bigram_probs(ex.bigram_counts(ids, V), alpha=0.01)
    assert np.abs(P_learned[e] - P_count[e]).max() < 0.03


def test_mlp_lm_beats_bigram():
    stoi, itos = ex.build_vocab(TEXT)
    V = len(stoi)
    tr, va = split(ex.encode(TEXT, stoi))
    big_va = ex.bigram_nll(ex.bigram_probs(ex.bigram_counts(tr, V), alpha=0.1), va)
    block = 5
    Xtr, Ytr = ex.build_dataset(tr, block)
    Xva, Yva = ex.build_dataset(va, block)
    torch.manual_seed(0)
    model = ex.MLPLanguageModel(V, block, emb_dim=10, hidden=100)
    assert isinstance(model.emb, nn.Embedding) and model.emb.weight.shape == (V, 10)
    assert model(Xtr[:7]).shape == (7, V)
    losses = ex.train_lm(model, Xtr, Ytr, steps=3000, batch_size=64, lr=3e-3, seed=0)
    assert len(losses) == 3000
    assert np.mean(losses[-200:]) < np.mean(losses[:200]) - 0.5
    va_loss = ex.evaluate_loss(model, Xva, Yva)
    assert isinstance(va_loss, float)
    assert va_loss < big_va - 0.15, (va_loss, big_va)       # context beats one previous char
    new = ex.generate(model, tr[:block], 40, seed=1)
    assert len(new) == 40 and all(0 <= i < V for i in new)
    assert ex.generate(model, tr[:block], 40, seed=1) == new


# ---- Part C ----
def test_rnn_forward_matches_nn_rnn():
    torch.manual_seed(0)
    T, B, D, H = 6, 3, 4, 5
    rnn = nn.RNN(D, H, nonlinearity="tanh")
    xs, h0 = torch.randn(T, B, D), torch.randn(B, H)
    with torch.no_grad():
        ref, _ = rnn(xs, h0[None])
        hs = ex.rnn_forward(xs, h0, rnn.weight_ih_l0.T, rnn.weight_hh_l0.T, rnn.bias_ih_l0 + rnn.bias_hh_l0)
    assert hs.shape == (T, B, H)
    assert torch.allclose(hs, ref, atol=1e-6)


def test_vanishing_and_exploding_gradients():
    small = ex.grad_norms_through_time(T=30, scale=0.5, seed=0)
    assert len(small) == 30 and all(isinstance(v, float) for v in small)
    assert small[0] / small[-1] < 1e-6                      # 30 steps back: vanished
    assert all(a < b for a, b in zip(small[:-1], small[1:]))  # shrinks steadily with distance
    big = ex.grad_norms_through_time(T=30, scale=3.0, seed=0)
    assert big[0] / big[-1] > 10                            # large recurrent weights: explodes
