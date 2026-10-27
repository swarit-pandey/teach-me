import copy
import math
import pytest
import torch
import torch.nn.functional as F
from _impl import load

ex = load(__file__)
import helpers  # noqa: E402  (load() puts this folder on sys.path)


# ---- Part A ----
def test_bradley_terry_prob_and_loss():
    rc, rr = torch.tensor([1.5, 0.0, 3.0]), torch.tensor([0.5, 0.0, -1.0])
    p = ex.bradley_terry_prob(rc, rr)
    assert torch.allclose(p, torch.sigmoid(torch.tensor([1.0, 0.0, 4.0])))
    assert torch.allclose(ex.bradley_terry_prob(rr, rc), 1 - p)                # P(a≻b) + P(b≻a) = 1
    assert math.isclose(ex.bradley_terry_loss(torch.zeros(5), torch.zeros(5)).item(), math.log(2), rel_tol=1e-6)
    # shift invariance: only reward DIFFERENCES matter
    assert torch.isclose(ex.bradley_terry_loss(rc + 7, rr + 7), ex.bradley_terry_loss(rc, rr))
    # stable for huge margins (log(sigmoid(-200)) would be -inf)
    big = ex.bradley_terry_loss(torch.tensor([-200.0]), torch.tensor([0.0]))
    assert torch.isfinite(big) and abs(big.item() - 200) < 1e-3


def test_bradley_terry_gradient():
    rc = torch.tensor([1.5, -0.2], requires_grad=True); rr = torch.tensor([0.5, 0.3])
    ex.bradley_terry_loss(rc, rr).backward()
    want = -(1 - torch.sigmoid(rc.detach() - rr)) / 2                           # −(1−σ(Δ))/N
    assert torch.allclose(rc.grad, want, atol=1e-6)


def test_reward_model_shapes():
    torch.manual_seed(0)
    rm = ex.RewardModel()
    toks = helpers.sample_reference(5, torch.Generator().manual_seed(0))
    out = rm(toks)
    assert out.shape == (5,)
    out.sum().backward()
    assert rm.head.weight.grad is not None and rm.body.tok.weight.grad is not None
    # the score must depend on EARLY tokens too (read the last position, not the first)
    t2 = toks.clone(); t2[:, 1] = (t2[:, 1] % 6) + 1
    assert not torch.allclose(rm(toks), rm(t2))


def test_train_reward_model_accuracy():
    ch, rj = helpers.preference_pairs(3000, seed=0)
    rm = ex.train_reward_model(ch, rj, steps=300, seed=0)
    tch, trj = helpers.preference_pairs(2000, seed=1)
    acc = ex.preference_accuracy(rm, tch, trj)
    # Bayes-optimal accuracy on these noisy BT labels: mean of σ(|Δ|) (ties count ½)
    d = (helpers.true_score(tch) - helpers.true_score(trj)).abs()
    bayes = torch.sigmoid(d).mean().item()
    assert isinstance(acc, float)
    assert acc > bayes - 0.06, (acc, bayes)
    # it learned the labellers' utility: agreement with the TRUE ordering on clear pairs
    s_ref = helpers.true_score(tch) - helpers.true_score(trj)
    with torch.no_grad():
        s_rm = rm(tch) - rm(trj)
    clear = s_ref.abs() >= 1
    assert ((s_rm[clear] > 0) == (s_ref[clear] > 0)).float().mean() > 0.85


# ---- Part B ----
def test_optimal_kl_policy():
    pr, r = torch.tensor([0.5, 0.3, 0.2]), torch.tensor([0.0, 1.0, 2.0])
    p = ex.optimal_kl_policy(pr, r, 1.0)
    assert torch.allclose(p, torch.tensor([0.17900, 0.29194, 0.52906]), atol=1e-4)
    assert torch.allclose(ex.optimal_kl_policy(pr, r, 1e6), pr, atol=1e-5)          # β→∞: stay at π_ref
    assert torch.allclose(ex.optimal_kl_policy(pr, r, 1e-3), torch.tensor([0., 0., 1.]), atol=1e-6)  # β→0: argmax
    # it really is the maximizer of E[r] − β KL(π‖π_ref): beats random alternatives
    obj = lambda q: (q * r).sum() - 0.7 * (q * (q / pr).log()).sum()
    best = obj(ex.optimal_kl_policy(pr, r, 0.7))
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        q = torch.softmax(torch.randn(3, generator=g) * 2, 0)
        assert obj(q) <= best + 1e-6
    # batched over leading dims
    P = ex.optimal_kl_policy(pr.expand(4, 3), r.expand(4, 3), 0.5)
    assert P.shape == (4, 3) and torch.allclose(P.sum(-1), torch.ones(4))


def test_sequence_logprobs():
    torch.manual_seed(0)
    m = helpers.TinyGPT()
    toks = helpers.sample_reference(3, torch.Generator().manual_seed(2))
    lp = ex.sequence_logprobs(m, toks)
    assert lp.shape == (3, 8) and lp.requires_grad
    with torch.no_grad():
        for b in range(3):
            for t in range(8):                                     # brute force, one prefix at a time
                logits = m(toks[b:b + 1, :t + 1])[0, -1]
                assert torch.isclose(lp[b, t], F.log_softmax(logits, -1)[toks[b, t + 1]], atol=1e-5)
    assert ex.sequence_logprobs(m, toks, prompt_len=3).shape == (3, 6)
    assert torch.allclose(ex.sequence_logprobs(m, toks, prompt_len=3), lp[:, 2:])


def test_kl_penalized_rewards():
    logp = torch.tensor([[-0.2, -1.0, -0.5]]); ref = torch.tensor([[-0.4, -1.5, -0.5]])
    r = ex.kl_penalized_rewards(logp, ref, torch.tensor([2.0]), beta=0.1)
    assert torch.allclose(r, torch.tensor([[-0.02, -0.05, 2.0]]))
    assert torch.allclose(logp, torch.tensor([[-0.2, -1.0, -0.5]]))          # inputs untouched
    g = torch.Generator().manual_seed(0)
    lp, rl, sc = torch.randn(4, 8, generator=g), torch.randn(4, 8, generator=g), torch.randn(4, generator=g)
    tot = ex.kl_penalized_rewards(lp, rl, sc, 0.3).sum(1)
    assert torch.allclose(tot, sc - 0.3 * (lp - rl).sum(1), atol=1e-6)


def test_gae_batched():
    g = torch.Generator().manual_seed(0)
    r, v = torch.randn(3, 6, generator=g), torch.randn(3, 6, generator=g)
    adv, ret = ex.gae(r, v, gamma=1.0, lam=1.0)
    rtg = r.flip(1).cumsum(1).flip(1)                                  # λ=γ=1: plain reward-to-go
    assert torch.allclose(ret, rtg, atol=1e-5) and torch.allclose(adv, rtg - v, atol=1e-5)
    adv0, _ = ex.gae(r, v, gamma=0.9, lam=0.0)
    nxt = torch.cat([v[:, 1:], torch.zeros(3, 1)], 1)
    assert torch.allclose(adv0, r + 0.9 * nxt - v, atol=1e-6)         # λ=0: TD residuals


def test_ppo_policy_loss():
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    old = torch.zeros(4)
    logp = torch.log(torch.tensor([1.5, 1.1, 0.7, 1.5])).requires_grad_()
    loss = ex.ppo_policy_loss(logp, old, adv, eps=0.2)
    assert torch.isclose(loss, -torch.tensor([1.2, 1.1, -0.8, -1.5]).mean())
    loss.backward()
    # clipped samples (0 and 2) get no gradient; dL/dlogp = −A ρ / n for the others
    assert torch.allclose(logp.grad, torch.tensor([0.0, -1.1 / 4, 0.0, 1.5 / 4]), atol=1e-6)


# ---- Part C ----
def test_ppo_rlhf_raises_reward_with_bounded_kl():
    ref = helpers.make_sft_model()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy = copy.deepcopy(ref)
    for p in policy.parameters():
        p.requires_grad_(True)
    torch.manual_seed(0)
    hist = ex.ppo_rlhf(policy, ref, helpers.ValueNet(), helpers.count_a, iters=25, beta=0.5, seed=0)
    s, kl = hist["score"], hist["kl"]
    assert len(s) == 25 and len(kl) == 25
    assert abs(kl[0]) < 0.3                                             # starts AT the reference
    assert sum(s[-5:]) / 5 > s[0] + 1.0, s                              # reward measurably up
    assert 0.2 < sum(kl[-5:]) / 5 < 8.0, kl                             # moved, but stayed close
    # the reference model itself must be untouched
    assert torch.allclose(helpers.make_sft_model().lm_head.weight, ref.lm_head.weight)
