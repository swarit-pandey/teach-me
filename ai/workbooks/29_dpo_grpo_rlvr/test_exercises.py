import copy
import math
import torch
import torch.nn.functional as F
from _impl import load

ex = load(__file__)
import helpers  # noqa: E402  (load() puts this folder on sys.path)


def trainable_copy(m):
    c = copy.deepcopy(m)
    for p in c.parameters():
        p.requires_grad_(True)
    return c


def frozen(m):
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ---- Part A ----
def test_token_and_sequence_logprobs():
    torch.manual_seed(0)
    m = helpers.TinyGPT()
    toks = helpers.sample_reference(4, torch.Generator().manual_seed(0))
    lp = ex.token_logprobs(m, toks)
    assert lp.shape == (4, 8) and lp.requires_grad
    with torch.no_grad():
        logits = m(toks[1:2, :4])[0, -1]
        assert torch.isclose(lp[1, 3], F.log_softmax(logits, -1)[toks[1, 4]], atol=1e-5)
    assert torch.allclose(ex.sequence_logprob(m, toks), lp.sum(1))
    assert ex.token_logprobs(m, toks, prompt_len=4).shape == (4, 5)
    assert torch.allclose(ex.sequence_logprob(m, toks, prompt_len=4), lp[:, 3:].sum(1))


def test_implicit_reward_recovers_reward_from_optimal_policy():
    # π* = π_ref exp(r/β)/Z  =>  β log(π*/π_ref) = r − β log Z : the reward up to a constant
    pref = torch.tensor([0.5, 0.2, 0.2, 0.1]); r = torch.tensor([0.3, -1.0, 2.0, 0.5]); beta = 0.4
    pstar = torch.softmax(pref.log() + r / beta, 0)
    rhat = ex.implicit_reward(pstar.log(), pref.log(), beta)
    shift = rhat - r
    assert torch.allclose(shift, shift[0].expand(4), atol=1e-5)
    Z = (pref * torch.exp(r / beta)).sum()
    assert torch.isclose(shift[0], -beta * Z.log(), atol=1e-5)


def test_dpo_loss_values_and_gradient():
    beta = 0.1
    pc = torch.tensor([-5.0, -3.0, -7.0], requires_grad=True); pr = torch.tensor([-6.0, -2.0, -7.0], requires_grad=True)
    rc, rr = torch.tensor([-5.5, -3.0, -6.0]), torch.tensor([-5.5, -2.5, -8.0])
    loss, r_c, r_r = ex.dpo_loss(pc, pr, rc, rr, beta)
    assert torch.allclose(r_c, beta * (pc.detach() - rc)) and not r_c.requires_grad
    margin = beta * ((pc - rc) - (pr - rr)).detach()
    assert torch.isclose(loss, -F.logsigmoid(margin).mean())
    loss.backward()
    w = torch.sigmoid(-margin)                           # weight: how WRONG the implicit ranking is
    assert torch.allclose(pc.grad, -beta * w / 3, atol=1e-6)
    assert torch.allclose(pr.grad, beta * w / 3, atol=1e-6)
    # at policy == reference the loss is log 2 and the implicit rewards are 0
    l0, a, b = ex.dpo_loss(rc, rr, rc, rr, beta)
    assert math.isclose(l0.item(), math.log(2), rel_tol=1e-6) and torch.all(a == 0) and torch.all(b == 0)


def test_train_dpo_moves_policy_toward_preferences():
    ref = frozen(helpers.make_sft_model())
    policy = trainable_copy(ref)
    ch, rj = helpers.preference_pairs(4000, seed=0)
    hist = ex.train_dpo(policy, ref, ch, rj, steps=150, beta=0.5, seed=0)
    assert len(hist["loss"]) == 150
    assert abs(hist["loss"][0] - math.log(2)) < 0.02                 # starts at policy == ref
    assert sum(hist["loss"][-10:]) / 10 < hist["loss"][0] - 0.03
    assert sum(hist["margin"][-10:]) / 10 > 0.1
    # held-out pairs: the implicit reward ranks them like the labellers (better than chance)
    tch, trj = helpers.preference_pairs(2000, seed=1)
    with torch.no_grad():
        m = (ex.sequence_logprob(policy, tch) - ex.sequence_logprob(ref, tch)) - \
            (ex.sequence_logprob(policy, trj) - ex.sequence_logprob(ref, trj))
    assert (m > 0).float().mean() > 0.6
    # and the preferred-response probability went up: the policy's samples score higher
    g = torch.Generator().manual_seed(5)
    before = helpers.true_score(helpers.generate(ref, 2000, generator=g)).mean()
    g = torch.Generator().manual_seed(5)
    after = helpers.true_score(helpers.generate(policy, 2000, generator=g)).mean()
    assert after > before + 0.1, (before, after)


# ---- Part B ----
def test_check_answer():
    t = torch.tensor([
        [12, 7, 10, 8, 11, 1, 5],     # 7+8=15  right
        [12, 7, 10, 8, 11, 1, 4],     # wrong
        [12, 2, 10, 3, 11, 0, 5],     # 2+3=05  right
        [12, 0, 10, 0, 11, 0, 0],     # 0+0=00  right
        [12, 9, 10, 9, 11, 1, 8],     # 9+9=18  right
        [12, 1, 10, 0, 11, 0, 11],    # '=' as a digit: wrong
        [12, 5, 10, 5, 11, 0, 10],    # 10*0 + '+'(10) == 10 — must still be wrong
    ])
    assert ex.check_answer(t).tolist() == [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_grpo_advantages():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    A = ex.grpo_advantages(r)
    assert A.shape == (3, 4)
    assert torch.allclose(A[0], torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-4)
    assert torch.all(A[1] == 0)                                     # no signal from a solved prompt
    assert torch.allclose(A[2], torch.tensor([-1, -1, -1, 3.0]) / math.sqrt(3), atol=1e-4)
    assert torch.allclose(A.mean(1), torch.zeros(3), atol=1e-6)


def test_kl_k3_unbiased_and_nonnegative():
    g = torch.Generator().manual_seed(0)
    pi = torch.softmax(torch.randn(6, generator=g), 0); ref = torch.softmax(torch.randn(6, generator=g), 0)
    k = ex.kl_k3(pi.log(), ref.log())
    assert torch.all(k >= 0)
    assert torch.isclose((pi * k).sum(), (pi * (pi / ref).log()).sum(), atol=1e-6)   # E_π[k3] = KL(π‖ref)
    assert torch.allclose(ex.kl_k3(pi.log(), pi.log()), torch.zeros(6))


def test_grpo_loss():
    lp = torch.zeros(2, 3, requires_grad=True)
    adv = torch.tensor([1.0, -2.0])
    loss = ex.grpo_loss(lp, torch.zeros(2, 3), torch.zeros(2, 3), adv, eps=0.2, beta=0.04)
    assert torch.isclose(loss, -adv.mean())                         # ρ = 1, KL = 0
    loss.backward()
    assert torch.allclose(lp.grad, -adv[:, None].expand(2, 3) / 6)
    # clipping + KL term
    lp2 = torch.log(torch.tensor([[1.5, 1.0]]))
    ref = torch.log(torch.tensor([[1.0, 0.5]]))
    l2 = ex.grpo_loss(lp2, torch.zeros(1, 2), ref, torch.tensor([1.0]), eps=0.2, beta=0.1)
    k = ex.kl_k3(lp2, ref)
    assert torch.isclose(l2, -((torch.tensor([1.2, 1.0]) - 0.1 * k).mean()))


def test_train_grpo_raises_accuracy():
    ref = frozen(helpers.make_arith_sft_model())
    policy = trainable_copy(ref)
    acc0 = helpers.sampled_accuracy(ref, ex.check_answer, n=2000)
    assert 0.2 < acc0 < 0.65                                         # the sloppy SFT model
    hist = ex.train_grpo(policy, ref, iters=60, seed=0)
    assert len(hist["accuracy"]) == 60 and len(hist["kl"]) == 60
    acc1 = helpers.sampled_accuracy(policy, ex.check_answer, n=2000)
    assert acc1 > acc0 + 0.2, (acc0, acc1)
    assert hist["kl"][-1] > hist["kl"][0]                            # it moved away from the reference...
