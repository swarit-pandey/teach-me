import numpy as np
import pytest
from _impl import load

ex = load(__file__)
from helpers import ChainEnv, rollout  # noqa: E402  (load() puts this folder on sys.path)


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


# ---- Part A ----
def test_softmax():
    p = ex.softmax(np.array([1000.0, 1000.0, 0.0]))          # must not overflow
    assert np.allclose(p, [0.5, 0.5, 0])
    P = ex.softmax(np.array([[0.0, np.log(3)], [2.0, 2.0]]))
    assert np.allclose(P, [[0.25, 0.75], [0.5, 0.5]])
    assert np.allclose(ex.softmax(np.array([1.0, 2.0, 3.0]) + 7), ex.softmax(np.array([1.0, 2.0, 3.0])))


def test_grad_log_softmax_matches_numeric():
    z = np.array([0.3, -1.2, 2.0, 0.5])
    for a in range(4):
        num = num_grad(lambda v: np.log(ex.softmax(v)[a]), z)
        assert np.allclose(ex.grad_log_softmax(z, a), num, atol=1e-6)
    assert abs(ex.grad_log_softmax(z, 2).sum()) < 1e-12               # sums to zero


def test_exact_bandit_gradient():
    th, mu = np.array([0.5, -0.3, 0.1]), np.array([1.0, 2.0, -0.5])
    num = num_grad(lambda v: ex.softmax(v) @ mu, th)
    assert np.allclose(ex.exact_bandit_gradient(th, mu), num, atol=1e-6)


def test_estimates_unbiased_and_baseline_cuts_variance():
    th, mu = np.zeros(3), np.array([0.2, 0.5, 0.9]) + 10.0          # all rewards ~10: big offset
    true = ex.exact_bandit_gradient(th, mu)
    G0 = ex.gradient_estimates(th, mu, 100_000, baseline=0.0, seed=0)
    b = ex.softmax(th) @ mu
    Gb = ex.gradient_estimates(th, mu, 100_000, baseline=b, seed=1)
    assert G0.shape == (100_000, 3)
    se0 = G0.std(0) / np.sqrt(len(G0))
    assert np.all(np.abs(G0.mean(0) - true) < 5 * se0)                 # unbiased without baseline
    assert np.allclose(Gb.mean(0), true, atol=0.01)                    # ...and with it
    ratio = G0.var(0).sum() / Gb.var(0).sum()
    assert ratio > 50, ratio                                           # ~100x variance reduction


def test_reinforce_bandit_learns_and_baseline_helps():
    mu = np.array([0.2, 0.5, 0.9]) + 5.0
    with_b = [ex.softmax(ex.train_reinforce_bandit(mu, 500, lr=0.05, use_baseline=True, seed=s)[0])[2] for s in range(10)]
    no_b = [ex.softmax(ex.train_reinforce_bandit(mu, 500, lr=0.05, use_baseline=False, seed=s)[0])[2] for s in range(10)]
    assert np.mean(with_b) > 0.8 and min(with_b) > 0.6
    assert np.mean(with_b) > np.mean(no_b) + 0.1
    theta, r = ex.train_reinforce_bandit(mu, 50, seed=0)
    assert theta.shape == (3,) and r.shape == (50,)


# ---- Part B ----
def test_reward_to_go():
    G = ex.reward_to_go(np.array([0.0, 0.0, 1.0]), 0.9)
    assert np.allclose(G, [0.81, 0.9, 1.0])


def test_reinforce_grad_matches_hand_computation():
    theta = np.array([[0.0, 0.0], [1.0, -1.0], [0.0, 0.0]])
    ep = {"states": np.array([0, 1, 1]), "actions": np.array([1, 0, 1]), "rewards": np.array([0.0, 0.0, 1.0])}
    g = ex.reinforce_grad(theta, [ep], gamma=0.5)
    p1 = ex.softmax(theta[1])
    # G = [0.25, 0.5, 1];  row 0: 0.25*(e1 - [.5,.5]);  row 1: 0.5*(e0 - p1) + 1*(e1 - p1)
    want = np.zeros((3, 2))
    want[0] = 0.25 * np.array([-0.5, 0.5])
    want[1] = 0.5 * (np.array([1, 0]) - p1) + (np.array([0, 1]) - p1)
    assert np.allclose(g, want)
    V = np.array([0.1, 0.2, 0.0])
    gb = ex.reinforce_grad(theta, [ep, ep], gamma=0.5, values=V)       # averaging over episodes
    want_b = np.zeros((3, 2))
    want_b[0] = 0.15 * np.array([-0.5, 0.5])
    want_b[1] = 0.3 * (np.array([1, 0]) - p1) + 0.8 * (np.array([0, 1]) - p1)
    assert np.allclose(gb, want_b)


def test_reinforce_solves_chain():
    env = ChainEnv(n=10, max_steps=25)
    theta, hist = ex.train_reinforce(env, iters=40, seed=0)
    assert theta.shape == (10, 2) and len(hist) == 40
    assert hist[:3].mean() < 0.5 and hist[-5:].mean() > 0.9
    pr = ex.softmax(theta)[:-1, 1]
    assert np.all(pr > 0.6) and pr.mean() > 0.8                        # learned to walk right


# ---- Part C ----
def test_td_and_gae():
    r = np.array([0.0, 0.0, 1.0]); V = np.array([0.5, 0.6, 0.8]); g = 0.9
    d = ex.td_residuals(r, V, g, last_value=0.0)
    assert np.allclose(d, [0.9 * 0.6 - 0.5, 0.9 * 0.8 - 0.6, 1 - 0.8])
    assert np.allclose(ex.gae(r, V, g, 0.0), d)                                  # λ=0: TD residual
    assert np.allclose(ex.gae(r, V, g, 1.0), ex.reward_to_go(r, g) - V)          # λ=1: MC advantage
    rng = np.random.default_rng(0)
    r, V = rng.normal(size=7), rng.normal(size=7)
    d = ex.td_residuals(r, V, 0.95, last_value=0.3)
    brute = [sum((0.95 * 0.7) ** l * d[t + l] for l in range(7 - t)) for t in range(7)]
    assert np.allclose(ex.gae(r, V, 0.95, 0.7, last_value=0.3), brute)


def test_ppo_clip_objective():
    rho = np.array([0.5, 1.0, 1.1, 1.5, 0.5, 1.5]); A = np.array([1, 1, 1, 1, -1, -1.0])
    assert np.allclose(ex.ppo_clip_objective(rho, A, 0.2), [0.5, 1.0, 1.1, 1.2, -0.8, -1.5])


def test_ppo_grad_matches_numeric():
    rng = np.random.default_rng(3)
    theta = rng.normal(size=(4, 3))
    states = rng.integers(0, 4, size=40); actions = rng.integers(0, 3, size=40)
    # old policy = a perturbed version so the ratios spread over clipped and unclipped regions
    old = ex.softmax(theta + rng.normal(scale=0.4, size=theta.shape))
    old_logp = np.log(old[states, actions]); adv = rng.normal(size=40)
    loss, g = ex.ppo_loss_and_grad(theta, states, actions, old_logp, adv, eps=0.2, ent_coef=0.05)
    ratio = ex.softmax(theta)[states, actions] / old[states, actions]
    assert np.any(np.abs(ratio - 1) > 0.25)                            # some samples are clipped
    f = lambda th: ex.ppo_loss_and_grad(th, states, actions, old_logp, adv, 0.2, 0.05)[0]
    assert np.isclose(loss, f(theta))
    assert np.allclose(g, num_grad(f, theta), atol=1e-5)
    # at θ = θ_old the clipped loss has the plain policy-gradient gradient
    lp = np.log(ex.softmax(theta)[states, actions])
    _, g1 = ex.ppo_loss_and_grad(theta, states, actions, lp, adv, eps=0.2)
    pg = np.zeros_like(theta)
    for s, a, A in zip(states, actions, adv):
        pg[s] += A * ex.grad_log_softmax(theta[s], a)
    assert np.allclose(-g1, pg / len(states))


def test_ppo_solves_chain_quickly():
    env = ChainEnv(n=10, max_steps=25)
    theta, hist = ex.train_ppo(env, iters=15, seed=0)
    assert len(hist) == 15 and hist[0] < 0.5
    assert hist[-5:].mean() > 0.95
