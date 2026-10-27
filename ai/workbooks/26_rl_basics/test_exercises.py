import math
import numpy as np
import pytest
from _impl import load

ex = load(__file__)
from helpers import MDP, TabularEnv, DEMO_LAYOUT  # noqa: E402  (load() puts this folder on sys.path)


def chain_mdp(n=6, step=-0.04, goal=1.0):
    """States 0..n-1 in a line, actions 0=left 1=right, deterministic; n-1 is the goal."""
    P = np.zeros((n, 2, n)); R = np.zeros((n, 2, n)); term = np.zeros(n, bool); term[-1] = True
    for s in range(n):
        if term[s]:
            P[s, :, s] = 1; continue
        for a, s2 in ((0, max(s - 1, 0)), (1, s + 1)):
            P[s, a, s2] = 1; R[s, a, s2] = goal if s2 == n - 1 else step
    return MDP(P, R, term, 0)


def chain_vstar(n, gamma, step=-0.04, goal=1.0):
    d = np.arange(n - 1, 0, -1)                 # steps to goal from s = 0..n-2
    return np.append([step * (1 - gamma ** (k - 1)) / (1 - gamma) + gamma ** (k - 1) * goal for k in d], 0.0)


# ---- Part A ----
def test_discounted_return():
    assert math.isclose(ex.discounted_return([1, 1, 1], 0.5), 1.75)
    assert math.isclose(ex.discounted_return([0, 0, 10], 0.9), 8.1)
    assert math.isclose(ex.discounted_return([3, -1, 2], 1.0), 4)
    assert ex.discounted_return([], 0.9) == 0


def test_returns_to_go():
    rng = np.random.default_rng(0)
    r = rng.normal(size=12)
    G = ex.returns_to_go(r, 0.9)
    assert G.shape == (12,)
    assert math.isclose(G[0], ex.discounted_return(list(r), 0.9))
    assert np.allclose(G[:-1], r[:-1] + 0.9 * G[1:])        # the recursion
    assert math.isclose(G[-1], r[-1])


def test_gridworld_deterministic():
    m = ex.make_gridworld(["S.G"], slip=0.0, step_reward=-0.04)
    assert m.P.shape == (3, 4, 3) and m.R.shape == (3, 4, 3)
    assert np.allclose(m.P.sum(axis=2), 1)
    assert m.P[0, 1, 1] == 1 and m.P[0, 0, 0] == 1 and m.P[0, 3, 0] == 1   # right moves, up/left bump
    assert m.P[1, 1, 2] == 1 and m.R[1, 1, 2] == 1.0 and m.R[0, 1, 1] == -0.04
    assert list(m.terminal) == [False, False, True] and m.start == 0
    assert m.P[2, 0, 2] == 1 and np.all(m.R[2] == 0)            # absorbing, no reward
    assert m.cells == [(0, 0), (0, 1), (0, 2)] and tuple(m.shape) == (1, 3)


def test_gridworld_slip_and_walls():
    m = ex.make_gridworld(["...", ".S.", "..#"], slip=0.2)
    assert m.P.shape[0] == 8 and m.start == 4
    assert np.allclose(m.P.sum(axis=2), 1)
    # from the centre (state 4): right -> 5 w.p. .8, slips up -> 1 and down -> 7 w.p. .1 each
    assert np.isclose(m.P[4, 1, 5], .8) and np.isclose(m.P[4, 1, 1], .1) and np.isclose(m.P[4, 1, 7], .1)
    # state 5 = (1,2): 'down' hits the wall at (2,2) -> stays w.p. .8, slips left (->4) / right (bump, stays)
    assert np.isclose(m.P[5, 2, 5], .9) and np.isclose(m.P[5, 2, 4], .1)
    assert not m.terminal.any()


def test_policy_evaluation_small():
    m = ex.make_gridworld(["S.G"])
    right = np.zeros((3, 4)); right[:, 1] = 1
    V = ex.policy_evaluation(m, right, 0.9)
    assert np.allclose(V, [-0.04 + 0.9, 1.0, 0.0], atol=1e-6)
    assert np.allclose(ex.policy_evaluation_exact(m, right, 0.9), V, atol=1e-6)


def test_policy_evaluation_iterative_matches_exact():
    m = ex.make_gridworld(DEMO_LAYOUT, slip=0.2)
    rng = np.random.default_rng(1)
    pi = rng.random((m.P.shape[0], 4)); pi /= pi.sum(1, keepdims=True)
    V1 = ex.policy_evaluation(m, pi, 0.9, tol=1e-10)
    V2 = ex.policy_evaluation_exact(m, pi, 0.9)
    assert np.allclose(V1, V2, atol=1e-7)
    assert np.allclose(V2[m.terminal], 0)
    # V^π(s) = Σ_a π(a|s) Q^π(s,a)
    Q = ex.q_from_v(m, V2, 0.9)
    assert Q.shape == pi.shape
    assert np.allclose((pi * Q).sum(1)[~m.terminal], V2[~m.terminal])


def test_value_iteration():
    m = ex.make_gridworld(DEMO_LAYOUT, slip=0.2)
    V, pi, k = ex.value_iteration(m, 0.9)
    assert pi.shape == (m.P.shape[0],) and 5 < k < 2000
    Q = ex.q_from_v(m, V, 0.9)
    assert np.allclose(Q.max(1)[~m.terminal], V[~m.terminal], atol=1e-6)   # Bellman optimality holds
    rand = np.full((len(V), 4), 0.25)
    assert np.all(V >= ex.policy_evaluation_exact(m, rand, 0.9) - 1e-6)      # V* dominates any policy
    det = np.eye(4)[pi]
    assert np.allclose(ex.policy_evaluation_exact(m, det, 0.9), V, atol=1e-5)  # greedy policy achieves V*
    # next to the goal, the optimal move is "right" into it
    s = m.cells.index((0, 5))
    assert pi[s] == 1 and V[s] > 0.9


def test_value_iteration_chain_closed_form():
    m = chain_mdp(6)
    V, pi, _ = ex.value_iteration(m, 0.95)
    assert np.allclose(V, chain_vstar(6, 0.95), atol=1e-6)
    assert np.all(pi[:-1] == 1)


# ---- Part B ----
def test_epsilon_greedy():
    rng = np.random.default_rng(0)
    q = np.array([0.1, 0.5, 0.2])
    assert all(ex.epsilon_greedy(q, 0.0, rng) == 1 for _ in range(50))
    counts = np.bincount([ex.epsilon_greedy(q, 1.0, rng) for _ in range(3000)], minlength=3)
    assert np.all(np.abs(counts / 3000 - 1 / 3) < 0.04)
    counts = np.bincount([ex.epsilon_greedy(q, 0.3, rng) for _ in range(5000)], minlength=3)
    assert abs(counts[1] / 5000 - (0.7 + 0.1)) < 0.03


def test_q_learning_learns_optimal_values():
    m = chain_mdp(6)
    env = TabularEnv(m, seed=3, max_steps=60)
    Q, rets = ex.q_learning(env, 400, alpha=0.2, gamma=0.95, eps=0.2, seed=0)
    assert Q.shape == (6, 2) and len(rets) == 400
    assert np.all(Q[:-1].argmax(1) == 1)                          # always go right
    assert np.allclose(Q[:-1].max(1), chain_vstar(6, 0.95)[:-1], atol=0.03)
    assert np.mean(rets[-50:]) > np.mean(rets[:20])                 # it improved


# ---- Part C ----
def test_ucb_scores():
    s = ex.ucb_scores([1.0, 2.0, 0.0], [4, 1, 0], t=6, c=2.0)
    assert np.isinf(s[2])
    assert np.isclose(s[0], 1 + 2 * math.sqrt(math.log(6) / 4))
    assert np.isclose(s[1], 2 + 2 * math.sqrt(math.log(6) / 1))


def test_regret_curve():
    means = np.array([0.2, 1.0, 0.5])
    reg = ex.regret_curve(means, [1, 1, 0, 2, 1])
    assert np.allclose(reg, [0, 0, 0.8, 1.3, 1.3])
    assert np.all(ex.regret_curve(means, [1] * 10) == 0)


def test_run_bandit_basics():
    means = np.array([0.0, 1.0, -1.0, 0.5])
    a, r = ex.run_bandit(means, 200, "ucb", seed=0)
    assert a.shape == (200,) and r.shape == (200,)
    assert sorted(a[:4]) == [0, 1, 2, 3]                 # UCB tries every arm once first
    a, _ = ex.run_bandit(means, 4000, "eps", eps=1.0, seed=1)
    assert np.all(np.abs(np.bincount(a, minlength=4) / 4000 - 0.25) < 0.03)
    a, _ = ex.run_bandit(means, 500, "greedy", seed=2)
    assert len(set(a[100:].tolist())) == 1                 # pure greedy has stopped exploring


def test_exploration_beats_greedy():
    final = {s: [] for s in ("greedy", "eps", "ucb")}
    for seed in range(20):
        means = np.random.default_rng(100 + seed).normal(0, 1, 10)
        for s in final:
            a, _ = ex.run_bandit(means, 1000, s, eps=0.1, c=2.0, seed=seed)
            reg = ex.regret_curve(means, a)
            assert np.all(np.diff(reg) >= -1e-12)
            final[s].append(reg[-1])
    g, e, u = (np.mean(final[s]) for s in ("greedy", "eps", "ucb"))
    assert u < e < g, (g, e, u)
