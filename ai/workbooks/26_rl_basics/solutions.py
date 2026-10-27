"""Workbook 26 — reference solutions. Try first; read after."""
import math
import numpy as np
from helpers import MDP, MOVES, GaussianBandit


# ----------------------------------------------------------------------------
# Part A — returns and dynamic programming
# ----------------------------------------------------------------------------

def discounted_return(rewards, gamma):
    g = 0.0
    for r in reversed(rewards):        # Horner's rule: G = r0 + γ(r1 + γ(r2 + ...))
        g = r + gamma * g
    return g


def returns_to_go(rewards, gamma):
    G = np.zeros(len(rewards))
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g     # G_t = r_t + γ G_{t+1}
        G[t] = g
    return G


def make_gridworld(layout, slip=0.0, step_reward=-0.04, goal_reward=1.0, pit_reward=-1.0):
    rows, cols = len(layout), len(layout[0])
    cells = [(r, c) for r in range(rows) for c in range(cols) if layout[r][c] != "#"]
    index = {cell: i for i, cell in enumerate(cells)}
    S, A = len(cells), 4
    P = np.zeros((S, A, S))
    R = np.zeros((S, A, S))
    terminal = np.array([layout[r][c] in "GP" for r, c in cells])
    start = next((index[(r, c)] for r, c in cells if layout[r][c] == "S"), 0)

    def land(r, c, d):
        nr, nc = r + MOVES[d][0], c + MOVES[d][1]
        if 0 <= nr < rows and 0 <= nc < cols and layout[nr][nc] != "#":
            return index[(nr, nc)]
        return index[(r, c)]              # bumped into an edge or wall: stay put

    def reward_on_entering(s2):
        ch = layout[cells[s2][0]][cells[s2][1]]
        return goal_reward if ch == "G" else pit_reward if ch == "P" else step_reward

    for s, (r, c) in enumerate(cells):
        for a in range(A):
            if terminal[s]:
                P[s, a, s] = 1.0              # absorbing, reward 0
                continue
            # intended direction, and the two perpendicular "slips"
            for d, p in ((a, 1 - slip), ((a + 1) % 4, slip / 2), ((a + 3) % 4, slip / 2)):
                if p == 0:
                    continue
                s2 = land(r, c, d)
                P[s, a, s2] += p
                R[s, a, s2] = reward_on_entering(s2)
    return MDP(P, R, terminal, start, cells, (rows, cols))


def expected_reward(mdp):
    """r(s, a) = sum_s2 P(s2|s,a) R(s,a,s2)."""
    return (mdp.P * mdp.R).sum(axis=2)


def policy_evaluation(mdp, policy, gamma, tol=1e-8, max_iters=100_000):
    r_sa = expected_reward(mdp)
    V = np.zeros(mdp.P.shape[0])
    for _ in range(max_iters):
        Q = r_sa + gamma * mdp.P @ V          # (S,A,S) @ (S,) -> (S,A)
        V_new = (policy * Q).sum(axis=1)
        V_new[mdp.terminal] = 0.0
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def policy_evaluation_exact(mdp, policy, gamma):
    S = mdp.P.shape[0]
    P_pi = np.einsum("sa,sat->st", policy, mdp.P)          # state-to-state matrix under π
    r_pi = (policy * expected_reward(mdp)).sum(axis=1)
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)  # V = r_π + γ P_π V


def q_from_v(mdp, V, gamma):
    return (mdp.P * (mdp.R + gamma * V[None, None, :])).sum(axis=2)


def value_iteration(mdp, gamma, tol=1e-8, max_iters=100_000):
    V = np.zeros(mdp.P.shape[0])
    for k in range(1, max_iters + 1):
        Q = q_from_v(mdp, V, gamma)
        V_new = Q.max(axis=1)
        V_new[mdp.terminal] = 0.0
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            break
    policy = q_from_v(mdp, V, gamma).argmax(axis=1)
    return V, policy, k


# ----------------------------------------------------------------------------
# Part B — learning from experience
# ----------------------------------------------------------------------------

def epsilon_greedy(q_values, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))


def q_learning(env, n_episodes, alpha=0.1, gamma=0.99, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(n_episodes):
        s, done, G = env.reset(), False, 0.0
        while not done:
            a = epsilon_greedy(Q[s], eps, rng)
            s2, r, done = env.step(a)
            # bootstrap from max_a' Q(s2, a') unless s2 is terminal (its value is 0)
            target = r + (0.0 if env.mdp.terminal[s2] else gamma * Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            s, G = s2, G + r
        returns.append(G)
    return Q, np.array(returns)


# ----------------------------------------------------------------------------
# Part C — bandits
# ----------------------------------------------------------------------------

def ucb_scores(Q, N, t, c):
    Q, N = np.asarray(Q, float), np.asarray(N, float)
    scores = np.full(len(Q), np.inf)                 # untried arms first
    tried = N > 0
    scores[tried] = Q[tried] + c * np.sqrt(math.log(t) / N[tried])
    return scores


def run_bandit(means, T, strategy, eps=0.1, c=2.0, seed=0):
    bandit = GaussianBandit(means, seed=seed)
    rng = np.random.default_rng(seed + 1)
    k = len(means)
    Q, N = np.zeros(k), np.zeros(k)
    actions, rewards = np.zeros(T, dtype=int), np.zeros(T)
    for t in range(1, T + 1):
        if strategy == "ucb":
            a = int(np.argmax(ucb_scores(Q, N, t, c)))
        else:
            a = epsilon_greedy(Q, eps if strategy == "eps" else 0.0, rng)
        r = bandit.pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]                    # incremental sample mean
        actions[t - 1], rewards[t - 1] = a, r
    return actions, rewards


def regret_curve(means, actions):
    means = np.asarray(means, float)
    return np.cumsum(means.max() - means[np.asarray(actions)])
