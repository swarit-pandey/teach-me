"""Workbook 26 — RL basics: MDPs, returns & bandits.

Rules of the game
  * NumPy only. Loops are fine where the algorithm is inherently sequential
    (an episode, a bandit run); vectorize the Bellman backups over states/actions.
  * helpers.py (provided) has the MDP container, a step()-able TabularEnv, a
    GaussianBandit and render_values() for printing a gridworld. Read it first.
  * Conventions (same as the lesson): taking action a_t in state s_t yields reward
    r_t and next state s_{t+1};  G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
    Arrays: P[s, a, s2] transition probabilities, R[s, a, s2] rewards,
    policy[s, a] = π(a|s) (rows sum to 1), V shape (S,), Q shape (S, A).
  * Run   ./course check 26   to grade.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
  * Try it:  python -c "import exercises as e, helpers as h; m = e.make_gridworld(h.DEMO_LAYOUT, 0.2);
             V, pi, k = e.value_iteration(m, 0.9); print(h.render_values(m, V, pi))"
"""
import math
import numpy as np
from helpers import MDP, MOVES, GaussianBandit


# ----------------------------------------------------------------------------
# Part A — returns and dynamic programming
# ----------------------------------------------------------------------------

def discounted_return(rewards, gamma):
    """G_0 = r_0 + γ r_1 + γ² r_2 + ...  for a finite list of rewards.

    discounted_return([1, 1, 1], 0.5) -> 1.75.   Empty list -> 0.0.
    Hint: loop backwards with g = r + gamma * g (Horner's rule) — no powers needed.
    """
    raise NotImplementedError


def returns_to_go(rewards, gamma):
    """Array G with G[t] = r_t + γ r_{t+1} + ... + γ^{T-1-t} r_{T-1}, for every t.

    Must be O(T): use the recursion G_t = r_t + γ G_{t+1}, G_T = 0.
    (You'll reuse this in lesson 27 for REINFORCE.)
    """
    raise NotImplementedError


def make_gridworld(layout, slip=0.0, step_reward=-0.04, goal_reward=1.0, pit_reward=-1.0):
    """Build the MDP of a gridworld described by a list of equal-length strings.

    Characters: '.' empty, '#' wall, 'S' start (an empty cell), 'G' goal, 'P' pit.
      * States are the non-wall cells, numbered in row-major order
        (top row left->right, then the next row, ...). cells[s] = (row, col).
      * Actions 0..3 = up, right, down, left; helpers.MOVES[a] = (d_row, d_col).
      * The intended move happens with prob 1 - slip; each of the two
        PERPENDICULAR moves happens with prob slip/2 (so 'up' may slip left/right).
        A move into a wall or off the grid leaves the agent where it is.
      * Reward of a transition depends on the cell you land in: goal_reward if it
        is 'G', pit_reward if 'P', step_reward otherwise (including bumping).
      * G and P are terminal: absorbing (P[s, a, s] = 1) with reward 0.
      * start = index of 'S' (0 if there is no 'S').

    Return helpers.MDP(P, R, terminal, start, cells, (n_rows, n_cols)).
    Hint: two directions can land on the same cell (e.g. both slips bump into
    walls) — accumulate probabilities with +=.
    """
    raise NotImplementedError


def policy_evaluation(mdp, policy, gamma, tol=1e-8, max_iters=100_000):
    """V^π by iterating the Bellman expectation backup until it stops changing:

        V_{k+1}(s) = Σ_a π(a|s) Σ_s2 P(s2|s,a) [ R(s,a,s2) + γ V_k(s2) ]

    Synchronous sweeps (compute the whole new V from the old one). Terminal states
    keep value 0. Stop when max|V_{k+1} - V_k| < tol.
    Hint: r(s,a) = (P * R).sum(axis=2) and  P @ V  has shape (S, A).
    """
    raise NotImplementedError


def policy_evaluation_exact(mdp, policy, gamma):
    """V^π in one shot: the Bellman expectation equation is LINEAR in V,

        V = r_π + γ P_π V   =>   (I - γ P_π) V = r_π,

    where P_π[s, s2] = Σ_a π(a|s) P(s2|s,a) and r_π[s] = Σ_a π(a|s) r(s,a).
    Use np.linalg.solve (never invert). Assumes gamma < 1.
    """
    raise NotImplementedError


def q_from_v(mdp, V, gamma):
    """One-step lookahead:  Q(s,a) = Σ_s2 P(s2|s,a) [R(s,a,s2) + γ V(s2)].  Shape (S, A)."""
    raise NotImplementedError


def value_iteration(mdp, gamma, tol=1e-8, max_iters=100_000):
    """Optimal values by iterating the Bellman OPTIMALITY backup

        V_{k+1}(s) = max_a Σ_s2 P(s2|s,a) [R(s,a,s2) + γ V_k(s2)],

    starting from V_0 = 0, terminal states pinned to 0, until max|ΔV| < tol.
    Return (V, policy, n_iters) where policy[s] = argmax_a Q(s, a) (an int array)
    and n_iters is the number of sweeps performed.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — learning from experience (no access to P or R)
# ----------------------------------------------------------------------------

def epsilon_greedy(q_values, eps, rng):
    """With probability eps pick a uniformly random action (rng.integers),
    otherwise argmax(q_values) (ties -> lowest index, i.e. np.argmax). Return an int.
    Draw rng.random() exactly once per call to decide.
    """
    raise NotImplementedError


def q_learning(env, n_episodes, alpha=0.1, gamma=0.99, eps=0.1, seed=0):
    """Tabular Q-learning on a helpers.TabularEnv.

    For each episode: s = env.reset(); repeat until done:
        a = epsilon_greedy(Q[s], eps, rng)
        s2, r, done = env.step(a)
        target = r                         if s2 is terminal (env.mdp.terminal[s2])
                 r + γ max_a' Q[s2, a']    otherwise
        Q[s, a] += alpha * (target - Q[s, a]);  s = s2
    Use rng = np.random.default_rng(seed). Q starts at zeros (env.n_states, env.n_actions).
    Return (Q, returns) where returns[i] is the undiscounted sum of rewards of episode i.
    Note: an episode cut off by the time limit is NOT terminal — still bootstrap.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — multi-armed bandits
# ----------------------------------------------------------------------------

def ucb_scores(Q, N, t, c):
    """UCB1 scores  Q[a] + c * sqrt(ln t / N[a]);  arms with N[a] == 0 score +inf.

    t is the 1-based index of the current step (t = 1 on the first pull).
    """
    raise NotImplementedError


def run_bandit(means, T, strategy, eps=0.1, c=2.0, seed=0):
    """Play T rounds on helpers.GaussianBandit(means, seed=seed).

    strategy: "greedy" (eps = 0), "eps" (ε-greedy with the given eps) or "ucb"
    (argmax of ucb_scores with the given c, t = 1..T).
    Keep sample-average estimates Q (start at 0) and counts N; after reward r on arm a:
        N[a] += 1;  Q[a] += (r - Q[a]) / N[a]          (incremental mean)
    For the ε coin flips use epsilon_greedy with rng = np.random.default_rng(seed + 1).
    Return (actions, rewards): int array (T,) and float array (T,).
    """
    raise NotImplementedError


def regret_curve(means, actions):
    """Cumulative (pseudo-)regret after each step: cumsum over t of
    (max_a means[a] - means[actions[t]]). Shape (T,), non-decreasing."""
    raise NotImplementedError
