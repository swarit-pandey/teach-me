"""Provided (non-exercise) code for workbook 26: the MDP container, a sampling
environment wrapper, a Gaussian bandit, and a pretty-printer for gridworld values.
You don't need to edit this file — but read it, it is short.
"""
from typing import NamedTuple, List, Tuple
import numpy as np

ACTIONS = ["up", "right", "down", "left"]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # (d_row, d_col) for each action
ARROWS = ["^", ">", "v", "<"]

# The gridworld used in the lesson figures (7 x 5). S start, G goal (+1), P pit (-1), # wall.
DEMO_LAYOUT = [
    "......G",
    ".###...",
    "...#.P.",
    ".#...#.",
    "S..#...",
]


class MDP(NamedTuple):
    """A finite MDP in array form.

    P[s, a, s2]  probability of landing in s2 after taking action a in state s
    R[s, a, s2]  reward received on that transition
    terminal[s]  True for absorbing terminal states (their value is 0)
    start        index of the start state
    cells        (row, col) of every state, in state-index order (gridworlds only)
    shape        (n_rows, n_cols) of the grid (gridworlds only)
    """
    P: np.ndarray
    R: np.ndarray
    terminal: np.ndarray
    start: int
    cells: List[Tuple[int, int]] = []
    shape: Tuple[int, int] = (0, 0)


class TabularEnv:
    """Turns an MDP into a step()-able environment (the agent never sees P or R).

        env = TabularEnv(mdp, seed=0)
        s = env.reset()
        s, r, done = env.step(a)

    Episodes are cut off after max_steps (done=True) so a bad policy cannot loop forever.
    """

    def __init__(self, mdp: MDP, seed=0, max_steps=100):
        self.mdp, self.max_steps = mdp, max_steps
        self.n_states, self.n_actions = mdp.P.shape[0], mdp.P.shape[1]
        self.rng = np.random.default_rng(seed)
        self.s, self.t = mdp.start, 0

    def reset(self):
        self.s, self.t = self.mdp.start, 0
        return self.s

    def step(self, a):
        p = self.mdp.P[self.s, a]
        s2 = int(self.rng.choice(len(p), p=p))
        r = float(self.mdp.R[self.s, a, s2])
        self.s, self.t = s2, self.t + 1
        done = bool(self.mdp.terminal[s2]) or self.t >= self.max_steps
        return s2, r, done


class GaussianBandit:
    """k-armed bandit: pulling arm a pays a sample from N(means[a], 1)."""

    def __init__(self, means, seed=0):
        self.means = np.asarray(means, dtype=float)
        self.k = len(self.means)
        self.rng = np.random.default_rng(seed)

    def pull(self, a):
        return float(self.rng.normal(self.means[a], 1.0))


def render_values(mdp: MDP, V, policy=None):
    """ASCII picture of a gridworld value function (and optional greedy policy)."""
    rows, cols = mdp.shape
    grid = [["  #####  " for _ in range(cols)] for _ in range(rows)]
    for s, (r, c) in enumerate(mdp.cells):
        arrow = " " if policy is None or mdp.terminal[s] else ARROWS[int(policy[s])]
        grid[r][c] = f"{V[s]:+6.2f}{arrow}  "
    return "\n".join("".join(row) for row in grid)
