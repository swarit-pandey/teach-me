"""Provided (non-exercise) code for workbook 27: a small episodic environment and a
rollout function. You don't need to edit this file.
"""
import numpy as np


class ChainEnv:
    """A corridor of n cells. Start in cell 0; the goal is cell n-1.

        actions: 0 = left, 1 = right   (left in cell 0 bumps the wall and stays)
        reward:  +1 when you step into the goal (episode ends), 0 otherwise
        time limit: the episode is cut off after max_steps steps

    A uniformly random policy is a random walk; from cell 0 it needs about
    (n-1)^2 steps on average to reach the goal, so with n=8 and max_steps=30 it
    usually times out. A good policy walks right: n-1 steps, return γ^(n-2).
    """
    n_actions = 2

    def __init__(self, n=8, max_steps=30):
        self.n, self.max_steps = n, max_steps
        self.n_states = n

    def reset(self):
        self.s, self.t = 0, 0
        return self.s

    def step(self, a):
        self.s = max(self.s - 1, 0) if a == 0 else self.s + 1
        self.t += 1
        if self.s == self.n - 1:
            return self.s, 1.0, True
        return self.s, 0.0, self.t >= self.max_steps


def rollout(env, probs_fn, rng):
    """Play one episode. probs_fn(s) -> probability vector over actions.

    Returns a dict of arrays:
      states[t], actions[t], rewards[t], logp[t] = log π(a_t | s_t)   (length T)
      terminal: True if the episode ended in the goal (False if it timed out)
    """
    s, done = env.reset(), False
    S, A, R, L = [], [], [], []
    while not done:
        p = probs_fn(s)
        a = int(rng.choice(len(p), p=p))
        s2, r, done = env.step(a)
        S.append(s); A.append(a); R.append(r); L.append(np.log(p[a]))
        s = s2
    return {"states": np.array(S), "actions": np.array(A), "rewards": np.array(R),
            "logp": np.array(L), "terminal": s == env.n - 1}
