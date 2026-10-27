"""Workbook 27 — reference solutions. Try first; read after."""
import numpy as np
from helpers import rollout


# ----------------------------------------------------------------------------
# Part A — the policy gradient on a bandit
# ----------------------------------------------------------------------------

def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))     # subtract max: no overflow
    return e / e.sum(axis=-1, keepdims=True)


def grad_log_softmax(logits, a):
    g = -softmax(logits)
    g[a] += 1.0                                        # e_a - π
    return g


def exact_bandit_gradient(theta, means):
    pi = softmax(theta)
    means = np.asarray(means, float)
    return pi * (means - pi @ means)                   # π_a (μ_a - μ̄)


def gradient_estimates(theta, means, n, baseline=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pi = softmax(theta)
    K = len(pi)
    a = rng.choice(K, size=n, p=pi)
    r = rng.normal(np.asarray(means, float)[a], 1.0)
    G = -np.tile(pi, (n, 1))                           # ∇ log π(a) = e_a - π, one row per sample
    G[np.arange(n), a] += 1.0
    return (r - baseline)[:, None] * G


def train_reinforce_bandit(means, steps, lr=0.1, use_baseline=True, seed=0):
    rng = np.random.default_rng(seed)
    means = np.asarray(means, float)
    theta = np.zeros(len(means))
    b, rewards = 0.0, np.zeros(steps)
    for t in range(steps):
        pi = softmax(theta)
        a = rng.choice(len(pi), p=pi)
        r = rng.normal(means[a], 1.0)
        theta += lr * (r - (b if use_baseline else 0.0)) * grad_log_softmax(theta, a)
        b += (r - b) / (t + 1)                         # running mean of rewards
        rewards[t] = r
    return theta, rewards


# ----------------------------------------------------------------------------
# Part B — REINFORCE on an episodic task
# ----------------------------------------------------------------------------

def reward_to_go(rewards, gamma):
    G, g = np.zeros(len(rewards)), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        G[t] = g
    return G


def reinforce_grad(theta, episodes, gamma, values=None):
    grad = np.zeros_like(theta)
    for ep in episodes:
        G = reward_to_go(ep["rewards"], gamma)
        if values is not None:
            G = G - values[ep["states"]]
        pi = softmax(theta[ep["states"]])               # (T, A)
        g = -pi
        g[np.arange(len(G)), ep["actions"]] += 1.0       # rows: ∇_{θ[s_t]} log π(a_t|s_t)
        np.add.at(grad, ep["states"], G[:, None] * g)    # several steps may share a state
    return grad / len(episodes)


def train_reinforce(env, iters=60, episodes_per_iter=16, lr=0.5, gamma=0.95,
                    use_baseline=True, value_lr=0.2, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions))
    V = np.zeros(env.n_states)
    history = []
    for _ in range(iters):
        eps = [rollout(env, lambda s: softmax(theta[s]), rng) for _ in range(episodes_per_iter)]
        theta += lr * reinforce_grad(theta, eps, gamma, V if use_baseline else None)
        for ep in eps:                                   # value baseline: move V(s) toward observed returns
            G = reward_to_go(ep["rewards"], gamma)
            for s, g in zip(ep["states"], G):
                V[s] += value_lr * (g - V[s])
        history.append(np.mean([ep["rewards"].sum() for ep in eps]))
    return theta, np.array(history)


# ----------------------------------------------------------------------------
# Part C — advantages and PPO
# ----------------------------------------------------------------------------

def td_residuals(rewards, values, gamma, last_value=0.0):
    rewards, values = np.asarray(rewards, float), np.asarray(values, float)
    next_v = np.append(values[1:], last_value)
    return rewards + gamma * next_v - values


def gae(rewards, values, gamma, lam, last_value=0.0):
    delta = td_residuals(rewards, values, gamma, last_value)
    A, running = np.zeros(len(delta)), 0.0
    for t in range(len(delta) - 1, -1, -1):
        running = delta[t] + gamma * lam * running       # A_t = δ_t + γλ A_{t+1}
        A[t] = running
    return A


def ppo_clip_objective(ratio, adv, eps=0.2):
    ratio, adv = np.asarray(ratio, float), np.asarray(adv, float)
    return np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)


def ppo_loss_and_grad(theta, states, actions, old_logp, adv, eps=0.2, ent_coef=0.0):
    n = len(states)
    pi = softmax(theta[states])                          # (n, A)
    logp = np.log(pi[np.arange(n), actions])
    ratio = np.exp(logp - old_logp)
    obj = ppo_clip_objective(ratio, adv, eps)
    logpi = np.log(pi)
    ent = -(pi * logpi).sum(axis=1)
    loss = -obj.mean() - ent_coef * ent.mean()

    # d obj_i / d logits: A ρ (e_a - π) unless the clipped branch is active (then 0)
    clipped = ((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))
    glogp = -pi.copy()
    glogp[np.arange(n), actions] += 1.0
    d_obj = np.where(clipped, 0.0, adv * ratio)[:, None] * glogp
    d_ent = -pi * (logpi + ent[:, None])                 # ∇_z H(softmax z) = -π (log π + H)
    d_loss = -(d_obj + ent_coef * d_ent) / n
    grad = np.zeros_like(theta)
    np.add.at(grad, states, d_loss)
    return loss, grad


def train_ppo(env, iters=30, episodes_per_iter=16, epochs=4, lr=0.5, gamma=0.95, lam=0.9,
              eps=0.2, ent_coef=0.01, value_lr=0.3, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions))
    V = np.zeros(env.n_states)
    history = []
    for _ in range(iters):
        eps_data = [rollout(env, lambda s: softmax(theta[s]), rng) for _ in range(episodes_per_iter)]
        S, A, OLD, ADV, RET = [], [], [], [], []
        for ep in eps_data:
            last = 0.0 if ep["terminal"] else V[ep["states"][-1]]   # timed out: bootstrap (approx.)
            adv = gae(ep["rewards"], V[ep["states"]], gamma, lam, last)
            S.append(ep["states"]); A.append(ep["actions"]); OLD.append(ep["logp"])
            ADV.append(adv); RET.append(adv + V[ep["states"]])        # λ-return target for V
        S, A, OLD, ADV, RET = map(np.concatenate, (S, A, OLD, ADV, RET))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)                # normalize advantages
        for _ in range(epochs):                                      # reuse the same batch
            _, g = ppo_loss_and_grad(theta, S, A, OLD, ADV, eps, ent_coef)
            theta -= lr * g * len(S) / episodes_per_iter             # per-episode scale (like REINFORCE)
        for s, r in zip(S, RET):
            V[s] += value_lr * (r - V[s])
        history.append(np.mean([ep["rewards"].sum() for ep in eps_data]))
    return theta, np.array(history)
