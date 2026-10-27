"""Workbook 27 — Policy gradients → PPO.

Rules of the game
  * NumPy only — and no autograd. Every gradient here is derived by hand (lesson
    sections 2–7) and checked by the tests against finite differences. That is the
    point: after this workbook, `loss.backward()` on a PPO loss holds no mystery.
  * Policies are TABULAR SOFTMAX: theta has shape (n_states, n_actions) (or (K,) for
    a bandit) and π(·|s) = softmax(theta[s]).
  * helpers.py (provided): ChainEnv (a corridor; +1 for reaching the far end) and
    rollout(env, probs_fn, rng), which plays one episode and returns a dict with
    arrays states, actions, rewards, logp and a bool terminal.
  * Run   ./course check 27   to grade.
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import numpy as np
from helpers import rollout


# ----------------------------------------------------------------------------
# Part A — the policy gradient on a bandit
# ----------------------------------------------------------------------------

def softmax(z):
    """Numerically stable softmax along the LAST axis (works for (K,) and (n, K)).
    Hint: subtract z.max(axis=-1, keepdims=True) first (lesson 14)."""
    raise NotImplementedError


def grad_log_softmax(logits, a):
    """∇_logits log softmax(logits)[a]  =  e_a − softmax(logits).   Shape (K,)."""
    raise NotImplementedError


def exact_bandit_gradient(theta, means):
    """The TRUE policy gradient ∇_θ J for a bandit with softmax policy π = softmax(θ)
    and expected rewards means[a]:   J(θ) = Σ_a π_a μ_a.

    Derive it (lesson §3): ∇J = Σ_a π_a μ_a (e_a − π) = π ⊙ (μ − π·μ).
    """
    raise NotImplementedError


def gradient_estimates(theta, means, n, baseline=0.0, seed=0):
    """n independent single-sample REINFORCE estimates of ∇J, as an (n, K) array.

    Sample i:  a_i ~ π (rng.choice(K, size=n, p=π)),  r_i ~ N(means[a_i], 1)
               (rng.normal(means[a], 1.0) with a the whole action array),
               estimate_i = (r_i − baseline) · (e_{a_i} − π).
    Use rng = np.random.default_rng(seed); draw all actions first, then all rewards.
    Vectorize: no Python loop over samples.
    """
    raise NotImplementedError


def train_reinforce_bandit(means, steps, lr=0.1, use_baseline=True, seed=0):
    """Learn a bandit policy with REINFORCE, one pull per step.

    theta starts at zeros; rng = np.random.default_rng(seed); each step:
        π = softmax(θ);  a = rng.choice(K, p=π);  r = rng.normal(means[a], 1.0)
        θ += lr · (r − b) · ∇ log π(a)       with b = running mean of ALL past rewards
                                             (b = 0 if use_baseline is False)
        then update the running mean with r   (b starts at 0; b += (r − b)/(t+1))
    Return (theta, rewards) with rewards the (steps,) array of observed rewards.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — REINFORCE on an episodic task
# ----------------------------------------------------------------------------

def reward_to_go(rewards, gamma):
    """G[t] = r_t + γ r_{t+1} + ... (backwards, O(T)). Same as workbook 26."""
    raise NotImplementedError


def reinforce_grad(theta, episodes, gamma, values=None):
    """Monte-Carlo policy-gradient estimate for a tabular softmax policy:

        ĝ = (1/N) Σ_episodes Σ_t  ∇_θ log π(a_t|s_t) · (G_t − b(s_t))

    episodes: list of dicts from helpers.rollout. G_t = reward_to_go(rewards, gamma).
    values: optional (n_states,) baseline b(s); None means b = 0.
    ∇_θ log π(a|s) is zero except in row s, where it is e_a − π(·|s).
    Returns an array shaped like theta.
    Hint: the same state can appear many times in one episode — np.add.at(grad, states, rows).
    """
    raise NotImplementedError


def train_reinforce(env, iters=60, episodes_per_iter=16, lr=0.5, gamma=0.95,
                    use_baseline=True, value_lr=0.2, seed=0):
    """REINFORCE with an optional learned state-value baseline.

    theta = zeros((env.n_states, env.n_actions)), V = zeros(env.n_states),
    rng = np.random.default_rng(seed). Each iteration:
      1. collect episodes_per_iter episodes:  rollout(env, lambda s: softmax(theta[s]), rng)
      2. theta += lr * reinforce_grad(theta, episodes, gamma, V if use_baseline else None)
      3. for every visited (s_t, G_t):  V[s_t] += value_lr * (G_t − V[s_t])
      4. record the mean undiscounted return of the batch
    Return (theta, history) with history an array of length iters.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — advantages and PPO
# ----------------------------------------------------------------------------

def td_residuals(rewards, values, gamma, last_value=0.0):
    """δ_t = r_t + γ V(s_{t+1}) − V(s_t), for t = 0..T-1.

    values[t] = V(s_t) for the T visited states; V(s_T) = last_value
    (0 if the episode really ended; the critic's guess if it was cut off).
    """
    raise NotImplementedError


def gae(rewards, values, gamma, lam, last_value=0.0):
    """Generalized advantage estimation:  A_t = Σ_l (γλ)^l δ_{t+l}.
    Compute it backwards:  A_t = δ_t + γλ A_{t+1},  A_T = 0.
    λ = 0 gives the one-step TD residual; λ = 1 gives G_t − V(s_t) (when last_value
    is the bootstrap for V(s_T)).
    """
    raise NotImplementedError


def ppo_clip_objective(ratio, adv, eps=0.2):
    """Per-sample PPO surrogate (to be MAXIMIZED):
        min( ρ·A,  clip(ρ, 1−ε, 1+ε)·A ).   Vectorized over arrays."""
    raise NotImplementedError


def ppo_loss_and_grad(theta, states, actions, old_logp, adv, eps=0.2, ent_coef=0.0):
    """PPO loss for a tabular softmax policy AND its exact gradient w.r.t. theta.

        ρ_i   = exp(log π_θ(a_i|s_i) − old_logp_i)
        loss  = −mean_i ppo_clip_objective(ρ_i, adv_i, eps)  −  ent_coef · mean_i H(π_θ(·|s_i))

    Return (loss: float, grad: array shaped like theta).

    Derive the gradient (lesson §7): for each sample, the surrogate's derivative is
      A·ρ·(e_a − π)   unless the CLIPPED branch is active,
                      i.e. (A > 0 and ρ > 1+ε) or (A < 0 and ρ < 1−ε) — then it is 0.
    Entropy of a softmax: ∇_z H = −π ⊙ (log π + H).
    Remember the minus signs and the 1/n of the mean. Accumulate with np.add.at.
    """
    raise NotImplementedError


def train_ppo(env, iters=30, episodes_per_iter=16, epochs=4, lr=0.5, gamma=0.95, lam=0.9,
              eps=0.2, ent_coef=0.01, value_lr=0.3, seed=0):
    """PPO with GAE and a tabular critic. Each iteration:
      1. collect episodes_per_iter episodes with the CURRENT policy (rng as in
         train_reinforce); their ep["logp"] are the old log-probs.
      2. per episode: last = 0 if ep["terminal"] else V[last state];
         adv = gae(rewards, V[states], gamma, lam, last);  ret = adv + V[states].
      3. concatenate everything; normalize adv to mean 0, std 1 (add 1e-8).
      4. repeat `epochs` times on the SAME batch:
             _, g = ppo_loss_and_grad(theta, S, A, OLD, ADV, eps, ent_coef)
             theta -= lr * g * len(S) / episodes_per_iter
         (the factor turns the per-sample mean into a per-episode sum, so lr means
          the same thing as in train_reinforce)
      5. critic: for each (s, ret):  V[s] += value_lr * (ret − V[s])
      6. record the mean undiscounted return of the batch.
    Return (theta, history).
    """
    raise NotImplementedError
