"""Workbook 28 — RLHF: reward models & PPO with a KL penalty.

Rules of the game
  * PyTorch, CPU. Use autograd freely now (you hand-derived these gradients in
    workbook 27) — but know what every tensor's shape is.
  * helpers.py (provided) has the whole toy world: TinyGPT, ValueNet (the critic),
    sample_reference, preference_pairs, true_score, count_a, generate, pretrain,
    make_sft_model and decode. Read its docstring first.
  * Shapes: a batch of completions is a LongTensor `tokens` of shape (B, 9):
    BOS followed by 8 generated letters. Per-token tensors (log-probs, rewards,
    values, advantages) have shape (B, 8): one entry per GENERATED token.
  * Run   ./course check 28   to grade (≈ 30–60 s: it trains small transformers).
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
  * After it passes: python train_rlhf.py  runs the whole pipeline end to end.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import helpers
from helpers import TinyGPT, generate


# ----------------------------------------------------------------------------
# Part A — preferences and the reward model
# ----------------------------------------------------------------------------

def bradley_terry_prob(r_chosen, r_rejected):
    """P(chosen ≻ rejected) under Bradley–Terry:  σ(r_chosen − r_rejected).  Elementwise."""
    raise NotImplementedError


def bradley_terry_loss(r_chosen, r_rejected):
    """Mean negative log-likelihood of the observed preferences:
        mean( −log σ(r_chosen − r_rejected) ).
    Hint: F.logsigmoid is numerically stable for large |Δ|; log(sigmoid(Δ)) is not.
    """
    raise NotImplementedError


class RewardModel(nn.Module):
    """A transformer that maps a whole completion to ONE scalar score.

    __init__: self.body = TinyGPT(d_model=d_model, n_head=n_head, n_layer=n_layer)
              self.head = nn.Linear(d_model, 1)
    forward(tokens (B, T)) -> (B,): apply the head to the body's hidden state at the
    LAST position (the only position whose causal attention has seen every token).
    Hint: self.body.hidden(tokens) has shape (B, T, d_model).
    """

    def __init__(self, d_model=32, n_head=2, n_layer=2):
        super().__init__()
        raise NotImplementedError

    def forward(self, tokens):
        raise NotImplementedError


def train_reward_model(chosen, rejected, steps=300, batch=128, lr=3e-3, seed=0):
    """Fit a RewardModel to preference pairs by minimizing bradley_terry_loss.

    torch.manual_seed(seed) before building the model; sample minibatch indices with
    torch.randint(len(chosen), (batch,), generator=torch.Generator().manual_seed(seed));
    optimizer torch.optim.AdamW(lr=lr, weight_decay=0.0). Return the trained model.
    """
    raise NotImplementedError


def preference_accuracy(rm, chosen, rejected):
    """Fraction of pairs where rm(chosen) > rm(rejected), as a Python float. No grad."""
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — the KL-regularized objective, token by token
# ----------------------------------------------------------------------------

def optimal_kl_policy(pi_ref, rewards, beta):
    """The maximizer of  E_π[r] − β KL(π ‖ π_ref)  over distributions π (lesson §6):

        π*(y) = π_ref(y) exp(r(y)/β) / Z.

    pi_ref: probabilities (..., K); rewards: (..., K). Normalize over the last dim.
    Hint: work in log space — softmax(log π_ref + r/β) — so tiny β doesn't overflow.
    """
    raise NotImplementedError


def sequence_logprobs(model, tokens, prompt_len=1):
    """Per-token log-probabilities of the generated tokens under `model`.

    tokens: (B, T). Run the model on tokens[:, :-1] (position t predicts token t+1),
    take log_softmax, and gather the log-prob of each actual next token tokens[:, 1:].
    Return only the generated part: shape (B, T − prompt_len).  (Keep the graph:
    PPO backpropagates through this.)
    Hint: logp.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1).
    """
    raise NotImplementedError


def kl_penalized_rewards(logp, ref_logp, score, beta):
    """Per-token RLHF rewards, shape (B, L):

        r_t = −β (log π(y_t|y_<t) − log π_ref(y_t|y_<t))     for every token t
        r_{L-1} additionally gets + score                     (the reward model's verdict)

    so that Σ_t r_t = score − β · (sample estimate of KL(π ‖ π_ref)).
    Don't modify the inputs in place.
    """
    raise NotImplementedError


def gae(rewards, values, gamma=1.0, lam=0.95):
    """Batched GAE over the generated tokens (workbook 27, now with tensors).

    rewards, values: (B, L). The episode ends after the last token, so the value
    after it is 0. Returns (advantages, returns) with returns = advantages + values
    (the critic's regression target). Loop backwards over t; vectorize over B.
    """
    raise NotImplementedError


def ppo_policy_loss(logp, old_logp, adv, eps=0.2):
    """Clipped PPO loss (to MINIMIZE), averaged over every token in the batch:
        −mean( min(ρ·A, clip(ρ, 1−ε, 1+ε)·A) ),   ρ = exp(logp − old_logp).
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part C — PPO fine-tuning of the transformer policy
# ----------------------------------------------------------------------------

def ppo_rlhf(policy, ref, value_net, reward_fn, iters=30, batch_size=64, beta=0.5, lr=1e-3,
             value_lr=3e-3, epochs=4, eps=0.2, lam=0.95, seed=0):
    """RLHF with PPO. `policy` starts as a copy of `ref` (the frozen SFT model);
    `value_net` is a helpers.ValueNet; reward_fn(tokens) -> (B,) scores.

    Setup: torch.manual_seed(seed); g = torch.Generator().manual_seed(seed);
           Adam optimizers for policy (lr) and value_net (value_lr).
    Each iteration:
      1. with torch.no_grad():
           tokens   = generate(policy, batch_size, generator=g)
           old_logp = sequence_logprobs(policy, tokens);  ref_logp = sequence_logprobs(ref, tokens)
           values   = value_net(tokens)[:, :-1]                 # V of each prefix, (B, 8)
           score    = reward_fn(tokens)
           rewards  = kl_penalized_rewards(old_logp, ref_logp, score, beta)
           adv, ret = gae(rewards, values, 1.0, lam);  adv = (adv − mean) / (std + 1e-8)
      2. record hist["score"] (mean score) and hist["kl"] (mean over the batch of
         Σ_t (old_logp − ref_logp): the sequence-level KL estimate)
      3. repeat `epochs` times: one Adam step on ppo_policy_loss(sequence_logprobs(policy,
         tokens), old_logp, adv, eps), then one Adam step on the critic loss
         mean((value_net(tokens)[:, :-1] − ret)^2).
    Return hist = {"score": [...], "kl": [...]} (one entry per iteration).
    """
    raise NotImplementedError
