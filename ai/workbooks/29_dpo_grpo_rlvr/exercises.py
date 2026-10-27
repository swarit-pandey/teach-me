"""Workbook 29 — DPO, GRPO & RL with verifiable rewards.

Rules of the game
  * PyTorch, CPU, autograd allowed. helpers.py (provided) has TinyGPT, generate,
    the LETTERS world from workbook 28 (make_sft_model, preference_pairs, true_score,
    count_a) and the ARITHMETIC world (arith_prompts, make_arith_sft_model,
    sampled_accuracy, arith_decode). Read its docstring first.
  * Shapes: tokens (B, T) = prompt followed by the completion. "Per-token" tensors
    are (B, T − prompt_len); "sequence" tensors are (B,).
    Letters: prompt_len = 1 (just BOS). Arithmetic: prompt_len = 5, answer = 2 tokens.
  * Run   ./course check 29   to grade (≈ 30–60 s: it trains small transformers).
  * Stuck > 20 min? Read the Hint. Still stuck? Peek at solutions.py, then close it
    and re-write from memory.
"""
import torch
import torch.nn.functional as F
import helpers
from helpers import generate


# ----------------------------------------------------------------------------
# Part A — DPO
# ----------------------------------------------------------------------------

def token_logprobs(model, tokens, prompt_len=1):
    """log π(y_t | x, y_<t) for every COMPLETION token: shape (B, T − prompt_len).
    Run the model on tokens[:, :-1], log_softmax, gather the actual next tokens,
    drop the first prompt_len − 1 columns. Keep the autograd graph."""
    raise NotImplementedError


def sequence_logprob(model, tokens, prompt_len=1):
    """log π(y | x) = Σ_t log π(y_t | x, y_<t)  ->  shape (B,)."""
    raise NotImplementedError


def implicit_reward(policy_logp, ref_logp, beta):
    """DPO's implicit reward  r̂(x, y) = β (log π(y|x) − log π_ref(y|x))  (lesson §2)."""
    raise NotImplementedError


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """The DPO loss on a batch of preference pairs, from SEQUENCE log-probs (B,):

        L = −mean log σ( r̂(x, y_w) − r̂(x, y_l) ),   r̂ = implicit_reward(..., beta)

    Return (loss, chosen_rewards, rejected_rewards); the two reward tensors are
    DETACHED (they're for logging: margins and accuracy).
    Hint: F.logsigmoid.
    """
    raise NotImplementedError


def train_dpo(policy, ref, chosen, rejected, steps=150, batch=64, beta=0.5, lr=1e-3, seed=0):
    """DPO fine-tuning. ref is frozen; precompute its sequence log-probs of all pairs
    once under torch.no_grad(). torch.manual_seed(seed); minibatch indices from
    torch.randint(len(chosen), (batch,), generator=torch.Generator().manual_seed(seed));
    torch.optim.Adam(policy.parameters(), lr=lr).
    Each step record hist["loss"], hist["margin"] (mean of r̂_w − r̂_l) and
    hist["acc"] (fraction with r̂_w > r̂_l) for the minibatch. Return hist.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Part B — GRPO with a verifiable reward
# ----------------------------------------------------------------------------

def check_answer(tokens):
    """The verifier. tokens (B, 7) = [BOS, a, +, b, =, d1, d2].
    Return a float tensor (B,): 1.0 if d1 and d2 are digits (≤ 9) and 10·d1 + d2 == a + b,
    else 0.0. (A real RLVR checker parses a math answer or runs unit tests; same idea.)"""
    raise NotImplementedError


def grpo_advantages(rewards, eps=1e-6):
    """Group-relative advantages. rewards: (n_prompts, G), one row per prompt's group.
        A = (r − mean_row) / (std_row + eps)      with the POPULATION std (unbiased=False).
    A group where every reward is equal gets advantage 0 (no learning signal)."""
    raise NotImplementedError


def kl_k3(logp, ref_logp):
    """Per-token KL estimator used by GRPO (Schulman's "k3"):
        exp(ref_logp − logp) − (ref_logp − logp) − 1.
    Always ≥ 0, and its expectation under π is exactly KL(π ‖ π_ref)."""
    raise NotImplementedError


def grpo_loss(logp, old_logp, ref_logp, adv, eps=0.2, beta=0.04):
    """GRPO loss to MINIMIZE. logp, old_logp, ref_logp: per-token (N, L); adv: (N,),
    one advantage per completion, shared by all its tokens.

        per token:  min(ρ·A, clip(ρ, 1−ε, 1+ε)·A) − β · kl_k3(logp, ref_logp),  ρ = exp(logp − old_logp)
        loss = −mean over all tokens.
    """
    raise NotImplementedError


def train_grpo(policy, ref, iters=60, prompts_per_iter=64, group_size=8, beta=0.04, eps=0.2,
               lr=1e-3, epochs=2, seed=0):
    """GRPO on the arithmetic task with check_answer as the reward. NO value network.

    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); Adam(lr).
    Each iteration, under torch.no_grad():
        prompts  = helpers.arith_prompts(prompts_per_iter, g).repeat_interleave(group_size, 0)
        tokens   = generate(policy, len(prompts), length=helpers.ANSWER_LEN, generator=g, prompts=prompts)
        rewards  = check_answer(tokens)
        adv      = grpo_advantages(rewards.view(prompts_per_iter, group_size)).view(-1)
        old_logp = token_logprobs(policy, tokens, helpers.PROMPT_LEN);  ref_logp likewise with ref
    record hist["accuracy"] (mean reward) and hist["kl"] (mean over completions of the
    summed kl_k3); then `epochs` Adam steps on grpo_loss(...) with fresh token_logprobs.
    Return hist.
    """
    raise NotImplementedError
