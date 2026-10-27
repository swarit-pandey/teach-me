"""Workbook 29 — reference solutions. Try first; read after."""
import torch
import torch.nn.functional as F
import helpers
from helpers import generate


# ----------------------------------------------------------------------------
# Part A — DPO
# ----------------------------------------------------------------------------

def token_logprobs(model, tokens, prompt_len=1):
    logp = F.log_softmax(model(tokens[:, :-1]), dim=-1)
    tok = logp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok[:, prompt_len - 1:]


def sequence_logprob(model, tokens, prompt_len=1):
    return token_logprobs(model, tokens, prompt_len).sum(-1)     # log π(y|x) = Σ_t log π(y_t|x, y_<t)


def implicit_reward(policy_logp, ref_logp, beta):
    return beta * (policy_logp - ref_logp)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    r_c = implicit_reward(policy_chosen, ref_chosen, beta)
    r_r = implicit_reward(policy_rejected, ref_rejected, beta)
    loss = -F.logsigmoid(r_c - r_r).mean()                         # Bradley–Terry on implicit rewards
    return loss, r_c.detach(), r_r.detach()


def train_dpo(policy, ref, chosen, rejected, steps=150, batch=64, beta=0.5, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():                                          # the reference is frozen: precompute once
        ref_c, ref_r = sequence_logprob(ref, chosen), sequence_logprob(ref, rejected)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    hist = {"loss": [], "margin": [], "acc": []}
    for _ in range(steps):
        i = torch.randint(len(chosen), (batch,), generator=g)
        loss, r_c, r_r = dpo_loss(sequence_logprob(policy, chosen[i]), sequence_logprob(policy, rejected[i]),
                                  ref_c[i], ref_r[i], beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        hist["margin"].append((r_c - r_r).mean().item())
        hist["acc"].append((r_c > r_r).float().mean().item())
    return hist


# ----------------------------------------------------------------------------
# Part B — GRPO with a verifiable reward
# ----------------------------------------------------------------------------

def check_answer(tokens):
    a, b = tokens[:, 1], tokens[:, 3]
    d1, d2 = tokens[:, 5], tokens[:, 6]
    valid = (d1 <= 9) & (d2 <= 9)                                  # '+', '=' or BOS as a digit: wrong
    return (valid & (10 * d1 + d2 == a + b)).float()


def grpo_advantages(rewards, eps=1e-6):
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)                          # all-equal group -> 0: no signal


def kl_k3(logp, ref_logp):
    d = ref_logp - logp
    return torch.exp(d) - d - 1                                    # ≥ 0, unbiased for KL(π‖π_ref)


def grpo_loss(logp, old_logp, ref_logp, adv, eps=0.2, beta=0.04):
    ratio = torch.exp(logp - old_logp)
    A = adv[:, None]                                               # one advantage per completion, every token
    surr = torch.minimum(ratio * A, torch.clamp(ratio, 1 - eps, 1 + eps) * A)
    return -(surr - beta * kl_k3(logp, ref_logp)).mean()


def train_grpo(policy, ref, iters=60, prompts_per_iter=64, group_size=8, beta=0.04, eps=0.2,
               lr=1e-3, epochs=2, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    P = helpers.PROMPT_LEN
    hist = {"accuracy": [], "kl": []}
    for _ in range(iters):
        with torch.no_grad():
            prompts = helpers.arith_prompts(prompts_per_iter, g).repeat_interleave(group_size, 0)
            tokens = generate(policy, len(prompts), length=helpers.ANSWER_LEN, generator=g, prompts=prompts)
            rewards = check_answer(tokens)                                       # the verifier
            adv = grpo_advantages(rewards.view(prompts_per_iter, group_size)).view(-1)
            old_logp = token_logprobs(policy, tokens, P)
            ref_logp = token_logprobs(ref, tokens, P)
        hist["accuracy"].append(rewards.mean().item())
        hist["kl"].append(kl_k3(old_logp, ref_logp).sum(1).mean().item())
        for _ in range(epochs):
            loss = grpo_loss(token_logprobs(policy, tokens, P), old_logp, ref_logp, adv, eps, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return hist
