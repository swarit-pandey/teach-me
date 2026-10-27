"""Workbook 28 — reference solutions. Try first; read after."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import helpers
from helpers import TinyGPT, generate


# ----------------------------------------------------------------------------
# Part A — preferences and the reward model
# ----------------------------------------------------------------------------

def bradley_terry_prob(r_chosen, r_rejected):
    return torch.sigmoid(r_chosen - r_rejected)


def bradley_terry_loss(r_chosen, r_rejected):
    # -log σ(Δ) computed stably: logsigmoid never forms σ(Δ) explicitly
    return -F.logsigmoid(r_chosen - r_rejected).mean()


class RewardModel(nn.Module):
    def __init__(self, d_model=32, n_head=2, n_layer=2):
        super().__init__()
        self.body = TinyGPT(d_model=d_model, n_head=n_head, n_layer=n_layer)
        self.head = nn.Linear(d_model, 1)

    def forward(self, tokens):
        h = self.body.hidden(tokens)[:, -1]          # the last position has attended to everything
        return self.head(h).squeeze(-1)


def train_reward_model(chosen, rejected, steps=300, batch=128, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rm = RewardModel()
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        i = torch.randint(len(chosen), (batch,), generator=g)
        loss = bradley_terry_loss(rm(chosen[i]), rm(rejected[i]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return rm


@torch.no_grad()
def preference_accuracy(rm, chosen, rejected):
    return (rm(chosen) > rm(rejected)).float().mean().item()


# ----------------------------------------------------------------------------
# Part B — the KL-regularized objective, token by token
# ----------------------------------------------------------------------------

def optimal_kl_policy(pi_ref, rewards, beta):
    # π* ∝ π_ref · exp(r/β), normalized in log space for stability
    return torch.softmax(torch.log(pi_ref) + rewards / beta, dim=-1)


def sequence_logprobs(model, tokens, prompt_len=1):
    logits = model(tokens[:, :-1])                      # position t predicts token t+1
    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok_logp[:, prompt_len - 1:]                 # only the generated tokens


def kl_penalized_rewards(logp, ref_logp, score, beta):
    rewards = -beta * (logp - ref_logp)                 # per-token KL penalty
    rewards = rewards.clone()
    rewards[:, -1] += score                             # sequence-level reward on the last token
    return rewards


def gae(rewards, values, gamma=1.0, lam=0.95):
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(B)
    for t in range(T - 1, -1, -1):
        next_v = values[:, t + 1] if t + 1 < T else torch.zeros(B)     # terminal after the last token
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        running = delta + gamma * lam * running
        adv[:, t] = running
    return adv, adv + values


def ppo_policy_loss(logp, old_logp, adv, eps=0.2):
    ratio = torch.exp(logp - old_logp)
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
    return -surr.mean()


# ----------------------------------------------------------------------------
# Part C — PPO fine-tuning of the transformer policy
# ----------------------------------------------------------------------------

def ppo_rlhf(policy, ref, value_net, reward_fn, iters=30, batch_size=64, beta=0.5, lr=1e-3,
             value_lr=3e-3, epochs=4, eps=0.2, lam=0.95, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_v = torch.optim.Adam(value_net.parameters(), lr=value_lr)
    hist = {"score": [], "kl": []}
    for _ in range(iters):
        # 1. rollout with the current policy (no gradients: this is "data collection")
        with torch.no_grad():
            tokens = generate(policy, batch_size, generator=g)
            old_logp = sequence_logprobs(policy, tokens)
            ref_logp = sequence_logprobs(ref, tokens)
            values = value_net(tokens)[:, :-1]
            score = reward_fn(tokens)
            rewards = kl_penalized_rewards(old_logp, ref_logp, score, beta)
            adv, ret = gae(rewards, values, 1.0, lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        hist["score"].append(score.mean().item())
        hist["kl"].append((old_logp - ref_logp).sum(1).mean().item())
        # 2. several epochs of clipped policy updates + critic regression on the same batch
        for _ in range(epochs):
            loss_pi = ppo_policy_loss(sequence_logprobs(policy, tokens), old_logp, adv, eps)
            opt_pi.zero_grad()
            loss_pi.backward()
            opt_pi.step()
            loss_v = ((value_net(tokens)[:, :-1] - ret) ** 2).mean()
            opt_v.zero_grad()
            loss_v.backward()
            opt_v.step()
    return hist
