"""The whole RLHF pipeline on the toy world, end to end (run after the tests pass):

    SFT model  ->  reward model from preferences  ->  PPO against the RM, for two β values

It reports, per PPO iteration, the reward-model score, the TRUE labeller score
(3 − |#a − 3|) and the KL to the reference. Watch for reward hacking at small β:
the RM score keeps climbing while the true score turns down.

    python train_rlhf.py              # uses your exercises.py
    COURSE_IMPL=solutions python train_rlhf.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _impl import load  # noqa: E402

ex = load(__file__)      # exercises.py, or solutions.py when COURSE_IMPL=solutions
import torch  # noqa: E402
import helpers  # noqa: E402


def main():
    torch.manual_seed(0)
    ref = helpers.make_sft_model()
    for p in ref.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        y = helpers.generate(ref, 2000, generator=torch.Generator().manual_seed(1))
    print(f"SFT model: mean #a = {helpers.count_a(y).mean():.2f}, true score = {helpers.true_score(y).mean():.3f}")
    print("samples:", ", ".join(helpers.decode(t)[1:] for t in y[:6]))

    ch, rj = helpers.preference_pairs(3000, seed=0)
    rm = ex.train_reward_model(ch, rj, steps=300, seed=0)
    tch, trj = helpers.preference_pairs(2000, seed=1)
    bayes = torch.sigmoid((helpers.true_score(tch) - helpers.true_score(trj)).abs()).mean().item()
    print(f"reward model: held-out preference accuracy {ex.preference_accuracy(rm, tch, trj):.3f} (Bayes limit {bayes:.3f})")
    with torch.no_grad():
        for n in range(9):
            t = torch.tensor([[0] + [1] * n + [2, 3] * 4])[:, :9]   # n 'a's then 'bcbc...'
            print(f"   RM score with {n} a's: {rm(t).item():+.2f}   (true {3 - abs(n - 3)})")
    for p in rm.parameters():
        p.requires_grad_(False)

    for beta in (0.3, 0.05):
        torch.manual_seed(0)
        policy = copy.deepcopy(ref)
        for p in policy.parameters():
            p.requires_grad_(True)
        true_hist = []
        def reward_fn(tokens):
            true_hist.append(helpers.true_score(tokens).mean().item())
            return rm(tokens)
        hist = ex.ppo_rlhf(policy, ref, helpers.ValueNet(), reward_fn, iters=60, beta=beta, seed=0)
        print(f"\nPPO with beta={beta}")
        print(" iter   RM score   true score   KL(pi||ref)")
        for i in range(0, 60, 6):
            print(f" {i:4d}   {hist['score'][i]:+8.3f}   {true_hist[i]:+10.3f}   {hist['kl'][i]:8.3f}")
        i = 59
        print(f" {i:4d}   {hist['score'][i]:+8.3f}   {true_hist[i]:+10.3f}   {hist['kl'][i]:8.3f}")
        with torch.no_grad():
            y = helpers.generate(policy, 6, generator=torch.Generator().manual_seed(2))
        print(" samples:", ", ".join(helpers.decode(t)[1:] for t in y))


if __name__ == "__main__":
    main()
