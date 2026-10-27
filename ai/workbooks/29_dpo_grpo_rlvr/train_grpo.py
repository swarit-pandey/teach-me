"""Longer demo runs for workbook 29 (run after the tests pass):

  1. DPO on the letters world: preferred-response probability and true score.
  2. GRPO with the verifier on arithmetic: accuracy, KL, and pass@k before/after.

    python train_grpo.py                       # uses your exercises.py
    COURSE_IMPL=solutions python train_grpo.py
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _impl import load  # noqa: E402

ex = load(__file__)
import torch  # noqa: E402
import helpers  # noqa: E402


def frozen(m):
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def trainable(m):
    m = copy.deepcopy(m)
    for p in m.parameters():
        p.requires_grad_(True)
    return m


@torch.no_grad()
def pass_at_k(model, k, n_prompts=500, seed=7):
    g = torch.Generator().manual_seed(seed)
    pr = helpers.arith_prompts(n_prompts, g).repeat_interleave(k, 0)
    toks = helpers.generate(model, len(pr), length=2, generator=g, prompts=pr)
    return ex.check_answer(toks).view(n_prompts, k).max(1).values.mean().item()


def main():
    print("=== DPO (letters world) ===")
    ref = frozen(helpers.make_sft_model())
    ch, rj = helpers.preference_pairs(4000, seed=0)
    for beta in (0.1, 0.5):
        policy = trainable(ref)
        hist = ex.train_dpo(policy, ref, ch, rj, steps=300, beta=beta, seed=0)
        g = torch.Generator().manual_seed(5)
        y0 = helpers.generate(ref, 3000, generator=g)
        g = torch.Generator().manual_seed(5)
        y1 = helpers.generate(policy, 3000, generator=g)
        with torch.no_grad():
            kl = (ex.sequence_logprob(policy, y1) - ex.sequence_logprob(ref, y1)).mean().item()
        print(f"beta={beta}: loss {hist['loss'][0]:.3f} -> {sum(hist['loss'][-20:]) / 20:.3f}   "
              f"margin -> {sum(hist['margin'][-20:]) / 20:.2f}   pair acc -> {sum(hist['acc'][-20:]) / 20:.2f}   "
              f"mean #a {helpers.count_a(y0).mean():.2f} -> {helpers.count_a(y1).mean():.2f}   "
              f"true score {helpers.true_score(y0).mean():.3f} -> {helpers.true_score(y1).mean():.3f}   KL {kl:.2f}")

    print("\n=== GRPO + verifier (arithmetic) ===")
    ref = frozen(helpers.make_arith_sft_model())
    policy = trainable(ref)
    print(f"SFT model: pass@1 {helpers.sampled_accuracy(ref, ex.check_answer, n=4000):.3f}   "
          f"greedy-ish (T=0.1) {helpers.sampled_accuracy(ref, ex.check_answer, n=4000, temperature=0.1):.3f}   "
          f"pass@8 {pass_at_k(ref, 8):.3f}")
    hist = ex.train_grpo(policy, ref, iters=60, seed=0)
    for i in list(range(0, 60, 6)) + [59]:
        print(f"  iter {i:3d}   batch accuracy {hist['accuracy'][i]:.3f}   KL {hist['kl'][i]:.3f}")
    print(f"after GRPO: pass@1 {helpers.sampled_accuracy(policy, ex.check_answer, n=4000):.3f}   pass@8 {pass_at_k(policy, 8):.3f}")
    g = torch.Generator().manual_seed(3)
    pr = helpers.arith_prompts(8, g)
    print("samples:", "  ".join(helpers.arith_decode(t) for t in helpers.generate(policy, 8, length=2, generator=g, prompts=pr)))


if __name__ == "__main__":
    main()
