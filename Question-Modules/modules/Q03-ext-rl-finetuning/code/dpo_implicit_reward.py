#!/usr/bin/env python3
"""Q03 demo: DPO's implicit reward on a preference-bandit toy.

Tracks three quantities over DPO updates on pairwise preferences:
  1. the implicit reward difference  beta*log(pi(y_w)/pi_ref(y_w))
                                   - beta*log(pi(y_l)/pi_ref(y_l))
  2. the Bradley-Terry loss         -log sigmoid(implicit diff)
  3. the resulting policy's preference accuracy

Shows: the implicit reward margin grows monotonically as the loss falls,
and the policy separates preferred from dispreferred responses -- the
DPO mechanism in miniature, with no reward model anywhere in the code.

Run: python3 dpo_implicit_reward.py   (NumPy only)
"""
import numpy as np

rng = np.random.default_rng(7)

# ---------- toy preference bandit ----------
# 6 candidate responses; each has a TRUE quality (for evaluation only).
# Preference dataset: pairs where the label follows true quality.
# (Real preference data is NOISY -- pairs can contradict; we use clean
#  labels here so the demo isolates the DPO mechanism itself.)
K = 6
true_q = rng.normal(0, 2.0, K)            # wider spread => decisive preferences
REF_LOGP = np.log(np.full(K, 1.0 / K))   # reference = uniform policy

def sample_pair():
    w, l = rng.choice(K, 2, replace=False)
    return (w, l) if true_q[w] > true_q[l] else (l, w)

# ---------- DPO on softmax policy ----------
beta = 2.0
logits = np.zeros(K)                      # pi_0 = uniform = reference
lr = 0.5
pairs = [sample_pair() for _ in range(400)]

losses, margins, accs = [], [], []

def record():
    logp = logits - np.log(np.exp(logits).sum())
    ms = [beta * (logp[w] - REF_LOGP[w]) - beta * (logp[l] - REF_LOGP[l])
          for w, l in pairs]
    losses.append(np.mean([-np.log(1 / (1 + np.exp(-m))) for m in ms]))
    margins.append(np.mean(ms))
    accs.append(np.mean([logp[w] > logp[l] for w, l in pairs]))

record()                                  # BASELINE: before any training
for epoch in range(60):
    for w, l in pairs:
        logp = logits - np.log(np.exp(logits).sum())
        margin = beta * (logp[w] - REF_LOGP[w]) - beta * (logp[l] - REF_LOGP[l])
        loss = -np.log(1 / (1 + np.exp(-margin)))
        # gradient of -log sigmoid(margin) wrt logits: -sigmoid(-margin)*beta*(e_w - e_l)
        g = -(1 / (1 + np.exp(margin))) * beta
        grad = np.zeros(K); grad[w] += g; grad[l] -= g
        logits -= lr * grad
    record()

print("DPO on a preference bandit (no reward model anywhere in the code)")
print(f"epochs: {len(losses)-1}   pairs/epoch: {len(pairs)}   beta: {beta}")
print(f"\nDPO loss      : {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"implicit margin: {margins[0]:+.4f} -> {margins[-1]:+.4f}")
print(f"pair accuracy : {accs[0]:.3f} -> {accs[-1]:.3f}")
pi = np.exp(logits) / np.exp(logits).sum()
print(f"\nfinal policy mass vs true quality rank:")
order = np.argsort(-true_q)
for i in order:
    print(f"  response {i}: q={true_q[i]:+.2f}  pi={pi[i]:.3f}  (ref=0.167)")

checks = [
    ("loss decreased", losses[-1] < 0.5 * losses[0]),
    ("implicit reward margin grew", margins[-1] > margins[0] + 0.5),
    ("pair accuracy >= 0.75", accs[-1] >= 0.75),
    ("policy concentrates on high-quality responses",
     pi[np.argsort(-true_q)[:2]].sum() > 0.5),
    ("KL to reference stays bounded (pi < 0.98 on any response)",
     pi.max() < 0.98),
]
print()
for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
assert all(ok for _, ok in checks), "DPO toy failed a check"
print("\nALL CHECKS PASSED: the log-ratio IS the reward; comparisons alone")
print("separate preferred from dispreferred responses -- no r_phi needed.")
