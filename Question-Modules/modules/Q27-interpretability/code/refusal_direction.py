#!/usr/bin/env python3
"""Q27 demo: a toy of the Arditi et al. (2024) refusal direction.

Builds a synthetic 2-class "safe/unsafe" representation space in which a
base "model" encodes a CAPABILITY direction (task content, needed for the
task) and a toy post-training stage adds a REFUSAL direction (an independent
direction that gates whether the model answers). We then:

  1. collect activations for harmful vs. harmless "requests",
  2. train a linear probe (difference-in-means, as in Arditi et al.) to find
     the refusal direction,
  3. ablate it (project it out of every activation) and show that refusal
     behavior disappears while task ability -- carried by the orthogonal
     capability direction -- survives,
  4. ablate a RANDOM direction as a control: neither effect appears.

Pure NumPy + stdlib. Run:  python3 refusal_direction.py
"""
import numpy as np

rng = np.random.default_rng(27)

D = 64            # activation dimensionality
N = 400           # examples per class

# ---------------------------------------------------------------------------
# 1. Synthetic "model": two independent generative directions.
#    c = capability direction (carries task signal; both classes need it)
#    r = refusal direction   (added by toy "post-training" for unsafe inputs)
# ---------------------------------------------------------------------------
c = rng.normal(size=D); c /= np.linalg.norm(c)
# make r orthogonal to c (capability must survive refusal ablation)
r = rng.normal(size=D); r -= (r @ c) * c; r /= np.linalg.norm(r)

def make_activations(n, kind, ablate_dir=None):
    """kind: 'harmless' or 'harmful'. Returns (activations, task_labels).
    Task label = sign of capability coordinate (the "task" the model must do).
    Harmful inputs additionally get a positive refusal coordinate."""
    task = rng.choice([-1, 1], size=n)          # task content, both classes
    noise = rng.normal(scale=0.35, size=(n, D))
    A = 1.6 * np.outer(task, c) + noise         # base-model representation
    if kind == "harmful":
        A = A + 2.0 * r                          # post-training refusal signal
    if ablate_dir is not None:
        # directional ablation: project out ablate_dir from every activation
        A = A - np.outer(A @ ablate_dir, ablate_dir)
    return A, task

# ---------------------------------------------------------------------------
def run_experiment(ablate_dir=None):
    """Run the toy model with an optional ablation direction.
    Returns dict with refusal rate on harmful inputs and task accuracy."""
    # harmless requests: no refusal signal injected
    A_harmless, task_h = make_activations(N, "harmless", ablate_dir)
    # harmful requests: refusal signal injected by post-training
    A_harmful, task_u = make_activations(N, "harmful", ablate_dir)

    # "behavior": the deployed model refuses when its refusal readout fires;
    # when not refusing, it answers using the capability readout.
    def behavior(A, task):
        refuses = (A @ r) > 0.9                 # model's own refusal gate
        answers = ~refuses
        task_readout = np.sign(A @ c)           # what the model would answer
        acc = np.mean((task_readout == task)[answers]) if answers.any() else 0.0
        return refuses.mean(), acc

    ref_h, acc_h = behavior(A_harmless, task_h)
    ref_u, acc_u = behavior(A_harmful, task_u)
    return {"refusal_harmless": ref_h, "refusal_harmful": ref_u,
            "task_acc_harmless": acc_h, "task_acc_harmful": acc_u}

# ---------------------------------------------------------------------------
# 2. Difference-in-means probe (Arditi et al. Sec 2.3): find the refusal
#    direction from labeled activations only -- no peeking at r.
# ---------------------------------------------------------------------------
A_harmless_probe, _ = make_activations(N, "harmless")
A_harmful_probe, _ = make_activations(N, "harmful")
r_hat = A_harmful_probe.mean(axis=0) - A_harmless_probe.mean(axis=0)
r_hat /= np.linalg.norm(r_hat)

align = abs(r_hat @ r)                      # |cos| with true refusal dir
cap_overlap = abs(r_hat @ c)                # should be ~0: probe spares c

# ---------------------------------------------------------------------------
# 3. Experiments: baseline, refusal-direction ablation, random-direction control
# ---------------------------------------------------------------------------
base = run_experiment(None)
abl_refusal = run_experiment(r_hat)
random_dir = rng.normal(size=D); random_dir /= np.linalg.norm(random_dir)
abl_random = run_experiment(random_dir)

print("Q27 toy: refusal direction (Arditi et al. 2024, in miniature)")
print("-" * 64)
print(f"probe recovery: |cos(r_hat, r_true)|      = {align:.3f}")
print(f"probe leakage:  |cos(r_hat, c_capability)| = {cap_overlap:.3f}")
print()
print(f"{'condition':<28}{'refuse(harmful)':>16}{'refuse(harmless)':>17}{'task acc':>10}")
print(f"{'baseline':<28}{base['refusal_harmful']:>16.1%}{base['refusal_harmless']:>17.1%}{base['task_acc_harmless']:>10.1%}")
print(f"{'ablate refusal dir r_hat':<28}{abl_refusal['refusal_harmful']:>16.1%}{abl_refusal['refusal_harmless']:>17.1%}{abl_refusal['task_acc_harmless']:>10.1%}")
print(f"{'ablate random dir (ctrl)':<28}{abl_random['refusal_harmful']:>16.1%}{abl_random['refusal_harmless']:>17.1%}{abl_random['task_acc_harmless']:>10.1%}")
print()

# ---------------------------------------------------------------------------
# 4. Pass/fail assertions (the demo's claims, checked)
# ---------------------------------------------------------------------------
checks = {
    "probe recovers refusal direction (|cos| > 0.95)":
        align > 0.95,
    "probe direction spares capability (|cos| < 0.10)":
        cap_overlap < 0.10,
    "baseline: harmful requests refused (rate > 0.95)":
        base["refusal_harmful"] > 0.95,
    "baseline: harmless requests not refused (rate < 0.05)":
        base["refusal_harmless"] < 0.05,
    "ablation kills refusal (harmful refusal rate < 0.05)":
        abl_refusal["refusal_harmful"] < 0.05,
    "task ability survives refusal ablation (acc > 0.95)":
        abl_refusal["task_acc_harmless"] > 0.95 and abl_refusal["task_acc_harmful"] > 0.95,
    "random-direction control changes nothing (refusal > 0.95)":
        abl_random["refusal_harmful"] > 0.95,
}
all_ok = True
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    all_ok &= ok
print()
print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
