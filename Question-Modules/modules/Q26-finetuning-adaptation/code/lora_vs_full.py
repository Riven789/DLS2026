"""Q26 demo: full fine-tuning vs LoRA on a two-task sequence (NumPy only).

A single linear layer W (64x64) is trained to high accuracy on task A, then
adapted to task B two ways, BOTH TRAINED TO CONVERGENCE (decayed LR + early
stopping -- the contrast below is a property of the update constraint, not of
undertraining one method):

  (a) full fine-tuning -- all 4096 weights of W updated;
  (b) rank-4 LoRA      -- W frozen; only B_ad (64x4) @ A_ad (4x64) trained.

We measure:
  * forgetting  = accuracy drop on task A after adaptation
  * plasticity  = accuracy gain on task B

Expected result (the note's central claim): LoRA preserves task A better than
full fine-tuning, at some cost on task B, because the rank-4 update is
confined to a 4-dimensional subspace of W's 64-dim row space.

Task construction (learnable structure, not random labels):
  * Inputs x in R^64. A FIXED readout P (8x64, orthonormal rows) maps the
    model output to class scores: logits = P (W x). P is part of the frozen
    model; the learned capability lives entirely in W.
  * Task tau is a target map M_tau (64x64): label(x) = argmax_k P[k].(M_tau x)
    -- deterministic, linearly learnable; W = M_tau solves task tau exactly.
  * Task B is a DOMAIN SHIFT of task A: M_B = M_A + P^T C, where the
    logit-space shift C = (rank-4 "signal" component) + (weak full-rank
    "tail"), and 10% of B's 1024 training labels are flipped (noise). This
    mirrors two empirical facts about real adaptation: the systematic part of
    a domain shift is approximately low-rank (Hu et al. 2022's "intrinsic
    rank" observation), while real fine-tuning data carries label noise a
    high-capacity model can memorize.

Why the contrast emerges at convergence:
  * Full FT has 4096 free parameters vs 1024 noisy samples: it fits B's
    low-rank signal AND memorizes the flipped labels (watch full's
    train-vs-test gap on B), and its update perturbs every row of W -- the
    input structure task A relies on is collateral damage (catastrophic
    forgetting).
  * Rank-4 LoRA can express the low-rank part of the shift but CANNOT
    memorize i.i.d. label noise (noise is full-rank); its update also leaves
    the row-space directions outside the adapter's 4-dim span untouched.
    Task A survives better; task B is fit slightly worse. Bounded update =
    soft protection -- the LoRA story from the note, and a soft analogue of
    EWC's importance-weighted consolidation.

Run:  python lora_vs_full.py      (deterministic: every RNG is seeded)
"""

import time

import numpy as np

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
D = 64          # dimension of W (W is D x D = 64 x 64)
K = 8           # classes (fixed readout P is K x D)
R = 4           # LoRA rank
ALPHA = 1.0     # LoRA scale: W_eff = W + (alpha/r) B_ad @ A_ad
N_A_TRAIN = 2048
N_B_TRAIN = 1024   # finite, partly-noisy adaptation set
N_TEST = 512
LR0 = 0.3
DECAY = 0.9998
MAX_STEPS = 20000
PATIENCE = 1500     # early-stopping patience (steps without improvement)
NOISE_FRAC = 0.10   # fraction of task-B training labels flipped

rng = np.random.default_rng(7)
_Q, _ = np.linalg.qr(rng.standard_normal((D, K)))
P = _Q.T                                   # K x D readout, orthonormal rows


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------
def make_target(seed):
    """Task target map M (D x D): label(x) = argmax_k P[k] . (M x)."""
    return np.random.default_rng(seed).standard_normal((D, D)) / np.sqrt(D)


def sample(seed, n):
    return np.random.default_rng(seed).standard_normal((n, D))


def labels(X, M):
    return np.argmax((X @ M.T) @ P.T, axis=1)


def accuracy(W, X, y):
    return float(np.mean(np.argmax((X @ W.T) @ P.T, axis=1) == y))


# ----------------------------------------------------------------------
# Loss / gradients (softmax cross-entropy over logits = P (W x))
# ----------------------------------------------------------------------
def softloss(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    return float(-np.mean(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))))


def loss_and_grad(W, X, y):
    """Mean loss and gradient wrt W. With Q = P W: dL/dQ = dZ^T X / n, dL/dW = P^T dL/dQ."""
    n = len(y)
    Z = (X @ W.T) @ P.T
    Zs = Z - Z.max(axis=1, keepdims=True)
    p = np.exp(Zs)
    p /= p.sum(axis=1, keepdims=True)
    p[np.arange(n), y] -= 1.0
    dQ = p.T @ X / n
    return softloss(Z, y), P.T @ dQ


# ----------------------------------------------------------------------
# Trainers (both to convergence: decayed LR + early stopping)
# ----------------------------------------------------------------------
def converged_sgd(W, X, y, tag):
    """Full fine-tuning: plain SGD on all of W, stopped at loss plateau."""
    lr = LR0
    best, best_W, patience, t = np.inf, W.copy(), 0, 0
    for t in range(MAX_STEPS):
        loss, gW = loss_and_grad(W, X, y)
        W = W - lr * gW
        lr *= DECAY
        if loss < best - 1e-5:
            best, best_W, patience = loss, W.copy(), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    print(f"    [{tag}] converged: step {t+1}, train loss {best:.4f}")
    return best_W


def converged_lora(W_base, A_ad, B_ad, X, y, tag):
    """LoRA: W_base frozen; SGD on B_ad (D,r) and A_ad (r,D) only.

    W_eff = W_base + (alpha/r) B_ad @ A_ad; with G = dL/dW_eff:
      dL/dB_ad = (alpha/r) G A_ad^T,   dL/dA_ad = (alpha/r) B_ad^T G.
    B_ad zero-initialized => the update starts at exactly zero.
    """
    lr = LR0
    best, bestA, bestB, patience, t = np.inf, A_ad.copy(), B_ad.copy(), 0, 0
    for t in range(MAX_STEPS):
        W_eff = W_base + (ALPHA / R) * (B_ad @ A_ad)
        loss, G = loss_and_grad(W_eff, X, y)
        dZ = (ALPHA / R) * G
        B_ad = B_ad - lr * (dZ @ A_ad.T)
        A_ad = A_ad - lr * (B_ad.T @ dZ)
        lr *= DECAY
        if loss < best - 1e-5:
            best, bestA, bestB, patience = loss, A_ad.copy(), B_ad.copy(), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    print(f"    [{tag}] converged: step {t+1}, train loss {best:.4f}")
    return bestA, bestB


# ----------------------------------------------------------------------
# Task A: train the base "pretrained" model
# ----------------------------------------------------------------------
print("== Task A: train base linear layer W (the 'pretrained' model) ==")
t0 = time.time()
M_A = make_target(seed=1)
XA, XAt = sample(11, N_A_TRAIN), sample(12, N_TEST)
yA, yAt = labels(XA, M_A), labels(XAt, M_A)

W0 = rng.standard_normal((D, D)) * 0.05
W_A = converged_sgd(W0.copy(), XA, yA, tag="A-train")
accA_base = accuracy(W_A, XAt, yAt)
print(f"base accuracy on A (held-out): {accA_base:.3f}")

# ----------------------------------------------------------------------
# Task B: a domain shift of A with low-rank signal + label noise
# ----------------------------------------------------------------------
print("\n== Task B: adapt the task-A model to a shifted, partly noisy task ==")
r99 = np.random.default_rng(99)
a_strong = r99.standard_normal((K, R)) / np.sqrt(R)
S_strong = r99.standard_normal((R, D)) / np.sqrt(D)
C = (a_strong @ S_strong) + 0.7 * r99.standard_normal((K, D)) / np.sqrt(D)
M_B = M_A + P.T @ C

XB, XBt = sample(21, N_B_TRAIN), sample(22, N_TEST)
yB_clean = labels(XB, M_B)
yB = yB_clean.copy()
flip = np.random.default_rng(55).random(N_B_TRAIN) < NOISE_FRAC
alt = np.random.default_rng(56).integers(0, K, N_B_TRAIN)
yB[flip] = alt[flip]
yBt = labels(XBt, M_B)      # test labels are clean
print(f"task-B training set: {N_B_TRAIN} samples, {int(flip.sum())} labels flipped "
      f"({NOISE_FRAC:.0%} noise)")

accB_base = accuracy(W_A, XBt, yBt)
print(f"task-A model on task B, before adapting: {accB_base:.3f}")

print("\n-- (a) full fine-tune (all 64x64 = 4096 weights updated) --")
W_full = converged_sgd(W_A.copy(), XB, yB, tag="full-B")

print("-- (b) LoRA rank-4 (W frozen; (64x4)+(4x64) = 512 factors trained) --")
A_ad = rng.standard_normal((R, D))
B_ad = np.zeros((D, R))
A_ad, B_ad = converged_lora(W_A, A_ad, B_ad, XB, yB, tag="lora-B")
W_lora = W_A + (ALPHA / R) * (B_ad @ A_ad)

# ----------------------------------------------------------------------
# Measure forgetting (drop on A) and plasticity (gain on B)
# ----------------------------------------------------------------------
accA_full = accuracy(W_full, XAt, yAt)
accB_full = accuracy(W_full, XBt, yBt)
accA_lora = accuracy(W_lora, XAt, yAt)
accB_lora = accuracy(W_lora, XBt, yBt)
btr_full = accuracy(W_full, XB, yB)     # on the noisy TRAIN labels
btr_lora = accuracy(W_lora, XB, yB)
chance = 1.0 / K

print(f"\n== Results (held-out accuracy; chance = {chance:.3f}) ==")
print(f"                          task A (retention)  task B (new task)")
print(f"base (task-A model):      {accA_base:17.3f}  {accB_base:16.3f}")
print(f"full fine-tune:           {accA_full:17.3f}  {accB_full:16.3f}")
print(f"LoRA rank-4:              {accA_lora:17.3f}  {accB_lora:16.3f}")

forget_full = accA_base - accA_full
forget_lora = accA_base - accA_lora
print(f"\nforgetting on A (drop):    full = {forget_full:.3f}   LoRA = {forget_lora:.3f}")
print(f"plasticity on B (gain):    full = {accB_full - accB_base:+.3f}   "
      f"LoRA = {accB_lora - accB_base:+.3f}")
print("\nnoise memorization (accuracy on noisy train labels minus clean test):")
print(f"  full fine-tune: {btr_full - accB_full:+.3f}   (fits the flipped labels)")
print(f"  LoRA rank-4:    {btr_lora - accB_lora:+.3f}   (rank-4 cannot memorize "
      f"i.i.d. label noise)")
dF, dL = np.linalg.norm(W_full - W_A), np.linalg.norm(W_lora - W_A)
rankF = np.linalg.matrix_rank(W_full - W_A, tol=0.01)
rankL = np.linalg.matrix_rank(W_lora - W_A, tol=0.01)
print(f"\nupdate size ||W - W_A||_F:  full = {dF:.3f}   LoRA = {dL:.3f}")
print(f"update rank (numerical):    full = {rankF}   LoRA = {rankL}  "
      f"(LoRA bounds the RANK, not the norm)")
print(f"trainable parameters:       full = {W_A.size}   LoRA = {A_ad.size + B_ad.size} "
      f"({(A_ad.size + B_ad.size) / W_A.size:.0%} of full)")

# ----------------------------------------------------------------------
# Pass/fail assertions (the note's central claim, checked)
# ----------------------------------------------------------------------
checks = [
    ("base model learned task A (acc > 0.80)", accA_base > 0.80),
    ("task A does not transfer to B (base < 0.60)", accB_base < 0.60),
    ("full fine-tune adapted to B (gain > 0.15)", accB_full - accB_base > 0.15),
    ("full fine-tune catastrophically forgot A (drop > 0.25)",
     forget_full > 0.25),
    ("LoRA retains task A far better than full fine-tune (gap > 0.08)",
     accA_lora - accA_full > 0.08),
    ("LoRA forgets task A less than full fine-tune", forget_lora < forget_full),
    ("LoRA still adapted to B (gain > 0.10)", accB_lora - accB_base > 0.10),
    ("...at a cost on B (below full fine-tune's B)", accB_lora < accB_full),
    ("full FT memorized noise (B train-test gap > 0.05)",
     btr_full - accB_full > 0.05),
    ("LoRA memorized less noise (smaller gap)",
     (btr_lora - accB_lora) < (btr_full - accB_full)),
]
print("\n== Assertions ==")
all_ok = True
for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    all_ok &= ok

print(f"\n(total wall-clock: {time.time() - t0:.1f}s)")
if all_ok:
    print("\nRESULT: at matched convergence, the rank-4 LoRA update preserves the "
          "base task\nbetter than full fine-tuning at some cost on the new task -- "
          "the bounded-update\nclaim, demonstrated. Full FT additionally memorizes "
          "label noise that the\nrank-constrained update cannot fit.")
else:
    print("\nRESULT: claim NOT demonstrated -- inspect the training losses above.")
raise SystemExit(0 if all_ok else 1)
