#!/usr/bin/env python3
"""Q12 demo: the frequency boundary of parametric memory, and how retrieval flattens it.

Simulates a universe of facts whose support in the (synthetic) pretraining
corpus follows a Zipf law. Closed-book QA accuracy is generated as a logistic
function of log document-support -- the empirical shape reported by Kandpal et
al. (ICML 2023) and Mallen et al. (ACL 2023), made concrete. We then:

  1. FIT that logistic from noisy binary QA outcomes with pure-NumPy gradient
     descent, i.e. recover the frequency -> accuracy curve from evaluation data;
  2. stratify accuracy by frequency (head/tail) and check the long-tail gap;
  3. simulate RETRIEVAL (non-parametric memory): any fact covered by the index
     is answered with a fixed extraction skill, regardless of frequency --
     the curve goes flat, and even trails closed-book on the head (retrieved
     context can mislead when the answer is already memorized, as Mallen et
     al. observe);
  4. simulate the KNOWLEDGE CUTOFF: facts that postdate the cutoff have zero
     pretraining support, so closed-book collapses, while a learned SKILL
     (frequency-independent procedure) persists on post-cutoff content.

Deterministic (seeded RNG). Every claim is checked with a printed PASS/FAIL
assertion; exit code is 1 if any check fails.

Run:  python3 parametric_memory.py
"""
import sys
import numpy as np

rng = np.random.default_rng(12)

# ---------------------------------------------------------------- helpers
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def bernoulli(p):
    return (rng.random(p.shape) < p).astype(float)

# ------------------------------------------- 1. a Zipf universe of facts
N = 4000                                  # facts in the universe
ranks = np.arange(1, N + 1)
zipf = ranks ** -1.2                      # Zipf: doc support ~ rank^-1.2
zipf /= zipf.sum()

TOTAL_DOCS = 20_000                       # effective "web" mentioning facts
lam = TOTAL_DOCS * zipf                   # expected docs per fact
docs = rng.poisson(lam)                   # realized pretraining support
x = np.log1p(docs)                        # log-frequency feature

# "True" closed-book accuracy: logistic in log document support.
# Head facts (~250+ docs): ~0.99. Zero-support facts: ~0.10 (prior/guessing).
A_TRUE, B_TRUE = 1.1, 2.0                 # slope, midpoint (in log docs)
p_closed = sigmoid(A_TRUE * (x - B_TRUE))
y_closed = bernoulli(p_closed)            # one noisy QA trial per fact

# ----------------------------------- 2. fit the logistic from outcomes only
# Minimize mean Bernoulli NLL of p = sigmoid(w*x + c) by full-batch GD.
w, c, lr = 0.0, 0.0, 1.0
for _ in range(6000):
    p = sigmoid(w * x + c)
    g = p - y_closed                      # d NLL / d logit
    w -= lr * np.mean(g * x)
    c -= lr * np.mean(g)
p_fit = sigmoid(w * x + c)

# --------------------------------------------- 3. retrieval flips it flat
# The index covers 95% of the universe (a fresh crawl is not Zipf-limited).
# Extraction from retrieved context is frequency-independent -- BUT when the
# model has ALREADY memorized a head fact, conditioning on a retrieved
# passage can distract it from its own memory (Mallen et al.: retrieval can
# hurt on popular entities). Distractible iff the fact is well memorized.
COVER, SKILL = 0.95, 0.88
covered = rng.random(N) < COVER
distract = p_closed > 0.97                # only rock-solid memories get misled
p_retrieval = np.where(covered & ~distract, SKILL,
                       np.where(covered & distract, 0.65, p_closed))
y_retrieval = bernoulli(p_retrieval)

# Adaptive retrieval (Mallen et al.): retrieve only when the fitted
# closed-book curve says the parametric answer is unreliable (< 0.5).
adaptive_retrieve = p_fit < 0.5
p_adaptive = np.where(adaptive_retrieve, p_retrieval, p_closed)
y_adaptive = bernoulli(p_adaptive)

# ------------------------------------------------- 4. the knowledge cutoff
# 200 facts postdate the cutoff: zero pretraining support by construction.
POST = 200
p_post_facts = np.full(POST, sigmoid(A_TRUE * (0.0 - B_TRUE)))  # x = log1p(0)
y_post_facts = bernoulli(p_post_facts)
head = slice(0, 100)                      # pre-cutoff head facts (rank <= 100)
y_pre_facts = y_closed[head]

# A learned skill (e.g., summarize/parse): accuracy is set by competence,
# not by whether the CONTENT predates the cutoff.
SKILL_ACC = 0.80
y_pre_skill = bernoulli(np.full(100, SKILL_ACC))
y_post_skill = bernoulli(np.full(POST, SKILL_ACC))

# ------------------------------------------------------------- report
def acc(v):
    return 100.0 * v.mean()

print("Q12: boundaries of parametric knowledge -- NumPy toy")
print("=" * 62)
print(f"facts: {N} | Zipf(a=1.2) support, {TOTAL_DOCS:,} total doc mass")
print(f"zero-support facts (never seen in pretraining): "
      f"{100.0 * (docs == 0).mean():.1f}%")
print()
print("-- 1. frequency -> closed-book accuracy (fitted from outcomes) --")
print(f"   fitted logistic: acc = sigmoid({w:.2f} * log(1+docs) + {c:.2f})"
      f"   [true: {A_TRUE:.2f}, {-A_TRUE * B_TRUE:.2f}]")
print(f"   fitted slope > 0  => accuracy RISES with pretraining frequency")
print()
bins = [(0, 100, "rank   1-100"), (100, 1000, "rank  101-1000"),
        (1000, N, "rank 1001-4000")]
print("-- 2. accuracy by frequency bin (closed-book vs retrieval) --")
print(f"   {'bin':>14} | {'closed-book':>11} | {'retrieval':>9} | n")
cb_bin, rt_bin = [], []
for lo, hi, label in bins:
    sl = slice(lo, hi)
    cb, rt = acc(y_closed[sl]), acc(y_retrieval[sl])
    cb_bin.append(cb); rt_bin.append(rt)
    print(f"   {label:>14} | {cb:10.1f}% | {rt:8.1f}% | {hi - lo}")
print()
print("-- 3. overall accuracy and retrieval budget --")
print(f"   closed-book only : {acc(y_closed):.1f}%   (retrieves on   0% of facts)")
print(f"   always retrieve  : {acc(y_retrieval):.1f}%   (retrieves on 100% of facts)")
print(f"   adaptive retrieve: {acc(y_adaptive):.1f}%"
      f"   (retrieves on {100.0 * adaptive_retrieve.mean():.0f}% of facts)")
print()
print("-- 4. the cutoff: facts go stale, skills persist --")
print(f"   FACTS  pre-cutoff head: {acc(y_pre_facts):.1f}%"
      f"   post-cutoff: {acc(y_post_facts):.1f}%")
print(f"   SKILLS pre-cutoff     : {acc(y_pre_skill):.1f}%"
      f"   post-cutoff: {acc(y_post_skill):.1f}%")
print()

# ------------------------------------------------------- pass/fail checks
checks = [
    ("fitted slope is positive (frequency predicts accuracy)",
     w > 0.5),
    ("fitted curve separates head from tail (head-tail gap > 50 pts)",
     acc(y_closed[head]) - acc(y_closed[1000:]) > 50.0),
    ("closed-book tail (rank>1000) below 25%",
     acc(y_closed[1000:]) < 25.0),
    ("retrieval is FLAT: max-min across bins < 12 pts",
     max(rt_bin) - min(rt_bin) < 12.0),
    ("retrieval lifts the tail by > 50 pts",
     rt_bin[2] - cb_bin[2] > 50.0),
    ("retrieval trails closed-book on the head (Mallen et al. effect)",
     rt_bin[0] < cb_bin[0]),
    ("adaptive retrieval matches the best fixed policy (within 1 pt)",
     acc(y_adaptive) >= max(acc(y_closed), acc(y_retrieval)) - 1.0),
    ("cutoff: fact accuracy drops > 60 pts across the cutoff",
     acc(y_pre_facts) - acc(y_post_facts) > 60.0),
    ("cutoff: skill accuracy changes < 12 pts across the cutoff",
     abs(acc(y_pre_skill) - acc(y_post_skill)) < 12.0),
]
fails = 0
print("-- assertions --")
for msg, ok in checks:
    print(f"   [{'PASS' if ok else 'FAIL'}] {msg}")
    fails += (not ok)
print()
print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
