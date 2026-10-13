#!/usr/bin/env python3
"""Q11 demo: why a statistical language model hallucinates rare facts.

A toy 'LM' -- a smoothed conditional (bigram-style) count model over
(subject -> object) associations -- is trained on a small, 100% truthful
corpus of fact statements ("SUBJ-r is ENT-r") plus general-text entity
mentions that set entity popularity. Everything below is computed from
that one generated corpus:

  1. Frequency vs. accuracy: facts stated often are answered correctly
     (closed-book, greedy decoding); facts stated once or never are
     'overwritten' by the most plausible continuation (the most popular
     entity). This is the compression/generalization account of
     hallucination in miniature: smoothing shrinks rare associations
     toward the popularity prior.
  2. Confidence vs. frequency: the model's own probability on the true
     answer rises with attestation frequency and crosses 0.5 only for
     the most-attested facts -- the calibration account. Sampling from
     the distribution hallucinates on a strictly larger set of facts
     than greedy decoding does (the reliability boundary sits to the
     right of the argmax-correct boundary).
  3. Good-Turing / monofacts: the fraction of facts stated exactly once
     estimates the fraction of facts never stated -- the quantity that,
     in Kalai & Vempala's idealization, lower-bounds the hallucination
     rate of any calibrated model on arbitrary facts.
  4. Error taxonomy: greedy errors are misattributions (a real,
     well-attested entity, wrong fact); sampled errors also include
     fabrications (objects never attested as anyone's answer, reachable
     only through the model's small 'novelty floor').

Honest scope: an n-gram-style count model is not a Transformer. This
demo isolates ONE statistical mechanism (shrinkage of rare associations
toward the prior under a smoothness penalty). It says nothing about
other causes (architecture, RLHF, multi-step reasoning slips), and it
does not implement the Kalai-Vempala theorem -- only its flavor.

Run:  python3 hallucination_toy.py      (NumPy + stdlib only; seeded)
"""
import sys
import numpy as np

# ----------------------------- world --------------------------------------
SEED = 11
N_SUBJ = 600          # subjects SUBJ-001..600, each with one true answer
N_EXTRA = 120         # unattested objects GHOST-001..120 (in the open
                      # vocabulary, never attested as anyone's answer)
N_ATT = N_SUBJ        # attested answer entities ENT-001..600 (ENT-r is
                      # the true answer of SUBJ-r; rank-correlated
                      # popularity: popular subjects have popular answers)
ZIPF_A = 1.3          # Zipf exponent for subject/entity popularity
F_FACT = 4000         # fact-statement mentions in the corpus (all true)
D_DISC = 6000         # general-text entity mentions (set popularity)
ALPHA = 12.0          # smoothing strength: the 'generalization pressure'
GAMMA = 0.05          # uniform novelty floor over the open vocabulary
M_TEST = 4000         # fresh fact mentions for the missing-mass estimate
N_SAMPLES = 20        # samples per subject in sampling mode
PER_BUCKET = 25       # subjects per bucket in sampling mode

rng = np.random.default_rng(SEED)

ranks = np.arange(1, N_SUBJ + 1)
pop = ranks ** (-ZIPF_A)
pop /= pop.sum()      # p(mention subject r); also entity popularity


def subj_name(i):     # i is 0-based index
    return f"SUBJ-{i+1:03d}"


def ent_name(i):
    return f"ENT-{i+1:03d}" if i < N_ATT else f"GHOST-{i-N_ATT+1:03d}"


# ------------------------- the corpus (all truthful) ----------------------
# Fact statements: F draws of (s, t(s)) -- every statement is true.
fact_counts = rng.multinomial(F_FACT, pop)              # n_s per subject
# General-text mentions: D draws of an entity, popularity-correlated.
disc_counts = rng.multinomial(D_DISC, pop)              # per entity
total_tokens = F_FACT + D_DISC
# Unigram entity popularity, estimated from the same corpus the model sees.
u = (fact_counts + disc_counts) / total_tokens

# ------------------------- the model --------------------------------------
# P(o | s) = (1-gamma) * (C(s,o) + alpha*u(o)) / (C(s) + alpha)  over the
# attested vocabulary, plus gamma * uniform over the open vocabulary
# (attested entities + GHOSTs). C(s, o) is nonzero only for o = t(s),
# with C(s, t(s)) = fact_counts[s]  (the corpus only ever states truths).
V_OPEN = N_ATT + N_EXTRA
floor = GAMMA / V_OPEN
att_scores = np.zeros((N_SUBJ, N_ATT))
att_scores[np.arange(N_SUBJ), np.arange(N_SUBJ)] = fact_counts
att_scores += ALPHA * u[None, :]
att_probs = att_scores / (fact_counts + ALPHA)[:, None]
att_probs *= (1.0 - GAMMA)

greedy_pred = att_probs.argmax(axis=1)          # flat floor never wins argmax
p_truth = att_probs[np.arange(N_SUBJ), np.arange(N_SUBJ)] + floor
correct = greedy_pred == np.arange(N_SUBJ)

# ------------------------- frequency buckets -------------------------------
BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 15), (16, 10**9)]
LABELS = ["0", "1", "2", "3-5", "6-15", "16+"]


def bucket_mask(lo, hi):
    return (fact_counts >= lo) & (fact_counts <= hi)


masks = [bucket_mask(lo, hi) for lo, hi in BUCKETS]
assert sum(m.sum() for m in masks) == N_SUBJ, "buckets must partition subjects"

acc = np.array([correct[m].mean() if m.any() else np.nan for m in masks])
conf = np.array([np.median(p_truth[m]) if m.any() else np.nan for m in masks])
n_in = np.array([m.sum() for m in masks])

# Argmax-correct boundary: first bucket where greedy accuracy is 1.0.
argmax_boundary = next((LABELS[i] for i in range(len(BUCKETS)) if acc[i] == 1.0), None)
# Reliability boundary: first bucket where median P(truth) >= 0.5.
reliab_boundary = next((LABELS[i] for i in range(len(BUCKETS)) if conf[i] >= 0.5), None)

# ------------------------- Good-Turing / monofacts -------------------------
# Monofact rate: fraction of training mentions stating a fact that appears
# exactly once (the Good-Turing estimate of the missing-fact mass).
monofact_rate = float((fact_counts == 1).sum()) / F_FACT
# Empirical missing-fact rate: fresh mentions from the same world process;
# fraction whose fact was never stated in training.
test_draws = rng.choice(N_SUBJ, size=M_TEST, p=pop)
missing_emp = float(np.mean(fact_counts[test_draws] == 0))
# True missing mass (population quantity, for reference only).
true_missing = float(pop[fact_counts == 0].sum())

# ------------------------- sampling mode ----------------------------------
top = int(np.argmax(u))   # most popular entity: the default 'plausible' answer
# Stratified subjects, N_SAMPLES draws each; classify every sampled answer.
rng_choice = np.random.default_rng(SEED + 1)
sample_rows = []      # (bucket label, subject index, sample object index)
for lbl, m in zip(LABELS, masks):
    idxs = np.where(m)[0]
    take = idxs[:PER_BUCKET] if len(idxs) > PER_BUCKET else idxs
    for s in take:
        cdf = np.cumsum(att_probs[s])
        cdf /= cdf[-1]
        gate = rng_choice.random(N_SAMPLES)
        outs = np.empty(N_SAMPLES, dtype=int)
        floor_mask = gate < GAMMA
        n_floor = int(floor_mask.sum())
        outs[floor_mask] = rng_choice.integers(0, V_OPEN, size=n_floor)
        r = rng.random(int((~floor_mask).sum()))
        outs[~floor_mask] = np.searchsorted(cdf, np.clip(r, 0, 1 - 1e-12))
        for o in outs:
            sample_rows.append((lbl, s, int(o)))

n_samp = len(sample_rows)
k_correct = k_mis = k_fab = 0
fab_examples = []
for lbl, s, o in sample_rows:
    if o >= N_ATT:
        k_fab += 1
        if len(fab_examples) < 3:
            fab_examples.append((lbl, subj_name(s), ent_name(o)))
    elif o == s:
        k_correct += 1
    else:
        k_mis += 1

# Greedy error taxonomy (on ALL subjects).
greedy_err = np.where(~correct)[0]
greedy_mis = [(i, int(greedy_pred[i])) for i in greedy_err]
# Analytically: for subject s, column 0 dominates every non-self column
# (alpha*u[0] >= alpha*u[j]), so an incorrect subject's argmax is exactly
# the most popular entity. Assert it rather than trust the algebra.
greedy_all_top = all(j == top for _, j in greedy_mis)

# ----------------------------- reporting ----------------------------------
print("=" * 72)
print("Q11 toy: a smoothed count 'LM' on a 100% truthful factual corpus")
print("=" * 72)
print(f"subjects: {N_SUBJ} (Zipf a={ZIPF_A}) | fact mentions: {F_FACT} | "
      f"discourse mentions: {D_DISC}")
print(f"smoothing alpha={ALPHA} (generalization pressure) | "
      f"novelty floor gamma={GAMMA} over {V_OPEN} open-vocab objects")
print()
print("Frequency vs. accuracy (closed-book QA, greedy decoding):")
print(f"{'train count':>12} {'#facts':>7} {'acc':>6} {'med P(truth)':>13}")
for i, lbl in enumerate(LABELS):
    print(f"{lbl:>12} {n_in[i]:>7} {acc[i]:>6.2f} {conf[i]:>13.3f}")
print()
print("ASCII accuracy curve (each '#' ~ 5%):")
for i, lbl in enumerate(LABELS):
    bar = '#' * int(round(acc[i] / 0.05))
    print(f"  count {lbl:>5} | {bar:<21} {acc[i]:.2f}")
print()
print(f"argmax-correct boundary (greedy accuracy = 1.0):      count >= {argmax_boundary}")
print(f"reliability boundary  (median P(truth) >= 0.5):        count >= {reliab_boundary}")
print("  -> the model's argmax turns correct BEFORE its own confidence")
print("     crosses 0.5: sampling hallucinates on a strictly larger set")
print("     of facts than greedy decoding.")
print()

print(f"most popular entity: {ent_name(top)} (unigram mass {u[top]:.3f}); "
      f"all greedy errors misattribute TO IT")
print("representative subjects (rank ~ attestation):")
print(f"{'subject':>9} {'n_train':>8} {'true':>8} {'greedy pred':>12} "
      f"{'P(true)':>8} {'P(pred)':>8}  verdict")
for r in [1, 8, 30, 60, 100, 150, 400, 550]:
    s = r - 1
    pred = int(greedy_pred[s])
    verdict = "correct" if correct[s] else "MISATTRIBUTED"
    print(f"{subj_name(s):>9} {fact_counts[s]:>8} {ent_name(s):>8} "
          f"{ent_name(pred):>12} {p_truth[s]:>8.3f} "
          f"{att_probs[s, pred] + floor:>8.3f}  {verdict}")
print()

print("Good-Turing / Kalai-Vempala flavor (facts = subject->answer pairs):")
print(f"  monofact rate (facts stated exactly once / mentions): {monofact_rate:.4f}")
print(f"  empirical missing-fact rate (fresh mentions):          {missing_emp:.4f}")
print(f"  true missing mass (population, reference):             {true_missing:.4f}")
print("  -> a calibrated model cannot know unattested facts; the")
print("     monofact rate estimates how often that happens.")
print()

print(f"Sampling mode ({n_samp} samples from P(o|s), stratified):")
print(f"  correct answers:        {k_correct:>5} ({k_correct/n_samp:6.1%})")
print(f"  misattributions:        {k_mis:>5} ({k_mis/n_samp:6.1%})")
print(f"  fabrications (GHOSTs):  {k_fab:>5} ({k_fab/n_samp:6.1%})")
for lbl, s, o in fab_examples:
    print(f"    e.g. bucket [{lbl}] {s} -> {o} (never attested as any answer)")
print()

# ----------------------------- assertions ---------------------------------
results = []


def check(name, ok):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}: {name}")


check("unattested facts are unanswerable (bucket-0 greedy accuracy == 0)",
      acc[0] == 0.0)
check("rare facts are overridden by plausible continuations "
      "(bucket-1 greedy accuracy <= 0.1)", acc[1] <= 0.1)
check("greedy accuracy is non-decreasing across frequency buckets",
      all(acc[i] <= acc[i + 1] + 1e-12 for i in range(len(acc) - 1)))
check("high-frequency facts are all correct (bucket 16+ accuracy == 1.0)",
      acc[-1] == 1.0)
check("Good-Turing: |monofact rate - empirical missing rate| <= 0.05",
      abs(monofact_rate - missing_emp) <= 0.05)
check("greedy errors exist and are misattributions to the single most "
      "popular entity (the plausible continuation)",
      len(greedy_mis) >= 1 and greedy_all_top)
check("sampling fabricates: >=1 unattested object sampled, and "
      "fabrications are rarer than misattributions",
      k_fab >= 1 and k_fab < k_mis)
check("confidence tracks attestation: median P(truth) strictly increases "
      "from bucket 1 to bucket 16+",
      conf[1] < conf[-1] and conf[1] > 0)

print()
if all(results):
    print(f"ALL {len(results)} ASSERTIONS PASSED")
    sys.exit(0)
else:
    print(f"{sum(not r for r in results)} ASSERTION(S) FAILED")
    sys.exit(1)
