#!/usr/bin/env python3
"""Q09 demo: MinHash fuzzy dedup, and its effect on a memorization proxy.

Part 1 - MinHash correctness:
  * Synthetic "documents" are random shingle sets over a small universe.
  * Verify P[signature agreement in one coordinate] ~= Jaccard similarity.
  * Dedup a corpus with planted near-duplicates; check recall of dups.

Part 2 - Dedup effect on a memorization proxy (train-loss gap, seen vs unseen):
  * Source: T topics = Dirichlet multinomials over a word vocabulary; each
    document is a bag of words drawn from ONE topic (with sampling noise).
  * Corpus: 150 "seed" docs, each with 3 noisy near-duplicate copies
    (like boilerplate/spam clusters in web crawl), = 600 docs, 4x repetition.
  * Run the Part-1 MinHash+LSH dedup over it (shingles = distinct words).
  * Train a tiny bag-of-words mixture model (pure NumPy, online hard-EM)
    two ways at the SAME budget of 600 doc-visits:
      D = duplicated corpus, 1 epoch   (each seed gets ~4 exposures)
      U = deduped seeds + fresh unique docs, 1 epoch (budget reallocated)
  * Memorization proxy = loss on the SEEN (seed) docs vs UNSEEN docs.
    Repetition buys low loss on the repeated docs (memorization) without
    buying coverage; reallocating the same budget to unique docs shrinks
    the gap and improves unseen-doc loss.
  * Repetition sweep on the deduped corpus alone (data-scarce regime):
    seen loss keeps falling with epochs while unseen-loss gains decay --
    the miniature of the ~4-epoch rule of Muennighoff et al. (2023).

Self-contained: stdlib + NumPy only. Prints PASS/FAIL assertions.
Run: python3 minhash_dedup_demo.py
"""
import numpy as np

P_MERSENNE = (1 << 31) - 1

# ---------------------------------------------------------------- Part 1: MinHash

class HashFamily:
    """k affine hashes h(s) = (a*s + b) mod p over integer shingles."""
    def __init__(self, k: int, seed: int = 0):
        r = np.random.default_rng(seed)
        self.a = r.integers(1, P_MERSENNE, size=k).astype(np.int64)
        self.b = r.integers(0, P_MERSENNE, size=k).astype(np.int64)
        self.k = k

    def signatures(self, shingle_array: np.ndarray) -> np.ndarray:
        """shingle_array: (n,) int64 -> (k,) signature of coordinate-wise minima."""
        M = (shingle_array[None, :] * self.a[:, None] + self.b[:, None]) % P_MERSENNE
        return M.min(axis=1)

def jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0

def sig_jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of matching signature coordinates estimates Jaccard."""
    return float(np.mean(sig_a == sig_b))

def lsh_dedup(shingle_sets, hashes: HashFamily, threshold=0.8, bands=32):
    """MinHash + LSH banding dedup. shingle_sets: list of frozensets of ints.

    Splits each k-length signature into `bands` bands of k/bands rows;
    documents sharing an identical band are candidates; a candidate pair is
    confirmed when the full-signature Jaccard estimate >= threshold.
    Clusters candidates (union-find) and keeps one representative each.
    Returns (kept_indices, cluster_root_of_each_doc).
    """
    k = hashes.k
    rows = k // bands
    if rows * bands != k:
        raise ValueError("signature length must be divisible by bands")
    sigs = [hashes.signatures(np.fromiter(sorted(s), dtype=np.int64)) for s in shingle_sets]
    buckets = {}
    for idx, sig in enumerate(sigs):
        for b in range(bands):
            key = (b, sig[b * rows:(b + 1) * rows].tobytes())
            buckets.setdefault(key, []).append(idx)
    parent = list(range(len(shingle_sets)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in buckets.values():
        if len(members) < 2:
            continue
        first = members[0]
        for other in members[1:]:
            if sig_jaccard_estimate(sigs[first], sigs[other]) >= threshold:
                ra, rb = find(first), find(other)
                if ra != rb:
                    parent[rb] = ra
    keep, seen_roots = [], set()
    for idx in range(len(shingle_sets)):
        root = find(idx)
        if root not in seen_roots:
            seen_roots.add(root)
            keep.append(idx)
    return keep, [find(i) for i in range(len(shingle_sets))]

print("=== Part 1: MinHash correctness ===")
rng = np.random.default_rng(2)
hashes = HashFamily(k=128, seed=1)
UNIV = 5000
trials, n_elem = 4000, 60
match_count = 0
true_js = []
for _ in range(trials):
    n_swaps = int(rng.integers(0, 40))          # vary Jaccard across trials
    a = rng.integers(0, UNIV, size=n_elem)
    b = np.concatenate([a[n_swaps:], rng.integers(0, UNIV, size=n_swaps)])
    rng.shuffle(b)
    sa, sb = frozenset(a.tolist()), frozenset(b.tolist())
    true_js.append(jaccard(sa, sb))
    match_count += int(np.sum(hashes.signatures(a) == hashes.signatures(b)))
mean_j = float(np.mean(true_js))
empirical_p = match_count / (trials * hashes.k)
print(f"mean true Jaccard      : {mean_j:.4f}")
print(f"empirical P[sig match] : {empirical_p:.4f}")
assert abs(empirical_p - mean_j) < 0.02, \
    f"MinHash estimate off: {empirical_p:.4f} vs {mean_j:.4f}"
print("PASS: P[one-coordinate signature agreement] ~= Jaccard (|diff| < 0.02)\n")

# planted near-duplicates: recall of dups, retention of distinct docs
rng = np.random.default_rng(3)
docs, is_dup = [], []
for _ in range(60):
    seed_doc = rng.integers(0, UNIV, size=80)
    docs.append(frozenset(seed_doc.tolist())); is_dup.append(False)
    for _ in range(2):  # two noisy copies: swap 4 of 80 shingles
        copy = seed_doc.copy()
        copy[:4] = rng.integers(0, UNIV, size=4)
        rng.shuffle(copy)
        docs.append(frozenset(copy.tolist())); is_dup.append(True)
for _ in range(60):
    docs.append(frozenset(rng.integers(0, UNIV, size=80).tolist())); is_dup.append(False)

kept, _ = lsh_dedup(docs, hashes, threshold=0.8, bands=32)
kept_set = set(kept)
total_dups, total_fresh = sum(is_dup), len(is_dup) - sum(is_dup)
dups_removed = sum(1 for i, d in enumerate(is_dup) if d and i not in kept_set)
fresh_kept = sum(1 for i, d in enumerate(is_dup) if not d and i in kept_set)
print(f"corpus: {len(docs)} docs = {total_fresh} originals + {total_dups} planted near-dups")
print(f"kept {len(kept)}; removed {len(docs) - len(kept)}")
print(f"near-dups removed : {dups_removed}/{total_dups}")
print(f"originals kept    : {fresh_kept}/{total_fresh}")
assert dups_removed / total_dups > 0.90, "dedup missed too many planted near-duplicates"
assert fresh_kept / total_fresh > 0.90, "dedup removed too many distinct documents"
print("PASS: >90% of planted near-dups removed, >90% of distinct docs retained\n")

# ------------------------------------------------- Part 2: memorization proxy

V = 120            # word vocabulary
T = 100            # source topics == model components
DOC_LEN = 80       # words per document

def make_topics(r):
    """Topics: Dirichlet multinomials over the vocabulary."""
    return r.dirichlet(np.ones(V) * 0.35, size=T)

def sample_doc(topics, topic_idx, r):
    """Bag-of-words counts from one topic (with sampling noise, as real docs)."""
    words = r.choice(V, size=DOC_LEN, p=topics[topic_idx])
    return np.bincount(words, minlength=V).astype(float)

def noisy_copy(counts, r, n_swaps=3):
    """Near-duplicate: keep the doc, resample a few word slots from the doc
    itself (web near-dups share most of their vocabulary)."""
    words = np.repeat(np.arange(V), counts.astype(int)).copy()
    positions = r.choice(len(words), size=n_swaps, replace=False)
    p = counts / counts.sum()
    words[positions] = r.choice(V, size=n_swaps, p=p / p.sum())
    return np.bincount(words, minlength=V).astype(float)

class TinyMixtureLM:
    """Bag-of-words mixture model fit by online hard-EM (NumPy only).

    Each of the K components is a categorical over the vocabulary. Each
    training visit: assign the doc to its best component (argmax likelihood)
    and move that component an EMA step toward the doc's empirical word
    distribution. Duplicated docs therefore pull their component tightly
    onto themselves (memorization); unique docs from one topic average into
    a shared topic estimate (generalization).
    """
    def __init__(self, n_components=T, vocab=V, seed=7):
        r = np.random.default_rng(seed)
        self.logp = -np.log(vocab) + r.normal(0, 0.01, (n_components, vocab))

    def train(self, corpus, epochs=1, alpha=0.15):
        r = np.random.default_rng(13)
        n = len(corpus)
        for _ in range(epochs):
            for idx in r.permutation(n):
                counts = corpus[idx]
                phat = counts / counts.sum()
                c = int(np.argmax(self.logp @ counts))
                p_c = np.exp(self.logp[c])
                p_c = p_c / p_c.sum()
                p_new = (1.0 - alpha) * p_c + alpha * phat
                self.logp[c] = np.log(p_new + 1e-12)
        return self

    def nll_per_word(self, docs):
        """Mean per-word NLL under the best-matching component."""
        tot, n = 0.0, 0
        for counts in docs:
            tot += -float((self.logp @ counts).max())
            n += counts.sum()
        return tot / n

print("=== Part 2: dedup effect on memorization proxy ===")
rng = np.random.default_rng(11)
topics = make_topics(rng)

N_SEEDS, COPIES = 150, 3
BUDGET = (COPIES + 1) * N_SEEDS          # 600 doc-visits for every run

# duplicated corpus: each seed + 3 noisy near-dups
seed_topic = rng.integers(0, T, size=N_SEEDS)
seeds = [sample_doc(topics, seed_topic[i], rng) for i in range(N_SEEDS)]
dup_corpus = []
for seed in seeds:
    dup_corpus.append(seed)
    for _ in range(COPIES):
        dup_corpus.append(noisy_copy(seed, rng))
assert len(dup_corpus) == BUDGET

# MinHash dedup of the duplicated corpus (shingles = distinct words present)
word_sets = [frozenset(np.nonzero(c)[0].tolist()) for c in dup_corpus]
kept_idx, _ = lsh_dedup(word_sets, hashes, threshold=0.7, bands=32)
dedup_corpus = [dup_corpus[i] for i in kept_idx]
print(f"duplicated corpus : {len(dup_corpus)} docs, {N_SEEDS} unique seeds (4x repetition)")
print(f"after MinHash dedup: {len(dedup_corpus)} docs kept")
assert 0.6 * N_SEEDS <= len(dedup_corpus) <= 1.3 * N_SEEDS, \
    "dedup should collapse the corpus to roughly the number of unique seeds"

# fresh unique docs (same source) for the reallocated budget + evaluation.
# Topics are drawn from the topics PRESENT in the seeds, so every unseen doc
# comes from a topic the model had the opportunity to learn (fair eval).
trained_topics = np.unique(seed_topic)
fresh_pool = [sample_doc(topics, int(t), rng)
              for t in trained_topics[rng.integers(0, len(trained_topics),
                                                   size=2 * BUDGET)]]
train_fresh = fresh_pool[: BUDGET - len(dedup_corpus)]
unseen = fresh_pool[BUDGET - len(dedup_corpus):][:150]  # held out, never trained on

# Model D: duplicated corpus, 1 epoch (each seed ~4 exposures)
mD = TinyMixtureLM().train(dup_corpus, epochs=1)
# Model U: deduped seeds + fresh unique docs, 1 epoch (same 600 doc-visits)
mU = TinyMixtureLM().train(dedup_corpus + train_fresh, epochs=1)

def report(name, model):
    seen = model.nll_per_word(seeds)
    un = model.nll_per_word(unseen)
    print(f"{name}: seen(seed) {seen:.4f} | unseen {un:.4f} | gap {un - seen:+.4f}")
    return seen, un

print(f"\ntraining budget for both runs: {BUDGET} doc-visits")
seen_D, unseen_D = report("D: duplicated corpus, 1 epoch           ", mD)
seen_U, unseen_U = report("U: deduped seeds + fresh docs, 1 epoch  ", mU)

assert seen_D < seen_U, "repetition should LOWER loss on the repeated docs (memorization)"
assert unseen_U < unseen_D, "unique data should beat duplicates on unseen docs"
assert (unseen_D - seen_D) > (unseen_U - seen_U), \
    "dedup + fresh data should shrink the seen/unseen memorization gap"
print("PASS: at equal budget, duplicates memorize (low seen-loss, no unseen gain);")
print("      replacing them with unique docs improves unseen loss and shrinks")
print("      the seen/unseen gap (the memorization proxy).\n")

# repetition sweep on the deduped corpus alone (data-scarce regime)
print("repetition sweep (only the deduped corpus available, fixed content):")
print(f"{'epochs':>7} {'visits':>7} {'seen':>8} {'unseen':>8} {'gap':>8}")
sweep = {}
for ep in [1, 2, 4, 8, 16, 32]:
    m = TinyMixtureLM().train(dedup_corpus, epochs=ep)
    s, u = m.nll_per_word(seeds), m.nll_per_word(unseen)
    sweep[ep] = (s, u)
    print(f"{ep:>7} {ep * len(dedup_corpus):>7} {s:>8.4f} {u:>8.4f} {u - s:>+8.4f}")

seen_at = {ep: sweep[ep][0] for ep in sweep}
unseen_at = {ep: sweep[ep][1] for ep in sweep}
gain_early = unseen_at[1] - unseen_at[4]      # improvement from 1 -> 4 epochs
gain_late = unseen_at[16] - unseen_at[32]     # improvement from 16 -> 32 epochs
assert seen_at[32] < seen_at[4], "seen loss should keep falling with epochs (memorization grows)"
assert unseen_at[4] <= unseen_at[1], "a few epochs of repetition should not hurt unseen loss"
assert gain_early > gain_late, "returns on repetition should decay (early gains > late gains)"
assert (unseen_at[32] - seen_at[32]) > (unseen_at[4] - seen_at[4]), \
    "memorization gap should grow with repetition"
print("PASS: repetition up to ~4 epochs is roughly free for unseen loss, while")
print("      seen loss keeps falling and the gap grows; gains from 16 -> 32")
print("      epochs are smaller than from 1 -> 4 -- diminishing returns, the")
print("      miniature of the ~4-epoch rule of Muennighoff et al. (2023).")

print("""
Interpretation:
* Duplicates give repeated gradient steps on the same content: lower loss on
  the duplicated docs (memorization) at no benefit to unseen docs. This is
  the efficiency + memorization argument for deduplication (Lee et al. 2022).
* Reallocating the same budget to unique documents (U) improves generalization
  and shrinks the seen/unseen loss gap.
* If unique data is scarce, repeating it for a few epochs is nearly free, but
  returns decay with more repetition (Muennighoff et al. 2023).""")
