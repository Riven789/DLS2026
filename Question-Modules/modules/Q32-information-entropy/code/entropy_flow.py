#!/usr/bin/env python3
"""
entropy_flow.py -- Module Q32, AI Course 2026 (author: Kukla)

Quantifying information flow with entropy, on quantities that are actually
safe to measure, plus a deliberate demonstration of the binning artefact that
makes "information plane" plots untrustworthy.

Runs CPU-only with just numpy. No GPU, no model download required: we simulate
realistic next-token / attention distributions so the file is self-contained
and deterministic. The math is identical to what you'd run on a real model's
logits -- swap `softmax(logits)` for your model's output and nothing changes.

Sections:
  (1) predictive (next-token) entropy
  (2) corpus cross-entropy and perplexity, with the H + KL decomposition
  (3) per-head attention entropy (exact, assumption-free)
  (4) THE CAUTIONARY DEMO: an "information" estimate that changes when you
      only change the number of bins -- i.e. the estimator is the measurement.

Usage:  python3 entropy_flow.py
Deps:   numpy (>=1.20).  Python 3.8+.
"""
from __future__ import annotations
import numpy as np

rng = np.random.default_rng(42)
LOG2 = np.log(2.0)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def entropy_bits(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in bits. Safe at p=0 (0*log0 := 0)."""
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis) / LOG2


# ---------------------------------------------------------------------------
# (1) predictive next-token entropy
# ---------------------------------------------------------------------------
def demo_predictive_entropy(vocab: int = 50000) -> None:
    print("=" * 68)
    print("(1) PREDICTIVE (next-token) ENTROPY")
    print("=" * 68)
    uniform_bits = np.log2(vocab)
    # A "confident" distribution (one dominant token) vs a "hedging" one.
    confident = np.full(vocab, 0.01 / (vocab - 1))
    confident[0] = 0.99
    base = rng.standard_normal(vocab) * 0.3
    hedging = softmax(base)  # near-flat-ish
    for name, p in [("confident (0.99 on top token)", confident),
                    ("hedging (near-flat)", hedging)]:
        h = entropy_bits(p)
        print(f"  {name:32s}  H = {h:6.2f} bits   "
              f"({100*h/uniform_bits:4.1f}% of uniform {uniform_bits:.1f})")
    # Temperature reshapes entropy -- the number is about the SAMPLING dist.
    logits = rng.standard_normal(vocab)
    print("\n  Same logits, different temperature (entropy is post-softmax):")
    for t in (0.5, 1.0, 1.5, 2.0):
        print(f"    T={t:<4}  H = {entropy_bits(softmax(logits, t)):6.2f} bits")
    print()


# ---------------------------------------------------------------------------
# (2) cross-entropy, perplexity, and the H + KL decomposition
# ---------------------------------------------------------------------------
def cross_entropy_bits(p_data: np.ndarray, p_model: np.ndarray) -> float:
    p_model = np.clip(p_model, 1e-12, 1.0)
    return float(-(p_data * np.log(p_model)).sum() / LOG2)


def kl_bits(p_data: np.ndarray, p_model: np.ndarray) -> float:
    p_data_c = np.clip(p_data, 1e-12, 1.0)
    p_model_c = np.clip(p_model, 1e-12, 1.0)
    return float((p_data * (np.log(p_data_c) - np.log(p_model_c))).sum() / LOG2)


def demo_cross_entropy(vocab: int = 2000) -> None:
    print("=" * 68)
    print("(2) CROSS-ENTROPY = PERPLEXITY = COMPRESSION  (CE = H + KL)")
    print("=" * 68)
    # "true" data distribution over a token position (Zipf-ish), and a sequence
    # of increasingly-good model estimates of it.
    ranks = np.arange(1, vocab + 1)
    p_data = (1.0 / ranks)
    p_data /= p_data.sum()
    H_data = entropy_bits(p_data)
    print(f"  Irreducible entropy of the data H(p_data) = {H_data:.3f} bits\n")
    print(f"  {'model':26s} {'CE (bits)':>10s} {'KL (bits)':>10s} {'PPL':>10s}")
    print(f"  {'-'*26} {'-'*10} {'-'*10} {'-'*10}")
    for name, noise in [("random init", 3.0),
                        ("after a little training", 0.8),
                        ("well trained", 0.15),
                        ("~= the data", 0.0)]:
        if noise == 0.0:
            p_model = p_data.copy()
        else:
            p_model = softmax(np.log(np.clip(p_data, 1e-12, 1)) + rng.standard_normal(vocab) * noise)
        ce = cross_entropy_bits(p_data, p_model)
        kl = kl_bits(p_data, p_model)
        ppl = 2.0 ** ce
        # Check the identity CE == H + KL (up to float error):
        assert abs(ce - (H_data + kl)) < 1e-6, "CE = H + KL identity broke!"
        print(f"  {name:26s} {ce:10.3f} {kl:10.3f} {ppl:10.1f}")
    print("\n  CE = H(p_data) + KL(p_data || p_model): training drives KL -> 0,\n"
          "  and CE bottoms out at the irreducible entropy of language itself.\n")


# ---------------------------------------------------------------------------
# (3) per-head attention entropy -- EXACT, assumption-free
# ---------------------------------------------------------------------------
def demo_attention_entropy(seq_len: int = 32, n_heads: int = 6) -> None:
    print("=" * 68)
    print("(3) ATTENTION ENTROPY (exact: weights are already a distribution)")
    print("=" * 68)
    # Simulate attention patterns a real model exhibits:
    heads = {}
    pos = np.arange(seq_len)
    # "pointing" head: sharp, attends to previous token
    s = np.exp(-((pos[None, :] - (pos[:, None] - 1)) ** 2) / 0.5)
    heads["previous-token (pointing)"] = s / s.sum(-1, keepdims=True)
    # "averaging" head: near-uniform over past
    tri = np.tril(np.ones((seq_len, seq_len)))
    heads["uniform-past (averaging)"] = tri / tri.sum(-1, keepdims=True)
    # "sink" head: dumps most mass on token 0
    s = np.full((seq_len, seq_len), 0.01)
    s[:, 0] = 5.0
    s = np.tril(s)
    heads["attention-sink (-> tok 0)"] = s / s.sum(-1, keepdims=True)
    print(f"  {'head':30s} {'mean H (bits)':>14s}  (max = log2(seq) = "
          f"{np.log2(seq_len):.2f})")
    print(f"  {'-'*30} {'-'*14}")
    for name, A in heads.items():
        # entropy per query row, averaged over valid (causal) rows
        h = entropy_bits(A, axis=-1)
        print(f"  {name:30s} {h.mean():14.2f}")
    print("\n  Low entropy = the head is 'pointing'; high = 'averaging'. This is\n"
          "  a legitimate internal information read-out -- no estimator needed.\n")


# ---------------------------------------------------------------------------
# (4) THE CAUTIONARY DEMO: the binning artefact
# ---------------------------------------------------------------------------
def binned_entropy_of_continuous(x: np.ndarray, n_bins: int) -> float:
    """Discretise a continuous variable and measure its (plug-in) entropy."""
    counts, _ = np.histogram(x, bins=n_bins)
    p = counts / counts.sum()
    return float(entropy_bits(p))


def demo_binning_artefact(n: int = 20000) -> None:
    print("=" * 68)
    print("(4) CAUTIONARY DEMO: 'information' that changes with the bin count")
    print("=" * 68)
    # One fixed continuous activation. The underlying variable NEVER changes.
    x = rng.standard_normal(n)
    print("  Same fixed activation x ~ N(0,1), n =", n)
    print("  Estimated entropy (bits) as a function of #bins ONLY:\n")
    print(f"  {'#bins':>8s} {'H_hat (bits)':>14s}")
    print(f"  {'-'*8} {'-'*14}")
    vals = []
    for b in (4, 8, 16, 64, 256, 1024, 4096):
        h = binned_entropy_of_continuous(x, b)
        vals.append(h)
        print(f"  {b:8d} {h:14.2f}")
    print(f"\n  The 'information' nearly doubled from {vals[0]:.2f} to "
          f"{vals[-1]:.2f} bits\n  WITHOUT changing the data -- only the "
          "estimator. This is exactly the\n  knob Saxe et al. (2018) showed "
          "drives the Information-Bottleneck\n  'compression phase'. The "
          "estimator IS the measurement.\n")


def main() -> None:
    demo_predictive_entropy()
    demo_cross_entropy()
    demo_attention_entropy()
    demo_binning_artefact()
    print("Done. All identities (CE = H + KL) asserted & passed.")


if __name__ == "__main__":
    main()
