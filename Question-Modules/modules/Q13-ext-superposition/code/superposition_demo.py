#!/usr/bin/env python3
"""Q13 demo: superposition — N features packed into d < N dimensions.

Simulates the core claim of Elhage et al. 2022 (Toy Models of
Superposition): a layer can store more features than it has dimensions by
giving features non-orthogonal (near-collapsed) directions, at the cost of
interference that must be filtered nonlinearly.

Model:
  * N features, each with a random unit direction in R^d (d < N).
  * Feature activity is SPARSE: at each sample only a few features are on.
  * Encoding: x = sum_i a_i * f_i  (linear superposition in R^d).
  * Readout: hat_a_i = relu(<f_i, x>) — project back onto each feature and
    clip negative crosstalk (ReLU as the interference filter).
  * Recovery: a feature is "recovered" if hat_a_i is closer to a_i than to
    any other feature's activity (argmax over correlations works too).

We show:
  1. d slightly < N  + sparse features  -> reliable recovery (PASS)
  2. d << N           + sparse features  -> interference dominates (FAIL)
  3. d slightly < N  + dense features   -> co-activation causes collisions
     (FAIL) — sparsity is what buys capacity.

Pure NumPy + stdlib. Run:  python superposition_demo.py
"""
import numpy as np

rng = np.random.default_rng(13)


def random_feature_bank(n_feat: int, dim: int):
    """N random unit directions in R^d — a non-orthogonal (superposed) bank."""
    F = rng.normal(size=(n_feat, dim))
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    return F


def gram_stats(F: np.ndarray):
    """Off-diagonal cosine similarities of the feature bank."""
    G = F @ F.T
    off = G[~np.eye(len(G), dtype=bool)]
    return off


def sample_activities(n_feat: int, n_samples: int, sparsity: float):
    """Sparse feature activities: each feature on with prob (1-sparsity)."""
    return (rng.random((n_samples, n_feat)) > sparsity).astype(float)


def readout(F: np.ndarray, A: np.ndarray):
    """Encode A (n_samples x N) into R^d, read back with ReLU projection."""
    X = A @ F                # (n_samples, d) — superposed code
    return np.maximum(X @ F.T, 0.0)   # (n_samples, N) — filtered readout


def recovery_rate(F: np.ndarray, A: np.ndarray, thresh: float = 0.35):
    """Fraction of ACTIVE features read out above threshold (true positive rate)."""
    R = readout(F, A)
    active = A > 0
    if not active.any():
        return float("nan")
    return float((R[active] > thresh).mean())


def crosstalk_rate(F: np.ndarray, A: np.ndarray, thresh: float = 0.35):
    """Fraction of INACTIVE features spuriously read out above threshold."""
    R = readout(F, A)
    inactive = A == 0
    if not inactive.any():
        return float("nan")
    return float((R[inactive] > thresh).mean())


def run_case(name, n_feat, dim, sparsity, n_samples=4000):
    F = random_feature_bank(n_feat, dim)
    off = gram_stats(F)
    A = sample_activities(n_feat, n_samples, sparsity)
    tp = recovery_rate(F, A)
    fp = crosstalk_rate(F, A)
    # max off-diagonal |cos|: how non-orthogonal the bank is
    max_cos = float(np.abs(off).max())
    mean_cos = float(np.abs(off).mean())
    ok = tp > 0.95 and fp < 0.05
    print(f"{name}")
    print(f"  N={n_feat} features in d={dim} dims | "
          f"sparsity={sparsity:.2f} (P(feature off))")
    print(f"  feature bank: mean |cos|={mean_cos:.3f}, max |cos|={max_cos:.3f} "
          f"(orthogonal would be 0)")
    print(f"  readout: true-positive rate={tp:.3f}, "
          f"false-positive (crosstalk) rate={fp:.3f}")
    print(f"  => {'PASS: sparse features recovered despite d<N' if ok else 'FAIL: interference dominates'}")
    print()
    return ok


def main():
    print("Q13 superposition demo — features in superposition (Elhage et al. 2022)\n")

    # Case 1: mild compression, sparse features -> superposition works.
    # (2% active => ~4 co-active features per sample, as in the sparse
    #  regime of Elhage et al.'s toy models.)
    ok1 = run_case("Case 1: d slightly < N, sparse features",
                   n_feat=200, dim=160, sparsity=0.98)

    # Case 2: heavy compression, same sparsity -> capacity exceeded.
    ok2 = run_case("Case 2: d << N, sparse features",
                   n_feat=200, dim=20, sparsity=0.98)

    # Case 3: mild compression but DENSE features -> co-activation collisions.
    ok3 = run_case("Case 3: d slightly < N, dense features",
                   n_feat=200, dim=160, sparsity=0.20)

    # Sanity: orthogonal control — d >= N with orthonormal basis is exact.
    d = 200
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    F = Q[:200]
    A = sample_activities(200, 4000, 0.95)
    tp = recovery_rate(F, A)
    fp = crosstalk_rate(F, A)
    print("Control: orthonormal basis (d == N)")
    print(f"  readout: true-positive rate={tp:.3f}, false-positive rate={fp:.3f}")
    print(f"  => {'PASS: no interference when directions are orthogonal' if tp > 0.99 and fp < 0.01 else 'FAIL: unexpected'}")
    print()

    # Assertions (exit code signals pass/fail).
    assert ok1, "Case 1 failed: mild compression + sparsity should be recoverable"
    assert not ok2, "Case 2 unexpectedly passed: d=20 cannot hold N=200 random directions"
    assert not ok3, "Case 3 unexpectedly passed: dense co-activation should collide"
    assert tp > 0.99 and fp < 0.01, "Control failed: orthonormal readout must be exact"
    print("ALL CHECKS PASSED:")
    print("  * sparse features survive d<N (superposition buys capacity),")
    print("  * too few dimensions (d<<N) or dense co-activation breaks readout")
    print("    (interference is the price),")
    print("  * orthogonal directions read out exactly (the ideal superposition")
    print("    approximates at high dimension).")
    print("\nMoral: sparsity is the resource that lets a layer store more")
    print("features than dimensions; the ReLU readout is the interference filter.")


if __name__ == "__main__":
    main()
