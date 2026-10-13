"""Fitness = GFLOP/s of y = A x (2*nnz flops) on ONE fixed sparsity pattern:
n=6000, tridiagonal band + ~5 random nnz per row (~8 nnz/row, no empty rows).

Prints 0 if the candidate's y disagrees with a trusted reference, else the
measured rate. The same pattern scores every candidate in every round — the
system specializes to it, which is the point.

Usage: python score.py [candidate.py]   (default: champion.py)
"""
import importlib.util
import sys
import time

import numpy as np

N = 6000
SEED = 123


def build_pattern():
    rng = np.random.default_rng(SEED)
    rows, cols = [], []
    for i in range(N):
        c = {max(0, i - 1), i, min(N - 1, i + 1)}
        c.update(rng.integers(0, N, size=5).tolist())
        cs = sorted(c)
        rows.extend([i] * len(cs))
        cols.extend(cs)
    rows = np.asarray(rows, dtype=np.int64)
    counts = np.zeros(N + 1, dtype=np.int64)
    np.add.at(counts, rows + 1, 1)
    indptr = np.cumsum(counts)
    data = rng.standard_normal(len(cols))
    return indptr, np.asarray(cols, dtype=np.int64), data


def reference_spmv(indptr, indices, data, x):
    # trusted oracle inside the scorer (no empty rows in this pattern,
    # so reduceat segment semantics are safe)
    return np.add.reduceat(data * x[indices], indptr[:-1])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "champion.py"
    indptr, indices, data = build_pattern()
    nnz = len(data)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(N)
    ref = reference_spmv(indptr, indices, data, x)
    try:
        spec = importlib.util.spec_from_file_location("cand_spmv", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        y = np.asarray(mod.spmv(N, indptr, indices, data, x), dtype=np.float64)
    except Exception:
        print(0)
        return
    if y.shape != ref.shape or not np.allclose(y, ref, atol=1e-9):
        print(0)
        return
    best = float("inf")
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            mod.spmv(N, indptr, indices, data, x)
        dt = (time.perf_counter() - t0) / reps
        best = min(best, dt)
        if reps * dt > 0.1 or reps >= 1024:
            break
        reps *= 4
    print(f"{2.0 * nnz / best / 1e9:.4f}")


if __name__ == "__main__":
    main()
