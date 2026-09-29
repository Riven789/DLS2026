"""Fitness function for the conv1d tournament: prints 0 if wrong, else an
effective GFLOP/s (2*n*k / best time — direct-convolution flop equivalent, so
FFT candidates score high by being fast, not by doing more work).

Every candidate is scored by this SAME harness; the champion is max fitness.
Sized to n=2^18, k=1025 so a full 8-candidate arena scores in seconds (the
prompt says 1e6 — scale n up on real hardware if you like).
"""
import importlib.util
import sys
import time

import numpy as np

N = 2**18
K = 1025


def load_conv(path):
    spec = importlib.util.spec_from_file_location("cand_conv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.conv1d_fft


def main():
    path = sys.argv[1]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N)
    k = rng.standard_normal(K)
    ref = np.convolve(x, k)
    try:
        f = load_conv(path)
        y = np.asarray(f(x, k), dtype=np.float64)
    except Exception:
        print(0)
        return
    if y.shape != ref.shape or not np.allclose(y, ref, atol=1e-5):
        print(0)
        return
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        f(x, k)
        best = min(best, time.perf_counter() - t0)
    print(f"{2.0 * N * K / best / 1e9:.2f}")


if __name__ == "__main__":
    main()
