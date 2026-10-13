"""Benchmark harness — the loop's feedback signal. Prints ONE number
(GFLOP/s) with --report gflops, so the shell can compare it with awk.

Workload: BATCH x (N x N) @ (N x N), float64 inputs. Timing window auto-grows
to ~0.2 s so fast kernels aren't measured in noise.
"""
import sys
import time

import numpy as np

import kernel

BATCH, N = 128, 48
FLOPS = 2.0 * BATCH * N**3


def measure():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((BATCH, N, N))
    B = rng.standard_normal((BATCH, N, N))
    kernel.run(A, B)                       # warm-up
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            kernel.run(A, B)
        dt = (time.perf_counter() - t0) / reps
        if dt * reps > 0.2 or reps >= 256:
            return FLOPS / dt / 1e9
        reps *= 2


if __name__ == "__main__":
    g = measure()
    print(f"{g:.2f}" if "--report" in sys.argv else f"{g:.2f} GFLOP/s")
