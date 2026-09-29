# Expected output — Problem 5 — Tool-augmented loop (measurement as verification)

The signal is a NUMBER, not pass/fail: `bench.py --report gflops` prints one
float, and awk comparisons implement both stop rules (target > 28 GFLOP/s, or
3 rounds with no improvement → PLATEAU). Captured trajectory:

- round 1 — v1 seed (broadcast-and-sum, giant temporary): 0.25 GFLOP/s
- round 2 — v2 (per-item BLAS dgemm loop): ~20 GFLOP/s (80× — still short)
- round 3 — v3 (one C-level batched matmul, no per-item Python dispatch):
  ~40 GFLOP/s → `HIT: target reached`

Each change is ONE transformation, and the mock re-runs `check.py` after every
edit — optimizations that break correctness don't count.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text
$ python check.py    # correctness gate before optimizing
kernel correct (v1)

round 1: 0.25 GFLOP/s (best 0)
[pi] wrote kernel.py — ONE change: block over the batch — per-item BLAS dgemm replaces the O(n^3)-sized broadcast temporary
[pi] correctness vs check.py: kernel correct (v2)
round 2: 20.22 GFLOP/s (best 0.25)
[pi] wrote kernel.py — ONE change: vectorization/dispatch — one C-level batched matmul instead of 128 Python-dispatched dgemm calls
[pi] correctness vs check.py: kernel correct (v3)
round 3: 39.88 GFLOP/s (best 20.22)
HIT: target reached
```

## What varies between runs/machines

- Every GFLOP/s figure — this is real wall-clock on shared vCPUs. On a fast
  desktop, v2 alone can cross 28 (`HIT` in round 2); on a throttled box v3 can
  miss it, v4 (preallocated output) adds little, and the run exits via
  `PLATEAU` instead. Both endings are designed stop-rule exits.
- `perf stat` is unavailable in containers/macOS: the `$(perf ...)` command
  substitution injects its error text into the prompt, harmlessly. With perf
  present, real cache-miss counters ride along in the iteration prompt.

With a real `pi`: proposed changes sometimes regress (the measurement, not the
model, is the judge — `best` tracking and the plateau counter absorb this).
Note this loop keeps no champion copy: recovering the best-seen version is
exactly what Problems 9-10 add.
