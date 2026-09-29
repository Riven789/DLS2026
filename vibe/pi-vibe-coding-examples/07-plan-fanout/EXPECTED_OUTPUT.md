# Expected output — Problem 7 — Plan → decompose → sub-agent fan-out

Three phases: (1) planner writes `PLAN.md` (shared `gemm(A,B)` interface + an
acceptance test per subtask); (2) three workers run IN PARALLEL, each in its
own directory and context, each implementing only its backend + unit test and
running it; (3) the integrator discovers `work/*/adapter.py`, re-runs every
unit test, and prints the timing table.

In the table: numpy and torch hit BLAS speed (~50-60 GFLOP/s); `custom` is an
honest hand-written rank-1-update kernel (~1 GFLOP/s, still 1e-14-correct) —
a 50× gap that makes the "why BLAS exists" point for the room.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text
[pi] wrote PLAN.md — 4 subtasks, common gemm(A,B) interface, acceptance tests
[pi] wrote adapter.py — numpy backend behind the shared gemm(A,B) interface
[pi] wrote test_adapter.py — acceptance: matches numpy on 256x256
[pi] wrote adapter.py — torch backend behind the shared gemm(A,B) interface
[pi] wrote test_adapter.py — acceptance: matches numpy on 256x256
[pi] wrote adapter.py — custom backend behind the shared gemm(A,B) interface
[pi] wrote test_adapter.py — acceptance: matches numpy on 256x256
[pi] unit test (torch): 1 passed in 0.21s
[pi] unit test (numpy): 1 passed in 0.24s
[pi] unit test (custom): 1 passed in 0.25s
[pi] wrote bench.py — discovers work/*/adapter.py, unit-checks, prints timing table
[pi] unit tests custom : 1 passed in 0.14s
[pi] unit tests numpy  : 1 passed in 0.11s
[pi] unit tests torch  : 1 passed in 0.11s
[pi] all adapters green — timing table:
backend         ms   GFLOP/s   max|err| vs numpy 
custom      32.201      1.04   3.20e-14           
numpy        0.474     70.85   0.00e+00           
torch        0.425     78.95   0.00e+00           (numpy fallback: torch not installed)
```

## What varies between runs/machines

- The `[pi] wrote ...` lines from the three workers interleave differently
  every run — that's the `&`/`wait` parallelism showing.
- Timings/GFLOP/s; and this container has no torch, so the torch adapter
  declares a flagged numpy fallback — the table prints
  `(numpy fallback: torch not installed)`. With torch installed, the worker
  uses real torch and the note disappears.

With a real `pi`: workers only see their own directory plus `PLAN.md` — the
whole point is that no worker drowns in the others' context. The orchestration
is plain shell: honest, portable, inspectable.
