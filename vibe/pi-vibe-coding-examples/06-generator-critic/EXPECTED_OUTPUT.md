# Expected output — Problem 6 — Generator + Critic (fresh eyes)

Round 1: the generator writes the natural first version — threads accumulating
into a shared global (`TOTAL += s`), tail elements silently dropped. The
critic runs in a FRESH context (`--fork-fresh`: no author bias) and files four
findings to `CRITIQUE.md`. Round 2: the generator addresses the checklist
(per-thread slots, `array_split` tail coverage, fixed-order `fsum` combine);
the critic finds nothing real and writes `NO_ISSUES` on line 1 — the machine-
checkable gate (`head -n1 | grep`) ends the loop: `OK: critic satisfied`.

The epilogue diffs the survivor against `math.fsum` on 1,000,003 elements
(deliberately not divisible by 4 threads): `|diff| = 0.000e+00`.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text
[pi] wrote reduce.py — threaded parallel sum over chunks
[pi] wrote CRITIQUE.md — 4 findings
[pi] fresh-context review: found a shared-accumulator race, a dropped tail, and non-deterministic FP ordering — see CRITIQUE.md
[pi] wrote reduce.py — addressed CRITIQUE.md: per-thread partials, array_split tail, fixed-order fsum
[pi] wrote CRITIQUE.md — zero real issues found
[pi] fresh-context review: no data race, tail covered, deterministic order — NO_ISSUES
OK: critic satisfied

$ python — quick numeric check of the final reduce.py
parallel_sum = 1000.5238657165388
math.fsum    = 1000.5238657165388
|diff|       = 0.000e+00
```

## What varies between runs/machines

- Nothing in the mock run; the epilogue diff can be a few ULPs (~1e-13) on
  other BLAS builds — anything at that scale is fine.

With a real `pi`: rounds-to-clean varies (usually 2-3); occasionally the
critic nitpicks a real-but-minor point for an extra round. Same model, two
roles, two context states — the separation is what finds the race the author
just wrote past.
