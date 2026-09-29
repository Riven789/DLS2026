# Expected output — Problem 10 — Capstone: Tournament + Meta multi-agent

The AlphaEvolve shape, composed from every prior part: rounds of
{mutate (P7/P9) → critique (P6) → benchmark (P5) → select (P9) → reflect (P8)}.
Captured fitness trajectory (GFLOP/s on the fixed sparsity pattern):

- round 0 — seed (pure-Python CSR loops): ~0.004
- round 1 — vectorized gather, row loop kept: ~0.009
- round 2 — np.add.reduceat over segments: ~0.44 (the big jump: the row loop
  was the bottleneck on 8-nnz rows)
- round 4 — format change to scipy CSR C matvec, matrix cached: ~1.9-2.0
- rounds 5-6 — plateau (the PDF notes the plateau stop rule is elided)

≈450-500× over the seed, and no human wrote the final kernel: `champion.py`
is whatever survived selection.

Watch the critic ensemble earn its keep every round: `brokenseg` (uniform-
stride segmentation) fails fresh-eyes review outright, and `stale` (caches its
result across calls) is DROPPED on *trial 2* — it returns stale output for a
new x. The benchmark alone would have CROWNED that one, because `score.py`
times a fixed x. That is why critique runs before benchmarking.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text
round 0 champion fitness: 0.0045
[pi] wrote gen/1/1/spmv.py — single mutation: vectorized gather (data * x[indices]) with the row loop kept
[pi] wrote gen/1/5/spmv.py — single mutation: vectorized gather (data * x[indices]) with the row loop kept
[pi] wrote gen/1/4/spmv.py — single mutation: vectorized gather (data * x[indices]) with the row loop kept
[pi] wrote gen/1/3/spmv.py — single mutation: uniform-stride segmentation (BROKEN: rows have variable nnz)
[pi] wrote gen/1/6/spmv.py — single mutation: uniform-stride segmentation (BROKEN: rows have variable nnz)
[pi] wrote gen/1/2/spmv.py — single mutation: cached result reused across calls (BROKEN: stale for new x)
[pi] KEEP gen/1/1/spmv.py — matches the dense reference on two independent x vectors
[pi] DROP gen/1/2/spmv.py — trial 2: output disagrees with the dense reference
[pi] DROP gen/1/3/spmv.py — trial 1: output disagrees with the dense reference
[pi] KEEP gen/1/4/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/1/5/spmv.py — matches the dense reference on two independent x vectors
[pi] DROP gen/1/6/spmv.py — trial 1: output disagrees with the dense reference
  gen/1/1/spmv.py → 0.0093
  gen/1/4/spmv.py → 0.0093
  gen/1/5/spmv.py → 0.0086
[pi] wrote STRATEGY.md — rewrote mutation strategy for the next round
round 1 champion fitness: 0.0093
[pi] wrote gen/2/1/spmv.py — single mutation: np.add.reduceat over indptr segments — row loop eliminated
[pi] wrote gen/2/5/spmv.py — single mutation: np.add.reduceat over indptr segments — row loop eliminated
[pi] wrote gen/2/3/spmv.py — single mutation: cached result reused across calls (BROKEN: stale for new x)
[pi] wrote gen/2/2/spmv.py — single mutation: vectorized gather (data * x[indices]) with the row loop kept
[pi] wrote gen/2/4/spmv.py — single mutation: uniform-stride segmentation (BROKEN: rows have variable nnz)
[pi] wrote gen/2/6/spmv.py — single mutation: vectorized gather (data * x[indices]) with the row loop kept
[pi] KEEP gen/2/1/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/2/2/spmv.py — matches the dense reference on two independent x vectors
[pi] DROP gen/2/3/spmv.py — trial 2: output disagrees with the dense reference
[pi] DROP gen/2/4/spmv.py — trial 1: output disagrees with the dense reference
[pi] KEEP gen/2/5/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/2/6/spmv.py — matches the dense reference on two independent x vectors
  gen/2/1/spmv.py → 0.4454
  gen/2/2/spmv.py → 0.0091
  gen/2/5/spmv.py → 0.4350
  gen/2/6/spmv.py → 0.0091
[pi] wrote STRATEGY.md — rewrote mutation strategy for the next round
round 2 champion fitness: 0.4454
[pi] wrote gen/3/2/spmv.py — single mutation: np.add.reduceat over indptr segments — row loop eliminated
[pi] wrote gen/3/3/spmv.py — single mutation: uniform-stride segmentation (BROKEN: rows have variable nnz)
[pi] wrote gen/3/1/spmv.py — single mutation: scatter-add via np.bincount with precomputed row ids
[pi] wrote gen/3/5/spmv.py — single mutation: cached result reused across calls (BROKEN: stale for new x)
[pi] wrote gen/3/4/spmv.py — single mutation: row-blocked reduceat (locality blocking, block=1024)
[pi] wrote gen/3/6/spmv.py — single mutation: np.add.reduceat over indptr segments — row loop eliminated
[pi] KEEP gen/3/1/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/3/2/spmv.py — matches the dense reference on two independent x vectors
[pi] DROP gen/3/3/spmv.py — trial 1: output disagrees with the dense reference
[pi] KEEP gen/3/4/spmv.py — matches the dense reference on two independent x vectors
[pi] DROP gen/3/5/spmv.py — trial 2: output disagrees with the dense reference
[pi] KEEP gen/3/6/spmv.py — matches the dense reference on two independent x vectors
  gen/3/1/spmv.py → 0.3871
  gen/3/2/spmv.py → 0.3787
  gen/3/4/spmv.py → 0.3389
  gen/3/6/spmv.py → 0.3804
[pi] wrote STRATEGY.md — rewrote mutation strategy for the next round
round 3 champion fitness: 0.4454
[pi] wrote gen/4/2/spmv.py — single mutation: scatter-add via np.bincount with precomputed row ids
[pi] wrote gen/4/4/spmv.py — single mutation: np.add.reduceat over indptr segments — row loop eliminated
[pi] wrote gen/4/1/spmv.py — single mutation: format change: hand the pattern to scipy.sparse CSR (C matvec), matrix cached
[pi] wrote gen/4/5/spmv.py — single mutation: uniform-stride segmentation (BROKEN: rows have variable nnz)
[pi] wrote gen/4/6/spmv.py — single mutation: cached result reused across calls (BROKEN: stale for new x)
[pi] wrote gen/4/3/spmv.py — single mutation: row-blocked reduceat (locality blocking, block=1024)
[pi] KEEP gen/4/1/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/4/2/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/4/3/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/4/4/spmv.py — matches the dense reference on two independent x vectors
[pi] DROP gen/4/5/spmv.py — trial 1: output disagrees with the dense reference
[pi] DROP gen/4/6/spmv.py — trial 2: output disagrees with the dense reference
  gen/4/1/spmv.py → 1.9302
  gen/4/2/spmv.py → 0.4532
  gen/4/3/spmv.py → 0.3663
  gen/4/4/spmv.py → 0.4695
[pi] wrote STRATEGY.md — rewrote mutation strategy for the next round
round 4 champion fitness: 1.9302
[pi] wrote gen/5/1/spmv.py — single mutation: format change: hand the pattern to scipy.sparse CSR (C matvec), matrix cached
[pi] wrote gen/5/5/spmv.py — single mutation: cached result reused across calls (BROKEN: stale for new x)
[pi] wrote gen/5/2/spmv.py — single mutation: row-blocked reduceat (locality blocking, block=1024)
[pi] wrote gen/5/4/spmv.py — single mutation: format change: hand the pattern to scipy.sparse CSR (C matvec), matrix cached
[pi] wrote gen/5/6/spmv.py — single mutation: uniform-stride segmentation (BROKEN: rows have variable nnz)
[pi] wrote gen/5/3/spmv.py — single mutation: scatter-add via np.bincount with precomputed row ids
[pi] KEEP gen/5/1/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/5/2/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/5/3/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/5/4/spmv.py — matches the dense reference on two independent x vectors
[pi] DROP gen/5/5/spmv.py — trial 2: output disagrees with the dense reference
[pi] DROP gen/5/6/spmv.py — trial 1: output disagrees with the dense reference
  gen/5/1/spmv.py → 1.9210
  gen/5/2/spmv.py → 0.3944
  gen/5/3/spmv.py → 0.4642
  gen/5/4/spmv.py → 1.9697
[pi] wrote STRATEGY.md — rewrote mutation strategy for the next round
round 5 champion fitness: 1.9697
[pi] wrote gen/6/2/spmv.py — single mutation: format change: hand the pattern to scipy.sparse CSR (C matvec), matrix cached
[pi] wrote gen/6/4/spmv.py — single mutation: row-blocked reduceat (locality blocking, block=1024)
[pi] wrote gen/6/1/spmv.py — single mutation: scatter-add via np.bincount with precomputed row ids
[pi] wrote gen/6/5/spmv.py — single mutation: uniform-stride segmentation (BROKEN: rows have variable nnz)
[pi] wrote gen/6/3/spmv.py — single mutation: np.add.reduceat over indptr segments — row loop eliminated
[pi] wrote gen/6/6/spmv.py — single mutation: cached result reused across calls (BROKEN: stale for new x)
[pi] KEEP gen/6/1/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/6/2/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/6/3/spmv.py — matches the dense reference on two independent x vectors
[pi] KEEP gen/6/4/spmv.py — matches the dense reference on two independent x vectors
[pi] DROP gen/6/5/spmv.py — trial 1: output disagrees with the dense reference
[pi] DROP gen/6/6/spmv.py — trial 2: output disagrees with the dense reference
  gen/6/1/spmv.py → 0.4636
  gen/6/2/spmv.py → 1.7468
  gen/6/3/spmv.py → 0.4436
  gen/6/4/spmv.py → 0.3599
[pi] wrote STRATEGY.md — rewrote mutation strategy for the next round
round 6 champion fitness: 1.9697
EVOLVED champion → champion.py @ 1.9697 GFLOP/s
```

## What varies between runs/machines

- All fitness numbers; whether `bincount` edges out `reduceat` in the middle
  rounds flips run to run (they're within noise of each other); round-3/5
  champions sometimes don't improve — `awk` keeps the incumbent.
- Worker lines interleave within each round (parallel mutation fan-out).
- `STRATEGY.md` is rewritten by the reflect step each round — open it after a
  run to see the surviving directions.

With a real `pi`: mutations come from the model reading `STRATEGY.md` (the
verbatim `$(cat gen/$round/$k ...)` injects nothing — it cats a directory;
kept as in the PDF), so diversity and dead-ends are organic. Budget/plateau
stop rules and champion checkpointing are the knobs you add for long runs.
