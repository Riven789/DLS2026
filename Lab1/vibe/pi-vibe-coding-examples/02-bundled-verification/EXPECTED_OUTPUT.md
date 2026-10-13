# Expected output — Problem 2 — Generate with its own verification

The agent writes `roofline.py` AND `test_roofline.py`, runs pytest itself, and
only then reports done ("If red, fix and rerun before finishing"). The harness
re-runs pytest afterward — never take the agent's word for green. Both runs
show `4 passed`.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text
[pi] wrote roofline.py — AI + attainable GFLOP/s (roofline min)
[pi] wrote test_roofline.py — SAXPY AI, GEMM AI, ridge point, div-by-zero
[pi] running pytest -q myself before reporting done:
....                                                                     [100%]
4 passed in 0.02s
[pi] done — pytest is green

$ python -m pytest -q
....                                                                     [100%]
4 passed in 0.01s
```

## What varies between runs/machines

- pytest timing (`0.02s`) only.

With a real `pi`: test names/count and the printed transcript vary; the
contract is invariant — the artifact ships with its own oracle, and the agent
must show green before claiming done. This in-turn check is the seed that
Problem 3 externalizes into a shell loop.
