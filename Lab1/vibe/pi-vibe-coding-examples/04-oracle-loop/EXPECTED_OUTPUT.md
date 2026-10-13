# Expected output — Problem 4 — Differential / oracle-driven loop

No test suite exists, so `check.py` (written inline by the harness) IS the
manufactured one: 1000 random tensors diffed against the naive log-sum-exp
reference at 1e-6. The seeded `fast.py` is vectorized-but-naive (float32, no
max-shift): the very first large-logit batch overflows `exp` → inf/inf → NaN →
`MISMATCH`, exit 1. Those numpy `RuntimeWarning` lines are the diagnostic
breadcrumbs the loop pipes into the fix prompt (`tail -n 5`). One fix later
(max-shift, float64), all 1000 tensors match: `OK: matches oracle`.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text
/home/claude/pi-vibe-coding-examples/04-oracle-loop/fast.py:11: RuntimeWarning: overflow encountered in exp
  e = np.exp(x)
/usr/local/lib/python3.11/dist-packages/numpy/_core/_methods.py:49: RuntimeWarning: overflow encountered in reduce
  return umr_sum(a, axis, dtype, out, keepdims, initial, where)
/home/claude/pi-vibe-coding-examples/04-oracle-loop/fast.py:12: RuntimeWarning: invalid value encountered in divide
  return e / e.sum(axis=-1, keepdims=True)
MISMATCH
[pi] wrote fast.py — max-shift before exp, float64 accumulation
[pi] diagnosis: float32 exp() overflows to inf for logits > ~88, giving inf/inf = NaN.
[pi] fix: subtract the row max (exp args <= 0) and keep float64. reference.py untouched.
OK
OK: matches oracle
```

## What varies between runs/machines

- Only the RuntimeWarning file paths.

With a real `pi`: the fix wording varies (some runs keep float32 with a
max-shift — that passes 1e-6 only marginally; the mock's float64 fix is the
robust version). The oracle is the spec: `reference.py` stays untouched, and
the loop is gated on `check.py`'s exit code.
