# Expected output — Problem 8 — Reflection / self-improvement loop

Two incidents of the same bug CLASS, so the self-improvement machinery has
something to do:

- Incident 1 (Euler + float32): rel_err 5.5e-5 — four orders over tolerance.
  Fix → RK4/float64 (rel_err 1.4e-14) AND a one-line root-cause LESSON
  appended to `~/.pi/agent/memory.md`.
- Incident 2 (RK4 but float32 — right order, wrong dtype): rel_err 7.6e-6,
  a subtle 7.6× miss. Fix, second LESSON appended — and because the class
  recurred, the lessons get PROMOTED into a reusable skill:
  `~/.pi/agent/skills/numerical-stability/SKILL.md` (printed at the end).

Attempt k+1 starts primed with the distilled lessons of attempts 1..k — the
loop improves the loop.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text
=== incident 1: flaky integrator (forward Euler, float32) ===
y(5) = 0.0067375777289271355   exact = 0.006737946999085467   rel_err = 5.480e-05   tol = 1e-06
FAIL: not converged
[pi] root cause: first-order method (forward Euler) + float32 accumulation.
[pi] wrote integrator.py — RK4, float64, t = t0 + i*dt
[pi] appended root-cause lesson to ~/.pi/agent/memory.md
y(5) = 0.006737946999085371   exact = 0.006737946999085467   rel_err = 1.429e-14   tol = 1e-06
PASS: converged
OK: converged

=== incident 2: the bug class RECURS (RK4 but float32) — same loop ===
y(5) = 0.006737998221069574   exact = 0.006737946999085467   rel_err = 7.602e-06   tol = 1e-06
FAIL: not converged
[pi] root cause: float32 state — truncation order was fine, rounding was not.
[pi] wrote integrator.py — RK4, float64, t = t0 + i*dt
[pi] appended root-cause lesson to ~/.pi/agent/memory.md
[pi] this class of bug has recurred — promoted the lessons to a skill: ~/.pi/agent/skills/numerical-stability/SKILL.md
y(5) = 0.006737946999085371   exact = 0.006737946999085467   rel_err = 1.429e-14   tol = 1e-06
PASS: converged
OK: converged

--- ~/.pi/agent/memory.md (persistent lessons) ---
# pi agent memory
LESSON: forward Euler is O(dt) — cannot reach 1e-6 rel. error at sane step counts; use RK4 (O(dt^4)), float64 state, and t = t0 + i*dt (never accumulate t by repeated addition).
LESSON: float32 state over ~2e5 steps loses ~4-5 digits to rounding (error random-walks ~ sqrt(n)*eps); keep integrator state in float64.

--- ~/.pi/agent/skills/numerical-stability/SKILL.md (promoted skill) ---
---
name: numerical-stability
description: Checklist for convergence/accuracy bugs in numerical kernels —
  read before debugging any integrator, reduction, or iterative solver.
---

# Numerical stability

Recurring root causes, distilled from failed convergence runs:

1. Method order first: forward Euler error is O(dt) — it will NOT reach 1e-6
   at feasible step counts. Reach for RK4 (O(dt^4)) or better.
2. float32 state over >1e4 steps loses ~4-5 digits (rounding random-walk
   ~ sqrt(n) * eps). Keep integrator/reduction state in float64.
3. Never accumulate t by repeated addition; compute t = t0 + i*dt.
4. For long sums, use compensated (Kahan) or pairwise summation.
```

## What varies between runs/machines

- rel_err digits only (float32 rounding noise).
- The mock honors `PI_MOCK_HOME`, so "~/.pi" lands in `./fakehome/` here. A
  real `pi` writes your actual `~/.pi/agent/memory.md` — which is the point:
  the lessons persist across every future run, and the skill loads whenever a
  numerical-stability task appears.

With a real `pi`: lesson wording varies; the durable contract is the
mechanism — fix, distill, persist, promote on recurrence.
