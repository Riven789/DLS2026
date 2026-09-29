#!/usr/bin/env bash
# Problem 7 — Plan → decompose → sub-agent fan-out
# Teaches: an orchestrator that splits work, spawns workers under INDEPENDENT
# context (each in its own dir), and integrates. Parallelism + context
# isolation, orchestrated at the shell layer — honest and portable.
#
# Note: if torch isn't installed, the torch worker writes an adapter that
# falls back to numpy and FLAGS the fallback — the integrator's table shows it.
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
rm -rf work PLAN.md bench.py .pi_mock_state.json __pycache__ .pytest_cache

# --- BEGIN VERBATIM (PDF p.4) ------------------------------------------------
# PLAN
pi -p 'Write PLAN.md decomposing a GEMM benchmark harness into independent
subtasks: one adapter per backend (numpy, torch, custom) with a common
interface gemm(A,B)→C, plus a runner. List each subtask + its acceptance test.'
# FAN-OUT: one isolated worker per backend, in parallel, each in its own dir/context
for be in numpy torch custom; do
( mkdir -p work/$be && cd work/$be
pi -p "Per ../../PLAN.md, implement ONLY the $be adapter to the shared
interface and a unit test proving it matches numpy on a 256x256 GEMM." ) &
done
wait
# INTEGRATE
pi -p 'All three adapters exist under work/*/. Wire them into bench.py behind the
common interface, run all unit tests, then produce a timing table. Fix integration
gaps only.'
# --- END VERBATIM ------------------------------------------------------------
