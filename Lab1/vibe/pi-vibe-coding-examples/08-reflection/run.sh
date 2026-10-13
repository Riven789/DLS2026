#!/usr/bin/env bash
# Problem 8 — Reflection / self-improvement loop
# Teaches: the loop edits itself. On failure it extracts a LESSON into
# ~/.pi/agent/memory.md, and when a bug class recurs it promotes the lesson
# into a skill. Attempt k+1 is primed with the distilled lessons of 1..k.
#
# Self-containment note: the mock honors PI_MOCK_HOME so this demo writes its
# "~/.pi" under ./fakehome instead of your real home. A real pi writes the
# real ~/.pi/agent/memory.md — which is exactly the point.
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
export PI_MOCK_HOME="$PWD/fakehome"
rm -rf fakehome __pycache__ .pi_mock_state.json
cp seeds/integrator_v0.py integrator.py   # flaky seed #1: Euler + float32

echo "=== incident 1: flaky integrator (forward Euler, float32) ==="
# --- BEGIN VERBATIM (PDF p.5) ------------------------------------------------
for i in $(seq 1 8); do
python converge_check.py && { echo "OK: converged"; break; }
pi -p "integrator.py fails the convergence check (see error below).
1) Fix it.
2) Append the ROOT-CAUSE LESSON (one line, e.g. 'use Kahan summation for the
accumulator') to ~/.pi/agent/memory.md so future runs start smarter.
3) If this class of bug recurs, promote the lesson into a skill at
~/.pi/agent/skills/numerical-stability/SKILL.md.
$(python converge_check.py 2>&1 | tail -n 15)"
done
# --- END VERBATIM ------------------------------------------------------------

echo
echo "=== incident 2: the bug class RECURS (RK4 but float32) — same loop ==="
cp seeds/integrator_v1.py integrator.py
for i in $(seq 1 8); do
python converge_check.py && { echo "OK: converged"; break; }
pi -p "integrator.py fails the convergence check (see error below).
1) Fix it.
2) Append the ROOT-CAUSE LESSON (one line, e.g. 'use Kahan summation for the
accumulator') to ~/.pi/agent/memory.md so future runs start smarter.
3) If this class of bug recurs, promote the lesson into a skill at
~/.pi/agent/skills/numerical-stability/SKILL.md.
$(python converge_check.py 2>&1 | tail -n 15)"
done

echo
echo '--- ~/.pi/agent/memory.md (persistent lessons) ---'
cat fakehome/.pi/agent/memory.md
echo
echo '--- ~/.pi/agent/skills/numerical-stability/SKILL.md (promoted skill) ---'
cat fakehome/.pi/agent/skills/numerical-stability/SKILL.md
