#!/usr/bin/env bash
# Problem 3 — Iterate-until-green (the canonical loop)
# Teaches: the verification-gated loop. Tests are the stop rule; the shell is
# the loop runner. The verifier is pytest's EXIT CODE, not the model's opinion.
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
rm -rf src .pi_mock_state.json __pycache__ .pytest_cache
mkdir -p src

# --- BEGIN VERBATIM (PDF p.2) ------------------------------------------------
# objective: pass tests | state: src/ + pytest output | verify: pytest exit code
# stop: green OR 8 attempts
for i in $(seq 1 8); do
if pytest -q; then echo "OK: green in $i"; break; fi
pi -p "pytest is failing. Read src/stencil.py and the failures below, make the
smallest fix, do not edit the tests. Failures:
$(pytest -q 2>&1 | tail -n 40)"
done
# --- END VERBATIM ------------------------------------------------------------
