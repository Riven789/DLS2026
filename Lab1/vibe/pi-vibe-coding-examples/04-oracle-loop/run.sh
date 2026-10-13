#!/usr/bin/env bash
# Problem 4 — Differential / oracle-driven loop
# Teaches: when there's no test suite, manufacture one from a slow-but-correct
# reference. The oracle is the spec. You go fast; the slow version keeps you
# honest.
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
rm -f check.py .pi_mock_state.json
rm -rf __pycache__
cp seeds/fast_v0.py fast.py               # start from the unstable fast version

# --- BEGIN VERBATIM (PDF pp.2-3; python indentation restored) ----------------
# verify: a harness that diffs fast vs reference and exits nonzero on mismatch
cat > check.py <<'PY'
import numpy as np, sys
from reference import reference_softmax
from fast import fast_softmax
rng = np.random.default_rng(0)
for _ in range(1000):
    x = rng.standard_normal((rng.integers(1,64), rng.integers(1,512))) * rng.integers(1,50)
    if not np.allclose(fast_softmax(x), reference_softmax(x), atol=1e-6):
        print("MISMATCH"); sys.exit(1)
print("OK")
PY
for i in $(seq 1 6); do
python check.py && { echo "OK: matches oracle"; break; }
pi -p "fast.py disagrees with the reference oracle. Fix fast_softmax for
numerical stability (think overflow on large logits). Don't touch reference.py.
$(python check.py 2>&1 | tail -n 5)"
done
# --- END VERBATIM ------------------------------------------------------------
