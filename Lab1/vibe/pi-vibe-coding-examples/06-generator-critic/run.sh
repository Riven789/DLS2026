#!/usr/bin/env bash
# Problem 6 — Generator + Critic (role separation, fresh eyes)
# Teaches: split making from judging. A critic with a CLEAN context window
# catches what the author is blind to. Same model, different role + different
# state — that's why it works. (--fork-fresh is shown conceptually in the PDF;
# in practice fork the session or run the critic with a stripped context.)
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
rm -f reduce.py CRITIQUE.md .pi_mock_state.json
rm -rf __pycache__

# --- BEGIN VERBATIM (PDF p.4) ------------------------------------------------
for i in $(seq 1 5); do
# GENERATOR
pi -p "Implement/improve reduce.py: a parallel sum over a large array.
Address any open items in CRITIQUE.md if it exists."
# CRITIC -- fresh session (no author bias), structured output, gates the loop
pi -p --fork-fresh "You are a hostile reviewer. Inspect reduce.py ONLY for:
data races, non-deterministic FP ordering, false sharing, off-by-one on the
tail. Write findings to CRITIQUE.md as a checklist. If and only if there are
zero real issues, write exactly NO_ISSUES on the first line."
head -n1 CRITIQUE.md | grep -q NO_ISSUES && { echo "OK: critic satisfied"; break; }
done
# --- END VERBATIM ------------------------------------------------------------

# Sanity-run the survivor against math.fsum (not part of the PDF snippet):
echo
echo '$ python — quick numeric check of the final reduce.py'
python - <<'EOF'
import math
import numpy as np
from reduce import parallel_sum
rng = np.random.default_rng(0)
a = rng.standard_normal(1_000_003)          # deliberately not divisible by 4
got = parallel_sum(a, nthreads=4)
exact = math.fsum(a.tolist())
print(f"parallel_sum = {got!r}")
print(f"math.fsum    = {exact!r}")
print(f"|diff|       = {abs(got - exact):.3e}")
EOF
