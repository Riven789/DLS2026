#!/usr/bin/env bash
# Problem 5 — Tool-augmented loop (real-world feedback signal)
# Teaches: verification doesn't have to be pass/fail — it can be a MEASUREMENT.
# The environment is the grader; plateau is the first real stop rule.
#
# Note: `perf stat` needs Linux perf + counter access. Where it's missing
# (containers, macOS), the $(perf ...) substitution just injects the error
# text into the prompt — the loop's real signal is bench.py's number either way.
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
rm -f .pi_mock_state.json
rm -rf __pycache__
cp seeds/kernel_v1.py kernel.py           # start from the slow seed kernel

echo '$ python check.py    # correctness gate before optimizing'
python check.py
echo

# --- BEGIN VERBATIM (PDF p.3) ------------------------------------------------
best=0
stale=0
for i in $(seq 1 12); do
gflops=$(python bench.py --report gflops)
# harness prints a number
echo "round $i: ${gflops} GFLOP/s (best ${best})"
awk "BEGIN{exit !($gflops > 28)}" && { echo "HIT: target reached"; break; }
awk "BEGIN{exit !($gflops > $best)}" && { best=$gflops; stale=0; } || stale=$((stale+1))
[ $stale -ge 3 ] && { echo "PLATEAU"; break; }
pi -p "Current kernel does ${gflops} GFLOP/s. Here is perf stat (cache-misses,
bandwidth) and the hot loop. Propose ONE change (blocking, vectorization, layout)
and apply it. Keep correctness vs check.py.
$(perf stat -e cache-misses,cache-references python bench.py 2>&1 | tail -n 20)"
done
# --- END VERBATIM ------------------------------------------------------------
