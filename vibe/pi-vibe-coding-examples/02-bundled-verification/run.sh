#!/usr/bin/env bash
# Problem 2 — Generate with its own verification (one shot)
# Teaches: bundle the oracle with the artifact. Never ship code without the
# thing that checks it. The agent runs its own check inside a single turn —
# the seed of the loop we externalize in Problem 3.
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
rm -f roofline.py test_roofline.py .pi_mock_state.json
rm -rf __pycache__ .pytest_cache

# --- BEGIN VERBATIM (PDF p.2) ------------------------------------------------
pi -p 'Write roofline.py with arithmetic_intensity(flops, bytes_moved) and
attainable_gflops(ai, peak_gflops, mem_bw_gbs). THEN write test_roofline.py
covering: AI of a SAXPY, AI of a GEMM, the ridge point where compute- and
memory-bound cross, and div-by-zero. Run pytest -q yourself and only report
done if it is green. If red, fix and rerun before finishing.'
# --- END VERBATIM ------------------------------------------------------------

# Don't take the agent's word for it — rerun the bundled oracle yourself:
echo
echo '$ python -m pytest -q'
python -m pytest -q
