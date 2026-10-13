#!/usr/bin/env bash
# Problem 9 — Tournament (best-of-N with a fitness function)
# Teaches: population search. Generate many INDEPENDENT candidates (forked
# sessions = cheap diversity), score them all with the SAME harness, keep the
# champion. Variance is a feature, not a bug.
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
rm -rf arena conv.py .pi_mock_state.json __pycache__

# --- BEGIN VERBATIM (PDF p.5) ------------------------------------------------
mkdir -p arena
for n in $(seq 1 8); do
( cd arena && mkdir -p cand_$n && cd cand_$n
pi -p "Implement conv1d_fft(signal, kernel) in conv.py. Optimize for speed
on length-1e6 inputs. It must match numpy.convolve within 1e-5." ) &
done
wait
# SCORE every candidate with the SAME harness; champion = max fitness
best=""; bestf=0
for d in arena/cand_*; do
f=$(python score.py "$d/conv.py")
# prints 0 if wrong, else GFLOP/s
echo "$d → $f"
awk "BEGIN{exit !($f > $bestf)}" && { bestf=$f; best=$d; }
done
echo "CHAMPION: $best @ $bestf GFLOP/s"
cp "$best/conv.py" conv.py
# --- END VERBATIM ------------------------------------------------------------
