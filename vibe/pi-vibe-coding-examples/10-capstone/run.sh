#!/usr/bin/env bash
# Problem 10 — Capstone: Tournament + Meta multi-agent (the AlphaEvolve shape)
# Rounds of {mutate → critique → benchmark → select → reflect}, composed
# entirely from parts introduced in Problems 5-9. No human writes the final
# kernel — the system discovers it.
#
# Transcription note: the PDF's first code line ran the trailing '---' of the
# comment into 'seed=...'; split back into two lines here. Everything else is
# verbatim (including `$(cat gen/$round/$k 2>/dev/null)`, which cats a
# directory and injects nothing — the mock, like a real agent, reads
# STRATEGY.md itself).
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
rm -rf gen champion.py .pi_mock_state.json __pycache__
cp seeds/STRATEGY.md STRATEGY.md

# --- BEGIN VERBATIM (PDF p.6; first line split, see note above) --------------
# --- meta loop: rounds of {mutate → critique → benchmark → select → reflect} ---
seed=spmv_seed.py; cp $seed champion.py; bestf=$(python score.py champion.py)
echo "round 0 champion fitness: $bestf"
for round in $(seq 1 6); do
# 1) TOURNAMENT: spawn K mutated workers off the current champion (fresh contexts)
for k in $(seq 1 6); do
( mkdir -p gen/$round/$k
pi -p "Mutate champion.py (CSR SpMV) using ONE idea from STRATEGY.md:
$(cat gen/$round/$k 2>/dev/null). Apply a single optimization (blocking,
SIMD, format change, prefetch) → gen/$round/$k/spmv.py. Keep it correct
vs check.py." ) &
done; wait
# 2) CRITIC ENSEMBLE: cheap fresh-eyes pass drops obviously-broken candidates
for c in gen/$round/*/spmv.py; do
pi -p --fork-fresh "Sanity-review $c for correctness + UB. Write KEEP or DROP
to ${c%.py}.verdict on line 1."
done
# 3) BENCHMARK → fitness on survivors; 4) SELECT champion of the round
for c in gen/$round/*/spmv.py; do
grep -q '^KEEP' "${c%.py}.verdict" || continue
f=$(python score.py "$c"); echo "  $c → $f"
awk "BEGIN{exit !($f > $bestf)}" && { bestf=$f; cp "$c" champion.py; }
done
# 5) REFLECT: champion rewrites the mutation strategy for the NEXT round
pi -p "Current champion fitness ${bestf} GFLOP/s. Review which mutation ideas
this round helped vs hurt (gen/$round/*) and REWRITE STRATEGY.md with the most
promising directions to try next. Be specific to this sparsity pattern."
echo "round $round champion fitness: $bestf"
# 6) STOP: plateau handled by comparing bestf across rounds (omitted for brevity)
done
echo "EVOLVED champion → champion.py @ $bestf GFLOP/s"
# --- END VERBATIM ------------------------------------------------------------
