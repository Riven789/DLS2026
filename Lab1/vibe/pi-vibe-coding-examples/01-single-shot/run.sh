#!/usr/bin/env bash
# Problem 1 — Single-shot utility script
# Teaches: a tight spec is the program. Precision in, artifact out.
# "The entire loop here is you reading the file."
cd "$(dirname "$0")"
export PATH="$(cd .. && pwd)/bin:$PATH"   # mock pi (delete this line for a real pi)
rm -f slurm_gpu_hours.py .pi_mock_state.json

# --- BEGIN VERBATIM (PDF p.1) ------------------------------------------------
pi -p 'Write slurm_gpu_hours.py. Input: a sacct CSV with columns
JobID,User,Elapsed,AllocTRES (TRES contains gres/gpu=N). Output: a table of
User → total GPU-hours, sorted descending. Parse Elapsed as [DD-]HH:MM:SS.
Handle missing gpu allocations as 0. stdlib only. Print a usage line if no arg.'
# --- END VERBATIM ------------------------------------------------------------

# The loop is you reading (and running) the artifact:
echo
echo '$ python slurm_gpu_hours.py'
python slurm_gpu_hours.py
echo
echo '$ python slurm_gpu_hours.py data/sacct.csv'
python slurm_gpu_hours.py data/sacct.csv
