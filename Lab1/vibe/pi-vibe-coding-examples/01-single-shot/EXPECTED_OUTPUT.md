# Expected output — Problem 1 — Single-shot utility script

One `pi -p` call writes `slurm_gpu_hours.py`; the harness then runs it twice —
no argument (usage line, exit 1) and on `data/sacct.csv`. The table is exact
and worth checking by hand: carol = 12:15:30×8 + 2-06:00:00×4 = 98.07 + 216 =
314.07; bob = 24h×2 + 3:45×1 = 51.75; alice's second job has no `gres/gpu`
(counts 0); dave's TRES field is empty. `gres/gpu:a100=1` (typed gres) parses.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text
[pi] wrote slurm_gpu_hours.py — stdlib csv+re; usage line on no arg; gres/gpu:type=N handled
[pi] try: python slurm_gpu_hours.py <sacct.csv>

$ python slurm_gpu_hours.py
usage: python slurm_gpu_hours.py SACCT_CSV

$ python slurm_gpu_hours.py data/sacct.csv
User    GPU-hours
-----  ----------
carol      314.07
bob         51.75
alice        8.00
dave         0.00
```

## What varies between runs/machines

- Nothing — this problem is fully deterministic given `data/sacct.csv`.

With a real `pi`: the generated file's style and table formatting will differ
run to run; the four GPU-hour totals must not. That is the lesson — the tight
spec pins the behavior, and "the entire loop here is you reading the file."
