# Expected output — Problem 9 — Tournament (best-of-N with a fitness function)

Eight independent candidates (forked sessions in a real run = cheap diversity;
the mock scripts 8 distinct implementations, three of them subtly wrong), then
ONE shared fitness function: `score.py` prints 0 if the output disagrees with
`np.convolve` at 1e-5, else effective GFLOP/s. Captured scoreboard:

- cand_2 (FFT, no padding → circular convolution): 0 — wrong
- cand_4 (float32 FFT — fast but misses the 1e-5 tolerance): 0 — wrong
- cand_8 (pow2 FFT, off-by-one trim): 0 — wrong
- cand_1 (direct np.convolve): ~19 · cand_6 (correlate+flip): ~15 ·
  cand_7 (exact-size mixed-radix rfft): ~8 · cand_3 (pow2-padded rfft): ~10
- cand_5 (overlap-add blocked rfft): ~36-42 → CHAMPION, copied to `conv.py`

Variance is a feature: nobody had to know in advance that overlap-add would
win on this size — the fitness function found out.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text
[pi] wrote conv.py — rfft with pow2 zero-padding, trimmed to n+k-1
[pi] wrote conv.py — FFT without padding — circular convolution (WRONG)
[pi] wrote conv.py — direct np.convolve (correct, no FFT)
[pi] wrote conv.py — np.correlate with a flipped kernel (correct, no FFT)
[pi] wrote conv.py — float32 rfft for speed (fails the 1e-5 tolerance)
[pi] wrote conv.py — rfft at exact size n+k-1 (mixed-radix, no pad waste)
[pi] wrote conv.py — overlap-add: blocked rfft, block 8192
[pi] wrote conv.py — pow2 FFT but off-by-one trim (WRONG length)
arena/cand_1 → 19.18
arena/cand_2 → 0
arena/cand_3 → 12.02
arena/cand_4 → 0
arena/cand_5 → 42.38
arena/cand_6 → 16.59
arena/cand_7 → 9.76
arena/cand_8 → 0
CHAMPION: arena/cand_5 @ 42.38 GFLOP/s
```

## What varies between runs/machines

- All GFLOP/s figures, and the champion can flip between the correct FFT
  variants (cand_3/5/7) by machine and FFT build. The three WRONG candidates
  score exactly 0 everywhere — correctness gates fitness.
- Worker `[pi] wrote ...` lines interleave (parallel fan-out).

With a real `pi`: candidates come from forked sessions with varied
temperature/model, so quality spread is organic; expect some of 8 to be wrong
or slow — that's why you score all of them with the SAME harness and crown
max fitness. `score.py` here runs n=2^18 so the arena scores in seconds;
scale N up on real hardware.
