# Q32 code — Quantifying information flow with entropy

**Author:** Kukla · **Module:** Q32 (AI Course 2026)

## What it does
`entropy_flow.py` computes the information-theoretic quantities that are *safe*
to measure on a language model, and then deliberately demonstrates the one that
is **not** safe (the binning artefact behind "information plane" plots):

1. Predictive (next-token) entropy — confidence / hedging read-out.
2. Corpus cross-entropy & perplexity, with the `CE = H(data) + KL(data‖model)`
   decomposition verified by assertion.
3. Per-head attention entropy — exact, assumption-free internal read-out.
4. **Cautionary demo:** an "information" estimate of a *fixed* continuous
   activation that nearly doubles when you change only the number of histogram
   bins — i.e. the estimator is the measurement (Saxe et al. 2018).

The distributions are simulated so the file is self-contained and
deterministic (seed 42). The math is identical on real logits: replace
`softmax(logits)` with your model's output distribution and nothing changes.

## Run
```bash
python3 entropy_flow.py
```

## Requirements
- Python 3.8+
- numpy >= 1.20

CPU-only. No GPU, no model download, no network. Runtime < 1 s.
