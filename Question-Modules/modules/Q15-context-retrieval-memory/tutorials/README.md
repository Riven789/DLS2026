# Architecture tutorials — Q15-context-retrieval-memory

Self-contained, runnable from-scratch tutorials (slides + code + tests) added to this module.

## `lstm-from-scratch/`
LSTM/RNN from scratch — NumPy BPTT with gradient checks, PyTorch LSTM, char-level Shakespeare + sunspot forecasting + the adding problem. Tests: all pass.

Slides: `lstm-from-scratch/slides/` (PDF + LaTeX source). Code: implementation, training script, and test suite in `lstm-from-scratch/`.

## Running the code

These labs use **NumPy + PyTorch**. A modern PyTorch (>= 2.0) is recommended — a
couple of labs (GNN, decoder-only) use APIs added after torch 1.11. On m1 the
`nougat-ocr` conda env (torch 2.3) runs everything; any recent `pip install torch numpy` works.

```
python test_*.py     # correctness tests (gradient checks, parity vs PyTorch)
python train.py      # or: python walkthrough.py  — see the lab's own README
```

Model checkpoints and large datasets are intentionally **not committed**; each
lab's `README.md` says where to fetch its data (one-time download).
