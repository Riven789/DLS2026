# Architecture tutorials — Q30-pytorch-compiler-autodiff

Self-contained, runnable from-scratch tutorials (slides + code + tests) added to this module.

## `mlp-from-scratch/`
Multilayer perceptron from scratch — NumPy forward/backprop with finite-difference gradient checks, a matching PyTorch implementation, and training on real tabular tasks (moons, MPG, Titanic). Tests: all pass.

Slides: `mlp-from-scratch/slides/` (PDF + LaTeX source). Code: implementation, training script, and test suite in `mlp-from-scratch/`.

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
