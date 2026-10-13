# Architecture tutorials — Q21-composed-systems

Self-contained, runnable from-scratch tutorials (slides + code + tests) added to this module.

## `gnn-from-scratch/`
Graph convolutional network from scratch — NumPy GCN + PyTorch GNN on the Cora citation graph (download step in README). Core tests (equivariance, batching, attention) pass; node-classification tests need the Cora data.

Slides: `gnn-from-scratch/slides/` (PDF + LaTeX source). Code: implementation, training script, and test suite in `gnn-from-scratch/`.

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
