# DLS-2026 Tutorials

Hands-on, build-it-from-scratch tutorials for the core deep-learning architectures.
Each tutorial is a **20-slide deck plus the code that produced every number and figure
on the slides** — a NumPy implementation (forward + hand-written backprop with gradient
checks), the equivalent PyTorch model, a training harness, worked hand-calculations, and
a test suite. Everything runs on CPU; no GPU or dataset download required.

## Start here

New to these? Read the matching PDF (or open the `.pptx` for the narrated speaker notes),
then open the tutorial's own `README.md` for setup, data contracts, and run commands.
A good learning order:

**MLP → CNN → RNN/LSTM → GNN → Transformer (decoder-only) → einsum architectures**

## What's here

### Standalone tutorials (source + deck)

| Dir | Topic | Highlights |
|---|---|---|
| [`mlp-tutorial/`](mlp-tutorial/) | **Multilayer Perceptrons** | NumPy backprop + gradient check, PyTorch twin, Titanic (classification) & Auto-MPG (regression); 6 tests (backprop vs finite-diff, NumPy vs PyTorch, leakage, XOR, overfit) |
| [`cnn-tutorial/`](cnn-tutorial/) | **Convolutional Neural Networks** | convolution/pooling from scratch in NumPy, LeNet-style PyTorch CNN on Fashion-MNIST; 7 tests incl. equivariance |
| [`lstm-tutorial/`](lstm-tutorial/) | **Recurrent Nets / LSTM** | LSTM + BPTT by hand, PyTorch RNN/LSTM/GRU, adding problem, sunspots, char-level TinyShakespeare; 8 tests |
| [`gnn-tutorial/`](gnn-tutorial/) | **Graph Neural Networks** | GCN from scratch (no graph library), plus GraphSAGE/GAT/GIN in PyTorch; karate club & Cora; 7 tests |
| [`einsum-architectures/`](einsum-architectures/) | **Einsum & architecture patterns** | `make` the Beamer deck (`deck.pdf`), with `SHA256SUMS` + `verification.txt` for reproducibility |

Each standalone dir contains: `slides/` (PDF + Beamer/TikZ source), `*_numpy.py`,
the PyTorch model, `data.py`, `train.py`, `experiments.py`, `walkthrough.py`,
`test_*.py`, and a `data/` folder with the bundled datasets.

Typical run (from inside a tutorial dir):

```bash
python test_mlp.py          # run the checks
python walkthrough.py       # the hand calculations from the slides
python train.py --task titanic
```

### Bundle: CNN + GNN + LSTM

| Item | What it is |
|---|---|
| [`CNN_GNN_LSTM_Complete_Tutorials/`](CNN_GNN_LSTM_Complete_Tutorials/) | A self-contained package of the CNN, GNN, and LSTM tutorials — each with PowerPoint, PDF, and a `*_lab/` with a `run_all.py` one-shot runner, `requirements.txt`, and `reference_runs/` (pre-trained models + measured results). See `START_HERE.txt`. |
| `CNN_GNN_LSTM_Complete_Tutorials.zip` | Zipped copy of the above (for easy download). |

The bundle labs are synthetic-data / self-contained: `python -m pip install -r
requirements.txt && python run_all.py` tests the implementation, trains
classification + regression models, and predicts on held-out inputs.

### MLP (expanded materials)

| Item | What it is |
|---|---|
| `MLP_Neural_Networks_Tutorial.pdf` / `.pptx` | The standalone MLP deck (PDF + narrated slides). |
| `MLP_Python_Training_Lab.zip` | The MLP hands-on training lab. |

### Transformer (decoder-only)

| Item | What it is |
|---|---|
| `transformer_tutorial.pdf` / `transformer-tutorial.zip` | The transformer tutorial deck + code. |
| `Decoder_Only_Transformer_Tutorial.pptx` | Narrated decoder-only transformer slides. |
| `Decoder_Only_Transformer_Code.zip` | The accompanying decoder-only transformer code. |

## Requirements

Python 3.10+, NumPy, and PyTorch (matplotlib only for the `experiments.py` figure
regeneration). The standalone tutorials list exact needs in their own `README.md`;
the bundle labs ship a `requirements.txt`.

## A note on provenance

Bundled datasets are small, redistributable research sets: Titanic & Auto-MPG
(seaborn-data), Fashion-MNIST (first 12k train + full 10k test), Cora (LINQS, via
tkipf/pygcn), monthly sunspots (SILSO via jbrownlee/Datasets), and TinyShakespeare
(Karpathy's char-rnn). All diagrams in the decks are TikZ/pgfplots *source*, not images.
