# Graph Neural Networks from the Ground Up

A 20-slide tutorial on graph neural networks with the code that produced every number and figure on the slides.
No graph library is used: the layers are written directly in NumPy and PyTorch.

## Files

| file | what it is |
|---|---|
| `slides/gnn_tutorial.pdf` | the deck |
| `slides/gnn_tutorial.tex` | Beamer + TikZ/pgfplots source (all diagrams are source, no images) |
| `data.py` | the three graphs (6-node example, Zachary's karate club, Cora) and the train/val/test node masks |
| `gcn_numpy.py` | normalised adjacency, GCN forward pass, masked loss, hand-written backpropagation and a gradient check in NumPy |
| `gnn.py` | GCN, GraphSAGE-mean, GAT and GIN layers in plain PyTorch with sparse matrices; batching and readout for graph-level tasks; label propagation |
| `train.py` | training and testing harness: full-batch Adam, masked loss, early stopping on validation nodes, test report with baselines |
| `experiments.py` | runs every experiment on the slides and writes `slides/generated/*` |
| `walkthrough.py` | the hand calculations on the slides |
| `test_gnn.py` | seven checks (backprop vs finite differences, NumPy vs PyTorch, permutation equivariance, batching, data and masks, no label leakage, can fit) |
| `data/` | `cora.content`, `cora.cites` from github.com/tkipf/pygcn (the original LINQS Cora files) |

## Run it

Requires Python 3.10+, NumPy, PyTorch. (networkx is used only as an optional cross-check in one test.)

```bash
python test_gnn.py
python walkthrough.py
python gcn_numpy.py                               # from-scratch GCN on the karate club
python train.py                                   # GCN on Cora, split 0
python train.py --model mlp                       # features only: the graph is ignored
python train.py --model gcn --no_features         # graph only: X = I
python train.py --model sage                      # or gat, gin
python train.py --model gat --hidden 8 --heads 8 --dropout 0.6 --lr 0.005
python train.py --layers 6 --residual             # depth and skip connections
python train.py --dataset karate --dropout 0 --weight_decay 0
python experiments.py                             # about 8 minutes; regenerates slide figures
python experiments.py F                           # only section F (A, B, C, D, F exist)
cd slides && latexmk -pdf gnn_tutorial.tex
```

## Data and splits

- **Cora**: 2,708 papers, 5,429 citation lines which become 5,278 undirected edges after merging
  duplicates and reciprocal citations, 1,433 binary word features (row-normalised), 7 topics.
- **Split**: for each seed, 20 random training nodes per class (140), then 500 validation and 1,000 test
  nodes drawn at random from the rest; the other 1,068 nodes are never labelled. These are the sizes of
  the Planetoid split, but it is *not* the fixed public Planetoid split used by Kipf & Welling, so the
  accuracies are not directly comparable with published tables. `--seed` sets both the split and the
  initial weights.
- **Karate club**: 34 nodes, 78 edges, embedded in `data.py`. Only nodes 0 and 33 are labelled;
  8 random other nodes are validation nodes and 24 are test nodes.

## Notes

- Slides 17 and 18 include lines of `gcn_numpy.py`, `gnn.py` and `train.py` by line number; if you edit
  those files, update the `linerange` values in the `.tex`.
- The seconds-per-run column on slide 12 is wall-clock time and will differ on your machine.
- Results are deterministic on one machine for a given seed; small differences between PyTorch
  versions are possible.
