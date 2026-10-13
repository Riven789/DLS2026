# Multilayer Perceptrons from the Ground Up

A 20-slide tutorial on MLPs with the code that produced every number and figure on the slides.

## Files

| file | what it is |
|---|---|
| `slides/mlp_tutorial.pdf` | the deck |
| `slides/mlp_tutorial.tex` | Beamer + TikZ/pgfplots source (all diagrams are source, no images) |
| `data.py` | raw table -> numbers: standardising numerical columns, 0/1 logical columns, one-hot categoricals, missing-value flags, label encoding, train/val/test split, mini-batches |
| `mlp_numpy.py` | an MLP with hand-written forward pass, backpropagation and SGD in NumPy, plus a gradient check |
| `mlp.py` | the same network in PyTorch |
| `train.py` | training and testing harness: optimizers, learning rate, batch size, early stopping, test report |
| `experiments.py` | runs every experiment on the slides and writes `slides/generated/*` |
| `walkthrough.py` | the hand calculations on the slides |
| `test_mlp.py` | six checks (backprop vs finite differences, NumPy vs PyTorch, encoder, leakage, XOR, overfit) |
| `data/` | Titanic (891 rows) and Auto MPG (398 rows) CSV files from the seaborn-data repository |

## Run it

Requires Python 3.10+, NumPy, PyTorch (matplotlib only for `experiments.py`).

```bash
python test_mlp.py
python walkthrough.py
python mlp_numpy.py
python train.py --task titanic            # classification
python train.py --task mpg                # regression
python train.py --task mpg --hidden       # no hidden layer: linear baseline
python train.py --task titanic --hidden 64 64 64 --optimizer sgd --lr 0.05 --batch_size 16 --dropout 0.2
python experiments.py                     # about a minute; regenerates slide figures
cd slides && latexmk -pdf mlp_tutorial.tex
```

To use your own table: add an entry to `SCHEMAS` in `data.py` naming the target column and which
columns are numerical, logical and categorical, and add the task name to `--task` in `train.py`.

## Notes

- Slide 18 includes lines of `mlp.py` and `train.py` by line number; if you edit those files,
  update the `linerange` values in the `.tex`.
- The ms/epoch column on slide 16 is wall-clock time and will differ on your machine.
