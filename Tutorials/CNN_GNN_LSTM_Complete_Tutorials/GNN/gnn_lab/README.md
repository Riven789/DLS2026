# Graph neural networks: executable laboratory

A small GraphSAGE-style network learns a graph-level label from node attributes and adjacency. It predicts whether most edges connect nodes with the same Boolean flag, or estimates the exact fraction.

Read the matching 20-slide presentation and `lecture_notes.md` for the full explanation. Everything needed to regenerate the data, train, evaluate and predict is included. Only Python and NumPy are required. There is no dataset download, GPU or framework dependency. The code implements the forward pass and parameter derivatives explicitly.

## Quick start

Use Python 3.10 or later. The reference runs used the version recorded in each checkpoint's `metadata.json` (NumPy 2.3.5). From this unzipped directory:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs seven automated tests, generates the datasets, trains both tasks, evaluates each test split and writes six unlabeled-example predictions. CPU runtime depends on your machine; the CNN is the slowest because its convolution gradients are explicit. For an initial smoke run, use `--epochs 3` in a separate training command. It will not reproduce the reference scores.

The supplied `reference_runs/` folder contains the completed reference experiments. A fresh run writes to `runs/`. Training refuses to overwrite a nonempty output directory. Give each new experiment a new name.

## Individual commands

```bash
python -m unittest discover -s tests -v
python lab.py generate --out data --seed 42

python lab.py train --task classification --data data/classification.npz --out runs/classification
python lab.py evaluate --data data/classification.npz --run runs/classification
python lab.py predict --data data/unlabeled.npz --run runs/classification --out runs/classification/new_predictions.csv

python lab.py train --task regression --data data/regression.npz --out runs/regression
python lab.py evaluate --data data/regression.npz --run runs/regression
python lab.py predict --data data/unlabeled.npz --run runs/regression --out runs/regression/new_predictions.csv
```

To use a reference checkpoint immediately, substitute `reference_runs/classification` or `reference_runs/regression` for the `--run` argument. Evaluation requires the original labeled archive because its byte hash guards the saved test indices. Prediction accepts new unlabeled examples that follow the same input contract. Avoid rerunning `generate` before reference evaluation: recompressing equivalent data can change an NPZ byte hash. Fresh training records the hash of its own generated archive.

## Data contract and labels

| Array | Shape | Contract |
|---|---|---|
| `X` | `[N,Nmax,2]` | Feature 0 is a Boolean flag; feature 1 is a finite measurement. |
| `A` | `[N,Nmax,Nmax]` | Symmetric, nonnegative adjacency, zero diagonal, no edges touching padding. |
| `mask` | `[N,Nmax]` | 1 for valid nodes, 0 for padding; at least one valid node. |
| `y` | `[N]` | One class ID or regression target per graph. |
| `split` (optional) | `[N]` | 0=train, 1=validation, 2=test. |

Generated graphs have 8–16 nodes and binary undirected edges. Labels 0/1 mean mostly different / mostly same flags on edges. Class 1 uses a strict fraction >0.5; ties are class 0. Regression predicts the fraction. Numerical feature scaling uses valid nodes in training graphs only. The Boolean flag remains 0/1. Padded features become zero after preprocessing. The adjacency convention is A[v,u] = contribution from source u to destination v.

An NPZ archive stores named NumPy arrays without Python pickles. To create one, use `np.savez_compressed("my_data.npz", X=X, y=y, ...)` with the additional arrays listed above. Prediction needs every input array but can omit `y` and `split`. There are no string inputs inside the network.

Classification labels must be contiguous integer IDs from 0 to K−1, with at least two classes and every class present in training. The default splitter needs at least ten examples per class (or ten total for regression). The code stores a stable name for each class in metadata. **For a new task, edit `data_api.class_names()` to match your class IDs before training**; the built-in names describe only the synthetic task. If K differs from that list length, the CLI uses numeric names. Never change the mapping after training without changing the associated metadata consistently. Missing or nonfinite labels are invalid; use the prediction command for unlabeled examples. Check annotation quality and feature/label row alignment before fitting.

Without an explicit split array, the seeded splitter assigns approximately 70% train, 15% validation and 15% test, stratified for classification. Rounding affects exact counts. Optional split IDs override it and must produce three nonempty partitions. Fit statistics on the training split only. Regression target mean/scale are also learned only on training labels, and reporting converts predictions back to original units.

## Model mathematics

For each of two layers, compute `M = row_normalize(A)` and `H = ReLU(H @ Ws + (M @ H) @ Wn + b)`, then apply the validity mask. A degree-zero row in M is zero. Self information has its own matrix, so the input adjacency excludes self loops. A masked mean pools valid final nodes before the linear output head.

This is a GraphSAGE-style teaching variant with mean aggregation and separate self/neighbor matrices. It does not implement neighbor sampling, hidden-vector unit normalization or edge features. Two features become 16 hidden features; the second layer keeps width 16. Classification has 642 parameters; scalar regression has 625. Dense adjacency is intentionally simple and has quadratic storage in node count.

All hidden and output weights are learned jointly. Classification uses a stable softmax cross-entropy on logits, with gradient `(softmax(z) - one_hot(y))/B`. Regression uses a linear output and MSE with gradient `2*(prediction-target)/B`. Output probabilities are mutually exclusive class probabilities, not independent multilabel predictions.

## The training loop

1. Initialize weights and compute training-only preprocessing.
2. Shuffle training examples and form minibatches, including the last partial batch.
3. Run the forward pass, compute the mean loss and backpropagate parameter gradients.
4. Clip the combined gradient norm if it exceeds 5, then apply AdamW.
5. Measure full training and validation loss at the end of the epoch.
6. Save the best validation weights in memory; stop after 15 non-improving epochs or 80 total.
7. Restore the selected weights, save the checkpoint, and evaluate the test split separately.

AdamW keeps exponential first and second gradient moments, applies bias correction and scales each parameter update using the second moment. Its defaults here are beta1=0.9, beta2=0.999 and epsilon=1e-8. Decoupled weight decay shrinks weight matrices by `(1 - learning_rate * weight_decay)` and excludes biases. The learning rate controls update size. A large rate can destabilize learning, while a very small one may train slowly. Gradient clipping bounds the norm of gradients before the optimizer, not the parameter norm. Validation loss excludes the weight-decay term.

| Training option | Default | Meaning |
|---|---:|---|
| `--epochs` | 80 | Maximum passes over training examples |
| `--batch` | 32 | Examples in each optimizer step |
| `--lr` | 0.003 | AdamW learning rate |
| `--weight-decay` | 0.0001 | Decoupled matrix-weight decay |
| `--clip` | 5 | Global gradient norm cap; 0 disables it |
| `--patience` | 15 | Epochs without validation improvement |
| `--seed` | 7 | Splits, model initialization and shuffle seed |

Example validation experiment: `python lab.py train --data data/classification.npz --out runs/lr_001 --lr 0.001`. Compare validation losses when choosing settings. Inspecting test scores repeatedly while tuning compromises the test estimate. Batch size affects gradient noise, memory and updates per epoch. A larger batch does not automatically improve quality. No learning-rate scheduler or exact training-resume mode is implemented.

## Saved artifacts

- `weights.npz`: validation-selected learned parameters.
- `metadata.json`: model configuration, preprocessing, target scale, split indices, dataset hash, label names, selected epoch, arguments and Python/NumPy versions.
- `history.json`: full training/validation losses and maximum pre-clipping gradient norm per epoch. Regression loss uses standardized targets.
- `test_metrics.json`: model scores and a training-only majority/mean baseline.
- `test_predictions.npz`: labels and raw output logits (classification) or original-unit values (regression).
- `new_predictions.csv`: class ID/name/probabilities or original-unit regression predictions.

Checkpoints support inference and evaluation. They do not contain optimizer moments, so restarting training from them would not exactly resume the original optimization trajectory. Only load your own trusted local code. NPZ reading disables pickle.

## Reference results

These are actual seeded CPU results from the included synthetic data, not expected scores for another dataset.

| Quantity | Classification | Regression |
|---|---:|---:|
| Test examples | 212 | 211 |
| Selected epoch | 80 | 79 |
| Accuracy / RMSE | 1.000000 | 0.011096 |
| Majority accuracy / mean RMSE | 0.514151 | 0.240194 |
| Macro-F1 / MAE | 1.000000 | 0.008088 |

Classification confusion matrix (rows=true, columns=predicted): `[[103, 0], [0, 109]]`. Regression R²: 0.997847. Small numerical differences across numerical libraries and platforms are possible. Accuracy alone can conceal class imbalance; inspect per-class recall and the confusion matrix. Regression RMSE and MAE use the target's original units.

## Tests and adaptation

The seven tests passed for this lab. Shared checks cover sampled finite differences for every parameter array under both losses, exact prediction round-trip after saving/reloading, stable extreme-logit loss, deterministic disjoint splits and training-only preprocessing. Permutation invariance, added-padding invariance and isolated-node behavior complement sampled finite-difference gradients.

Preserve the shared ordering between X, both axes of A and mask. Node IDs are bookkeeping, not automatically meaningful numerical features. One-hot categories, learned embeddings and edge attributes require explicit schema/model changes. This CLI supports graph-level labels only. Node tasks require a node label mask; edge tasks require a pair head and an explicit negative/positive sampling protocol. Remove held-out edges when their presence would reveal a link-prediction answer. Use whole-graph or entity/time-aware splits to match the intended inductive or transductive setting.

An exact algorithm can count matching-flag edges and solve this synthetic task with zero error. The included majority/mean baselines are deliberately weak; the reported result demonstrates the implementation, not an advantage over a task-specific algorithm.

The tests are focused correctness checks, not proof of correctness for every possible input. For a new domain, add checks for its data schema and required invariances, then compare against meaningful simple baselines. The tutorial's compact CPU implementation is educational rather than optimized for large-scale training.

## File guide

| File | Responsibility |
|---|---|
| `data_api.py` | Synthetic generator, input validation, feature preprocessing and schema |
| `model.py` | Model-specific forward computation and explicit derivatives |
| `common.py` | Stable losses, AdamW, splits, metrics and training loop |
| `lab.py` | Train/evaluate/predict command line and checkpoint metadata |
| `run_all.py` | End-to-end reproducible laboratory runner |
| `tests/` | Gradient, persistence, preprocessing and structural checks |
| `lecture_notes.md` | Full explanatory notes for all 20 slides |

## Primary references

- Hamilton, Ying and Leskovec (2017). Inductive Representation Learning on Large Graphs. https://arxiv.org/abs/1706.02216
- Kipf and Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. https://arxiv.org/abs/1609.02907
- Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. https://arxiv.org/abs/1704.01212
- Velickovic et al. (2018). Graph Attention Networks. https://arxiv.org/abs/1710.10903
- Loshchilov and Hutter (2019). Decoupled Weight Decay Regularization. https://arxiv.org/abs/1711.05101
- scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html
