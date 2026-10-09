# Long short-term memory networks: executable laboratory

A masked recurrent model learns to retain the last marked numerical value in a short sequence, predicting either its sign or its value.

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
| `X` | `[N,T,2]` | Feature 0 is a finite numerical value; feature 1 is a Boolean marker. |
| `mask` | `[N,T]` | Valid prefix of 1s followed by 0s; at least one valid step. |
| `y` | `[N]` | One class ID or regression target per sequence. |
| `split` (optional) | `[N]` | 0=train, 1=validation, 2=test. |

Generated sequences have 6–12 valid time steps and exactly two marked positions. Numerical values lie in [-1,1]. The last marked value is the regression target. Classification IDs 0/1 mean nonpositive/positive. Numerical scaling uses valid training steps only, and the marker stays 0/1. Padding has its own mask because an observed value of zero is meaningful.

An NPZ archive stores named NumPy arrays without Python pickles. To create one, use `np.savez_compressed("my_data.npz", X=X, y=y, ...)` with the additional arrays listed above. Prediction needs every input array but can omit `y` and `split`. There are no string inputs inside the network.

Classification labels must be contiguous integer IDs from 0 to K−1, with at least two classes and every class present in training. The default splitter needs at least ten examples per class (or ten total for regression). The code stores a stable name for each class in metadata. **For a new task, edit `data_api.class_names()` to match your class IDs before training**; the built-in names describe only the synthetic task. If K differs from that list length, the CLI uses numeric names. Never change the mapping after training without changing the associated metadata consistently. Missing or nonfinite labels are invalid; use the prediction command for unlabeled examples. Check annotation quality and feature/label row alignment before fitting.

Without an explicit split array, the seeded splitter assigns approximately 70% train, 15% validation and 15% test, stratified for classification. Rounding affects exact counts. Optional split IDs override it and must produce three nonempty partitions. Fit statistics on the training split only. Regression target mean/scale are also learned only on training labels, and reporting converts predictions back to original units.

## Model mathematics

A one-layer, forward LSTM uses D=2 features and H=16 hidden units. Concatenate each input with the previous hidden state and multiply by W with shape `(D+H,4H)`. Gate order is input, forget, candidate, output. Sigmoid transforms i/f/o; tanh transforms g.

`c_new = f*c_prev + i*g` and `h_new = o*tanh(c_new)`. At padded steps, both states remain unchanged. Independent sequences start from zero states. The forget-gate bias starts at 1. There are `4H(D+H+1)=1,216` recurrent parameters, plus 34 for a two-class head (1,250 total), or 17 for regression (1,233 total).

The backward loop uses full backpropagation through time, sums shared parameter gradients and preserves identity gradients through padding. It has no peepholes, projection, stacked layer, bidirectionality or truncated BPTT.

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
| Test examples | 242 | 240 |
| Selected epoch | 36 | 80 |
| Accuracy / RMSE | 0.983471 | 0.021065 |
| Majority accuracy / mean RMSE | 0.500000 | 0.575167 |
| Macro-F1 / MAE | 0.983471 | 0.016098 |

Classification confusion matrix (rows=true, columns=predicted): `[[119, 2], [2, 119]]`. Regression R²: 0.998658. Small numerical differences across numerical libraries and platforms are possible. Accuracy alone can conceal class imbalance; inspect per-class recall and the confusion matrix. Regression RMSE and MAE use the target's original units.

## Tests and adaptation

The seven tests passed for this lab. Shared checks cover sampled finite differences for every parameter array under both losses, exact prediction round-trip after saving/reloading, stable extreme-logit loss, deterministic disjoint splits and training-only preprocessing. A hand-calculated cell, right-padding invariance of predictions and gradients, and rejection of invalid mask order complement the shared tests.

Shuffle independent sequences, while keeping time order inside each one. The generated examples are independent, so random splitting is appropriate. Real forecasting data usually need chronological or rolling splits; overlapping windows and related entities must not leak across boundaries. Supply explicit split IDs. Record prediction cutoffs and target horizons before constructing windows.

One-hot features or embeddings can represent categories or tokens after a deliberate schema update. Imputation and missingness indicators represent missing observations; padding represents absent time steps. The current CLI produces one output per sequence. Per-step labels require a masked sequence loss. Stateful streaming requires explicit state ownership and reset logic; this lab resets each sequence.

An exact marker scan solves the synthetic task with zero error. The neural model is a memory and training demonstration, not a reason to replace that simple rule.

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

- Hochreiter and Schmidhuber (1997). Long Short-Term Memory. Neural Computation 9(8), 1735-1780. https://www.bioinf.jku.at/publications/older/2604.pdf
- Gers, Schmidhuber and Cummins (2000). Learning to Forget: Continual Prediction with LSTM. https://pubmed.ncbi.nlm.nih.gov/11032042/
- PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
- Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html
- Loshchilov and Hutter (2019). Decoupled Weight Decay Regularization. https://arxiv.org/abs/1711.05101
- scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html
