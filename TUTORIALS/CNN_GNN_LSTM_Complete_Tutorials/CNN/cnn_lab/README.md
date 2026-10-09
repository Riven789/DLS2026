# Convolutional neural networks: executable laboratory

A compact spatial feature extractor learns bar orientation (classification) or bar amplitude (regression) from noisy synthetic images.

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
| `X` | `[N,C,H,W]` | Finite numerical channels; even H and W, both at least 4. |
| `y` | `[N]` | One class ID or regression target per image. |
| `split` (optional) | `[N]` | 0=train, 1=validation, 2=test. |

Generated images are `[N,1,12,12]`. Labels 0/1/2 mean horizontal/vertical/diagonal. Regression labels are the added bar amplitude, sampled from 0.6 to 1.5. Gaussian noise has standard deviation 0.12. Per-channel means and standard deviations come only from training pixels; use the saved statistics for all later images. Do not normalize each image separately when absolute brightness carries the target.

An NPZ archive stores named NumPy arrays without Python pickles. To create one, use `np.savez_compressed("my_data.npz", X=X, y=y, ...)` with the additional arrays listed above. Prediction needs every input array but can omit `y` and `split`. There are no string inputs inside the network.

Classification labels must be contiguous integer IDs from 0 to K−1, with at least two classes and every class present in training. The default splitter needs at least ten examples per class (or ten total for regression). The code stores a stable name for each class in metadata. **For a new task, edit `data_api.class_names()` to match your class IDs before training**; the built-in names describe only the synthetic task. If K differs from that list length, the CLI uses numeric names. Never change the mapping after training without changing the associated metadata consistently. Missing or nonfinite labels are invalid; use the prediction command for unlabeled examples. Check annotation quality and feature/label row alignment before fitting.

Without an explicit split array, the seeded splitter assigns approximately 70% train, 15% validation and 15% test, stratified for classification. Rounding affects exact counts. Optional split IDs override it and must produce three nonempty partitions. Fit statistics on the training split only. Regression target mean/scale are also learned only on training labels, and reporting converts predictions back to original units.

## Model mathematics

`Conv(1,6,3,pad=1) → ReLU → AvgPool(2,stride=2) → Conv(6,8,3,pad=1) → ReLU → global spatial mean → Linear(8,K)`.

The classifier has 527 parameters for K=3. Scalar regression has 509. Convolution uses cross-correlation: no kernel flip. Filters share weights across spatial positions. Backpropagation accumulates weight gradients from all positions and adds input gradients where receptive fields overlap. Average pooling distributes each derivative equally to its four input pixels.

A numerical check: input `[[1,2,0],[0,1,3],[2,1,0]]` and kernel `[[1,0],[0,-1]]`, with no padding, stride 1 and zero bias, produce `[[0,-1],[-1,1]]`.

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
| Test examples | 183 | 180 |
| Selected epoch | 80 | 78 |
| Accuracy / RMSE | 1.000000 | 0.036067 |
| Majority accuracy / mean RMSE | 0.344262 | 0.266264 |
| Macro-F1 / MAE | 1.000000 | 0.027471 |

Classification confusion matrix (rows=true, columns=predicted): `[[63, 0, 0], [0, 59, 0], [0, 0, 61]]`. Regression R²: 0.981558. Small numerical differences across numerical libraries and platforms are possible. Accuracy alone can conceal class imbalance; inspect per-class recall and the confusion matrix. Regression RMSE and MAE use the target's original units.

## Tests and adaptation

The seven tests passed for this lab. Shared checks cover sampled finite differences for every parameter array under both losses, exact prediction round-trip after saving/reloading, stable extreme-logit loss, deterministic disjoint splits and training-only preprocessing. A hand-calculated convolution, pooling derivative and overlapping input gradients complement the shared gradient and checkpoint tests.

For RGB input, provide three consistently ordered channels and train a fresh model. For binary image masks, use 0/1 values with a documented channel meaning. Categorical class names belong in the label mapping, not in pixel values. A label per pixel would be segmentation and requires a spatial output head and loss, which this CLI does not implement. Split related images by specimen, patient, scene or capture session. Any training augmentation must preserve the label meaning; a 90-degree rotation changes the orientation class in this dataset.

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

- Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/
- Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 9. https://www.deeplearningbook.org/contents/convnets.html
- Loshchilov and Hutter (2019). Decoupled Weight Decay Regularization. https://arxiv.org/abs/1711.05101
- scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html
