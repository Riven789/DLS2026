# Convolutional Neural Networks from the Ground Up

A 20-slide tutorial on CNNs with the code that produced every number and figure on the slides.

## Files

| file | what it is |
|---|---|
| `slides/cnn_tutorial.pdf` | the deck |
| `slides/cnn_tutorial.tex` | Beamer + TikZ/pgfplots source (all diagrams and heat maps are source, no images) |
| `data.py` | Fashion-MNIST: idx-file parser, scaling and standardisation, train/val/test split, shifts, augmentation |
| `conv_numpy.py` | convolution from scratch in NumPy: forward (padding, stride), backward, max/average pooling, a NumPy forward pass of the whole network |
| `cnn.py` | the PyTorch models: LeNet-style CNN (optionally batch norm, dropout, GAP head), MLP baseline, softmax regression |
| `train.py` | training and testing harness: optimizers, learning rate, batch size, augmentation, best checkpoint, early stopping, test report |
| `experiments.py` | runs every experiment on the slides and writes `slides/generated/*` |
| `walkthrough.py` | the hand calculations on the slides |
| `test_cnn.py` | seven checks (forward vs PyTorch, backward vs finite differences and autograd, pooling, slide examples and equivariance, NumPy network vs PyTorch, data pipeline, overfitting a tiny batch) |
| `data/fashion/` | the first 12,000 Fashion-MNIST training images and the full 10,000-image test set (idx, gzip) |

## Data

Fashion-MNIST (Xiao, Rasul, Vollgraf 2017, MIT licence), from
`https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion`.
To keep the project small, `data/fashion/train12k-*` holds only the first 12,000 of the 60,000 training
images; each run draws 10,000 for training and 2,000 for validation from them (the seed sets the split).
The test files are the complete official test set. To rebuild the subset, download the four original
`.gz` files into `data/fashion/` and run `python data.py --subset`.

## Run it

Requires Python 3.10+, NumPy and PyTorch. Everything runs on one CPU core.

```bash
python test_cnn.py
python walkthrough.py
python conv_numpy.py
python train.py                               # CNN, about a minute
python train.py --model mlp                   # fully connected baseline, 784 -> 128 -> 64 -> 10
python train.py --model linear                # softmax regression baseline
python train.py --batchnorm --dropout 0.3 --augment 2
python train.py --model cnn_gap               # global-average-pooling head
python experiments.py                         # 5 to 10 minutes; regenerates all slide figures and numbers
cd slides && latexmk -pdf cnn_tutorial.tex
```

`python train.py --model {cnn,mlp,linear} --seed {0,1,2}` reproduces the nine runs behind the results
table on slide 17 (the same seeds give the same numbers on the same machine and PyTorch version).

## Notes

- Slides 15 and 16 include lines of `conv_numpy.py`, `cnn.py` and `train.py` by line number; if you edit
  those files, update the `linerange` values in the `.tex`.
- Every measured number on the slides is a macro from `slides/generated/results.tex`, written by
  `experiments.py`. The ImageNet figures on slide 19 are quoted from the cited papers, not measured.
- Results use a 10,000-image training subset and 12 epochs so that everything fits a small shared CPU.
  They are not state-of-the-art numbers for Fashion-MNIST.
- Training times on the slides are wall-clock on the machine that built the deck and will differ on yours.
