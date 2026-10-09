"""
data.py -- Fashion-MNIST from the raw idx files to normalised image tensors.

    python data.py            # print shapes, class counts and the pixel statistics
    python data.py --subset   # (re)build the 12,000-image training subset from the full 60,000-image file

An image is a grid of numbers.  One grey-scale picture is a tensor of shape (C, H, W) = (1, 28, 28)
with integer pixel values 0..255; a batch of N pictures has shape (N, C, H, W).

Files in data/fashion/ (idx format, gzip-compressed, from github.com/zalandoresearch/fashion-mnist):
    train12k-images-idx3-ubyte.gz, train12k-labels-idx1-ubyte.gz   first 12,000 of the 60,000 training images
    t10k-images-idx3-ubyte.gz,     t10k-labels-idx1-ubyte.gz       the full official test set, 10,000 images
"""
from __future__ import annotations

import gzip
import os
import struct
import sys

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fashion")
URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
N_SUBSET = 12_000


def read_idx(path: str) -> np.ndarray:
    """Parse an idx file: 2 zero bytes, a type byte (8 = uint8), the number of dimensions,
    then one big-endian 32-bit size per dimension, then the raw bytes."""
    with gzip.open(path, "rb") as f:
        zero, dtype, ndim = struct.unpack(">HBB", f.read(4))
        assert zero == 0 and dtype == 8, "not an unsigned-byte idx file"
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def write_idx(path: str, a: np.ndarray) -> None:
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">HBB", 0, 8, a.ndim) + struct.pack(">" + "I" * a.ndim, *a.shape) + a.tobytes())


def make_subset(full_dir: str) -> None:
    """Cut the first N_SUBSET images out of the full training files (download those from URL first)."""
    for kind in ("images-idx3", "labels-idx1"):
        a = read_idx(os.path.join(full_dir, f"train-{kind}-ubyte.gz"))
        write_idx(os.path.join(DATA_DIR, f"train12k-{kind}-ubyte.gz"), a[:N_SUBSET])


def load_fashion(n_train: int = 10_000, n_val: int = 2_000, seed: int = 0, permute_pixels: bool = False) -> dict:
    """Train / validation / test splits as float32 arrays of shape (N, 1, 28, 28).

    Pixels are scaled to [0, 1] and then standardised with ONE mean and ONE standard deviation computed
    on the training split (one pair per channel; a grey image has one channel).
    permute_pixels=True applies one fixed random permutation of the 784 pixel positions to every image
    (the experiment on slide 2: it destroys the spatial layout but keeps all the information)."""
    assert n_train + n_val <= N_SUBSET
    xs = read_idx(os.path.join(DATA_DIR, "train12k-images-idx3-ubyte.gz"))
    ys = read_idx(os.path.join(DATA_DIR, "train12k-labels-idx1-ubyte.gz")).astype(np.int64)
    x_test = read_idx(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
    y_test = read_idx(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)

    order = np.random.default_rng(seed).permutation(len(xs))          # shuffle once, then cut
    tr, va = order[:n_train], order[n_train:n_train + n_val]
    scale = lambda a: a.astype(np.float32)[:, None, :, :] / 255.0     # (N, 28, 28) uint8 -> (N, 1, 28, 28) in [0, 1]
    x_train, x_val, x_test = scale(xs[tr]), scale(xs[va]), scale(x_test)
    mean, std = float(x_train.mean()), float(x_train.std())           # training split only
    d = dict(task="classification", classes=CLASSES, n_out=10, mean=mean, std=std, raw_train=xs[tr],
             X_train=(x_train - mean) / std, y_train=ys[tr], X_val=(x_val - mean) / std, y_val=ys[va],
             X_test=(x_test - mean) / std, y_test=y_test)
    if permute_pixels:
        perm = np.random.default_rng(12345).permutation(28 * 28)
        for s in ("train", "val", "test"):
            X = d["X_" + s]
            d["X_" + s] = X.reshape(len(X), -1)[:, perm].reshape(X.shape)
    return d


def shift_images(X: np.ndarray, dy: int, dx: int, fill: float) -> np.ndarray:
    """Move every image dy pixels down and dx pixels right; vacated pixels get `fill` (the background)."""
    out = np.full_like(X, fill)
    H, W = X.shape[-2:]
    out[..., max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        X[..., max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out


def augment_shift(X: np.ndarray, rng: np.random.Generator, max_shift: int, fill: float) -> np.ndarray:
    """Data augmentation: shift each image of the batch by its own random (dy, dx) in [-max_shift, max_shift]."""
    out = np.empty_like(X)
    for n, (dy, dx) in enumerate(rng.integers(-max_shift, max_shift + 1, size=(len(X), 2))):
        out[n] = shift_images(X[n], int(dy), int(dx), fill)
    return out


def batches(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    """One epoch of shuffled mini-batches."""
    order = rng.permutation(len(X))
    for i in range(0, len(X), batch_size):
        idx = order[i:i + batch_size]
        yield X[idx], y[idx]


if __name__ == "__main__":
    if "--subset" in sys.argv:
        make_subset(sys.argv[sys.argv.index("--subset") + 1] if len(sys.argv) > 2 else DATA_DIR)
    d = load_fashion()
    for s in ("train", "val", "test"):
        print(f"{s:5s} X {d['X_' + s].shape} {d['X_' + s].dtype}   y {d['y_' + s].shape}   "
              f"per-class counts {np.bincount(d['y_' + s], minlength=10).tolist()}")
    print(f"pixel mean {d['mean']:.4f}, std {d['std']:.4f} (training split, after scaling to [0, 1])")
    print("classes:", ", ".join(f"{i}={c}" for i, c in enumerate(CLASSES)))
