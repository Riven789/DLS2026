"""
train.py -- training and testing harness for the tutorial CNN and its baselines (Fashion-MNIST).

    python train.py                               # LeNet-style CNN, 10,000 training images
    python train.py --model mlp                   # fully connected baseline, 784 -> 128 -> 64 -> 10
    python train.py --model linear                # softmax regression baseline
    python train.py --batchnorm --dropout 0.3 --augment 2
    python train.py --optimizer sgd --lr 0.05 --batch_size 32 --epochs 20

The three data splits have three different jobs:
    train       the only images that change the weights
    validation  watched during training: best checkpoint, early stopping, choosing hyperparameters
    test        touched once, at the end, for the number you report
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from cnn import build_model
from data import augment_shift, batches, load_fashion, shift_images

torch.set_num_threads(1)          # the tutorial machine is small and shared; one thread is the fastest there


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "cnn_gap", "mlp", "linear"], default="cnn")
    p.add_argument("--n_train", type=int, default=10_000, help="training images (at most 10,000 with --n_val 2000)")
    p.add_argument("--n_val", type=int, default=2_000)
    p.add_argument("--batchnorm", action="store_true", help="batch normalisation after each convolution")
    p.add_argument("--dropout", type=float, default=0.0, help="dropout probability before the last layer")
    p.add_argument("--augment", type=int, default=0, help="random shift of up to this many pixels per training image")
    p.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="adam")
    p.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3,
                   help="stop after this many epochs without a better validation loss (0 = never)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def make_optimizer(model: torch.nn.Module, name: str, lr: float, weight_decay: float = 0.0):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, chunk: int = 1000) -> torch.Tensor:
    """Logits for all of X, computed in chunks.  eval() switches dropout off and makes batch
    normalisation use its running statistics."""
    model.eval()
    return torch.cat([model(torch.from_numpy(X[i:i + chunk])) for i in range(0, len(X), chunk)])


def loss_and_accuracy(model: torch.nn.Module, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    out, y = predict(model, X), torch.from_numpy(y)
    return F.cross_entropy(out, y).item(), (out.argmax(dim=1) == y).float().mean().item()


def fit(model: torch.nn.Module, d: dict, optimizer: str = "adam", lr: float = 2e-3, weight_decay: float = 0.0,
        batch_size: int = 64, epochs: int = 12, patience: int = 3, augment: int = 0, seed: int = 0,
        verbose: bool = False) -> list[dict]:
    """Train on d['X_train'], d['y_train'].  Returns one history row per epoch.
    With patience > 0 the weights from the best validation epoch are restored at the end."""
    opt = make_optimizer(model, optimizer, lr, weight_decay)
    rng = np.random.default_rng(seed)
    background = (0.0 - d["mean"]) / d["std"]                 # value of a black pixel after normalisation
    history, best_val, best_state, bad_epochs, t0 = [], float("inf"), None, 0, time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in batches(d["X_train"], d["y_train"], batch_size, rng):   # one epoch
            if augment:
                xb = augment_shift(xb, rng, augment, background)      # new random shifts every epoch
            logits = model(torch.from_numpy(xb))                      # 1. forward pass   (B, 10)
            loss = F.cross_entropy(logits, torch.from_numpy(yb))      # 2. softmax + negative log-likelihood
            opt.zero_grad()
            loss.backward()                                           # 3. backpropagation
            opt.step()                                                # 4. update every weight

        train_loss, train_acc = loss_and_accuracy(model, d["X_train"][:2000], d["y_train"][:2000])
        val_loss, val_acc = loss_and_accuracy(model, d["X_val"], d["y_val"])
        row = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc,
                   seconds=time.time() - t0)
        history.append(row)
        if verbose:
            print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.3f} | {row['seconds']:.0f} s")
        if not np.isfinite(train_loss):                               # diverged (learning rate too large)
            break
        if val_loss < best_val - 1e-5:
            best_val, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())            # best checkpoint so far
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:                   # early stopping
                if verbose:
                    print(f"early stop at epoch {epoch}; best was epoch {epoch - bad_epochs}")
                break
    if patience and best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate(model: torch.nn.Module, d: dict, split: str = "test", shift: tuple[int, int] = (0, 0)) -> dict:
    """Accuracy and confusion matrix on one split, optionally with every image shifted by (dy, dx) pixels."""
    X, y = d["X_" + split], d["y_" + split]
    if shift != (0, 0):
        X = shift_images(X, shift[0], shift[1], (0.0 - d["mean"]) / d["std"])
    pred = predict(model, X).argmax(dim=1).numpy()                    # most probable class
    K = d["n_out"]
    confusion = np.zeros((K, K), dtype=int)                           # rows: truth, columns: prediction
    np.add.at(confusion, (y, pred), 1)
    majority = np.bincount(d["y_train"], minlength=K).argmax()        # trivial baseline: one answer for everything
    return dict(accuracy=float((pred == y).mean()), confusion=confusion,
                baseline_accuracy=float((y == majority).mean()), n=len(y))


def shift_accuracy(model: torch.nn.Module, d: dict, s: int, n: int | None = None) -> float:
    """Test accuracy with every image moved by s pixels; mean over the four directions up, down, left, right.
    n limits the evaluation to the first n test images."""
    sub = dict(d, X_test=d["X_test"][:n], y_test=d["y_test"][:n])
    if s == 0:
        return evaluate(model, sub)["accuracy"]
    return float(np.mean([evaluate(model, sub, "test", sh)["accuracy"] for sh in ((s, 0), (-s, 0), (0, s), (0, -s))]))


def most_confused(confusion: np.ndarray, classes: list[str], top: int = 3) -> list[tuple[str, str, int]]:
    """Unordered class pairs with the most mistakes between them (both directions added)."""
    sym = confusion + confusion.T
    pairs = [(sym[a, b], a, b) for a in range(len(classes)) for b in range(a + 1, len(classes))]
    return [(classes[a], classes[b], int(n)) for n, a, b in sorted(pairs, reverse=True)[:top]]


def print_report(d: dict, m: dict) -> None:
    c, names = m["confusion"], d["classes"]
    print(f"\nTEST SET ({m['n']} images never used for training or tuning)")
    print(f"  accuracy {m['accuracy']:.4f}   (always guessing one class: {m['baseline_accuracy']:.4f})")
    print("  confusion matrix (rows = true class, columns = predicted class)")
    print("  " + " " * 9 + "".join(f"{n[:6]:>7}" for n in names))
    for n, row in zip(names, c):
        print("  " + f"{n:>9}" + "".join(f"{v:7d}" for v in row))
    print("  per-class recall: " + ", ".join(f"{n} {c[k, k] / c[k].sum():.2f}" for k, n in enumerate(names)))
    print("  most confused pairs: " + "; ".join(f"{a} / {b}: {n}" for a, b, n in most_confused(c, names)))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = args.out_dir or os.path.join("out", args.model)
    os.makedirs(out_dir, exist_ok=True)

    d = load_fashion(args.n_train, args.n_val, seed=args.seed)
    model = build_model(args.model, batchnorm=args.batchnorm, dropout=args.dropout)
    print(f"Fashion-MNIST: {len(d['X_train'])} train / {len(d['X_val'])} val / {len(d['X_test'])} test images, "
          f"shape {d['X_train'].shape[1:]}")
    print(f"model {args.model}: {model.num_params():,} parameters; {args.optimizer}, lr {args.lr}, "
          f"batch {args.batch_size}, seed {args.seed}")

    history = fit(model, d, args.optimizer, args.lr, args.weight_decay, args.batch_size, args.epochs,
                  args.patience, args.augment, args.seed, verbose=True)

    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    torch.save({"model": model.state_dict(), "args": vars(args), "mean": d["mean"], "std": d["std"]},
               os.path.join(out_dir, "model.pt"))
    print_report(d, evaluate(model, d, "test"))
    print("  accuracy with the test images shifted by s pixels (mean of up/down/left/right): "
          + ", ".join(f"s={s}: {shift_accuracy(model, d, s):.3f}" for s in (1, 2, 3)))
    print(f"\nlog -> {out_dir}/log.csv   weights -> {out_dir}/model.pt")


if __name__ == "__main__":
    main()
