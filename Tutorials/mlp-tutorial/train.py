"""
train.py -- training and testing harness for the tutorial MLP.

    python train.py --task titanic                 # classification: did the passenger survive?
    python train.py --task mpg                     # regression: miles per gallon of a car
    python train.py --task moons --hidden 16 16    # 2-D toy classification
    python train.py --task mpg --hidden            # no hidden layer = linear-model baseline
    python train.py --task titanic --optimizer sgd --lr 0.05 --batch_size 16

The three data splits have three different jobs:
    train       the only data that changes the weights
    validation  watched during training: early stopping, choosing hyperparameters
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

from data import batches, load_task
from mlp import ACTIVATIONS, MLP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["titanic", "mpg", "moons"], default="titanic")
    p.add_argument("--hidden", type=int, nargs="*", default=[32, 32], help="neurons per hidden layer")
    p.add_argument("--activation", choices=list(ACTIVATIONS), default="relu")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["sgd", "momentum", "adam", "adamw"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=30,
                   help="stop after this many epochs without a better validation loss (0 = never)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def make_optimizer(model: MLP, name: str, lr: float, weight_decay: float = 0.0):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def compute_loss(task: str, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if task == "classification":
        return F.cross_entropy(out, y)            # softmax + negative log-likelihood
    return F.mse_loss(out.squeeze(1), y)          # mean squared error


@torch.no_grad()
def predict(model: MLP, X: np.ndarray) -> torch.Tensor:
    model.eval()                                  # switches dropout off
    return model(torch.from_numpy(X))


def split_loss(model: MLP, d: dict, split: str) -> float:
    out = predict(model, d["X_" + split])
    return compute_loss(d["task"], out, torch.from_numpy(d["y_" + split])).item()


def fit(model: MLP, d: dict, optimizer: str = "adam", lr: float = 1e-3, weight_decay: float = 0.0,
        batch_size: int = 32, epochs: int = 200, patience: int = 20, seed: int = 0,
        verbose: bool = False) -> list[dict]:
    """Train `model` on d['X_train'], d['y_train'].  Returns one history row per epoch.
    With patience > 0 the weights from the best validation epoch are restored at the end."""
    opt = make_optimizer(model, optimizer, lr, weight_decay)
    rng = np.random.default_rng(seed)
    history, best_val, best_state, bad_epochs, t0 = [], float("inf"), None, 0, time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in batches(d["X_train"], d["y_train"], batch_size, rng):    # one epoch
            out = model(torch.from_numpy(xb))                 # 1. forward pass
            loss = compute_loss(d["task"], out, torch.from_numpy(yb))  # 2. how wrong?
            opt.zero_grad()
            loss.backward()                                   # 3. backpropagation: all gradients
            opt.step()                                        # 4. update every weight

        row = dict(epoch=epoch, train_loss=split_loss(model, d, "train"),
                   val_loss=split_loss(model, d, "val"), seconds=time.time() - t0)
        history.append(row)
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"epoch {epoch:4d} | train loss {row['train_loss']:.4f} | val loss {row['val_loss']:.4f}")
        if not np.isfinite(row["train_loss"]):                # diverged (learning rate too large)
            break
        if row["val_loss"] < best_val - 1e-5:
            best_val, bad_epochs = row["val_loss"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:           # early stopping
                if verbose:
                    print(f"early stop at epoch {epoch}; best was epoch {epoch - bad_epochs}")
                break
    if patience and best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate(model: MLP, d: dict, split: str = "test") -> dict:
    """Metrics on one split, in the original units of the problem."""
    out, y = predict(model, d["X_" + split]), d["y_" + split]
    if d["task"] == "classification":
        pred = out.argmax(dim=1).numpy()                      # most probable class
        K = d["n_out"]
        confusion = np.zeros((K, K), dtype=int)               # rows: truth, columns: prediction
        for t, p in zip(y, pred):
            confusion[t, p] += 1
        majority = np.bincount(d["y_train"], minlength=K).argmax()
        return dict(accuracy=float((pred == y).mean()), confusion=confusion,
                    baseline_accuracy=float((y == majority).mean()), n=len(y))
    sc = d["scaler"]
    pred, true = sc.inverse(out.squeeze(1).numpy()), sc.inverse(y)
    ss_res, ss_tot = ((true - pred) ** 2).sum(), ((true - true.mean()) ** 2).sum()
    return dict(rmse=float(np.sqrt(((true - pred) ** 2).mean())), mae=float(np.abs(true - pred).mean()),
                r2=float(1 - ss_res / ss_tot),
                baseline_rmse=float(np.sqrt(((true - sc.mean) ** 2).mean())),   # always predict the train mean
                n=len(y), pred=pred, true=true)


def print_report(d: dict, m: dict) -> None:
    print(f"\nTEST SET ({m['n']} examples never used for training or tuning)")
    if d["task"] == "classification":
        print(f"  accuracy {m['accuracy']:.3f}   (always guessing the majority class: {m['baseline_accuracy']:.3f})")
        print("  confusion matrix (rows = true class, columns = predicted class)")
        print("  " + " " * 10 + "".join(f"{c:>8}" for c in d["classes"]))
        for c, row in zip(d["classes"], m["confusion"]):
            print("  " + f"{c:>10}" + "".join(f"{v:8d}" for v in row))
        for k, c in enumerate(d["classes"]):
            col, row = m["confusion"][:, k].sum(), m["confusion"][k].sum()
            print(f"  class {c!r}: precision {m['confusion'][k, k] / max(col, 1):.3f}, "
                  f"recall {m['confusion'][k, k] / max(row, 1):.3f}")
    else:
        print(f"  RMSE {m['rmse']:.3f}   MAE {m['mae']:.3f}   R^2 {m['r2']:.3f}"
              f"   (always predicting the training mean: RMSE {m['baseline_rmse']:.3f})")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = args.out_dir or os.path.join("out", args.task)
    os.makedirs(out_dir, exist_ok=True)

    d = load_task(args.task, seed=args.seed)
    n_in = d["X_train"].shape[1]
    model = MLP(n_in, args.hidden, d["n_out"], args.activation, args.dropout)
    print(f"task {args.task} ({d['task']}): {len(d['X_train'])} train / {len(d['X_val'])} val / "
          f"{len(d['X_test'])} test examples, {n_in} input features")
    print(f"model: {n_in} -> {' -> '.join(map(str, args.hidden + [d['n_out']]))}, "
          f"{model.num_params():,} parameters; {args.optimizer}, lr {args.lr}, batch {args.batch_size}")

    history = fit(model, d, args.optimizer, args.lr, args.weight_decay, args.batch_size,
                  args.epochs, args.patience, args.seed, verbose=True)

    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    torch.save({"model": model.state_dict(), "args": vars(args), "feature_names": d["feature_names"]},
               os.path.join(out_dir, "model.pt"))
    print_report(d, evaluate(model, d, "test"))
    print(f"\nlog -> {out_dir}/log.csv   weights -> {out_dir}/model.pt")


if __name__ == "__main__":
    main()
