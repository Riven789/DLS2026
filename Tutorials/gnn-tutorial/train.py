"""
train.py -- training and testing harness for node classification on ONE graph.

    python train.py                                # GCN on Cora (2 layers, 16 hidden units)
    python train.py --model mlp                    # same features, graph ignored
    python train.py --model gcn --no_features      # graph only: every node gets a one-hot id
    python train.py --model sage | gat | gin       # other message-passing layers
    python train.py --layers 6 [--residual]        # depth and over-smoothing
    python train.py --dataset karate --hidden 2 --dropout 0 --weight_decay 0 --seed 0

Semi-supervised ("transductive") setting: the whole graph, with the features of ALL nodes, is the
input of every forward pass.  What is split into train / validation / test is the set of node LABELS:
    train mask   the only labels that enter the loss
    val mask     watched during training: early stopping, choosing hyperparameters
    test mask    touched once, at the end, for the number you report
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

from data import load_graph, make_split
from gnn import GNN, graph_operators, label_propagation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cora", "karate", "toy"], default="cora")
    p.add_argument("--model", choices=["gcn", "mlp", "sage", "gat", "gin"], default="gcn")
    p.add_argument("--layers", type=int, default=2, help="number of message-passing layers")
    p.add_argument("--hidden", type=int, default=16, help="units per hidden layer (per head for gat)")
    p.add_argument("--heads", type=int, default=8, help="attention heads (gat only)")
    p.add_argument("--residual", action="store_true", help="skip connections between hidden layers")
    p.add_argument("--no_features", action="store_true", help="replace X by the identity matrix")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01, help="learning rate (Adam)")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50,
                   help="stop after this many epochs without a better validation loss (0 = never)")
    p.add_argument("--seed", type=int, default=0, help="seeds the split AND the initial weights")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def prepare(g: dict, seed: int, no_features: bool = False) -> dict:
    """Tensors for one experiment: features, labels, graph operators and the three node masks."""
    X = np.eye(g["n"], dtype=np.float32) if no_features else g["X"]
    masks = {k: torch.from_numpy(v) for k, v in make_split(g, seed).items()}
    X = torch.from_numpy(X)
    if (X != 0).float().mean() < 0.1:                     # mostly zeros (bag of words, one-hot ids):
        X = X.to_sparse().coalesce()                      # store only the non-zeros
    return dict(X=X, y=torch.from_numpy(g["y"]), ops=graph_operators(g["edges"], g["n"]),
                masks=masks, n_classes=len(g["classes"]), classes=g["classes"])


@torch.no_grad()
def evaluate(model: GNN, d: dict, split: str) -> dict:
    """Loss and accuracy on the nodes of one mask.  The forward pass still uses the whole graph."""
    model.eval()                                          # switches dropout off
    logits, mask = model(d["X"], d["ops"]), d["masks"][split]
    pred = logits.argmax(dim=1)
    return dict(loss=F.cross_entropy(logits[mask], d["y"][mask]).item(),
                accuracy=(pred[mask] == d["y"][mask]).float().mean().item(), pred=pred)


def fit(model: GNN, d: dict, lr: float = 0.01, weight_decay: float = 5e-4, epochs: int = 200,
        patience: int = 50, verbose: bool = False) -> list[dict]:
    """Full-batch training: one epoch = one forward pass over the whole graph = one weight update.
    With patience > 0 the weights from the best validation epoch are restored at the end."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train = d["masks"]["train"]
    history, best_val, best_state, bad_epochs, t0 = [], float("inf"), None, 0, time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(d["X"], d["ops"])                  # 1. forward pass: ALL nodes, all edges
        loss = F.cross_entropy(logits[train], d["y"][train])   # 2. loss: labelled nodes only
        opt.zero_grad()
        loss.backward()                                   # 3. backpropagation
        opt.step()                                        # 4. Adam update

        val = evaluate(model, d, "val")                   # no weight update from these nodes
        history.append(dict(epoch=epoch, train_loss=loss.item(), val_loss=val["loss"],
                            val_acc=val["accuracy"], seconds=time.time() - t0))
        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(f"epoch {epoch:4d} | train loss {loss.item():.4f} | val loss {val['loss']:.4f}"
                  f" | val acc {val['accuracy']:.3f}")
        if val["loss"] < best_val - 1e-5:
            best_val, bad_epochs = val["loss"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:       # early stopping
                if verbose:
                    print(f"early stop at epoch {epoch}; best was epoch {epoch - bad_epochs}")
                break
    if patience and best_state is not None:
        model.load_state_dict(best_state)
    return history


def baselines(d: dict) -> dict:
    """Two reference points on the test nodes: the majority class, and label propagation."""
    y, train, test = d["y"], d["masks"]["train"], d["masks"]["test"]
    majority = torch.bincount(y[train], minlength=d["n_classes"]).argmax()
    lp = label_propagation(d["ops"], y, train, d["n_classes"]).argmax(dim=1)
    return dict(majority=(y[test] == majority).float().mean().item(),
                label_propagation=(lp[test] == y[test]).float().mean().item())


def run(g: dict, seed: int, model: str = "gcn", layers: int = 2, hidden: int = 16, heads: int = 8,
        residual: bool = False, no_features: bool = False, dropout: float = 0.5, lr: float = 0.01,
        weight_decay: float = 5e-4, epochs: int = 200, patience: int = 50, verbose: bool = False):
    """Split, build, train, test.  Returns (model, data dict, history, test metrics)."""
    d = prepare(g, seed, no_features)
    torch.manual_seed(seed)
    net = GNN(d["X"].shape[1], hidden, d["n_classes"], layers, model, dropout, residual, heads)
    history = fit(net, d, lr, weight_decay, epochs, patience, verbose)
    return net, d, history, evaluate(net, d, "test")


def print_report(d: dict, m: dict) -> None:
    y, test = d["y"], d["masks"]["test"]
    b = baselines(d)
    print(f"\nTEST NODES ({int(test.sum())} nodes whose labels were never used for training or tuning)")
    print(f"  accuracy {m['accuracy']:.3f}   (majority class {b['majority']:.3f}, "
          f"label propagation without features {b['label_propagation']:.3f})")
    K = d["n_classes"]
    confusion = np.zeros((K, K), dtype=int)               # rows: truth, columns: prediction
    for t, p in zip(y[test].tolist(), m["pred"][test].tolist()):
        confusion[t, p] += 1
    print("  confusion matrix (rows = true class, columns = predicted class)")
    for k, (c, row) in enumerate(zip(d["classes"], confusion)):
        print(f"  {c[:22]:>22}" + "".join(f"{v:5d}" for v in row) + f"   recall {row[k] / max(row.sum(), 1):.2f}")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or os.path.join("out", f"{args.dataset}_{args.model}")
    os.makedirs(out_dir, exist_ok=True)
    g = load_graph(args.dataset)
    kw = {k: v for k, v in vars(args).items() if k not in ("dataset", "seed", "out_dir")}
    net, d, history, m = run(g, args.seed, verbose=True, **kw)
    sizes = {k: int(v.sum()) for k, v in d["masks"].items()}
    print(f"\n{args.dataset}: {g['n']} nodes, {len(g['edges'])} edges, {d['X'].shape[1]} features; labels: "
          f"{sizes['train']} train / {sizes['val']} val / {sizes['test']} test (split seed {args.seed})")
    print(f"model: {args.model}, {args.layers} layers, hidden {args.hidden}, {net.num_params():,} parameters; "
          f"Adam lr {args.lr}, weight decay {args.weight_decay}, dropout {args.dropout}; "
          f"{len(history)} epochs in {history[-1]['seconds']:.1f} s")

    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    torch.save({"model": net.state_dict(), "args": vars(args)}, os.path.join(out_dir, "model.pt"))
    print_report(d, m)
    print(f"\nlog -> {out_dir}/log.csv   weights -> {out_dir}/model.pt")


if __name__ == "__main__":
    main()
