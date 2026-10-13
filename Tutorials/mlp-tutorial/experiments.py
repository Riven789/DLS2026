"""
experiments.py -- runs every experiment shown on the slides and writes the
results into slides/generated/ (data files for pgfplots + LaTeX macros).

    python experiments.py          # about two minutes on a laptop CPU

A  two moons: linear model vs MLP decision boundary        (slide 6)
B  learning-rate sweep                                     (slide 14)
C  optimizers: SGD, momentum, Adam across learning rates   (slide 15)
D  batch size                                              (slide 16)
E  overfitting and early stopping                          (slide 17)
F  final test results, MLP vs linear, 10 random splits     (slide 19)
"""
from __future__ import annotations

import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_task
from mlp import MLP
from train import evaluate, fit, predict, split_loss

GEN = "slides/generated"
os.makedirs(GEN, exist_ok=True)
MACROS: dict[str, str] = {}


def macro(name: str, value) -> None:
    MACROS[name] = str(value)
    print(f"  {name} = {value}")


def new_model(d: dict, hidden: list[int], seed: int, dropout: float = 0.0) -> MLP:
    torch.manual_seed(seed)
    return MLP(d["X_train"].shape[1], hidden, d["n_out"], dropout=dropout)


def run(d, hidden, seed=0, **kw):
    """Train a fresh model; history gets an epoch-0 row (loss before any training)."""
    model = new_model(d, hidden, seed)
    first = dict(epoch=0, train_loss=split_loss(model, d, "train"), val_loss=split_loss(model, d, "val"), seconds=0.0)
    return model, [first] + fit(model, d, seed=seed, **kw)


def write_curves(path: str, curves: dict[str, list[float]]) -> None:
    n = max(len(v) for v in curves.values())
    with open(path, "w") as f:
        f.write("epoch " + " ".join(curves) + "\n")
        for i in range(n):
            vals = [f"{v[i]:.5f}" if i < len(v) and math.isfinite(v[i]) and v[i] < 1e3 else "nan" for v in curves.values()]
            f.write(f"{i} " + " ".join(vals) + "\n")


# ------------------------------------------------------------------ activations
with open(f"{GEN}/act.dat", "w") as f:
    f.write("x sigmoid tanh relu dsigmoid dtanh drelu\n")
    for i in range(-50, 51):
        x = i / 10
        s = 1 / (1 + math.exp(-x))
        f.write(f"{x:.1f} {s:.4f} {math.tanh(x):.4f} {max(x, 0):.4f} {s * (1 - s):.4f} "
                f"{1 - math.tanh(x) ** 2:.4f} {float(x > 0):.1f}\n")

# ------------------------------------------------------------------ A: two moons
print("A  two moons")
d = load_task("moons", seed=0)
grid_x, grid_y = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.2, 1.7, 300))
grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
for name, hidden in (("linear", []), ("mlp", [16, 16])):
    model, _ = run(d, hidden, optimizer="adam", lr=0.01, batch_size=32, epochs=300, patience=30)
    p1 = torch.softmax(predict(model, grid), dim=1)[:, 1].numpy().reshape(grid_x.shape)
    segments = plt.contour(grid_x, grid_y, p1, levels=[0.5]).allsegs[0]       # the p = 0.5 curve
    with open(f"{GEN}/moons_{name}.dat", "w") as f:
        f.write("x y\n")
        for seg in segments:
            for x, y in seg[::3]:
                f.write(f"{x:.3f} {y:.3f}\n")
            f.write("\n")
    macro("moons" + name.capitalize() + "Acc", f"{100 * evaluate(model, d)['accuracy']:.0f}")
    macro("moons" + name.capitalize() + "Params", model.num_params())
with open(f"{GEN}/moons_points.dat", "w") as f:
    f.write("x y label\n")
    for (x, y), lab in zip(d["X_train"][:220], d["y_train"][:220]):
        f.write(f"{x:.3f} {y:.3f} {lab}\n")

# ------------------------------------------------------------------ B: learning rate
print("B  learning rate (mpg, plain SGD, batch 32)")
d = load_task("mpg", seed=0)
HID, EPOCHS = [32, 32], 60
curves = {}
for lr, tag in ((0.0005, "Small"), (0.02, "Good"), (0.25, "Noisy"), (0.3, "Broken")):
    _, h = run(d, HID, optimizer="sgd", lr=lr, batch_size=32, epochs=EPOCHS, patience=0)
    curves[tag] = [r["train_loss"] for r in h]
    macro("lr" + tag + "Final", f"{h[-1]['train_loss']:.3f}")
write_curves(f"{GEN}/lr.dat", curves)
macro("lrInit", f"{curves['Good'][0]:.2f}")
_, h = run(d, HID, optimizer="sgd", lr=0.35, batch_size=32, epochs=5, patience=0)
assert not math.isfinite(h[-1]["train_loss"])            # lr 0.35 overflows to NaN in the first epoch

# ------------------------------------------------------------------ C: optimizers
print("C  optimizers (mpg, batch 32): best validation loss in 60 epochs, for each learning rate")
LRS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
table = {}
for opt in ("sgd", "momentum", "adam"):
    table[opt] = []
    for lr in LRS:
        _, h = run(d, HID, optimizer=opt, lr=lr, batch_size=32, epochs=EPOCHS, patience=0)
        table[opt].append(min(r["val_loss"] for r in h))
    print(f"  {opt:9s}", " ".join(f"{v:.3f}" for v in table[opt]))
    best = int(np.argmin(table[opt]))
    macro("opt" + opt.capitalize() + "Lr", LRS[best])
    macro("opt" + opt.capitalize() + "Val", f"{table[opt][best]:.3f}")
    macro("opt" + opt.capitalize() + "Worst", f"{max(table[opt][:-1]):.3f}")     # worst over lr <= 0.1
with open(f"{GEN}/optgrid.dat", "w") as f:
    f.write("lr sgd momentum adam\n")
    for i, lr in enumerate(LRS):
        f.write(f"{lr} " + " ".join(f"{table[o][i]:.4f}" for o in table) + "\n")

# ------------------------------------------------------------------ D: batch size
print("D  batch size (mpg, plain SGD, lr 0.02)")
curves, n_train = {}, len(d["X_train"])
macro("mpgNtrain", n_train)
for bs, tag in ((1, "One"), (8, "Eight"), (32, "ThirtyTwo"), (n_train, "Full")):
    t0 = time.time()
    _, h = run(d, HID, optimizer="sgd", lr=0.02, batch_size=bs, epochs=EPOCHS, patience=0)
    curves["b" + tag] = [r["val_loss"] for r in h]
    macro("bs" + tag + "Updates", math.ceil(n_train / bs))
    macro("bs" + tag + "Ms", f"{1000 * (time.time() - t0) / EPOCHS:.0f}")
    macro("bs" + tag + "Val", f"{h[-1]['val_loss']:.3f}")
write_curves(f"{GEN}/batch.dat", curves)

# ------------------------------------------------------------------ E: overfitting
print("E  overfitting (titanic, 256-256 hidden units, no regularisation)")
d = load_task("titanic", seed=0)
model, h = run(d, [256, 256], optimizer="adam", lr=1e-3, batch_size=32, epochs=150, patience=0)
write_curves(f"{GEN}/overfit.dat", {"train": [r["train_loss"] for r in h], "val": [r["val_loss"] for r in h]})
best_epoch = int(np.argmin([r["val_loss"] for r in h]))
macro("ofParams", f"{model.num_params():,}".replace(",", "{,}"))
macro("ofBestEpoch", best_epoch)
macro("ofBestVal", f"{h[best_epoch]['val_loss']:.3f}")
macro("ofLastVal", f"{h[-1]['val_loss']:.3f}")
macro("ofLastTrain", f"{h[-1]['train_loss']:.3f}")
macro("ofLastAcc", f"{100 * evaluate(model, d)['accuracy']:.1f}")
model, _ = run(d, [256, 256], optimizer="adam", lr=1e-3, batch_size=32, epochs=150, patience=20)
macro("ofStoppedAcc", f"{100 * evaluate(model, d)['accuracy']:.1f}")

# ------------------------------------------------------------------ F: final results
print("F  final results over 10 random splits")
SETTINGS = dict(mlp=dict(hidden=[32, 32], optimizer="adam", lr=1e-3, weight_decay=1e-4, batch_size=32,
                         epochs=500, patience=30),
                linear=dict(hidden=[], optimizer="adam", lr=1e-2, weight_decay=0.0, batch_size=32,
                            epochs=500, patience=30))
for task, key, fmt in (("titanic", "accuracy", "{:.1f}"), ("mpg", "rmse", "{:.2f}")):
    for name, s in SETTINGS.items():
        vals = []
        for seed in range(10):
            d = load_task(task, seed=seed)
            kw = {k: v for k, v in s.items() if k != "hidden"}
            model, h = run(d, s["hidden"], seed=seed, **kw)
            m = evaluate(model, d)
            vals.append(m[key] * (100 if key == "accuracy" else 1))
            if seed == 0:                                   # details for the slide
                tag = task + name.capitalize()
                macro(tag + "Params", model.num_params())
                macro(tag + "Epochs", len(h) - 1)
                if task == "titanic" and name == "mlp":
                    c = m["confusion"]
                    for nm, v in (("TN", c[0, 0]), ("FP", c[0, 1]), ("FN", c[1, 0]), ("TP", c[1, 1])):
                        macro("cm" + nm, v)
                    macro("titanicSeedAcc", f"{100 * m['accuracy']:.1f}")
                    macro("titanicBaseAcc", f"{100 * m['baseline_accuracy']:.1f}")
                    macro("titanicNtest", m["n"])
                    macro("titanicNfeat", d["X_train"].shape[1])
                if task == "mpg" and name == "mlp":
                    macro("mpgSeedRmse", f"{m['rmse']:.2f}")
                    macro("mpgSeedMae", f"{m['mae']:.2f}")
                    macro("mpgSeedRsq", f"{m['r2']:.2f}")
                    macro("mpgBaseRmse", f"{m['baseline_rmse']:.2f}")
                    macro("mpgNtest", m["n"])
                    macro("mpgNfeat", d["X_train"].shape[1])
                    with open(f"{GEN}/mpg_scatter.dat", "w") as f:
                        f.write("true pred\n")
                        for t, p in zip(m["true"], m["pred"]):
                            f.write(f"{t:.2f} {p:.2f}\n")
        tag = task + name.capitalize()
        macro(tag + "Mean", fmt.format(np.mean(vals)))
        macro(tag + "Std", fmt.format(np.std(vals, ddof=1)))
        globals()[tag + "_vals"] = vals
for k in ("titanicMlp", "titanicLinear", "mpgMlp", "mpgLinear"):
    print(f"  {k:14s}", " ".join(f"{v:.2f}" for v in globals()[k + "_vals"]))
macro("titanicMlpBetter", int(sum(a > b for a, b in zip(titanicMlp_vals, titanicLinear_vals))))
macro("titanicTies", int(sum(a == b for a, b in zip(titanicMlp_vals, titanicLinear_vals))))
macro("mpgMlpBetter", int(sum(a < b for a, b in zip(mpgMlp_vals, mpgLinear_vals))))

with open(f"{GEN}/results.tex", "w") as f:
    for k, v in MACROS.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(MACROS)} macros to {GEN}/results.tex")
