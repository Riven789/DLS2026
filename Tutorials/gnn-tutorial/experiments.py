"""
experiments.py -- runs every experiment shown on the slides and writes the
results into slides/generated/ (data files for pgfplots + LaTeX macros).

    python experiments.py          # 5 to 12 minutes on two CPU cores, depending on load

    python experiments.py F        # only the listed sections; other macros are kept

A  Cora statistics and k-hop neighbourhood sizes                 (slides 2, 5, 10)
B  smoothing without training: similarity of rows of A_hat^k X   (slide 11)
C  GCN vs MLP vs graph-only baselines, 10 random splits;
   training curves; effect of dropout + weight decay             (slides 12, 15, 19)
D  depth 1..8, with and without skip connections, 5 splits       (slide 11)
F  Zachary's karate club from two labels                         (slide 16)
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np
import torch

from data import load_graph
from gnn import graph_operators
from train import baselines, evaluate, prepare, run

GEN = "slides/generated"
os.makedirs(GEN, exist_ok=True)
MACROS: dict[str, str] = {}
T0 = time.time()


def macro(name: str, value) -> None:
    MACROS[name] = str(value)
    print(f"  {name} = {value}")


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def mean_std(tag: str, accs: list[float]) -> None:
    macro(tag + "Mean", pct(np.mean(accs)))
    macro(tag + "Std", pct(np.std(accs, ddof=1)))


def pair_cosine(H: torch.Tensor, pairs: torch.Tensor) -> float:
    """Mean cosine similarity between the rows of H for the given node pairs."""
    a, b = H[pairs[:, 0]], H[pairs[:, 1]]
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1, eps=1e-12).mean())


cora = load_graph("cora")
n, E = cora["n"], cora["edges"]
ops = graph_operators(E, n)
pairs = torch.randint(0, n, (5000, 2), generator=torch.Generator().manual_seed(0))   # random node pairs


def section_A() -> None:
    print("A  Cora statistics")
    deg = np.bincount(E.ravel(), minlength=n)
    macro("coraNodes", f"{n:,}".replace(",", "{,}"))
    macro("coraEdges", f"{len(E):,}".replace(",", "{,}"))
    macro("coraFeatures", f"{cora['X'].shape[1]:,}".replace(",", "{,}"))
    macro("coraClasses", len(cora["classes"]))
    macro("coraMeanDeg", f"{deg.mean():.1f}")
    macro("coraMedianDeg", f"{np.median(deg):.0f}")
    macro("coraMaxDeg", deg.max())
    macro("coraHomophily", pct((cora["y"][E[:, 0]] == cora["y"][E[:, 1]]).mean(), 0))
    macro("coraNonzero", pct((cora["X"] != 0).mean()))
    macro("coraWordsPerPaper", f"{(cora['X'] != 0).sum(1).mean():.0f}")
    macro("coraLabelledPct", pct(140 / n))

    # k-hop neighbourhoods: R_k[v, u] = 1 if u is within k hops of v
    R = torch.eye(n)
    A_loop = (ops["A"] + torch.eye(n).to_sparse()).coalesce()
    with open(f"{GEN}/hops.dat", "w") as f:
        f.write("k mean median frac\n")
        for k in range(0, 9):
            size = R.sum(1)
            f.write(f"{k} {size.mean():.1f} {size.median():.0f} {100 * size.mean() / n:.2f}\n")
            if k in (1, 2, 3, 4, 8):
                name = {1: "One", 2: "Two", 3: "Three", 4: "Four", 8: "Eight"}[k]
                macro("hop" + name + "Mean", f"{size.mean():.0f}")
                macro("hop" + name + "Pct", f"{100 * size.mean() / n:.0f}" if k > 2 else f"{100 * size.mean() / n:.1f}")
            R = (torch.sparse.mm(A_loop, R) > 0).float()


def section_B() -> None:
    print("B  repeated propagation A_hat^k X (no weights, no training)")
    H = torch.from_numpy(cora["X"])
    with open(f"{GEN}/smooth.dat", "w") as f:
        f.write("k cos\n")
        for k in range(0, 65):
            c = pair_cosine(H, pairs)
            if k in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
                f.write(f"{k} {c:.4f}\n")
            if k in (0, 2, 8, 64):
                macro("smooth" + {0: "Zero", 2: "Two", 8: "Eight", 64: "SixtyFour"}[k], f"{c:.2f}")
            H = torch.sparse.mm(ops["A_hat"], H)


def section_C() -> None:
    print("C  Cora, 10 random splits (20 labels per class / 500 val / 1000 test)")
    SEEDS = list(range(10))
    CONFIGS = {                       # tag: (keyword arguments for run(), seeds)
        "Gcn": (dict(model="gcn"), SEEDS),
        "Mlp": (dict(model="mlp"), SEEDS),
        "GcnNoFeat": (dict(model="gcn", no_features=True), SEEDS),
        "Sage": (dict(model="sage"), SEEDS),
        "Gat": (dict(model="gat", hidden=8, heads=8, dropout=0.6, lr=0.005), SEEDS[:5]),
        "GcnNoReg": (dict(model="gcn", dropout=0.0, weight_decay=0.0), SEEDS),
    }
    acc: dict[str, list[float]] = {"Lp": [], "Majority": []}
    hist: dict[str, list[dict]] = {}
    for seed in SEEDS:
        b = baselines(prepare(cora, seed))
        acc["Lp"].append(b["label_propagation"])
        acc["Majority"].append(b["majority"])
    for tag, (kw, seeds) in CONFIGS.items():
        acc[tag], t = [], time.time()
        for seed in seeds:
            net, d, history, m = run(cora, seed, **kw)
            acc[tag].append(m["accuracy"])
            if seed == 0:
                macro("cora" + tag + "Params", f"{net.num_params():,}".replace(",", "{,}"))
                macro("cora" + tag + "SeedAcc", pct(m["accuracy"]))
                macro("cora" + tag + "Epochs", len(history))
                if tag in ("Gcn", "Mlp", "GcnNoReg"):
                    hist[tag] = history
                if tag == "Gcn":
                    y, test = d["y"], d["masks"]["test"]
                    per_class = [(m["pred"][test & (y == c)] == c).float().mean().item() for c in range(7)]
                    with open(f"{GEN}/perclass.dat", "w") as f:
                        f.write("class name recall n\n")
                        for c, r in enumerate(per_class):
                            f.write(f"{c} {cora['classes'][c]} {100 * r:.1f} {int((test & (y == c)).sum())}\n")
        macro("time" + tag, f"{(time.time() - t) / len(seeds):.1f}")
    for tag, vals in acc.items():
        mean_std("cora" + tag, vals)
        print(f"  {tag:10s}", " ".join(f"{100 * v:.1f}" for v in vals))
    macro("coraSeeds", len(SEEDS))
    macro("coraGatSeeds", len(CONFIGS["Gat"][1]))
    macro("gcnBeatsMlp", sum(a > b for a, b in zip(acc["Gcn"], acc["Mlp"])))
    macro("gcnBeatsLp", sum(a > b for a, b in zip(acc["Gcn"], acc["Lp"])))
    macro("gcnBeatsSage", sum(a > b for a, b in zip(acc["Gcn"], acc["Sage"])))
    macro("gatBeatsGcn", sum(a > b for a, b in zip(acc["Gat"], acc["Gcn"])))
    macro("coraGcnFirstFiveMean", pct(np.mean(acc["Gcn"][:5])))
    macro("coraGcnMin", pct(min(acc["Gcn"])))
    macro("coraGcnMax", pct(max(acc["Gcn"])))
    macro("gcnMinusMlp", f"{100 * (np.mean(acc['Gcn']) - np.mean(acc['Mlp'])):.0f}")
    with open(f"{GEN}/seeds.dat", "w") as f:               # one dot per split, for the strip plot
        order = ["Majority", "Mlp", "GcnNoFeat", "Lp", "Sage", "Gcn", "Gat"]
        f.write("row model acc\n")
        for r, tag in enumerate(order):
            for v in acc[tag]:
                f.write(f"{r} {tag} {100 * v:.1f}\n")
    with open(f"{GEN}/means.dat", "w") as f:
        f.write("row model mean std\n")
        for r, tag in enumerate(order):
            f.write(f"{r} {tag} {100 * np.mean(acc[tag]):.2f} {100 * np.std(acc[tag], ddof=1):.2f}\n")

    # training curves of split 0
    print("E  training curves (split 0)")
    with open(f"{GEN}/curves.dat", "w") as f:
        f.write("epoch gcn_train gcn_val gcn_acc mlp_train mlp_val mlp_acc noreg_train noreg_val noreg_acc\n")
        hists = [hist['Gcn'], hist['Mlp'], hist['GcnNoReg']]
        for i in range(max(len(h) for h in hists)):
            cells = []
            for h in hists:
                cells += ([f"{h[i]['train_loss']:.4f}", f"{h[i]['val_loss']:.4f}", f"{100 * h[i]['val_acc']:.1f}"]
                          if i < len(h) else ["nan"] * 3)
            f.write(f"{i + 1} " + " ".join(cells) + "\n")
    best = int(np.argmin([r["val_loss"] for r in hist['Gcn']]))
    macro("gcnBestEpoch", best + 1)
    macro("gcnBestVal", f"{hist['Gcn'][best]['val_loss']:.2f}")
    macro("gcnFinalTrain", f"{hist['Gcn'][-1]['train_loss']:.2f}")
    macro("noregFinalTrain", f"{hist['GcnNoReg'][-1]['train_loss']:.3f}")
    macro("noregBestVal", f"{min(r['val_loss'] for r in hist['GcnNoReg']):.2f}")
    macro("mlpBestVal", f"{min(r['val_loss'] for r in hist['Mlp']):.2f}")


def section_D() -> None:
    print("D  depth (Cora, 5 splits, hidden 16; skip connections between the hidden layers)")
    DEPTHS, DSEEDS = list(range(1, 9)), list(range(5))
    rows = []
    for L in DEPTHS:
        row = dict(L=L)
        for res in (False, True):
            if res and L < 3:                              # no pair of equal-width hidden layers yet
                row.update(res_mean=row["mean"], res_std=row["std"], res_train=row["train"])
                continue
            accs, trains, coss = [], [], []
            for seed in DSEEDS:
                net, d, history, m = run(cora, seed, layers=L, residual=res)
                accs.append(m["accuracy"])
                trains.append(evaluate(net, d, "train")["accuracy"])
                with torch.no_grad():                      # how alike are the last hidden vectors?
                    hidden = net(d["X"], d["ops"], return_hidden=True)[1]
                coss.append(pair_cosine(hidden.to_dense() if hidden.is_sparse else hidden, pairs))
            p = "res_" if res else ""
            row.update({p + "mean": 100 * np.mean(accs), p + "std": 100 * np.std(accs, ddof=1),
                        p + "train": 100 * np.mean(trains), p + "cos": np.mean(coss)})
            print(f"  layers {L} residual {res}: test {row[p + 'mean']:.1f} +- {row[p + 'std']:.1f}"
                  f"   train {row[p + 'train']:.1f}   cos {row[p + 'cos']:.3f}")
        row.setdefault("res_cos", row["cos"])
        rows.append(row)
    with open(f"{GEN}/depth.dat", "w") as f:
        keys = ["L", "mean", "std", "train", "cos", "res_mean", "res_std", "res_train", "res_cos"]
        f.write(" ".join(keys) + "\n")
        for row in rows:
            f.write(" ".join(f"{row[k]:.3f}" if k != "L" else str(row[k]) for k in keys) + "\n")
    names = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"]
    for row in rows:
        macro("depth" + names[row["L"] - 1], f"{row['mean']:.1f}")
        macro("depthTrain" + names[row["L"] - 1], f"{row['train']:.0f}")
        macro("depthRes" + names[row["L"] - 1], f"{row['res_mean']:.1f}")
    macro("depthSeeds", len(DSEEDS))
    macro("depthBest", int(np.argmax([r["mean"] for r in rows])) + 1)


def section_F() -> None:
    print("F  karate club: two labelled nodes")
    karate = load_graph("karate")
    KSEEDS = list(range(20))
    KW = dict(hidden=16, dropout=0.0, weight_decay=0.0, lr=0.01, epochs=200, patience=50)
    k_acc = {"gcn": [], "lp": [], "majority": [], "mlp": []}
    for seed in KSEEDS:
        net, d, history, m = run(karate, seed, **KW)
        k_acc["gcn"].append(m["accuracy"])
        b = baselines(d)
        k_acc["lp"].append(b["label_propagation"])
        k_acc["majority"].append(b["majority"])
        k_acc["mlp"].append(run(karate, seed, model="mlp", **KW)[3]["accuracy"])
        if seed == 0:
            with torch.no_grad():
                logits = net(d["X"], d["ops"])
            pred = logits.argmax(1)
            rest = ~d["masks"]["train"]
            macro("karateSeedWrong", int((pred[rest] != d["y"][rest]).sum()))
            macro("karateSeedWrongNodes", ", ".join(str(i) for i in torch.nonzero((pred != d["y"]) & rest).flatten().tolist()))
            macro("karateSeedAcc", pct(m["accuracy"]))
            macro("karateParams", net.num_params())
            macro("karateEpochs", len(history))
            # a force-directed layout (Fruchterman-Reingold), only for drawing the graph
            rng = np.random.default_rng(3)
            A = np.zeros((34, 34))
            A[karate["edges"][:, 0], karate["edges"][:, 1]] = A[karate["edges"][:, 1], karate["edges"][:, 0]] = 1
            pos, kk = rng.uniform(-1, 1, size=(34, 2)), 1 / np.sqrt(34)
            for it in range(300):
                delta = pos[:, None, :] - pos[None, :, :]
                dist = np.maximum(np.linalg.norm(delta, axis=2), 0.01)
                force = (delta * (kk * kk / dist ** 2 - A * dist / kk)[:, :, None]).sum(1)
                step = 0.1 * (1 - it / 300)
                pos += force / np.maximum(np.linalg.norm(force, axis=1, keepdims=True), 1e-9) * step
            pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0))
            with open(f"{GEN}/karate_nodes.dat", "w") as f:
                f.write("id x y zA zB label pred role wrong\n")   # role 1 = labelled (train) node
                for i in range(34):
                    f.write(f"{i} {pos[i, 0]:.4f} {pos[i, 1]:.4f} {logits[i, 0]:.3f} {logits[i, 1]:.3f} "
                            f"{karate['y'][i]} {pred[i]} {int(d['masks']['train'][i])} {int(pred[i] != d['y'][i])}\n")
            with open(f"{GEN}/karate_edges.dat", "w") as f:
                f.write("x y\n")
                for a, b_ in karate["edges"]:
                    f.write(f"{pos[a, 0]:.4f} {pos[a, 1]:.4f}\n{pos[b_, 0]:.4f} {pos[b_, 1]:.4f}\n\n")
    for tag, vals in k_acc.items():
        mean_std("karate" + tag.capitalize(), vals)
    macro("karateSeeds", len(KSEEDS))
    macro("karateGcnBeatsLp", sum(a > b for a, b in zip(k_acc["gcn"], k_acc["lp"])))
    macro("karateLpBeatsGcn", sum(a < b for a, b in zip(k_acc["gcn"], k_acc["lp"])))


if __name__ == "__main__":
    wanted = [a.upper() for a in sys.argv[1:]] or ["A", "B", "C", "D", "F"]
    if len(wanted) < 5 and os.path.exists(f"{GEN}/results.tex"):       # keep the other sections' macros
        for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\n", open(f"{GEN}/results.tex").read()):
            MACROS[k] = v
    for name in wanted:
        globals()["section_" + name]()
    if len(wanted) == 5:
        macro("experimentMinutes", f"{(time.time() - T0) / 60:.0f}")
    with open(f"{GEN}/results.tex", "w") as f:
        for k, v in MACROS.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(MACROS)} macros to {GEN}/results.tex in {(time.time() - T0) / 60:.1f} min")
