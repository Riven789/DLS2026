"""
experiments.py -- runs every experiment shown on the slides and writes the results into
slides/generated/ (TikZ heat maps, data files for pgfplots, LaTeX macros in results.tex).

    python experiments.py          # 5 to 10 minutes on one CPU core

A  pictures as numbers: sample images, a patch of raw pixel values           (slide 2)
B  Sobel edge detectors applied to a real image                              (slide 4)
C  translation equivariance of the convolution, measured                     (slide 8)
D  architecture table and parameter counts                                   (slides 2, 7, 11)
E  backward pass: finite differences and PyTorch autograd                    (slide 12)
F  learning-rate sweep, SGD vs Adam                                          (slide 13)
G  CNN vs MLP vs softmax regression, 3 seeds; training curves; confusions    (slides 13, 17)
H  batch normalisation, dropout, shift augmentation; shifted test images     (slide 14)
I  pixel-permutation experiment                                              (slide 2)
J  learned first-layer filters and feature maps                              (slide 18)
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from cnn import build_model, shape_table
from conv_numpy import conv2d, conv2d_backward, gradient_check
from data import CLASSES, load_fashion, read_idx, shift_images, DATA_DIR
from train import evaluate, fit, most_confused, shift_accuracy

GEN = "slides/generated"
os.makedirs(GEN, exist_ok=True)
MACROS: dict[str, str] = {}
T_START = time.time()
SHIFT_N = 2500                 # shifted-image accuracy uses the first 2,500 test images x 4 directions


def macro(name: str, value) -> None:
    MACROS[name] = str(value)
    print(f"  {name} = {value}")


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


# ------------------------------------------------------------------ TikZ heat maps
# Each helper returns \fill commands in "pixel" units: pixel (row r, column c) is the square
# (x0 + c, -r) .. (x0 + c + 1, -r - 1).  The slide chooses the physical size with [x=..mm, y=..mm].
def grid(a: np.ndarray, x0: float = 0.0, y0: float = 0.0, signed: bool = False, vmax: float | None = None,
         frame: bool = True) -> list[str]:
    """Grey map (dark = large) or, with signed=True, orange for positive and blue for negative values."""
    H, W = a.shape
    vmax = float(np.abs(a).max()) if vmax is None else vmax
    pct = np.clip(np.rint(100 * np.abs(a) / (vmax + 1e-12)), 0, 100).astype(int)
    out = []
    for r in range(H):
        c = 0
        while c < W:                                      # merge runs of equal colour into one rectangle
            e = c
            key = (pct[r, c], a[r, c] >= 0)
            while e + 1 < W and (pct[r, e + 1], a[r, e + 1] >= 0) == key:
                e += 1
            if pct[r, c] >= 3:
                colour = ("cAtt" if a[r, c] >= 0 else "cEmb") if signed else "cInk"
                out.append(rf"\fill[{colour}!{pct[r, c]}] ({x0 + c:.2f},{y0 - r:.2f}) rectangle ++({e - c + 1.04:.2f},-1.04);")
            c = e + 1
    if frame:
        out.append(rf"\draw[cInk!45,line width=0.3pt] ({x0:.2f},{y0:.2f}) rectangle ++({W},-{H});")
    return out


def label(text: str, x: float, y: float, opts: str = "") -> str:
    return rf"\node[lab{',' + opts if opts else ''}] at ({x:.2f},{y:.2f}) {{{text}}};"


def write_tex(name: str, lines: list[str]) -> None:
    with open(f"{GEN}/{name}.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def write_table(name: str, header: str, rows: list[list]) -> None:
    with open(f"{GEN}/{name}.dat", "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(" ".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in row) + "\n")


def run(model_name: str, d: dict, seed: int = 0, batchnorm: bool = False, dropout: float = 0.0, **kw):
    torch.manual_seed(seed)
    model = build_model(model_name, batchnorm=batchnorm, dropout=dropout)
    t0 = time.time()
    history = fit(model, d, seed=seed, **kw)
    return model, history, time.time() - t0


d = load_fashion(seed=0)
raw_test = read_idx(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
background = (0.0 - d["mean"]) / d["std"]
macro("nTrain", thousands(len(d["X_train"])))
macro("nVal", thousands(len(d["X_val"])))
macro("nTest", thousands(len(d["X_test"])))
macro("pixMean", f"{d['mean']:.3f}")
macro("pixStd", f"{d['std']:.3f}")

# ------------------------------------------------------------------ A: pictures as numbers
print("A  sample images")
IMG = 0                                                   # test image 0: an ankle boot
boot = raw_test[IMG].astype(float)
write_tex("img_boot", grid(boot, vmax=255))
r0, c0 = 9, 13                                            # a 5 x 5 patch on the front edge of the boot's shaft
patch = raw_test[IMG][r0:r0 + 5, c0:c0 + 5]
lines = grid(patch.astype(float), vmax=255 / 0.55, frame=False)        # pale background for black digits
for r in range(5):
    for c in range(5):
        lines.append(label(str(patch[r, c]), c + 0.5, -r - 0.5))
lines.append(r"\draw[cInk!45,line width=0.3pt,step=1] (0,0) grid (5,-5);")
write_tex("img_patch", lines)
macro("patchRow", r0)
macro("patchCol", c0)
v = int(patch[2, 2])
macro("patchValue", v)
macro("patchScaled", f"{v / 255:.3f}")
macro("patchNorm", f"{(v / 255 - d['mean']) / d['std']:+.2f}")
macro("blackNorm", f"{background:.2f}")
lines = []
for k in range(10):                                       # first test image of every class
    i = int(np.argmax(d["y_test"] == k))
    lines += grid(raw_test[i].astype(float), x0=k * 31.0, vmax=255)
    lines.append(label(CLASSES[k], k * 31.0 + 14, -31.5))
write_tex("img_classes", lines)

# ------------------------------------------------------------------ B: Sobel
print("B  Sobel edge detectors")
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
x = (boot / 255.0)[None, None]                            # (1, 1, 28, 28) in [0, 1]
gx = conv2d(x, SOBEL_X[None, None], padding=1)[0, 0]      # responds to left-right changes: vertical edges
gy = conv2d(x, SOBEL_X.T[None, None], padding=1)[0, 0]    # responds to up-down changes: horizontal edges
mag = np.sqrt(gx ** 2 + gy ** 2)
vmax = max(np.abs(gx).max(), np.abs(gy).max())
write_tex("sobel_gx", grid(gx, signed=True, vmax=vmax))
write_tex("sobel_gy", grid(gy, signed=True, vmax=vmax))
write_tex("sobel_mag", grid(mag))
macro("sobelMax", f"{vmax:.2f}")
macro("sobelZeroFrac", f"{100 * np.mean(mag < 0.05 * mag.max()):.0f}")

# ------------------------------------------------------------------ C: equivariance
print("C  translation equivariance")
rng = np.random.default_rng(0)
w = rng.normal(size=(4, 1, 5, 5))
xn = d["X_test"][IMG:IMG + 1].astype(np.float64)
S = (2, 3)                                                # shift: 2 pixels down, 3 pixels right
a = conv2d(shift_images(xn, *S, background), w, padding=2)            # shift, then convolve
b = shift_images(conv2d(xn, w, padding=2), *S, 0.0)                   # convolve, then shift
m = 2 + max(S)                                            # ignore a margin where the border is involved
macro("equivInterior", f"{np.abs(a - b)[:, :, m:-m, m:-m].max():.0e}" if np.abs(a - b)[:, :, m:-m, m:-m].max() > 0 else "0")
macro("equivBorder", f"{np.abs(a - b).max():.2f}")
W_dense = rng.normal(size=(64, 784)) / 28.0               # a random fully connected layer for contrast
h0, h1 = W_dense @ xn.reshape(-1), W_dense @ shift_images(xn, *S, background).reshape(-1)
macro("denseChange", f"{100 * np.linalg.norm(h1 - h0) / np.linalg.norm(h0):.0f}")

# ------------------------------------------------------------------ D: architecture table
print("D  architecture")
NAMES = {"Conv2d": r"conv $5\times5$", "ReLU": "ReLU", "MaxPool2d": r"max-pool $2\times2$", "Flatten": "flatten",
         "Linear": "linear", "input": "input image"}
cnn = build_model("cnn")
rows, total, conv_no = [], 0, 0
for layer, shape, n in shape_table(cnn):
    total += n
    if layer == "ReLU":                                   # same shape as the row before: fold it into that row
        rows[-1][0] += ", ReLU"
        continue
    name = NAMES[layer]
    if layer == "Conv2d":
        conv_no += 1
        name += ", pad 2" if conv_no == 1 else ", pad 0"
    shp = r"\times".join(map(str, shape))
    rows.append([name, rf"${shp}$ & {thousands(int(np.prod(shape)))} & {thousands(n) if n else '--'}\\"])
rows = [" & ".join(r) for r in rows]
assert total == cnn.num_params()
write_tex("arch_table", [r"\newcommand{\archRows}{%"] + rows + ["}"])      # a macro: \input inside a tabular breaks \midrule
for name, tag in (("cnn", "Cnn"), ("mlp", "Mlp"), ("linear", "Lin"), ("cnn_gap", "Gap")):
    macro("params" + tag, thousands(build_model(name).num_params()))
macro("paramRatio", f"{build_model('mlp').num_params() / cnn.num_params():.1f}")
conv_params = sum(p.numel() for p in cnn.features.parameters())
macro("paramsConvShare", f"{100 * conv_params / cnn.num_params():.0f}")
macro("paramsConv", thousands(conv_params))

# ------------------------------------------------------------------ E: backward pass
print("E  gradient check")
worst = max(max(gradient_check(0, s, p).values()) for s, p in ((1, 1), (2, 1), (1, 0)))
macro("gradFD", f"{worst:.0e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}")
xg, wg, bg = rng.normal(size=(2, 3, 8, 8)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
xt, wt, bt = (torch.tensor(v, requires_grad=True) for v in (xg, wg, bg))
yt = F.conv2d(xt, wt, bt, stride=2, padding=1)
G = rng.normal(size=tuple(yt.shape))
(yt * torch.from_numpy(G)).sum().backward()
dx, dw, db = conv2d_backward(G, xg, wg, stride=2, padding=1)
err = max(np.abs(dx - xt.grad.numpy()).max(), np.abs(dw - wt.grad.numpy()).max(), np.abs(db - bt.grad.numpy()).max())
macro("gradTorch", f"{err:.0e}".replace("e-", r"\times10^{-") + "}")
fwd = np.abs(conv2d(xg, wg, bg, 2, 1) - yt.detach().numpy()).max()
macro("fwdTorch", f"{fwd:.0e}".replace("e-", r"\times10^{-") + "}")

# ------------------------------------------------------------------ F: learning rate
print("F  learning rate (CNN, 3,000 training images, 2 epochs)")
small = load_fashion(n_train=3000, n_val=2000, seed=0)
LRS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
table = {}
for opt in ("sgd", "adam"):
    table[opt] = []
    for lr in LRS:
        _, h, _ = run("cnn", small, optimizer=opt, lr=lr, epochs=2, patience=0)
        table[opt].append(min(h[-1]["val_loss"], 2.5) if np.isfinite(h[-1]["val_loss"]) else 2.5)
    print(f"  {opt:5s}", " ".join(f"{v:.3f}" for v in table[opt]))
    best = int(np.argmin(table[opt]))
    macro("lr" + opt.capitalize() + "Best", LRS[best])
    macro("lr" + opt.capitalize() + "BestVal", f"{table[opt][best]:.2f}")
    macro("lr" + opt.capitalize() + "Range", f"{LRS[min(i for i, v in enumerate(table[opt]) if v < 1.0)]:g} to "
                                             f"{LRS[max(i for i, v in enumerate(table[opt]) if v < 1.0)]:g}")
write_table("lr", "lr sgd adam", [[lr, table["sgd"][i], table["adam"][i]] for i, lr in enumerate(LRS)])
macro("lrChance", f"{np.log(10):.2f}")

# ------------------------------------------------------------------ G: main comparison
print("G  CNN vs MLP vs softmax regression, 3 seeds")
SEEDS = (0, 1, 2)
acc = {m: [] for m in ("cnn", "mlp", "linear")}
shifted = {m: [] for m in ("cnn", "mlp", "linear")}
SHIFTS = (0, 1, 2, 3, 4)
for seed in SEEDS:
    ds = load_fashion(seed=seed)
    for name in acc:
        model, h, secs = run(name, ds, seed=seed)
        res = evaluate(model, ds)
        acc[name].append(100 * res["accuracy"])
        shifted[name].append([100 * shift_accuracy(model, ds, s, SHIFT_N) for s in SHIFTS])
        print(f"  seed {seed} {name:6s} test {acc[name][-1]:.2f}  epochs {len(h)}  {secs:.0f} s  shifted {np.round(shifted[name][-1], 1)}")
        if seed == 0:
            tag = name.capitalize()
            macro("epochs" + tag, len(h))
            macro("seconds" + tag, f"{secs:.0f}")
            macro("seedAcc" + tag, f"{acc[name][-1]:.1f}")
        if seed == 0 and name == "cnn":
            cnn0, hist0, res0 = model, h, res
for name, tag in (("cnn", "Cnn"), ("mlp", "Mlp"), ("linear", "Lin")):
    macro("acc" + tag + "Mean", f"{np.mean(acc[name]):.1f}")
    macro("acc" + tag + "Std", f"{np.std(acc[name], ddof=1):.1f}")
    macro("acc" + tag + "All", ", ".join(f"{v:.1f}" for v in acc[name]))
    macro("err" + tag, f"{100 - np.mean(acc[name]):.1f}")
macro("accBase", f"{100 * res0['baseline_accuracy']:.0f}")
macro("gainCnnMlp", f"{np.mean(acc['cnn']) - np.mean(acc['mlp']):.1f}")
macro("gainMlpLin", f"{np.mean(acc['mlp']) - np.mean(acc['linear']):.1f}")
macro("errCut", f"{100 * (1 - (100 - np.mean(acc['cnn'])) / (100 - np.mean(acc['mlp']))):.0f}")
macro("nSeeds", len(SEEDS))

write_table("curves", "epoch train val trainacc valacc",
            [[r["epoch"], r["train_loss"], r["val_loss"], 100 * r["train_acc"], 100 * r["val_acc"]] for r in hist0])
best = int(np.argmin([r["val_loss"] for r in hist0]))
macro("bestEpoch", hist0[best]["epoch"])
macro("bestVal", f"{hist0[best]['val_loss']:.2f}")
macro("lastTrain", f"{hist0[-1]['train_loss']:.2f}")
macro("lastVal", f"{hist0[-1]['val_loss']:.2f}")
macro("lastTrainAcc", f"{100 * hist0[-1]['train_acc']:.0f}")
macro("lastValAcc", f"{100 * hist0[-1]['val_acc']:.0f}")

C = res0["confusion"]                                     # confusion matrix of the seed-0 CNN
lines = []
for r in range(10):
    for c in range(10):
        n = int(C[r, c])
        if r == c:
            lines.append(rf"\fill[cFfn!{int(np.rint(55 * n / C.max()))}] ({c},{-r}) rectangle ++(1,-1);")
        elif n:
            lines.append(rf"\fill[cBad!{int(np.rint(min(100, 85 * n / (C - np.diag(np.diag(C))).max())))}] ({c},{-r}) rectangle ++(1,-1);")
        if n:
            lines.append(label(str(n), c + 0.5, -r - 0.5))
    lines.append(label(CLASSES[r], -0.15, -r - 0.5, "anchor=east"))
    lines.append(label(CLASSES[r], r + 0.5, 0.15, "anchor=west,rotate=60"))
lines.append(r"\draw[cInk!30,line width=0.2pt,step=1] (0,0) grid (10,-10);")
write_tex("confusion", lines)
for (a_, b_, n), tag in zip(most_confused(C, CLASSES), ("One", "Two", "Three")):
    macro("conf" + tag, f"{a_} / {b_}")
    macro("conf" + tag + "N", n)
recall = C.diagonal() / C.sum(1)
macro("worstClass", CLASSES[int(recall.argmin())])
macro("worstRecall", f"{100 * recall.min():.0f}")
macro("bestClass", CLASSES[int(recall.argmax())])
macro("bestRecall", f"{100 * recall.max():.0f}")
shirts = [CLASSES.index(c) for c in ("T-shirt", "Pullover", "Coat", "Shirt", "Dress")]
off = C - np.diag(np.diag(C))
macro("upperShare", f"{100 * off[np.ix_(shirts, shirts)].sum() / off.sum():.0f}")

# ------------------------------------------------------------------ H: regularisation and shifted test images
print("H  batch norm, dropout, augmentation (seed 0)")
d = load_fashion(seed=0)
VARIANTS = (("Bn", dict(batchnorm=True)), ("Drop", dict(dropout=0.3)), ("Aug", dict(augment=2)),
            ("All", dict(batchnorm=True, dropout=0.3, augment=2)))
abl = {"Plain": (hist0, 100 * res0["accuracy"], shifted["cnn"][0])}
for tag, kw in VARIANTS:
    model, h, secs = run("cnn", d, seed=0, **kw)
    abl[tag] = (h, 100 * evaluate(model, d)["accuracy"], [100 * shift_accuracy(model, d, s, SHIFT_N) for s in SHIFTS])
    print(f"  {tag:5s} test {abl[tag][1]:.2f}  epochs {len(h)}  {secs:.0f} s  shifted {np.round(abl[tag][2], 1)}")
    if tag == "All":
        write_table("curves_reg", "epoch train val", [[r["epoch"], r["train_loss"], r["val_loss"]] for r in h])
for tag, (h, test_acc, sh) in abl.items():
    b_ = int(np.argmin([r["val_loss"] for r in h]))
    macro("abl" + tag + "Test", f"{test_acc:.1f}")
    macro("abl" + tag + "Val", f"{100 * h[b_]['val_acc']:.1f}")
    macro("abl" + tag + "Gap", f"{h[b_]['val_loss'] - h[b_]['train_loss']:.2f}")
    macro("abl" + tag + "ShiftTwo", f"{sh[2]:.1f}")
    macro("abl" + tag + "Epochs", len(h))
model, h, _ = run("mlp", d, seed=0, augment=2)
mlp_aug = [100 * shift_accuracy(model, d, s, SHIFT_N) for s in SHIFTS]
macro("mlpAugTest", f"{100 * evaluate(model, d)['accuracy']:.1f}")
macro("mlpAugShiftTwo", f"{mlp_aug[2]:.1f}")
mean_shift = {m: np.mean(shifted[m], axis=0) for m in shifted}
write_table("shift", "s cnn mlp linear cnnaug mlpaug",
            [[s, float(mean_shift["cnn"][i]), float(mean_shift["mlp"][i]), float(mean_shift["linear"][i]),
              float(abl["Aug"][2][i]), float(mlp_aug[i])] for i, s in enumerate(SHIFTS)])
for name, tag in (("cnn", "Cnn"), ("mlp", "Mlp"), ("linear", "Lin")):
    macro("shiftZero" + tag, f"{mean_shift[name][0]:.1f}")
    macro("shiftTwo" + tag, f"{mean_shift[name][2]:.1f}")
    macro("shiftFour" + tag, f"{mean_shift[name][4]:.1f}")
    macro("shiftDrop" + tag, f"{mean_shift[name][0] - mean_shift[name][2]:.0f}")
macro("shiftN", thousands(SHIFT_N))

# ------------------------------------------------------------------ I: pixel permutation
print("I  fixed random permutation of the 784 pixel positions (seed 0)")
dp = load_fashion(seed=0, permute_pixels=True)
for name, tag in (("cnn", "Cnn"), ("mlp", "Mlp")):
    model, h, _ = run(name, dp, seed=0)
    macro("perm" + tag, f"{100 * evaluate(model, dp)['accuracy']:.1f}")
macro("permCnnDrop", f"{acc['cnn'][0] - float(MACROS['permCnn']):.1f}")
macro("permMlpDrop", f"{acc['mlp'][0] - float(MACROS['permMlp']):.1f}")

# ------------------------------------------------------------------ J: what the network learned
print("J  learned filters and feature maps (seed-0 CNN)")
W1 = cnn0.features[0].weight.detach().numpy()[:, 0]       # (8, 5, 5)
with torch.no_grad():
    maps = torch.relu(cnn0.features[0](torch.from_numpy(d["X_test"][IMG:IMG + 1])))[0].numpy()   # (8, 28, 28)
order = np.argsort(-maps.reshape(8, -1).mean(1))          # most active filter first
lines_f, lines_m = [], []
for pos, k in enumerate(order):
    lines_f += grid(W1[k], x0=pos * 6.0, signed=True, vmax=float(np.abs(W1).max()))
    lines_f.append(label(str(pos + 1), pos * 6.0 + 2.5, 1.0))
    lines_m += grid(maps[k], x0=(pos % 4) * 30.0, y0=-(pos // 4) * 33.0, vmax=float(maps.max()))
    lines_m.append(label(f"map {pos + 1}", (pos % 4) * 30.0 + 14, -(pos // 4) * 33.0 - 29.6))
write_tex("filters", lines_f)
write_tex("feature_maps", lines_m)
macro("deadMaps", int((maps.reshape(8, -1).max(1) == 0).sum()))
macro("mapSparsity", f"{100 * (maps == 0).mean():.0f}")
macro("predClass", CLASSES[int(cnn0(torch.from_numpy(d['X_test'][IMG:IMG + 1])).argmax())])
macro("trueClass", CLASSES[int(d["y_test"][IMG])])

macro("totalMinutes", f"{(time.time() - T_START) / 60:.0f}")
with open(f"{GEN}/results.tex", "w") as f:
    for k, v in MACROS.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(MACROS)} macros to {GEN}/results.tex in {(time.time() - T_START) / 60:.1f} min")
