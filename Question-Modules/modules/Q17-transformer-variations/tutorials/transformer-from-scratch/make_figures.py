"""
make_figures.py -- turn a finished training run into the data the slides read.

    python make_figures.py          # after train.py; writes slides/generated/*

Outputs:  act.dat            activation-function curves (slide 10)
          attn_heatmap.tex   TikZ heat maps of two attention heads (slide 9)
          log.csv            loss curve (slide 19)
          sample.txt         generated text (slide 19)
          results.tex        headline numbers as LaTeX macros (slide 19)
"""
import csv
import math
import os
import shutil

import torch

from model import GPT, GPTConfig
from tokenizer import load_tokenizer

OUT, GEN = "out", "slides/generated"
PROMPT = "I will not be the man"
HEADS = (0, 2)          # which heads of block 1 to draw
os.makedirs(GEN, exist_ok=True)

# ---- activation curves ----------------------------------------------------
with open(f"{GEN}/act.dat", "w") as f:
    f.write("x relu gelu silu\n")
    for i in range(-40, 33):
        x = i / 10
        gelu = x * 0.5 * (1 + math.erf(x / math.sqrt(2)))
        f.write(f"{x:.1f} {max(x, 0):.4f} {gelu:.4f} {x / (1 + math.exp(-x)):.4f}\n")

# ---- model ----------------------------------------------------------------
tok = load_tokenizer(f"{OUT}/tokenizer.json")
ckpt = torch.load(f"{OUT}/ckpt.pt", map_location="cpu")
model = GPT(GPTConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.eval()


def tex_escape(s: str) -> str:
    s = s.replace("\n", r"\textbackslash n").replace(" ", r"\textvisiblespace ")
    for ch in "&%$#_{}":
        s = s.replace(ch, "\\" + ch)
    return s


# ---- attention heat maps --------------------------------------------------
ids = tok.encode(PROMPT)
labels = [tex_escape(tok.token_str(i)) for i in ids]
with torch.no_grad():
    model(torch.tensor([ids]))
attn = model.blocks[0].attn.last_attn[0]                  # (h, T, T)
T = len(ids)
c = round(min(4.0, 25.0 / T), 2)                                     # cell size in mm
lines = [r"\begin{tikzpicture}[x=1mm,y=1mm,font=\ttfamily\fontsize{5.6}{6}\selectfont]"]
for n, head in enumerate(HEADS):
    x0 = n * (T * c + 4.0)
    for t in range(T):
        for s in range(T):
            w = attn[head, t, s].item()
            fill = f"cAtt!{round(100 * w)}!white" if s <= t else "cInk!6"
            lines.append(rf"\fill[{fill}] ({x0 + s * c:.1f},{-t * c:.1f}) rectangle ++({c},{-c});")
    lines.append(rf"\draw[cInk!30,line width=0.2pt] ({x0:.1f},0) rectangle ++({T * c},{-T * c});")
    for s, lab in enumerate(labels):
        lines.append(rf"\node[rotate=60,anchor=south west,inner sep=0.5pt] at ({x0 + s * c + 0.6:.1f},0.3) {{{lab}}};")
    if n == 0:
        for t, lab in enumerate(labels):
            lines.append(rf"\node[anchor=east,inner sep=1pt] at (0,{-t * c - c / 2:.1f}) {{{lab}}};")
    lines.append(rf"\node[font=\scriptsize,text=cInk] at ({x0 + T * c / 2:.1f},{-T * c - 2.2:.1f}) {{head {head + 1}}};")
lines.append(r"\end{tikzpicture}")
with open(f"{GEN}/attn_heatmap.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
for head in HEADS:
    print(f"head {head + 1} weights (rows = query):")
    print((attn[head] * 100).round().int().numpy())

# ---- loss curve, sample, headline numbers ---------------------------------
shutil.copy(f"{OUT}/log.csv", f"{GEN}/log.csv")
rows = list(csv.DictReader(open(f"{OUT}/log.csv")))
best = min(rows, key=lambda r: float(r["val_loss"]))
last = rows[-1]

torch.manual_seed(1337)
out = model.generate(torch.tensor([tok.encode("\n")]), max_new_tokens=220, temperature=0.8, top_k=50)
sample = tok.decode(out[0].tolist()).strip("\n").split("\n")
with open(f"{GEN}/sample.txt", "w") as f:
    f.write("\n".join(sample[:9]) + "\n")

with open(f"{GEN}/results.tex", "w") as f:
    f.write(f"\\newcommand{{\\resSteps}}{{{int(last['step']):,}}}\n".replace(",", "{,}"))
    f.write(f"\\newcommand{{\\resMinutes}}{{{float(last['elapsed_s']) / 60:.0f}}}\n")
    f.write(f"\\newcommand{{\\resInit}}{{{float(rows[0]['val_loss']):.2f}}}\n")
    f.write(f"\\newcommand{{\\resTrain}}{{{float(best['train_loss']):.2f}}}\n")
    f.write(f"\\newcommand{{\\resVal}}{{{float(best['val_loss']):.2f}}}\n")
    f.write(f"\\newcommand{{\\resBestStep}}{{{int(best['step']):,}}}\n".replace(",", "{,}"))
    f.write(f"\\newcommand{{\\resPpl}}{{{math.exp(float(best['val_loss'])):.0f}}}\n")
print(open(f"{GEN}/results.tex").read())
print(open(f"{GEN}/sample.txt").read())
