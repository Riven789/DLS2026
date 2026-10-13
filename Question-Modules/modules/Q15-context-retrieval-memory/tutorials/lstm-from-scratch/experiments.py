"""
experiments.py -- runs every experiment shown on the slides and writes the results into
slides/generated/ (data files for pgfplots, text samples, and LaTeX macros in results.tex).

    python experiments.py                  # everything: several minutes on 2 CPU cores
    python experiments.py grad clip        # only the named sections

grad         gradient norm vs distance back in time, RNN vs LSTM                (slides 6, 9)
adding       the adding problem at four lengths, RNN vs LSTM vs GRU, 3 seeds    (slide 11)
clip         gradient clipping on/off, simple RNN with SGD                      (slide 12)
sunspots     forecasting vs persistence, seasonal-naive and linear AR           (slides 3, 17)
shakespeare  character-level language model vs bigram counts, samples           (slide 18)
misc         parameter counts                                                   (slide 10)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

from data import CharData, adding_batch, forecast_baselines, load_forecast, rmse_mae
from lstm import SeqModel
from train import char_batches, fit, generate, run_adding, run_charlm, run_forecast, seq_loss

GEN = "slides/generated"
os.makedirs(GEN, exist_ok=True)
CACHE = f"{GEN}/macros.json"
MACROS: dict[str, str] = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
SECTIONS = sys.argv[1:] or ["grad", "adding", "clip", "sunspots", "shakespeare", "misc"]
T0 = time.time()


def macro(name: str, value) -> None:
    MACROS[name] = str(value)
    print(f"  {name} = {value}")


def sci(v: float) -> str:
    """1.3e-07 -> LaTeX 1.3\\times10^{-7}"""
    m, e = f"{v:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def table(path: str, header: str, rows) -> None:
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(" ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r) + "\n")


# ------------------------------------------------------------------ grad: how far back does the gradient reach?
def grad_vs_lag(model: SeqModel, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """Loss at the LAST step only.  Returns ||dL/d(state_t)|| for t = 1..T, where the state is
    h_t for the RNN and (h_t, c_t) for the LSTM, by unrolling one step at a time."""
    state, states = None, []
    for t in range(x.shape[1]):
        out, state = model.rnn(x[:, t:t + 1], state)
        parts = state if isinstance(state, tuple) else (state,)
        for s in parts:
            s.retain_grad()
        states.append(parts)
    seq_loss(model.head(states[-1][0][0]), y).backward()      # read out from h_T, as SeqModel does
    return np.array([math.sqrt(sum(float((s.grad ** 2).sum()) for s in parts if s.grad is not None))
                     for parts in states])


if "grad" in SECTIONS:
    print("grad  gradient norm vs lag (T = 100, n_h = 32, loss at the last step, 5 seeds)")
    T, curves = 100, {}
    x, y = adding_batch(64, T, torch.Generator().manual_seed(123))
    for tag, kind, fb in (("rnn", "rnn", 0.0), ("lstmzero", "lstm", 0.0), ("lstmone", "lstm", 1.0)):
        logs = []
        for seed in range(5):
            torch.manual_seed(seed)
            g = grad_vs_lag(SeqModel(kind, 2, 32, 1, forget_bias=fb), x, y)
            logs.append(np.log10(np.maximum(g / g[-1], 1e-300)))
        curves[tag] = 10 ** np.mean(logs, axis=0)             # geometric mean over seeds, 1 at lag 0
    model, hist, rep = run_adding("lstm", T, seed=0, steps=3000, solved=0.001)
    g = grad_vs_lag(model, x, y)
    curves["lstmtrained"] = g / g[-1]
    macro("gradTrainedMse", f"{rep['test_mse']:.4f}")
    macro("gradTrainedSteps", rep["steps"])
    lags = np.arange(T)[::-1]                                 # lag = T - t: steps between state and loss
    table(f"{GEN}/gradlag.dat", "lag " + " ".join(curves),
          [(int(l), *(max(float(c[i]), 1e-30) for c in curves.values())) for i, l in enumerate(lags)])
    for tag, c in curves.items():
        for lag, name in ((20, "Twenty"), (50, "Fifty"), (99, "End")):
            macro(f"grad{tag.capitalize()}{name}", sci(float(c[T - 1 - lag])))
    with torch.no_grad():                                     # the forget gates of the trained LSTM
        out, state, fs = None, None, []
        for t in range(T):
            h_prev = torch.zeros(64, 32) if state is None else state[0][0]
            z = x[:, t] @ model.rnn.weight_ih_l0.T + h_prev @ model.rnn.weight_hh_l0.T \
                + model.rnn.bias_ih_l0 + model.rnn.bias_hh_l0
            fs.append(torch.sigmoid(z[:, 32:64]))
            out, state = model.rnn(x[:, t:t + 1], state)
        f_all = torch.stack(fs)                               # (T, B, n_h)
        macro("gradTrainedFmax", f"{f_all.mean(dim=(0, 1)).max().item():.3f}")
        macro("gradTrainedFmean", f"{f_all.mean().item():.2f}")

# ------------------------------------------------------------------ adding: does gating matter?
if "adding" in SECTIONS:
    print("adding  RNN vs LSTM vs GRU on the adding problem (n_h = 32, Adam 1e-2, clip 1, <= 2000 steps)")
    LENGTHS, NAMES, SEEDS = (10, 20, 40, 80), ("Ten", "Twenty", "Forty", "Eighty"), range(3)
    results, seed_rows, curve = {}, [], {}
    for kind in ("rnn", "lstm", "gru"):
        for T, name in zip(LENGTHS, NAMES):
            mses, steps = [], []
            for seed in SEEDS:
                _, hist, r = run_adding(kind, T, seed=seed, steps=2000, solved=0.001)
                mses.append(r["test_mse"])
                steps.append(next((h["step"] for h in hist if h["val_loss"] < 0.01), None))
                seed_rows.append((T, kind, seed, r["test_mse"]))
                if T == 80 and seed == 0:
                    curve[kind] = {h["step"]: h["val_loss"] for h in hist}
            results[kind, T] = mses
            solved = sum(m < 0.01 for m in mses)
            done = [s for s in steps if s is not None]
            macro(f"add{kind.capitalize()}Solved{name}", solved)
            macro(f"add{kind.capitalize()}Steps{name}", int(np.median(done)) if len(done) == len(steps) else "--")
            print(f"  {kind:5s} T={T:3d}  test MSE {' '.join(f'{m:.4f}' for m in mses)}  steps to val<0.01: {steps}"
                  f"  [{time.time() - T0:.0f}s]", flush=True)
    macro("addBaseline", f"{r['baseline_mse']:.3f}")
    macro("addSeeds", len(SEEDS))
    table(f"{GEN}/adding.dat", "T rnn lstm gru",
          [(T, *(float(np.exp(np.mean(np.log(results[k, T])))) for k in ("rnn", "lstm", "gru"))) for T in LENGTHS])
    for kind in ("rnn", "lstm", "gru"):
        table(f"{GEN}/adding_seeds_{kind}.dat", "T mse", [(T, m) for T, k, _, m in seed_rows if k == kind])
    steps = sorted(set().union(*[set(c) for c in curve.values()]))
    last = {k: None for k in curve}
    rows = []
    for s in steps:                                           # a run that stopped early keeps its last value
        for k in curve:
            last[k] = curve[k].get(s, last[k])
        rows.append((s, *(float(last[k]) for k in ("rnn", "lstm", "gru"))))
    table(f"{GEN}/adding_curve.dat", "step rnn lstm gru", rows)

    print("  forget-gate bias 0 vs 1 (LSTM, T = 40, 3 seeds)")
    for fb, name in ((0.0, "Zero"), (1.0, "One")):
        steps = []
        for seed in SEEDS:
            _, hist, r = run_adding("lstm", 40, seed=seed, steps=2000, solved=0.001, forget_bias=fb)
            steps.append(next((h["step"] for h in hist if h["val_loss"] < 0.01), 9999))
        macro(f"fb{name}Steps", ", ".join("not solved" if s == 9999 else str(s) for s in steps))
        macro(f"fb{name}Median", "--" if np.median(steps) == 9999 else int(np.median(steps)))

# ------------------------------------------------------------------ clip: exploding gradients and clipping
if "clip" in SECTIONS:
    print("clip  simple RNN (n_h = 128) on Shakespeare characters, SGD lr 2.0, BPTT length 100, 400 steps")
    cd = CharData()
    traces = {}
    for clip, tag in ((0.0, "Off"), (0.5, "On")):
        finals, maxima = [], []
        for seed in range(3):
            torch.manual_seed(seed)
            model = SeqModel("rnn", 32, 128, cd.vocab, vocab=cd.vocab, readout="all")
            trace: list = []
            fit(model, char_batches(cd.train, 32, 100), lambda m: 0.0, 400, 2.0, "sgd", clip, eval_every=400,
                stateful=True, trace=trace)
            finals.append(float(np.mean([t[1] for t in trace[-20:]])))
            maxima.append(max(t[2] for t in trace))
            if seed == 0:
                traces[tag] = trace
        print(f"  clip {clip}: final loss per seed {np.round(finals, 3)}, max grad norm {np.round(maxima, 1)}")
        macro(f"clip{tag}Final", f"{finals[0]:.2f}")
        macro(f"clip{tag}MaxNorm", sci(maxima[0]) if maxima[0] >= 1000 else f"{maxima[0]:.1f}")
        macro(f"clip{tag}Diverged", sum(f > 3.0 for f in finals))
        macro(f"clip{tag}FinalAll", ", ".join(f"{f:.2f}" for f in finals))
    n = min(len(t) for t in traces.values())
    table(f"{GEN}/clip.dat", "step lossoff normoff losson normon",
          [(traces["Off"][i][0], min(traces["Off"][i][1], 99.0), traces["Off"][i][2], traces["On"][i][1],
            traces["On"][i][2]) for i in range(n)])
    macro("clipUniform", f"{math.log(cd.vocab):.2f}")

# ------------------------------------------------------------------ sunspots: forecasting vs baselines
if "sunspots" in SECTIONS:
    print("sunspots  24-month window -> 1 and 6 months ahead; 3 seeds per recurrent model")
    dates, raw = load_forecast()["dates"], load_forecast()["raw"]
    table(f"{GEN}/sunspots.dat", "year value",
          [(int(d[:4]) + (int(d[5:]) - 0.5) / 12, float(v)) for d, v in list(zip(dates, raw))[::3]])
    for h, hname in ((1, "One"), (6, "Six")):
        d = load_forecast(24, h)
        true = d["raw"][d["t_test"]]
        base = forecast_baselines(d, "test")
        for name, tag in (("persistence", "Persist"), ("seasonal-naive", "Seasonal"), ("linear AR", "Linear")):
            rmse, mae = rmse_mae(base[name], true)
            macro(f"sun{tag}Rmse{hname}", f"{rmse:.1f}")
            macro(f"sun{tag}Mae{hname}", f"{mae:.1f}")
        lin_rmse = rmse_mae(base["linear AR"], true)[0]
        for kind in ("rnn", "gru", "lstm"):
            scores = []
            for seed in range(3):
                model, hist, r, _ = run_forecast(kind, 24, h, seed=seed)
                scores.append(r["scores"][kind.upper()])
                if kind == "lstm" and seed == 0:
                    lstm_pred, lstm_steps, lstm_params = r["preds"]["LSTM"], hist[-1]["step"], model.num_params()
            s = np.array(scores)
            print(f"  h={h} {kind:5s} RMSE {np.round(s[:, 0], 2)}  MAE {np.round(s[:, 1], 2)}  (linear AR {lin_rmse:.2f})"
                  f"  [{time.time() - T0:.0f}s]", flush=True)
            tag = kind.capitalize()
            macro(f"sun{tag}Rmse{hname}", f"{s[:, 0].mean():.1f}")
            macro(f"sun{tag}RmseStd{hname}", f"{s[:, 0].std(ddof=1):.1f}")
            macro(f"sun{tag}Mae{hname}", f"{s[:, 1].mean():.1f}")
            macro(f"sun{tag}Wins{hname}", int((s[:, 0] < lin_rmse).sum()))
        if h == 6:
            years = [int(dates[t][:4]) + (int(dates[t][5:]) - 0.5) / 12 for t in d["t_test"]]
            table(f"{GEN}/forecast.dat", "year true lstm linear persistence",
                  [(float(yr), float(a), float(b), float(c), float(e)) for yr, a, b, c, e in
                   zip(years, true, lstm_pred, base["linear AR"], base["persistence"])])
        else:
            macro("sunParams", f"{lstm_params:,}".replace(",", "{,}"))
            macro("sunSteps", lstm_steps)
    c = d["cuts"]
    macro("sunNtrain", c["train"][1])
    macro("sunNval", c["val"][1] - c["val"][0])
    macro("sunNtest", len(true))
    macro("sunTrainEnd", dates[c["train"][1] - 1][:4])
    macro("sunValEnd", dates[c["val"][1] - 1][:4])
    macro("sunTrainEndYear", f"{int(dates[c['train'][1]][:4]) + (int(dates[c['train'][1]][5:]) - 1) / 12:.2f}")
    macro("sunValEndYear", f"{int(dates[c['val'][1]][:4]) + (int(dates[c['val'][1]][5:]) - 1) / 12:.2f}")
    macro("sunMean", f"{d['scaler'].mean:.1f}")
    macro("sunStd", f"{d['scaler'].std:.1f}")
    macro("sunTrainMax", f"{raw[:c['train'][1]].max():.0f}")
    macro("sunTestMax", f"{raw[c['test'][0]:].max():.0f}")

# ------------------------------------------------------------------ shakespeare: character-level language model
if "shakespeare" in SECTIONS:
    print("shakespeare  LSTM (n_h = 256) character model, 1500 steps of truncated BPTT; one run")
    t0 = time.time()
    model, hist, r, cd = run_charlm(verbose=True)
    macro("shkSeconds", f"{hist[-1]['seconds']:.0f}")
    table(f"{GEN}/shk_curve.dat", "step train val", [(h["step"], h["train_loss"], h["val_loss"]) for h in hist])
    best = min(hist, key=lambda h: h["val_loss"])
    macro("shkValLoss", f"{best['val_loss']:.2f}")
    macro("shkValPpl", f"{math.exp(best['val_loss']):.2f}")
    macro("shkTestLoss", f"{r['test_loss']:.2f}")
    macro("shkTestPpl", f"{math.exp(r['test_loss']):.2f}")
    macro("shkTestBits", f"{r['test_loss'] / math.log(2):.2f}")
    for name in ("bigram", "unigram", "uniform"):
        macro(f"shk{name.capitalize()}Loss", f"{r[name]:.2f}")
        macro(f"shk{name.capitalize()}Ppl", f"{math.exp(r[name]):.2f}")
    macro("shkParams", f"{model.num_params():,}".replace(",", "{,}"))
    macro("shkSteps", hist[-1]["step"])
    macro("shkVocab", cd.vocab)
    macro("shkNtrain", f"{len(cd.train):,}".replace(",", "{,}"))
    macro("shkNtest", f"{r['n']:,}".replace(",", "{,}"))
    macro("shkEpochs", f"{hist[-1]['step'] * 32 * 64 / len(cd.train):.1f}")

    def wrap(s: str, lines: int, width: int = 62) -> str:     # keep the slide listing inside its column
        out = []
        for line in s.split("\n"):
            while len(line) > width:
                cut = line.rfind(" ", 0, width)
                cut = cut if cut > 0 else width
                out.append(line[:cut])
                line = line[cut:].lstrip()
            out.append(line)
        return "\n".join(out[:lines]).rstrip()

    for temp, tag in ((0.5, "05"), (1.0, "10")):
        with open(f"{GEN}/sample_{tag}.txt", "w") as f:
            f.write(wrap(generate(model, cd, "ROMEO:", 330, temp, seed=0), lines=7) + "\n")
    gen, ids, bi = torch.Generator().manual_seed(0), [int(cd.stoi["R"])], torch.tensor(r["bigram_table"])
    for _ in range(200):                                      # the bigram baseline also generates text
        ids.append(int(torch.multinomial(bi[ids[-1]], 1, generator=gen)))
    with open(f"{GEN}/sample_bigram.txt", "w") as f:
        f.write(wrap(cd.decode(ids).replace("\n", " / "), lines=2) + "\n")

# ------------------------------------------------------------------ misc: parameter counts
if "misc" in SECTIONS:
    print("misc  parameter counts for n_in = 32, n_h = 256 (the character model)")
    n_in, n_h = 32, 256
    for name, k, mod in (("Rnn", 1, torch.nn.RNN), ("Gru", 3, torch.nn.GRU), ("Lstm", 4, torch.nn.LSTM)):
        macro(f"count{name}Book", f"{k * (n_in + n_h + 1) * n_h:,}".replace(",", "{,}"))
        macro(f"count{name}Torch", f"{sum(p.numel() for p in mod(n_in, n_h).parameters()):,}".replace(",", "{,}"))
    macro("countScratch", f"{SeqModel('scratch', n_in, n_h, 65).rnn.cell.lin.weight.numel() + 4 * n_h:,}".replace(",", "{,}"))

json.dump(MACROS, open(CACHE, "w"), indent=0)
with open(f"{GEN}/results.tex", "w") as f:
    for k, v in MACROS.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(MACROS)} macros to {GEN}/results.tex   [{time.time() - T0:.0f}s]")
