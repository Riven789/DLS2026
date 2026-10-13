"""
train.py -- training and testing harness for the recurrent models of the tutorial.

    python train.py --task adding --seq_len 50                # long-range memory: LSTM
    python train.py --task adding --seq_len 50 --model rnn    # ... and the simple RNN that fails
    python train.py --task sunspots                           # forecasting, with three baselines
    python train.py --task shakespeare                        # character-level language model (minutes)
    python train.py --task shakespeare --model gru --hidden 128 --steps 500 --temperature 0.5 1.0

Every task uses the same loop (`fit`): forward through time, loss, backpropagation through
time, gradient clipping, update; the validation split picks the checkpoint, the test split
is used once at the end and is always reported next to simple baselines.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import (CharData, adding_batch, forecast_baselines, load_forecast, ngram_baselines, rmse_mae,
                  stream_batches)
from lstm import SeqModel, detach

torch.set_num_threads(1)       # the models are tiny: extra threads only add overhead on a shared CPU

DEFAULTS = {   # per-task defaults for flags left unset
    "adding": dict(hidden=32, seq_len=50, batch_size=32, lr=1e-2, steps=2000, eval_every=100, patience=0),
    "sunspots": dict(hidden=32, seq_len=24, batch_size=64, lr=2e-3, steps=3000, eval_every=50, patience=10),
    "shakespeare": dict(hidden=256, seq_len=64, batch_size=32, lr=3e-3, steps=1500, eval_every=250,
                        patience=0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(DEFAULTS), default="sunspots")
    p.add_argument("--model", choices=["lstm", "rnn", "gru", "scratch"], default="lstm")
    p.add_argument("--hidden", type=int, help="size of the hidden (and cell) state")
    p.add_argument("--layers", type=int, default=1, help="stacked recurrent layers")
    p.add_argument("--bidirectional", action="store_true", help="adding/sunspots only (not for generation)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--embed", type=int, default=32, help="shakespeare: embedding width")
    p.add_argument("--seq_len", type=int, help="adding: T; sunspots: window; shakespeare: BPTT length")
    p.add_argument("--horizon", type=int, default=1, help="sunspots: months ahead to forecast")
    p.add_argument("--forget_bias", type=float, default=1.0, help="initial LSTM forget-gate bias")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, help="learning rate")
    p.add_argument("--clip", type=float, default=1.0, help="clip the gradient norm to this (0 = off)")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--steps", type=int, help="maximum number of weight updates")
    p.add_argument("--eval_every", type=int, help="validate every this many steps")
    p.add_argument("--patience", type=int,
                   help="stop after this many evaluations without improvement (0 = never)")
    p.add_argument("--temperature", type=float, nargs="*", default=[0.5, 1.0], help="shakespeare: sampling")
    p.add_argument("--sample_chars", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    for k, v in DEFAULTS[args.task].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------- the loop
def seq_loss(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if y.dtype == torch.long:                                 # next-token prediction: (B, T, V) vs (B, T)
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))
    return F.mse_loss(out.squeeze(-1), y)                     # regression: (B, 1) vs (B,)


def fit(model: SeqModel, batches, val_fn, steps: int, lr: float = 1e-3, optimizer: str = "adam",
        clip: float = 1.0, eval_every: int = 100, patience: int = 0, stateful: bool = False,
        solved: float = 0.0, verbose: bool = False, trace: list | None = None) -> list[dict]:
    """Train `model` on the iterator `batches` (yielding x, y, first).  One history row per evaluation.
    stateful: carry the hidden state from one batch to the next (truncated BPTT on a long stream).
    The weights with the best validation loss are restored at the end."""
    opt = (torch.optim.Adam if optimizer == "adam" else torch.optim.SGD)(model.parameters(), lr=lr)
    history, best_val, best_state, bad, t0 = [], float("inf"), None, 0, time.time()
    state, losses, norms = None, [], []
    for step in range(1, steps + 1):
        x, y, first = next(batches)
        state = detach(state) if stateful and not first else None   # keep the values, cut the graph
        model.train()
        out, state = model(x, state)                          # forward through all T steps
        loss = seq_loss(out, y)
        opt.zero_grad()
        loss.backward()                                       # backpropagation through time
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else float("inf"))
        opt.step()
        losses.append(loss.item())
        norms.append(norm.item())                             # gradient norm BEFORE clipping
        if trace is not None:
            trace.append((step, losses[-1], norms[-1]))

        if step % eval_every == 0 or step == steps or not math.isfinite(losses[-1]):
            row = dict(step=step, train_loss=float(np.mean(losses)), val_loss=val_fn(model),
                       grad_norm=float(np.max(norms)), seconds=time.time() - t0)
            history.append(row)
            losses, norms = [], []
            if verbose:
                print(f"step {step:5d} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} | "
                      f"max grad norm {row['grad_norm']:.2f} | {row['seconds']:.0f}s", flush=True)
            if not math.isfinite(row["train_loss"]):          # diverged
                break
            if row["val_loss"] < best_val:
                best_val, bad, best_state = row["val_loss"], 0, copy.deepcopy(model.state_dict())
            else:
                bad += 1
            if (patience and bad >= patience) or best_val < solved:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def dataset_loss(model: SeqModel, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()                                              # switches dropout off
    return seq_loss(model(X)[0], y).item()


@torch.no_grad()
def stream_loss(model: SeqModel, ids: torch.Tensor, B: int = 16, T: int = 200) -> float:
    """Mean cross-entropy (nats per character) along a text, carrying the state across chunks."""
    model.eval()
    total, count, state = 0.0, 0, None
    for x, y, _ in stream_batches(ids, B, T):
        out, state = model(x, state)
        total += F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1), reduction="sum").item()
        count += y.numel()
    return total / count


# ---------------------------------------------------------------------------- task (a): adding
def adding_batches(B: int, T: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    while True:                                               # fresh random sequences forever
        yield (*adding_batch(B, T, gen), True)


def run_adding(kind: str = "lstm", T: int = 50, hidden: int = 32, seed: int = 0, steps: int = 2000,
               lr: float = 1e-2, batch_size: int = 32, clip: float = 1.0, eval_every: int = 100,
               patience: int = 0, forget_bias: float = 1.0, solved: float = 0.0, verbose: bool = False, **kw):
    torch.manual_seed(seed)
    model = SeqModel(kind, 2, hidden, 1, forget_bias=forget_bias, **kw)
    Xv, yv = adding_batch(1000, T, torch.Generator().manual_seed(10_000 + seed))    # fixed validation set
    Xt, yt = adding_batch(2000, T, torch.Generator().manual_seed(20_000 + seed))    # fixed test set
    history = fit(model, adding_batches(batch_size, T, seed), lambda m: dataset_loss(m, Xv, yv), steps, lr,
                  clip=clip, eval_every=eval_every, patience=patience, solved=solved, verbose=verbose)
    report = dict(test_mse=dataset_loss(model, Xt, yt), baseline_mse=float(((yt - 1.0) ** 2).mean()),
                  steps=history[-1]["step"], n=len(yt))
    return model, history, report


# ---------------------------------------------------------------------------- task (b): forecasting
def window_batches(X: torch.Tensor, y: torch.Tensor, B: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    while True:                                               # one pass of this loop = one epoch
        perm = torch.randperm(len(X), generator=gen)          # shuffling WINDOWS inside the train period is fine
        for s in range(0, len(X), B):
            idx = perm[s:s + B]
            yield X[idx], y[idx], True


@torch.no_grad()
def forecast(model: SeqModel, d: dict, split: str = "test") -> np.ndarray:
    model.eval()
    return d["scaler"].inverse(model(d["X_" + split])[0].squeeze(-1).numpy())      # back to sunspot numbers


def run_forecast(kind: str = "lstm", window: int = 24, horizon: int = 1, hidden: int = 32, seed: int = 0,
                 steps: int = 3000, lr: float = 2e-3, batch_size: int = 64, clip: float = 1.0,
                 eval_every: int = 50, patience: int = 10, verbose: bool = False, **kw):
    torch.manual_seed(seed)
    d = load_forecast(window, horizon)
    model = SeqModel(kind, 1, hidden, 1, **kw)
    history = fit(model, window_batches(d["X_train"], d["y_train"], batch_size, seed),
                  lambda m: dataset_loss(m, d["X_val"], d["y_val"]), steps, lr, clip=clip,
                  eval_every=eval_every, patience=patience, verbose=verbose)
    true = d["raw"][d["t_test"]]
    preds = {kind.upper(): forecast(model, d), **forecast_baselines(d, "test")}
    report = dict(true=true, preds=preds, scores={k: rmse_mae(v, true) for k, v in preds.items()}, n=len(true))
    return model, history, report, d


# ---------------------------------------------------------------------------- task (c): characters
def char_batches(ids: torch.Tensor, B: int, T: int):
    while True:                                               # one pass of this loop = one epoch
        yield from stream_batches(ids, B, T)


@torch.no_grad()
def generate(model: SeqModel, cd: CharData, prompt: str, n: int, temperature: float = 1.0,
             seed: int = 0) -> str:
    """Feed the prompt, then repeatedly sample one character and feed it back in."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    out, state = model(cd.encode(prompt)[None])               # logits (1, len(prompt), V) and the state
    ids = []
    for _ in range(n):
        p = torch.softmax(out[0, -1] / temperature, dim=-1)   # temperature < 1 sharpens, > 1 flattens
        nxt = torch.multinomial(p, 1, generator=gen)          # sample one id
        ids.append(nxt.item())
        out, state = model(nxt[None], state)                  # one step: only the state carries the past
    return prompt + cd.decode(ids)


def run_charlm(kind: str = "lstm", hidden: int = 256, embed: int = 32, seq_len: int = 64, seed: int = 0,
               steps: int = 1500, lr: float = 3e-3, batch_size: int = 32, clip: float = 1.0,
               eval_every: int = 250, patience: int = 0, optimizer: str = "adam", verbose: bool = False, **kw):
    torch.manual_seed(seed)
    cd = CharData()
    model = SeqModel(kind, embed, hidden, cd.vocab, vocab=cd.vocab, readout="all", **kw)
    history = fit(model, char_batches(cd.train, batch_size, seq_len), lambda m: stream_loss(m, cd.val), steps,
                  lr, optimizer, clip, eval_every, patience, stateful=True, verbose=verbose)
    report = dict(test_loss=stream_loss(model, cd.test), n=len(cd.test) - 1, **ngram_baselines(cd, "test"))
    return model, history, report, cd


# ---------------------------------------------------------------------------- command line
def main() -> None:
    a = parse_args()
    out_dir = a.out_dir or os.path.join("out", a.task)
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(kind=a.model, hidden=a.hidden, seed=a.seed, steps=a.steps, lr=a.lr, batch_size=a.batch_size,
              clip=a.clip, eval_every=a.eval_every, patience=a.patience, verbose=True, layers=a.layers,
              dropout=a.dropout, forget_bias=a.forget_bias)
    print(f"task {a.task} | model {a.model}, hidden {a.hidden}, layers {a.layers} | {a.optimizer}, lr {a.lr}, "
          f"batch {a.batch_size}, clip {a.clip}, seq_len {a.seq_len}")

    if a.task == "adding":
        model, history, r = run_adding(T=a.seq_len, bidirectional=a.bidirectional, **kw)
        print(f"\nTEST SET ({r['n']} fresh sequences of length {a.seq_len})")
        print(f"  MSE {r['test_mse']:.4f}   (always predicting 1.0: {r['baseline_mse']:.4f})")
    elif a.task == "sunspots":
        model, history, r, d = run_forecast(window=a.seq_len, horizon=a.horizon,
                                            bidirectional=a.bidirectional, **kw)
        lo, hi = d["cuts"]["test"]
        print(f"\nsplit by time: train {d['dates'][0]}..{d['dates'][d['cuts']['train'][1] - 1]}, "
              f"val ..{d['dates'][lo - 1]}, test {d['dates'][lo]}..{d['dates'][hi - 1]}")
        print(f"TEST SET ({r['n']} months never used for training or tuning; {a.horizon}-month-ahead forecast)")
        for name, (rmse, mae) in r["scores"].items():
            print(f"  {name:15s} RMSE {rmse:6.2f}   MAE {mae:6.2f}")
    else:
        model, history, r, cd = run_charlm(embed=a.embed, seq_len=a.seq_len, optimizer=a.optimizer, **kw)
        print(f"\nTEST SET ({r['n']:,} characters never used for training or tuning), nats per character")
        print(f"  {a.model.upper():8s} loss {r['test_loss']:.3f}   perplexity {math.exp(r['test_loss']):.2f}")
        for name in ("bigram", "unigram", "uniform"):
            print(f"  {name:8s} loss {r[name]:.3f}   perplexity {math.exp(r[name]):.2f}")
        for temp in a.temperature:
            print(f"\n--- sample, temperature {temp} ---")
            print(generate(model, cd, "ROMEO:", a.sample_chars, temp, a.seed))

    print(f"\nmodel parameters: {model.num_params():,}")
    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(out_dir, "model.pt"))
    print(f"log -> {out_dir}/log.csv   weights -> {out_dir}/model.pt")


if __name__ == "__main__":
    main()
