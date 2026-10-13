"""
data.py -- turning sequences into tensors, for the three tasks of the tutorial.

    adding       synthetic long-range task: add the two marked numbers of a length-T sequence
    sunspots     monthly sunspot numbers 1749-1983: sliding windows, chronological split, scaling
    shakespeare  characters -> integer ids, contiguous train/val/test split, parallel streams

plus the baselines every result is compared with (persistence, seasonal-naive,
linear autoregression, unigram and bigram character counts).
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ (a) the adding problem
def adding_batch(B: int, T: int, gen: torch.Generator):
    """x: (B, T, 2).  Channel 0 holds random numbers in [0, 1); channel 1 is 0 except for two 1s,
    one in each half of the sequence.  Target: the sum of the two marked numbers.
    Predicting the mean (1.0) gives MSE 1/6 = 0.167; anything better requires remembering."""
    x = torch.zeros(B, T, 2)
    x[:, :, 0] = torch.rand(B, T, generator=gen)
    rows = torch.arange(B)
    first = torch.randint(0, T // 2, (B,), generator=gen)
    second = torch.randint(T // 2, T, (B,), generator=gen)
    x[rows, first, 1] = 1.0
    x[rows, second, 1] = 1.0
    return x, x[rows, first, 0] + x[rows, second, 0]


# ------------------------------------------------------------------ (b) time-series forecasting
def load_series(name: str = "monthly-sunspots.csv"):
    with open(os.path.join(HERE, "data", name)) as f:
        rows = list(csv.reader(f))[1:]
    return [r[0] for r in rows], np.array([float(r[1]) for r in rows])


class Scaler:
    """Standardise with the mean and standard deviation of the TRAINING period only."""

    def fit(self, v: np.ndarray) -> "Scaler":
        self.mean, self.std = float(v.mean()), float(v.std())
        return self

    def transform(self, v):
        return (v - self.mean) / self.std

    def inverse(self, v):
        return v * self.std + self.mean


def make_windows(series: np.ndarray, window: int, horizon: int, lo: int, hi: int):
    """Every (input window, target) pair whose TARGET index t lies in [lo, hi).
    Input: the `window` values ending `horizon` steps before t.  Returns X (N, window), y (N,), t (N,)."""
    ts = np.arange(max(lo, window + horizon - 1), hi)
    X = np.stack([series[t - horizon - window + 1:t - horizon + 1] for t in ts])
    return X, series[ts], ts


def load_forecast(window: int = 24, horizon: int = 1, name: str = "monthly-sunspots.csv",
                  fractions=(0.70, 0.15)) -> dict:
    """Chronological split: the first 70% of the months train, the next 15% validate, the last 15% test.
    A validation or test window may reach back into earlier months for its inputs (that is the
    past, which a forecaster is allowed to see); its target never lies in an earlier split."""
    dates, raw = load_series(name)
    n = len(raw)
    n_train, n_val = int(fractions[0] * n), int(fractions[1] * n)
    cuts = dict(train=(0, n_train), val=(n_train, n_train + n_val), test=(n_train + n_val, n))
    scaler = Scaler().fit(raw[:n_train])                       # statistics from the training months only
    z = scaler.transform(raw)
    d = dict(task="forecast", raw=raw, dates=dates, scaler=scaler, window=window, horizon=horizon, cuts=cuts)
    for split, (lo, hi) in cuts.items():
        X, y, t = make_windows(z, window, horizon, lo, hi)
        d["X_" + split] = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)    # (N, window, 1)
        d["y_" + split] = torch.tensor(y, dtype=torch.float32)
        d["t_" + split] = t
    return d


def forecast_baselines(d: dict, split: str = "test", season: int = 132) -> dict:
    """Predictions in original units, for the target months of `split`.
    persistence     the last observed value
    seasonal-naive  the value one season earlier (132 months: the mean length of the solar cycle)
    linear AR       least-squares linear map from the same window to the target, fitted on train"""
    raw, t, h = d["raw"], d["t_" + split], d["horizon"]
    lag = season * math.ceil(h / season)                       # most recent same-phase month already observed
    A = np.c_[d["X_train"][:, :, 0].numpy(), np.ones(len(d["X_train"]))]
    coef, *_ = np.linalg.lstsq(A, d["y_train"].numpy(), rcond=None)
    X = d["X_" + split][:, :, 0].numpy()
    return {"persistence": raw[t - h], "seasonal-naive": raw[t - lag],
            "linear AR": d["scaler"].inverse(np.c_[X, np.ones(len(X))] @ coef)}


def rmse_mae(pred: np.ndarray, true: np.ndarray):
    return float(np.sqrt(((pred - true) ** 2).mean())), float(np.abs(pred - true).mean())


# ------------------------------------------------------------------ (c) character-level language modelling
class CharData:
    """Tiny Shakespeare as integer ids.  Split by position (first 90% train, next 5% val, last 5% test):
    text is a sequence too, so no shuffling before the split."""

    def __init__(self, name: str = "tinyshakespeare.txt"):
        with open(os.path.join(HERE, "data", name)) as f:
            self.text = f.read()
        self.chars = sorted(set(self.text))                    # vocabulary: 65 distinct characters
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        ids = torch.tensor([self.stoi[ch] for ch in self.text])
        n = len(ids)
        self.train, self.val, self.test = ids[:int(0.9 * n)], ids[int(0.9 * n):int(0.95 * n)], ids[int(0.95 * n):]

    @property
    def vocab(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[ch] for ch in s])

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


def stream_batches(ids: torch.Tensor, B: int, T: int):
    """Cut `ids` into B parallel streams and walk along them T characters at a time.
    Yields (x, y, first): x, y are (B, T) with y the next character after each x;
    batch k+1 continues exactly where batch k stopped, so the hidden state can be carried over."""
    n = (len(ids) - 1) // B                                    # characters per stream
    x = ids[:B * n].view(B, n)
    y = ids[1:B * n + 1].view(B, n)
    for s in range(0, n - 1, T):
        yield x[:, s:s + T], y[:, s:s + T], s == 0


def ngram_baselines(cd: CharData, split: str = "test") -> dict:
    """Mean negative log-likelihood (nats per character) of count-based models with add-one smoothing."""
    V, tr, ev = cd.vocab, cd.train.numpy(), getattr(cd, split).numpy()
    uni = np.bincount(tr, minlength=V) + 1.0
    uni /= uni.sum()
    bi = np.ones((V, V))
    np.add.at(bi, (tr[:-1], tr[1:]), 1.0)                      # bi[a, b] = 1 + count of "a then b"
    bi /= bi.sum(axis=1, keepdims=True)
    return {"uniform": math.log(V), "unigram": float(-np.log(uni[ev[1:]]).mean()),
            "bigram": float(-np.log(bi[ev[:-1], ev[1:]]).mean()), "bigram_table": bi}
