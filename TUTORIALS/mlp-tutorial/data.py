"""
data.py -- from a raw table to the numbers a network can train on.

A multilayer perceptron only understands vectors of real numbers of similar
scale.  This file does the translation, with nothing but the csv module and
NumPy so every step is visible:

    numerical column  -> (optional log) -> standardise to mean 0, std 1
                         missing value  -> 0 (the mean) plus a 0/1 "was missing" flag
    logical column    -> False/True -> 0/1
    categorical column-> one-hot (one 0/1 column per category seen in training)
    class label       -> integer id 0..K-1          (classification)
    numeric target    -> standardised               (regression)

Statistics (means, stds, category lists) are computed on the TRAINING split
only and then reused unchanged for validation and test data.
"""
from __future__ import annotations

import csv
import math

import numpy as np

# Which column plays which role.  This is the one thing you must decide by
# looking at your data; everything else is mechanical.
SCHEMAS = {
    "titanic": dict(                      # will this passenger survive?  (classification)
        path="data/titanic.csv", task="classification", target="alive",
        numerical=["age", "fare", "sibsp", "parch"], log=["fare"],
        logical=["adult_male", "alone"],
        categorical=["sex", "class", "embark_town"]),
    "mpg": dict(                          # fuel economy of a car in miles/gallon  (regression)
        path="data/mpg.csv", task="regression", target="mpg",
        numerical=["cylinders", "displacement", "horsepower", "weight",
                   "acceleration", "model_year"], log=[],
        logical=[],
        categorical=["origin"]),
}


def read_csv(path: str) -> list[dict]:
    """Rows as dictionaries of strings; an empty string means 'missing'."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TabularEncoder:
    """Learns the encoding on training rows (fit), applies it to any rows (transform)."""

    def __init__(self, numerical=(), logical=(), categorical=(), log=()):
        self.numerical, self.logical = list(numerical), list(logical)
        self.categorical, self.log = list(categorical), set(log)
        self.stats: dict[str, tuple[float, float]] = {}      # column -> (mean, std)
        self.has_missing: dict[str, bool] = {}
        self.categories: dict[str, list[str]] = {}
        self.feature_names: list[str] = []

    def _number(self, col: str, raw: str) -> float:
        if raw == "":
            return math.nan
        v = float(raw)
        return math.log1p(v) if col in self.log else v       # log tames long right tails

    def fit(self, rows: list[dict]) -> "TabularEncoder":
        self.feature_names = []
        for col in self.numerical:
            v = np.array([self._number(col, r[col]) for r in rows])
            self.stats[col] = (float(np.nanmean(v)), float(np.nanstd(v)) or 1.0)
            self.has_missing[col] = bool(np.isnan(v).any())
            self.feature_names.append(col)
            if self.has_missing[col]:
                self.feature_names.append(col + "_missing")
        self.feature_names += self.logical
        for col in self.categorical:
            self.categories[col] = sorted({r[col] for r in rows if r[col] != ""})
            self.feature_names += [f"{col}={c}" for c in self.categories[col]]
        return self

    def transform(self, rows: list[dict]) -> np.ndarray:
        X = np.zeros((len(rows), len(self.feature_names)), dtype=np.float32)
        for i, r in enumerate(rows):
            j = 0
            for col in self.numerical:
                v = self._number(col, r[col])
                mean, std = self.stats[col]
                X[i, j] = 0.0 if math.isnan(v) else (v - mean) / std
                j += 1
                if self.has_missing[col]:
                    X[i, j] = float(math.isnan(v))
                    j += 1
            for col in self.logical:
                X[i, j] = float(r[col].strip().lower() in ("true", "1", "yes"))
                j += 1
            for col in self.categorical:
                cats = self.categories[col]
                if r[col] in cats:                             # unseen/missing -> all zeros
                    X[i, j + cats.index(r[col])] = 1.0
                j += len(cats)
        return X


class LabelEncoder:
    """Class names <-> integer ids.  'no','yes' -> 0,1."""

    def fit(self, labels: list[str]) -> "LabelEncoder":
        self.classes = sorted(set(labels))
        self.index = {c: i for i, c in enumerate(self.classes)}
        return self

    def transform(self, labels: list[str]) -> np.ndarray:
        return np.array([self.index[c] for c in labels], dtype=np.int64)


class TargetScaler:
    """Standardise a regression target; inverse() returns to the original units."""

    def fit(self, y: np.ndarray) -> "TargetScaler":
        self.mean, self.std = float(y.mean()), float(y.std())
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean


def split_rows(rows: list, seed: int, frac=(0.70, 0.15, 0.15)):
    """Shuffle once, then cut into train / validation / test."""
    order = np.random.default_rng(seed).permutation(len(rows))
    n_train, n_val = int(frac[0] * len(rows)), int(frac[1] * len(rows))
    pick = lambda idx: [rows[i] for i in idx]
    return pick(order[:n_train]), pick(order[n_train:n_train + n_val]), pick(order[n_train + n_val:])


def make_moons(n: int, noise: float, seed: int):
    """Two interleaved half-circles: a 2-D problem no straight line can solve."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, math.pi, n)
    y = rng.integers(0, 2, n)
    x = np.where(y == 0, np.cos(t), 1 - np.cos(t))
    z = np.where(y == 0, np.sin(t), 0.5 - np.sin(t))
    X = np.stack([x, z], axis=1) + rng.normal(0, noise, (n, 2))
    return X.astype(np.float32), y.astype(np.int64)


def load_task(name: str, seed: int = 0) -> dict:
    """Everything the harness needs: encoded splits plus the fitted encoders."""
    if name == "moons":
        X, y = make_moons(600, 0.20, seed)
        cut = (420, 510)
        d = dict(task="classification", n_out=2, classes=["0", "1"], feature_names=["x1", "x2"])
        for split, sl in (("train", slice(0, cut[0])), ("val", slice(*cut)), ("test", slice(cut[1], None))):
            d["X_" + split], d["y_" + split] = X[sl], y[sl]
        return d

    s = SCHEMAS[name]
    rows = [r for r in read_csv(s["path"]) if r[s["target"]] != ""]
    train, val, test = split_rows(rows, seed)
    enc = TabularEncoder(s["numerical"], s["logical"], s["categorical"], s["log"]).fit(train)
    d = dict(task=s["task"], encoder=enc, feature_names=enc.feature_names,
             rows_train=train, rows_test=test)
    target = lambda part: [r[s["target"]] for r in part]
    if s["task"] == "classification":
        lab = LabelEncoder().fit(target(train))
        d.update(n_out=len(lab.classes), classes=lab.classes, label_encoder=lab)
        encode_y = lab.transform
    else:
        sc = TargetScaler().fit(np.array(target(train), dtype=np.float64))
        d.update(n_out=1, scaler=sc)
        encode_y = lambda t: sc.transform(np.array(t, dtype=np.float64))
    for split, part in (("train", train), ("val", val), ("test", test)):
        d["X_" + split], d["y_" + split] = enc.transform(part), encode_y(target(part))
    return d


def batches(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator | None):
    """Yield mini-batches.  With an rng the rows are reshuffled first (do this every epoch)."""
    order = rng.permutation(len(X)) if rng is not None else np.arange(len(X))
    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    d = load_task("titanic", seed=0)
    enc, row = d["encoder"], d["rows_train"][0]
    print("raw row      :", {k: row[k] for k in enc.numerical + enc.logical + enc.categorical}, "->", row["alive"])
    print("feature names:", enc.feature_names)
    print("encoded x    :", d["X_train"][0])
    print("label id     :", d["y_train"][0], " classes:", d["classes"])
    for col in enc.numerical:
        print(f"  {col:6s} mean {enc.stats[col][0]:7.3f}  std {enc.stats[col][1]:6.3f}"
              f"  missing in train: {enc.has_missing[col]}" + ("   (after log1p)" if col in enc.log else ""))
    print("shapes       :", {k: v.shape for k, v in d.items() if k.startswith(("X_", "y_"))})
