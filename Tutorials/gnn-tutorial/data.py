"""
data.py -- the three graphs used in the tutorial, and the train/val/test node masks.

    toy_graph()      6 nodes, 7 edges: the worked example on the slides
    karate_club()    Zachary's karate club, 34 nodes, 78 edges, 2 factions   [Zachary 1977]
    load_cora()      Cora citation network, 2,708 papers, 7 topics           [Sen et al. 2008]

Every loader returns a dict with
    n        number of nodes
    edges    (E, 2) int array, each undirected edge once with i < j, no self-loops
    X        (n, F) float32 node features
    y        (n,)   int64 node labels
    classes  list of class names
"""
from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ the 6-node worked example
def toy_graph() -> dict:
    """Two triangles joined by one edge.  Slides number the nodes 1..6; arrays are 0-based."""
    edges = np.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5]])
    X = np.array([[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [1, 0]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1])
    return dict(name="toy", n=6, edges=edges, X=X, y=y, classes=["left", "right"])


# ------------------------------------------------------------------ Zachary's karate club
_KARATE_EDGES = """
0-1 0-2 0-3 0-4 0-5 0-6 0-7 0-8 0-10 0-11 0-12 0-13 0-17 0-19 0-21 0-31 1-2 1-3 1-7 1-13 1-17 1-19
1-21 1-30 2-3 2-7 2-8 2-9 2-13 2-27 2-28 2-32 3-7 3-12 3-13 4-6 4-10 5-6 5-10 5-16 6-16 8-30 8-32
8-33 9-33 13-33 14-32 14-33 15-32 15-33 18-32 18-33 19-33 20-32 20-33 22-32 22-33 23-25 23-27 23-29
23-32 23-33 24-25 24-27 24-31 25-31 26-29 26-33 27-33 28-31 28-33 29-32 29-33 30-32 30-33 31-32
31-33 32-33
"""
_KARATE_CLUB = "0000000001000011001010111111111111"     # faction after the split; node 0 = instructor


def karate_club() -> dict:
    """Friendships in a university karate club that later split in two [Zachary 1977].
    Node 0 is the instructor ("Mr. Hi"), node 33 the club president ("Officer").
    There are no node features, so X is the identity: each node gets a one-hot id."""
    edges = np.array([[int(v) for v in e.split("-")] for e in _KARATE_EDGES.split()])
    y = np.array([int(c) for c in _KARATE_CLUB])
    return dict(name="karate", n=34, edges=edges, X=np.eye(34, dtype=np.float32), y=y,
                classes=["Mr. Hi", "Officer"])


# ------------------------------------------------------------------ Cora
def load_cora(path: str = os.path.join(HERE, "data"), row_normalise: bool = True) -> dict:
    """Cora as distributed with github.com/tkipf/pygcn (the original LINQS files).

    cora.content   one line per paper:  <paper id> <1433 binary word indicators> <topic>
    cora.cites     one line per citation:  <cited paper id> <citing paper id>

    Citations are treated as undirected and duplicates are merged: 5,429 lines -> 5,278 edges.
    Features are divided by their row sum, as in Kipf & Welling."""
    ids, feats, labels = [], [], []
    with open(os.path.join(path, "cora.content")) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[0])
            feats.append(np.array(parts[1:-1], dtype=np.float32))
            labels.append(parts[-1])
    X = np.stack(feats)
    if row_normalise:
        X = X / np.maximum(X.sum(1, keepdims=True), 1)
    classes = sorted(set(labels))
    y = np.array([classes.index(c) for c in labels])
    index = {pid: i for i, pid in enumerate(ids)}
    pairs = set()
    with open(os.path.join(path, "cora.cites")) as f:
        for line in f:
            a, b = (index[t] for t in line.split())
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    return dict(name="cora", n=len(ids), edges=np.array(sorted(pairs)), X=X, y=y, classes=classes)


def load_graph(name: str) -> dict:
    return {"toy": toy_graph, "karate": karate_club, "cora": load_cora}[name]()


# ------------------------------------------------------------------ dense helpers (small graphs)
def adjacency(n: int, edges: np.ndarray) -> np.ndarray:
    """Dense symmetric 0/1 adjacency matrix A of an undirected graph."""
    A = np.zeros((n, n))
    A[edges[:, 0], edges[:, 1]] = 1
    A[edges[:, 1], edges[:, 0]] = 1
    return A


# ------------------------------------------------------------------ which nodes carry labels
def planetoid_style_split(y: np.ndarray, seed: int, per_class: int = 20, n_val: int = 500,
                          n_test: int = 1000) -> dict:
    """Random split with the SIZES of Yang et al. (Planetoid): `per_class` labelled training nodes
    per class, then n_val validation and n_test test nodes drawn from the rest.  The remaining
    nodes are unlabelled: the model sees their features and edges but never their labels.
    This is NOT the fixed public Planetoid split; it is a fresh random one for every seed."""
    rng = np.random.default_rng(seed)
    train = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:per_class] for c in np.unique(y)])
    rest = rng.permutation(np.setdiff1d(np.arange(len(y)), train))
    masks = {k: np.zeros(len(y), dtype=bool) for k in ("train", "val", "test")}
    masks["train"][train] = True
    masks["val"][rest[:n_val]] = True
    masks["test"][rest[n_val:n_val + n_test]] = True
    return masks


def karate_split(y: np.ndarray, seed: int = 0) -> dict:
    """The classic setting: only the two leaders are labelled (nodes 0 and 33).  Of the other 32
    members a seeded random 8 are validation nodes (early stopping) and 24 are test nodes."""
    rest = np.random.default_rng(seed).permutation(np.arange(1, 33))
    masks = {k: np.zeros(len(y), dtype=bool) for k in ("train", "val", "test")}
    masks["train"][[0, 33]] = True
    masks["val"][rest[:8]] = True
    masks["test"][rest[8:]] = True
    return masks


def make_split(g: dict, seed: int) -> dict:
    if g["name"] == "karate":
        return karate_split(g["y"], seed)
    if g["name"] == "toy":
        m = {k: np.zeros(6, dtype=bool) for k in ("train", "val", "test")}
        m["train"][[0, 5]], m["val"][[1, 4]], m["test"][[2, 3]] = True, True, True
        return m
    return planetoid_style_split(g["y"], seed)


if __name__ == "__main__":
    for name in ("toy", "karate", "cora"):
        g = load_graph(name)
        deg = np.bincount(g["edges"].ravel(), minlength=g["n"])
        same = (g["y"][g["edges"][:, 0]] == g["y"][g["edges"][:, 1]]).mean()
        m = make_split(g, 0)
        print(f"{name:7s} {g['n']:5d} nodes  {len(g['edges']):5d} edges  {g['X'].shape[1]:5d} features  "
              f"{len(g['classes'])} classes  mean degree {deg.mean():.2f}  max {deg.max():3d}  "
              f"same-class edges {100 * same:.1f}%  split {m['train'].sum()}/{m['val'].sum()}/{m['test'].sum()}")
