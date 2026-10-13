"""
gcn_numpy.py -- a graph convolutional network (GCN) written out by hand in NumPy:
normalised adjacency, forward pass, masked cross-entropy, backpropagation, gradient descent.

    python gcn_numpy.py        # gradient check, then learn the karate-club factions from 2 labels

Notation (same as the slides):
    A      (n, n) adjacency matrix            A~ = A + I   (add self-loops)
    D~     diagonal matrix of the degrees of A~
    A_hat  = D~^(-1/2) A~ D~^(-1/2)           the normalised adjacency
    H^0 = X,   H^l = ReLU(A_hat H^(l-1) W^l + b^l),   last layer without ReLU -> logits Z
"""
from __future__ import annotations

import numpy as np


def normalized_adjacency(A: np.ndarray, self_loops: bool = True, norm: str = "sym") -> np.ndarray:
    """A_hat = D~^(-1/2) (A + I) D~^(-1/2)   ("sym", the GCN choice)
       or      D~^(-1)   (A + I)             ("mean": every row sums to 1)."""
    A_tilde = A + np.eye(len(A)) if self_loops else A.astype(float)
    d = A_tilde.sum(axis=1)                              # degrees, self-loop included
    if norm == "mean":
        return A_tilde / d[:, None]
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]   # entry (v,u) = 1/sqrt(d_v d_u)


def gcn_layer(A_hat: np.ndarray, H: np.ndarray, W: np.ndarray, b: np.ndarray, relu: bool = True):
    """One GCN layer: mix each node with its neighbours (A_hat @ H), then a shared linear map."""
    Z = A_hat @ H @ W + b
    return np.maximum(Z, 0) if relu else Z


def softmax(Z: np.ndarray) -> np.ndarray:
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


class NumpyGCN:
    """L-layer GCN.  sizes = [F, hidden, ..., K].  Weights are shared by all nodes."""

    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / (m + k)), size=(m, k)) for m, k in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(k) for k in sizes[1:]]

    def forward(self, A_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
        self.A_hat, self.H, self.S = A_hat, [X], []      # caches for the backward pass
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            self.S.append(A_hat @ self.H[-1])            # aggregate: weighted sum over neighbours
            Z = self.S[-1] @ W + b                       # update: shared linear map
            self.H.append(np.maximum(Z, 0) if l < len(self.W) - 1 else Z)
        return self.H[-1]                                # logits, one row per node

    def loss(self, A_hat, X, y, mask) -> float:
        """Mean cross-entropy over the labelled nodes only (mask is a boolean vector)."""
        P = softmax(self.forward(A_hat, X))
        return float(-np.log(P[mask, y[mask]]).mean())

    def backward(self, y: np.ndarray, mask: np.ndarray):
        """Gradients of the masked loss.  Call after forward()/loss()."""
        P = softmax(self.H[-1])
        dZ = np.zeros_like(P)
        dZ[mask] = P[mask]
        dZ[mask, y[mask]] -= 1                           # p - onehot(y) on labelled rows, 0 elsewhere
        dZ /= mask.sum()
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.S[l].T @ dZ                     # dL/dW = (A_hat H)^T dZ
            gb[l] = dZ.sum(axis=0)
            if l > 0:
                dH = self.A_hat.T @ dZ @ self.W[l].T     # the error also flows back along the edges
                dZ = dH * (self.H[l] > 0)                # through the ReLU
        return gW, gb

    def gd_step(self, A_hat, X, y, mask, lr: float) -> float:
        loss = self.loss(A_hat, X, y, mask)
        gW, gb = self.backward(y, mask)
        for l in range(len(self.W)):
            self.W[l] -= lr * gW[l]
            self.b[l] -= lr * gb[l]
        return loss

    def predict(self, A_hat, X) -> np.ndarray:
        return self.forward(A_hat, X).argmax(axis=1)


def gradient_check(net: NumpyGCN, A_hat, X, y, mask, eps: float = 1e-5) -> float:
    """Largest absolute difference between backprop and central finite differences."""
    net.loss(A_hat, X, y, mask)
    gW, gb = net.backward(y, mask)
    worst = 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for P, G in zip(params, grads):
            for idx in np.ndindex(P.shape):
                old = P[idx]
                P[idx] = old + eps
                up = net.loss(A_hat, X, y, mask)
                P[idx] = old - eps
                down = net.loss(A_hat, X, y, mask)
                P[idx] = old
                worst = max(worst, abs((up - down) / (2 * eps) - G[idx]))
    return worst


if __name__ == "__main__":
    from data import adjacency, karate_club, karate_split

    g = karate_club()
    A_hat = normalized_adjacency(adjacency(g["n"], g["edges"]))
    masks = karate_split(g["y"], seed=0)
    net = NumpyGCN([34, 8, 2], seed=0)
    print(f"gradient check (max |backprop - finite difference|): "
          f"{gradient_check(net, A_hat, g['X'], g['y'], masks['train']):.2e}")
    for step in range(1, 201):                           # full-batch gradient descent
        loss = net.gd_step(A_hat, g["X"], g["y"], masks["train"], lr=0.3)
        if step in (1, 10, 50, 100, 200):
            acc = (net.predict(A_hat, g["X"]) == g["y"])[~masks["train"]].mean()
            print(f"step {step:3d} | loss on the 2 labelled nodes {loss:.4f} | "
                  f"accuracy on the other 32 nodes {acc:.3f}")
