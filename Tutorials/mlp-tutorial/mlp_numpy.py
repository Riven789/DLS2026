"""
mlp_numpy.py -- a multilayer perceptron in ~60 lines of NumPy, no autograd.

Forward pass, backpropagation and the SGD update are written out by hand so
each equation on the slides can be read off the code.  Notation:

    a[0] = x                         the input batch,      shape (B, n_0)
    z[l] = a[l-1] @ W[l] + b[l]      pre-activation,       shape (B, n_l)
    a[l] = relu(z[l])                hidden activation
    last layer: no activation; the loss turns z[L] into a prediction.

    python mlp_numpy.py              # gradient check, then learn XOR
"""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class NumpyMLP:
    def __init__(self, sizes: list[int], task: str = "classification", seed: int = 0):
        """sizes = [n_in, hidden..., n_out], e.g. [2, 8, 2]."""
        rng = np.random.default_rng(seed)
        self.task = task
        # He initialisation: std = sqrt(2 / fan_in) keeps ReLU signals at a steady scale
        self.W = [rng.normal(0, np.sqrt(2 / m), (m, n)) for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n) for n in sizes[1:]]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.a, self.z = [x], []
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.a[-1] @ W + b
            self.z.append(z)
            last = l == len(self.W) - 1
            self.a.append(z if last else np.maximum(z, 0))            # ReLU on hidden layers
        return self.a[-1]

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        out = self.forward(x)
        if self.task == "classification":                             # cross-entropy
            return float(-np.log(softmax(out)[np.arange(len(y)), y]).mean())
        return float(((out[:, 0] - y) ** 2).mean())                   # mean squared error

    def backward(self, y: np.ndarray):
        """Gradients of the mean loss for the batch last passed to forward()."""
        B, out = len(y), self.a[-1]
        if self.task == "classification":
            delta = softmax(out)
            delta[np.arange(B), y] -= 1                               # p - onehot(y)
            delta /= B
        else:
            delta = (2 * (out[:, 0] - y) / B)[:, None]                # d MSE / d out
        gW, gb = [], []
        for l in reversed(range(len(self.W))):
            gW.insert(0, self.a[l].T @ delta)                         # dL/dW[l]
            gb.insert(0, delta.sum(axis=0))                           # dL/db[l]
            if l > 0:                                                 # send the error one layer back
                delta = (delta @ self.W[l].T) * (self.z[l - 1] > 0)   # times ReLU'(z)
        return gW, gb

    def sgd_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        loss = self.loss(x, y)
        gW, gb = self.backward(y)
        for l in range(len(self.W)):
            self.W[l] -= lr * gW[l]                                   # theta <- theta - lr * grad
            self.b[l] -= lr * gb[l]
        return loss


def gradient_check(net: NumpyMLP, x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Largest gap between backprop and (loss(w+eps) - loss(w-eps)) / 2eps over all weights."""
    net.loss(x, y)
    gW, gb = net.backward(y)
    worst = 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for p, g in zip(params, grads):
            for i in np.ndindex(p.shape):
                old = p[i]
                p[i] = old + eps
                up = net.loss(x, y)
                p[i] = old - eps
                down = net.loss(x, y)
                p[i] = old
                worst = max(worst, abs((up - down) / (2 * eps) - g[i]))
    return worst


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for task, y in (("classification", rng.integers(0, 3, 5)), ("regression", rng.normal(size=5))):
        net = NumpyMLP([4, 6, 5, 3 if task == "classification" else 1], task, seed=1)
        print(f"gradient check ({task}): max |backprop - finite difference| = "
              f"{gradient_check(net, rng.normal(size=(5, 4)), y):.2e}")

    # XOR: the classic problem a single neuron cannot solve but one hidden layer can.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    net = NumpyMLP([2, 8, 2], seed=0)
    for step in range(2001):
        loss = net.sgd_step(X, y, lr=0.5)
        if step % 500 == 0:
            print(f"XOR step {step:4d}  loss {loss:.4f}  predictions {net.forward(X).argmax(1)}")
