"""
conv_numpy.py -- the convolution layer from scratch in NumPy: forward pass, backward pass, pooling.

    python conv_numpy.py      # checks against torch.nn.functional and against finite differences

Shapes follow PyTorch:  x (N, C_in, H, W)   w (C_out, C_in, k, k)   b (C_out,)   out (N, C_out, H_out, W_out)
with  H_out = floor((H + 2p - k) / s) + 1.

"Convolution" in deep learning is really cross-correlation: the kernel is NOT flipped.
"""
from __future__ import annotations

import numpy as np


def out_size(n: int, k: int, s: int = 1, p: int = 0) -> int:
    """Output length along one axis: input n, kernel k, stride s, padding p."""
    return (n + 2 * p - k) // s + 1


def conv2d(x: np.ndarray, w: np.ndarray, b=None, stride: int = 1, padding: int = 0) -> np.ndarray:
    """y[n, o, i, j] = b[o] + sum_{c, u, v} w[o, c, u, v] * x_padded[n, c, i*s + u, j*s + v]"""
    N, C, H, W = x.shape
    Co, Ci, k, _ = w.shape
    assert C == Ci, "kernel depth must equal the number of input channels"
    s, p = stride, padding
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))              # zeros around the border
    Ho, Wo = out_size(H, k, s, p), out_size(W, k, s, p)
    y = np.zeros((N, Co, Ho, Wo), dtype=x.dtype)
    for i in range(Ho):                                           # slide the window over the image
        for j in range(Wo):
            patch = xp[:, :, i * s:i * s + k, j * s:j * s + k]    # (N, C_in, k, k)
            y[:, :, i, j] = np.einsum("nckl,ockl->no", patch, w)  # one dot product per filter
    return y if b is None else y + b[None, :, None, None]


def conv2d_backward(dy: np.ndarray, x: np.ndarray, w: np.ndarray, stride: int = 1, padding: int = 0):
    """Given dy = dL/dy, return (dL/dx, dL/dw, dL/db).  Every forward product w * x_patch sends
    gradient dy * x_patch to the kernel and dy * w back to the patch."""
    k, s, p = w.shape[2], stride, padding
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    dxp, dw = np.zeros_like(xp), np.zeros_like(w)
    for i in range(dy.shape[2]):
        for j in range(dy.shape[3]):
            patch = xp[:, :, i * s:i * s + k, j * s:j * s + k]
            dw += np.einsum("no,nckl->ockl", dy[:, :, i, j], patch)   # shared kernel: every position adds its vote
            dxp[:, :, i * s:i * s + k, j * s:j * s + k] += np.einsum("no,ockl->nckl", dy[:, :, i, j], w)
    dx = dxp[:, :, p:p + x.shape[2], p:p + x.shape[3]]            # drop the gradient of the padding
    return dx, dw, dy.sum(axis=(0, 2, 3))


def conv2d_windows(x: np.ndarray, w: np.ndarray, b=None, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Same result without Python loops: view all windows at once, then one tensor product."""
    p, k = padding, w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::stride, ::stride]
    y = np.tensordot(win, w, axes=([1, 4, 5], [1, 2, 3]))        # win (N, C_in, H_out, W_out, k, k)
    y = y.transpose(0, 3, 1, 2)                                   # (N, H_out, W_out, C_out) -> (N, C_out, H_out, W_out)
    return y if b is None else y + b[None, :, None, None]


def pool2d(x: np.ndarray, k: int = 2, mode: str = "max") -> np.ndarray:
    """Non-overlapping k x k pooling (stride k).  Rows/columns that do not fill a window are dropped."""
    N, C, H, W = x.shape
    blocks = x[:, :, :H // k * k, :W // k * k].reshape(N, C, H // k, k, W // k, k)
    return blocks.max(axis=(3, 5)) if mode == "max" else blocks.mean(axis=(3, 5))


def max_pool2d_backward(dy: np.ndarray, x: np.ndarray, k: int = 2) -> np.ndarray:
    """The gradient goes only to the winner of each window (ties: split equally)."""
    N, C, H, W = x.shape
    blocks = x[:, :, :H // k * k, :W // k * k].reshape(N, C, H // k, k, W // k, k)
    mask = blocks == blocks.max(axis=(3, 5), keepdims=True)
    mask = mask / mask.sum(axis=(3, 5), keepdims=True)
    dx = np.zeros_like(x)
    dx[:, :, :H // k * k, :W // k * k] = (mask * dy[:, :, :, None, :, None]).reshape(N, C, H // k * k, W // k * k)
    return dx


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cnn_forward(params: dict, x: np.ndarray) -> np.ndarray:
    """The whole network of cnn.py (default settings) in NumPy, from a PyTorch state_dict.
    conv 5x5 pad 2 -> ReLU -> pool -> conv 5x5 -> ReLU -> pool -> flatten -> linear -> ReLU -> linear."""
    P = {name: np.asarray(v, dtype=np.float64) for name, v in params.items()}
    h = pool2d(relu(conv2d(x, P["features.0.weight"], P["features.0.bias"], padding=2)))
    h = pool2d(relu(conv2d(h, P["features.4.weight"], P["features.4.bias"])))
    h = h.reshape(len(h), -1)                                     # flatten (N, 16, 5, 5) -> (N, 400)
    h = relu(h @ P["head.1.weight"].T + P["head.1.bias"])
    return h @ P["head.4.weight"].T + P["head.4.bias"]            # logits (N, 10)


def gradient_check(seed: int = 0, stride: int = 1, padding: int = 1, eps: float = 1e-6) -> dict:
    """Compare conv2d_backward with centred finite differences of the scalar loss L = sum(conv2d(x, w, b) * G)."""
    rng = np.random.default_rng(seed)
    x, w, b = rng.normal(size=(2, 3, 6, 5)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    G = rng.normal(size=conv2d(x, w, b, stride, padding).shape)   # dL/dy for this L
    loss = lambda: float((conv2d(x, w, b, stride, padding) * G).sum())
    dx, dw, db = conv2d_backward(G, x, w, stride, padding)
    worst = {}
    for name, arr, grad in (("x", x, dx), ("w", w, dw), ("b", b, db)):
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + eps; up = loss()
            arr[idx] = old - eps; down = loss()
            arr[idx] = old
            num[idx] = (up - down) / (2 * eps)
        worst[name] = float(np.abs(num - grad).max() / np.abs(num).max())
    return worst


if __name__ == "__main__":
    import time
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    print("forward: conv2d vs torch.nn.functional.conv2d (max |difference|)")
    for (H, W, k, s, p) in ((28, 28, 5, 1, 2), (28, 28, 3, 2, 1), (9, 7, 3, 1, 0), (10, 10, 4, 3, 2)):
        x, w, b = rng.normal(size=(2, 3, H, W)), rng.normal(size=(4, 3, k, k)), rng.normal(size=4)
        ref = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b), stride=s, padding=p).numpy()
        ours, fast = conv2d(x, w, b, s, p), conv2d_windows(x, w, b, s, p)
        print(f"  {H}x{W}, k={k}, s={s}, p={p}: out {ours.shape[2]}x{ours.shape[3]}"
              f"   loops {np.abs(ours - ref).max():.1e}   vectorised {np.abs(fast - ref).max():.1e}")

    print("backward: conv2d_backward vs finite differences (max relative error)")
    for s, p in ((1, 1), (2, 1), (1, 0)):
        print(f"  stride {s}, padding {p}:", {k: f"{v:.1e}" for k, v in gradient_check(0, s, p).items()})

    x, w = rng.normal(size=(16, 8, 14, 14)), rng.normal(size=(16, 8, 5, 5))
    xt, wt = torch.from_numpy(x), torch.from_numpy(w)
    for name, fn in (("NumPy loops", lambda: conv2d(x, w)), ("NumPy vectorised", lambda: conv2d_windows(x, w)),
                     ("PyTorch", lambda: F.conv2d(xt, wt))):
        fn()                                                      # warm-up
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        print(f"  {name:17s} {(time.perf_counter() - t0) / 5 * 1000:7.2f} ms   (batch of 16, 8 -> 16 channels, 14x14, k=5)")
