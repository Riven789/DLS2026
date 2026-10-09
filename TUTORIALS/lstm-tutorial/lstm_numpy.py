"""
lstm_numpy.py -- an LSTM layer in NumPy, no autograd: one step, a full forward
pass, and backpropagation through time (BPTT) written out by hand.

Notation (column-vector maths, batches stored as rows):

    z_t = W_x x_t + W_h h_{t-1} + b          4 n_h pre-activations, stacked as [i, f, g, o]
    i_t = sigmoid(z_i)    input gate         f_t = sigmoid(z_f)    forget gate
    g_t = tanh(z_g)       candidate          o_t = sigmoid(z_o)    output gate
    c_t = f_t * c_{t-1} + i_t * g_t          cell state   (additive update)
    h_t = o_t * tanh(c_t)                    hidden state (what the next layer sees)

W_x is (4 n_h, n_in), W_h is (4 n_h, n_h), b is (4 n_h,): the layout of torch.nn.LSTM,
whose two bias vectors b_ih + b_hh are summed into the single b used here.

    python lstm_numpy.py        # compare with torch.nn.LSTM, then check BPTT by finite differences
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def lstm_step(x, h_prev, c_prev, Wx, Wh, b):
    """One time step for a batch.  x: (B, n_in); h_prev, c_prev: (B, n_h)."""
    n_h = h_prev.shape[1]
    z = x @ Wx.T + h_prev @ Wh.T + b                     # (B, 4 n_h)
    i = sigmoid(z[:, 0 * n_h:1 * n_h])                   # input gate
    f = sigmoid(z[:, 1 * n_h:2 * n_h])                   # forget gate
    g = np.tanh(z[:, 2 * n_h:3 * n_h])                   # candidate values
    o = sigmoid(z[:, 3 * n_h:4 * n_h])                   # output gate
    c = f * c_prev + i * g                               # keep some of the old, add some of the new
    tc = np.tanh(c)
    h = o * tc
    cache = (x, h_prev, c_prev, i, f, g, o, tc)
    return h, c, cache


def lstm_forward(X, Wx, Wh, b, h0=None, c0=None):
    """X: (B, T, n_in).  Returns all hidden states (B, T, n_h), the final (h, c) and the caches."""
    B, T, _ = X.shape
    n_h = Wh.shape[1]
    h = np.zeros((B, n_h)) if h0 is None else h0
    c = np.zeros((B, n_h)) if c0 is None else c0
    H, caches = np.zeros((B, T, n_h)), []
    for t in range(T):                                   # the same Wx, Wh, b at every step
        h, c, cache = lstm_step(X[:, t], h, c, Wx, Wh, b)
        H[:, t] = h
        caches.append(cache)
    return H, (h, c), caches


def lstm_backward(dH, caches, Wx, Wh):
    """Backpropagation through time.

    dH: (B, T, n_h), the gradient of the loss with respect to every h_t *as an output*.
    Returns dX (B, T, n_in), dWx, dWh, db and the gradients w.r.t. the initial (h0, c0)."""
    B, T, n_h = dH.shape
    dWx, dWh, db = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros(4 * n_h)
    dX = np.zeros((B, T, Wx.shape[1]))
    dh_next, dc_next = np.zeros((B, n_h)), np.zeros((B, n_h))   # messages from step t+1
    for t in reversed(range(T)):
        x, h_prev, c_prev, i, f, g, o, tc = caches[t]
        dh = dH[:, t] + dh_next                          # from the output at t and from step t+1
        dc = dc_next + dh * o * (1 - tc ** 2)            # h = o * tanh(c)
        dz = np.concatenate([dc * g * i * (1 - i),       # through the input gate   (sigmoid' = s(1-s))
                             dc * c_prev * f * (1 - f),  # through the forget gate
                             dc * i * (1 - g ** 2),      # through the candidate    (tanh' = 1 - tanh^2)
                             dh * tc * o * (1 - o)],     # through the output gate
                            axis=1)                      # (B, 4 n_h)
        dWx += dz.T @ x                                  # shared weights: gradients add over time
        dWh += dz.T @ h_prev
        db += dz.sum(axis=0)
        dX[:, t] = dz @ Wx
        dh_next = dz @ Wh                                # to h_{t-1}: through a weight matrix
        dc_next = dc * f                                 # to c_{t-1}: only an elementwise product
    return dX, dWx, dWh, db, (dh_next, dc_next)


def rnn_step(x, h_prev, Wx, Wh, b):
    """The simple (Elman) recurrent step, for comparison:  h_t = tanh(W_x x_t + W_h h_{t-1} + b)."""
    return np.tanh(x @ Wx.T + h_prev @ Wh.T + b)


def params_from_torch(lstm):
    """Copy the weights of a one-layer torch.nn.LSTM into the (Wx, Wh, b) used here."""
    return (lstm.weight_ih_l0.detach().numpy().astype(np.float64),
            lstm.weight_hh_l0.detach().numpy().astype(np.float64),
            (lstm.bias_ih_l0 + lstm.bias_hh_l0).detach().numpy().astype(np.float64))


def gradient_check(Wx, Wh, b, X, target, eps: float = 1e-6) -> float:
    """Largest relative difference between BPTT and centred finite differences
    for the loss  L = 0.5 * sum((H - target)^2)  over all time steps."""
    def loss():
        H, _, _ = lstm_forward(X, Wx, Wh, b)
        return 0.5 * ((H - target) ** 2).sum()

    H, _, caches = lstm_forward(X, Wx, Wh, b)
    _, dWx, dWh, db, _ = lstm_backward(H - target, caches, Wx, Wh)
    worst = 0.0
    for P, dP in ((Wx, dWx), (Wh, dWh), (b, db)):
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + eps
            up = loss()
            P[idx] = old - eps
            down = loss()
            P[idx] = old
            numeric = (up - down) / (2 * eps)
            worst = max(worst, abs(numeric - dP[idx]) / max(1e-8, abs(numeric) + abs(dP[idx])))
    return worst


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    B, T, n_in, n_h = 3, 7, 4, 5

    ref = torch.nn.LSTM(n_in, n_h, batch_first=True).double()
    Wx, Wh, b = params_from_torch(ref)
    X = rng.normal(size=(B, T, n_in))
    H, (h, c), _ = lstm_forward(X, Wx, Wh, b)
    H_ref, (h_ref, c_ref) = ref(torch.from_numpy(X))
    print(f"forward: max |NumPy - torch.nn.LSTM| over all h_t = {np.abs(H - H_ref.detach().numpy()).max():.2e},"
          f" final c = {np.abs(c - c_ref[0].detach().numpy()).max():.2e}")

    target = rng.normal(size=(B, T, n_h))
    loss = 0.5 * ((H_ref - torch.from_numpy(target)) ** 2).sum()
    loss.backward()
    _, caches = lstm_forward(X, Wx, Wh, b)[1:]
    _, dWx, dWh, db, _ = lstm_backward(H - target, caches, Wx, Wh)
    worst = max(np.abs(dWx - ref.weight_ih_l0.grad.numpy()).max(), np.abs(dWh - ref.weight_hh_l0.grad.numpy()).max(),
                np.abs(db - ref.bias_ih_l0.grad.numpy()).max())
    print(f"backward: max |BPTT - autograd| = {worst:.2e}")
    print(f"gradient check: worst relative error vs finite differences = "
          f"{gradient_check(Wx, Wh, b, X, target):.2e}")
    print(f"parameters: 4 * (n_in + n_h + 1) * n_h = {4 * (n_in + n_h + 1) * n_h}"
          f"   (torch.nn.LSTM stores {sum(p.numel() for p in ref.parameters())}: two bias vectors)")
