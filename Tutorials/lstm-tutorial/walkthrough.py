"""
walkthrough.py -- the worked examples on the slides, computed step by step.

    python walkthrough.py
"""
import math

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from data import Scaler, make_windows
from lstm import LSTMCellScratch
from lstm_numpy import lstm_step, rnn_step, sigmoid

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def header(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- preparing a time series (slide 3)
header("1. A TIME SERIES BECOMES WINDOWS  (scaling fitted on the training period only)")
series = np.array([10.0, 12, 16, 14, 18, 20, 24, 22])
n_train, window = 6, 3
sc = Scaler().fit(series[:n_train])
z = sc.transform(series)
print("series        =", series, f"  train = first {n_train} values, test = last {len(series) - n_train}")
print(f"train mean = {sc.mean:.3f}, train std = {sc.std:.3f}   (the test values 24, 22 are NOT used)")
print("standardised  =", z)
for split, lo, hi in (("train", 0, n_train), ("test", n_train, len(series))):
    X, y, t = make_windows(z, window, 1, lo, hi)
    for xi, yi, ti in zip(X, y, t):
        print(f"  {split:5s} target index {ti}: window {xi} -> target {yi:.4f}")
bad = Scaler().fit(series)
print(f"leaky version: mean over ALL values = {bad.mean:.3f}, std = {bad.std:.3f}  (uses the future)")

header("2. TEXT BECOMES IDS, IDS BECOME VECTORS")
text = "hello"
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = [stoi[c] for c in text]
print("vocabulary:", stoi)
print("ids:", ids, "  input x =", ids[:-1], "  target y (next character) =", ids[1:])
print("one-hot of id 2:", np.eye(len(chars))[2], " | an embedding is a learned row of a V x d table instead")

# ---------------------------------------------------------------- the simple RNN (slide 4)
header("3. SIMPLE RNN, TWO STEPS:  h_t = tanh(W_x x_t + W_h h_{t-1} + b)")
Wx = np.array([[0.5], [-1.0]])
Wh = np.array([[0.5, -0.5], [0.25, 1.0]])
b = np.array([0.0, 0.5])
h = np.zeros((1, 2))
print("W_x =", Wx.ravel(), "  W_h =", Wh.tolist(), "  b =", b, "  h_0 = (0, 0)")
for t, x in enumerate((1.0, 2.0), start=1):
    zt = np.array([[x]]) @ Wx.T + h @ Wh.T + b
    h_new = rnn_step(np.array([[x]]), h, Wx, Wh, b)
    print(f"t = {t}: x_t = {x}, z_t = W_x x_t + W_h h_{t - 1} + b = {zt[0]},  h_{t} = tanh(z_t) = {h_new[0]}")
    h = h_new
J = np.diag(1 - h[0] ** 2) @ Wh
print("Jacobian dh_2/dh_1 = diag(1 - h_2^2) W_h =\n", J, "\n largest singular value:", np.linalg.svd(J)[1][0].round(4))
print(f"parameters (n_in + n_h + 1) n_h = (1 + 2 + 1) * 2 = {(1 + 2 + 1) * 2}, the same at every time step")

# ---------------------------------------------------------------- vanishing / exploding (slide 5)
header("4. A PRODUCT OF T FACTORS")
for w in (0.9, 1.1):
    print(f"|w| = {w}:  w^20 = {w ** 20:.4g}   w^50 = {w ** 50:.4g}   w^100 = {w ** 100:.4g}")
print("LSTM cell path, factor = forget gate f:")
for zf in (0.0, 1.0, 4.6):
    f = 1 / (1 + math.exp(-zf))
    print(f"  f = sigmoid({zf}) = {f:.3f}:  f^20 = {f ** 20:.3g}   f^50 = {f ** 50:.3g}   f^100 = {f ** 100:.3g}")

# ---------------------------------------------------------------- one LSTM step (slide 8)
header("5. ONE LSTM STEP BY HAND  (n_in = 1, n_h = 1)")
rows = dict(i=(1.0, -2.0, 0.0), f=(0.5, 1.0, 1.0), g=(0.5, 1.0, 0.0), o=(1.0, 2.0, -1.0))   # (w_x, w_h, bias)
x, h_prev, c_prev = 1.0, 0.5, 1.0
print(f"x_t = {x}, h_(t-1) = {h_prev}, c_(t-1) = {c_prev};  rows (w_x, w_h, b):", rows)
zs = {k: wx * x + wh * h_prev + bb for k, (wx, wh, bb) in rows.items()}
f, i, g, o = sigmoid(zs["f"]), sigmoid(zs["i"]), math.tanh(zs["g"]), sigmoid(zs["o"])
print(f"forget    f = sigmoid({zs['f']:+.1f}) = {f:.4f}")
print(f"input     i = sigmoid({zs['i']:+.1f}) = {i:.4f}")
print(f"candidate g = tanh({zs['g']:+.1f})    = {g:.4f}")
print(f"output    o = sigmoid({zs['o']:+.1f}) = {o:.4f}")
c = f * c_prev + i * g
h = o * math.tanh(c)
print(f"c_t = f * c_(t-1) + i * g = {f:.4f} * {c_prev} + {i:.4f} * {g:.4f} = {f * c_prev:.4f} + {i * g:.4f} = {c:.4f}")
print(f"h_t = o * tanh(c_t) = {o:.4f} * {math.tanh(c):.4f} = {h:.4f}")

order = "ifgo"                                               # the layout used by the code
Wx = np.array([[rows[k][0]] for k in order])
Wh = np.array([[rows[k][1]] for k in order])
bb = np.array([rows[k][2] for k in order])
h_np, c_np, _ = lstm_step(np.array([[x]]), np.array([[h_prev]]), np.array([[c_prev]]), Wx, Wh, bb)
cell = LSTMCellScratch(1, 1).double()
with torch.no_grad():
    cell.lin.weight.copy_(torch.from_numpy(np.concatenate([Wx, Wh], axis=1)))
    cell.lin.bias.copy_(torch.from_numpy(bb))
h_t, c_t = cell(torch.tensor([[x]], dtype=torch.float64), (torch.tensor([[h_prev]], dtype=torch.float64),
                                                         torch.tensor([[c_prev]], dtype=torch.float64)))
print(f"lstm_numpy.lstm_step: h = {h_np[0, 0]:.4f}, c = {c_np[0, 0]:.4f}   "
      f"lstm.LSTMCellScratch: h = {h_t.item():.4f}, c = {c_t.item():.4f}")
print(f"direct path dc_t/dc_(t-1) = f = {f:.4f}  (no weight matrix, no squashing)")
h2, c2, _ = lstm_step(np.array([[0.0]]), h_np, c_np, Wx, Wh, bb)
print(f"next step with x = 0: c = {c2[0, 0]:.4f}, h = {h2[0, 0]:.4f}")
print(f"parameters: 4 (n_in + n_h + 1) n_h = 4 * 3 * 1 = {4 * 3 * 1}")

# ---------------------------------------------------------------- counting parameters (slide 10)
header("6. COUNTING PARAMETERS  (n_in = 32, n_h = 256: the character model)")
n_in, n_h = 32, 256
for name, k, mod in (("RNN", 1, torch.nn.RNN), ("GRU", 3, torch.nn.GRU), ("LSTM", 4, torch.nn.LSTM)):
    textbook = k * (n_in + n_h + 1) * n_h
    actual = sum(p.numel() for p in mod(n_in, n_h).parameters())
    print(f"{name:5s} {k} (n_in + n_h + 1) n_h = {textbook:8,d}   torch.nn.{name}: {actual:8,d}"
          f"  = textbook + {k} n_h (second bias vector: {actual - textbook})")
print("lstm.LSTMCellScratch(32, 256):", sum(p.numel() for p in LSTMCellScratch(n_in, n_h).parameters()))

# ---------------------------------------------------------------- clipping (slide 12)
header("7. GRADIENT CLIPPING BY NORM")
gvec, tau = np.array([3.0, 4.0]), 1.0
norm = np.linalg.norm(gvec)
print(f"g = {gvec}, ||g|| = {norm}, threshold = {tau}:  g <- g * min(1, {tau}/||g||) = {gvec * min(1, tau / norm)}")

# ---------------------------------------------------------------- padding and packing (slide 13)
header("8. SEQUENCES OF DIFFERENT LENGTH: PAD, THEN PACK")
torch.manual_seed(0)
seqs = [torch.randn(n, 2) for n in (5, 3, 2)]
padded = pad_sequence(seqs, batch_first=True)                 # (3, 5, 2), zeros after each sequence's end
lengths = torch.tensor([5, 3, 2])
packed = pack_padded_sequence(padded, lengths, batch_first=True)
print("lengths:", lengths.tolist(), " padded batch shape:", tuple(padded.shape))
print("packed.batch_sizes (sequences still running at each step):", packed.batch_sizes.tolist())
lstm = torch.nn.LSTM(2, 3, batch_first=True)
_, (h_packed, _) = lstm(packed)
_, (h_naive, _) = lstm(padded)
for k, s in enumerate(seqs):
    _, (h_alone, _) = lstm(s[None])
    print(f"  sequence {k} (length {len(s)}): |h_packed - h_alone| = {(h_packed[0, k] - h_alone[0, 0]).abs().max():.1e}"
          f"   |h_unpacked - h_alone| = {(h_naive[0, k] - h_alone[0, 0]).abs().max():.1e}")

# ---------------------------------------------------------------- output heads (slide 14)
header("9. SOFTMAX HEAD: CROSS-ENTROPY, PERPLEXITY, TEMPERATURE")
logits = np.array([2.0, 1.0, 0.0, -1.0])
for temp in (0.5, 1.0, 2.0):
    e = np.exp(logits / temp)
    print(f"temperature {temp}: p = softmax(logits / T) = {e / e.sum()}")
p = np.exp(logits) / np.exp(logits).sum()
print(f"true next token is the 2nd: loss = -ln {p[1]:.3f} = {-math.log(p[1]):.3f}")
probs = np.array([0.5, 0.25, 0.125, 0.5])
nll = -np.log(probs)
print("probabilities given to the true next token at 4 positions:", probs)
print(f"losses {nll}, mean = {nll.mean():.3f} nats, perplexity = exp(mean) = {math.exp(nll.mean()):.3f},"
      f" bits per token = {nll.mean() / math.log(2):.3f}")
print(f"uniform guess over 65 characters: loss ln 65 = {math.log(65):.3f}, perplexity 65")
