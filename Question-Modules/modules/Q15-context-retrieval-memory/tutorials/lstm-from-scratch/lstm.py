"""
lstm.py -- the recurrent models of the tutorial, in PyTorch.

    LSTMCellScratch   one LSTM step written with a single nn.Linear, mirroring the six equations
    ScratchLSTM       that cell unrolled over a sequence (slow, readable)
    SeqModel          [embedding] -> nn.RNN / nn.LSTM / nn.GRU / ScratchLSTM -> linear head

Shapes: a batch of sequences is (B, T, n_in); hidden and cell states are (B, n_h).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMCellScratch(nn.Module):
    """One time step.  All four gates read the same vector [x_t, h_{t-1}]."""

    def __init__(self, n_in: int, n_h: int, forget_bias: float = 1.0):
        super().__init__()
        self.n_h = n_h
        self.lin = nn.Linear(n_in + n_h, 4 * n_h)            # rows: input, forget, candidate, output
        with torch.no_grad():
            self.lin.bias[n_h:2 * n_h].fill_(forget_bias)    # start with the forget gate open

    def forward(self, x, state):
        h_prev, c_prev = state
        z = self.lin(torch.cat([x, h_prev], dim=1))          # (B, 4 n_h) pre-activations
        zi, zf, zg, zo = z.chunk(4, dim=1)
        i = torch.sigmoid(zi)                                # input gate:  how much to write
        f = torch.sigmoid(zf)                                # forget gate: how much to keep
        g = torch.tanh(zg)                                   # candidate:   what to write
        o = torch.sigmoid(zo)                                # output gate: how much to show
        c = f * c_prev + i * g                               # cell state: additive update
        h = o * torch.tanh(c)                                # hidden state: gated view of the cell
        return h, c


class ScratchLSTM(nn.Module):
    """The cell applied left to right.  Same call signature as nn.LSTM(batch_first=True), one layer."""

    def __init__(self, n_in: int, n_h: int, forget_bias: float = 1.0):
        super().__init__()
        self.cell = LSTMCellScratch(n_in, n_h, forget_bias)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        if state is None:
            h = c = x.new_zeros(B, self.cell.n_h)
        else:
            h, c = state[0][0], state[1][0]
        outs = []
        for t in range(T):                                   # the same weights at every step
            h, c = self.cell(x[:, t], (h, c))
            outs.append(h)
        return torch.stack(outs, dim=1), (h[None], c[None])


def set_forget_bias(lstm: nn.LSTM, value: float = 1.0) -> None:
    """PyTorch adds two bias vectors (b_ih + b_hh), each ordered [input, forget, candidate, output].
    Put `value` in the forget slice of the first and zero in the second, so their sum is `value`."""
    n_h = lstm.hidden_size
    with torch.no_grad():
        for name, p in lstm.named_parameters():
            if name.startswith("bias_ih"):
                p[n_h:2 * n_h].fill_(value)
            elif name.startswith("bias_hh"):
                p[n_h:2 * n_h].zero_()


class SeqModel(nn.Module):
    """A recurrent network with a linear read-out.

    kind     'rnn' (Elman, tanh) | 'lstm' | 'gru' | 'scratch' (ScratchLSTM)
    vocab    if given, inputs are integer ids and pass through an embedding table of width n_in
    readout  'last': one output per sequence, from h_T (many-to-one)
             'all' : one output per time step (many-to-many)
    """

    def __init__(self, kind: str, n_in: int, n_h: int, n_out: int, layers: int = 1, dropout: float = 0.0,
                 bidirectional: bool = False, vocab: int | None = None, readout: str = "last",
                 forget_bias: float = 1.0):
        super().__init__()
        self.kind, self.readout = kind, readout
        self.embed = nn.Embedding(vocab, n_in) if vocab else None
        kw = dict(num_layers=layers, batch_first=True, bidirectional=bidirectional,
                  dropout=dropout if layers > 1 else 0.0)    # dropout between stacked layers only
        if kind == "rnn":
            self.rnn = nn.RNN(n_in, n_h, nonlinearity="tanh", **kw)
        elif kind == "gru":
            self.rnn = nn.GRU(n_in, n_h, **kw)
        elif kind == "lstm":
            self.rnn = nn.LSTM(n_in, n_h, **kw)
            set_forget_bias(self.rnn, forget_bias)
        elif kind == "scratch":
            assert layers == 1 and not bidirectional
            self.rnn = ScratchLSTM(n_in, n_h, forget_bias)
        else:
            raise ValueError(kind)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_h * (2 if bidirectional else 1), n_out)

    def forward(self, x, state=None):
        """x: (B, T) ids or (B, T, n_in) floats.  Returns (output, state)."""
        if self.embed is not None:
            x = self.embed(x)
        hs, state = self.rnn(x, state)                       # hs: (B, T, n_h), top layer, every step
        if self.readout == "last":
            hs = hs[:, -1]                                   # (B, n_h)
        return self.head(self.drop(hs)), state

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def detach(state):
    """Cut the graph between two truncated-BPTT segments but keep the values."""
    if state is None:
        return None
    return tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()
