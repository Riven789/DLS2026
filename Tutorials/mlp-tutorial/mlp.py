"""
mlp.py -- the same network in PyTorch, which computes the gradients for us.

    MLP(n_in=13, hidden=[32, 32], n_out=2)   # two hidden layers of 32 neurons
    MLP(n_in=13, hidden=[],       n_out=2)   # no hidden layer: a linear model
"""
from __future__ import annotations

import torch
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], n_out: int,
                 activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        layers, width = [], n_in
        for h in hidden:
            layers.append(nn.Linear(width, h))            # z = a W + b
            layers.append(ACTIVATIONS[activation]())      # a = phi(z)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))        # randomly silence neurons (training only)
            width = h
        layers.append(nn.Linear(width, n_out))            # output layer: raw scores, no activation
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                if activation in ("relu", "gelu"):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")   # He initialisation
                else:
                    nn.init.xavier_normal_(m.weight)                         # Glorot initialisation
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
