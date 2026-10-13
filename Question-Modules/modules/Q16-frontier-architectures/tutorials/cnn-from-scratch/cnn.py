"""
cnn.py -- the models of the tutorial in PyTorch.

    CNN()                     LeNet-style: 2 x (conv 5x5 -> ReLU -> max-pool) -> flatten -> 64 -> 10
    CNN(batchnorm=True, dropout=0.3)
    CNN(head="gap")           global average pooling instead of flatten + hidden layer
    MLP(hidden=(128, 64))     fully connected baseline on the 784 raw pixels
    MLP(hidden=())            no hidden layer: softmax (multinomial logistic) regression

    python cnn.py             # print the shape of every layer's output and the parameter counts
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, channels: tuple[int, int] = (8, 16), hidden: int = 64, n_out: int = 10,
                 batchnorm: bool = False, dropout: float = 0.0, head: str = "flatten"):
        super().__init__()
        c1, c2 = channels
        norm = nn.BatchNorm2d if batchnorm else (lambda c: nn.Identity())
        self.features = nn.Sequential(                       # input            (N,  1, 28, 28)
            nn.Conv2d(1, c1, kernel_size=5, padding=2),      # 28 + 4 - 5 + 1   (N,  8, 28, 28)
            norm(c1), nn.ReLU(), nn.MaxPool2d(2),            # 28 / 2           (N,  8, 14, 14)
            nn.Conv2d(c1, c2, kernel_size=5),                # 14 - 5 + 1       (N, 16, 10, 10)
            norm(c2), nn.ReLU(), nn.MaxPool2d(2),            # 10 / 2           (N, 16,  5,  5)
        )
        if head == "flatten":
            self.head = nn.Sequential(
                nn.Flatten(),                                # 16 * 5 * 5       (N, 400)
                nn.Linear(c2 * 5 * 5, hidden), nn.ReLU(),    #                  (N, 64)
                nn.Dropout(dropout),                         # active in training only
                nn.Linear(hidden, n_out),                    # logits           (N, 10)
            )
        else:                                                # global average pooling: one number per channel
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c2, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))                   # softmax lives inside the loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLP(nn.Module):
    """Fully connected baseline.  The image is flattened first, so the network never sees which
    pixels are neighbours."""

    def __init__(self, hidden: tuple[int, ...] = (128, 64), n_in: int = 28 * 28, n_out: int = 10, dropout: float = 0.0):
        super().__init__()
        layers, width = [nn.Flatten()], n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(name: str, batchnorm: bool = False, dropout: float = 0.0, hidden: tuple[int, ...] = (128, 64)) -> nn.Module:
    if name == "cnn":
        return CNN(batchnorm=batchnorm, dropout=dropout)
    if name == "cnn_gap":
        return CNN(batchnorm=batchnorm, head="gap")
    if name == "mlp":
        return MLP(hidden, dropout=dropout)
    if name == "linear":
        return MLP(())
    raise ValueError(name)


def shape_table(model: nn.Module, input_shape=(1, 1, 28, 28)) -> list[tuple[str, tuple, int]]:
    """(layer, output shape without the batch axis, number of parameters) for every layer that does something."""
    rows, x = [("input", tuple(input_shape[1:]), 0)], torch.zeros(input_shape)
    model.eval()
    with torch.no_grad():
        for block in model.children():
            for layer in block:
                x = layer(x)
                if not isinstance(layer, (nn.Identity, nn.Dropout)):
                    n = sum(p.numel() for p in layer.parameters())
                    rows.append((str(layer).split("(")[0], tuple(x.shape[1:]), n))
    return rows


if __name__ == "__main__":
    for name in ("cnn", "cnn_gap", "mlp", "linear"):
        model = build_model(name)
        print(f"\n{name}: {model.num_params():,} parameters")
        for layer, shape, n in shape_table(model):
            print(f"  {layer:18s} {str(shape):16s} {n:8,d}")
