"""
test_mlp.py -- checks worth running before trusting any result.

    python test_mlp.py
"""
import numpy as np
import torch

from data import LabelEncoder, TabularEncoder, load_task
from mlp import MLP
from mlp_numpy import NumpyMLP, gradient_check
from train import evaluate, fit


def test_backprop_matches_finite_differences():
    rng = np.random.default_rng(0)
    net = NumpyMLP([4, 6, 3], "classification", seed=1)
    assert gradient_check(net, rng.normal(size=(5, 4)), rng.integers(0, 3, 5)) < 1e-7
    net = NumpyMLP([4, 6, 1], "regression", seed=1)
    assert gradient_check(net, rng.normal(size=(5, 4)), rng.normal(size=5)) < 1e-7


def test_numpy_and_pytorch_agree():
    """Same weights, same batch -> same loss and same gradients from both implementations."""
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(8, 4)), rng.integers(0, 3, 8)
    net = NumpyMLP([4, 6, 3], "classification", seed=2)
    model = MLP(4, [6], 3).double()
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, W, b in zip(linears, net.W, net.b):
            lin.weight.copy_(torch.from_numpy(W.T))       # PyTorch stores W as (out, in)
            lin.bias.copy_(torch.from_numpy(b))
    loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(x)), torch.from_numpy(y))
    loss.backward()
    assert abs(loss.item() - net.loss(x, y)) < 1e-10
    gW, gb = net.backward(y)
    for lin, g in zip(linears, gW):
        assert np.allclose(lin.weight.grad.numpy().T, g, atol=1e-10)


def test_encoder():
    train = [dict(age="20", ok="True", town="A"), dict(age="", ok="False", town="B"),
             dict(age="40", ok="True", town="A")]
    enc = TabularEncoder(["age"], ["ok"], ["town"]).fit(train)
    assert enc.feature_names == ["age", "age_missing", "ok", "town=A", "town=B"]
    X = enc.transform(train)
    assert np.allclose(X[:, 0], [-1, 0, 1])               # standardised; missing -> 0
    assert X[:, 1].tolist() == [0, 1, 0]                  # missing flag
    assert X[:, 2].tolist() == [1, 0, 1]                  # logical -> 0/1
    assert np.allclose(X[:, 3:].sum(1), 1)                # one-hot
    unseen = enc.transform([dict(age="30", ok="False", town="Z")])
    assert unseen[0, 3:].tolist() == [0, 0]               # unseen category -> all zeros
    assert np.isclose(unseen[0, 0], 0)                    # uses the TRAIN mean (30)
    assert LabelEncoder().fit(["yes", "no", "yes"]).transform(["no", "yes"]).tolist() == [0, 1]


def test_no_information_leaks_from_test_rows():
    d = load_task("titanic", seed=0)
    assert abs(d["X_train"][:, 2].mean()) < 1e-5          # fare is centred on the train split...
    assert abs(d["X_test"][:, 2].mean()) > 1e-4           # ...so the test split is not exactly centred


def test_xor_needs_a_hidden_layer():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    d = dict(task="classification", n_out=2, X_train=X, y_train=y, X_val=X, y_val=y, X_test=X, y_test=y)
    torch.manual_seed(0)
    linear, mlp = MLP(2, [], 2), MLP(2, [8], 2)
    for model in (linear, mlp):
        fit(model, d, optimizer="adam", lr=0.05, batch_size=4, epochs=300, patience=0)
    assert evaluate(linear, d)["accuracy"] <= 0.75        # a line gets at most 3 of 4 right
    assert evaluate(mlp, d)["accuracy"] == 1.0


def test_can_overfit_a_tiny_batch():
    d = load_task("mpg", seed=0)
    small = dict(d, X_train=d["X_train"][:8], y_train=d["y_train"][:8])
    torch.manual_seed(0)
    model = MLP(small["X_train"].shape[1], [64, 64], 1)
    h = fit(model, small, optimizer="adam", lr=0.01, batch_size=8, epochs=300, patience=0)
    assert h[-1]["train_loss"] < 1e-3, h[-1]["train_loss"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok   {name}")
