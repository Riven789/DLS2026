"""
test_cnn.py -- checks worth running before trusting any result.

    python test_cnn.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from cnn import CNN, MLP, build_model, shape_table
from conv_numpy import (cnn_forward, conv2d, conv2d_backward, conv2d_windows, gradient_check, max_pool2d_backward,
                        out_size, pool2d)
from data import augment_shift, load_fashion, shift_images
from train import evaluate, fit


def test_conv_forward_matches_pytorch():
    rng = np.random.default_rng(0)
    for H, W, k, s, p in ((28, 28, 5, 1, 2), (28, 28, 3, 2, 1), (9, 7, 3, 1, 0), (10, 10, 4, 3, 2), (8, 8, 1, 1, 0)):
        x, w, b = rng.normal(size=(2, 3, H, W)), rng.normal(size=(4, 3, k, k)), rng.normal(size=4)
        ref = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b), stride=s, padding=p).numpy()
        assert ref.shape[2:] == (out_size(H, k, s, p), out_size(W, k, s, p))      # the output-size formula
        assert np.abs(conv2d(x, w, b, s, p) - ref).max() < 1e-12
        assert np.abs(conv2d_windows(x, w, b, s, p) - ref).max() < 1e-12


def test_conv_backward_matches_finite_differences_and_autograd():
    for s, p in ((1, 1), (2, 1), (1, 0)):
        assert max(gradient_check(0, s, p).values()) < 1e-6
    rng = np.random.default_rng(1)
    x, w, b = rng.normal(size=(2, 3, 8, 8)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    xt, wt, bt = (torch.tensor(v, requires_grad=True) for v in (x, w, b))
    y = F.conv2d(xt, wt, bt, stride=2, padding=1)
    G = rng.normal(size=tuple(y.shape))
    (y * torch.from_numpy(G)).sum().backward()
    dx, dw, db = conv2d_backward(G, x, w, stride=2, padding=1)
    for ours, ref in ((dx, xt.grad), (dw, wt.grad), (db, bt.grad)):
        assert np.abs(ours - ref.numpy()).max() < 1e-12


def test_pooling_matches_pytorch():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 3, 7, 6))                      # 7 is odd: the last row is dropped
    xt = torch.tensor(x, requires_grad=True)
    assert np.allclose(pool2d(x, 2, "max"), F.max_pool2d(xt, 2).detach().numpy())
    assert np.allclose(pool2d(x, 2, "mean"), F.avg_pool2d(xt, 2).detach().numpy())
    G = rng.normal(size=(2, 3, 3, 3))
    (F.max_pool2d(xt, 2) * torch.from_numpy(G)).sum().backward()
    assert np.allclose(max_pool2d_backward(G, x, 2), xt.grad.numpy())


def test_slide_examples_and_equivariance():
    X = np.array([[1, 2, 0, 1, 3], [0, 1, 2, 3, 1], [2, 0, 1, 1, 0], [1, 3, 0, 2, 2], [0, 1, 1, 0, 1]], dtype=float)
    K = np.array([[1, 0, -1]] * 3, dtype=float)
    Y = conv2d(X[None, None], K[None, None])[0, 0]
    assert Y.tolist() == [[0, -2, -1], [0, -2, 0], [1, 1, -1]]                    # slide 3
    assert [out_size(*a) for a in ((5, 3, 1, 0), (5, 3, 2, 1), (28, 5, 1, 2), (14, 5), (8, 3, 2), (224, 7, 2, 3))] \
        == [3, 3, 28, 10, 3, 112]                                                 # slide 5
    Xb, Wb, D = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 1.0]]), np.array([[1, -1], [0, 2.0]]), np.array([[1, 0], [-1, 2.0]])
    dx, dw, db = conv2d_backward(D[None, None], Xb[None, None], Wb[None, None])
    assert dw[0, 0].tolist() == [[3, 7], [0, 2]] and dx[0, 0].tolist() == [[1, -1, 0], [-1, 5, -2], [0, -2, 4]]   # slide 12
    # equivariance: shift then convolve == convolve then shift, away from the border
    rng = np.random.default_rng(3)
    x, w = rng.normal(size=(1, 1, 20, 20)), rng.normal(size=(3, 1, 5, 5))
    a = conv2d(shift_images(x, 2, 3, 0.0), w, padding=2)
    b = shift_images(conv2d(x, w, padding=2), 2, 3, 0.0)
    assert np.abs(a - b)[:, :, 5:-5, 5:-5].max() < 1e-12
    assert np.abs(a - b).max() > 1e-3                                             # ... but not at the border


def test_numpy_network_matches_pytorch_and_parameter_count():
    torch.manual_seed(0)
    model = CNN()
    assert model.num_params() == 8 * (1 * 25 + 1) + 16 * (8 * 25 + 1) + (400 + 1) * 64 + (64 + 1) * 10 == 29_738
    assert sum(n for _, _, n in shape_table(model)) == model.num_params()
    assert [s for _, s, _ in shape_table(model)][-5:] == [(16, 5, 5), (400,), (64,), (64,), (10,)]
    assert MLP().num_params() == 109_386 and build_model("linear").num_params() == 7_850
    x = load_fashion()["X_test"][:16]
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    ours = cnn_forward(model.state_dict(), x.astype(np.float64))
    assert np.abs(ours - ref).max() < 1e-4                                        # float32 vs float64


def test_data_pipeline():
    d, d1 = load_fashion(seed=0), load_fashion(seed=1)
    assert d["X_train"].shape == (10_000, 1, 28, 28) and d["X_val"].shape == (2_000, 1, 28, 28)
    assert d["X_test"].shape == (10_000, 1, 28, 28) and d["X_train"].dtype == np.float32
    assert abs(d["X_train"].mean()) < 1e-3 and abs(d["X_train"].std() - 1) < 1e-3   # standardised on the train split
    assert abs(d["X_test"].mean()) > 1e-4                                         # ... so test is not exactly centred
    assert not np.array_equal(d["y_train"][:50], d1["y_train"][:50])              # the seed changes the split
    assert np.array_equal(d["y_test"], d1["y_test"])                              # the test set never changes
    assert set(d["y_train"]) == set(range(10))
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    s = shift_images(x, 1, -1, -9.0)                                              # 1 down, 1 left
    assert s[0, 0, 0].tolist() == [-9, -9, -9, -9] and s[0, 0, 1].tolist() == [1, 2, 3, -9]
    aug = augment_shift(d["X_train"][:8], np.random.default_rng(0), 2, 0.0)
    assert aug.shape == (8, 1, 28, 28) and not np.array_equal(aug, d["X_train"][:8])
    p = load_fashion(permute_pixels=True)
    assert np.allclose(np.sort(p["X_test"][0].ravel()), np.sort(d["X_test"][0].ravel()))   # same pixels, new places


def test_cnn_can_overfit_a_tiny_batch():
    d = load_fashion()
    tiny = dict(d, X_train=d["X_train"][:32], y_train=d["y_train"][:32], X_val=d["X_train"][:32], y_val=d["y_train"][:32])
    torch.manual_seed(0)
    model = CNN()
    h = fit(model, tiny, lr=3e-3, batch_size=32, epochs=60, patience=0)
    assert h[-1]["train_acc"] == 1.0 and h[-1]["train_loss"] < 0.05, h[-1]
    assert evaluate(model, d)["accuracy"] > 0.3                                   # 32 images: above chance, far from good


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok   {name}")
