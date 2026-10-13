"""
test_lstm.py -- checks worth running before trusting any result.

    python test_lstm.py
"""
import math

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from data import CharData, load_forecast, stream_batches
from lstm import LSTMCellScratch, ScratchLSTM, SeqModel, detach
from lstm_numpy import gradient_check, lstm_backward, lstm_forward, lstm_step, params_from_torch
from train import run_adding, stream_loss


def test_numpy_forward_matches_torch():
    torch.manual_seed(0)
    ref = torch.nn.LSTM(4, 5, batch_first=True).double()
    X = np.random.default_rng(0).normal(size=(3, 7, 4))
    H, (h, c), _ = lstm_forward(X, *params_from_torch(ref))
    H_ref, (h_ref, c_ref) = ref(torch.from_numpy(X))
    assert np.allclose(H, H_ref.detach().numpy(), atol=1e-12)
    assert np.allclose(c, c_ref[0].detach().numpy(), atol=1e-12)


def test_bptt_matches_finite_differences_and_autograd():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    ref = torch.nn.LSTM(3, 4, batch_first=True).double()
    Wx, Wh, b = params_from_torch(ref)
    X, target = rng.normal(size=(2, 6, 3)), rng.normal(size=(2, 6, 4))
    assert gradient_check(Wx, Wh, b, X, target) < 1e-4
    H, _, caches = lstm_forward(X, Wx, Wh, b)
    _, dWx, dWh, db, _ = lstm_backward(H - target, caches, Wx, Wh)
    (0.5 * ((ref(torch.from_numpy(X))[0] - torch.from_numpy(target)) ** 2).sum()).backward()
    assert np.allclose(dWx, ref.weight_ih_l0.grad.numpy(), atol=1e-10)
    assert np.allclose(dWh, ref.weight_hh_l0.grad.numpy(), atol=1e-10)
    assert np.allclose(db, ref.bias_ih_l0.grad.numpy(), atol=1e-10)


def test_scratch_cell_matches_torch_and_parameter_count():
    torch.manual_seed(2)
    n_in, n_h = 3, 5
    mine, ref = ScratchLSTM(n_in, n_h), torch.nn.LSTM(n_in, n_h, batch_first=True)
    with torch.no_grad():                                     # [W_x  W_h] side by side, biases summed
        mine.cell.lin.weight.copy_(torch.cat([ref.weight_ih_l0, ref.weight_hh_l0], dim=1))
        mine.cell.lin.bias.copy_(ref.bias_ih_l0 + ref.bias_hh_l0)
    x = torch.randn(4, 9, n_in)
    assert torch.allclose(mine(x)[0], ref(x)[0], atol=1e-6)
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(mine) == 4 * (n_in + n_h + 1) * n_h                  # the textbook formula
    assert count(ref) == 4 * (n_in + n_h + 2) * n_h                   # PyTorch keeps two bias vectors
    assert LSTMCellScratch(2, 3).lin.bias[3:6].tolist() == [1.0, 1.0, 1.0]   # forget bias starts at 1


def test_hand_example_from_the_slides():
    Wx, Wh = np.array([[1.0], [0.5], [0.5], [1.0]]), np.array([[-2.0], [1.0], [1.0], [2.0]])   # rows i, f, g, o
    b = np.array([0.0, 1.0, 0.0, -1.0])
    h, c, _ = lstm_step(np.array([[1.0]]), np.array([[0.5]]), np.array([[1.0]]), Wx, Wh, b)
    assert abs(c[0, 0] - 1.2616) < 1e-4 and abs(h[0, 0] - 0.6225) < 1e-4


def test_forecast_split_is_chronological_and_leak_free():
    d = load_forecast(window=24, horizon=1)
    assert d["t_train"].max() < d["t_val"].min() <= d["t_val"].max() < d["t_test"].min()
    n_train = d["cuts"]["train"][1]
    assert abs(d["raw"][:n_train].mean() - d["scaler"].mean) < 1e-9   # scaler fitted on train months only
    assert abs(d["raw"].mean() - d["scaler"].mean) > 1.0              # ... not on the whole series
    k = 5                                                             # a window ends right before its target
    assert np.isclose(d["X_test"][k, -1, 0].item(), d["scaler"].transform(d["raw"][d["t_test"][k] - 1]), atol=1e-5)
    assert np.isclose(d["y_test"][k].item(), d["scaler"].transform(d["raw"][d["t_test"][k]]), atol=1e-5)


def test_packing_ignores_padding():
    torch.manual_seed(3)
    lstm = torch.nn.LSTM(2, 3, batch_first=True)
    seqs = [torch.randn(n, 2) for n in (5, 3, 2)]
    packed = pack_padded_sequence(pad_sequence(seqs, batch_first=True), torch.tensor([5, 3, 2]), batch_first=True)
    h_packed = lstm(packed)[1][0][0]
    for k, s in enumerate(seqs):
        assert torch.allclose(h_packed[k], lstm(s[None])[1][0][0, 0], atol=1e-6)


def test_carried_state_equals_one_long_pass_and_detach_cuts_the_graph():
    torch.manual_seed(4)
    model = SeqModel("lstm", 8, 16, 11, vocab=11, readout="all")
    ids = torch.randint(0, 11, (2, 30))
    whole, _ = model(ids)
    first, state = model(ids[:, :12])
    second, _ = model(ids[:, 12:], detach(state))
    assert torch.allclose(whole, torch.cat([first, second], dim=1), atol=1e-6)
    model.zero_grad()
    second.sum().backward()                                   # must not reach back into the first chunk
    x, y, is_first = next(stream_batches(torch.arange(100), B=4, T=5))
    assert is_first and (y == x + 1).all()                    # target = next symbol


def test_models_start_at_chance_and_lstm_learns_to_add():
    torch.manual_seed(5)
    cd = CharData()
    model = SeqModel("lstm", 16, 32, cd.vocab, vocab=cd.vocab, readout="all")
    assert abs(stream_loss(model, cd.val[:5000]) - math.log(cd.vocab)) < 0.15   # untrained: about ln 65
    _, _, r = run_adding("lstm", T=10, seed=0, steps=400)
    assert r["test_mse"] < 0.2 * r["baseline_mse"], r


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok   {name}")
