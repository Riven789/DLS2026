"""
test_model.py -- sanity checks worth running before any long training job.

    python test_model.py
"""
import math

import torch

from model import GPT, GPTConfig, sample_next
from tokenizer import BPETokenizer, CharTokenizer

CFG = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_tokenizers_round_trip():
    text = "The quick brown fox -- naive cafe, 123! The quick brown fox."
    for tok in (CharTokenizer.train(text), BPETokenizer.train(text, 280)):
        assert tok.decode(tok.encode(text)) == text
    # byte-level BPE can encode text it has never seen
    bpe = BPETokenizer.train(text, 280)
    assert bpe.decode(bpe.encode("zebra été")) == "zebra été"


def test_initial_loss_is_ln_V():
    torch.manual_seed(0)
    model = GPT(CFG)
    x = torch.randint(0, CFG.vocab_size, (64, CFG.block_size))
    y = torch.randint(0, CFG.vocab_size, (64, CFG.block_size))
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.15, loss.item()


def test_causality():
    """Changing token t must not change any prediction at positions < t."""
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    a = torch.randint(0, CFG.vocab_size, (1, CFG.block_size))
    b = a.clone()
    t = 9
    b[0, t] = (b[0, t] + 1) % CFG.vocab_size
    la, lb = model(a)[0], model(b)[0]
    assert torch.equal(la[0, :t], lb[0, :t])
    assert not torch.allclose(la[0, t:], lb[0, t:])


def test_attention_rows_are_distributions():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    model(torch.randint(0, CFG.vocab_size, (2, CFG.block_size)))
    attn = model.blocks[0].attn.last_attn                    # (B, h, T, T)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)
    assert torch.all(attn.triu(1) == 0)                      # nothing above the diagonal


def test_can_overfit_one_batch():
    """A correct model + optimizer can memorise a single small batch."""
    torch.manual_seed(0)
    model = GPT(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
    y = torch.roll(x, -1, dims=1)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, loss.item()


def test_sampling_filters():
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])
    assert sample_next(logits, temperature=0.0).item() == 0             # greedy
    torch.manual_seed(0)
    ks = {sample_next(logits, top_k=2).item() for _ in range(200)}
    assert ks == {0, 1}
    ps = {sample_next(logits, top_p=0.9).item() for _ in range(500)}
    assert ps == {0, 1, 2, 3}                                           # 'moon' is cut


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok   {name}")
