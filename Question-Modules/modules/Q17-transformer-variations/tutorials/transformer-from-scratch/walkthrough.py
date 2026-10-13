"""
walkthrough.py -- every stage of a decoder-only transformer, by hand.

Part A uses NumPy only and numbers small enough to check with a calculator.
These are the worked examples on the slides.

Part B (runs if out/ckpt.pt exists) pushes a real prompt through the trained
model and prints what each stage produces.

    python walkthrough.py
"""
from __future__ import annotations

import math
import os

import numpy as np

from tokenizer import BPETokenizer

np.set_printoptions(precision=3, suppress=True, linewidth=120)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)          # subtract the max: same answer, no overflow
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU(x) = x * Phi(x), Phi = standard normal CDF."""
    return x * 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)             # gain = 1, bias = 0


def header(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ---------------------------------------------------------------- Part A
def stage_tokenization() -> None:
    header("1. TOKENIZATION  (byte-pair encoding)")
    corpus = ("low low low low low lower lower newest newest newest newest "
              "newest newest widest widest widest")
    print("corpus:", corpus)
    tok = BPETokenizer.train(corpus, 256 + 6, verbose=True)
    for s in ["lowest", " lower", "newer"]:
        ids = tok.encode(s)
        print(f"  encode({s!r}) = {[tok.token_str(i) for i in ids]} = {ids}")
        assert tok.decode(ids) == s                  # encoding is lossless


def stage_embedding() -> None:
    header("2. EMBEDDING  (a table lookup)")
    V, d = 5, 3
    E = np.array([[0.1, 0.3, -0.2],
                  [0.7, -0.5, 0.2],
                  [-0.4, 0.6, 0.9],
                  [0.0, 0.8, -0.1],
                  [0.5, 0.5, 0.5]])                  # (V, d)
    P = np.array([[0.00, 0.10, 0.00],
                  [0.10, 0.00, 0.10],
                  [0.20, -0.10, 0.00]])              # (T, d) position embeddings
    ids = np.array([2, 0, 3])
    onehot = np.eye(V)[ids]                          # (T, V)
    print("token ids x =", ids)
    print("E =\n", E)
    print("E[x] (row lookup) =\n", E[ids])
    assert np.allclose(onehot @ E, E[ids])           # lookup == one-hot times E
    print("one-hot(x) @ E gives the same thing: True")
    print("X0 = E[x] + P =\n", E[ids] + P)


def stage_attention() -> np.ndarray:
    header("3. SELF-ATTENTION  (single head, T = 3, d_k = 2)")
    Q = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    K = np.array([[1.0, 1.0], [0.0, 2.0], [2.0, 0.0]])
    Vm = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    d_k = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d_k)
    M = np.triu(np.full((3, 3), -np.inf), k=1)       # -inf above the diagonal
    A = softmax(S + M)
    out = A @ Vm
    print("Q =\n", Q, "\nK =\n", K, "\nV =\n", Vm)
    print("scores S = Q K^T / sqrt(d_k) =\n", S)
    print("masked S + M =\n", S + M)
    print("attention weights A = softmax(S + M)  (each row sums to 1) =\n", A)
    print("output A V =\n", out)
    assert np.allclose(A.sum(-1), 1.0) and np.allclose(np.triu(A, 1), 0.0)
    return A


def stage_feedforward() -> None:
    header("4. FEED-FORWARD NETWORK  (d = 2 -> 4d = 8 -> 2, GELU)")
    rng = np.random.default_rng(0)
    W1, b1 = rng.normal(0, 0.5, (2, 8)), np.zeros(8)
    W2, b2 = rng.normal(0, 0.5, (8, 2)), np.zeros(2)
    x = np.array([1.0, -0.5])
    hidden = x @ W1 + b1
    print("x =", x)
    print("pre-activation W1 x + b1 =", hidden)
    print("GELU(...)                =", gelu(hidden))
    print("FFN(x) = W2 GELU(...) + b2 =", gelu(hidden) @ W2 + b2)
    xs = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    print("\n   x     ReLU    GELU    SiLU")
    for v, r, g, s in zip(xs, np.maximum(xs, 0), gelu(xs), xs / (1 + np.exp(-xs))):
        print(f"{v:5.1f}  {r:6.3f}  {g:6.3f}  {s:6.3f}")
    print("layer_norm([2, 4, 6, 8]) =", layer_norm(np.array([2.0, 4.0, 6.0, 8.0])))


def stage_softmax() -> np.ndarray:
    header("5. SOFTMAX AND TEMPERATURE")
    words = ["mat", "sofa", "floor", "roof", "moon"]
    z = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
    print("candidates:", words)
    print("logits z  :", z)
    for tau in [0.5, 1.0, 2.0]:
        print(f"softmax(z / {tau}) =", softmax(z / tau))
    return softmax(z)


def stage_sampling(p: np.ndarray) -> None:
    header("6. CHOOSING THE NEXT TOKEN")
    words = ["mat", "sofa", "floor", "roof", "moon"]
    print("p =", p)
    print("greedy  : argmax ->", words[int(p.argmax())])
    k = 2
    pk = np.where(np.arange(len(p)) < k, p, 0.0)     # p is already sorted here
    print(f"top-k (k={k}) : keep {words[:k]}, renormalise ->", pk / pk.sum())
    top_p = 0.9
    before = np.cumsum(p) - p                        # mass strictly before each token
    keep = before <= top_p
    pn = np.where(keep, p, 0.0)
    print(f"top-p (p={top_p}): cumulative =", np.cumsum(p))
    print(f"                keep {[w for w, m in zip(words, keep) if m]}, renormalise ->", pn / pn.sum())
    rng = np.random.default_rng(0)
    draws = rng.choice(len(p), size=10_000, p=p)
    print("10,000 samples from p, empirical frequencies:", np.bincount(draws, minlength=5) / 10_000)


def stage_loss(p: np.ndarray) -> None:
    header("7. CROSS-ENTROPY LOSS AND ITS GRADIENT")
    z = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
    y = 1                                            # suppose the true next token is 'sofa'
    loss = -math.log(p[y])
    print(f"true token index y = {y}; loss = -log p[y] = {loss:.4f}")
    grad = p.copy()
    grad[y] -= 1.0                                   # dL/dz = p - onehot(y)
    print("analytic gradient dL/dz = p - onehot(y) =", grad)
    num, eps = np.zeros_like(z), 1e-6
    for i in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        num[i] = (-math.log(softmax(zp)[y]) + math.log(softmax(zm)[y])) / (2 * eps)
    print("finite-difference gradient              =", num)
    assert np.allclose(grad, num, atol=1e-6)
    print(f"sanity: a uniform guess over V=512 tokens costs ln 512 = {math.log(512):.3f} nats")


def stage_adamw() -> None:
    header("8. ONE AdamW STEP BY HAND")
    theta, g = np.array([0.50, -0.30]), np.array([0.20, -0.10])
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.95, 1e-8, 0.1
    m, v = np.zeros(2), np.zeros(2)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    m_hat, v_hat = m / (1 - b1 ** 1), v / (1 - b2 ** 1)       # bias correction, t = 1
    new = theta - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * theta)
    print("theta =", theta, " gradient g =", g)
    print("m =", m, " v =", v, " m_hat =", m_hat, " v_hat =", v_hat)
    print("step direction m_hat / sqrt(v_hat) =", m_hat / (np.sqrt(v_hat) + eps),
          " <- first step is +-1 per coordinate: Adam normalises scale")
    np.set_printoptions(precision=6, suppress=True)
    print("theta_new =", new)
    np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------- Part B
def real_model(prompt: str = "O Romeo, Romeo! wherefore art thou") -> None:
    if not os.path.exists("out/ckpt.pt"):
        print("\n(no out/ckpt.pt yet -- run train.py to see Part B)")
        return
    import torch
    import torch.nn.functional as F
    from model import GPT, GPTConfig
    from tokenizer import load_tokenizer

    header("PART B: THE SAME STAGES INSIDE THE TRAINED MODEL")
    tok = load_tokenizer("out/tokenizer.json")
    ckpt = torch.load("out/ckpt.pt", map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tok.encode(prompt)
    print(f"prompt  : {prompt!r}")
    print(f"tokens  : {[tok.token_str(i) for i in ids]}")
    print(f"ids     : {ids}      (T = {len(ids)})")
    idx = torch.tensor([ids])
    with torch.no_grad():
        x = model.tok_emb(idx) + model.pos_emb(torch.arange(len(ids)))
        print(f"embedding X0        : shape {tuple(x.shape)}  (B, T, d)")
        for i, block in enumerate(model.blocks):
            x = block(x)
            a = block.attn.last_attn                 # (B, h, T, T)
            print(f"block {i + 1}: X shape {tuple(x.shape)}, attention {tuple(a.shape)} (B, h, T, T); "
                  f"last token, head 1 attends most to "
                  f"{tok.token_str(ids[int(a[0, 0, -1].argmax())])!r}")
        logits = model.head(model.ln_f(x))
        print(f"logits Z            : shape {tuple(logits.shape)}  (B, T, V)")
        p = F.softmax(logits[0, -1], dim=-1)
        top = torch.topk(p, 5)
        print("next-token distribution, top 5:")
        for prob, i in zip(top.values, top.indices):
            print(f"    {tok.token_str(int(i))!r:>10}  p = {prob:.3f}")
        torch.manual_seed(0)
        out = model.generate(idx, max_new_tokens=40, temperature=0.8, top_k=50)
        print("continuation (T=0.8, top-k=50):", repr(tok.decode(out[0].tolist())))


if __name__ == "__main__":
    stage_tokenization()
    stage_embedding()
    stage_attention()
    stage_feedforward()
    p = stage_softmax()
    stage_sampling(p)
    stage_loss(p)
    stage_adamw()
    real_model()
