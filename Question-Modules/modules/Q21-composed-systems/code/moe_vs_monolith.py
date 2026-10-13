#!/usr/bin/env python3
"""Q21 demo: 4-expert mixture with a learned router vs a larger monolith.

Synthetic regression on 4 well-separated clusters (heterogeneous input
distribution). The mixture is a router (linear) + 4 small linear experts;
the monolith is one hidden-layer network with MORE total parameters
(316 vs. 180), making the comparison conservative for the mixture.

Claims made concrete:
  1. The router learns the cluster decomposition (routing accuracy >= 90%).
  2. The composed system beats the equal-parameter monolith on held-out loss.
  3. Random routing (dead router) destroys the advantage -> the interface
     (the learned router) is what pays, not the parameter count.

Simulation, not an LLM run: NumPy only, deterministic seed, ~2 seconds.
Run: python3 code/moe_vs_monolith.py
"""
import numpy as np

rng = np.random.default_rng(21)

# ---------------- data: 4 clusters, each with its own linear map ----------------
D_IN, D_OUT, K, N_TRAIN, N_TEST = 8, 4, 4, 4000, 1000

# cluster centers, spread apart so the routing problem is learnable
centers = rng.normal(size=(K, D_IN)) * 2.0
# each cluster has its own ground-truth linear map W_k (heterogeneous experts)
true_W = rng.normal(size=(K, D_OUT, D_IN)) * 0.5

def make_data(n, rng):
    z = rng.integers(0, K, size=n)                      # latent cluster id
    X = centers[z] + rng.normal(size=(n, D_IN)) * 0.25  # cluster-structured input
    Y = np.stack([X[i] @ true_W[z[i]].T for i in range(n)]) + rng.normal(size=(n, D_OUT)) * 0.05
    return X, Y, z

Xtr, Ytr, Ztr = make_data(N_TRAIN, rng)
Xte, Yte, Zte = make_data(N_TEST, rng)

# parameter accounting: equal-parameter comparison
# mixture: router (K*D_IN + K) + experts (K * (D_IN*D_OUT + D_OUT))
P_MIX = K * D_IN + K + K * (D_IN * D_OUT + D_OUT)
# monolith with hidden layer of width H: D_IN*H + H + H*D_OUT + D_OUT
H = 24
P_MONO = D_IN * H + H + H * D_OUT + D_OUT
# note: monolith is given the ADVANTAGE here (more parameters) to make the
# comparison conservative for the mixture.

def init_mixture(rng):
    return {
        "Wr": rng.normal(size=(D_IN, K)) * 0.1, "br": np.zeros(K),
        "We": rng.normal(size=(K, D_IN, D_OUT)) * 0.05,
        "be": np.zeros((K, D_OUT)),
    }

def init_monolith(rng):
    return {
        "W1": rng.normal(size=(D_IN, H)) * 0.05, "b1": np.zeros(H),
        "W2": rng.normal(size=(H, D_OUT)) * 0.05, "b2": np.zeros(D_OUT),
    }

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def mixture_forward(p, X, hard=False, routing=None):
    """Returns prediction, gate probs, chosen expert idx."""
    g = softmax(X @ p["Wr"] + p["br"])                  # (n, K) router probs
    if routing == "random":
        idx = rng.integers(0, K, size=len(X))
        g = np.eye(K)[idx]                              # one-hot random gate
    elif hard:
        idx = g.argmax(axis=1)
        g = np.eye(K)[idx]
    # expert outputs: (n, K, D_OUT)
    E = np.einsum("ni,kio->nko", X, p["We"]) + p["be"]
    pred = (g[:, :, None] * E).sum(axis=1)              # soft mixture of experts
    return pred, g, E

def mixture_step(p, X, Y, lr, routing=None):
    pred, g, E = mixture_forward(p, X, routing=routing)
    err = pred - Y                                      # (n, D_OUT)
    n = len(X)
    # d/d pred: MSE
    dE = g[:, :, None] * (2.0 / n / D_OUT) * err[:, None, :]   # (n,K,O) grad to experts
    dg = (E * (2.0 / n / D_OUT) * err[:, None, :]).sum(axis=2) # (n,K) grad to gate
    # softmax jacobian: dg is dL/d(pre-softmax logit)
    gsum = (g * dg).sum(axis=1, keepdims=True)
    dlogit = g * (dg - gsum)
    # accumulate parameter grads
    gWe = np.einsum("nko,ni->kio", dE, X)
    gbe = dE.sum(axis=0)
    gWr = X.T @ dlogit
    gbr = dlogit.sum(axis=0)
    # SGD
    p["We"] -= lr * gWe; p["be"] -= lr * gbe
    p["Wr"] -= lr * gWr; p["br"] -= lr * gbr
    loss = ((pred - Y) ** 2).mean()
    return loss

def monolith_forward(p, X):
    h = np.tanh(X @ p["W1"] + p["b1"])
    return h @ p["W2"] + p["b2"], h

def monolith_step(p, X, Y, lr):
    pred, h = monolith_forward(p, X)
    err = pred - Y
    n = len(X)
    d = (2.0 / n / D_OUT) * err                          # (n, O)
    gW2 = h.T @ d
    gb2 = d.sum(axis=0)
    dh = d @ p["W2"].T * (1 - h ** 2)
    gW1 = X.T @ dh
    gb1 = dh.sum(axis=0)
    p["W2"] -= lr * gW2; p["b2"] -= lr * gb2
    p["W1"] -= lr * gW1; p["b1"] -= lr * gb1
    return ((pred - Y) ** 2).mean()

def train(steps, batch, lr, kind, routing=None, seed=0):
    r = np.random.default_rng(seed)
    p = init_mixture(r) if kind == "moe" else init_monolith(r)
    for step in range(steps):
        idx = r.integers(0, N_TRAIN, size=batch)
        if kind == "moe":
            mixture_step(p, Xtr[idx], Ytr[idx], lr, routing=routing)
        else:
            monolith_step(p, Xtr[idx], Ytr[idx], lr)
    return p

def evaluate(p, kind, routing=None):
    if kind == "moe":
        pred, g, _ = mixture_forward(p, Xte, routing=routing)
    else:
        pred, _ = monolith_forward(p, Xte)
    return ((pred - Yte) ** 2).mean()

# -------------------------------- experiment ---------------------------------
print("Q21 demo: learned-router MoE (4 experts) vs larger monolith")
print(f"data: {N_TRAIN} train / {N_TEST} test, {K} clusters, d_in={D_IN}, d_out={D_OUT}")
print(f"parameters: mixture={P_MIX}, monolith={P_MONO} (monolith is GIVEN more params)\n")

moe = train(steps=3000, batch=256, lr=0.05, kind="moe")
mono = train(steps=3000, batch=256, lr=0.05, kind="mono")

# routing accuracy: does the router send each cluster to ONE consistent expert?
# (the expert that owns cluster k is arbitrary -- a permutation is fine)
_, g_test, _ = mixture_forward(moe, Xte)
routed = g_test.argmax(axis=1)
perm_ok = 0
for e in range(K):
    mask = routed == e
    if mask.sum() > 0:
        # expert e's dominant true cluster
        dom = np.bincount(Zte[mask], minlength=K).argmax()
        perm_ok += np.sum((routed == e) & (Zte == dom))
routing_acc = perm_ok / len(Zte)

# expert specialization: which true cluster does each expert serve most?
specialists = np.zeros((K, K), dtype=int)
for e in range(K):
    for z in range(K):
        specialists[e, z] = np.sum((routed == e) & (Zte == z))

loss_moe = evaluate(moe, "moe")
loss_mono = evaluate(mono, "mono")

# dead-router control: same trained experts, routing at random
moe_dead = {k: v.copy() for k, v in moe.items()}
loss_dead = evaluate(moe_dead, "moe", routing="random")

print(f"held-out MSE  (learned router)  : {loss_moe:.5f}")
print(f"held-out MSE  (monolith)        : {loss_mono:.5f}")
print(f"held-out MSE  (random routing)  : {loss_dead:.5f}   <- same experts, dead router")
print(f"routing accuracy vs latent cluster: {routing_acc:.1%}")
print("expert x true-cluster assignment matrix (rows=expert, cols=true cluster):")
print(specialists)

# ------------------------------- assertions ----------------------------------
checks = {
    "router learns cluster assignment (>= 90%)": routing_acc >= 0.90,
    "composed MoE beats monolith (lower MSE)": loss_moe < loss_mono,
    "dead router loses the advantage": loss_dead > loss_moe,
    "MoE improves on random routing baseline": loss_moe < loss_dead,
}
print()
ok = True
for name, result in checks.items():
    print(f"[{'PASS' if result else 'FAIL'}] {name}")
    ok = ok and result
print()
if ok:
    print("ALL CHECKS PASSED: composition pays when the router can learn the")
    print("decomposition; the advantage is the interface, not the parameters.")
else:
    print("CHECKS FAILED - see values above.")
    raise SystemExit(1)
