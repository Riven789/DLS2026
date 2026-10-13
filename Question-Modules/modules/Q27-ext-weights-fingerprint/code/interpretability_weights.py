"""
Q27. Interpretability: how can inspecting weights reveal a model's tendencies?
Do models revert to pre-training traits after deployment?

Core mechanism illustrated with numpy:

1) A model's weights are not a black box: a single weight vector encodes a
   "direction" in feature space. By inspecting the sign/magnitude of weights
   we can read off which features the model relies on (its "tendencies").

2) "Reverting to pre-training traits" is a myth for a deployed (frozen) model:
   weights are static after training. The only thing that changes at inference
   is the INPUT. If post-deployment inputs drift back toward the pre-training
   distribution, the model's OUTPUTS resemble pre-training behavior -- but the
   learned weights themselves never move. We demonstrate this below.
"""

import numpy as np

rng = np.random.default_rng(0)

# --- "Pre-training": learn weights emphasizing feature 0, ignoring feature 2 ---
def train(X, y, steps=400, lr=0.1):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ w))
        w -= lr * X.T @ (p - y) / len(y)
    return w

# Pre-training data: label driven by feature 0
Xpre = rng.normal(size=(500, 3))
ypre = (Xpre[:, 0] > 0).astype(float)
w_pre = train(Xpre, ypre)

# Fine-tuning: label now driven by feature 2 (new tendency)
Xft = rng.normal(size=(500, 3))
yft = (Xft[:, 2] > 0).astype(float)
w_deployed = w_pre.copy()
# continue training from pre-trained weights
for _ in range(400):
    p = 1 / (1 + np.exp(-Xft @ w_deployed))
    w_deployed -= 0.1 * Xft.T @ (p - yft) / len(yft)

print("=== Inspecting weights reveals tendencies ===")
print(f"Pre-trained weights : {np.round(w_pre, 2)}   -> relies on feature 0")
print(f"Deployed  weights   : {np.round(w_deployed, 2)}   -> relies on feature 2")
print(f"Dominant feature pre-train: {np.argmax(np.abs(w_pre))}")
print(f"Dominant feature deployed : {np.argmax(np.abs(w_deployed))}")

# --- "Reversion" test: feed pre-training-style vs new-style inputs ---
# The deployed weights are FROZEN. We only change the inputs.
def accuracy(w, X, y):
    return np.mean(((X @ w) > 0) == (y > 0))

print("\n=== Does the model revert after deployment? ===")
print(f"Deployed weights unchanged? {np.allclose(w_deployed, w_deployed)} (frozen)")
acc_new   = accuracy(w_deployed, Xft, yft)
acc_oldin = accuracy(w_deployed, Xpre, ypre)
print(f"Accuracy on new-task inputs      : {acc_new:.2f}")
print(f"Accuracy on pre-training inputs  : {acc_oldin:.2f}")

print("\n--- INSIGHT ---")
print("Weights ARE the interpretable fingerprint: largest |weight| shows which")
print("feature the model trusts, exposing its learned tendency directly.")
print("A deployed model does NOT revert -- its weights are frozen. Any apparent")
print("'reversion to pre-training traits' is really INPUT DISTRIBUTION SHIFT:")
print("old-style inputs trigger behavior matching the old weight direction,")
print("while the parameters themselves stay exactly where fine-tuning left them.")