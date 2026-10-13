"""
walkthrough.py -- the worked examples on the slides, computed step by step.

    python walkthrough.py
"""
import math

import numpy as np

from data import load_task
from mlp_numpy import NumpyMLP, softmax

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def header(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ---------------------------------------------------------------- one neuron
header("1. ONE NEURON:  a = phi(w . x + b)")
x, w, b = np.array([2.0, 1.0]), np.array([0.5, -1.0]), 0.5
z = w @ x + b
print(f"x = {x}, w = {w}, b = {b}")
print(f"z = w.x + b = {w[0]}*{x[0]} + {w[1]}*{x[1]} + {b} = {z}")
print(f"ReLU(z) = {max(z, 0):.3f}   sigmoid(z) = {1 / (1 + math.exp(-z)):.3f}   tanh(z) = {math.tanh(z):.3f}")

# ---------------------------------------------------------------- forward / backward / update
header("2. A 2-2-1 NETWORK: FORWARD, BACKWARD, ONE SGD STEP")
net = NumpyMLP([2, 2, 1], task="regression")
net.W = [np.array([[0.5, -1.0], [0.25, 0.5]]), np.array([[2.0], [-1.0]])]
net.b = [np.array([0.0, -0.5]), np.array([0.5])]
X, y = np.array([[1.0, 2.0]]), np.array([2.0])

out = net.forward(X)
print("x  =", X[0], "  target y =", y[0])
print("z1 = x W1 + b1 =", net.z[0][0])
print("a1 = ReLU(z1)  =", net.a[1][0])
print("yhat = a1 W2 + b2 =", out[0, 0])
loss0 = net.loss(X, y)
print("loss = (yhat - y)^2 =", loss0)

gW, gb = net.backward(y)
print("\nbackward pass")
print("dL/dyhat = 2 (yhat - y) =", 2 * (out[0, 0] - y[0]))
print("dL/dW2 =", gW[1].ravel(), "  dL/db2 =", gb[1])
print("delta1 = (dL/dyhat * W2^T) * ReLU'(z1) =", (2 * (out - y) @ net.W[1].T * (net.z[0] > 0))[0])
print("dL/dW1 =\n", gW[0], "\ndL/db1 =", gb[0])

for lr in (0.01, 0.1):
    trial = NumpyMLP([2, 2, 1], task="regression")
    trial.W, trial.b = [w.copy() for w in net.W], [v.copy() for v in net.b]
    trial.sgd_step(X, y, lr)
    print(f"\nafter one SGD step with lr = {lr}:")
    print("  W1 =", trial.W[0].tolist(), " b1 =", trial.b[0].tolist())
    print("  W2 =", trial.W[1].ravel().tolist(), " b2 =", trial.b[1].tolist())
    print(f"  new yhat = {trial.forward(X)[0, 0]:.4f}   new loss = {trial.loss(X, y):.4f}   (was {loss0})")

# ---------------------------------------------------------------- classification head
header("3. CLASSIFICATION HEAD: SOFTMAX AND CROSS-ENTROPY")
logits, label = np.array([[2.0, 0.5, -1.0]]), 1
p = softmax(logits)[0]
print("logits =", logits[0], " -> softmax p =", p, " (sums to", p.sum(), ")")
print(f"true class = {label}: loss = -ln p[{label}] = {-math.log(p[label]):.4f}")
g = p.copy()
g[label] -= 1
print("gradient dL/dlogits = p - onehot =", g)
print(f"binary case: sigmoid(1.2) = {1 / (1 + math.exp(-1.2)):.4f}; "
      f"softmax([0, 1.2])[1] = {softmax(np.array([[0, 1.2]]))[0, 1]:.4f}  (the same thing)")

# ---------------------------------------------------------------- regression head
header("4. REGRESSION HEAD: MSE, RMSE, MAE, R^2")
true, pred = np.array([18.0, 24.0, 31.0, 15.0]), np.array([20.0, 23.0, 28.0, 16.0])
err = pred - true
print("true =", true, " predicted =", pred, " errors =", err)
print(f"MSE = {np.mean(err ** 2):.3f}   RMSE = {np.sqrt(np.mean(err ** 2)):.3f}   MAE = {np.mean(np.abs(err)):.3f}"
      f"   R^2 = {1 - np.sum(err ** 2) / np.sum((true - true.mean()) ** 2):.3f}")

# ---------------------------------------------------------------- encoding a real row
header("5. ENCODING A REAL ROW (Titanic, statistics from the training split)")
d = load_task("titanic", seed=0)
enc = d["encoder"]
i = next(k for k, r in enumerate(d["rows_train"]) if r["age"] != "" and r["alone"] == "True")
row = d["rows_train"][i]
print("raw:", {k: row[k] for k in enc.numerical + enc.logical + enc.categorical}, "| label:", row["alive"])
for name, v in zip(enc.feature_names, d["X_train"][i]):
    print(f"  {name:26s} {v: .3f}")
m, s = enc.stats["age"]
print(f"age: ({row['age']} - {m:.2f}) / {s:.2f} = {(float(row['age']) - m) / s:.3f}")
m, s = enc.stats["fare"]
print(f"fare: (ln(1 + {row['fare']}) - {m:.3f}) / {s:.3f} = {(math.log1p(float(row['fare'])) - m) / s:.3f}")
print("label:", row["alive"], "->", d["y_train"][i], "  classes:", d["classes"])
fares = np.array([float(r["fare"]) for r in d["rows_train"]])
print(f"raw fare: min {fares.min():.2f}, median {np.median(fares):.2f}, max {fares.max():.2f}")
ages = [r["age"] for r in d["rows_train"]]
print(f"missing ages in the training split: {ages.count('')} of {len(ages)}")
