"""
Q11. Why do models hallucinate, and can they know when they are unsure?

Core mechanism illustrated here:
A language/classification model outputs a probability distribution via softmax.
"Hallucination" happens when the model confidently picks an answer even when
its evidence is weak or off-distribution. The softmax ALWAYS produces a
normalized distribution that sums to 1 --- it is forced to "say something"
even for inputs it has never seen. So high confidence does NOT imply correctness.

BUT: the model CAN partly know it is unsure. The entropy of its output
distribution measures uncertainty. Flat distributions (high entropy) signal
"I don't know", while peaked distributions (low entropy) signal confidence.
The danger: for out-of-distribution inputs the model is sometimes confidently
wrong (low entropy, wrong answer) --- that is a hallucination.

This script trains a tiny linear classifier on 2 clusters, then probes it with
in-distribution and far out-of-distribution inputs and inspects confidence+entropy.
"""

import numpy as np

rng = np.random.default_rng(0)

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def entropy(p):
    return -np.sum(p * np.log(p + 1e-12), axis=-1)

# --- training data: two well-separated classes ---
A = rng.normal([-3, 0], 0.5, (100, 2))
B = rng.normal([ 3, 0], 0.5, (100, 2))
X = np.vstack([A, B])
y = np.array([0]*100 + [1]*100)
Y = np.eye(2)[y]

# --- train a simple linear softmax classifier via gradient descent ---
W = np.zeros((2, 2)); b = np.zeros(2)
for _ in range(400):
    P = softmax(X @ W + b)
    gW = X.T @ (P - Y) / len(X)
    gb = (P - Y).mean(axis=0)
    W -= 0.5 * gW; b -= 0.5 * gb

def probe(x, label):
    p = softmax(x @ W + b)[0]
    conf = p.max()
    pred = p.argmax()
    h = entropy(p[None])[0]
    print(f"{label:28s} pred={pred} conf={conf:.3f} entropy={h:.3f}")
    return conf, h

print("=== Probing the trained model ===")
probe(np.array([[-3.0, 0.0]]), "in-dist class 0")
probe(np.array([[ 3.0, 0.0]]), "in-dist class 1")
probe(np.array([[ 0.0, 0.0]]), "boundary (honestly unsure)")
probe(np.array([[ 0.0, 50.0]]), "FAR out-of-distribution")
probe(np.array([[-30.0, 0.0]]), "extreme extrapolation")

print("\n--- Insight ---")
print("Softmax is FORCED to output a valid distribution for ANY input, so the")
print("model always 'answers'. Near the boundary it honestly reports high")
print("entropy (knows it's unsure). But far out-of-distribution inputs can yield")
print("LOW entropy / HIGH confidence while being meaningless -> that is a")
print("hallucination: confident, fluent, and unjustified. Entropy helps detect")
print("uncertainty, but confidence alone is NOT a reliable signal of truth.")