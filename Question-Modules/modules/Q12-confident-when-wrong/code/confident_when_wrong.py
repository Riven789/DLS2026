"""
Module Q12: Why do models give confident answers even when wrong?

Core mechanism: A classifier's confidence comes from the softmax of its
output scores (logits). Softmax ALWAYS produces a probability distribution
that sums to 1, forcing the model to assign high probability to *some*
class -- even when the input is garbage or far outside the training data.
The model has no built-in notion of "I don't know." High confidence only
means one logit is larger than the others, NOT that the answer is correct.

This script trains a tiny linear classifier on two well-separated clusters,
then feeds it an out-of-distribution point far from everything it has seen.
We observe that the model reports very high confidence on this nonsense input.
"""

import numpy as np

rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# --- Training data: two clusters (class 0 near origin, class 1 far right) ---
n = 200
X0 = rng.normal([0, 0], 0.5, (n, 2))
X1 = rng.normal([4, 0], 0.5, (n, 2))
X = np.vstack([X0, X1])
y = np.array([0] * n + [1] * n)
Y = np.eye(2)[y]  # one-hot

# --- Train a linear model with gradient descent on softmax cross-entropy ---
W = rng.normal(0, 0.01, (2, 2))
b = np.zeros(2)
lr = 0.1
for _ in range(400):
    probs = softmax(X @ W + b)
    grad = (probs - Y) / len(X)
    W -= lr * (X.T @ grad)
    b -= lr * grad.sum(axis=0)

# --- In-distribution test point vs. far out-of-distribution point ---
in_dist = np.array([[4.0, 0.0]])       # clearly class 1
ood = np.array([[100.0, 100.0]])        # absurd, nothing like training data

for name, pt in [("in-distribution", in_dist), ("OUT-of-distribution", ood)]:
    p = softmax(pt @ W + b)[0]
    pred = p.argmax()
    print(f"{name:22s} point={pt[0]}  pred=class {pred}  "
          f"confidence={p[pred]*100:5.1f}%")

print("\nINSIGHT:")
print("The far-away garbage point gets an extreme, near-certain confidence")
print("even though it resembles NO training example. Softmax normalizes any")
print("logit gap into a crisp probability, so large unchecked logits ->")
print("high confidence. Confidence measures relative score, not correctness")
print("or familiarity -- hence models are confidently wrong on bad inputs.")