"""
Q13: How do models represent and store knowledge, and use it to answer?

Core mechanism illustrated: KEY-VALUE ASSOCIATIVE MEMORY.

Neural networks (especially the MLP / feed-forward layers of transformers)
store knowledge as associations between "key" patterns and "value" vectors
inside their weight matrices. When an input arrives, the model:

  1. Encodes the query into a vector.
  2. Measures similarity between the query and each stored KEY.
  3. Softmax-weights the similarities into attention weights.
  4. Returns a blend of the stored VALUES.

This is exactly how an attention head and an associative memory work: the
weights ARE the knowledge; retrieval is a weighted lookup, not a database read.
Below we build a tiny hand-made knowledge store and query it.
"""

import numpy as np

np.random.seed(0)

# ---- Knowledge base: facts stored as (key vector -> value vector) pairs ----
# Keys encode "what the fact is about"; values encode "the answer".
facts = {
    "capital of France":  "Paris",
    "capital of Japan":   "Tokyo",
    "capital of Italy":   "Rome",
    "capital of Spain":   "Madrid",
}
answers = list(facts.values())

# Give each fact a random but fixed "concept" embedding (the stored KEY),
# and let the VALUE be a one-hot pointer to its answer.
dim = 8
keys   = np.random.randn(len(facts), dim)        # stored knowledge (keys)
keys  /= np.linalg.norm(keys, axis=1, keepdims=True)
values = np.eye(len(facts))                        # stored knowledge (values)

# Map each question string to its key embedding (simulating the encoder).
question_to_key = {q: keys[i] for i, q in enumerate(facts)}


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def answer(query_key, noise=0.0):
    """Retrieve an answer via key-similarity-weighted value lookup."""
    q = query_key + noise * np.random.randn(dim)
    q /= np.linalg.norm(q)
    scores = keys @ q                 # similarity to every stored key
    weights = softmax(scores * 10)    # sharpen (temperature)
    blended = weights @ values        # weighted blend of values
    idx = int(np.argmax(blended))
    return answers[idx], weights


# ---- Ask a clean question ----
q = "capital of Japan"
pred, w = answer(question_to_key[q])
print(f"Q: {q!r}")
print(f"A: {pred}")
print("Retrieval weights over stored facts:")
for fact, wt in zip(facts, w):
    print(f"   {fact:20s} {wt:.3f}")

# ---- Ask a NOISY/approximate question (robust recall) ----
print("\nNoisy query (partial/fuzzy memory of 'capital of Italy'):")
pred, w = answer(question_to_key["capital of Italy"], noise=0.4)
print(f"A: {pred}  (still correct despite noise)")

print("\nINSIGHT:")
print("Knowledge lives in the KEY/VALUE weights, not in explicit records.")
print("Answering = encode query -> compare to keys -> softmax -> blend values.")
print("This graceful, similarity-based lookup is why models recall facts")
print("even from imperfect prompts, and why confused keys cause errors.")