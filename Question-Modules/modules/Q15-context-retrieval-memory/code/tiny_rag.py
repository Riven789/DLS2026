"""tiny_rag.py -- a minimal embed/index/retrieve loop to show how RAG grounds answers (Module Q15).
Pure stdlib. Uses a toy bag-of-words 'embedding' + cosine similarity (NO external model), so it runs
anywhere. Demonstrates the retrieval MECHANISM; a real system swaps in a neural embedder + vector DB.
"""
import math, re
from collections import Counter

DOCS = [
    "The KV cache stores keys and values for each token so generation does not recompute the past.",
    "Retrieval-augmented generation fetches relevant documents and places them in the context window.",
    "Self-attention compute grows quadratically with sequence length.",
    "Paris is the capital of France; the Eiffel Tower is located there.",
    "Grouped-query attention reduces the size of the KV cache by sharing keys and values across heads.",
]

def vec(text):
    return Counter(re.findall(r"[a-z]+", text.lower()))

def cosine(a, b):
    dot = sum(a[t]*b[t] for t in a)
    na = math.sqrt(sum(v*v for v in a.values())); nb = math.sqrt(sum(v*v for v in b.values()))
    return dot/(na*nb) if na and nb else 0.0

def retrieve(query, k=2):
    q = vec(query)
    scored = sorted(((cosine(q, vec(d)), d) for d in DOCS), reverse=True)
    return scored[:k]

if __name__ == "__main__":
    index = {i: d for i, d in enumerate(DOCS)}
    print(f"indexed {len(index)} documents\n")
    for query in ["How does the KV cache reduce recomputation?",
                  "What is the capital of France?",
                  "How do you shrink the KV cache?"]:
        print(f"QUERY: {query}")
        for score, doc in retrieve(query, k=2):
            print(f"  [{score:.3f}] {doc}")
        top = retrieve(query, 1)[0][1]
        print(f"  -> grounding the answer on: {top!r}\n")
    print("Lesson: retrieval puts only the RELEVANT chunks in context, so the model answers from")
    print("supplied evidence (groundable + citable) instead of parametric guesswork.")
