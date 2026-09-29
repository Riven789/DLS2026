"""First vectorized attempt — fast, but numerically naive.

float32 "for speed", no max-shift: exp() overflows to inf for logits > ~88,
and inf/inf = NaN. The oracle harness catches it on the first batch.
"""
import numpy as np


def fast_softmax(x):
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
