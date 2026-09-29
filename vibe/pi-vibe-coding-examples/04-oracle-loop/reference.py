"""The slow-but-obviously-correct oracle. The agent must not touch this.

Naive: explicit row loop, log-sum-exp shift per row. You can read it and
believe it — that's its whole job.
"""
import numpy as np


def reference_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        row = x[i]
        m = row.max()              # log-sum-exp shift
        e = np.exp(row - m)
        out[i] = e / e.sum()
    return out
