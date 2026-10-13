"""Seed SpMV — naive CSR, pure-Python loops. Correct but slow.

Interface every candidate must keep:
    spmv(n, indptr, indices, data, x) -> y   (numpy array, length n)

No human writes the final kernel; the system evolves this one.
"""
import numpy as np


def spmv(n, indptr, indices, data, x):
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for jj in range(indptr[i], indptr[i + 1]):
            s += data[jj] * x[indices[jj]]
        y[i] = s
    return y
