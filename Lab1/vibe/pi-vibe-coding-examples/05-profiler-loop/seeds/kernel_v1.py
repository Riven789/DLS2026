"""Batched matmul kernel — v1 (seed): broadcast-and-sum, no BLAS.

C[b,i,k] = sum_j A[b,i,j] * B[b,j,k], materialized as a (BATCH, N, N, N)
temporary. Memory-bound and allocation-heavy on purpose — this is the
starting point the loop optimizes away.
"""
import numpy as np

VERSION = 1


def run(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    return (A[:, :, :, None] * B[:, None, :, :]).sum(axis=2)
