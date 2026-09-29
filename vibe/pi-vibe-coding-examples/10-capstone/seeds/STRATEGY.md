# Mutation strategy — round 0 (seed)

Optimization ideas for the CSR SpMV on this pattern (tridiagonal band +
~5 random nnz per row, n=6000, ~8 nnz/row — short rows, so per-row Python
overhead dominates):

- gather-vectorize: compute data * x[indices] in one shot; keep the row loop
- reduceat: replace the row loop with np.add.reduceat over indptr segments
- bincount: scatter-add products via np.bincount(row_ids, weights=...)
- scipy-csr: format change — hand the pattern to scipy.sparse's C matvec
- row-blocking: process rows in fixed-size blocks for locality
- prefetch/precompute: hoist x-independent index arrays out of the hot call

Apply ONE idea per mutation. Anything x-dependent must be recomputed per
call — cache only pattern data.
