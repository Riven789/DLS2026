"""Quick correctness gate for an SpMV candidate vs a dense reference (n=200).

Usage: python check.py [candidate.py]   (default: champion.py)
"""
import importlib.util
import sys

import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "champion.py"
    rng = np.random.default_rng(5)
    n = 200
    dense = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.05)
    np.fill_diagonal(dense, rng.standard_normal(n))       # no empty rows
    indptr, indices, data = [0], [], []
    for i in range(n):
        nz = np.nonzero(dense[i])[0]
        indices.extend(nz.tolist())
        data.extend(dense[i, nz].tolist())
        indptr.append(len(indices))
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    data = np.asarray(data)
    x = rng.standard_normal(n)
    spec = importlib.util.spec_from_file_location("cand_spmv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    y = np.asarray(mod.spmv(n, indptr, indices, data, x), dtype=float)
    if y.shape == (n,) and np.allclose(y, dense @ x, atol=1e-9):
        print(f"{path}: correct")
        return 0
    print(f"{path}: WRONG")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
