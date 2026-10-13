"""Correctness gate for kernel.py: batched matmul vs an einsum reference.

Any optimization the loop applies must keep this green.
"""
import numpy as np

import kernel


def main():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((4, 64, 64))
    B = rng.standard_normal((4, 64, 64))
    ref = np.einsum("bij,bjk->bik", A, B)
    got = np.asarray(kernel.run(A, B), dtype=np.float64)
    if got.shape != ref.shape or not np.allclose(got, ref, atol=1e-3):
        print("KERNEL WRONG")
        raise SystemExit(1)
    print("kernel correct (v%d)" % getattr(kernel, "VERSION", 0))


if __name__ == "__main__":
    main()
