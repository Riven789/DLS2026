# Expected output — Problem 3 — Iterate-until-green (the canonical loop)

The full arc of the canonical verification-gated loop, in three iterations:

1. Iteration 1: `src/stencil.py` doesn't exist — pytest exits nonzero on a
   collection error. Attempt 1 implements the stencil with `np.roll`, which
   wraps around at the edges (a classic).
2. Iteration 2: 2 failed, 1 passed — the failure tail (interior mismatches at
   the boundary rows + boundary test) goes straight into the next prompt.
   Attempt 2 makes the smallest fix: interior-slice update, edges untouched.
3. Iteration 3: `3 passed` → `OK: green in 3`.

The verdict driving the loop is pytest's EXIT CODE — the model's prose never
gates anything.

## Captured run

Captured from one run of `run.sh` with the bundled mock `pi` in a 2-vCPU Linux container (Python 3.11, numpy 2.4.4, OpenBLAS). With the mock, the loop *structure* below is deterministic; wall-clock-derived numbers are not.

```text

==================================== ERRORS ====================================
_______________________ ERROR collecting test_stencil.py _______________________
ImportError while importing test module '/home/claude/pi-vibe-coding-examples/03-iterate-until-green/test_stencil.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
test_stencil.py:5: in <module>
    from src.stencil import apply_stencil
E   ModuleNotFoundError: No module named 'src.stencil'
=========================== short test summary info ============================
ERROR test_stencil.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.25s
[pi] wrote src/stencil.py — vectorized neighbour average via np.roll
[pi] tests failed on missing module — implemented apply_stencil; rerun pytest
FF.                                                                      [100%]
=================================== FAILURES ===================================
____________________________ test_matches_reference ____________________________

    def test_matches_reference():
        rng = np.random.default_rng(42)
        g = rng.standard_normal((16, 12))
>       np.testing.assert_allclose(apply_stencil(g), reference(g), atol=1e-12)
E       AssertionError: 
E       Not equal to tolerance rtol=1e-07, atol=1e-12
E       
E       Mismatched elements: 52 / 192 (27.1%)
E       First 5 mismatches are at indices:
E        [0, 0]: 0.2774600671141234 (ACTUAL), 0.30471707975443135 (DESIRED)
E        [0, 1]: 0.6004479959731945 (ACTUAL), -1.0399841062404955 (DESIRED)
E        [0, 2]: -0.010709319607743295 (ACTUAL), 0.7504511958064572 (DESIRED)
E        [0, 3]: -0.23839693641768214 (ACTUAL), 0.9405647163912139 (DESIRED)
E        [0, 4]: 0.108973108018252 (ACTUAL), -1.9510351886538364 (DESIRED)
E       Max absolute difference among violations: 2.0600083
E       Max relative difference among violations: 39.507746
E        ACTUAL: array([[ 0.27746 ,  0.600448, -0.010709, -0.238397,  0.108973, -0.31158 ,
E               -0.139184, -0.290839, -0.680577,  0.458149,  0.717739,  0.212517],
E              [ 0.212275, -0.214644,  0.387677,  0.535567, -0.839119,  0.093961,...
E        DESIRED: array([[ 0.304717, -1.039984,  0.750451,  0.940565, -1.951035, -1.30218 ,
E                0.12784 , -0.316243, -0.016801, -0.853044,  0.879398,  0.777792],
E              [ 0.066031, -0.214644,  0.387677,  0.535567, -0.839119,  0.093961,...

test_stencil.py:21: AssertionError
____________________ test_boundary_rows_and_cols_unchanged _____________________

    def test_boundary_rows_and_cols_unchanged():
        g = np.arange(30, dtype=float).reshape(5, 6)
        out = apply_stencil(g)
>       np.testing.assert_array_equal(out[0, :], g[0, :])
E       AssertionError: 
E       Arrays are not equal
E       
E       Mismatched elements: 6 / 6 (100%)
E       First 5 mismatches are at indices:
E        [0]: 9.0 (ACTUAL), 0.0 (DESIRED)
E        [1]: 8.5 (ACTUAL), 1.0 (DESIRED)
E        [2]: 9.5 (ACTUAL), 2.0 (DESIRED)
E        [3]: 10.5 (ACTUAL), 3.0 (DESIRED)
E        [4]: 11.5 (ACTUAL), 4.0 (DESIRED)
E       Max absolute difference among violations: 9.
E       Max relative difference among violations: 7.5
E        ACTUAL: array([ 9. ,  8.5,  9.5, 10.5, 11.5, 11. ])
E        DESIRED: array([0., 1., 2., 3., 4., 5.])

test_stencil.py:27: AssertionError
=========================== short test summary info ============================
FAILED test_stencil.py::test_matches_reference - AssertionError: 
FAILED test_stencil.py::test_boundary_rows_and_cols_unchanged - AssertionError: 
2 failed, 1 passed in 0.14s
[pi] wrote src/stencil.py — smallest fix: np.roll wraps at the edges; boundary must pass through
[pi] root cause: periodic wraparound at the boundary. Interior-slice update, edges copied unchanged. Tests untouched.
...                                                                      [100%]
3 passed in 0.16s
OK: green in 3
```

## What varies between runs/machines

- pytest durations; the assertion-diff arrays elide differently by terminal
  width.

With a real `pi`: green typically lands in 2-4 iterations (sometimes attempt 1
is simply correct → `OK: green in 2`); the stop rule caps it at 8. The mock
scripts the classic wraparound bug on attempt 1 so the loop's second turn is
visible.
