"""Provided tests — YOU supply these; the agent earns green. Do not let it
edit this file. Correctness vs a loop reference + a boundary case."""
import numpy as np

from src.stencil import apply_stencil


def reference(grid):
    out = grid.astype(float).copy()
    for i in range(1, grid.shape[0] - 1):
        for j in range(1, grid.shape[1] - 1):
            out[i, j] = 0.25 * (
                grid[i - 1, j] + grid[i + 1, j] + grid[i, j - 1] + grid[i, j + 1]
            )
    return out


def test_matches_reference():
    rng = np.random.default_rng(42)
    g = rng.standard_normal((16, 12))
    np.testing.assert_allclose(apply_stencil(g), reference(g), atol=1e-12)


def test_boundary_rows_and_cols_unchanged():
    g = np.arange(30, dtype=float).reshape(5, 6)
    out = apply_stencil(g)
    np.testing.assert_array_equal(out[0, :], g[0, :])
    np.testing.assert_array_equal(out[-1, :], g[-1, :])
    np.testing.assert_array_equal(out[:, 0], g[:, 0])
    np.testing.assert_array_equal(out[:, -1], g[:, -1])


def test_input_not_mutated():
    g = np.ones((4, 4))
    g0 = g.copy()
    apply_stencil(g)
    np.testing.assert_array_equal(g, g0)
