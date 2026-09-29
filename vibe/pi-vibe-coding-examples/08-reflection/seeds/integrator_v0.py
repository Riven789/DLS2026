"""ODE integrator — flaky seed #1: forward Euler, float32 'for speed'.

Two stacked sins: O(dt) truncation error AND float32 accumulation
(y, t, and dt all rounded every step).
"""
import numpy as np


def integrate(f, y0, t0, t1, n):
    y = np.float32(y0)
    t = np.float32(t0)
    dt = np.float32((t1 - t0) / n)
    for _ in range(n):
        y = y + dt * np.float32(f(t, y))   # forward Euler step
        t = t + dt                          # t accumulated by repeated addition
    return float(y)
