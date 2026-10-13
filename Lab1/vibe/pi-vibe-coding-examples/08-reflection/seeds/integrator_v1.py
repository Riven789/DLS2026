"""ODE integrator — flaky seed #2 (the bug class RECURS): RK4, but float32.

Right method order this time; wrong dtype. Rounding error random-walks
~ sqrt(n) * eps_f32 over 2e5 steps — stalls around 1e-5, misses 1e-6.
"""
import numpy as np


def integrate(f, y0, t0, t1, n):
    y = np.float32(y0)
    dt = np.float32((t1 - t0) / n)
    t0 = np.float32(t0)
    for i in range(n):
        t = t0 + np.float32(i) * dt
        k1 = np.float32(f(t, y))
        k2 = np.float32(f(t + np.float32(0.5) * dt, y + np.float32(0.5) * dt * k1))
        k3 = np.float32(f(t + np.float32(0.5) * dt, y + np.float32(0.5) * dt * k2))
        k4 = np.float32(f(t + dt, y + dt * k3))
        y = y + (dt / np.float32(6.0)) * (k1 + np.float32(2) * k2 + np.float32(2) * k3 + k4)
    return float(y)
