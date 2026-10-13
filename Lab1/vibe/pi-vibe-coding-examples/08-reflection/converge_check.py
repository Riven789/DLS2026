"""Convergence gate: integrate y' = -y, y(0)=1 to t=5; compare to exp(-5).

Known analytic solution, machine-checkable verdict: exits nonzero unless the
relative error beats 1e-6. This file is the verifier — the agent fixes
integrator.py, never this.
"""
import math

from integrator import integrate

TOL = 1e-6


def main():
    exact = math.exp(-5.0)
    got = integrate(lambda t, y: -y, y0=1.0, t0=0.0, t1=5.0, n=200_000)
    rel = abs(got - exact) / abs(exact)
    print(f"y(5) = {got!r}   exact = {exact!r}   rel_err = {rel:.3e}   tol = {TOL:.0e}")
    if not (rel < TOL):
        print("FAIL: not converged")
        raise SystemExit(1)
    print("PASS: converged")


if __name__ == "__main__":
    main()
