"""scaling_fit.py -- fit and extrapolate a power-law scaling curve (Module Q07).
Pure stdlib. Generates synthetic loss-vs-compute data with a known power law + noise,
fits L(C) = Linf + a*C^(-alpha) via a small grid/refine search, and tests extrapolation.
NOTE: synthetic data -- this demonstrates the FITTING METHOD and its fragility, not a real model run.
"""
import math, random

def power_law(C, Linf, a, alpha): return Linf + a * C ** (-alpha)

def make_data(seed=0):
    rng = random.Random(seed)
    true = dict(Linf=1.8, a=12.0, alpha=0.32)
    Cs = [10**k for k in (2, 3, 4, 5, 6)]            # train range: 1e2..1e6
    data = [(C, power_law(C, **true) * (1 + rng.uniform(-0.01, 0.01))) for C in Cs]
    return true, data

def fit(data):
    best, besterr = None, float("inf")
    for Linf in [x/10 for x in range(10, 25)]:
        for a in [x/2 for x in range(2, 40)]:
            for alpha in [x/100 for x in range(10, 60)]:
                err = sum((power_law(C, Linf, a, alpha) - L)**2 for C, L in data)
                if err < besterr: besterr, best = err, (Linf, a, alpha)
    return best

if __name__ == "__main__":
    true, data = make_data()
    print("true params:", true)
    print("train data (C, loss):")
    for C, L in data: print(f"  C={C:>9.0e}  L={L:.4f}")
    Linf, a, alpha = fit(data)
    print(f"\nfitted: Linf={Linf:.2f} a={a:.2f} alpha={alpha:.2f}")
    C_new = 1e7                                        # extrapolate 10x beyond training
    pred = power_law(C_new, Linf, a, alpha)
    actual = power_law(C_new, **true)
    print(f"extrapolate to C={C_new:.0e}: predicted L={pred:.4f}, true L={actual:.4f}, "
          f"err={abs(pred-actual):.4f}")
    print("Lesson: power-law fits extrapolate well IN-REGIME; small exponent errors compound far out.")
