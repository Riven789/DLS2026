"""best_of_n.py -- simulate best-of-n coverage vs. single-sample accuracy (Module Q05).
Pure stdlib. Demonstrates the diminishing-returns trade-off of test-time compute.
NOTE: this is a MATH SIMULATION of the coverage identity 1-(1-p)^n, NOT a real LLM run.
"""
import math, random

def coverage_exact(p1, n):
    """Analytic coverage: probability >=1 of n independent samples is correct."""
    return 1 - (1 - p1) ** n

def coverage_empirical(p1, n, trials=20000, seed=0):
    """Monte-Carlo check of the analytic formula."""
    rng = random.Random(seed)
    hits = sum(any(rng.random() < p1 for _ in range(n)) for _ in range(trials))
    return hits / trials

if __name__ == "__main__":
    print("Coverage = Pr[at least one of n samples correct] = 1-(1-p1)^n\n")
    for p1 in (0.3, 0.5, 0.7):
        print(f"single-sample accuracy p1={p1}")
        print(f"{'n':>3} {'exact':>8} {'monte-carlo':>12} {'marginal gain':>14}")
        prev = 0.0
        for n in (1, 2, 4, 8, 16, 32):
            ex = coverage_exact(p1, n)
            mc = coverage_empirical(p1, n)
            print(f"{n:>3} {ex:>8.3f} {mc:>12.3f} {ex-prev:>14.3f}")
            prev = ex
        print()
    # smallest n for >=0.95 coverage at p1=0.5 (the quiz question)
    p1 = 0.5
    n = math.ceil(math.log(0.05) / math.log(1 - p1))
    print(f"At p1={p1}, smallest n for >=0.95 coverage: n={n}  (coverage={coverage_exact(p1,n):.3f})")
    print("Marginal gain SHRINKS as n grows -> diminishing returns -> an optimal budget exists.")
