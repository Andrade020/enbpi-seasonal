"""
diag_linesearch.py
==================
How much coverage does the minimum-width line search cost, and when does
it pay for itself?

The line search picks the narrowest of m candidate quantile pairs computed
from the same n residuals.  Selecting the narrowest of many noisy
candidates biases coverage downwards, and the bias grows with m and shrinks
with n.  Against that, when the residual distribution is skewed the
asymmetric pair genuinely is narrower than the symmetric one, so there is
something real to find.

Both effects are measured here on residual buffers drawn from a known
distribution: symmetric (t3, standardised), where the optimal pair IS the
symmetric one and the search can only hurt, and skewed (standardised
lognormal), where it has something to gain.  Conformal levels throughout,
so the only difference between the columns is the selection.
"""
import numpy as np
import pandas as pd
import fastcal as fc

ALPHA = 0.10
N_EVAL = 200000
N_DRAW = 3000


def sample(kind, n, rng):
    if kind == "symmetric":
        return rng.standard_t(3, size=n) / np.sqrt(3.0)
    z = np.exp(rng.standard_normal(n))
    return (z - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)


def interval(buf, alpha, m):
    """Conformal order-statistic endpoints; m = 1 fixes beta at alpha/2."""
    a = np.sort(buf)
    n = len(a)
    beta = np.array([alpha / 2]) if m == 1 else np.linspace(0, alpha, m)
    k_lo, k_hi, _ = fc.conformal_indices(beta, 1 - alpha + beta, n)
    lo, hi = fc.order_stat(a, k_lo), fc.order_stat(a, k_hi)
    j = int(np.argmin(hi - lo))
    return float(lo[j]), float(hi[j])


if __name__ == "__main__":
    rows = []
    for kind in ("symmetric", "skewed"):
        rng = np.random.default_rng(17)
        ev = fc.Evaluator(sample(kind, N_EVAL, rng), ALPHA)
        for n in (10, 20, 35, 50, 100, 200, 500):
            for m in (1, 5, 21, 41, 201):
                cov = wid = sco = 0.0
                for _ in range(N_DRAW):
                    lo, hi = interval(sample(kind, n, rng), ALPHA, m)
                    cov += ev.coverage(lo, hi) / N_DRAW
                    wid += (hi - lo) / N_DRAW
                    sco += ev.score(lo, hi) / N_DRAW
                rows.append({"dist": kind, "n": n, "m": m,
                             "coverage": cov, "width": wid, "score": sco})
    df = pd.DataFrame(rows)
    df.to_csv("../tables/X9_linesearch.csv", index=False)
    for kind in ("symmetric", "skewed"):
        print(f"\n=== {kind} residuals ===")
        print(f"{'n':>5} " + "".join(f"{'m='+str(m):>16}" for m in (1, 21, 201)))
        print(f"{'':>5} " + "".join(f"{'cov / score':>16}" for _ in range(3)))
        for n in (10, 20, 35, 50, 100, 200, 500):
            cells = []
            for m in (1, 21, 201):
                r = df[(df.dist == kind) & (df.n == n) & (df.m == m)].iloc[0]
                cells.append(f"{r['coverage']:.3f} /{r['score']:6.3f}")
            print(f"{n:>5} " + "".join(f"{c:>16}" for c in cells))
