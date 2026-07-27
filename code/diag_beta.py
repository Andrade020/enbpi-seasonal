"""
diag_beta.py
============
Is the minimum-width line search worth its selection bias?

EnbPI chooses the asymmetric quantile pair by minimising the width over a
grid of beta.  Choosing the narrowest of 41 candidates from the same
residuals that define them is a selection, and selections bias coverage
downwards; the shorter the buffer, the worse.  The alternative is to fix
beta = alpha/2 and give up the adaptation to skewness.

This compares the two, over the weight grid, on the corrected family.
"""
import numpy as np
import pandas as pd
import fastcal as fc
import diag_grid as dg

ALPHA, N_EVAL, N_DRAW = 0.10, 40000, 200
LAM = np.round(np.linspace(0, 1, 11), 2)
BETA_FULL = np.linspace(0.0, ALPHA, 41)
BETA_SYM = np.array([ALPHA / 2])


def run(kind, Ts, s_star=1, seed=5):
    rng = np.random.default_rng(seed)
    ev = fc.Evaluator(dg.draw(kind, s_star, N_EVAL, rng), ALPHA)
    acc = {}
    for tag, beta in (("search", BETA_FULL), ("fixed", BETA_SYM)):
        acc[tag] = {k: np.zeros(len(LAM)) for k in ("score", "cov", "wid")}
    for _ in range(N_DRAW):
        buf = dg.buffers(kind, Ts, rng)
        for tag, beta in (("search", BETA_FULL), ("fixed", BETA_SYM)):
            ep = fc.Endpoints(buf, s_star, ALPHA, beta, corrected=True)
            for j, lam in enumerate(LAM):
                lo, hi = ep.interval(lam)
                acc[tag]["score"][j] += ev.score(lo, hi) / N_DRAW
                acc[tag]["cov"][j] += ev.coverage(lo, hi) / N_DRAW
                acc[tag]["wid"][j] += (hi - lo) / N_DRAW
    rows = []
    for tag in acc:
        for j, lam in enumerate(LAM):
            rows.append({"dgp": kind, "Ts": Ts, "beta": tag, "lam": lam,
                         "score": acc[tag]["score"][j],
                         "coverage": acc[tag]["cov"][j],
                         "width": acc[tag]["wid"][j]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    for kind in ("scale", "shape", "null"):
        for Ts in (20, 50):
            out.append(run(kind, Ts))
    df = pd.concat(out, ignore_index=True)
    df.to_csv("../tables/X7_beta.csv", index=False)
    for kind in ("scale", "shape", "null"):
        for Ts in (20, 50):
            print(f"\n=== {kind}, Ts={Ts} ===")
            for tag in ("search", "fixed"):
                s = df[(df.dgp == kind) & (df.Ts == Ts) & (df.beta == tag)]
                s = s.sort_values("lam")
                j = int(np.argmin(s["score"].values))
                print(f"  beta {tag:<7} best lam={s['lam'].values[j]:.1f}"
                      f" score={s['score'].values[j]:.3f}"
                      f" | lam=0 cov={s['coverage'].values[0]:.3f}"
                      f" | lam=1 cov={s['coverage'].values[-1]:.3f}"
                      f" wid={s['width'].values[-1]:.3f}")
