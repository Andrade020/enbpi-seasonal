"""
diag_selection.py
=================
Which rule should pick the weight?

Three candidates are compared against the population-optimal weight,
computed by scoring each candidate weight on a large independent sample
from the true season distribution:

    plugin   moment rule built from a within-season bootstrap
    cv       K-fold cross-validation on the Winkler interval score
    fixed    the best constant weight, for reference

The comparison is run on residual buffers drawn directly from known
distributions, so that the behaviour of the selection rule is isolated
from the forecasting pipeline.  The criterion is the interval score of
the resulting interval, which is what the calibration step is ultimately
for; coverage and width are reported alongside it.
"""

import numpy as np
import shrinkage as sh

S = 12
ALPHA = 0.10
LAM_GRID = np.linspace(0.0, 1.0, 21)
N_EVAL = 20000
N_DRAW = 200


def draw(kind, s, n, rng):
    if kind == "scale":
        sig = 2.0 if s in (1, 2, 8, 9) else 1.0
        return sig * rng.standard_t(3, size=n) / np.sqrt(3.0)
    if kind == "shape":
        if s in (1, 2, 8, 9):
            z = np.exp(rng.standard_normal(n))
            return (z - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
        return rng.standard_normal(n)
    raise ValueError(kind)


def buffers(kind, Ts, rng):
    return {s: draw(kind, s, Ts, rng) for s in range(1, S + 1)}


def interval_for_lambda(buf, s_star, lam, alpha=ALPHA):
    return sh.hybrid_interval(buf, s_star, alpha, lam, lam, coherent=True)


def evaluate(kind, Ts, s_star=1, n_draw=N_DRAW, seed=3):
    rng = np.random.default_rng(seed)
    y_eval = draw(kind, s_star, N_EVAL, rng)

    score = {k: [] for k in ("plugin", "cv", "best_fixed")}
    cover = {k: [] for k in ("plugin", "cv", "best_fixed")}
    width = {k: [] for k in ("plugin", "cv", "best_fixed")}
    lam_used = {k: [] for k in ("plugin", "cv")}
    grid_score = np.zeros(len(LAM_GRID))

    for _ in range(n_draw):
        buf = buffers(kind, Ts, rng)

        # population score of every fixed weight, on this buffer
        gs = np.empty(len(LAM_GRID))
        ints = []
        for j, lam in enumerate(LAM_GRID):
            lo, hi = interval_for_lambda(buf, s_star, lam)
            ints.append((lo, hi))
            gs[j] = sh.winkler_score(lo, hi, y_eval, ALPHA).mean()
        grid_score += gs / n_draw

        lam_p = float(np.mean(sh.lambda_plugin(
            buf, s_star, np.array([ALPHA / 2, 1 - ALPHA / 2]),
            n_boot=150, rng=rng)))
        lam_c = sh.lambda_cv(buf, s_star, ALPHA, LAM_GRID,
                             n_folds=min(10, len(buf[s_star])), rng=rng)
        lam_used["plugin"].append(lam_p)
        lam_used["cv"].append(lam_c)

        for name, lam in (("plugin", lam_p), ("cv", lam_c),
                          ("best_fixed", None)):
            if lam is None:
                continue
            lo, hi = interval_for_lambda(buf, s_star, lam)
            score[name].append(sh.winkler_score(lo, hi, y_eval, ALPHA).mean())
            cover[name].append(float(np.mean((y_eval >= lo) & (y_eval <= hi))))
            width[name].append(hi - lo)

    j_best = int(np.argmin(grid_score))
    lam_best = LAM_GRID[j_best]
    # rerun the best fixed weight
    rng = np.random.default_rng(seed)
    _ = draw(kind, s_star, N_EVAL, rng)
    for _ in range(n_draw):
        buf = buffers(kind, Ts, rng)
        lo, hi = interval_for_lambda(buf, s_star, lam_best)
        score["best_fixed"].append(sh.winkler_score(lo, hi, y_eval, ALPHA).mean())
        cover["best_fixed"].append(float(np.mean((y_eval >= lo) & (y_eval <= hi))))
        width["best_fixed"].append(hi - lo)

    print(f"\n=== {kind} DGP, Ts = {Ts}, season {s_star} ===")
    print(f"  population-optimal fixed weight: {lam_best:.2f}"
          f"   (score {grid_score[j_best]:.3f})")
    print(f"  score at lam=0 (N): {grid_score[0]:.3f}"
          f"   at lam=1 (S): {grid_score[-1]:.3f}")
    print(f"  {'rule':<11}{'mean lam':>9}{'score':>9}{'coverage':>10}{'width':>8}")
    for k in ("plugin", "cv", "best_fixed"):
        lm = np.mean(lam_used[k]) if k in lam_used else lam_best
        print(f"  {k:<11}{lm:>9.3f}{np.mean(score[k]):>9.3f}"
              f"{np.mean(cover[k]):>10.3f}{np.mean(width[k]):>8.3f}")


if __name__ == "__main__":
    for kind in ("scale", "shape"):
        for Ts in (20, 35, 50):
            evaluate(kind, Ts)
