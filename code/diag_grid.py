"""
diag_grid.py
============
The design question, laid out over the whole weight grid.

Two facts established by diag_bootstrap.py and diag_selection.py have to be
settled before anything goes into the paper.

1. At Ts = 20 the empirical quantile of a season's own residuals is badly
   biased INWARDS (measured: +0.28 and -0.35 at the 5% and 95% levels in
   the scale DGP), of the same size as the shape discrepancy the shrinkage
   trades against.  So the choice of the lam = 1 endpoint matters: the
   finite-sample conformal levels partly undo that bias.

2. Minimising the interval score and hitting nominal coverage are not the
   same objective; in the scale DGP at Ts = 20 the score-optimal weight
   delivers 0.873 coverage.  The paper has to say which one it optimises.

For each DGP and each Ts this reports the whole profile of interval score,
coverage and width over the weight grid, for the plain family (endpoints
N and S) and for the corrected one (N+ and S+), and then the weights that
three feasible rules select.
"""

import os
import numpy as np
import pandas as pd

import fastcal as fc

S = 12
ALPHA = 0.10
LAM_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)
BETA = np.linspace(0.0, ALPHA, 41)
N_EVAL = 40000
N_DRAW = 200
HIGH = (1, 2, 8, 9)


def draw(kind, s, n, rng):
    if kind == "scale":
        sig = 2.0 if s in HIGH else 1.0
        return sig * rng.standard_t(3, size=n) / np.sqrt(3.0)
    if kind == "shape":
        if s in HIGH:
            z = np.exp(rng.standard_normal(n))
            return (z - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
        return rng.standard_normal(n)
    if kind == "mixed":
        sig = 2.0 if s in HIGH else 1.0
        if s in HIGH:
            z = np.exp(rng.standard_normal(n))
            return sig * (z - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
        return sig * rng.standard_normal(n)
    if kind == "null":
        return rng.standard_normal(n)
    raise ValueError(kind)


def buffers(kind, Ts, rng):
    return {s: draw(kind, s, Ts, rng) for s in range(1, S + 1)}


# ------------------------------------------------------- selection rules
def cv_weights(buf, s_star, alpha, corrected, rng, n_folds=10):
    """
    Out-of-fold score and coverage for every weight on the grid.
    Folds are removed from the season buffer and therefore also from the
    pooled standardised buffer, as they would be at prediction time.
    """
    own = np.asarray(buf[s_star], dtype=float)
    n = len(own)
    n_folds = min(n_folds, n)
    folds = np.array_split(rng.permutation(n), n_folds)
    tot = np.zeros(len(LAM_GRID))
    hit = np.zeros(len(LAM_GRID))
    wid = np.zeros(len(LAM_GRID))
    for f in folds:
        keep = np.setdiff1d(np.arange(n), f)
        if len(keep) < 3:
            continue
        sub = dict(buf)
        sub[s_star] = own[keep]
        ep = fc.Endpoints(sub, s_star, alpha, BETA, corrected, n_levels_S=n)
        for j, lam in enumerate(LAM_GRID):
            lo, hi = ep.interval(lam)
            tot[j] += fc.winkler(lo, hi, own[f], alpha).sum()
            hit[j] += np.sum((own[f] >= lo) & (own[f] <= hi))
            wid[j] += (hi - lo) * len(f)
    return tot / n, hit / n, wid / n


def rule_from_cv(score, cov, wid, alpha):
    lam_score = float(LAM_GRID[int(np.argmin(score))])
    gap = np.abs(cov - (1 - alpha))
    best = np.flatnonzero(gap <= gap.min() + 1e-12)
    lam_cov = float(LAM_GRID[best[int(np.argmax(wid[best]))]])
    return lam_score, lam_cov


# ------------------------------------------------------------------ main
def run(kind, Ts, s_star=1, seed=5):
    rng = np.random.default_rng(seed)
    ev = fc.Evaluator(draw(kind, s_star, N_EVAL, rng), ALPHA)

    prof = {c: {k: np.zeros(len(LAM_GRID)) for k in ("score", "cov", "wid")}
            for c in (False, True)}
    sel = {k: [] for k in ("cv_score", "cv_cov")}
    sel_perf = {k: {"score": [], "cov": [], "wid": []}
                for k in ("cv_score", "cv_cov")}

    for _ in range(N_DRAW):
        buf = buffers(kind, Ts, rng)
        eps = {}
        for corrected in (False, True):
            ep = fc.Endpoints(buf, s_star, ALPHA, BETA, corrected)
            eps[corrected] = ep
            for j, lam in enumerate(LAM_GRID):
                lo, hi = ep.interval(lam)
                prof[corrected]["score"][j] += ev.score(lo, hi) / N_DRAW
                prof[corrected]["cov"][j] += ev.coverage(lo, hi) / N_DRAW
                prof[corrected]["wid"][j] += (hi - lo) / N_DRAW

        sc, cv, wd = cv_weights(buf, s_star, ALPHA, True, rng)
        lam_s, lam_c = rule_from_cv(sc, cv, wd, ALPHA)
        for name, lam in (("cv_score", lam_s), ("cv_cov", lam_c)):
            sel[name].append(lam)
            lo, hi = eps[True].interval(lam)
            sel_perf[name]["score"].append(ev.score(lo, hi))
            sel_perf[name]["cov"].append(ev.coverage(lo, hi))
            sel_perf[name]["wid"].append(hi - lo)

    rows = []
    for corrected in (False, True):
        for j, lam in enumerate(LAM_GRID):
            rows.append({"dgp": kind, "Ts": Ts, "family":
                         "corrected" if corrected else "plain",
                         "lam": lam, "rule": "",
                         "score": prof[corrected]["score"][j],
                         "coverage": prof[corrected]["cov"][j],
                         "width": prof[corrected]["wid"][j]})
    for k in sel:
        rows.append({"dgp": kind, "Ts": Ts, "family": "corrected",
                     "lam": float(np.mean(sel[k])), "rule": k,
                     "score": float(np.mean(sel_perf[k]["score"])),
                     "coverage": float(np.mean(sel_perf[k]["cov"])),
                     "width": float(np.mean(sel_perf[k]["wid"]))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    for kind in ("scale", "shape", "mixed", "null"):
        for Ts in (20, 35, 50):
            print(f"--- {kind}, Ts={Ts} ---", flush=True)
            out.append(run(kind, Ts))
    df = pd.concat(out, ignore_index=True)
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "tables", "X6_lambda_profile.csv")
    df.to_csv(path, index=False)
    print("saved", path)

    for kind in ("scale", "shape", "mixed", "null"):
        for Ts in (20, 35, 50):
            sub = df[(df.dgp == kind) & (df.Ts == Ts)]
            print(f"\n=== {kind}, Ts={Ts} ===")
            for fam in ("plain", "corrected"):
                s = sub[(sub.family == fam) & (sub.rule == "")].sort_values("lam")
                j = int(np.argmin(s["score"].values))
                print(f"  {fam:<10} best-score lam={s['lam'].values[j]:.1f}"
                      f" score={s['score'].values[j]:.3f}"
                      f" cov={s['coverage'].values[j]:.3f}"
                      f" | lam=0 cov={s['coverage'].values[0]:.3f}"
                      f" score={s['score'].values[0]:.3f}"
                      f" | lam=1 cov={s['coverage'].values[-1]:.3f}"
                      f" score={s['score'].values[-1]:.3f}")
            for _, r in sub[sub.rule != ""].iterrows():
                print(f"  rule {r['rule']:<9} mean lam={r['lam']:.2f}"
                      f" score={r['score']:.3f} cov={r['coverage']:.3f}"
                      f" width={r['width']:.3f}")
