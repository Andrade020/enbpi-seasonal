"""
simulation_family.py
====================
The decisive experiment, run in the full forecasting pipeline.

Three diagnostics on synthetic buffers (diag_bootstrap, diag_grid,
diag_beta) point at two finite-sample defects of stratified conformal
calibration that have nothing to do with the seasonal bias the paper is
about:

  (a) the empirical quantile of a short buffer is biased inwards, which
      the finite-sample conformal levels partly repair;
  (b) the minimum-width line search picks the narrowest of many candidate
      quantile pairs computed from the same residuals, and that selection
      biases coverage downwards; the damage grows as the buffer shortens,
      so it falls almost entirely on the stratified scheme.

This script checks whether the two repairs survive in the real pipeline,
where the buffers hold leave-one-out residuals from a bootstrap ensemble
and slide through the test period.  Methods (all sharing one ensemble per
replication, so they differ only in calibration):

  pool      pooled buffer, line search, plain levels      (Xu and Xie)
  norm      pooled standardised buffer, line search, plain
  strat     season buffer, line search, plain             (EnbPI-S)
  pool_f    pooled buffer, beta = alpha/2, conformal levels
  norm_f    normalised, beta = alpha/2, conformal levels
  strat_f   season buffer, beta = alpha/2, conformal levels
  hyb_cv    family of strat_f and norm_f, weight chosen per season by
            cross-validated coverage on the training buffer
  hyb_50    the same family at a fixed weight of one half

Run:
    python simulation_family.py            # n_rep = 200
    python simulation_family.py --quick    # n_rep = 25
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import deque

import simulation as sim
import simulation_baselines as sb
import fastcal as fc

S = sim.S
ALPHA = sim.ALPHA
B_BOOT = sim.B_BOOT
P_LAGS = sim.P_LAGS
T1 = sim.T1
HIGH = sim.HIGH_SEAS

BETA_SEARCH = np.linspace(0.0, ALPHA, 41)
BETA_FIXED = np.array([ALPHA / 2])
LAM_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)

METHODS = ["pool", "norm", "strat", "pool_f", "norm_f", "strat_f",
           "hyb_cv", "hyb_50"]


# ------------------------------------------------------------------ DGPs
def gen(kind, n, rng):
    """Series generator; the AR structure is common to all four DGPs."""
    if kind == "scale":
        return sim.generate_series(n, sim.SIGMA_HIGH, sim.SIGMA_LOW, rng)
    if kind == "homosk":
        return sim.generate_series(n, 1.0, 1.0, rng)
    if kind == "shape":
        return sb.generate_series_shape(n, rng, sigma_high=1.0, sigma_low=1.0)
    if kind == "mixed":
        return sb.generate_series_shape(n, rng, sigma_high=2.0, sigma_low=1.0)
    raise ValueError(kind)


# --------------------------------------------------- calibration schemes
def pooled_interval(buf, alpha, beta_grid, corrected):
    a = np.sort(np.asarray(buf, dtype=float))
    n = len(a)
    lo_lv, hi_lv = beta_grid, 1.0 - alpha + beta_grid
    if corrected:
        k_lo, k_hi, _ = fc.conformal_indices(lo_lv, hi_lv, n)
        lo, hi = fc.order_stat(a, k_lo), fc.order_stat(a, k_hi)
    else:
        lo = fc.q_sorted(a, lo_lv)
        hi = fc.q_sorted(a, hi_lv)
    b = int(np.argmin(hi - lo))
    return float(lo[b]), float(hi[b])


def cv_lambda_coverage(res_by_season, s_star, alpha, rng, n_folds=10):
    """Weight whose out-of-fold coverage is closest to nominal; ties go to
    the wider interval."""
    own = np.asarray(res_by_season[s_star], dtype=float)
    n = len(own)
    if n < 4:
        return 1.0
    n_folds = min(n_folds, n)
    folds = np.array_split(rng.permutation(n), n_folds)
    hit = np.zeros(len(LAM_GRID))
    wid = np.zeros(len(LAM_GRID))
    base_scales = fc.season_scales(res_by_season)
    for f in folds:
        keep = np.setdiff1d(np.arange(n), f)
        if len(keep) < 3:
            continue
        sub = dict(res_by_season)
        sub[s_star] = own[keep]
        # only the held-out season's scale changes from fold to fold
        scales = dict(base_scales)
        scales[s_star] = fc.mad_scale(sub[s_star])
        # the conformal indices are those of the buffer that will be used at
        # prediction time, not of the shortened cross-validation buffer
        ep = fc.Endpoints(sub, s_star, alpha, BETA_FIXED, corrected=True,
                          n_levels_S=n, scales=scales)
        for j, lam in enumerate(LAM_GRID):
            lo, hi = ep.interval(lam)
            hit[j] += np.sum((own[f] >= lo) & (own[f] <= hi))
            wid[j] += (hi - lo) * len(f)
    cov = hit / n
    gap = np.abs(cov - (1 - alpha))
    best = np.flatnonzero(gap <= gap.min() + 1e-12)
    return float(LAM_GRID[best[int(np.argmax(wid[best]))]])


# ------------------------------------------------------ one replication
def run_one(Y, T, T1, alpha, B, S, p, rng):
    tr_idx = np.arange(p, T)
    n_tr = len(tr_idx)
    X_tr = sim.build_features(Y, tr_idx, p)
    y_tr = Y[tr_idx]
    models, Sb = sim.fit_ensemble(X_tr, y_tr, B, "ridge", rng)

    all_idx = np.arange(p, T + T1)
    X_all = sim.build_features(Y, all_idx, p)
    y_all = Y[all_idx]
    preds = sim.batch_predict(models, X_all)
    ens = preds.mean(axis=0)
    loo = y_tr - sim.loo_predictions(preds[:, :n_tr], Sb, n_tr)
    full_res = y_all - ens
    test_preds = ens[T - p: T - p + T1]

    def get_res(j0):
        k = j0 - p
        return float(full_res[k]) if 0 <= k < len(full_res) else 0.0

    buf_pool = deque(loo.tolist())
    bufs = {s: deque() for s in range(1, S + 1)}
    for k, i in enumerate(tr_idx):
        bufs[sim.season(i, S)].append(float(loo[k]))

    # weights are chosen once, on the training buffers
    res0 = {s: np.array(bufs[s]) for s in bufs}
    lam_cv = {s: cv_lambda_coverage(res0, s, alpha, rng)
              for s in range(1, S + 1)}

    out = {m: (np.empty(T1, dtype=bool), np.empty(T1), np.empty(T1))
           for m in METHODS}

    for step in range(T1):
        t0 = T + step
        f = float(test_preds[step])
        y_t = float(Y[t0])
        s_star = sim.season(t0, S)
        res = {s: np.array(bufs[s]) if bufs[s] else np.zeros(1) for s in bufs}
        arr_pool = np.array(buf_pool)

        scales = fc.season_scales(res)
        ep_search = fc.Endpoints(res, s_star, alpha, BETA_SEARCH, False,
                                 scales=scales)
        ep_fixed = fc.Endpoints(res, s_star, alpha, BETA_FIXED, True,
                                scales=scales)

        cand = {}
        cand["pool"] = pooled_interval(arr_pool, alpha, BETA_SEARCH, False)
        cand["pool_f"] = pooled_interval(arr_pool, alpha, BETA_FIXED, True)
        cand["strat"] = ep_search.interval(1.0)
        cand["norm"] = ep_search.interval(0.0)
        cand["strat_f"] = ep_fixed.interval(1.0)
        cand["norm_f"] = ep_fixed.interval(0.0)
        cand["hyb_cv"] = ep_fixed.interval(lam_cv[s_star])
        cand["hyb_50"] = ep_fixed.interval(0.5)

        for m, (q_lo, q_hi) in cand.items():
            lo, hi = f + q_lo, f + q_hi
            cov, lo_a, hi_a = out[m]
            lo_a[step], hi_a[step] = lo, hi
            cov[step] = bool(lo <= y_t <= hi)

        j0 = T + step - 1
        if j0 >= p:
            eps = get_res(j0)
            s_j = sim.season(j0, S)
            buf_pool.popleft(); buf_pool.append(eps)
            if bufs[s_j]:
                bufs[s_j].popleft()
            bufs[s_j].append(eps)

    return out, lam_cv


# ------------------------------------------------------------ Monte Carlo
def monte_carlo(kind, Ts, n_rep, seed_base):
    T = Ts * S
    acc = {m: [] for m in METHODS}
    lam_acc = []
    for rep in range(n_rep):
        rng = np.random.default_rng(seed_base + rep * 1000)
        Y = gen(kind, T + T1, rng)
        out, lam = run_one(Y, T, T1, ALPHA, B_BOOT, S, P_LAGS, rng)
        lam_acc.append(np.mean(list(lam.values())))
        for m in METHODS:
            cov, lo, hi = out[m]
            met = sim.compute_metrics(cov, lo, hi, T, S)
            y = np.array([Y[T + k] for k in range(T1)])
            met["score"] = float(fc.winkler(lo, hi, y, ALPHA).mean())
            acc[m].append(met)
        if (rep + 1) % 25 == 0:
            print(f"    rep {rep+1}/{n_rep}", flush=True)
    return acc, float(np.mean(lam_acc))


def summarise(acc, n_rep, kind, Ts, lam_mean):
    rows = []
    for m in METHODS:
        summ = sim.summarise(acc[m])
        per = [summ["seasonal_coverage"][s][0] for s in range(1, S + 1)]
        rows.append({
            "dgp": kind, "Ts": Ts, "method": m,
            "coverage": summ["overall_coverage_mean"],
            "coverage_se": summ["overall_coverage_std"] / np.sqrt(n_rep),
            "spread": float(np.max(per) - np.min(per)),
            "worst": float(np.min(per)),
            "width": summ["overall_width_mean"],
            "score": float(np.mean([r["score"] for r in acc[m]])),
            "lam_mean": lam_mean if m == "hyb_cv" else np.nan,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_rep = 25 if quick else 200
    here = os.path.dirname(os.path.abspath(__file__))
    tab = os.path.join(os.path.dirname(here), "tables")
    path = os.path.join(tab, "X8_family.csv")

    # results are written after every configuration, and configurations
    # already present in the file are skipped, so an interrupted run can be
    # resumed by launching the script again
    done = set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        done = set(zip(prev["dgp"], prev["Ts"]))
        print(f"resuming: {len(done)} configuration(s) already stored")

    # optional wall-clock budget: stop starting new configurations once it is
    # used up, so that a run can be split over several invocations
    budget = None
    if "--budget" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget") + 1])
    started = time.time()

    for kind, seed in (("scale", 1000), ("shape", 6000),
                       ("mixed", 7000), ("homosk", 4000)):
        for Ts in (20, 35, 50):
            if (kind, Ts) in done:
                print(f"--- skip {kind}, Ts={Ts} (already done) ---", flush=True)
                continue
            if budget is not None and time.time() - started > budget:
                print("--- budget exhausted, stopping ---", flush=True)
                break
            print(f"=== {kind}, Ts={Ts} ===", flush=True)
            acc, lam = monte_carlo(kind, Ts, n_rep, seed)
            frame = summarise(acc, n_rep, kind, Ts, lam)
            frame.to_csv(path, mode="a", index=False,
                         header=not os.path.exists(path))
            print(f"    written to {os.path.basename(path)}", flush=True)

    df = pd.read_csv(path)
    df = df[df["dgp"] != "dgp"]          # drop any repeated header rows
    for c in ("Ts", "coverage", "coverage_se", "spread", "worst",
              "width", "score"):
        df[c] = pd.to_numeric(df[c])
    df.to_csv(path, index=False)

    for kind in ("scale", "shape", "mixed", "null"):
        for Ts in (20, 35, 50):
            sub = df[(df.dgp == kind) & (df.Ts == Ts)]
            print(f"\n=== {kind}, Ts={Ts} ===")
            print(f"{'method':<9}{'cover':>8}{'spread':>8}{'worst':>8}"
                  f"{'width':>8}{'score':>8}")
            for _, r in sub.iterrows():
                print(f"{r['method']:<9}{r['coverage']:>8.3f}{r['spread']:>8.3f}"
                      f"{r['worst']:>8.3f}{r['width']:>8.2f}{r['score']:>8.2f}")
