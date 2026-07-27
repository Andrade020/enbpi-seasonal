"""
empirical_family.py
===================
The calibration family on the two Brazilian series.

Same eight schemes as simulation_family.py, same ensemble within each
application, so the columns differ only in how the residual buffer is
turned into an interval.  The weight of the hybrid is chosen per season by
cross-validated coverage on the training buffer and is reported, since it
is itself informative: a weight near one says the season's residuals carry
information that the pooled standardised buffer does not.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import simulation as sim
import fastcal as fc
import simulation_family as sf
from empirical_baselines import load_food, load_exports, cal_season

S = sim.S
ALPHA = sim.ALPHA
B_BOOT = sim.B_BOOT
P_LAGS = sim.P_LAGS
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
METHODS = sf.METHODS


def run_application(Y, T, T1, month0, seed=42):
    rng = np.random.default_rng(seed)
    p = P_LAGS
    all_idx = np.arange(p, T + T1)
    X_all = sim.build_features(Y, all_idx, p)
    y_all = Y[all_idx]
    n_tr = T - p
    models, Sb = sim.fit_ensemble(X_all[:n_tr], y_all[:n_tr], B_BOOT,
                                  "ridge", rng)
    preds = sim.batch_predict(models, X_all)
    ens = preds.mean(axis=0)
    loo = y_all[:n_tr] - sim.loo_predictions(preds[:, :n_tr], Sb, n_tr)
    full_res = y_all - ens
    test_preds = ens[T - p: T - p + T1]

    def get_res(j0):
        k = j0 - p
        return float(full_res[k]) if 0 <= k < len(full_res) else 0.0

    buf_pool = deque(loo.tolist())
    bufs = {s: deque() for s in range(1, S + 1)}
    for k, idx in enumerate(all_idx[:n_tr]):
        bufs[cal_season(idx, month0)].append(float(loo[k]))

    res0 = {s: np.array(bufs[s]) for s in bufs}
    lam_cv = {s: sf.cv_lambda_coverage(res0, s, ALPHA, rng)
              for s in range(1, S + 1)}

    out = {m: (np.empty(T1, dtype=bool), np.empty(T1), np.empty(T1))
           for m in METHODS}

    for step in range(T1):
        t0 = T + step
        f = float(test_preds[step])
        y_t = float(Y[t0])
        ss = cal_season(t0, month0)
        res = {s: np.array(bufs[s]) if bufs[s] else np.zeros(1) for s in bufs}
        arr_pool = np.array(buf_pool)

        ep_search = fc.Endpoints(res, ss, ALPHA, sf.BETA_SEARCH, False)
        ep_fixed = fc.Endpoints(res, ss, ALPHA, sf.BETA_FIXED, True)
        cand = {
            "pool": sf.pooled_interval(arr_pool, ALPHA, sf.BETA_SEARCH, False),
            "pool_f": sf.pooled_interval(arr_pool, ALPHA, sf.BETA_FIXED, True),
            "strat": ep_search.interval(1.0),
            "norm": ep_search.interval(0.0),
            "strat_f": ep_fixed.interval(1.0),
            "norm_f": ep_fixed.interval(0.0),
            "hyb_cv": ep_fixed.interval(lam_cv[ss]),
            "hyb_50": ep_fixed.interval(0.5),
        }
        for m, (q_lo, q_hi) in cand.items():
            lo, hi = f + q_lo, f + q_hi
            cov, lo_a, hi_a = out[m]
            lo_a[step], hi_a[step] = lo, hi
            cov[step] = bool(lo <= y_t <= hi)

        j0 = T + step - 1
        if j0 >= p:
            eps = get_res(j0)
            sj = cal_season(j0, month0)
            buf_pool.popleft(); buf_pool.append(eps)
            if bufs[sj]:
                bufs[sj].popleft()
            bufs[sj].append(eps)

    return out, lam_cv


def summarise(out, Y, T, T1, month0, app, per_rows=None):
    rows = []
    per_rows = [] if per_rows is None else per_rows
    y = np.array([Y[T + k] for k in range(T1)])
    for m in METHODS:
        cov, lo, hi = out[m]
        wid = hi - lo
        per_c, per_w = [], []
        for s in range(1, S + 1):
            mask = np.array([cal_season(T + k, month0) == s
                             for k in range(T1)])
            per_c.append(float(cov[mask].mean()))
            per_w.append(float(wid[mask].mean()))
        for s_ in range(1, S + 1):
            per_rows.append({"application": app, "method": m, "season": s_,
                             "month": MONTHS[s_ - 1],
                             "coverage": per_c[s_ - 1],
                             "width": per_w[s_ - 1]})
        rows.append({
            "application": app, "method": m,
            "coverage": float(cov.mean()),
            "worst": float(np.min(per_c)),
            "spread": float(np.max(per_c) - np.min(per_c)),
            "width": float(wid.mean()),
            "width_ratio": float(np.max(per_w) / np.min(per_w)),
            "score": float(fc.winkler(lo, hi, y, ALPHA).mean()),
            "argmax_month": MONTHS[int(np.argmax(per_w))],
            "argmin_month": MONTHS[int(np.argmin(per_w))],
        })
    return pd.DataFrame(rows), pd.DataFrame(per_rows)


if __name__ == "__main__":
    tab = os.path.join(os.path.dirname(_HERE), "tables")
    frames, lam_rows, per_frames = [], [], []
    for app, loader in (("food", load_food), ("exports", load_exports)):
        Y, T, T1, m0, dates = loader()
        print(f"=== {app}: T={T} T1={T1} month0={m0} ===")
        out, lam = run_application(Y, T, T1, m0)
        df, per = summarise(out, Y, T, T1, m0, app)
        per_frames.append(per)
        print(df[["method", "coverage", "worst", "spread", "width",
                  "width_ratio", "score"]].to_string(index=False))
        print("  selected weights: " + ", ".join(
            f"{MONTHS[s-1]}={lam[s]:.1f}" for s in range(1, S + 1)))
        frames.append(df)
        for s in range(1, S + 1):
            lam_rows.append({"application": app, "season": s,
                             "month": MONTHS[s - 1], "lambda": lam[s]})
    pd.concat(frames, ignore_index=True).to_csv(
        os.path.join(tab, "X10_empirical_family.csv"), index=False)
    pd.concat(per_frames, ignore_index=True).to_csv(
        os.path.join(tab, "X11_empirical_by_season.csv"), index=False)
    pd.DataFrame(lam_rows).to_csv(
        os.path.join(tab, "X10_empirical_lambda.csv"), index=False)
    print("\nsaved tables/X10_empirical_family.csv and X10_empirical_lambda.csv")
