"""
empirical_baselines.py
======================
Runs the competing calibration schemes of simulation_baselines.py on
the two empirical applications of the paper:

  food   IPCA -- Alimentacao no domicilio (BCB SGS 1635), Jan 1995
         onwards, monthly percentage change, index 0 = January.
  exports Brazilian total exports (BCB SGS 1402), log-returns from
         March 1979, index 0 = MARCH (month0 = 3).

Every method shares the same bootstrap ensemble and the same LOO
residuals within an application, so the only difference is the
calibration step.  Offline by default: reads the cached CSVs in data/.

Usage
-----
    python code/empirical_baselines.py
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
from simulation_baselines import (interval_from_buffer, mad_scale, METHODS)

PROJ = os.path.dirname(_HERE)
DATA = os.path.join(PROJ, "data")
TAB = os.path.join(PROJ, "tables")

S = sim.S
ALPHA = sim.ALPHA
B_BOOT = sim.B_BOOT
P_LAGS = sim.P_LAGS
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def cal_season(t0, month0, S_=S):
    """True calendar season of array index t0 when index 0 is month0."""
    return ((t0 + month0 - 1) % S_) + 1


# ---------------------------------------------------------------- data
def load_food():
    df = pd.read_csv(os.path.join(DATA, "ipca_food_raw.csv"),
                     parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp()
    sub = df.loc[(df.index >= pd.Timestamp("1995-01")) &
                 (df.index <= pd.Timestamp("2024-12")), "value"]
    Y = sub.values.astype(float)
    dates = sub.index
    T = int((dates <= pd.Timestamp("2014-12")).sum())
    T1 = int((dates >= pd.Timestamp("2015-01")).sum())
    return Y, T, T1, int(dates[0].month), dates


def load_exports():
    df = pd.read_csv(os.path.join(DATA, "exports_raw.csv"),
                     parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp()
    lr = (100.0 * np.log(df["value"] / df["value"].shift(1))).dropna()
    lr = lr.loc[lr.index <= pd.Timestamp("2024-12")]
    Y = lr.values.astype(float)
    dates = lr.index
    T = int((dates <= pd.Timestamp("2014-12")).sum())
    T1 = int(((dates >= pd.Timestamp("2015-01")) &
              (dates <= pd.Timestamp("2024-12"))).sum())
    return Y, T, T1, int(dates[0].month), dates


# ------------------------------------------------------------- methods
def run_all_methods(Y, T, T1, month0, alpha=ALPHA, B=B_BOOT,
                    p=P_LAGS, s0=1, seed=42):
    rng = np.random.default_rng(seed)
    all_idx = np.arange(p, T + T1)
    X_all = sim.build_features(Y, all_idx, p)
    y_all = Y[all_idx]
    n_tr = T - p
    X_tr, y_tr = X_all[:n_tr], y_all[:n_tr]

    models, Sb = sim.fit_ensemble(X_tr, y_tr, B, "ridge", rng)
    preds_B = sim.batch_predict(models, X_all)
    ens = preds_B.mean(axis=0)
    loo_res = y_tr - sim.loo_predictions(preds_B[:, :n_tr], Sb, n_tr)
    full_res = y_all - ens

    def get_res(j0):
        k = j0 - p
        return float(full_res[k]) if 0 <= k < len(full_res) else 0.0

    test_preds = ens[T - p: T - p + T1]

    buf_p = deque(loo_res.tolist())
    bufs_s = {s: deque() for s in range(1, S + 1)}
    for k, idx in enumerate(all_idx[:n_tr]):
        bufs_s[cal_season(idx, month0)].append(float(loo_res[k]))
    sigma_hat = {s: mad_scale(np.array(bufs_s[s])) for s in range(1, S + 1)}
    buf_n = deque(float(loo_res[k]) / sigma_hat[cal_season(idx, month0)]
                  for k, idx in enumerate(all_idx[:n_tr]))

    out = {m: (np.empty(T1, dtype=bool), np.empty(T1), np.empty(T1))
           for m in METHODS}

    for step in range(T1):
        t0 = T + step
        f = float(test_preds[step])
        y_t = float(Y[t0])
        ss = cal_season(t0, month0)
        arr_p = np.array(buf_p)
        arr_s = np.array(bufs_s[ss]) if bufs_s[ss] else np.zeros(1)
        arr_n = np.array(buf_n)
        sc = sigma_hat[ss]
        for name, arr, scale, corr in [
                ("Pooled", arr_p, 1.0, False),
                ("EnbPI-N", arr_n, sc, False),
                ("EnbPI-S", arr_s, 1.0, False),
                ("Pooled-C", arr_p, 1.0, True),
                ("EnbPI-N-C", arr_n, sc, True),
                ("EnbPI-S-C", arr_s, 1.0, True)]:
            q_lo, q_hi = interval_from_buffer(arr, alpha, corr)
            cov, lo_a, hi_a = out[name]
            lo_a[step] = f + scale * q_lo
            hi_a[step] = f + scale * q_hi
            cov[step] = bool(lo_a[step] <= y_t <= hi_a[step])

        if (step + 1) % s0 == 0:
            for delta in range(s0):
                j0 = T + step - 1 - delta
                if j0 >= p:
                    eps = get_res(j0)
                    sj = cal_season(j0, month0)
                    buf_p.popleft(); buf_p.append(eps)
                    buf_n.popleft(); buf_n.append(eps / sigma_hat[sj])
                    if bufs_s[sj]:
                        bufs_s[sj].popleft()
                    bufs_s[sj].append(eps)
    return out


def summarise(out, T, T1, month0):
    rows = []
    for m in METHODS:
        cov, lo, hi = out[m]
        wid = hi - lo
        per_c, per_w = [], []
        for s in range(1, S + 1):
            mask = np.array([cal_season(T + k, month0) == s
                             for k in range(T1)])
            per_c.append(float(cov[mask].mean()))
            per_w.append(float(wid[mask].mean()))
        rows.append({
            "method": m,
            "coverage": float(cov.mean()),
            "min_season_cov": float(np.min(per_c)),
            "spread": float(np.max(per_c) - np.min(per_c)),
            "width_mean": float(wid.mean()),
            "width_min": float(np.min(per_w)),
            "width_max": float(np.max(per_w)),
            "width_ratio": float(np.max(per_w) / np.min(per_w)),
            "argmax_month": MONTHS[int(np.argmax(per_w))],
            "argmin_month": MONTHS[int(np.argmin(per_w))],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------- diagnostic
def seasonal_discrepancy(Y, T, month0, alpha=ALPHA, B=B_BOOT,
                         p=P_LAGS, seed=42, n_perm=999):
    """
    Two sup-norm discrepancies computed from the training LOO residuals:

      bhat   = max_s sup_x |Fhat_s(x) - Fhat_pool(x)|
               the empirical counterpart of the seasonal bias b_{s*}
               (Section 3): how far pooled calibration is from
               season-specific calibration.

      bhat_N = max_s sup_x |Ghat_s(x) - Ghat_pool(x)| computed on
               residuals standardised by the season-specific scale:
               the part of the seasonal discrepancy that a scale
               normalisation cannot remove.

    Both statistics are compared with the distribution they would have
    if the season labels carried no information, obtained by randomly
    permuting the labels among the residuals (n_perm draws).  The
    p-value is the share of permutations whose statistic is at least as
    large as the observed one.  Under a permutation the season scales
    are re-estimated, so the reference respects the null in both cases.
    """
    rng = np.random.default_rng(seed)
    tr_idx = np.arange(p, T)
    X_tr = sim.build_features(Y, tr_idx, p)
    y_tr = Y[tr_idx]
    models, Sb = sim.fit_ensemble(X_tr, y_tr, B, "ridge", rng)
    preds = sim.batch_predict(models, X_tr)
    res = y_tr - sim.loo_predictions(preds, Sb, len(tr_idx))

    seas = np.array([cal_season(i, month0) for i in tr_idx])

    def sup_gap(sample, pool_sorted):
        f1 = np.searchsorted(np.sort(sample), pool_sorted,
                             side="right") / len(sample)
        f0 = np.searchsorted(pool_sorted, pool_sorted,
                             side="right") / len(pool_sorted)
        return float(np.max(np.abs(f1 - f0)))

    def stats(labels):
        pool_sorted = np.sort(res)
        raw = max(sup_gap(res[labels == s], pool_sorted)
                  for s in range(1, S + 1))
        sc = {s: mad_scale(res[labels == s]) for s in range(1, S + 1)}
        zz = res / np.array([sc[l] for l in labels])
        z_sorted = np.sort(zz)
        nrm = max(sup_gap(zz[labels == s], z_sorted)
                  for s in range(1, S + 1))
        return raw, nrm

    b_raw, b_norm = stats(seas)

    rng_p = np.random.default_rng(seed + 7)
    ge_raw = ge_norm = 0
    for _ in range(n_perm):
        perm = rng_p.permutation(seas)
        r, n = stats(perm)
        ge_raw += (r >= b_raw)
        ge_norm += (n >= b_norm)

    Ts = int(np.median([np.sum(seas == s) for s in range(1, S + 1)]))
    return {"bhat": b_raw, "p_bhat": (ge_raw + 1) / (n_perm + 1),
            "bhat_norm": b_norm, "p_bhat_norm": (ge_norm + 1) / (n_perm + 1),
            "Ts": Ts, "n_perm": n_perm}


# --------------------------------------------------------------- LaTeX
def latex_table(df_food, df_exp):
    def block(df, label):
        out = []
        for m in ["Pooled", "EnbPI-N", "EnbPI-S", "EnbPI-S-C"]:
            r = df[df["method"] == m].iloc[0]
            name = (r"\texttt{EnbPI-S}" if m == "EnbPI-S" else
                    r"\texttt{EnbPI-S+}" if m == "EnbPI-S-C" else
                    r"\texttt{EnbPI-N}" if m == "EnbPI-N" else "Pooled")
            first = label if m == "Pooled" else ""
            out.append(
                f"{first} & {name} & {r['coverage']:.3f} &"
                f" {r['min_season_cov']:.3f} & {r['spread']:.3f} &"
                f" {r['width_mean']:.2f} & {r['width_ratio']:.2f} " + r"\\")
        return out

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Competing calibration schemes on the two empirical",
        r"         applications ($\alpha = 0.10$, test window",
        r"         Jan 2015--Dec 2024, $n_s = 10$ observations per season).",
        r"         \texttt{EnbPI-N} divides each residual by a",
        r"         season-specific robust scale and calibrates on the",
        r"         pooled standardised buffer; \texttt{EnbPI-S+} is",
        r"         \texttt{EnbPI-S} with the finite-sample conformal",
        r"         quantile levels.  Spread is the largest minus the",
        r"         smallest per-season coverage; width ratio is the",
        r"         largest over the smallest per-season mean width.}",
        r"\label{tab:empirical_baselines}",
        r"\begin{tabular}{l l ccc cc}",
        r"\toprule",
        r"Application & Method & Coverage & Worst season & Spread"
        r" & Mean width & Width ratio \\",
        r"\midrule",
    ]
    lines += block(df_food, r"IPCA food ($\Ts = 20$)")
    lines.append(r"\midrule")
    lines += block(df_exp, r"Exports ($\Ts = 35$)")
    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\footnotesize Nominal coverage $0.90$."
        r"  Widths are in percentage points for the IPCA application and",
        r" in per cent}\\",
        r"\multicolumn{7}{l}{\footnotesize for the export-growth",
        r" application, so they are comparable only within an",
        r" application.}\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print("=== Food IPCA (series 1635) ===")
    Y, T, T1, m0, dates = load_food()
    print(f"  T={T}  T1={T1}  month0={m0}  ({dates[0].date()} .. {dates[-1].date()})")
    out_food = run_all_methods(Y, T, T1, m0)
    df_food = summarise(out_food, T, T1, m0)
    print(df_food.to_string(index=False))
    diag_food = seasonal_discrepancy(Y, T, m0)
    print("  diagnostic:", {k: round(v, 3) for k, v in diag_food.items()})

    print("\n=== Exports (series 1402) ===")
    Y, T, T1, m0, dates = load_exports()
    print(f"  T={T}  T1={T1}  month0={m0}  ({dates[0].date()} .. {dates[-1].date()})")
    out_exp = run_all_methods(Y, T, T1, m0)
    df_exp = summarise(out_exp, T, T1, m0)
    print(df_exp.to_string(index=False))
    diag_exp = seasonal_discrepancy(Y, T, m0)
    print("  diagnostic:", {k: round(v, 3) for k, v in diag_exp.items()})

    pd.DataFrame([dict(application="food", **diag_food),
                  dict(application="exports", **diag_exp)]).to_csv(
        os.path.join(TAB, "empirical_diagnostic.csv"), index=False)

    df_food.insert(0, "application", "food")
    df_exp.insert(0, "application", "exports")
    pd.concat([df_food, df_exp], ignore_index=True).to_csv(
        os.path.join(TAB, "empirical_baselines.csv"), index=False)
    with open(os.path.join(TAB, "tab_empirical_baselines.tex"), "w",
              encoding="utf-8") as fh:
        fh.write(latex_table(df_food, df_exp))
    print("\nSaved tables/empirical_baselines.csv and "
          "tables/tab_empirical_baselines.tex")
