"""
simulation_hybrid.py
====================
Experiment E8: the hybrid between the two seasonal corrections.

Normalising and stratifying are the two ways of letting the
calibration depend on the season.  Combining them by standardising
the residuals and then stratifying the standardised buffer is NOT a
new method: within a season every residual is divided by the same
constant and the resulting quantiles are multiplied back by it, so
the interval is exactly the one \texttt{EnbPI-S} produces.  A genuine
hybrid has to interpolate between the two estimators rather than
compose them.  Write

    q_lo^S, q_hi^S            offsets from the season-s buffer of raw
                              residuals              (EnbPI-S)
    sigma_s * q_lo^N, ...     offsets from the pooled buffer of
                              standardised residuals (EnbPI-N)

and take, for lambda in [0, 1],

    q^{NS}(lambda) = lambda * q^S + (1 - lambda) * sigma_s * q^N.

lambda = 1 is EnbPI-S, lambda = 0 is EnbPI-N, and intermediate values
shrink the season-specific quantile towards the normalised pooled
one.  Each endpoint of the interval is shrunk separately, so the
asymmetry of the two schemes is interpolated as well.

Run:
    python simulation_hybrid.py            # n_rep = 200
    python simulation_hybrid.py --quick    # n_rep = 30
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import deque

import simulation as sim
import simulation_baselines as sb

S = sim.S
ALPHA = sim.ALPHA
B_BOOT = sim.B_BOOT
P_LAGS = sim.P_LAGS
T1 = sim.T1
HIGH = sim.HIGH_SEAS

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def lam_name(lam):
    return f"NS-{lam:.2f}"


METHODS_H = [lam_name(l) for l in LAMBDAS]


def run_one_rep_hybrid(Y, T, T1, alpha, B, s0, S, model_type, p, rng):
    """All lambda values at once, sharing one ensemble."""
    tr_idx = np.arange(p, T)
    n_tr = len(tr_idx)
    X_tr = sim.build_features(Y, tr_idx, p)
    y_tr = Y[tr_idx]

    models, Sb_sets = sim.fit_ensemble(X_tr, y_tr, B, model_type, rng)
    all_idx = np.arange(p, T + T1)
    X_all = sim.build_features(Y, all_idx, p)
    y_all = Y[all_idx]
    preds_B = sim.batch_predict(models, X_all)
    ens = preds_B.mean(axis=0)
    loo_res = y_tr - sim.loo_predictions(preds_B[:, :n_tr], Sb_sets, n_tr)
    full_res = y_all - ens

    def get_res(j0):
        k = j0 - p
        return float(full_res[k]) if 0 <= k < len(full_res) else 0.0

    test_preds = ens[T - p: T - p + T1]

    bufs_s = {s: deque() for s in range(1, S + 1)}
    for k, i in enumerate(tr_idx):
        bufs_s[sim.season(i, S)].append(float(loo_res[k]))
    sigma_hat = {s: sb.mad_scale(np.array(bufs_s[s])) for s in range(1, S + 1)}
    buf_n = deque(float(loo_res[k]) / sigma_hat[sim.season(i, S)]
                  for k, i in enumerate(tr_idx))

    out = {m: (np.empty(T1, dtype=bool), np.empty(T1), np.empty(T1))
           for m in METHODS_H}

    for step in range(T1):
        t0 = T + step
        f = float(test_preds[step])
        y_t = float(Y[t0])
        s_star = sim.season(t0, S)
        arr_s = np.array(bufs_s[s_star]) if bufs_s[s_star] else np.zeros(1)
        arr_n = np.array(buf_n)
        sc = sigma_hat[s_star]

        lo_S, hi_S = sb.interval_from_buffer(arr_s, alpha, False)
        lo_N, hi_N = sb.interval_from_buffer(arr_n, alpha, False)
        lo_N, hi_N = sc * lo_N, sc * hi_N

        for lam in LAMBDAS:
            lo = f + lam * lo_S + (1 - lam) * lo_N
            hi = f + lam * hi_S + (1 - lam) * hi_N
            cov, lo_a, hi_a = out[lam_name(lam)]
            lo_a[step] = lo
            hi_a[step] = hi
            cov[step] = bool(lo <= y_t <= hi)

        if (step + 1) % s0 == 0:
            for delta in range(s0):
                j0 = T + step - 1 - delta
                if j0 >= p:
                    eps = get_res(j0)
                    s_j = sim.season(j0, S)
                    buf_n.popleft()
                    buf_n.append(eps / sigma_hat[s_j])
                    if bufs_s[s_j]:
                        bufs_s[s_j].popleft()
                    bufs_s[s_j].append(eps)
    return out


def monte_carlo_hybrid(n_rep, T, seed_base, dgp="scale"):
    acc = {m: [] for m in METHODS_H}
    for rep in range(n_rep):
        rng = np.random.default_rng(seed_base + rep * 1000)
        if dgp == "scale":
            Y = sim.generate_series(T + T1, sim.SIGMA_HIGH, sim.SIGMA_LOW, rng)
        else:
            Y = sb.generate_series_shape(T + T1, rng)
        res = run_one_rep_hybrid(Y, T, T1, ALPHA, B_BOOT, 1, S,
                                 "ridge", P_LAGS, rng)
        for m in METHODS_H:
            cov, lo, hi = res[m]
            acc[m].append(sim.compute_metrics(cov, lo, hi, T, S))
        if (rep + 1) % 25 == 0:
            print(f"    rep {rep+1}/{n_rep}", flush=True)
    return acc


def summarise(acc, n_rep, T, dgp):
    rows = []
    for m in METHODS_H:
        summ = sim.summarise(acc[m])
        per = [summ["seasonal_coverage"][s][0] for s in range(1, S + 1)]
        hv = [summ["seasonal_coverage"][s][0] for s in sorted(HIGH)]
        lv = [summ["seasonal_coverage"][s][0]
              for s in range(1, S + 1) if s not in HIGH]
        wh = [summ["seasonal_width"][s][0] for s in sorted(HIGH)]
        wl = [summ["seasonal_width"][s][0]
              for s in range(1, S + 1) if s not in HIGH]
        rows.append({
            "dgp": dgp, "T": T, "lambda": float(m.split("-")[1]),
            "coverage": summ["overall_coverage_mean"],
            "coverage_se": summ["overall_coverage_std"] / np.sqrt(n_rep),
            "cov_high": float(np.mean(hv)), "cov_low": float(np.mean(lv)),
            "spread": float(np.max(per) - np.min(per)),
            "worst": float(np.min(per)),
            "width": summ["overall_width_mean"],
            "w_high": float(np.mean(wh)), "w_low": float(np.mean(wl)),
        })
    return pd.DataFrame(rows)


def latex_table(df, tab_dir):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Experiment E8: the hybrid",
        r"         $q^{NS}(\lambda) = \lambda\,q^{S}",
        r"         + (1-\lambda)\,\hat{\sigma}_{s^*} q^{N}$ between",
        r"         normalisation ($\lambda = 0$) and stratification",
        r"         ($\lambda = 1$), at $T = 480$ under the two",
        r"         data-generating processes of Experiments E6 and E7",
        r"         ($\alpha = 0.10$, $n_{\mathrm{rep}} = 200$).",
        r"         Spread is the largest minus the smallest per-season",
        r"         coverage.}",
        r"\label{tab:e8_hybrid}",
        r"\begin{tabular}{c cccc c cccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Seasonal scale heterogeneity (E6)} & &"
        r" \multicolumn{4}{c}{Seasonal shape heterogeneity (E7)} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){7-10}",
        r"$\lambda$ & Coverage & High & Low & Spread & &"
        r" Coverage & Skewed & Gaussian & Spread \\",
        r"\midrule",
    ]
    for lam in LAMBDAS:
        a = df[(df["dgp"] == "scale") & (df["lambda"] == lam)].iloc[0]
        b = df[(df["dgp"] == "shape") & (df["lambda"] == lam)].iloc[0]
        lines.append(
            f"{lam:.2f} & {a['coverage']:.3f} & {a['cov_high']:.3f} & "
            f"{a['cov_low']:.3f} & {a['spread']:.3f} & & "
            f"{b['coverage']:.3f} & {b['cov_high']:.3f} & "
            f"{b['cov_low']:.3f} & {b['spread']:.3f} " + r"\\")
    se = df["coverage_se"].max()
    lines += [
        r"\bottomrule",
        r"\multicolumn{10}{l}{\footnotesize Nominal coverage $0.90$;"
        f" Monte Carlo standard errors do not exceed ${se:.3f}$."
        r"  $\lambda = 1$ reproduces \texttt{EnbPI-S} and}\\",
        r"\multicolumn{10}{l}{\footnotesize $\lambda = 0$ reproduces"
        r" \texttt{EnbPI-N} of Table~\ref{tab:e6_schemes}.}\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path = os.path.join(tab_dir, "tab_E8_hybrid.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_rep = 30 if quick else 200
    here = os.path.dirname(os.path.abspath(__file__))
    tab = os.path.join(os.path.dirname(here), "tables")

    frames = []
    for dgp, seed in [("scale", 1000), ("shape", 6000)]:
        print(f"=== hybrid, {dgp} DGP, T=480 ===")
        acc = monte_carlo_hybrid(n_rep, 480, seed_base=seed, dgp=dgp)
        frames.append(summarise(acc, n_rep, 480, dgp))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(tab, "X5_hybrid.csv"), index=False)
    print(df.round(3).to_string(index=False))
    latex_table(df, tab)
