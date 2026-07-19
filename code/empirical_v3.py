"""
empirical_v3.py
===============
Second empirical application: Brazilian monthly export growth
(BCB SGS series 1402 -- Exportacoes brasileiras, US$ millions).

This application provides a test case with T_s = 35-36 observations
per season (training 1979-2014), in the regime where Theorem 1-S
and Experiment E2 predict reliable coverage improvements for
EnbPI-S over pooled calibration.

Seasonal heteroskedasticity mechanism:
  Brazilian exports are dominated by agricultural commodities
  (soybeans, coffee, sugarcane/ethanol, beef).  Harvest-cycle
  uncertainty creates clear seasonal differences in forecast-error
  variance: the harvest-export ramp-up (Mar-May, peaking in April)
  exhibits high variance; September is the most predictable month.
  Because the series is USD-denominated, it is unaffected by
  the Brazilian hyperinflation of the 1980s.

Y_t is the monthly log-return:  Y_t = 100 * log(V_t / V_{t-1})

Train / test split
------------------
  Full series (levels)     : Feb 1979 - Dec 2024
  Log-returns              : Mar 1979 - Dec 2024
  Training    : Mar 1979 - Dec 2014  (T = 430; 35 or 36 obs per season)
  Test        : Jan 2015 - Dec 2024  (T1 = 120, same window as food IPCA)

IMPORTANT: the log-return series starts in MARCH 1979, so array
index 0 corresponds to March, not January.  All season labels are
therefore derived from the true calendar month via cal_season()
below (a bug in an earlier version used season(), which assumes
index 0 = January and mislabelled every month by +2).

Usage
-----
  python code/empirical_v3.py           # download and run
  python code/empirical_v3.py --offline # use cached data
"""

import sys, os, warnings
import numpy as np
import pandas as pd
from collections import deque
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from simulation import (
    season, build_features, fit_ensemble, batch_predict,
    loo_predictions, empirical_quantile, line_search_beta,
    B_BOOT, P_LAGS, ALPHA, S,
)

# ── Config ────────────────────────────────────────────────────────────
BCB_SERIES  = 1402
SERIES_NAME = "Exportacoes brasileiras (USD mi)"
TRAIN_END   = "2014-12"
TEST_START  = "2015-01"
TEST_END    = "2024-12"
BCB_URL     = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{BCB_SERIES}/dados"
DATA_CACHE  = os.path.join(_HERE, "..", "data", "exports_raw.csv")

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


def cal_season(t0: int, month0: int, S_: int = S) -> int:
    """True calendar season of array index t0, where index 0 falls in
    calendar month `month0` (1-12).  Returns a value in 1..S_."""
    return ((t0 + month0 - 1) % S_) + 1


# ── 1. Data ───────────────────────────────────────────────────────────

def download_series() -> pd.DataFrame:
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")
    print(f"  Downloading BCB series {BCB_SERIES} ({SERIES_NAME})...",
          end=" ", flush=True)
    r = requests.get(BCB_URL, params={
        "formato": "json",
        "dataInicial": "01/01/1979",
        "dataFinal":   "31/12/2024",
    }, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"]  = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["value"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df[["date","value"]].dropna().sort_values("date").reset_index(drop=True)
    print(f"done ({len(df)} obs, {df['date'].min().date()} to {df['date'].max().date()})")
    return df


def load_series(offline: bool = False) -> pd.DataFrame:
    os.makedirs(os.path.dirname(DATA_CACHE), exist_ok=True)
    if offline and os.path.exists(DATA_CACHE):
        print(f"  Loading cached data from {DATA_CACHE}")
        return pd.read_csv(DATA_CACHE, parse_dates=["date"])
    df = download_series()
    df.to_csv(DATA_CACHE, index=False)
    return df


def prepare_arrays(df: pd.DataFrame):
    """Compute log-returns, split into train/test arrays."""
    df = df.set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp()

    # Monthly log-return: Y_t = 100 * log(X_t / X_{t-1})
    lvl = df["value"]
    lr  = 100.0 * np.log(lvl / lvl.shift(1))
    lr  = lr.dropna()

    # Keep full series; identify train/test split
    train_mask = lr.index <= pd.Timestamp(TRAIN_END)
    test_mask  = (lr.index >= pd.Timestamp(TEST_START)) & \
                 (lr.index <= pd.Timestamp(TEST_END))

    T  = int(train_mask.sum())
    T1 = int(test_mask.sum())

    Y_full = lr.values.astype(float)
    dates  = lr.index

    print(f"  Series  : {dates[0].date()} to {dates[-1].date()} ({len(Y_full)} log-returns)")
    print(f"  Training: {dates[train_mask][0].date()} to "
          f"{dates[train_mask][-1].date()} (T={T}, T_s={T//S})")
    print(f"  Test    : {dates[test_mask][0].date()} to "
          f"{dates[test_mask][-1].date()} (T1={T1})")
    return Y_full, T, T1, dates


# ── 2. Run both methods (shared with empirical_v2) ────────────────────

def run_both(Y, T, T1, alpha=ALPHA, B=B_BOOT, s0=1, S=S, p=P_LAGS, seed=42,
             month0=1):
    rng = np.random.default_rng(seed)
    all_idx = np.arange(p, T + T1)
    X_all   = build_features(Y, all_idx, p)
    y_all   = Y[all_idx]
    n_tr    = T - p
    X_tr, y_tr = X_all[:n_tr], y_all[:n_tr]

    print("  Fitting ensemble ...", end=" ", flush=True)
    models, Sb_sets = fit_ensemble(X_tr, y_tr, B, "ridge", rng)
    print("done")
    print("  Computing residuals ...", end=" ", flush=True)
    all_preds_B = batch_predict(models, X_all)
    ens_preds   = all_preds_B.mean(axis=0)
    loo_p_tr    = loo_predictions(all_preds_B[:, :n_tr], Sb_sets, n_tr)
    loo_res     = y_tr - loo_p_tr
    full_res    = y_all - ens_preds

    def get_res(j0):
        k = j0 - p
        return float(full_res[k]) if 0 <= k < len(full_res) else 0.0

    test_preds = ens_preds[T - p: T - p + T1]

    buf_p  = deque(loo_res.tolist())
    bufs_s = {s_: deque() for s_ in range(1, S + 1)}
    for k, idx in enumerate(all_idx[:n_tr]):
        bufs_s[cal_season(idx, month0, S)].append(float(loo_res[k]))

    cov_p = np.empty(T1, dtype=bool); cov_s = np.empty(T1, dtype=bool)
    lo_p  = np.empty(T1); hi_p = np.empty(T1)
    lo_s  = np.empty(T1); hi_s = np.empty(T1)

    print("done")
    print("  Running prediction loop ...", end=" ", flush=True)
    for step in range(T1):
        t0  = T + step
        f   = float(test_preds[step])
        y_t = float(Y[t0])
        ss  = cal_season(t0, month0, S)

        arr_p = np.array(buf_p)
        b_p   = line_search_beta(arr_p, alpha)
        lo_p[step] = f + empirical_quantile(arr_p, b_p)
        hi_p[step] = f + empirical_quantile(arr_p, 1 - alpha + b_p)
        cov_p[step] = bool(lo_p[step] <= y_t <= hi_p[step])

        arr_s = np.array(bufs_s[ss]) if bufs_s[ss] else np.zeros(1)
        b_s   = line_search_beta(arr_s, alpha)
        lo_s[step] = f + empirical_quantile(arr_s, b_s)
        hi_s[step] = f + empirical_quantile(arr_s, 1 - alpha + b_s)
        cov_s[step] = bool(lo_s[step] <= y_t <= hi_s[step])

        if (step + 1) % s0 == 0:
            for delta in range(s0):
                j0 = T + step - 1 - delta
                if j0 >= p:
                    eps = get_res(j0)
                    buf_p.popleft(); buf_p.append(eps)
                    sj = cal_season(j0, month0, S)
                    if bufs_s[sj]: bufs_s[sj].popleft()
                    bufs_s[sj].append(eps)
    print("done")
    return (cov_p, lo_p, hi_p), (cov_s, lo_s, hi_s)


# ── 3. Results ────────────────────────────────────────────────────────

def season_summary(cov_p, lo_p, hi_p, cov_s, lo_s, hi_s, T, month0=1):
    T1 = len(cov_p)
    rows = []
    for s_ in range(1, S + 1):
        mask = np.array([cal_season(T + k, month0, S) == s_ for k in range(T1)])
        if mask.any():
            rows.append({
                "season":    s_,
                "month":     MONTH_NAMES[s_ - 1],
                "n":         int(mask.sum()),
                "cov_pool":  float(cov_p[mask].mean()),
                "cov_strat": float(cov_s[mask].mean()),
                "wid_pool":  float((hi_p - lo_p)[mask].mean()),
                "wid_strat": float((hi_s - lo_s)[mask].mean()),
            })
    return pd.DataFrame(rows)


def latex_table(df, T, T1, Ts, alpha=ALPHA):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-season empirical coverage and average width "
        r"for Brazilian monthly export growth forecasting.",
        f"Training: Mar 1979--Dec 2014 ($T={T}$; $T_s = 35$ or $36$ per season); "
        f"test: Jan 2015--Dec 2024 ($T_1={T1}$); $\\alpha={alpha}$.",
        r"Bold coverage = closest to nominal; italic width = narrowest.}",
        r"\label{tab:empirical_exports_coverage}",
        r"\begin{tabular}{l r cc cc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Coverage} & "
        r"\multicolumn{2}{c}{Width (\%)} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Month & $n$ & Pooled & EnbPI-S & Pooled & EnbPI-S \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cp, cs = row["cov_pool"], row["cov_strat"]
        wp, ws = row["wid_pool"], row["wid_strat"]
        cp_s = f"\\textbf{{{cp:.2f}}}" if abs(cp-(1-alpha))<=abs(cs-(1-alpha)) else f"{cp:.2f}"
        cs_s = f"\\textbf{{{cs:.2f}}}" if abs(cs-(1-alpha))< abs(cp-(1-alpha)) else f"{cs:.2f}"
        wp_s = f"\\textit{{{wp:.2f}}}"  if wp <= ws else f"{wp:.2f}"
        ws_s = f"\\textit{{{ws:.2f}}}"  if ws <  wp else f"{ws:.2f}"
        lines.append(f"{row['month']} & {row['n']} & {cp_s} & {cs_s} "
                     f"& {wp_s} & {ws_s} \\\\")
    oc_p = df["cov_pool"].mean(); oc_s = df["cov_strat"].mean()
    ow_p = df["wid_pool"].mean(); ow_s = df["wid_strat"].mean()
    lines += [
        r"\midrule",
        f"\\textbf{{Overall}} & {int(df['n'].sum())} "
        f"& {oc_p:.2f} & {oc_s:.2f} & {ow_p:.2f} & {ow_s:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── 4. Figures ────────────────────────────────────────────────────────

def make_figures(df_season, Y, T, T1, dates_test, figures_dir,
                 lo_p=None, hi_p=None, cov_p=None,
                 lo_s=None, hi_s=None, cov_s=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import matplotlib.dates as mdates
    except ImportError:
        print("  matplotlib not available — skip figures.")
        return

    x = np.arange(1, S + 1)
    w = 0.35

    # Figure 1: per-season coverage
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(x - w/2, df_season["cov_pool"],  w, label="Pooled",  color="tab:red",  alpha=0.75)
    ax.bar(x + w/2, df_season["cov_strat"], w, label="EnbPI-S", color="tab:blue", alpha=0.75)
    ax.axhline(1 - ALPHA, color="k", ls="--", lw=0.9,
               label=f"Nominal ({100*(1-ALPHA):.0f}%)")
    ax.set_xticks(x); ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Month"); ax.set_ylabel("Empirical coverage")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Per-season coverage — Brazilian export growth ({TEST_START}–{TEST_END})")
    fig.tight_layout()
    p = os.path.join(figures_dir, "fig_empirical_exports_coverage.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # Figure 2: interval width by season
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(x - w/2, df_season["wid_pool"],  w, label="Pooled",  color="tab:red",  alpha=0.75)
    ax.bar(x + w/2, df_season["wid_strat"], w, label="EnbPI-S", color="tab:blue", alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.set_xlabel("Month"); ax.set_ylabel("Mean interval width (%)")
    ax.legend(fontsize=8)
    ax.set_title("Interval width by season — Brazilian export growth")
    fig.tight_layout()
    p = os.path.join(figures_dir, "fig_empirical_exports_width.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # Figure 3: time-series comparison
    if lo_p is not None:
        Y_test    = Y[T: T + T1]
        dates_dt  = pd.to_datetime(dates_test)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.subplots_adjust(hspace=0.08)

        for ax, lo, hi, cov, label, lc, fc in [
            (axes[0], lo_p, hi_p, cov_p, "Pooled EnbPI",  "tab:red",  "#ffd5d5"),
            (axes[1], lo_s, hi_s, cov_s, "EnbPI-S",       "tab:blue", "#d5e8ff"),
        ]:
            ax.fill_between(dates_dt, lo, hi, color=fc, alpha=0.9,
                            label=f"90% interval ({label})")
            ax.plot(dates_dt, lo, color=lc, lw=0.5, alpha=0.6)
            ax.plot(dates_dt, hi, color=lc, lw=0.5, alpha=0.6)
            ax.plot(dates_dt, Y_test, color="black", lw=0.9,
                    label="Realised log-return")
            missed = ~np.array(cov, dtype=bool)
            if missed.any():
                ax.scatter(dates_dt[missed], Y_test[missed],
                           color="red", zorder=5, s=18,
                           label=f"Missed ({missed.sum()})")
            ax.set_ylabel("% (log-ret.)", fontsize=9)
            ax.legend(fontsize=7.5, loc="upper right", ncol=2)
            ax.set_title(label, fontsize=10, pad=3)
            ax.axhline(0, color="gray", lw=0.4, ls=":")

        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate(rotation=0, ha="center")
        fig.suptitle(
            "Prediction intervals vs realised export growth\n"
            f"(test: {TEST_START}–{TEST_END}, $\\alpha=0.10$)",
            fontsize=11)
        p = os.path.join(figures_dir, "fig_empirical_exports_timeseries.pdf")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {p}")


# ── 5. Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    offline = "--offline" in sys.argv

    proj   = os.path.dirname(_HERE)
    tables = os.path.join(proj, "tables");  os.makedirs(tables, exist_ok=True)
    figs   = os.path.join(proj, "figures"); os.makedirs(figs,   exist_ok=True)

    print(f"=== Empirical: {SERIES_NAME} ===\n")

    print("--- Data ---")
    df_raw       = load_series(offline)
    Y, T, T1, dates = prepare_arrays(df_raw)
    Ts = T // S

    month0 = int(dates[0].month)   # calendar month of array index 0 (March)
    print(f"    (array index 0 = calendar month {month0})")

    print("\n--- Running methods ---")
    (cov_p, lo_p, hi_p), (cov_s, lo_s, hi_s) = run_both(Y, T, T1, month0=month0)

    dates_test = dates[dates >= pd.Timestamp(TEST_START)]
    df_s       = season_summary(cov_p, lo_p, hi_p, cov_s, lo_s, hi_s, T,
                                month0=month0)

    print("\n--- Results ---")
    print(f"{'Month':<6} {'Cov_P':>7} {'Cov_S':>7} {'Wid_P':>8} {'Wid_S':>8}")
    print("-" * 44)
    for _, r in df_s.iterrows():
        marker = " *" if r["cov_strat"] > r["cov_pool"] else ""
        print(f"{r['month']:<6} {r['cov_pool']:>7.3f} {r['cov_strat']:>7.3f} "
              f"{r['wid_pool']:>8.3f} {r['wid_strat']:>8.3f}{marker}")
    print("-" * 44)
    oc_p = df_s["cov_pool"].mean(); oc_s = df_s["cov_strat"].mean()
    print(f"{'Mean':<6} {oc_p:>7.3f} {oc_s:>7.3f} "
          f"{df_s['wid_pool'].mean():>8.3f} {df_s['wid_strat'].mean():>8.3f}")
    print(f"\nNominal: {1-ALPHA:.0%}")
    print(f"T_s = {Ts}  (training seasons per month)")
    print(f"Pooled coverage gap  : {abs(oc_p-(1-ALPHA)):.3f}")
    print(f"EnbPI-S coverage gap : {abs(oc_s-(1-ALPHA)):.3f}")

    # Width ratio: compute from actual data
    widths_s = df_s["wid_strat"].values
    wid_max  = widths_s.max()
    wid_min  = widths_s.min()
    month_max = df_s.loc[df_s["wid_strat"].idxmax(), "month"]
    month_min = df_s.loc[df_s["wid_strat"].idxmin(), "month"]
    print(f"\nEnbPI-S width range: {wid_min:.2f} ({month_min}) to "
          f"{wid_max:.2f} ({month_max}), ratio = {wid_max/wid_min:.2f}")
    print(f"Pooled  width range: {df_s['wid_pool'].min():.2f} to "
          f"{df_s['wid_pool'].max():.2f}  (ratio = "
          f"{df_s['wid_pool'].max()/df_s['wid_pool'].min():.2f})")

    # Save
    path_csv = os.path.join(tables, "empirical_exports_coverage.csv")
    df_s.to_csv(path_csv, index=False); print(f"\n  Saved {path_csv}")

    df_full = pd.DataFrame({
        "date": dates_test[:T1], "y": Y[T: T + T1],
        "lo_pool": lo_p, "hi_pool": hi_p, "covered_pool": cov_p,
        "lo_strat":lo_s, "hi_strat":hi_s, "covered_strat":cov_s,
    })
    path_full = os.path.join(tables, "empirical_exports_intervals.csv")
    df_full.to_csv(path_full, index=False); print(f"  Saved {path_full}")

    tex = latex_table(df_s, T, T1, Ts)
    path_tex = os.path.join(tables, "tab_empirical_exports_coverage.tex")
    with open(path_tex, "w", encoding="utf-8") as f: f.write(tex)
    print(f"  Saved {path_tex}")

    print("\n--- Figures ---")
    make_figures(df_s, Y, T, T1, dates_test, figs,
                 lo_p=lo_p, hi_p=hi_p, cov_p=cov_p,
                 lo_s=lo_s, hi_s=hi_s, cov_s=cov_s)
    print("\nDone.")
