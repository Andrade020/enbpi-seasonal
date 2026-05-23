"""
empirical_v2.py
===============
Empirical application to monthly Brazilian food-at-home inflation
(IPCA - Alimentacao no domicilio, BCB SGS series 1635).

Food prices exhibit strong seasonal heteroskedasticity driven by
harvest cycles (lower volatility in harvest months Mar-Jun),
administered-price resets (higher volatility in January), and
mid-year adjustments (higher volatility in Jul-Aug).  This makes
food IPCA a cleaner test case for EnbPI-S than headline IPCA,
whose 2020-22 regime shift confounds the analysis.

Data source
-----------
  BCB SGS series 1635 -- IPCA Alimentacao no domicilio, % variation m/m
  URL: https://api.bcb.gov.br/dados/serie/bcdata.sgs.1635/dados
  Period used: Jan 1995 - Dec 2024  (360 months)

Train / test split
------------------
  Train: Jan 1995 - Dec 2014  (T = 240, T_s = 20/season)
  Test : Jan 2015 - Dec 2024  (T1 = 120, 10 obs/season)

Usage
-----
  python code/empirical_v2.py           # download and run
  python code/empirical_v2.py --offline # use cached data
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
BCB_SERIES  = 1635
SERIES_NAME = "IPCA - Alimentacao no domicilio"
TRAIN_START = "1995-01"
TRAIN_END   = "2014-12"
TEST_START  = "2015-01"
TEST_END    = "2024-12"
BCB_URL     = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{BCB_SERIES}/dados"
DATA_CACHE  = os.path.join(_HERE, "..", "data", "ipca_food_raw.csv")

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


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
        "dataInicial": "01/01/1994",
        "dataFinal":   "31/12/2025",
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
    df = df.set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp()
    req_start = pd.Timestamp(TRAIN_START)
    req_end   = pd.Timestamp(TEST_END)
    sub = df.loc[(df.index >= req_start) & (df.index <= req_end), "value"]
    Y_full = sub.values.astype(float)
    dates  = sub.index
    T  = int((dates <= pd.Timestamp(TRAIN_END)).sum())
    T1 = int((dates >= pd.Timestamp(TEST_START)).sum())
    print(f"  Training: {dates[:T][0].date()} to {dates[:T][-1].date()} (T={T})")
    print(f"  Test    : {dates[T:][0].date()} to {dates[T:][-1].date()} (T1={T1})")
    return Y_full, T, T1, dates


# ── 2. Run both methods ───────────────────────────────────────────────

def run_both(Y, T, T1, alpha=ALPHA, B=B_BOOT, s0=1, S=S, p=P_LAGS, seed=42):
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

    buf_p = deque(loo_res.tolist())
    bufs_s = {s_: deque() for s_ in range(1, S + 1)}
    for k, idx in enumerate(all_idx[:n_tr]):
        bufs_s[season(idx, S)].append(float(loo_res[k]))

    cov_p = np.empty(T1, dtype=bool); cov_s = np.empty(T1, dtype=bool)
    lo_p  = np.empty(T1); hi_p = np.empty(T1)
    lo_s  = np.empty(T1); hi_s = np.empty(T1)

    print("  Running prediction loop ...", end=" ", flush=True)
    for step in range(T1):
        t0  = T + step
        f   = float(test_preds[step])
        y_t = float(Y[t0])
        ss  = season(t0, S)

        arr_p  = np.array(buf_p)
        b_p    = line_search_beta(arr_p, alpha)
        lo_p[step] = f + empirical_quantile(arr_p, b_p)
        hi_p[step] = f + empirical_quantile(arr_p, 1 - alpha + b_p)
        cov_p[step] = bool(lo_p[step] <= y_t <= hi_p[step])

        arr_s  = np.array(bufs_s[ss]) if bufs_s[ss] else np.zeros(1)
        b_s    = line_search_beta(arr_s, alpha)
        lo_s[step] = f + empirical_quantile(arr_s, b_s)
        hi_s[step] = f + empirical_quantile(arr_s, 1 - alpha + b_s)
        cov_s[step] = bool(lo_s[step] <= y_t <= hi_s[step])

        if (step + 1) % s0 == 0:
            for delta in range(s0):
                j0 = T + step - 1 - delta
                if j0 >= p:
                    eps = get_res(j0)
                    buf_p.popleft(); buf_p.append(eps)
                    sj = season(j0, S)
                    if bufs_s[sj]: bufs_s[sj].popleft()
                    bufs_s[sj].append(eps)
    print("done")
    return (cov_p, lo_p, hi_p), (cov_s, lo_s, hi_s)


# ── 3. Results ────────────────────────────────────────────────────────

def season_summary(cov_p, lo_p, hi_p, cov_s, lo_s, hi_s, T):
    T1 = len(cov_p)
    rows = []
    for s_ in range(1, S + 1):
        mask = np.array([season(T + k, S) == s_ for k in range(T1)])
        if mask.any():
            rows.append({
                "season":     s_,
                "month":      MONTH_NAMES[s_ - 1],
                "n":          int(mask.sum()),
                "cov_pool":   float(cov_p[mask].mean()),
                "cov_strat":  float(cov_s[mask].mean()),
                "wid_pool":   float((hi_p - lo_p)[mask].mean()),
                "wid_strat":  float((hi_s - lo_s)[mask].mean()),
            })
    return pd.DataFrame(rows)


def latex_table(df, T, T1, alpha=ALPHA):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-season empirical coverage and average width for "
        r"IPCA food-at-home monthly inflation forecasting.",
        f"Training: {TRAIN_START}--{TRAIN_END} ($T={T}$); "
        f"test: {TEST_START}--{TEST_END} ($T_1={T1}$); $\\alpha={alpha}$.}}",
        r"\label{tab:empirical_food_coverage}",
        r"\begin{tabular}{l r cc cc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Coverage} & "
        r"\multicolumn{2}{c}{Width (\%\,p.p.)} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Month & $n$ & Pooled & EnbPI-S & Pooled & EnbPI-S \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cp, cs = row["cov_pool"], row["cov_strat"]
        wp, ws = row["wid_pool"], row["wid_strat"]
        cp_s = f"\\textbf{{{cp:.2f}}}" if abs(cp-(1-alpha)) <= abs(cs-(1-alpha)) else f"{cp:.2f}"
        cs_s = f"\\textbf{{{cs:.2f}}}" if abs(cs-(1-alpha)) <  abs(cp-(1-alpha)) else f"{cs:.2f}"
        wp_s = f"\\textit{{{wp:.3f}}}" if wp <= ws else f"{wp:.3f}"
        ws_s = f"\\textit{{{ws:.3f}}}" if ws <  wp else f"{ws:.3f}"
        lines.append(f"{row['month']} & {row['n']} & {cp_s} & {cs_s} "
                     f"& {wp_s} & {ws_s} \\\\")
    oc_p = df["cov_pool"].mean();  oc_s = df["cov_strat"].mean()
    ow_p = df["wid_pool"].mean();  ow_s = df["wid_strat"].mean()
    lines += [
        r"\midrule",
        f"\\textbf{{Overall}} & {int(df['n'].sum())} "
        f"& {oc_p:.2f} & {oc_s:.2f} & {ow_p:.3f} & {ow_s:.3f} \\\\",
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
    except ImportError:
        print("  matplotlib not available — skip figures.")
        return

    # Figure 1: per-season coverage bars
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(1, S + 1)
    w = 0.35
    ax.bar(x - w/2, df_season["cov_pool"],  w, label="Pooled",   color="tab:red",  alpha=0.75)
    ax.bar(x + w/2, df_season["cov_strat"], w, label="EnbPI-S",  color="tab:blue", alpha=0.75)
    ax.axhline(1 - ALPHA, color="k", ls="--", lw=0.9, label=f"Nominal ({100*(1-ALPHA):.0f}%)")
    ax.set_xticks(x); ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Month"); ax.set_ylabel("Empirical coverage")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Per-season coverage — IPCA food-at-home ({TEST_START}–{TEST_END})")
    fig.tight_layout()
    path = os.path.join(figures_dir, "fig_empirical_food_coverage.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")

    # Figure 2: interval widths by season (bar chart comparing methods)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(x - w/2, df_season["wid_pool"],  w, label="Pooled",  color="tab:red",  alpha=0.75)
    ax.bar(x + w/2, df_season["wid_strat"], w, label="EnbPI-S", color="tab:blue", alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.set_xlabel("Month"); ax.set_ylabel("Mean interval width (p.p.)")
    ax.legend(fontsize=8)
    ax.set_title(f"Interval width by season — IPCA food-at-home")
    fig.tight_layout()
    path = os.path.join(figures_dir, "fig_empirical_food_width.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")

    # Figure 3: time-series comparison — intervals vs realised values
    if lo_p is not None and lo_s is not None:
        import matplotlib.dates as mdates

        Y_test = Y[T: T + T1]
        dates_dt = pd.to_datetime(dates_test)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.subplots_adjust(hspace=0.08)

        configs = [
            (axes[0], lo_p, hi_p, cov_p, "Pooled EnbPI",
             "tab:red",  "#ffd5d5"),
            (axes[1], lo_s, hi_s, cov_s, "EnbPI-S",
             "tab:blue", "#d5e8ff"),
        ]

        for ax, lo, hi, cov, label, line_col, fill_col in configs:
            # Shaded interval
            ax.fill_between(dates_dt, lo, hi,
                            color=fill_col, alpha=0.9,
                            label=f"90% interval ({label})")
            # Interval boundaries (thin lines)
            ax.plot(dates_dt, lo, color=line_col, lw=0.6, alpha=0.7)
            ax.plot(dates_dt, hi, color=line_col, lw=0.6, alpha=0.7)
            # Realised series
            ax.plot(dates_dt, Y_test, color="black", lw=1.0,
                    label="Realised IPCA food")
            # Mark missed observations with red circles
            missed = ~np.array(cov, dtype=bool)
            if missed.any():
                ax.scatter(dates_dt[missed], Y_test[missed],
                           color="red", zorder=5, s=18,
                           label=f"Missed ({missed.sum()})")
            ax.set_ylabel("% (m/m)", fontsize=9)
            ax.legend(fontsize=7.5, loc="upper right", ncol=2)
            ax.set_title(label, fontsize=10, pad=3)
            ax.axhline(0, color="gray", lw=0.4, ls=":")

        # X-axis formatting on bottom panel
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[7]))
        fig.autofmt_xdate(rotation=0, ha="center")

        fig.suptitle(
            "Prediction intervals vs realised food-at-home IPCA\n"
            f"(test period: {TEST_START}–{TEST_END}, $\\alpha=0.10$)",
            fontsize=11)

        path = os.path.join(figures_dir, "fig_empirical_food_timeseries.pdf")
        fig.savefig(path, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {path}")


# ── 5. Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    offline = "--offline" in sys.argv

    proj   = os.path.dirname(_HERE)
    tables = os.path.join(proj, "tables");  os.makedirs(tables, exist_ok=True)
    figs   = os.path.join(proj, "figures"); os.makedirs(figs,   exist_ok=True)

    print(f"=== Empirical: {SERIES_NAME} ===\n")

    print("--- Data ---")
    df_raw = load_series(offline)
    Y, T, T1, dates = prepare_arrays(df_raw)

    print("\n--- Running methods ---")
    (cov_p, lo_p, hi_p), (cov_s, lo_s, hi_s) = run_both(Y, T, T1)

    dates_test = dates[dates >= pd.Timestamp(TEST_START)]
    df_s = season_summary(cov_p, lo_p, hi_p, cov_s, lo_s, hi_s, T)

    print("\n--- Results ---")
    print(f"{'Month':<6} {'Cov_P':>7} {'Cov_S':>7} {'Wid_P':>8} {'Wid_S':>8}")
    print("-" * 44)
    for _, r in df_s.iterrows():
        print(f"{r['month']:<6} {r['cov_pool']:>7.3f} {r['cov_strat']:>7.3f} "
              f"{r['wid_pool']:>8.3f} {r['wid_strat']:>8.3f}")
    print("-" * 44)
    print(f"{'Mean':<6} {df_s['cov_pool'].mean():>7.3f} {df_s['cov_strat'].mean():>7.3f} "
          f"{df_s['wid_pool'].mean():>8.3f} {df_s['wid_strat'].mean():>8.3f}")
    print(f"\nNominal: {1-ALPHA:.0%}")
    print(f"Width ratio (hi-vol / lo-vol):")
    hi_months = [0, 1, 6, 7]   # Jan Feb Jul Aug (0-indexed)
    lo_months = [2, 3, 4, 5]   # Mar Apr May Jun
    r_p_hi = df_s.iloc[hi_months]["wid_pool"].mean()
    r_s_hi = df_s.iloc[hi_months]["wid_strat"].mean()
    r_p_lo = df_s.iloc[lo_months]["wid_pool"].mean()
    r_s_lo = df_s.iloc[lo_months]["wid_strat"].mean()
    print(f"  Pooled  : {r_p_hi:.3f} / {r_p_lo:.3f} = {r_p_hi/r_p_lo:.2f}")
    print(f"  EnbPI-S : {r_s_hi:.3f} / {r_s_lo:.3f} = {r_s_hi/r_s_lo:.2f}")

    # Save season summary
    path_csv = os.path.join(tables, "empirical_food_coverage.csv")
    df_s.to_csv(path_csv, index=False); print(f"\n  Saved {path_csv}")

    # Save full interval series
    df_full = pd.DataFrame({
        "date":    dates_test[:T1],
        "y":       Y[T: T + T1],
        "lo_pool": lo_p, "hi_pool": hi_p, "covered_pool": cov_p,
        "lo_strat":lo_s, "hi_strat":hi_s, "covered_strat":cov_s,
    })
    path_full = os.path.join(tables, "empirical_food_intervals.csv")
    df_full.to_csv(path_full, index=False); print(f"  Saved {path_full}")

    tex = latex_table(df_s, T, T1)
    path_tex = os.path.join(tables, "tab_empirical_food_coverage.tex")
    with open(path_tex, "w", encoding="utf-8") as f: f.write(tex)
    print(f"  Saved {path_tex}")

    print("\n--- Figures ---")
    make_figures(df_s, Y, T, T1, dates_test, figs,
                 lo_p=lo_p, hi_p=hi_p, cov_p=cov_p,
                 lo_s=lo_s, hi_s=hi_s, cov_s=cov_s)
    print("\nDone.")
