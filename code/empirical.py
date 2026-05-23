"""
empirical.py
============
Empirical application of Pooled EnbPI and EnbPI-S to Brazilian monthly IPCA
(Índice de Preços ao Consumidor Amplo), sourced from the Banco Central do
Brasil (BCB) open-data API.

Data
----
  Series: BCB SGS 433 — IPCA monthly variation (%)
  Period used: 1994-08 to present
  URL: https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados
  Licence: public domain (Brazilian government open data)

Analysis
--------
  Train: 1995-01 to 2014-12  (T = 240 months, 20 years)
  Test:  2015-01 to 2024-12  (T1 = 120 months, 10 years)
  α = 0.10, S = 12

Usage
-----
  python empirical.py            # download fresh data and run analysis
  python empirical.py --offline  # use cached data/ipca_raw.csv if it exists

Outputs
-------
  tables/empirical_coverage.csv           per-season coverage and width
  tables/empirical_intervals.csv          full test-period interval series
  tables/tab_empirical_coverage.tex       LaTeX booktabs table
  figures/fig_empirical_coverage.pdf      per-season coverage bar chart
  figures/fig_empirical_intervals.pdf     interval time-series plot (subset)
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from collections import deque

warnings.filterwarnings("ignore")

# ── import core utilities from simulation.py ───────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from simulation import (
    season,
    build_features,
    fit_ensemble,
    batch_predict,
    loo_predictions,
    empirical_quantile,
    line_search_beta,
    compute_metrics,
    B_BOOT, P_LAGS, ALPHA, S,
)


# =====================================================================
# 0.  Constants
# =====================================================================

TRAIN_START = "1995-01"
TRAIN_END   = "2014-12"
TEST_START  = "2015-01"
TEST_END    = "2024-12"

BCB_URL   = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
DATA_CACHE = os.path.join(_HERE, "..", "data", "ipca_raw.csv")


# =====================================================================
# 1.  Data acquisition
# =====================================================================

def download_ipca() -> pd.DataFrame:
    """
    Download IPCA monthly variation (%) from the BCB open-data API.
    Returns a DataFrame with columns ['date', 'ipca'] indexed by date.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests  (needed for IPCA download)")

    params = {
        "formato": "json",
        "dataInicial": "01/01/1994",
        "dataFinal":   "31/12/2025",
    }
    print("  Downloading IPCA series from BCB API...", end=" ", flush=True)
    resp = requests.get(BCB_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    df["date"]  = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["ipca"]  = pd.to_numeric(df["valor"], errors="coerce")
    df = df[["date", "ipca"]].dropna().sort_values("date").reset_index(drop=True)
    print(f"done ({len(df)} observations, {df['date'].min().date()} – {df['date'].max().date()})")
    return df


def load_ipca(offline: bool = False) -> pd.DataFrame:
    """Load IPCA data: from cache if --offline, else download."""
    os.makedirs(os.path.dirname(DATA_CACHE), exist_ok=True)
    if offline and os.path.exists(DATA_CACHE):
        print(f"  Loading cached data from {DATA_CACHE}")
        df = pd.read_csv(DATA_CACHE, parse_dates=["date"])
        return df
    df = download_ipca()
    df.to_csv(DATA_CACHE, index=False)
    print(f"  Cached to {DATA_CACHE}")
    return df


def prepare_arrays(df: pd.DataFrame):
    """
    Slice training and test periods; return numpy arrays.
    Returns (Y_train_test, train_mask, test_mask, dates) all aligned.
    """
    df = df.set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp()

    # Extend if needed using available data
    avail_start = df.index.min()
    avail_end   = df.index.max()

    req_start = pd.Timestamp(TRAIN_START)
    req_end   = pd.Timestamp(TEST_END)

    if avail_start > req_start:
        raise ValueError(
            f"Data starts {avail_start.date()}, need {req_start.date()}")
    if avail_end < pd.Timestamp(TEST_END):
        actual_test_end = avail_end.strftime("%Y-%m")
        print(f"  WARNING: data ends {avail_end.date()}, "
              f"test will cover up to {actual_test_end}.")

    mask = (df.index >= req_start) & (df.index <= req_end)
    df_sub = df.loc[mask, "ipca"].copy()
    Y_full  = df_sub.values.astype(float)
    dates   = df_sub.index

    train_mask = dates <= pd.Timestamp(TRAIN_END)
    test_mask  = dates >= pd.Timestamp(TEST_START)

    T  = int(train_mask.sum())
    T1 = int(test_mask.sum())
    print(f"  Training: {dates[train_mask][0].date()} – "
          f"{dates[train_mask][-1].date()} (T={T})")
    print(f"  Test:     {dates[test_mask][0].date()} – "
          f"{dates[test_mask][-1].date()} (T1={T1})")
    return Y_full, T, T1, dates


# =====================================================================
# 2.  Run both methods on IPCA data
# =====================================================================

def run_empirical(Y: np.ndarray, T: int, T1: int,
                  alpha: float = ALPHA,
                  B: int = B_BOOT,
                  s0: int = 1,
                  S: int = S,
                  p: int = P_LAGS,
                  seed: int = 42):
    """
    Fit ensemble on Y[0:T], predict Y[T:T+T1] with both methods.
    Returns (results_pooled, results_strat) as (covered, lo, hi) tuples.
    """
    rng = np.random.default_rng(seed)

    all_idx   = np.arange(p, T + T1)
    X_all     = build_features(Y, all_idx, p)
    y_all     = Y[all_idx]
    n_tr      = T - p

    X_tr, y_tr = X_all[:n_tr], y_all[:n_tr]

    print(f"  Fitting bootstrap ensemble (B={B}) ...", end=" ", flush=True)
    models, Sb_sets = fit_ensemble(X_tr, y_tr, B, "ridge", rng)
    print("done")

    print("  Computing LOO residuals and predictions ...", end=" ", flush=True)
    all_preds_B = batch_predict(models, X_all)
    ens_preds   = all_preds_B.mean(axis=0)

    loo_p_tr = loo_predictions(all_preds_B[:, :n_tr], Sb_sets, n_tr)
    loo_res  = y_tr - loo_p_tr
    full_res = y_all - ens_preds

    def get_res(j0: int) -> float:
        k = j0 - p
        return float(full_res[k]) if 0 <= k < len(full_res) else 0.0

    test_slice = T - p
    test_preds = ens_preds[test_slice: test_slice + T1]
    print("done")

    # ---- Initialise buffers ----
    buf_pooled = deque(loo_res.tolist())
    bufs_strat = {s_: deque() for s_ in range(1, S + 1)}
    for k, idx in enumerate(all_idx[:n_tr]):
        bufs_strat[season(idx, S)].append(float(loo_res[k]))

    covered_p = np.empty(T1, dtype=bool)
    covered_s = np.empty(T1, dtype=bool)
    lo_p, hi_p = np.empty(T1), np.empty(T1)
    lo_s, hi_s = np.empty(T1), np.empty(T1)

    print("  Running prediction loop ...", end=" ", flush=True)
    for step in range(T1):
        t0     = T + step
        fhat   = float(test_preds[step])
        y_t    = float(Y[t0])
        s_star = season(t0, S)

        # Pooled
        arr_p  = np.array(buf_pooled)
        beta_p = line_search_beta(arr_p, alpha)
        lo_p[step] = fhat + empirical_quantile(arr_p, beta_p)
        hi_p[step] = fhat + empirical_quantile(arr_p, 1.0 - alpha + beta_p)
        covered_p[step] = bool(lo_p[step] <= y_t <= hi_p[step])

        # Stratified
        buf_s  = bufs_strat[s_star]
        arr_s  = np.array(buf_s) if buf_s else np.zeros(1)
        beta_s = line_search_beta(arr_s, alpha)
        lo_s[step] = fhat + empirical_quantile(arr_s, beta_s)
        hi_s[step] = fhat + empirical_quantile(arr_s, 1.0 - alpha + beta_s)
        covered_s[step] = bool(lo_s[step] <= y_t <= hi_s[step])

        # Update
        if (step + 1) % s0 == 0:
            for delta in range(s0):
                j0 = T + step - 1 - delta
                if j0 >= p:
                    eps = get_res(j0)
                    buf_pooled.popleft()
                    buf_pooled.append(eps)
                    s_j = season(j0, S)
                    if bufs_strat[s_j]:
                        bufs_strat[s_j].popleft()
                    bufs_strat[s_j].append(eps)

    print("done")
    return (covered_p, lo_p, hi_p), (covered_s, lo_s, hi_s)


# =====================================================================
# 3.  Results tables and LaTeX output
# =====================================================================

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


def build_results_df(covered_p, lo_p, hi_p,
                     covered_s, lo_s, hi_s,
                     dates_test: pd.DatetimeIndex):
    """
    Build a long-format DataFrame with per-observation results.
    """
    T1 = len(covered_p)
    rows = []
    for step in range(T1):
        s_ = season(step, S)   # season relative to test start
        rows.append({
            "date":       dates_test[step],
            "season":     s_,
            "month_name": MONTH_NAMES[s_ - 1],
            "covered_p":  bool(covered_p[step]),
            "lo_p":       float(lo_p[step]),
            "hi_p":       float(hi_p[step]),
            "width_p":    float(hi_p[step] - lo_p[step]),
            "covered_s":  bool(covered_s[step]),
            "lo_s":       float(lo_s[step]),
            "hi_s":       float(hi_s[step]),
            "width_s":    float(hi_s[step] - lo_s[step]),
        })
    return pd.DataFrame(rows)


def build_season_summary(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by season (month).
    """
    rows = []
    for s_ in range(1, S + 1):
        sub = df_results[df_results["season"] == s_]
        rows.append({
            "season":         s_,
            "month":          MONTH_NAMES[s_ - 1],
            "n_obs":          len(sub),
            "coverage_p":     sub["covered_p"].mean(),
            "width_p":        sub["width_p"].mean(),
            "coverage_s":     sub["covered_s"].mean(),
            "width_s":        sub["width_s"].mean(),
        })
    return pd.DataFrame(rows)


def latex_empirical_table(df_season: pd.DataFrame,
                          T_val: int, T1_val: int,
                          alpha: float = ALPHA) -> str:
    """
    Generate a LaTeX booktabs table for the empirical results.
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-season empirical coverage and average width for monthly IPCA forecasting.",
        f"Training: {TRAIN_START}--{TRAIN_END} ($T={T_val}$); "
        f"test: {TEST_START}--{TEST_END} ($T_1={T1_val}$); $\\alpha={alpha}$.}}",
        r"\label{tab:empirical_coverage}",
        r"\begin{tabular}{l r cc cc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{Width (\%\,p.p.)} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Month & $n$ & Pooled & EnbPI-S & Pooled & EnbPI-S \\",
        r"\midrule",
    ]

    for _, row in df_season.iterrows():
        cp = row["coverage_p"]
        cs = row["coverage_s"]
        wp = row["width_p"]
        ws = row["width_s"]
        n  = int(row["n_obs"])

        # Bold = closer to 0.90
        if abs(cs - (1 - alpha)) <= abs(cp - (1 - alpha)):
            cp_str = f"{cp:.2f}"
            cs_str = f"\\textbf{{{cs:.2f}}}"
        else:
            cp_str = f"\\textbf{{{cp:.2f}}}"
            cs_str = f"{cs:.2f}"

        # Italic = narrower
        if ws <= wp:
            wp_str = f"{wp:.3f}"
            ws_str = f"\\textit{{{ws:.3f}}}"
        else:
            wp_str = f"\\textit{{{wp:.3f}}}"
            ws_str = f"{ws:.3f}"

        lines.append(
            f"{row['month']} & {n} & {cp_str} & {cs_str} "
            f"& {wp_str} & {ws_str} \\\\"
        )

    # Overall row
    overall_cp = df_season["coverage_p"].mean()
    overall_cs = df_season["coverage_s"].mean()
    overall_wp = df_season["width_p"].mean()
    overall_ws = df_season["width_s"].mean()
    lines += [
        r"\midrule",
        f"\\textbf{{Overall}} & {int(df_season['n_obs'].sum())} "
        f"& {overall_cp:.2f} & {overall_cs:.2f} "
        f"& {overall_wp:.3f} & {overall_ws:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =====================================================================
# 4.  Figures
# =====================================================================

def make_empirical_figures(df_results: pd.DataFrame,
                           df_season: pd.DataFrame,
                           Y_full: np.ndarray,
                           T: int,
                           figures_dir: str):
    """Generate empirical result figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import matplotlib.dates as mdates
    except ImportError:
        print("  matplotlib not available — skipping figures.")
        return

    # ── Figure 1: per-season coverage ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x     = np.arange(1, S + 1)
    width = 0.35
    cov_p = df_season["coverage_p"].values
    cov_s = df_season["coverage_s"].values
    bars_p = ax.bar(x - width/2, cov_p, width, label="Pooled",
                    color="tab:red",  alpha=0.75)
    bars_s = ax.bar(x + width/2, cov_s, width, label="EnbPI-S",
                    color="tab:blue", alpha=0.75)
    ax.axhline(1 - ALPHA, color="black", linestyle="--", linewidth=0.9,
               label=f"Nominal ({1-ALPHA:.0%})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Empirical coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Per-season coverage — IPCA monthly forecasting")
    fig.tight_layout()
    path = os.path.join(figures_dir, "fig_empirical_coverage.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Figure 2: interval time series (last 3 years of test) ──────
    # Show actual IPCA + intervals for both methods, Jan 2022 – Dec 2024
    test_dates = df_results["date"].values
    plot_start = pd.Timestamp("2022-01-01")
    mask       = df_results["date"] >= plot_start
    sub        = df_results[mask]

    if len(sub) == 0:
        print("  Not enough test data for interval plot — skipping.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ipca_test = Y_full[T: T + len(df_results)]
    ipca_sub  = ipca_test[mask.values]
    x_dates   = sub["date"].values

    for ax, lo_col, hi_col, cov_col, title, color in [
        (axes[0], "lo_p", "hi_p", "covered_p", "Pooled EnbPI",  "tab:red"),
        (axes[1], "lo_s", "hi_s", "covered_s", "EnbPI-S",       "tab:blue"),
    ]:
        ax.fill_between(x_dates, sub[lo_col], sub[hi_col],
                        alpha=0.35, color=color, label="90% interval")
        ax.plot(x_dates, ipca_sub, "k-", linewidth=0.9, label="Actual IPCA")
        ax.set_ylabel("% (m/m)")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title)
        uncov_dates = sub.loc[~sub[cov_col], "date"]
        for d in uncov_dates:
            ax.axvline(d, color="red", linewidth=0.5, alpha=0.5)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    fig.autofmt_xdate(rotation=30)
    fig.suptitle("Prediction intervals — IPCA (Jan 2022 – Dec 2024)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(figures_dir, "fig_empirical_intervals.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# 5.  Main
# =====================================================================

if __name__ == "__main__":
    offline = "--offline" in sys.argv

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    tables_dir  = os.path.join(project_dir, "tables")
    figures_dir = os.path.join(project_dir, "figures")
    os.makedirs(tables_dir,  exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=== Empirical application: IPCA forecasting ===")

    # ── Data ──────────────────────────────────────────────────────
    print("\n--- Data acquisition ---")
    df_raw = load_ipca(offline=offline)
    Y_full, T, T1, dates = prepare_arrays(df_raw)

    # ── Analysis ──────────────────────────────────────────────────
    print("\n--- Running EnbPI and EnbPI-S ---")
    (cov_p, lo_p, hi_p), (cov_s, lo_s, hi_s) = run_empirical(
        Y_full, T, T1,
        alpha=ALPHA, B=B_BOOT, s0=1, S=S, p=P_LAGS, seed=42)

    # ── Results ───────────────────────────────────────────────────
    dates_test = dates[dates >= pd.Timestamp(TEST_START)]
    df_results = build_results_df(cov_p, lo_p, hi_p,
                                  cov_s, lo_s, hi_s, dates_test)
    df_season  = build_season_summary(df_results)

    print("\n--- Per-season results ---")
    print(f"{'Month':<6} {'Cov_P':>7} {'Cov_S':>7} {'Wid_P':>8} {'Wid_S':>8}")
    print("-" * 42)
    for _, row in df_season.iterrows():
        print(f"{row['month']:<6} {row['coverage_p']:>7.3f} {row['coverage_s']:>7.3f} "
              f"{row['width_p']:>8.3f} {row['width_s']:>8.3f}")
    print("-" * 42)
    print(f"{'Mean':<6} {df_season['coverage_p'].mean():>7.3f} "
          f"{df_season['coverage_s'].mean():>7.3f} "
          f"{df_season['width_p'].mean():>8.3f} "
          f"{df_season['width_s'].mean():>8.3f}")
    print(f"\nNominal coverage: {1-ALPHA:.2f}")

    # ── Save CSV ──────────────────────────────────────────────────
    path_season  = os.path.join(tables_dir, "empirical_coverage.csv")
    path_full    = os.path.join(tables_dir, "empirical_intervals.csv")
    df_season.to_csv(path_season,  index=False)
    df_results.to_csv(path_full,   index=False)
    print(f"\n  Saved {path_season}")
    print(f"  Saved {path_full}")

    # ── LaTeX table ───────────────────────────────────────────────
    tex = latex_empirical_table(df_season, T, T1)
    path_tex = os.path.join(tables_dir, "tab_empirical_coverage.tex")
    with open(path_tex, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  Saved {path_tex}")

    # ── Figures ───────────────────────────────────────────────────
    print("\n--- Generating figures ---")
    make_empirical_figures(df_results, df_season, Y_full, T, figures_dir)

    print("\nEmpirical analysis complete.")
