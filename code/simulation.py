"""
simulation.py
=============
Monte Carlo experiments for the EnbPI-S paper.

DGP (monthly, S=12):
    Y[t] = 0.6*Y[t-12] + 0.3*Y[t-1] + sigma_{s(t)} * eta[t]
    eta[t] ~ t_3 i.i.d.
    sigma_{s} = sigma_high  for s in {1,2,8,9}  (high-volatility months)
              = sigma_low   otherwise

Experiments
-----------
E1 — Per-season coverage vs training size T in {120, 240, 480}
E2 — Overall coverage vs stratum size Ts = T/S in {5,10,20,30,50}
E3 — Per-season coverage vs sigma_high in {1.0, 1.5, 2.0, 3.0}
E4 — Null case (sigma_s=1 for all s)
E5 — Predictor robustness: Ridge vs Random Forest

Run:
    python simulation.py              # full run (N_REP=200, ~30-50 min)
    python simulation.py --quick      # reduced run (N_REP=30, ~3 min)
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import t as t_dist
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# 0.  Global constants
# =====================================================================

S          = 12
ALPHA      = 0.10
B_BOOT     = 50        # bootstrap replicates
P_LAGS     = 13        # AR feature lags (covers lag-1 and lag-12)
T1         = 120       # test length (10 years)
N_REP_FULL = 200       # production replications
N_REP_FAST = 30        # quick-test replications
PHI1       = 0.3
PHI12      = 0.6
DF_T       = 3         # t-distribution d.f.
SIGMA_HIGH = 2.0
SIGMA_LOW  = 1.0
HIGH_SEAS  = frozenset({1, 2, 8, 9})
BURN_IN    = 60        # initial observations to discard


# =====================================================================
# 1.  Helper: season label (1-based)
# =====================================================================

def season(t0: int, S: int = S) -> int:
    """1-based season for 0-indexed time t0."""
    return (t0 % S) + 1


# =====================================================================
# 2.  DGP
# =====================================================================

def sigma_t(t0: int, sigma_high: float, sigma_low: float,
            S: int = S, high_seas=HIGH_SEAS) -> float:
    return sigma_high if season(t0, S) in high_seas else sigma_low


def generate_series(n: int, sigma_high: float, sigma_low: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Simulate n observations from the seasonal-heteroskedastic DGP.
    Returns Y[0]...Y[n-1].
    """
    eta = rng.standard_t(df=DF_T, size=n + BURN_IN)
    Y_full = np.zeros(n + BURN_IN)
    for t in range(n + BURN_IN):
        sig  = sigma_t(t, sigma_high, sigma_low)
        lag1  = Y_full[t - 1]  if t >= 1  else 0.0
        lag12 = Y_full[t - 12] if t >= 12 else 0.0
        Y_full[t] = PHI12 * lag12 + PHI1 * lag1 + sig * eta[t]
    return Y_full[BURN_IN:]   # discard burn-in


# =====================================================================
# 3.  AR feature matrix
# =====================================================================

def build_features(Y: np.ndarray, indices, p: int = P_LAGS) -> np.ndarray:
    """
    Build design matrix X with rows [Y[t-1],...,Y[t-p]] for each t in indices.
    Pads with 0 when t-k < 0.
    """
    idx = np.asarray(indices, dtype=int)
    n, d = len(idx), p
    X = np.zeros((n, d))
    for col, lag in enumerate(range(1, p + 1)):
        valid = idx - lag >= 0
        X[valid, col] = Y[idx[valid] - lag]
    return X


# =====================================================================
# 4.  Bootstrap ensemble
# =====================================================================

def fit_ensemble(X_tr: np.ndarray, y_tr: np.ndarray,
                 B: int, model_type: str,
                 rng: np.random.Generator):
    """
    Fit B bootstrap models.
    Returns (models, Sb_sets) where Sb_sets[b] is a Python set of
    bootstrap-sampled indices for model b.
    """
    n = len(y_tr)
    models, Sb_sets = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        Sb_sets.append(set(idx.tolist()))
        Xb, yb = X_tr[idx], y_tr[idx]
        if model_type == "ridge":
            m = Ridge(alpha=1.0)
        elif model_type == "rf":
            m = RandomForestRegressor(
                n_estimators=30, max_depth=6, n_jobs=1,
                random_state=int(rng.integers(0, 99999)))
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")
        m.fit(Xb, yb)
        models.append(m)
    return models, Sb_sets


def batch_predict(models, X: np.ndarray) -> np.ndarray:
    """Returns array of shape (B, n) with predictions from each model."""
    return np.array([m.predict(X) for m in models])


def loo_predictions(all_preds_B: np.ndarray, Sb_sets: list,
                    n_tr: int) -> np.ndarray:
    """
    Compute LOO ensemble predictions for all n_tr training observations.
    For obs i: average over models b with i not in Sb.
    Falls back to full ensemble if every model included i.
    all_preds_B has shape (B, n_tr).
    """
    loo_p = np.empty(n_tr)
    for i in range(n_tr):
        mask = np.array([i not in Sb for Sb in Sb_sets])
        if mask.any():
            loo_p[i] = all_preds_B[mask, i].mean()
        else:
            loo_p[i] = all_preds_B[:, i].mean()
    return loo_p


# =====================================================================
# 5.  Quantile utilities
# =====================================================================

def empirical_quantile(values: np.ndarray, p: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.quantile(values, np.clip(p, 0.0, 1.0), method="linear"))


def line_search_beta(values: np.ndarray, alpha: float,
                     n_grid: int = 200) -> float:
    """
    beta* = argmin_{beta in [0,alpha]} [Q(1-alpha+beta) - Q(beta)]
    minimises interval width while keeping coverage level fixed at 1-alpha.
    """
    betas = np.linspace(0.0, alpha, n_grid + 1)
    widths = np.array([
        empirical_quantile(values, 1.0 - alpha + b) -
        empirical_quantile(values, b)
        for b in betas
    ])
    return float(betas[int(np.argmin(widths))])


# =====================================================================
# 6.  Core: run one MC replication (both methods simultaneously)
# =====================================================================

def run_one_rep(Y: np.ndarray, T: int, T1: int,
                alpha: float, B: int, s0: int, S: int,
                model_type: str, p: int,
                rng: np.random.Generator):
    """
    Fit bootstrap ensemble on Y[0:T], then predict Y[T:T+T1].
    Returns two tuples (covered_p, lo_p, hi_p) and (covered_s, lo_s, hi_s)
    for pooled EnbPI and stratified EnbPI-S respectively.
    """
    # ------------------------------------------------------------------
    # Training design matrix: indices p, p+1, ..., T-1  (need p lags)
    # ------------------------------------------------------------------
    tr_idx = np.arange(p, T)
    n_tr   = len(tr_idx)
    X_tr   = build_features(Y, tr_idx, p)
    y_tr   = Y[tr_idx]

    # Fit bootstrap models
    models, Sb_sets = fit_ensemble(X_tr, y_tr, B, model_type, rng)

    # ------------------------------------------------------------------
    # Batch predictions for ALL relevant observations in one call per model
    # Needed:
    #   - n_tr training obs for LOO residuals
    #   - T1 test obs for prediction
    #   - observations T-1, T, ..., T+T1-2 for sliding updates
    #     (all within range [p, T+T1-1])
    # Build one contiguous block: all_idx = p, ..., T+T1-1
    # ------------------------------------------------------------------
    all_idx = np.arange(p, T + T1)
    X_all   = build_features(Y, all_idx, p)
    y_all   = Y[all_idx]

    all_preds_B = batch_predict(models, X_all)      # shape (B, n_all)
    ens_preds   = all_preds_B.mean(axis=0)           # ensemble mean

    # LOO residuals: training slice (first n_tr rows of all_idx)
    loo_p_tr = loo_predictions(all_preds_B[:, :n_tr], Sb_sets, n_tr)
    loo_res  = y_tr - loo_p_tr                       # length n_tr

    # Full-ensemble residuals for the update buffer (used after predictions)
    full_res = y_all - ens_preds                     # length n_all
    # Map: all_idx[k] = p+k  →  original 0-indexed obs (p+k) has full_res[k]

    def get_res(j0: int) -> float:
        k = j0 - p
        return float(full_res[k]) if 0 <= k < len(full_res) else 0.0

    # Test ensemble predictions (slice starting at T, length T1)
    test_slice_start = T - p          # index into all_idx where test starts
    test_preds = ens_preds[test_slice_start: test_slice_start + T1]

    # ------------------------------------------------------------------
    # Initialise buffers
    # ------------------------------------------------------------------
    # Pooled buffer: all n_tr LOO residuals in temporal order
    buf_pooled = deque(loo_res.tolist())

    # Stratified buffers: season-separated, in temporal order
    bufs_strat = {s: deque() for s in range(1, S + 1)}
    for k, i in enumerate(tr_idx):
        bufs_strat[season(i, S)].append(float(loo_res[k]))

    # ------------------------------------------------------------------
    # Prediction + buffer-update loop
    # ------------------------------------------------------------------
    covered_p = np.empty(T1, dtype=bool)
    covered_s = np.empty(T1, dtype=bool)
    lo_p = np.empty(T1)
    hi_p = np.empty(T1)
    lo_s = np.empty(T1)
    hi_s = np.empty(T1)

    for step in range(T1):
        t0   = T + step           # 0-indexed observation being predicted
        fhat = float(test_preds[step])
        y_t  = float(Y[t0])

        # ---- Pooled interval ----
        arr_p  = np.array(buf_pooled)
        beta_p = line_search_beta(arr_p, alpha)
        lo_p[step] = fhat + empirical_quantile(arr_p, beta_p)
        hi_p[step] = fhat + empirical_quantile(arr_p, 1.0 - alpha + beta_p)
        covered_p[step] = bool(lo_p[step] <= y_t <= hi_p[step])

        # ---- Stratified interval ----
        s_star = season(t0, S)
        buf_s  = bufs_strat[s_star]
        arr_s  = np.array(buf_s) if buf_s else np.zeros(1)
        beta_s = line_search_beta(arr_s, alpha)
        lo_s[step] = fhat + empirical_quantile(arr_s, beta_s)
        hi_s[step] = fhat + empirical_quantile(arr_s, 1.0 - alpha + beta_s)
        covered_s[step] = bool(lo_s[step] <= y_t <= hi_s[step])

        # ---- Sliding buffer update ----
        # At 1-indexed time T+step+1, update with j0 = T+step-1,...,T+step-s0
        if (step + 1) % s0 == 0:
            for delta in range(s0):
                j0 = T + step - 1 - delta
                if j0 >= p:
                    eps = get_res(j0)
                    # Pooled
                    buf_pooled.popleft()
                    buf_pooled.append(eps)
                    # Stratified
                    s_j = season(j0, S)
                    if bufs_strat[s_j]:
                        bufs_strat[s_j].popleft()
                    bufs_strat[s_j].append(eps)

    return (covered_p, lo_p, hi_p), (covered_s, lo_s, hi_s)


# =====================================================================
# 7.  Metrics
# =====================================================================

def compute_metrics(covered: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                    T_start: int, S: int = S):
    """
    Compute per-season and overall coverage and width.
    T_start: 0-indexed start of the test period in Y (used for season labels).
    """
    T1    = len(covered)
    width = hi - lo

    overall_cov = float(covered.mean())
    overall_wid = float(width.mean())

    seas_cov, seas_wid = {}, {}
    for s in range(1, S + 1):
        mask = np.array([season(T_start + k, S) == s for k in range(T1)])
        if mask.any():
            seas_cov[s] = float(covered[mask].mean())
            seas_wid[s] = float(width[mask].mean())
        else:
            seas_cov[s] = float("nan")
            seas_wid[s] = float("nan")

    return {
        "overall_coverage": overall_cov,
        "overall_width":    overall_wid,
        "seasonal_coverage": seas_cov,
        "seasonal_width":    seas_wid,
    }


# =====================================================================
# 8.  Monte Carlo wrapper
# =====================================================================

def monte_carlo(n_rep: int, T: int, T1: int,
                alpha: float, sigma_high: float, sigma_low: float,
                model_type: str = "ridge", B: int = B_BOOT,
                s0: int = 1, S: int = S, p: int = P_LAGS,
                seed_base: int = 0, verbose: bool = False):
    """
    Run n_rep replications. Returns (list_pooled_metrics, list_strat_metrics).
    """
    pool_metrics, strat_metrics = [], []
    for rep in range(n_rep):
        rng  = np.random.default_rng(seed_base + rep * 1000)
        Y    = generate_series(T + T1, sigma_high, sigma_low, rng)
        res_p, res_s = run_one_rep(Y, T, T1, alpha, B, s0, S,
                                   model_type, p, rng)
        m_p = compute_metrics(res_p[0], res_p[1], res_p[2], T, S)
        m_s = compute_metrics(res_s[0], res_s[1], res_s[2], T, S)
        pool_metrics.append(m_p)
        strat_metrics.append(m_s)
        if verbose and (rep + 1) % 10 == 0:
            print(f"    rep {rep+1}/{n_rep}", flush=True)
    return pool_metrics, strat_metrics


def summarise(results: list, S: int = S):
    """Aggregate MC results into mean ± std dicts."""
    seas_cov_all = {s: [] for s in range(1, S + 1)}
    seas_wid_all = {s: [] for s in range(1, S + 1)}
    overall_cov_all, overall_wid_all = [], []

    for r in results:
        overall_cov_all.append(r["overall_coverage"])
        overall_wid_all.append(r["overall_width"])
        for s in range(1, S + 1):
            v_c = r["seasonal_coverage"].get(s, float("nan"))
            v_w = r["seasonal_width"].get(s, float("nan"))
            if not np.isnan(v_c):
                seas_cov_all[s].append(v_c)
            if not np.isnan(v_w):
                seas_wid_all[s].append(v_w)

    def safe_stats(lst):
        a = np.asarray(lst)
        return float(a.mean()) if len(a) else float("nan"), \
               float(a.std())  if len(a) else float("nan")

    seas_cov = {s: safe_stats(seas_cov_all[s]) for s in range(1, S + 1)}
    seas_wid = {s: safe_stats(seas_wid_all[s]) for s in range(1, S + 1)}

    return {
        "overall_coverage_mean": float(np.mean(overall_cov_all)),
        "overall_coverage_std":  float(np.std(overall_cov_all)),
        "overall_width_mean":    float(np.mean(overall_wid_all)),
        "overall_width_std":     float(np.std(overall_wid_all)),
        "seasonal_coverage":     seas_cov,
        "seasonal_width":        seas_wid,
    }


# =====================================================================
# 9.  Experiments
# =====================================================================

def experiment_E1(n_rep, tables_dir):
    """Per-season coverage for T in {120, 240, 480}."""
    print("\n=== E1: Per-season coverage vs training size ===")
    rows = []
    for T in [120, 240, 480]:
        print(f"  T={T} ...", end=" ", flush=True)
        pm, sm = monte_carlo(n_rep, T, T1, ALPHA,
                             SIGMA_HIGH, SIGMA_LOW,
                             seed_base=1000, verbose=False)
        p_sum = summarise(pm)
        s_sum = summarise(sm)
        for seas in range(1, S + 1):
            for method, summ in [("Pooled", p_sum), ("EnbPI-S", s_sum)]:
                rows.append({
                    "T":             T,
                    "season":        seas,
                    "method":        method,
                    "coverage_mean": summ["seasonal_coverage"][seas][0],
                    "coverage_std":  summ["seasonal_coverage"][seas][1],
                    "width_mean":    summ["seasonal_width"][seas][0],
                })
        print("done")

    df = pd.DataFrame(rows)
    path = os.path.join(tables_dir, "E1_coverage_by_season.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")
    return df


def experiment_E2(n_rep, tables_dir):
    """Overall coverage vs stratum size Ts = T/S."""
    print("\n=== E2: Overall coverage vs stratum size ===")
    rows = []
    for Ts in [5, 10, 20, 30, 50]:
        T = Ts * S
        print(f"  Ts={Ts} (T={T}) ...", end=" ", flush=True)
        pm, sm = monte_carlo(n_rep, T, T1, ALPHA,
                             SIGMA_HIGH, SIGMA_LOW,
                             seed_base=2000, verbose=False)
        p_sum = summarise(pm)
        s_sum = summarise(sm)
        for method, summ in [("Pooled", p_sum), ("EnbPI-S", s_sum)]:
            rows.append({
                "Ts":              Ts,
                "T":               T,
                "method":          method,
                "coverage_mean":   summ["overall_coverage_mean"],
                "coverage_std":    summ["overall_coverage_std"],
                "width_mean":      summ["overall_width_mean"],
                "width_std":       summ["overall_width_std"],
            })
        print("done")

    df = pd.DataFrame(rows)
    path = os.path.join(tables_dir, "E2_stratum_size.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")
    return df


def experiment_E3(n_rep, tables_dir):
    """Per-season coverage vs heteroskedasticity intensity sigma_high."""
    print("\n=== E3: Per-season coverage vs sigma_high ===")
    T = 240
    rows = []
    for sh in [1.0, 1.5, 2.0, 3.0]:
        print(f"  sigma_high={sh} ...", end=" ", flush=True)
        pm, sm = monte_carlo(n_rep, T, T1, ALPHA,
                             sh, SIGMA_LOW,
                             seed_base=3000, verbose=False)
        p_sum = summarise(pm)
        s_sum = summarise(sm)
        for seas in range(1, S + 1):
            for method, summ in [("Pooled", p_sum), ("EnbPI-S", s_sum)]:
                rows.append({
                    "sigma_high":    sh,
                    "season":        seas,
                    "method":        method,
                    "coverage_mean": summ["seasonal_coverage"][seas][0],
                    "coverage_std":  summ["seasonal_coverage"][seas][1],
                    "width_mean":    summ["seasonal_width"][seas][0],
                })
        print("done")

    df = pd.DataFrame(rows)
    path = os.path.join(tables_dir, "E3_heteroskedasticity.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")
    return df


def experiment_E4(n_rep, tables_dir):
    """Null case: homoskedastic (sigma_s = 1 for all s)."""
    print("\n=== E4: Null case (homoskedastic) ===")
    T = 240
    rows = []
    print("  sigma_high=sigma_low=1.0 ...", end=" ", flush=True)
    pm, sm = monte_carlo(n_rep, T, T1, ALPHA,
                         sigma_high=1.0, sigma_low=1.0,
                         seed_base=4000, verbose=False)
    p_sum = summarise(pm)
    s_sum = summarise(sm)
    for seas in range(1, S + 1):
        for method, summ in [("Pooled", p_sum), ("EnbPI-S", s_sum)]:
            rows.append({
                "season":        seas,
                "method":        method,
                "coverage_mean": summ["seasonal_coverage"][seas][0],
                "coverage_std":  summ["seasonal_coverage"][seas][1],
                "width_mean":    summ["seasonal_width"][seas][0],
            })
    print("done")

    # Also record overall coverage (should be ~0.90 for both)
    print(f"  Overall coverage: Pooled={p_sum['overall_coverage_mean']:.3f}, "
          f"EnbPI-S={s_sum['overall_coverage_mean']:.3f}")

    df = pd.DataFrame(rows)
    path = os.path.join(tables_dir, "E4_null_case.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")
    return df


def experiment_E5(n_rep, tables_dir):
    """Predictor robustness: Ridge vs Random Forest."""
    print("\n=== E5: Predictor robustness ===")
    T = 240
    rows = []
    for model_type in ["ridge", "rf"]:
        n_rep_use = n_rep if model_type == "ridge" else max(n_rep // 2, 20)
        print(f"  model={model_type} (n_rep={n_rep_use}) ...", end=" ", flush=True)
        pm, sm = monte_carlo(n_rep_use, T, T1, ALPHA,
                             SIGMA_HIGH, SIGMA_LOW,
                             model_type=model_type,
                             seed_base=5000, verbose=False)
        p_sum = summarise(pm)
        s_sum = summarise(sm)
        for seas in range(1, S + 1):
            for method, summ in [("Pooled", p_sum), ("EnbPI-S", s_sum)]:
                rows.append({
                    "model":         model_type,
                    "season":        seas,
                    "method":        method,
                    "coverage_mean": summ["seasonal_coverage"][seas][0],
                    "coverage_std":  summ["seasonal_coverage"][seas][1],
                    "width_mean":    summ["seasonal_width"][seas][0],
                })
        print("done")

    df = pd.DataFrame(rows)
    path = os.path.join(tables_dir, "E5_predictor_robustness.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")
    return df


# =====================================================================
# 10.  LaTeX table helpers
# =====================================================================

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


def latex_E1_table(df: pd.DataFrame, T_val: int = 240) -> str:
    """
    Produce LaTeX booktabs table: seasons as rows, Pooled vs EnbPI-S columns.
    Coverage and width for T=T_val.
    """
    sub = df[df["T"] == T_val].copy()
    pivot_cov = sub.pivot(index="season", columns="method",
                          values="coverage_mean")
    pivot_wid = sub.pivot(index="season", columns="method",
                          values="width_mean")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Empirical coverage (90\% nominal) and average width by season.",
        f"$T={T_val}$, $\\alpha=0.10$, $S=12$, $n_{{\\mathrm{{rep}}}}={N_REP_FULL}$.}}",
        r"\label{tab:e1_coverage_T" + str(T_val) + "}",
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{Width} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Month & Pooled & EnbPI-S & Pooled & EnbPI-S \\",
        r"\midrule",
    ]

    for s in range(1, S + 1):
        cov_p = pivot_cov.loc[s, "Pooled"]
        cov_s = pivot_cov.loc[s, "EnbPI-S"]
        wid_p = pivot_wid.loc[s, "Pooled"]
        wid_s = pivot_wid.loc[s, "EnbPI-S"]

        # Bold the better coverage (closer to 0.90)
        if abs(cov_s - (1 - ALPHA)) < abs(cov_p - (1 - ALPHA)):
            cov_p_str = f"{cov_p:.3f}"
            cov_s_str = f"\\textbf{{{cov_s:.3f}}}"
        else:
            cov_p_str = f"\\textbf{{{cov_p:.3f}}}"
            cov_s_str = f"{cov_s:.3f}"

        # Italicise narrower width
        if wid_s < wid_p:
            wid_p_str = f"{wid_p:.3f}"
            wid_s_str = f"\\textit{{{wid_s:.3f}}}"
        else:
            wid_p_str = f"\\textit{{{wid_p:.3f}}}"
            wid_s_str = f"{wid_s:.3f}"

        high = "*" if s in HIGH_SEAS else ""
        line = (f"{MONTH_NAMES[s-1]}{high} & {cov_p_str} & {cov_s_str} "
                f"& {wid_p_str} & {wid_s_str} \\\\")
        lines.append(line)

    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{l}{\footnotesize * High-volatility months "
        r"($\sigma_{\mathrm{high}}=2$).}\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_E2_table(df: pd.DataFrame) -> str:
    """Produce LaTeX table for E2: overall coverage vs Ts."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Overall empirical coverage and average width vs stratum size $T_s = T/S$.",
        f"$\\alpha=0.10$, $S=12$, $n_{{\\mathrm{{rep}}}}={N_REP_FULL}$.}}",
        r"\label{tab:e2_stratum_size}",
        r"\begin{tabular}{r r cc cc}",
        r"\toprule",
        r"$T_s$ & $T$ & \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{Width} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"& & Pooled & EnbPI-S & Pooled & EnbPI-S \\",
        r"\midrule",
    ]

    for Ts in df["Ts"].unique():
        sub = df[df["Ts"] == Ts]
        T_val = int(sub["T"].iloc[0])
        row_p = sub[sub["method"] == "Pooled"].iloc[0]
        row_s = sub[sub["method"] == "EnbPI-S"].iloc[0]

        line = (f"{Ts} & {T_val} "
                f"& {row_p['coverage_mean']:.3f} & {row_s['coverage_mean']:.3f} "
                f"& {row_p['width_mean']:.3f} & {row_s['width_mean']:.3f} \\\\")
        lines.append(line)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_latex_tables(df_E1, df_E2, tables_dir):
    for T_val in [120, 240, 480]:
        tex = latex_E1_table(df_E1, T_val)
        path = os.path.join(tables_dir, f"tab_E1_T{T_val}.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"  Saved {path}")

    tex = latex_E2_table(df_E2)
    path = os.path.join(tables_dir, "tab_E2.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  Saved {path}")


# =====================================================================
# 11.  Optional plots
# =====================================================================

def make_plots(df_E1, df_E3, figures_dir):
    """Generate coverage plots. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  matplotlib not available — skipping plots.")
        return

    # ---- Figure 1: Per-season coverage, T=240, E1 ----
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sub = df_E1[df_E1["T"] == 240]
    seas = list(range(1, 13))
    for method, ls, color in [("Pooled", "--", "tab:red"),
                               ("EnbPI-S", "-",  "tab:blue")]:
        cov = [sub[(sub["method"] == method) &
                   (sub["season"] == s)]["coverage_mean"].values[0]
               for s in seas]
        ax.plot(seas, cov, linestyle=ls, color=color, marker="o",
                markersize=4, label=method)

    ax.axhline(1 - ALPHA, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Season (month)")
    ax.set_ylabel("Coverage probability")
    ax.set_xticks(seas)
    ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend()
    ax.set_title(r"Per-season coverage ($T=240$, $\alpha=0.10$)")

    # Shade high-volatility seasons
    for s in HIGH_SEAS:
        ax.axvspan(s - 0.5, s + 0.5, alpha=0.10, color="gray")

    fig.tight_layout()
    path = os.path.join(figures_dir, "fig_E1_coverage_T240.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ---- Figure 2: Coverage gap vs sigma_high (high-vol seasons only) ----
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sig_vals = sorted(df_E3["sigma_high"].unique())
    for method, ls, color in [("Pooled", "--", "tab:red"),
                               ("EnbPI-S", "-",  "tab:blue")]:
        gaps = []
        for sh in sig_vals:
            sub = df_E3[(df_E3["sigma_high"] == sh) &
                        (df_E3["method"] == method) &
                        (df_E3["season"].isin(HIGH_SEAS))]
            mean_cov = sub["coverage_mean"].mean()
            gaps.append(abs(mean_cov - (1 - ALPHA)))
        ax.plot(sig_vals, gaps, linestyle=ls, color=color,
                marker="s", markersize=5, label=method)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"$\sigma_{\mathrm{high}}$")
    ax.set_ylabel(r"|Coverage $-$ $(1-\alpha)$|")
    ax.legend()
    ax.set_title("Coverage gap in high-volatility seasons vs intensity")
    fig.tight_layout()
    path = os.path.join(figures_dir, "fig_E3_coverage_gap.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# 12.  Main
# =====================================================================

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_rep = N_REP_FAST if quick else N_REP_FULL

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    tables_dir  = os.path.join(project_dir, "tables")
    figures_dir = os.path.join(project_dir, "figures")
    os.makedirs(tables_dir,  exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"EnbPI-S simulation — n_rep={n_rep}, alpha={ALPHA}, T1={T1}, B={B_BOOT}")
    if quick:
        print("  [QUICK MODE: reduced replications]")

    df_E1 = experiment_E1(n_rep, tables_dir)
    df_E2 = experiment_E2(n_rep, tables_dir)
    df_E3 = experiment_E3(n_rep, tables_dir)
    df_E4 = experiment_E4(n_rep, tables_dir)
    df_E5 = experiment_E5(n_rep, tables_dir)

    print("\n--- Generating LaTeX tables ---")
    save_latex_tables(df_E1, df_E2, tables_dir)

    print("\n--- Generating figures ---")
    make_plots(df_E1, df_E3, figures_dir)

    # ---- Summary printout ----
    print("\n====== SUMMARY (T=240, sigma_high=2.0) ======")
    sub240 = df_E1[df_E1["T"] == 240]
    print(f"{'Month':<8}  {'Pooled':>8}  {'EnbPI-S':>8}")
    print("-" * 30)
    for s in range(1, S + 1):
        cov_p = sub240[(sub240["method"] == "Pooled") &
                       (sub240["season"] == s)]["coverage_mean"].values[0]
        cov_s = sub240[(sub240["method"] == "EnbPI-S") &
                       (sub240["season"] == s)]["coverage_mean"].values[0]
        marker = " *" if s in HIGH_SEAS else ""
        print(f"{MONTH_NAMES[s-1]:<8}  {cov_p:>8.3f}  {cov_s:>8.3f}{marker}")
    print("-" * 30)
    overall_p = sub240[sub240["method"] == "Pooled"]["coverage_mean"].mean()
    overall_s = sub240[sub240["method"] == "EnbPI-S"]["coverage_mean"].mean()
    print(f"{'Mean':<8}  {overall_p:>8.3f}  {overall_s:>8.3f}")
    print(f"\nNominal coverage: {1 - ALPHA:.2f}")
    print("* = high-volatility seasons (sigma_high=2.0)")
    print("\nAll experiments complete.")
