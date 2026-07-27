"""
simulation_baselines.py
=======================
Additional Monte Carlo experiments for the EnbPI-S paper: competing
calibration schemes that also target seasonal heteroskedasticity, and
the finite-sample (conformal) quantile correction.

All methods share the SAME bootstrap ensemble and the SAME LOO
residuals within each replication, so the only difference between them
is the calibration step.

Methods
-------
Pooled      single calibration buffer            (Xu and Xie, 2023)
EnbPI-N     pooled buffer of residuals divided by a season-specific
            scale estimate; interval rescaled by that scale
            (the normalised / locally-weighted competitor)
EnbPI-S     one calibration buffer per season     (this paper)
*-C         same method with the finite-sample conformal quantile
            correction: level p evaluated at ceil((n+1)p)/n on the
            upper side and floor((n+1)p)/n on the lower side

Experiments
-----------
X1 -- per-season coverage and width, T in {240, 480}
X2 -- overall coverage and width vs stratum size Ts in {5,10,20,30,50}

Run:
    python simulation_baselines.py            # n_rep = 200
    python simulation_baselines.py --quick    # n_rep = 30
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import deque

import simulation as sim   # DGP, ensemble and helpers (has a __main__ guard)

S       = sim.S
ALPHA   = sim.ALPHA
B_BOOT  = sim.B_BOOT
P_LAGS  = sim.P_LAGS
T1      = sim.T1
HIGH    = sim.HIGH_SEAS
MONTHS  = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

METHODS = ["Pooled", "EnbPI-N", "EnbPI-S",
           "Pooled-C", "EnbPI-N-C", "EnbPI-S-C"]

MAD_CONST = 0.6744897501960817     # Phi^{-1}(0.75)


# =====================================================================
# 1.  Fast quantiles on a pre-sorted array (matches np.quantile 'linear')
# =====================================================================

def q_sorted(a_sorted: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """Linear-interpolation quantiles of a sorted sample at levels ps."""
    n = len(a_sorted)
    if n == 1:
        return np.full(np.shape(ps), a_sorted[0], dtype=float)
    h = (n - 1) * np.clip(ps, 0.0, 1.0)
    lo = np.floor(h).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = h - lo
    return a_sorted[lo] + frac * (a_sorted[hi] - a_sorted[lo])


def interval_from_buffer(values: np.ndarray, alpha: float,
                         corrected: bool, n_grid: int = 200):
    """
    Minimum-width asymmetric interval offsets (lo, hi) obtained by the
    line search over beta in [0, alpha].

    corrected=False  -> plain empirical quantiles (Xu and Xie, 2023)
    corrected=True   -> finite-sample conformal levels:
                        lower level  floor((n+1)*beta)/n
                        upper level  ceil((n+1)*(1-alpha+beta))/n
    """
    a = np.sort(np.asarray(values, dtype=float))
    n = len(a)
    betas = np.linspace(0.0, alpha, n_grid + 1)
    lvl_lo = betas
    lvl_hi = 1.0 - alpha + betas
    if corrected:
        lvl_lo = np.floor((n + 1) * lvl_lo) / n
        lvl_hi = np.ceil((n + 1) * lvl_hi) / n
    q_lo = q_sorted(a, lvl_lo)
    q_hi = q_sorted(a, lvl_hi)
    k = int(np.argmin(q_hi - q_lo))
    return float(q_lo[k]), float(q_hi[k])


def mad_scale(values: np.ndarray) -> float:
    """Robust scale: MAD / Phi^{-1}(0.75), with guards."""
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return 1.0
    s = float(np.median(np.abs(a - np.median(a))) / MAD_CONST)
    if not np.isfinite(s) or s <= 1e-8:
        s = float(np.std(a))
    return s if (np.isfinite(s) and s > 1e-8) else 1.0


# =====================================================================
# 2.  One replication, all methods at once
# =====================================================================

def run_one_rep_multi(Y, T, T1, alpha, B, s0, S, model_type, p, rng):
    """
    Fit the ensemble once on Y[0:T] and produce intervals for every
    method in METHODS over the test window Y[T:T+T1].
    Returns dict method -> (covered, lo, hi).
    """
    tr_idx = np.arange(p, T)
    n_tr = len(tr_idx)
    X_tr = sim.build_features(Y, tr_idx, p)
    y_tr = Y[tr_idx]

    models, Sb_sets = sim.fit_ensemble(X_tr, y_tr, B, model_type, rng)

    all_idx = np.arange(p, T + T1)
    X_all = sim.build_features(Y, all_idx, p)
    y_all = Y[all_idx]
    all_preds_B = sim.batch_predict(models, X_all)
    ens_preds = all_preds_B.mean(axis=0)

    loo_p_tr = sim.loo_predictions(all_preds_B[:, :n_tr], Sb_sets, n_tr)
    loo_res = y_tr - loo_p_tr
    full_res = y_all - ens_preds

    def get_res(j0):
        k = j0 - p
        return float(full_res[k]) if 0 <= k < len(full_res) else 0.0

    test_preds = ens_preds[T - p: T - p + T1]

    # ---- buffers -----------------------------------------------------
    buf_pool = deque(loo_res.tolist())
    bufs_str = {s: deque() for s in range(1, S + 1)}
    for k, i in enumerate(tr_idx):
        bufs_str[sim.season(i, S)].append(float(loo_res[k]))

    # season scale estimates, frozen at the end of training
    sigma_hat = {s: mad_scale(np.array(bufs_str[s])) for s in range(1, S + 1)}
    buf_norm = deque(float(loo_res[k]) / sigma_hat[sim.season(i, S)]
                     for k, i in enumerate(tr_idx))

    out = {m: (np.empty(T1, dtype=bool), np.empty(T1), np.empty(T1))
           for m in METHODS}

    for step in range(T1):
        t0 = T + step
        fhat = float(test_preds[step])
        y_t = float(Y[t0])
        s_star = sim.season(t0, S)

        arr_p = np.array(buf_pool)
        arr_s = np.array(bufs_str[s_star]) if bufs_str[s_star] else np.zeros(1)
        arr_n = np.array(buf_norm)
        sc = sigma_hat[s_star]

        for name, arr, scale, corr in [
                ("Pooled",    arr_p, 1.0, False),
                ("EnbPI-N",   arr_n, sc,  False),
                ("EnbPI-S",   arr_s, 1.0, False),
                ("Pooled-C",  arr_p, 1.0, True),
                ("EnbPI-N-C", arr_n, sc,  True),
                ("EnbPI-S-C", arr_s, 1.0, True)]:
            q_lo, q_hi = interval_from_buffer(arr, alpha, corr)
            lo = fhat + scale * q_lo
            hi = fhat + scale * q_hi
            cov, lo_a, hi_a = out[name]
            lo_a[step] = lo
            hi_a[step] = hi
            cov[step] = bool(lo <= y_t <= hi)

        # ---- sliding update (same rule for every method) -------------
        if (step + 1) % s0 == 0:
            for delta in range(s0):
                j0 = T + step - 1 - delta
                if j0 >= p:
                    eps = get_res(j0)
                    s_j = sim.season(j0, S)
                    buf_pool.popleft()
                    buf_pool.append(eps)
                    buf_norm.popleft()
                    buf_norm.append(eps / sigma_hat[s_j])
                    if bufs_str[s_j]:
                        bufs_str[s_j].popleft()
                    bufs_str[s_j].append(eps)

    return out


# =====================================================================
# 3.  Monte Carlo driver
# =====================================================================

def monte_carlo_multi(n_rep, T, sigma_high, sigma_low,
                      seed_base, model_type="ridge", verbose=True):
    """Returns dict method -> list of per-replication metric dicts."""
    acc = {m: [] for m in METHODS}
    for rep in range(n_rep):
        rng = np.random.default_rng(seed_base + rep * 1000)
        Y = sim.generate_series(T + T1, sigma_high, sigma_low, rng)
        res = run_one_rep_multi(Y, T, T1, ALPHA, B_BOOT, 1, S,
                                model_type, P_LAGS, rng)
        for m in METHODS:
            cov, lo, hi = res[m]
            acc[m].append(sim.compute_metrics(cov, lo, hi, T, S))
        if verbose and (rep + 1) % 25 == 0:
            print(f"    rep {rep+1}/{n_rep}", flush=True)
    return acc


def season_table(acc, n_rep):
    """Per-season mean coverage/width plus Monte Carlo standard errors."""
    rows = []
    for m in METHODS:
        summ = sim.summarise(acc[m])
        for s in range(1, S + 1):
            cmean, cstd = summ["seasonal_coverage"][s]
            wmean, _ = summ["seasonal_width"][s]
            rows.append({"method": m, "season": s,
                         "coverage_mean": cmean,
                         "coverage_se": cstd / np.sqrt(n_rep),
                         "width_mean": wmean})
        rows.append({"method": m, "season": 0,
                     "coverage_mean": summ["overall_coverage_mean"],
                     "coverage_se": summ["overall_coverage_std"] / np.sqrt(n_rep),
                     "width_mean": summ["overall_width_mean"]})
    return pd.DataFrame(rows)


# =====================================================================
# 4.  Experiments
# =====================================================================

def generate_series_shape(n, rng, sigma_high=1.0, sigma_low=1.0,
                          skew_seasons=HIGH):
    """
    Same AR structure as the main DGP, but the seasonal difference is in
    the SHAPE of the innovation and not (only) in its scale.

    Innovations are standardised to mean 0 and variance 1 in every
    season: a standardised lognormal (strongly right-skewed) in
    `skew_seasons`, a standard normal elsewhere.  With
    sigma_high = sigma_low = 1 the seasonal scales are identical, so a
    calibration scheme that only rescales residuals has nothing to
    correct, while the season-specific CDFs still differ.
    """
    m = n + sim.BURN_IN
    z = rng.standard_normal(m)
    ln = np.exp(rng.standard_normal(m))
    ln = (ln - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)   # mean 0, var 1
    Y = np.zeros(m)
    for t in range(m):
        s = sim.season(t, S)
        eta = ln[t] if s in skew_seasons else z[t]
        sig = sigma_high if s in skew_seasons else sigma_low
        lag1 = Y[t - 1] if t >= 1 else 0.0
        lag12 = Y[t - 12] if t >= 12 else 0.0
        Y[t] = sim.PHI12 * lag12 + sim.PHI1 * lag1 + sig * eta
    return Y[sim.BURN_IN:]


def monte_carlo_shape(n_rep, T, seed_base, sigma_high=1.0):
    acc = {m: [] for m in METHODS}
    for rep in range(n_rep):
        rng = np.random.default_rng(seed_base + rep * 1000)
        Y = generate_series_shape(T + T1, rng, sigma_high=sigma_high)
        res = run_one_rep_multi(Y, T, T1, ALPHA, B_BOOT, 1, S,
                                "ridge", P_LAGS, rng)
        for m in METHODS:
            cov, lo, hi = res[m]
            acc[m].append(sim.compute_metrics(cov, lo, hi, T, S))
        if (rep + 1) % 25 == 0:
            print(f"    rep {rep+1}/{n_rep}", flush=True)
    return acc


def experiment_X3(n_rep, tables_dir):
    """Seasonality in distributional SHAPE rather than scale."""
    print("\n=== X3: shape heterogeneity (equal seasonal scales) ===")
    frames = []
    for T in [240, 480]:
        print(f"  T={T}")
        acc = monte_carlo_shape(n_rep, T, seed_base=6000)
        df = season_table(acc, n_rep)
        df.insert(0, "T", T)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(tables_dir, "X3_shape_heterogeneity.csv")
    out.to_csv(path, index=False)
    print(f"  Saved {path}")
    return out


def experiment_X1(n_rep, tables_dir):
    """Per-season coverage and width for all calibration schemes."""
    print("\n=== X1: calibration schemes, per season ===")
    frames = []
    for T in [240, 480]:
        print(f"  T={T}")
        acc = monte_carlo_multi(n_rep, T, sim.SIGMA_HIGH, sim.SIGMA_LOW,
                                seed_base=1000)
        df = season_table(acc, n_rep)
        df.insert(0, "T", T)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    path = os.path.join(tables_dir, "X1_baselines_by_season.csv")
    out.to_csv(path, index=False)
    print(f"  Saved {path}")
    return out


def experiment_X2(n_rep, tables_dir):
    """Overall coverage and width vs stratum size."""
    print("\n=== X2: calibration schemes vs stratum size ===")
    rows = []
    for Ts in [5, 10, 20, 30, 50]:
        T = Ts * S
        print(f"  Ts={Ts} (T={T})")
        acc = monte_carlo_multi(n_rep, T, sim.SIGMA_HIGH, sim.SIGMA_LOW,
                                seed_base=2000)
        for m in METHODS:
            summ = sim.summarise(acc[m])
            hv = [summ["seasonal_coverage"][s][0] for s in sorted(HIGH)]
            lv = [summ["seasonal_coverage"][s][0]
                  for s in range(1, S + 1) if s not in HIGH]
            allc = [summ["seasonal_coverage"][s][0] for s in range(1, S + 1)]
            rows.append({
                "Ts": Ts, "T": T, "method": m,
                "coverage_mean": summ["overall_coverage_mean"],
                "coverage_se": summ["overall_coverage_std"] / np.sqrt(n_rep),
                "width_mean": summ["overall_width_mean"],
                "cov_high": float(np.mean(hv)),
                "cov_low": float(np.mean(lv)),
                "spread": float(np.max(allc) - np.min(allc)),
                "min_season_cov": float(np.min(allc)),
            })
    out = pd.DataFrame(rows)
    path = os.path.join(tables_dir, "X2_baselines_stratum_size.csv")
    out.to_csv(path, index=False)
    print(f"  Saved {path}")
    return out


# =====================================================================
# 5.  Reporting helpers
# =====================================================================

def report(df_x1, df_x2):
    for T in sorted(df_x1["T"].unique()):
        sub = df_x1[df_x1["T"] == T]
        print(f"\n=== T = {T} ===")
        print(f"{'method':<11} {'overall':>8} {'high-vol':>9} {'low-vol':>8} "
              f"{'spread':>7} {'min':>6} {'w-high':>7} {'w-low':>7}")
        for m in METHODS:
            d = sub[sub["method"] == m]
            per = d[d["season"] > 0]
            hv = per[per["season"].isin(HIGH)]
            lv = per[~per["season"].isin(HIGH)]
            ov = d[d["season"] == 0]["coverage_mean"].values[0]
            print(f"{m:<11} {ov:>8.3f} {hv['coverage_mean'].mean():>9.3f} "
                  f"{lv['coverage_mean'].mean():>8.3f} "
                  f"{per['coverage_mean'].max()-per['coverage_mean'].min():>7.3f} "
                  f"{per['coverage_mean'].min():>6.3f} "
                  f"{hv['width_mean'].mean():>7.2f} "
                  f"{lv['width_mean'].mean():>7.2f}")

    print("\n=== X2: overall coverage by stratum size ===")
    print(f"{'Ts':>4} " + " ".join(f"{m:>10}" for m in METHODS))
    for Ts in sorted(df_x2["Ts"].unique()):
        sub = df_x2[df_x2["Ts"] == Ts]
        vals = [sub[sub["method"] == m]["coverage_mean"].values[0] for m in METHODS]
        print(f"{Ts:>4} " + " ".join(f"{v:>10.3f}" for v in vals))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n_rep = 30 if quick else 200

    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    tables_dir = os.path.join(proj, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    print(f"Baseline comparison -- n_rep={n_rep}, alpha={ALPHA}, "
          f"T1={T1}, B={B_BOOT}")
    df_x1 = experiment_X1(n_rep, tables_dir)
    df_x2 = experiment_X2(n_rep, tables_dir)
    df_x3 = experiment_X3(n_rep, tables_dir)
    report(df_x1, df_x2)
    print("\n=== X3: shape heterogeneity ===")
    for T in sorted(df_x3["T"].unique()):
        sub = df_x3[df_x3["T"] == T]
        print(f"\n--- T = {T} ---")
        print(f"{'method':<11} {'overall':>8} {'skew':>8} {'gauss':>8} "
              f"{'spread':>7} {'w-skew':>7} {'w-gauss':>8}")
        for m in METHODS:
            d = sub[sub["method"] == m]
            per = d[d["season"] > 0]
            hv = per[per["season"].isin(HIGH)]
            lv = per[~per["season"].isin(HIGH)]
            ov = d[d["season"] == 0]["coverage_mean"].values[0]
            print(f"{m:<11} {ov:>8.3f} {hv['coverage_mean'].mean():>8.3f} "
                  f"{lv['coverage_mean'].mean():>8.3f} "
                  f"{per['coverage_mean'].max()-per['coverage_mean'].min():>7.3f} "
                  f"{hv['width_mean'].mean():>7.2f} "
                  f"{lv['width_mean'].mean():>8.2f}")
    print("\nDone.")
