"""
test_sarima_benchmark.py
Quick sanity check (N_REP=20) before committing to the full 200-rep run.
Compares SARIMA-Gaussian vs Pooled EnbPI vs EnbPI-S on E1 DGP, T=480.
"""
import sys, os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from simulation import (
    generate_series, season, build_features,
    fit_ensemble, batch_predict, loo_predictions,
    empirical_quantile, line_search_beta,
    B_BOOT, P_LAGS, ALPHA, S,
    HIGH_SEAS, SIGMA_HIGH, SIGMA_LOW,
)
from collections import deque
from scipy import stats

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not installed — run: pip install statsmodels")
    sys.exit(1)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── SARIMA Gaussian baseline ──────────────────────────────────────────

def run_sarima_gaussian(Y, T, T1, alpha=ALPHA, S=S):
    """
    Fit SARIMA(1,0,0)(1,0,0)_12 on Y[:T].
    Return Gaussian prediction intervals for Y[T:T+T1].
    Uses model-implied sigma (NOT conformal calibration).
    """
    y_tr = Y[:T]
    try:
        mod = SARIMAX(y_tr,
                      order=(1, 0, 0),
                      seasonal_order=(1, 0, 0, S),
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        res = mod.fit(disp=False, maxiter=300)

        # In-sample sigma estimate from residuals
        sigma_hat = float(np.std(res.resid, ddof=1))

        # Test point forecasts
        fc = res.forecast(steps=T1)

        # Gaussian symmetric interval
        z = stats.norm.ppf(1.0 - alpha / 2)   # = 1.645 for alpha=0.10
        lo = fc - z * sigma_hat
        hi = fc + z * sigma_hat
        covered = (lo <= Y[T:T+T1]) & (Y[T:T+T1] <= hi)
        return np.array(covered), lo, hi

    except Exception as e:
        # Fallback: return empty coverage on failure
        print(f"    SARIMA failed: {e}")
        return np.zeros(T1, dtype=bool), np.zeros(T1), np.zeros(T1)


# ── Pooled EnbPI  ─────────────────────────────────────────────────────

def run_pooled(Y, T, T1, alpha=ALPHA, B=B_BOOT, S=S, p=P_LAGS, seed=0):
    rng = np.random.default_rng(seed)
    all_idx = np.arange(p, T + T1)
    X_all = build_features(Y, all_idx, p)
    y_all = Y[all_idx]; n_tr = T - p
    models, Sb = fit_ensemble(X_all[:n_tr], y_all[:n_tr], B, "ridge", rng)
    pB = batch_predict(models, X_all); ep = pB.mean(axis=0)
    lr = y_all[:n_tr] - loo_predictions(pB[:,:n_tr], Sb, n_tr)
    fr = y_all - ep
    def res(j): k=j-p; return float(fr[k]) if 0<=k<len(fr) else 0.
    tp = ep[T-p:T-p+T1]
    buf = deque(lr.tolist())
    cov=np.empty(T1,dtype=bool); lo=np.empty(T1); hi=np.empty(T1)
    for step in range(T1):
        t0=T+step; f=float(tp[step]); y=float(Y[t0])
        a=np.array(buf); b=line_search_beta(a,alpha)
        lo[step]=f+empirical_quantile(a,b)
        hi[step]=f+empirical_quantile(a,1-alpha+b)
        cov[step]=bool(lo[step]<=y<=hi[step])
        j0=T+step-1
        if j0>=p: buf.popleft(); buf.append(res(j0))
    return cov,lo,hi


# ── EnbPI-S  ──────────────────────────────────────────────────────────

def run_strat(Y, T, T1, alpha=ALPHA, B=B_BOOT, S=S, p=P_LAGS, seed=0):
    rng = np.random.default_rng(seed)
    all_idx = np.arange(p, T + T1)
    X_all = build_features(Y, all_idx, p)
    y_all = Y[all_idx]; n_tr = T - p
    models, Sb = fit_ensemble(X_all[:n_tr], y_all[:n_tr], B, "ridge", rng)
    pB = batch_predict(models, X_all); ep = pB.mean(axis=0)
    lr = y_all[:n_tr] - loo_predictions(pB[:,:n_tr], Sb, n_tr)
    fr = y_all - ep
    def res(j): k=j-p; return float(fr[k]) if 0<=k<len(fr) else 0.
    tp = ep[T-p:T-p+T1]
    bufs={s:deque() for s in range(1,S+1)}
    for k,idx in enumerate(all_idx[:n_tr]):
        bufs[season(idx,S)].append(float(lr[k]))
    cov=np.empty(T1,dtype=bool); lo=np.empty(T1); hi=np.empty(T1)
    for step in range(T1):
        t0=T+step; ss=season(t0,S); f=float(tp[step]); y=float(Y[t0])
        arr=np.array(bufs[ss]) if bufs[ss] else np.zeros(1)
        b=line_search_beta(arr,alpha)
        lo[step]=f+empirical_quantile(arr,b)
        hi[step]=f+empirical_quantile(arr,1-alpha+b)
        cov[step]=bool(lo[step]<=y<=hi[step])
        j0=T+step-1
        if j0>=p:
            e=res(j0); sj=season(j0,S)
            if bufs[sj]: bufs[sj].popleft()
            bufs[sj].append(e)
    return cov,lo,hi


# ── Mini Monte Carlo ──────────────────────────────────────────────────

def season_cov(cov, T, T1, S=S):
    return {s: float(cov[[season(T+k,S)==s for k in range(T1)]].mean())
            for s in range(1,S+1)}

def run_mini_mc(n_rep=20, T=480, T1=120):
    results = {"sarima": [], "pooled": [], "strat": []}
    for rep in range(n_rep):
        print(f"  rep {rep+1}/{n_rep}", end="\r", flush=True)
        seed = rep * 1000
        rng  = np.random.default_rng(seed)
        Y    = generate_series(T + T1, SIGMA_HIGH, SIGMA_LOW, rng)

        c_sa, _, _ = run_sarima_gaussian(Y, T, T1)
        c_po, _, _ = run_pooled(Y, T, T1, seed=seed)
        c_st, _, _ = run_strat (Y, T, T1, seed=seed)

        results["sarima"].append(season_cov(c_sa, T, T1))
        results["pooled"].append(season_cov(c_po, T, T1))
        results["strat" ].append(season_cov(c_st, T, T1))
    print()
    return results


def summarise(results):
    out = {}
    for key, reps in results.items():
        out[key] = {s: np.mean([r[s] for r in reps]) for s in range(1,13)}
    return out


if __name__ == "__main__":
    T = 480
    print(f"Quick benchmark test: N_REP=20, T={T}, alpha={ALPHA}")
    print("Running (takes ~2 min)...\n")
    results = run_mini_mc(n_rep=20, T=T)
    summ    = summarise(results)

    print(f"\n{'Month':<6} {'SARIMA':>8} {'Pooled':>8} {'EnbPI-S':>8}  {'Type'}")
    print("-" * 50)
    for s in range(1, 13):
        tag = " *high-vol*" if s in HIGH_SEAS else ""
        print(f"{MONTH_NAMES[s-1]:<6} "
              f"{summ['sarima'][s]:>8.3f} "
              f"{summ['pooled'][s]:>8.3f} "
              f"{summ['strat'][s]:>8.3f}"
              f"{tag}")
    print("-" * 50)
    print(f"{'Mean':<6} "
          f"{np.mean(list(summ['sarima'].values())):>8.3f} "
          f"{np.mean(list(summ['pooled'].values())):>8.3f} "
          f"{np.mean(list(summ['strat'].values())):>8.3f}")
    print(f"\nNominal: {1-ALPHA:.2f}")

    # Coverage range (max - min across seasons)
    for key, label in [("sarima","SARIMA"),("pooled","Pooled"),("strat","EnbPI-S")]:
        vals = list(summ[key].values())
        print(f"{label} coverage range: {min(vals):.3f} – {max(vals):.3f} "
              f"(spread = {max(vals)-min(vals):.3f})")

    print("\n--- VERDICT ---")
    hi_vol = list(HIGH_SEAS)
    for key, label in [("sarima","SARIMA"), ("pooled","Pooled"), ("strat","EnbPI-S")]:
        hi = np.mean([summ[key][s] for s in hi_vol])
        lo = np.mean([summ[key][s] for s in range(1,13) if s not in HIGH_SEAS])
        print(f"{label:<10}: high-vol={hi:.3f}  low-vol={lo:.3f}  "
              f"gap={abs(hi-(1-ALPHA)):.3f}/{abs(lo-(1-ALPHA)):.3f}")
