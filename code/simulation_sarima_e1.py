"""
simulation_sarima_e1.py
Experiment E1 with three methods: SARIMA-Gaussian, Pooled EnbPI, EnbPI-S.
N_REP=200, T in {120, 240, 480}, T1=120, alpha=0.10.

Run: python code/simulation_sarima_e1.py
Output: tables/E1_three_methods.csv  +  tables/tab_E1_three_methods_T480.tex
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
from scipy import stats as scipy_stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
N_REP = 200
T1    = 120


# ── three methods ─────────────────────────────────────────────────────

def sarima_gaussian(Y, T, T1, alpha=ALPHA, S=S):
    y_tr = Y[:T]
    try:
        mod = SARIMAX(y_tr, order=(1,0,0), seasonal_order=(1,0,0,S),
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False, maxiter=300)
        sig = float(np.std(res.resid, ddof=1))
        fc  = res.forecast(steps=T1)
        z   = scipy_stats.norm.ppf(1.0 - alpha / 2)
        lo, hi = fc - z*sig, fc + z*sig
        cov = (lo <= Y[T:T+T1]) & (Y[T:T+T1] <= hi)
        return np.array(cov), lo, hi
    except Exception:
        return np.zeros(T1, dtype=bool), np.zeros(T1), np.zeros(T1)


def pooled_enbpi(Y, T, T1, alpha=ALPHA, B=B_BOOT, S=S, p=P_LAGS, seed=0):
    rng = np.random.default_rng(seed)
    ai  = np.arange(p, T+T1); Xa = build_features(Y, ai, p); ya = Y[ai]
    nt  = T - p
    ms, Sb = fit_ensemble(Xa[:nt], ya[:nt], B, "ridge", rng)
    pB = batch_predict(ms, Xa); ep = pB.mean(0)
    lr = ya[:nt] - loo_predictions(pB[:,:nt], Sb, nt)
    fr = ya - ep
    def r(j): k=j-p; return float(fr[k]) if 0<=k<len(fr) else 0.
    tp = ep[T-p:T-p+T1]
    buf= deque(lr.tolist())
    cov=np.empty(T1,dtype=bool); lo=np.empty(T1); hi=np.empty(T1)
    for step in range(T1):
        t0=T+step; f=float(tp[step]); y=float(Y[t0])
        a=np.array(buf); b=line_search_beta(a,alpha)
        lo[step]=f+empirical_quantile(a,b)
        hi[step]=f+empirical_quantile(a,1-alpha+b)
        cov[step]=bool(lo[step]<=y<=hi[step])
        j0=T+step-1
        if j0>=p: buf.popleft(); buf.append(r(j0))
    return cov,lo,hi


def enbpis(Y, T, T1, alpha=ALPHA, B=B_BOOT, S=S, p=P_LAGS, seed=0):
    rng = np.random.default_rng(seed)
    ai  = np.arange(p, T+T1); Xa = build_features(Y, ai, p); ya = Y[ai]
    nt  = T - p
    ms, Sb = fit_ensemble(Xa[:nt], ya[:nt], B, "ridge", rng)
    pB = batch_predict(ms, Xa); ep = pB.mean(0)
    lr = ya[:nt] - loo_predictions(pB[:,:nt], Sb, nt)
    fr = ya - ep
    def r(j): k=j-p; return float(fr[k]) if 0<=k<len(fr) else 0.
    tp = ep[T-p:T-p+T1]
    bufs={s:deque() for s in range(1,S+1)}
    for k,idx in enumerate(ai[:nt]): bufs[season(idx,S)].append(float(lr[k]))
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
            e=r(j0); sj=season(j0,S)
            if bufs[sj]: bufs[sj].popleft()
            bufs[sj].append(e)
    return cov,lo,hi


# ── helpers ───────────────────────────────────────────────────────────

def season_cov_width(cov, lo, hi, T, T1, S=S):
    w = hi - lo
    return {s: (float(cov[[season(T+k,S)==s for k in range(T1)]].mean()),
                float(w  [[season(T+k,S)==s for k in range(T1)]].mean()))
            for s in range(1, S+1)}


# ── main experiment ───────────────────────────────────────────────────

def run_e1(tables_dir):
    rows = []
    for T in [120, 240, 480]:
        print(f"  T={T} ...", end=" ", flush=True)
        sa_cov = {s:[] for s in range(1,S+1)}
        po_cov = {s:[] for s in range(1,S+1)}
        st_cov = {s:[] for s in range(1,S+1)}
        sa_wid = {s:[] for s in range(1,S+1)}
        po_wid = {s:[] for s in range(1,S+1)}
        st_wid = {s:[] for s in range(1,S+1)}

        for rep in range(N_REP):
            seed = rep * 1000
            rng  = np.random.default_rng(seed)
            Y    = generate_series(T + T1, SIGMA_HIGH, SIGMA_LOW, rng)

            c_sa, l_sa, h_sa = sarima_gaussian(Y, T, T1)
            c_po, l_po, h_po = pooled_enbpi(Y, T, T1, seed=seed)
            c_st, l_st, h_st = enbpis(Y, T, T1, seed=seed)

            for s in range(1, S+1):
                mask = np.array([season(T+k,S)==s for k in range(T1)])
                sa_cov[s].append(float(c_sa[mask].mean()))
                po_cov[s].append(float(c_po[mask].mean()))
                st_cov[s].append(float(c_st[mask].mean()))
                sa_wid[s].append(float((h_sa-l_sa)[mask].mean()))
                po_wid[s].append(float((h_po-l_po)[mask].mean()))
                st_wid[s].append(float((h_st-l_st)[mask].mean()))

        for s in range(1, S+1):
            rows.append({
                "T": T, "season": s, "month": MONTH_NAMES[s-1],
                "high_vol": s in HIGH_SEAS,
                "cov_sarima": float(np.mean(sa_cov[s])),
                "cov_pooled": float(np.mean(po_cov[s])),
                "cov_strat":  float(np.mean(st_cov[s])),
                "wid_sarima": float(np.mean(sa_wid[s])),
                "wid_pooled": float(np.mean(po_wid[s])),
                "wid_strat":  float(np.mean(st_wid[s])),
            })
        print("done")

    df = pd.DataFrame(rows)
    path = os.path.join(tables_dir, "E1_three_methods.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}")
    return df


def latex_table_three(df, T_val=480, alpha=ALPHA):
    """Three-method coverage table for given T."""
    sub = df[df["T"] == T_val].copy()
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-season empirical coverage ($90\%$ nominal) for three methods.",
        f"$T={T_val}$, $\\alpha=0.10$, $S=12$, $n_{{\\mathrm{{rep}}}}={N_REP}$.}}",
        r"\label{tab:e1_three_methods_T" + str(T_val) + "}",
        r"\begin{tabular}{l ccc ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Coverage} & \multicolumn{3}{c}{Width} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Month & SARIMA & Pooled & EnbPI-S & SARIMA & Pooled & EnbPI-S \\",
        r"\midrule",
    ]

    def best_cov(vals):
        dists = [abs(v - (1-alpha)) for v in vals]
        return dists.index(min(dists))

    for _, row in sub.iterrows():
        cvs = [row["cov_sarima"], row["cov_pooled"], row["cov_strat"]]
        wds = [row["wid_sarima"], row["wid_pooled"], row["wid_strat"]]
        bi = best_cov(cvs)
        wi = wds.index(min(wds))

        cv_strs, wd_strs = [], []
        for i, (c, w) in enumerate(zip(cvs, wds)):
            cv_strs.append(f"\\textbf{{{c:.3f}}}" if i==bi else f"{c:.3f}")
            wd_strs.append(f"\\textit{{{w:.2f}}}"  if i==wi else f"{w:.2f}")

        tag = "*" if row["high_vol"] else ""
        lines.append(
            f"{row['month']}{tag} & "
            f"{cv_strs[0]} & {cv_strs[1]} & {cv_strs[2]} & "
            f"{wd_strs[0]} & {wd_strs[1]} & {wd_strs[2]} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\footnotesize "
        r"Bold coverage = closest to $90\%$ nominal; "
        r"italic width = narrowest.  "
        r"* = high-volatility months ($\sigma_{\mathrm{high}}=2$).} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    proj   = os.path.dirname(_HERE)
    tables = os.path.join(proj, "tables")
    os.makedirs(tables, exist_ok=True)

    print(f"E1 with 3 methods: SARIMA-Gaussian, Pooled EnbPI, EnbPI-S")
    print(f"N_REP={N_REP}, T1={T1}, alpha={ALPHA}\n")

    df = run_e1(tables)

    for T_val in [120, 240, 480]:
        tex  = latex_table_three(df, T_val)
        path = os.path.join(tables, f"tab_E1_three_methods_T{T_val}.tex")
        with open(path, "w", encoding="utf-8") as f: f.write(tex)
        print(f"  Saved {path}")

    # Summary for T=480
    print(f"\n{'Month':<6} {'SARIMA':>8} {'Pooled':>8} {'EnbPI-S':>8}")
    print("-" * 38)
    sub480 = df[df["T"]==480]
    for _, row in sub480.iterrows():
        tag = " *" if row["high_vol"] else ""
        print(f"{row['month']:<6} {row['cov_sarima']:>8.3f} "
              f"{row['cov_pooled']:>8.3f} {row['cov_strat']:>8.3f}{tag}")
    print("-" * 38)
    for col, label in [("cov_sarima","SARIMA"),("cov_pooled","Pooled"),("cov_strat","EnbPI-S")]:
        m = sub480[col].mean()
        hi = sub480[sub480["high_vol"]][col].mean()
        lo = sub480[~sub480["high_vol"]][col].mean()
        sp = sub480[col].max() - sub480[col].min()
        print(f"{label:<10}: mean={m:.3f}  hi-vol={hi:.3f}  "
              f"lo-vol={lo:.3f}  spread={sp:.3f}")
    print(f"\nNominal: {1-ALPHA:.2f}")
    print("\nAll done.")
