"""
make_tables_E3E5.py
===================
Build the LaTeX tables for Experiments E3, E4 and E5 out of the CSV
files produced by simulation.py.  These three experiments were
previously reported only in the text.

Monte Carlo standard errors are reported as sd / sqrt(n_rep), with
n_rep = 200 for every run except the Random Forest arm of E5
(n_rep = 100), exactly as documented in Section 6.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
TAB = os.path.join(PROJ, "tables")

HIGH = {1, 2, 8, 9}
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
N_REP = 200
N_REP_RF = 100


def se(std, n=N_REP):
    return std / np.sqrt(n)


# ---------------------------------------------------------------- E3
def table_E3():
    d = pd.read_csv(os.path.join(TAB, "E3_heteroskedasticity.csv"))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Experiment E3: effect of the heteroskedasticity",
        r"         intensity $\sigma_{\mathrm{high}}$ on per-season",
        r"         coverage ($T = 240$, $\Ts = 20$, $\alpha = 0.10$,",
        r"         $n_{\mathrm{rep}} = 200$).  High-volatility seasons are",
        r"         January, February, August and September.  Coverage",
        r"         entries are averages over the seasons in each group;",
        r"         Monte Carlo standard errors are at most $0.010$.",
        r"         Spread is the largest minus the smallest per-season",
        r"         coverage.}",
        r"\label{tab:e3_intensity}",
        r"\begin{tabular}{c cc cc cc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{High-volatility coverage} &"
        r" \multicolumn{2}{c}{Low-volatility coverage} &"
        r" \multicolumn{2}{c}{Coverage spread} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"$\sigma_{\mathrm{high}}$ & Pooled & \texttt{EnbPI-S}"
        r" & Pooled & \texttt{EnbPI-S} & Pooled & \texttt{EnbPI-S} \\",
        r"\midrule",
    ]
    for sh in sorted(d["sigma_high"].unique()):
        row = [f"{sh:.1f}"]
        for grp in ["high", "low", "spread"]:
            for m in ["Pooled", "EnbPI-S"]:
                sub = d[(d["sigma_high"] == sh) & (d["method"] == m)]
                if grp == "high":
                    v = sub[sub["season"].isin(HIGH)]["coverage_mean"].mean()
                elif grp == "low":
                    v = sub[~sub["season"].isin(HIGH)]["coverage_mean"].mean()
                else:
                    v = (sub["coverage_mean"].max() -
                         sub["coverage_mean"].min())
                row.append(f"{v:.3f}")
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\footnotesize Nominal coverage $1-\alpha = 0.90$."
        r"  $\sigma_{\mathrm{high}} = 1.0$ is the homoskedastic null case"
        r" of Experiment E4.}\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- E4
def table_E4():
    d = pd.read_csv(os.path.join(TAB, "E4_null_case.csv"))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Experiment E4: null (homoskedastic) case,",
        r"         $\sigma_s = 1$ for every $s$, $T = 240$,",
        r"         $\Ts = 20$, $\alpha = 0.10$,",
        r"         $n_{\mathrm{rep}} = 200$.  Monte Carlo standard errors",
        r"         are in parentheses.  The gap between the two columns is",
        r"         the cost of stratification when there is no seasonal",
        r"         bias to remove.}",
        r"\label{tab:e4_null}",
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{Width} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Month & Pooled & \texttt{EnbPI-S} & Pooled & \texttt{EnbPI-S} \\",
        r"\midrule",
    ]
    cov = {m: [] for m in ["Pooled", "EnbPI-S"]}
    wid = {m: [] for m in ["Pooled", "EnbPI-S"]}
    for s in range(1, 13):
        row = [MONTHS[s - 1]]
        for m in ["Pooled", "EnbPI-S"]:
            r = d[(d["season"] == s) & (d["method"] == m)].iloc[0]
            cov[m].append(r["coverage_mean"])
            row.append(f"{r['coverage_mean']:.3f} ({se(r['coverage_std']):.3f})")
        for m in ["Pooled", "EnbPI-S"]:
            r = d[(d["season"] == s) & (d["method"] == m)].iloc[0]
            wid[m].append(r["width_mean"])
            row.append(f"{r['width_mean']:.2f}")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join(
        ["Mean"] +
        [f"{np.mean(cov[m]):.3f}" for m in ["Pooled", "EnbPI-S"]] +
        [f"{np.mean(wid[m]):.2f}" for m in ["Pooled", "EnbPI-S"]]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{5}{l}{\footnotesize Nominal coverage"
        r" $1-\alpha = 0.90$; the theoretical seasonal bias is"
        r" $b_{s^*} = 0$ throughout.}\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- E5
def table_E5():
    d = pd.read_csv(os.path.join(TAB, "E5_predictor_robustness.csv"))
    names = {"ridge": "Ridge", "rf": "Random Forest"}
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Experiment E5: predictor robustness",
        r"         ($T = 240$, $\Ts = 20$, $\sigma_{\mathrm{high}} = 2$,",
        r"         $\alpha = 0.10$; $n_{\mathrm{rep}} = 200$ for Ridge and",
        r"         $n_{\mathrm{rep}} = 100$ for the Random Forest).",
        r"         Coverage entries are averages over the seasons in each",
        r"         group.  The seasonal pattern is the same under both",
        r"         base learners, which is what makes the gain from",
        r"         stratification model-agnostic.}",
        r"\label{tab:e5_predictor}",
        r"\begin{tabular}{l l cc c c}",
        r"\toprule",
        r"Base learner & Method & High-vol.\ & Low-vol.\ & Spread"
        r" & Width ratio \\",
        r"\midrule",
    ]
    for model in ["ridge", "rf"]:
        for j, m in enumerate(["Pooled", "EnbPI-S"]):
            sub = d[(d["model"] == model) & (d["method"] == m)]
            hv = sub[sub["season"].isin(HIGH)]
            lv = sub[~sub["season"].isin(HIGH)]
            spread = sub["coverage_mean"].max() - sub["coverage_mean"].min()
            ratio = hv["width_mean"].mean() / lv["width_mean"].mean()
            label = names[model] if j == 0 else ""
            meth = r"\texttt{EnbPI-S}" if m == "EnbPI-S" else m
            lines.append(
                f"{label} & {meth} & {hv['coverage_mean'].mean():.3f} &"
                f" {lv['coverage_mean'].mean():.3f} & {spread:.3f} &"
                f" {ratio:.2f} " + r"\\")
        if model == "ridge":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\footnotesize Width ratio = mean"
        r" high-volatility width / mean low-volatility width; the true"
        r" scale ratio is $2$.}\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for name, fn in [("tab_E3", table_E3), ("tab_E4", table_E4),
                     ("tab_E5", table_E5)]:
        path = os.path.join(TAB, f"{name}.tex")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(fn())
        print("wrote", path)

    # --- numbers quoted in the text, for verification ----------------
    d3 = pd.read_csv(os.path.join(TAB, "E3_heteroskedasticity.csv"))
    print("\nE3 checks")
    for sh in sorted(d3["sigma_high"].unique()):
        for m in ["Pooled", "EnbPI-S"]:
            sub = d3[(d3["sigma_high"] == sh) & (d3["method"] == m)]
            jan = sub[sub["season"] == 1]["coverage_mean"].values[0]
            feb = sub[sub["season"] == 2]["coverage_mean"].values[0]
            lo = sub[~sub["season"].isin(HIGH)]["coverage_mean"]
            print(f"  sh={sh} {m:<8} Jan={jan:.3f} Feb={feb:.3f} "
                  f"low=[{lo.min():.3f},{lo.max():.3f}]")
    w3 = d3[(d3["sigma_high"] == 3.0)]
    for s in [1, 2]:
        wp = w3[(w3["method"] == "Pooled") & (w3["season"] == s)]["width_mean"].values[0]
        ws = w3[(w3["method"] == "EnbPI-S") & (w3["season"] == s)]["width_mean"].values[0]
        print(f"  sh=3 season {s}: width ratio EnbPI-S/Pooled = {ws/wp:.2f}")

    d4 = pd.read_csv(os.path.join(TAB, "E4_null_case.csv"))
    print("\nE4 checks")
    for m in ["Pooled", "EnbPI-S"]:
        sub = d4[d4["method"] == m]["coverage_mean"]
        print(f"  {m:<8} mean={sub.mean():.3f} range=[{sub.min():.3f},{sub.max():.3f}]")

    d5 = pd.read_csv(os.path.join(TAB, "E5_predictor_robustness.csv"))
    print("\nE5 checks")
    for model in ["ridge", "rf"]:
        for m in ["Pooled", "EnbPI-S"]:
            sub = d5[(d5["model"] == model) & (d5["method"] == m)]
            hv = sub[sub["season"].isin(HIGH)]["coverage_mean"].mean()
            lv = sub[~sub["season"].isin(HIGH)]["coverage_mean"].mean()
            sp = sub["coverage_mean"].max() - sub["coverage_mean"].min()
            print(f"  {model:<6} {m:<8} high={hv:.3f} low={lv:.3f} spread={sp:.3f}")
