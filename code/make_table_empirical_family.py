"""LaTeX table for the calibration family on the two Brazilian series."""
import os
import pandas as pd

TAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tables")
d = pd.read_csv(os.path.join(TAB, "X10_empirical_family.csv"))
lam = pd.read_csv(os.path.join(TAB, "X10_empirical_lambda.csv"))

NICE = {"pool": "Pooled", "norm": r"\texttt{EnbPI-N}",
        "strat": r"\texttt{EnbPI-S}",
        "pool_f": r"Pooled$^\dagger$", "norm_f": r"\texttt{EnbPI-N}$^\dagger$",
        "strat_f": r"\texttt{EnbPI-S}$^\dagger$",
        "hyb_cv": r"\texttt{EnbPI-H}($\hat\lambda$)",
        "hyb_50": r"\texttt{EnbPI-H}($\tfrac12$)"}
ORDER = ["pool", "norm", "strat", "pool_f", "norm_f", "strat_f",
         "hyb_cv", "hyb_50"]

lines = [
    r"\begin{table}[ht]", r"\centering", r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\caption{The calibration schemes on the two applications",
    r"         ($\alpha = 0.10$, test window Jan 2015--Dec 2024,",
    r"         $n_s = 10$ observations per season).  A dagger marks the",
    r"         conformal order statistics with $\beta$ fixed at",
    r"         $\alpha/2$.  Spread is the largest minus the smallest",
    r"         per-season coverage and Worst the smallest; the width ratio",
    r"         is the largest over the smallest per-season mean width.",
    r"         Widths are in percentage points for the IPCA application",
    r"         and in per cent for export growth, so they are comparable",
    r"         only within an application.}",
    r"\label{tab:empirical_family}",
    r"\begin{tabular}{l l ccc cc}",
    r"\toprule",
    r"Application & Method & Coverage & Worst & Spread & Width"
    r" & Width ratio \\",
    r"\midrule",
]
for j, (app, label) in enumerate((("food", r"IPCA food ($\Ts = 20$)"),
                                  ("exports", r"Exports ($\Ts = 35$)"))):
    if j:
        lines.append(r"\midrule")
    for i, m in enumerate(ORDER):
        r = d[(d.application == app) & (d.method == m)].iloc[0]
        first = label if i == 0 else ""
        lines.append(
            f"{first} & {NICE[m]} & {r['coverage']:.3f} & {r['worst']:.2f} &"
            f" {r['spread']:.2f} & {r['width']:.2f} &"
            f" {r['width_ratio']:.2f} " + r"\\")
lines += [
    r"\bottomrule",
    r"\multicolumn{7}{l}{\footnotesize Nominal coverage $0.90$."
    r"  With ten test observations per season a per-season coverage is a}\\",
    r"\multicolumn{7}{l}{\footnotesize multiple of $0.1$, so Spread and"
    r" Worst are coarse; they are reported for completeness.}\\",
    r"\end{tabular}", r"\end{table}",
]
open(os.path.join(TAB, "tab_empirical_family.tex"), "w",
     encoding="utf-8").write("\n".join(lines) + "\n")
print("wrote tab_empirical_family.tex")
for app in ("food", "exports"):
    v = lam[lam.application == app]
    print(f"  {app} weights: " + ", ".join(
        f"{r['month']}={r['lambda']:.1f}" for _, r in v.iterrows()))
    print(f"    mean {v['lambda'].mean():.2f}, at 1.0 in {(v['lambda']==1).sum()} months, at 0.0 in {(v['lambda']==0).sum()}")
