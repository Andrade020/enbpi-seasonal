"""LaTeX tables for the calibration-family experiments, from X8_family.csv."""
import os
import pandas as pd

TAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tables")
d = pd.read_csv(os.path.join(TAB, "X8_family.csv"))

NICE = {"pool": "Pooled", "norm": r"\texttt{EnbPI-N}",
        "strat": r"\texttt{EnbPI-S}",
        "pool_f": r"Pooled$^\dagger$", "norm_f": r"\texttt{EnbPI-N}$^\dagger$",
        "strat_f": r"\texttt{EnbPI-S}$^\dagger$",
        "hyb_cv": r"\texttt{EnbPI-H}($\hat\lambda$)",
        "hyb_50": r"\texttt{EnbPI-H}($\tfrac12$)"}
ORDER = ["pool", "norm", "strat", "pool_f", "norm_f", "strat_f",
         "hyb_cv", "hyb_50"]
DGPS = [("scale", "Scale"), ("shape", "Shape"),
        ("mixed", "Both"), ("homosk", "None")]


def table_main(Ts=20):
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Experiment E6: calibration schemes across four",
        rf"         data-generating processes at $\Ts = {Ts}$",
        r"         ($\alpha = 0.10$, $n_{\mathrm{rep}} = 200$; Monte Carlo",
        r"         standard errors of the coverage entries do not exceed",
        f"         ${d[d.Ts==Ts].coverage_se.max():.4f}$).  A dagger marks the two",
        r"         repairs of Section~\ref{sec:algorithm:defects}: conformal",
        r"         order statistics and $\beta$ fixed at $\alpha/2$.  Spread",
        r"         is the largest minus the smallest per-season coverage,",
        r"         Worst the smallest.  All schemes share one ensemble per",
        r"         replication.}",
        r"\label{tab:e6_schemes}",
        r"\begin{tabular}{l cccc cccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Coverage} & \multicolumn{4}{c}{Spread} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Method & " + " & ".join(lbl for _, lbl in DGPS)
        + " & " + " & ".join(lbl for _, lbl in DGPS)
        + r" \\",
        r"\midrule",
    ]
    for m in ORDER:
        cells = []
        for col in ("coverage", "spread"):
            for dgp, _ in DGPS:
                r = d[(d.dgp == dgp) & (d.Ts == Ts) & (d.method == m)].iloc[0]
                cells.append(f"{r[col]:.3f}")
        lines.append(f"{NICE[m]} & " + " & ".join(cells) + r" \\")
        if m == "strat":
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\multicolumn{9}{l}{\footnotesize Nominal coverage $0.90$.  Column"
        r" headings name the seasonal structure of the errors: scale only,}\\",
        r"\multicolumn{9}{l}{\footnotesize shape only, both, or none.}\\",
        r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def table_ts():
    sub = ["strat", "strat_f", "hyb_50", "hyb_cv", "pool", "norm_f"]
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Experiment E6, continued: coverage and mean width as the",
        r"         per-stratum size grows, under seasonal scale",
        r"         heterogeneity ($n_{\mathrm{rep}} = 200$).  The",
        r"         non-monotonicity of the daggered schemes is the",
        r"         discreteness of the conformal index: at $\Ts = 20$ and",
        r"         $35$ the endpoints are the buffer extremes, at $\Ts = 50$",
        r"         they move inside.}",
        r"\label{tab:e6_ts}",
        r"\begin{tabular}{l cc cc cc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{$\Ts = 20$} & \multicolumn{2}{c}{$\Ts = 35$}"
        r" & \multicolumn{2}{c}{$\Ts = 50$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Method & Coverage & Width & Coverage & Width & Coverage & Width \\",
        r"\midrule",
    ]
    for m in sub:
        cells = []
        for Ts in (20, 35, 50):
            r = d[(d.dgp == "scale") & (d.Ts == Ts) & (d.method == m)].iloc[0]
            cells += [f"{r['coverage']:.3f}", f"{r['width']:.2f}"]
        lines.append(f"{NICE[m]} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\footnotesize Nominal coverage $0.90$.}\\",
        r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for name, fn in (("tab_E6_schemes", lambda: table_main(20)),
                     ("tab_E6_ts", table_ts)):
        p = os.path.join(TAB, f"{name}.tex")
        open(p, "w", encoding="utf-8").write(fn())
        print("wrote", name)
