"""LaTeX table for Experiment E9 (line-search selection bias)."""
import os
import pandas as pd

TAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tables")
d = pd.read_csv(os.path.join(TAB, "X9_linesearch.csv"))
NS = [10, 20, 35, 50, 100, 200, 500]
MS = [1, 21, 201]

lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\small",
    r"\setlength{\tabcolsep}{5pt}",
    r"\caption{Experiment E9: what the minimum-width line search costs.",
    r"         Coverage and mean width of a $90\%$ interval built from a",
    r"         single buffer of $n$ residuals, with finite-sample conformal",
    r"         levels throughout, so that the only difference across",
    r"         columns is whether the asymmetry $\beta$ is fixed at",
    r"         $\alpha/2$ ($m = 1$) or chosen as the narrowest of $m$",
    r"         candidates.  Symmetric residuals are standardised $t_3$,",
    r"         skewed ones standardised lognormal; $3000$ buffers per cell,",
    r"         evaluated against $200\,000$ draws.}",
    r"\label{tab:e9_linesearch}",
    r"\begin{tabular}{r cc cc cc}",
    r"\toprule",
    r" & \multicolumn{2}{c}{$m = 1$ (fixed)} &"
    r" \multicolumn{2}{c}{$m = 21$} & \multicolumn{2}{c}{$m = 201$} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"$n$ & Coverage & Width & Coverage & Width & Coverage & Width \\",
    r"\midrule",
    r"\multicolumn{7}{l}{\emph{Symmetric residuals}} \\",
]

for dist, label in (("symmetric", None), ("skewed", r"\emph{Skewed residuals}")):
    if label:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{l}}{{{label}}} \\")
    for n in NS:
        cells = []
        for m in MS:
            r = d[(d.dist == dist) & (d.n == n) & (d.m == m)].iloc[0]
            cells += [f"{r['coverage']:.3f}", f"{r['width']:.2f}"]
        lines.append(f"{n} & " + " & ".join(cells) + r" \\")

lines += [
    r"\bottomrule",
    r"\multicolumn{7}{l}{\footnotesize Nominal coverage $0.90$.  The upper",
    r" conformal level is below one only for $n \geq 39$ at this nominal}\\",
    r"\multicolumn{7}{l}{\footnotesize level, so for shorter buffers the",
    r" upper endpoint is the largest residual in the buffer.}\\",
    r"\end{tabular}",
    r"\end{table}",
]
with open(os.path.join(TAB, "tab_E9_linesearch.tex"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("wrote tab_E9_linesearch.tex")
