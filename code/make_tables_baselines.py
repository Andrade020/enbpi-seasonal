"""
make_tables_baselines.py
========================
LaTeX tables for the calibration-scheme comparison (Experiments E6 and
E7), built from the CSVs written by simulation_baselines.py.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(os.path.dirname(HERE), "tables")
HIGH = {1, 2, 8, 9}

NICE = {"Pooled": "Pooled", "EnbPI-N": r"\texttt{EnbPI-N}",
        "EnbPI-S": r"\texttt{EnbPI-S}",
        "Pooled-C": r"Pooled$+$", "EnbPI-N-C": r"\texttt{EnbPI-N}$+$",
        "EnbPI-S-C": r"\texttt{EnbPI-S}$+$"}
ORDER = ["Pooled", "EnbPI-N", "EnbPI-S", "Pooled-C", "EnbPI-N-C", "EnbPI-S-C"]


def panel(df, T, group_label):
    """One panel of rows for a given training size."""
    sub = df[df["T"] == T]
    lines = [rf"\multicolumn{{8}}{{l}}{{\emph{{$T = {T}$}}}} \\"]
    for m in ORDER:
        d = sub[sub["method"] == m]
        per = d[d["season"] > 0]
        hv = per[per["season"].isin(HIGH)]
        lv = per[~per["season"].isin(HIGH)]
        ov = d[d["season"] == 0]
        spread = per["coverage_mean"].max() - per["coverage_mean"].min()
        lines.append(
            f"{NICE[m]} & {ov['coverage_mean'].values[0]:.3f} & "
            f"{hv['coverage_mean'].mean():.3f} & {lv['coverage_mean'].mean():.3f} & "
            f"{spread:.3f} & {per['coverage_mean'].min():.3f} & "
            f"{hv['width_mean'].mean():.2f} & {lv['width_mean'].mean():.2f} " + r"\\")
    return lines


def make_table(csv, out, label, caption, col_high, col_low, Ts):
    df = pd.read_csv(os.path.join(TAB, csv))
    max_se = df["coverage_se"].max()
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        caption.replace("MAXSE", f"{max_se:.3f}"),
        rf"\label{{{label}}}",
        r"\begin{tabular}{l cccc c cc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Coverage} & &"
        r" \multicolumn{2}{c}{Mean width} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){7-8}",
        rf"Method & Overall & {col_high} & {col_low} & Spread & Worst"
        rf" & {col_high} & {col_low} \\",
        r"\midrule",
    ]
    for j, T in enumerate(Ts):
        if j:
            lines.append(r"\midrule")
        lines += panel(df, T, "")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(TAB, out), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", out, f"(max MC s.e. {max_se:.4f})")


def table_X2():
    df = pd.read_csv(os.path.join(TAB, "X2_baselines_stratum_size.csv"))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Experiment E6, continued: overall coverage as a",
        r"         function of the per-stratum sample size $\Ts$",
        r"         ($\sigma_{\mathrm{high}} = 2$, $\alpha = 0.10$,",
        r"         $n_{\mathrm{rep}} = 200$).  A $+$ marks the variant",
        r"         with the finite-sample conformal quantile levels.}",
        r"\label{tab:e6_stratum_size}",
        r"\begin{tabular}{c cccccc}",
        r"\toprule",
        r"$\Ts$ & " + " & ".join(NICE[m] for m in ORDER) + r" \\",
        r"\midrule",
    ]
    for Ts in sorted(df["Ts"].unique()):
        sub = df[df["Ts"] == Ts]
        vals = [sub[sub["method"] == m]["coverage_mean"].values[0]
                for m in ORDER]
        lines.append(f"{Ts} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\footnotesize Nominal coverage $0.90$;"
        r" Monte Carlo standard errors do not exceed"
        f" ${df['coverage_se'].max():.3f}$." + r"}\\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(TAB, "tab_E6_stratum_size.tex"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote tab_E6_stratum_size.tex")


if __name__ == "__main__":
    make_table(
        "X1_baselines_by_season.csv", "tab_E6_schemes.tex",
        "tab:e6_schemes",
        "\n".join([
            r"\caption{Experiment E6: calibration schemes under seasonal",
            r"         scale heterogeneity ($\sigma_{\mathrm{high}} = 2$,",
            r"         $\alpha = 0.10$, $n_{\mathrm{rep}} = 200$; Monte",
            r"         Carlo standard errors at most MAXSE).",
            r"         All schemes share the same bootstrap ensemble and",
            r"         the same LOO residuals, so they differ only in the",
            r"         calibration step.  A $+$ marks the variant with the",
            r"         finite-sample conformal quantile levels.  Spread is",
            r"         the largest minus the smallest per-season coverage;",
            r"         Worst is the smallest per-season coverage.}"]),
        "High-vol.", "Low-vol.", [240, 480])

    make_table(
        "X3_shape_heterogeneity.csv", "tab_E7_shape.tex",
        "tab:e7_shape",
        "\n".join([
            r"\caption{Experiment E7: seasonality in distributional shape",
            r"         rather than scale.  All seasons have unit",
            r"         innovation variance; four seasons draw",
            r"         standardised lognormal innovations (strongly",
            r"         right-skewed) and eight draw standard normal ones",
            r"         ($\alpha = 0.10$, $n_{\mathrm{rep}} = 200$; Monte",
            r"         Carlo standard errors at most MAXSE).",
            r"         Because the seasonal scales are equal, a scheme",
            r"         that only rescales residuals has nothing to",
            r"         correct.}"]),
        "Skewed", "Gaussian", [240, 480])

    table_X2()
