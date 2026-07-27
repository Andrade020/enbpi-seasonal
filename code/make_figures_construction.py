"""
make_figures_construction.py
============================
The two figures that carry the paper's main finding.

fig_coverage_by_month
    Per-season coverage of the season-specific interval on the two
    applications, under the two constructions: the empirical quantile
    with the minimum-width line search, and the conformal order
    statistic with a fixed asymmetry.  Same residuals, same ensemble,
    same buffer; only the reading of the buffer differs.  A dumbbell
    per month makes the shift the visual subject.

fig_linesearch_cost
    What the line search costs as a function of buffer length, from the
    controlled experiment on residual buffers drawn from a known
    distribution.  The two vertical rules are the exact thresholds
    2/alpha - 1 and 4/alpha - 1.

Colours are the Okabe-Ito blue and vermillion, which stay separable
under protanopia and deuteranopia; marker shape carries the same
distinction so the figures survive greyscale printing.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
TAB = os.path.join(PROJ, "tables")
FIG = os.path.join(PROJ, "figures")

ORIG = "#D55E00"      # original construction: empirical quantile + line search
REPA = "#0072B2"      # conformal order statistic + fixed asymmetry
INK = "#333333"
MUTED = "#8a8a8a"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NOMINAL = 0.90


def _tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def figure_coverage_by_month():
    d = pd.read_csv(os.path.join(TAB, "X11_empirical_by_season.csv"))
    tot = pd.read_csv(os.path.join(TAB, "X10_empirical_family.csv"))
    apps = [("food", "IPCA food at home  ($T_s = 20$)"),
            ("exports", "Export growth  ($T_s = 35$)")]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    for ax, (app, title) in zip(axes, apps):
        sub = d[d.application == app]
        o = [sub[(sub.method == "strat") & (sub.month == m)].coverage.values[0]
             for m in MONTHS]
        r = [sub[(sub.method == "strat_f") & (sub.month == m)].coverage.values[0]
             for m in MONTHS]
        x = np.arange(12)

        _tidy(ax)
        ax.axhline(NOMINAL, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                   zorder=1)
        ax.annotate("nominal 0.90", xy=(-0.35, NOMINAL + 0.012), fontsize=7.5,
                    color=MUTED, va="bottom", ha="left")
        # the shift, drawn first so the markers sit on top of it
        for xi, a, b in zip(x, o, r):
            if abs(a - b) > 1e-9:
                ax.plot([xi, xi], [a, b], color="#c9c9c9", linewidth=2.0,
                        solid_capstyle="round", zorder=2)
        ax.plot(x, o, "o", markersize=7, markerfacecolor="white",
                markeredgecolor=ORIG, markeredgewidth=1.8, linestyle="none",
                zorder=3, label="empirical quantile, line search")
        ax.plot(x, r, "s", markersize=7, color=REPA, linestyle="none",
                zorder=4, label="conformal order statistic, $\\beta=\\alpha/2$")

        ov_o = tot[(tot.application == app) & (tot.method == "strat")].coverage.values[0]
        ov_r = tot[(tot.application == app) & (tot.method == "strat_f")].coverage.values[0]
        ax.set_title(f"{title}      overall {ov_o:.3f} $\\rightarrow$ {ov_r:.3f}",
                     fontsize=9.5, color=INK, loc="left", pad=8)
        ax.set_ylim(0.55, 1.05)
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylabel("coverage", fontsize=8.5, color=INK)

    axes[-1].set_xticks(np.arange(12))
    axes[-1].set_xticklabels(MONTHS)
    axes[0].legend(fontsize=8, frameon=False, loc="upper center", ncol=2,
                   handletextpad=0.4, columnspacing=1.8,
                   bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    out = os.path.join(FIG, "fig_coverage_by_month.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(out))


def figure_linesearch_cost():
    d = pd.read_csv(os.path.join(TAB, "X9_linesearch.csv"))
    panels = [("symmetric", "Symmetric residuals  ($t_3$)"),
              ("skewed", "Right-skewed residuals  (lognormal)")]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, (dist, title) in zip(axes, panels):
        s = d[d.dist == dist]
        n = sorted(s.n.unique())
        fixed = [s[(s.n == v) & (s.m == 1)].coverage.values[0] for v in n]
        search = [s[(s.n == v) & (s.m == 21)].coverage.values[0] for v in n]

        _tidy(ax)
        for thr, lab in ((19, "$2/\\alpha-1$"), (39, "$4/\\alpha-1$")):
            ax.axvline(thr, color="#e0e0e0", linewidth=1.0, zorder=0)
            ax.annotate(lab, xy=(thr, 0.775), fontsize=7, color=MUTED,
                        rotation=90, va="bottom", ha="right")
        ax.axhline(NOMINAL, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                   zorder=1)
        ax.plot(n, fixed, "-s", color=REPA, linewidth=2.0, markersize=6,
                zorder=3, label="$\\beta$ fixed at $\\alpha/2$")
        ax.plot(n, search, "--o", color=ORIG, linewidth=2.0, markersize=6,
                markerfacecolor="white", markeredgewidth=1.6, zorder=3,
                label="line search over 21 candidates")
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200, 500])
        ax.set_xticklabels(["10", "20", "50", "100", "200", "500"])
        ax.set_xlabel("buffer length $n$", fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)
        ax.set_ylim(0.76, 0.97)

    axes[0].set_ylabel("coverage", fontsize=8.5, color=INK)
    axes[0].annotate("nominal 0.90", xy=(210, NOMINAL - 0.006), fontsize=7.5,
                     color=MUTED, va="top", ha="left")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_linesearch_cost.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    figure_coverage_by_month()
    figure_linesearch_cost()
