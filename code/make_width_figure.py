"""
make_width_figure.py
Replaces the time-series-with-misses figure with an interval-width-over-time
figure. Shows the seasonal pulsing of EnbPI-S widths vs the flat Pooled width.
No realized values, no red circles -- highlights the key contribution.
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

_HERE = os.path.dirname(os.path.abspath(__file__))
proj  = os.path.dirname(_HERE)

MONTH_ABBR = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']


def make_width_over_time(csv_path, out_path, title,
                         ylabel, alpha_line=0.10,
                         highlight_months=None):
    """
    Two-panel figure:
      Top: interval widths over time (Pooled flat, EnbPI-S seasonal)
      Bottom: same data collapsed to mean width per calendar month (bar)
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df['width_pool']  = df['hi_pool']  - df['lo_pool']
    df['width_strat'] = df['hi_strat'] - df['lo_strat']
    df['month']       = df['date'].dt.month

    fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                             gridspec_kw={'height_ratios': [3, 2]})
    fig.subplots_adjust(hspace=0.35)

    # ── Panel 1: width over time ──────────────────────────────────────
    ax = axes[0]
    dates = df['date']

    ax.plot(dates, df['width_pool'],  color='tab:red',  lw=1.2,
            label='Pooled (constant width)', alpha=0.85)
    ax.plot(dates, df['width_strat'], color='tab:blue', lw=1.2,
            label='EnbPI-S (seasonal width)', alpha=0.85)

    # Shade high-uncertainty months (if provided)
    if highlight_months:
        for m in highlight_months:
            mask = df['month'] == m
            for _, row in df[mask].iterrows():
                ax.axvspan(row['date'] - pd.DateOffset(days=15),
                           row['date'] + pd.DateOffset(days=15),
                           alpha=0.07, color='gold', zorder=0)

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.set_xlim(dates.min(), dates.max())

    # ── Panel 2: mean width by calendar month (bar) ───────────────────
    ax2    = axes[1]
    x      = np.arange(1, 13)
    w      = 0.35
    mean_p = [df[df.month == m]['width_pool'].mean()  for m in x]
    mean_s = [df[df.month == m]['width_strat'].mean() for m in x]

    ax2.bar(x - w/2, mean_p, w, color='tab:red',  alpha=0.75, label='Pooled')
    ax2.bar(x + w/2, mean_s, w, color='tab:blue', alpha=0.75, label='EnbPI-S')
    ax2.set_xticks(x)
    ax2.set_xticklabels(MONTH_ABBR, fontsize=8)
    ax2.set_ylabel('Mean width', fontsize=9)
    ax2.set_title('Average interval width by calendar month', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}')


if __name__ == '__main__':
    figs = os.path.join(proj, 'figures')
    os.makedirs(figs, exist_ok=True)

    # Export growth figure
    make_width_over_time(
        csv_path=os.path.join(proj, 'tables', 'empirical_exports_intervals.csv'),
        out_path=os.path.join(figs, 'fig_empirical_exports_width_time.pdf'),
        title='Interval width over time — Brazilian export growth (2015–2024)',
        ylabel='Width (%)',
        highlight_months=[3, 4, 5],   # Mar-May: harvest-export ramp-up (peak April)
    )

    # IPCA food figure (optional)
    make_width_over_time(
        csv_path=os.path.join(proj, 'tables', 'empirical_food_intervals.csv'),
        out_path=os.path.join(figs, 'fig_empirical_food_width_time.pdf'),
        title='Interval width over time — IPCA food-at-home (2015–2024)',
        ylabel='Width (p.p.)',
        highlight_months=[5, 6],   # May-Jun: pre-harvest
    )

    print('Done.')
