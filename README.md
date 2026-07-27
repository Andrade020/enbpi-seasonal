# EnbPI-S: Distribution-Free Prediction Intervals for Seasonally Heteroskedastic Time Series

**Author:** Lucas Rafael de Andrade  
**Status:** Working paper

---

## Overview

This repository contains the replication code and data for the paper:

> **"Distribution-Free Prediction Intervals for Seasonally Heteroskedastic Time Series"**  
> Lucas Rafael de Andrade

We study how the calibration step of the Ensemble batch Prediction Intervals (EnbPI) algorithm of [Xu & Xie (2023)](https://doi.org/10.1109/TPAMI.2023.3272339) should depend on the calendar season, and find that most of what looks like the cost of conditioning on the season is in fact the cost of how the interval is built.

### Seasonal calibration (simulation, T = 480, 200 replications)

Pooled calibration carries a season-dependent bias that data cannot remove.

| Method | High-vol coverage | Low-vol coverage | Coverage spread |
|---|---|---|---|
| SARIMA-Gaussian | 0.699 | 0.865 | 0.243 |
| Pooled EnbPI | 0.786 | 0.946 | 0.179 |
| **EnbPI-S** | **0.842** | **0.840** | **0.019** |

### The interval construction (simulation, 20 residuals per season, 200 replications)

The same season buffer, read as an empirical quantile with the minimum-width
line search or as the conformal order statistic with a fixed asymmetry:

| Seasonal structure | Empirical quantile | Conformal order statistic |
|---|---|---|
| scale only | 0.785 | **0.898** |
| shape only | 0.790 | **0.900** |
| both | 0.790 | **0.898** |
| neither | 0.789 | **0.901** |

On the IPCA food application the same change takes coverage from 0.792 to
exactly 0.900, with identical coverage in all twelve calendar months.

Nominal coverage is 90% throughout. High-vol months: σ_high = 2; low-vol: σ_low = 1.

---

## Repository structure

```
EnbPI_Sazonal/
├── main.tex                    # Main LaTeX file
├── references.bib              # Bibliography
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_model.tex
│   ├── 03_pooled_gap.tex
│   ├── 04_algorithm.tex
│   ├── 05_theory.tex
│   ├── 06_simulations.tex
│   ├── 07_empirical.tex
│   ├── 08_conclusion.tex
│   ├── A_proofs.tex
│   └── B_additional_results.tex
├── code/
│   ├── simulation.py             # Monte Carlo experiments E1-E5
│   ├── simulation_sarima_e1.py   # SARIMA benchmark comparison
│   ├── fastcal.py                # calibration primitives: conformal order
│   │                             #   statistics, endpoints, exact scoring
│   ├── simulation_family.py      # Experiment E7: eight schemes x four DGPs
│   ├── diag_linesearch.py        # Experiment E6: cost of the line search
│   ├── diag_bootstrap.py         # inward quantile bias; bootstrap accuracy
│   ├── diag_grid.py              # weight profile over the four DGPs
│   ├── diag_beta.py              # fixed asymmetry against the line search
│   ├── diag_selection.py         # weight-selection rules against the oracle
│   ├── shrinkage.py              # the moment-based weight (reported as tried
│   │                             #   and abandoned in Section 5)
│   ├── test_shrinkage.py         # unit checks for the weight
│   ├── simulation_baselines.py   # earlier scheme comparison and the shape DGP
│   ├── simulation_hybrid.py      # earlier weight-grid run, superseded by
│   │                             #   simulation_family.py
│   ├── run_null_all.py           # homoskedastic case under every scheme
│   ├── make_tables_E3E5.py       # LaTeX tables for E3-E5
│   ├── make_tables_baselines.py  # LaTeX tables for the earlier comparison
│   ├── make_tables_family.py     # LaTeX tables for E7
│   ├── make_table_E9.py          # LaTeX table for E6
│   ├── make_table_empirical_family.py  # empirical scheme table
│   ├── make_figures_construction.py    # the two construction figures
│   ├── empirical_v2.py           # IPCA food-at-home application
│   ├── empirical_v3.py           # Brazilian export growth application
│   ├── empirical_baselines.py    # competing schemes + permutation diagnostic
│   ├── empirical_family.py       # the eight schemes on both series
│   └── test_series.py            # data availability checker
├── tables/                     # Auto-generated LaTeX tables
├── figures/                    # Auto-generated PDF figures
└── data/                       # Cached downloaded series
```

---

## Requirements

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib requests
```

Python 3.9+.

---

## Replication

### Monte Carlo simulation (E1–E5, ~3 hours)

```bash
python code/simulation.py
```

Quick test (30 reps, ~5 min):

```bash
python code/simulation.py --quick
```

### SARIMA benchmark (E1 only, ~15 min)

```bash
python code/simulation_sarima_e1.py
```

### Calibration schemes and the interval construction (E6-E7, ~40 min)

```bash
python code/simulation_family.py              # E7, the main grid (200 reps)
python code/simulation_family.py --quick      # 25 reps
python code/simulation_family.py --budget 480 # stop after 8 minutes; rerun to resume
python code/diag_linesearch.py                # E6, cost of the line search
python code/diag_bootstrap.py                 # inward bias of extreme quantiles
python code/run_null_all.py                   # homoskedastic case, all schemes
```

### Anonymous (double-blind) version of the paper

```bash
pdflatex -jobname=main_blind "\def\blindmode{1}\input{main}"
bibtex main_blind
pdflatex -jobname=main_blind "\def\blindmode{1}\input{main}"
pdflatex -jobname=main_blind "\def\blindmode{1}\input{main}"
```

### LaTeX tables

```bash
python code/make_tables_E3E5.py       # tables for E3, E4, E5
python code/make_tables_family.py     # tables for E7
python code/make_table_E9.py          # table for E6
python code/make_table_empirical_family.py  # empirical scheme table
python code/make_figures_construction.py  # coverage-by-month and line-search figures
```

### Empirical applications

```bash
# Application 1: IPCA food-at-home (BCB series 1635)
python code/empirical_v2.py

# Application 2: Brazilian export growth (BCB series 1402)
python code/empirical_v3.py

# Use cached data (no download):
python code/empirical_v2.py --offline
python code/empirical_v3.py --offline

# Competing calibration schemes and the permutation diagnostic
# (offline: reads the cached CSVs in data/)
python code/empirical_baselines.py
python code/empirical_family.py
```

All outputs (tables and figures) are saved to `tables/` and `figures/`.

---

## Data sources

| Series | Source | Description |
|---|---|---|
| BCB SGS 1635 | [Banco Central do Brasil](https://www.bcb.gov.br) | IPCA food-at-home (% m/m) |
| BCB SGS 1402 | [Banco Central do Brasil](https://www.bcb.gov.br) | Brazilian exports (USD mi) |

Both series are downloaded automatically via the [BCB open-data API](https://api.bcb.gov.br).

---

## Reference

If you use this code, please cite:

```bibtex
@article{andrade2025enbpis,
  author  = {Andrade, Lucas Rafael de},
  title   = {Distribution-Free Prediction Intervals for
             Seasonally Heteroskedastic Time Series},
  year    = {2025},
  note    = {Working paper}
}
```

---

## License

MIT License. See `LICENSE` for details.
