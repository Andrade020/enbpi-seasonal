# EnbPI-S: Distribution-Free Prediction Intervals for Seasonally Heteroskedastic Time Series

**Author:** Lucas Rafael de Andrade  
**Status:** Working paper

---

## Overview

This repository contains the replication code and data for the paper:

> **"Distribution-Free Prediction Intervals for Seasonally Heteroskedastic Time Series"**  
> Lucas Rafael de Andrade

We propose **EnbPI-S**, a stratified extension of the Ensemble batch Prediction Intervals (EnbPI) algorithm of [Xu & Xie (2023)](https://doi.org/10.1109/TPAMI.2023.3272339) that corrects for seasonal heteroskedasticity in time series forecasting.

### Key findings (simulation, T = 480, n = 200 replications)

| Method | High-vol coverage | Low-vol coverage | Coverage spread |
|---|---|---|---|
| SARIMA-Gaussian | 0.699 | 0.865 | 0.243 |
| Pooled EnbPI | 0.786 | 0.946 | 0.179 |
| **EnbPI-S** | **0.842** | **0.840** | **0.019** |

Nominal coverage: 90%. High-vol months: σ_high = 2; low-vol: σ_low = 1.

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
│   ├── simulation.py           # Monte Carlo experiments E1–E5
│   ├── simulation_sarima_e1.py # SARIMA benchmark comparison
│   ├── empirical_v2.py         # IPCA food-at-home application
│   ├── empirical_v3.py         # Brazilian export growth application
│   └── test_series.py          # Data availability checker
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

### Empirical applications

```bash
# Application 1: IPCA food-at-home (BCB series 1635)
python code/empirical_v2.py

# Application 2: Brazilian export growth (BCB series 1402)
python code/empirical_v3.py

# Use cached data (no download):
python code/empirical_v2.py --offline
python code/empirical_v3.py --offline
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
