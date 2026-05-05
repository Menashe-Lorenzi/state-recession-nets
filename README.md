# State Co-Movement Networks for Recession Prediction

UCL COMP0047 — Data Science individual project. Builds a dynamic
correlation network over the 50 US states (Philadelphia Fed coincident
indexes, 1979–2025) and tests whether network features add predictive
power to a properly engineered macro recession baseline.

**Headline result.** Against the engineered 38-feature macro baseline,
network features add **nothing** at any horizon under either sparse-L1
logistic or tuned XGBoost; the 90 % paired-bootstrap CI brackets zero
everywhere. The descriptive recession typology shows *why*: the four
NBER events in the sample (1990, 2001, 2008-09, 2020) have
qualitatively different network signatures, so a single
fixed-coefficient predictor cannot fit them simultaneously.

The notebook `state_comovement_networks.ipynb` assembles the full
10-section analysis (data → networks → features → lead-lag →
walk-forward → robustness → recession typology → conclusion).

## Layout

```
state_recession/
├── state_comovement_networks.ipynb  — 10-section analysis notebook
├── data/
│   ├── raw/         — Phil-Fed Excel + FRED USREC
│   ├── processed/   — section-prefixed parquet/npz outputs
│   └── external/    — group-project baseline CSVs
├── src/             — data, networks, features, leadlag, modeling, plotting
├── scripts/         — section-prefixed runners (04, 06, 07, 08, 09)
├── experiments/
│   └── alt_feature_family/   — alternative network-feature sensitivity check
├── figures/         — section-prefixed PNG outputs
└── reports/
    └── tables/      — section-prefixed CSV tables
```

## Pipeline → notebook section mapping

The notebook is organised into 10 sections; all processed data,
figures, tables, and scripts are prefixed with the matching section
number so files sort by pipeline order.

| Section | Topic | Key outputs |
| --- | --- | --- |
| 1 | Returns + sanity | `01_state_returns.parquet`, `figures/01_returns_sanity.png` |
| 2 | Stationarity | `reports/tables/02_stationarity_tests.csv`, `figures/02_acf_pacf.png` |
| 3 | Rolling correlations | `03_rolling_corr_*.npz`, `figures/03_mean_corr.png`, `figures/03_robustness_mean_corr.png` |
| 4 | Network features | `04_network_features_*.parquet`, `figures/04_network_features.png`, `figures/04_features_polished.png`, `figures/04_mst_snapshots.png` |
| 5 | Lead-lag | `05_leadlag_*.parquet`, `figures/05_leadlag_auc.png`, `figures/05_subsamples.png` |
| 6 | Walk-forward vs simple 6-feature macro baseline | `06_walkforward_simple.parquet`, `06_simple_*` robustness, `figures/06_auc_comparison.png`, `figures/06_bootstrap.png` |
| 7 | Walk-forward vs engineered 38-feature baseline (apples-to-apples) | `07_walkforward_engineered{,_xgb}.parquet`, `reports/tables/07_*.csv` |
| 8 | Statistical robustness (paired bootstrap, drop-COVID, C-sweep, alt-feature family) | `08_*.parquet`, `experiments/alt_feature_family/` |
| 9 | Recession typology + PMFG (descriptive) | `reports/tables/09_recession_typology.csv`, `figures/09_pmfg_*`, `figures/09_typology_*` |
| 10 | Conclusion | notebook section |

## Reproducing the pipeline

Order matters. Each script reads inputs from the previous section
and writes its outputs into `data/processed/`, `figures/`, or
`reports/tables/`.

```bash
# Sections 1–5 are produced by the notebook itself
# (state_comovement_networks.ipynb) or by helper modules under src/.
# Sections 6–9 have standalone runners:
python scripts/06_run_walkforward_simple.py
python scripts/06_run_simple_robustness.py
python scripts/07_run_engineered_walkforward.py    # sanity + logistic + xgboost
python scripts/08_run_robustness.py
python scripts/09_run_recession_typology.py        # MST fingerprint + PMFG snapshots

# Polished plots (figures/04_features_polished.png, 04_mst_snapshots.png,
# 06_auc_comparison.png):
python scripts/04_plot_network_features.py
python scripts/06_plot_auc_comparison.py
```

Then open `state_comovement_networks.ipynb` to view the assembled
analysis.
