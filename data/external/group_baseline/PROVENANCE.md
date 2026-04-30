# External data — group-project baseline

This folder contains the single macro panel imported from the UCL
COMP0047 group-presentation repository so that my project can run
end-to-end without depending on the group's working tree.

## File

- **`master_dataset_v20260317.csv`** — copied verbatim from:
  `UCL_COMP_0047_GROUP_PRESENTATION/DS_SOURCES_PIPELINE/master_dataset_v20260317.csv`

  Quarterly rows from 1948-Q1 onward with **untransformed** raw-level
  macro series (T10Y2Y, UNRATE, INDPRO, CPIAUCSL, TEDRATE, BAA10Y,
  FEDFUNDS) plus the pre-shifted 1Q / 2Q / 3Q-ahead recession targets.
  This is the single source feeding both of my baselines:

  - **Simple 6-feature baseline (Section 6).** Takes 6 of the 7 raw
    signals as levels (T10Y2Y, BAA10Y, UNRATE, INDPRO, CPIAUCSL,
    FEDFUNDS) — no lags, no rolling means, no transforms. TEDRATE is
    excluded because of extensive pre-1986 missingness that would
    otherwise drop the 1985-89 warm-up window.
  - **Engineered 38-feature baseline (Section 7).** Same six signals,
    each augmented with three lags, two rolling means (3Q and 6Q
    windows), plus two structural-break dummies (`post_2008`,
    `post_2020`). The Section 7 sanity harness (STEP 1 block of
    `scripts/07_run_engineered_walkforward.py`) validates that this
    reproduces the 0.836 / 0.820 / 0.765 logistic L1 AUCs on the full
    1948-2022 panel — matching the original group project's headline
    logistic numbers.

## Columns used by my project

- **6 signal columns** (shared by both baselines): `T10Y2Y`, `BAA10Y`,
  `UNRATE`, `INDPRO`, `CPIAUCSL`, `FEDFUNDS`.
- **Targets (3)**: `Target_1Q_ahead`, `Target_2Q_ahead`,
  `Target_3Q_ahead`.
- **Reference**: `USRECD` (contemporaneous NBER flag), `Date`.

The `*_missing` indicator columns are ignored.

## Why it's external, not `data/raw/`

`data/raw/` holds data *I* fetched from primary sources (Phil-Fed
coincident indexes, FRED USREC). `data/external/` holds data I did
**not** produce — I am consuming the group's untransformed levels
panel as-is.

## What it's used for

- `load_simple_baseline()` in `src/modeling.py` reads this CSV and
  returns the 6-signal levels panel for the **Section 6** walk-forward
  (simple 6-feature baseline + 5 network features).
- `load_macro_source()` + `build_engineered_features()` read the same
  CSV and build the 38-feature panel for the **Section 7** engineered
  baseline (headline comparison for both the logistic and XGBoost
  models).

NBER recession dates for my own sanity checks come from FRED directly
(see `data/raw/USREC.csv` and `data/raw/nber_usrec.parquet`), not from
the group's `USRECD.csv` which had too short a history.

## Earlier transformed CSV (removed)

A second file, `master_dataset_transformed_v20260317.csv`, used to live
in this directory and fed an earlier version of the Section 6 baseline
via a deprecated `load_raw_baseline()` path. That baseline was
7-feature *pre-transformed* (first-differences + log-YoY growth) and
produced a narrower 145-row 1986-Q1 → 2022-Q1 panel. It has been
**removed** to keep the project design to a single source of macro
data. If you need to re-pull it, the upstream group repo is the source
of truth.

## Source folder

These files originated in the
`UCL_COMP_0047_GROUP_PRESENTATION/DS_SOURCES_PIPELINE/` directory of
the group project. **The full 660 MB group folder has been removed
from this repo to keep it slim** — every file my pipeline depends on
has been copied here verbatim. If you need to re-pull anything, the
upstream group repo is the source of truth.
