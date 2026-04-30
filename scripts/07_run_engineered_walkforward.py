"""Section 7 — engineered-baseline walk-forward (logistic + XGBoost + sanity).

Single entry point for the three Section 7 runs previously split across
07_run_walkforward_engineered.py, 07_run_walkforward_engineered_xgb.py and
07_sanity_engineered.py. The outputs are bit-for-bit identical to the old
set because the protocol constants and the underlying helpers in
src.modeling haven't changed.

Three steps, in order:

  STEP 1  (sanity)     Reproduce the group project's headline logistic AUCs
                       (0.836 / 0.820 / 0.766) on the *full* 1948-2022
                       engineered panel — confirms the feature-engineering
                       replication before any network merge.

  STEP 2  (logistic)   Sparse-L1 logistic on the restricted 1985-2022 panel
                       (162 rows, created by the inner join on the EWMA
                       network features). Three variants: baseline_engineered
                       (38 feats), network (5 feats), combined_engineered
                       (43 feats). Writes the Section 7 headline logistic
                       table.

  STEP 3  (xgboost)    Tuned XGBoost on the same restricted panel. One-shot
                       TimeSeriesSplit(5) grid search on the full panel for
                       hyperparameters, then expanding-window walk-forward
                       refit (documented leakage: hyperparameter choice,
                       not model weights). Writes the 4-way logistic+XGB
                       comparison table.

Writes:
  data/processed/07_walkforward_engineered.parquet       (logistic run)
  data/processed/07_walkforward_engineered_xgb.parquet   (logistic+xgb run)
  reports/tables/07_auc_table.csv                        (logistic pivot)
  reports/tables/07_full_comparison.csv                  (4-way pivot)

Protocol (applies to steps 2 and 3):
  walk-forward initial=60, test=8, step=8, reject_partial=True
  logistic: L1, C=0.25, class_weight="balanced"
  xgboost:  grid over depth/lr/leaves/lambda/n_est, TSSplit(5), frozen
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import (  # noqa: E402
    NETWORK_COLS,
    TARGET_COLS,
    build_panel_engineered,
    walk_forward_eval,
)

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

EWMA_FEATURES = PROCESSED / "04_network_features_ewma.parquet"
FULL_PANEL_CSV = ROOT / "data" / "external" / "group_baseline" / "master_dataset_v20260317.csv"

INITIAL = 60
STEP = 8
TEST = 8
L1_C = 0.25
L1_CLASS_WEIGHT = "balanced"

# Sanity-step inputs (mirrors the old 07_sanity_engineered.py definitions)
SANITY_SIGNALS = ["T10Y2Y", "BAA10Y", "UNRATE", "INDPRO", "CPIAUCSL", "FEDFUNDS"]
SANITY_TARGETS = ["Target_1Q_ahead", "Target_2Q_ahead", "Target_3Q_ahead"]

# XGBoost tuning grid — small and overfit-resistant for a 162-row panel.
XGB_GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 3],
    "reg_lambda": [1.0, 5.0],
}
XGB_TS_CV_SPLITS = 5


# ============================================================================
# STEP 1 — sanity replication of the full-panel engineered logistic
# ============================================================================

def _sanity_build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    edf = df.copy()
    feat: list[str] = []
    for col in SANITY_SIGNALS:
        feat.append(col)
        for lag in [1, 2, 3]:
            n = f"{col}_lag{lag}"
            edf[n] = edf[col].shift(lag)
            feat.append(n)
        for win in [3, 6]:
            n = f"{col}_roll{win}"
            edf[n] = edf[col].rolling(win, min_periods=win).mean()
            feat.append(n)
    edf["post_2008"] = (edf["Date"] >= pd.Timestamp("2008-01-01")).astype(int)
    edf["post_2020"] = (edf["Date"] >= pd.Timestamp("2020-01-01")).astype(int)
    feat += ["post_2008", "post_2020"]
    return edf, feat


def _expanding_splits(n: int):
    out = []
    start = INITIAL
    while start + TEST <= n:
        out.append((np.arange(start), np.arange(start, start + TEST)))
        start += STEP
    return out


def _sanity_eval_horizon(edf: pd.DataFrame, feat: list[str], horizon: str):
    data = edf.dropna(subset=feat + [horizon]).copy().reset_index(drop=True)
    X = data[feat]
    y = data[horizon].astype(int)
    ys: list[int] = []
    preds: list[float] = []
    for tr, te in _expanding_splits(len(data)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(imp.fit_transform(Xtr))
        Xte_s = scaler.transform(imp.transform(Xte))
        if len(np.unique(ytr)) < 2:
            p = np.full(len(Xte), float(ytr.mean()))
        else:
            clf = LogisticRegression(
                penalty="l1", C=L1_C, class_weight=L1_CLASS_WEIGHT,
                max_iter=5000, solver="liblinear",
            )
            clf.fit(Xtr_s, ytr)
            p = clf.predict_proba(Xte_s)[:, 1]
        ys.extend(yte.tolist())
        preds.extend(p.tolist())
    ys_arr = np.array(ys)
    preds_arr = np.array(preds)
    return float(roc_auc_score(ys_arr, preds_arr)), len(ys_arr), int(ys_arr.sum())


def run_step1_sanity() -> None:
    print("\n" + "=" * 72)
    print("STEP 1 — full-panel sanity replication (0.836 / 0.820 / 0.766 target)")
    print("=" * 72)
    df = pd.read_csv(FULL_PANEL_CSV, parse_dates=["Date"])
    print(f"Loaded {len(df)} rows, {df['Date'].min().date()} → {df['Date'].max().date()}")
    edf, feat = _sanity_build_features(df)
    print(f"Feature count: {len(feat)}")
    print("Replication targets: 0.836 / 0.820 / 0.766\n")
    for h in SANITY_TARGETS:
        auc, n, pos = _sanity_eval_horizon(edf, feat, h)
        print(f"  {h}: AUC={auc:.4f}  n_oos={n}  positives={pos}")


# ============================================================================
# STEP 2 — restricted-panel sparse L1 logistic (Section 7 headline)
# ============================================================================

def run_step2_logistic(panel: pd.DataFrame, engineered_cols: list[str]) -> None:
    print("\n" + "=" * 72)
    print("STEP 2 — restricted-panel logistic (baseline / network / combined)")
    print("=" * 72)

    variants = {
        "baseline_engineered": engineered_cols,
        "network": NETWORK_COLS,
        "combined_engineered": engineered_cols + NETWORK_COLS,
    }

    rows = []
    for horizon, tgt in zip([1, 2, 3], TARGET_COLS):
        y = panel[tgt].astype(int)
        for variant, cols in variants.items():
            X = panel[cols]
            auc, _ = walk_forward_eval(
                X, y,
                initial=INITIAL, step=STEP, test=TEST,
                reject_partial=True, impute=True,
                model_type="logistic", C=L1_C, class_weight=L1_CLASS_WEIGHT,
            )
            rows.append(
                {"horizon_q": horizon, "variant": variant,
                 "model": "logistic", "auc": auc}
            )
            print(f"  {tgt:20s} {variant:22s}  AUC = {auc:.4f}")

    results = pd.DataFrame(rows)
    results.to_parquet(PROCESSED / "07_walkforward_engineered.parquet")

    pivot = results.pivot_table(index="horizon_q", columns="variant", values="auc")
    pivot["delta_comb_vs_base"] = (
        pivot["combined_engineered"] - pivot["baseline_engineered"]
    )
    pivot = pivot[
        ["baseline_engineered", "network", "combined_engineered", "delta_comb_vs_base"]
    ]
    pivot.to_csv(TABLES / "07_auc_table.csv")

    print("\n=== SECTION 7 HEADLINE (engineered baseline + EWMA network) ===")
    print(pivot.round(3).to_string())

    # Side-by-side with Section 6 EWMA+logistic for context.
    simple_parquet = PROCESSED / "06_walkforward_simple.parquet"
    if simple_parquet.exists():
        simple_results = pd.read_parquet(simple_parquet)
        simple_ewma = simple_results[
            (simple_results["estimator"] == "EWMA_hl24")
            & (simple_results["model"] == "logistic")
        ].pivot_table(index="horizon_q", columns="variant", values="auc")
        simple_ewma["delta_comb_vs_base"] = (
            simple_ewma["combined_simple"] - simple_ewma["baseline_simple"]
        )
        simple_ewma = simple_ewma[
            ["baseline_simple", "network", "combined_simple", "delta_comb_vs_base"]
        ]
        print("\n=== SECTION 6 (simple 6-feature untransformed baseline, for comparison) ===")
        print(simple_ewma.round(3).to_string())


# ============================================================================
# STEP 3 — restricted-panel tuned XGBoost (+ logistic re-run for 4-way table)
# ============================================================================

def _xgb(params: dict, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        **params,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
    )


def _grid_search(X_train: np.ndarray, y_train: np.ndarray) -> tuple[dict, float]:
    keys = list(XGB_GRID.keys())
    combos = list(itertools.product(*XGB_GRID.values()))
    tscv = TimeSeriesSplit(n_splits=XGB_TS_CV_SPLITS)
    best_auc = -np.inf
    best_params: dict | None = None
    for combo in combos:
        params = dict(zip(keys, combo))
        aucs: list[float] = []
        for tr_idx, va_idx in tscv.split(X_train):
            y_tr = y_train[tr_idx]
            y_va = y_train[va_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                continue
            pos = max(int(y_tr.sum()), 1)
            spw = float((len(y_tr) - y_tr.sum()) / pos)
            imp = SimpleImputer(strategy="median")
            X_tr_s = imp.fit_transform(X_train[tr_idx])
            X_va_s = imp.transform(X_train[va_idx])
            clf = _xgb(params, spw)
            clf.fit(X_tr_s, y_tr)
            p = clf.predict_proba(X_va_s)[:, 1]
            aucs.append(roc_auc_score(y_va, p))
        if not aucs:
            continue
        avg = float(np.mean(aucs))
        if avg > best_auc:
            best_auc = avg
            best_params = params
    assert best_params is not None, "grid search produced no valid fold"
    return best_params, best_auc


def _walk_forward_xgb(X: pd.DataFrame, y: pd.Series, params: dict) -> float:
    X = X.sort_index()
    y = y.reindex(X.index)
    n = len(X)
    ys: list[int] = []
    preds: list[float] = []
    start = INITIAL
    while start + TEST <= n:
        X_tr = X.iloc[:start].to_numpy(dtype=float)
        y_tr = y.iloc[:start].to_numpy(dtype=int)
        X_te = X.iloc[start : start + TEST].to_numpy(dtype=float)
        y_te = y.iloc[start : start + TEST].to_numpy(dtype=int)
        pos = max(int(y_tr.sum()), 1)
        spw = float((len(y_tr) - y_tr.sum()) / pos)
        imp = SimpleImputer(strategy="median")
        X_tr_s = imp.fit_transform(X_tr)
        X_te_s = imp.transform(X_te)
        if len(np.unique(y_tr)) < 2:
            p = np.full(len(y_te), float(y_tr.mean()))
        else:
            clf = _xgb(params, spw)
            clf.fit(X_tr_s, y_tr)
            p = clf.predict_proba(X_te_s)[:, 1]
        ys.extend(y_te.tolist())
        preds.extend(p.tolist())
        start += STEP
    return float(roc_auc_score(np.asarray(ys), np.asarray(preds)))


def run_step3_xgboost(panel: pd.DataFrame, engineered_cols: list[str]) -> None:
    print("\n" + "=" * 72)
    print("STEP 3 — restricted-panel tuned XGBoost (+ 4-way comparison table)")
    print("=" * 72)

    variants = {
        "baseline_engineered": engineered_cols,
        "combined_engineered": engineered_cols + NETWORK_COLS,
    }

    rows = []

    # Sparse L1 logistic re-run on exactly the 2 variants xgb sees, so the
    # comparison table pairs apples to apples (no "network only" column).
    print("\n--- Logistic (sparse L1, C=0.25) ---")
    for horizon, tgt in zip([1, 2, 3], TARGET_COLS):
        y = panel[tgt].astype(int)
        for variant, cols in variants.items():
            X = panel[cols]
            auc, _ = walk_forward_eval(
                X, y,
                initial=INITIAL, step=STEP, test=TEST,
                reject_partial=True, impute=True,
                model_type="logistic", C=L1_C, class_weight=L1_CLASS_WEIGHT,
            )
            rows.append({
                "model": "logistic", "variant": variant,
                "horizon_q": horizon, "auc": auc, "hyperparams": None,
            })
            print(f"  {tgt:20s} {variant:22s}  AUC = {auc:.4f}")

    print("\n--- XGBoost tuned (grid search, TimeSeriesSplit(5) on full panel) ---")
    for horizon, tgt in zip([1, 2, 3], TARGET_COLS):
        y = panel[tgt].astype(int)
        for variant, cols in variants.items():
            X = panel[cols]
            X_full = X.to_numpy(dtype=float)
            y_full = y.to_numpy(dtype=int)
            best_params, best_cv_auc = _grid_search(X_full, y_full)
            auc = _walk_forward_xgb(X, y, best_params)
            rows.append({
                "model": "xgboost", "variant": variant,
                "horizon_q": horizon, "auc": auc,
                "hyperparams": str(best_params),
            })
            compact = ", ".join(f"{k}={v}" for k, v in best_params.items())
            print(
                f"  {tgt:20s} {variant:22s}  AUC = {auc:.4f}  "
                f"(cv={best_cv_auc:.3f}  {compact})"
            )

    results = pd.DataFrame(rows)
    results.to_parquet(PROCESSED / "07_walkforward_engineered_xgb.parquet")

    pivot = results.pivot_table(
        index="horizon_q", columns=["model", "variant"], values="auc",
    )
    pivot[("delta", "logistic")] = (
        pivot[("logistic", "combined_engineered")]
        - pivot[("logistic", "baseline_engineered")]
    )
    pivot[("delta", "xgboost")] = (
        pivot[("xgboost", "combined_engineered")]
        - pivot[("xgboost", "baseline_engineered")]
    )
    pivot.to_csv(TABLES / "07_full_comparison.csv")

    print("\n=== SECTION 7 FULL COMPARISON (post-1985, engineered features) ===")
    print(pivot.round(3).to_string())


# ============================================================================
# Driver
# ============================================================================

def main() -> None:
    run_step1_sanity()

    print("\nLoading EWMA network features + building restricted engineered panel...")
    feats = pd.read_parquet(EWMA_FEATURES)
    panel, engineered_cols = build_panel_engineered(feats)
    print(f"  panel shape: {panel.shape}")
    print(f"  range: {panel.index.min().date()} → {panel.index.max().date()}")
    print(f"  class balance USRECD: {panel['USRECD'].mean():.3f}")
    print(f"  engineered feature count: {len(engineered_cols)}")

    run_step2_logistic(panel, engineered_cols)
    run_step3_xgboost(panel, engineered_cols)


if __name__ == "__main__":
    main()
