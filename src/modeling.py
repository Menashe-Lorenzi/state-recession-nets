"""Walk-forward predictive evaluation on two feature families.

Two baselines are defined on the group project's *untransformed* macro levels
panel (``master_dataset_v20260317.csv``):

- **simple**: 6 untransformed macro signals (T10Y2Y, BAA10Y, UNRATE, INDPRO,
  CPIAUCSL, FEDFUNDS) used as levels, no lags, no rolling means. This is the
  simple feature set used in Section 6.
- **engineered**: 38 features built from the same 6 signals
  (6 × {level + 3 lags + 2 rolling means} + 2 post-break dummies). This is the
  richer feature set used in Sections 7-8 and matches what the original group
  project's logistic / XGBoost baselines were trained on.

Both feature families read from the same source CSV and can be combined with
the 5 network features and scored under either ``logistic`` (sparse L1) or
``xgboost``, giving the two-axis ``{simple, engineered} × {logistic, xgboost}``
design used throughout the project.

Pooled out-of-sample AUC is the headline metric (walk-forward expanding-window,
refit every ``step`` quarters).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


ROOT = Path(__file__).resolve().parents[1]
# Untransformed macro levels CSV sourced from the UCL COMP0047 group project
# and copied into data/external/group_baseline/ — see PROVENANCE.md in that
# folder. Both the simple 6-feature baseline (Section 6) and the engineered
# 38-feature baseline (Sections 7-8) read from this file.
MACRO_SOURCE_CSV = (
    ROOT
    / "data"
    / "external"
    / "group_baseline"
    / "master_dataset_v20260317.csv"
)

SIMPLE_FEATURE_COLS = [
    "T10Y2Y",
    "BAA10Y",
    "UNRATE",
    "INDPRO",
    "CPIAUCSL",
    "FEDFUNDS",
]
NETWORK_COLS = [
    "mst_length",
    "mean_corr",
    "largest_eigenvalue",
    "network_density",
    "n_communities",
]
TARGET_COLS = ["Target_1Q_ahead", "Target_2Q_ahead", "Target_3Q_ahead"]

# Engineered feature set (38 columns = 6 signals × (level + 3 lags + 2 rolling
# means) + 2 structural-break dummies). Built from the untransformed macro
# source. TEDRATE is deliberately excluded because of extensive pre-1986
# missingness that would otherwise drop the 1985-89 warm-up window.
ENGINEERED_SIGNAL_COLS = [
    "T10Y2Y",
    "BAA10Y",
    "UNRATE",
    "INDPRO",
    "CPIAUCSL",
    "FEDFUNDS",
]
ENGINEERED_LAGS = [1, 2, 3]
ENGINEERED_ROLL_WINDOWS = [3, 6]


def aggregate_to_quarterly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Take the last observation inside each quarter and re-stamp to quarter-start.

    The baseline dataset is indexed on quarter-start dates (1986-01-01,
    1986-04-01, ...). Our network features are month-end. For each quarter
    we take the March/June/September/December value and move the index to
    the corresponding quarter-start so the merge lines up.
    """
    q_end = monthly.resample("QE").last()
    q_end.index = q_end.index.to_period("Q").to_timestamp(how="start")
    return q_end


def load_simple_baseline() -> pd.DataFrame:
    """6-feature untransformed macro levels panel used by Section 6.

    Reads the same ``MACRO_SOURCE_CSV`` the engineered baseline reads and
    returns it verbatim. Callers select ``SIMPLE_FEATURE_COLS`` out of it.
    """
    return load_macro_source()


def load_macro_source() -> pd.DataFrame:
    """Untransformed macro levels — shared source for both baselines."""
    df = (
        pd.read_csv(MACRO_SOURCE_CSV, parse_dates=["Date"])
        .set_index("Date")
        .sort_index()
    )
    return df


def build_engineered_features(
    macro_source: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the 38-feature engineered panel from the untransformed levels.

    For each of the 6 signals, keep the level and add 3 lags and 2
    rolling-window means (3Q and 6Q), then append two structural-break
    dummies (post-2008 and post-2020). Returns (engineered_df, feature_cols).
    """
    edf = macro_source.copy()
    feature_cols: list[str] = []
    for col in ENGINEERED_SIGNAL_COLS:
        feature_cols.append(col)
        for lag in ENGINEERED_LAGS:
            name = f"{col}_lag{lag}"
            edf[name] = edf[col].shift(lag)
            feature_cols.append(name)
        for win in ENGINEERED_ROLL_WINDOWS:
            name = f"{col}_roll{win}"
            edf[name] = edf[col].rolling(win, min_periods=win).mean()
            feature_cols.append(name)
    edf["post_2008"] = (edf.index >= pd.Timestamp("2008-01-01")).astype(int)
    edf["post_2020"] = (edf.index >= pd.Timestamp("2020-01-01")).astype(int)
    feature_cols += ["post_2008", "post_2020"]
    return edf, feature_cols


def build_panel_simple(network_features: pd.DataFrame) -> pd.DataFrame:
    """Merge quarterly-aggregated network features onto the simple 6-feature panel.

    The simple baseline takes the 6 untransformed signals as levels (no lags,
    no rolling means), joins quarterly network features, and drops any row
    missing one of the 6 signals, the 5 network features, the 3 targets, or
    the contemporaneous NBER flag.
    """
    base = load_simple_baseline()
    net_q = aggregate_to_quarterly(network_features)
    panel = base.join(net_q, how="inner")
    panel = panel.dropna(
        subset=SIMPLE_FEATURE_COLS + NETWORK_COLS + TARGET_COLS + ["USRECD"]
    )
    return panel


def build_panel_engineered(
    network_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the engineered 38-feature panel merged with network features.

    Reads the untransformed macro levels, runs ``build_engineered_features``,
    inner-joins against quarterly-aggregated network features, and drops rows
    missing any engineered / network / target column. Returns
    ``(panel, engineered_cols)`` so callers can reference the engineered
    columns by name without re-deriving them.
    """
    source = load_macro_source()
    engineered, engineered_cols = build_engineered_features(source)
    net_q = aggregate_to_quarterly(network_features)
    panel = engineered.join(net_q, how="inner")
    panel = panel.dropna(
        subset=engineered_cols + NETWORK_COLS + TARGET_COLS + ["USRECD"]
    )
    return panel, engineered_cols


def _fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    model_type: str,
    C: float = 1.0,
    class_weight: str | None = None,
    max_iter: int = 2000,
    impute: bool = False,
) -> np.ndarray:
    if model_type == "logistic":
        if impute:
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_tr)
            X_te = imp.transform(X_te)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_tr)
        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
        )
        # Guard against folds where the training y is all one class.
        if len(np.unique(y_tr)) < 2:
            return np.full(len(X_te), float(y_tr.mean()))
        clf.fit(Xs, y_tr)
        return clf.predict_proba(scaler.transform(X_te))[:, 1]
    if model_type == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        if len(np.unique(y_tr)) < 2:
            return np.full(len(X_te), float(y_tr.mean()))
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
        )
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]
    raise ValueError(f"unknown model_type {model_type!r}")


def walk_forward_eval(
    X: pd.DataFrame,
    y: pd.Series,
    initial: int = 60,
    step: int = 4,
    model_type: str = "logistic",
    C: float = 1.0,
    class_weight: str | None = None,
    test: int | None = None,
    reject_partial: bool = False,
    impute: bool = False,
) -> tuple[float, pd.DataFrame]:
    """Expanding-window walk-forward. Returns pooled AUC + per-fold predictions.

    At each fold the model is refit on all history up to the fold boundary
    and asked to score the next ``step`` quarters. Predictions from every
    fold are concatenated and scored with a single pooled AUC — this is
    more stable than averaging per-fold AUCs when folds only contain a
    handful of recession quarters.
    """
    X = X.sort_index()
    y = y.reindex(X.index)
    n = len(X)
    test_size = test if test is not None else step
    fold_frames: list[pd.DataFrame] = []
    fold_id = 0
    start = initial
    while start < n:
        end = start + test_size
        if end > n:
            if reject_partial:
                break
            end = n
        X_tr = X.iloc[:start].to_numpy(dtype=float)
        y_tr = y.iloc[:start].to_numpy(dtype=int)
        X_te = X.iloc[start:end].to_numpy(dtype=float)
        p = _fit_predict(
            X_tr,
            y_tr,
            X_te,
            model_type,
            C=C,
            class_weight=class_weight,
            impute=impute,
        )
        fold_frames.append(
            pd.DataFrame(
                {
                    "pred": p,
                    "y": y.iloc[start:end].to_numpy(dtype=int),
                    "fold_id": fold_id,
                    "train_end": X.index[start - 1],
                },
                index=X.index[start:end],
            )
        )
        fold_id += 1
        start += step
    all_pred = pd.concat(fold_frames)
    y_true = all_pred["y"].to_numpy(dtype=int)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, all_pred["pred"].to_numpy())
    return float(auc), all_pred


def run_walkforward_simple(
    panel: pd.DataFrame,
    initial: int = 60,
    step: int = 4,
    models: tuple[str, ...] = ("logistic", "xgboost"),
) -> pd.DataFrame:
    """Run baseline_simple / network / combined_simple × 3 horizons × 2 models."""
    variants = {
        "baseline_simple": SIMPLE_FEATURE_COLS,
        "network": NETWORK_COLS,
        "combined_simple": SIMPLE_FEATURE_COLS + NETWORK_COLS,
    }
    rows = []
    for horizon, tgt in zip([1, 2, 3], TARGET_COLS):
        y = panel[tgt].astype(int)
        for variant, cols in variants.items():
            X = panel[cols]
            for model in models:
                auc, _ = walk_forward_eval(
                    X, y, initial=initial, step=step, model_type=model
                )
                rows.append(
                    {
                        "horizon_q": horizon,
                        "variant": variant,
                        "model": model,
                        "auc": auc,
                    }
                )
    return pd.DataFrame(rows)
