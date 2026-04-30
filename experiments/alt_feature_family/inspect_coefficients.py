"""Full-panel sparse-L1 logistic fit to inspect which alt features the model keeps.

Mirrors the Section 8 coefficient-inspection step. Fits the combined_alt feature
set (38 engineered + 5 alt) on the full Section 7 restricted panel at 3Q horizon
with C=0.25, class_weight="balanced", and prints all non-zero coefficients by
|weight|. Flags whether any alt feature survives and, if so, which macro lags
it substitutes for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.modeling import (  # noqa: E402
    aggregate_to_quarterly,
    build_panel_engineered,
)


PROCESSED = ROOT / "data" / "processed"
ALT_DIR = ROOT / "experiments" / "alt_feature_family"
ALT_PARQUET = ALT_DIR / "data" / "alt_features.parquet"
EWMA_FEATURES = PROCESSED / "04_network_features_ewma.parquet"

ALT_COLS = [
    "corr_std",
    "corr_skewness",
    "corr_kurtosis",
    "pmfg_sum_sq_corr",
    "pmfg_separators_cliques_ratio",
]


def main() -> None:
    net_feats = pd.read_parquet(EWMA_FEATURES)
    base_panel, engineered_cols = build_panel_engineered(net_feats)

    alt_monthly = pd.read_parquet(ALT_PARQUET)
    alt_q = aggregate_to_quarterly(alt_monthly)
    panel = base_panel.join(alt_q[ALT_COLS], how="left")
    panel = panel.dropna(subset=ALT_COLS)

    feature_cols = engineered_cols + ALT_COLS
    X = panel[feature_cols].to_numpy(dtype=float)
    y = panel["Target_3Q_ahead"].astype(int).to_numpy()

    imp = SimpleImputer(strategy="median")
    X_i = imp.fit_transform(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_i)

    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.25,
        class_weight="balanced",
        max_iter=5000,
    )
    clf.fit(Xs, y)
    coef = pd.Series(clf.coef_.ravel(), index=feature_cols)

    nz = coef[coef.abs() > 1e-10].sort_values(key=lambda s: s.abs(), ascending=False)
    print(f"Panel shape for fit: {X.shape}  (rows × features)")
    print(f"Full-panel C=0.25 3Q sparse-L1 fit — {len(nz)} of {len(feature_cols)} features kept:")
    print()
    for name, w in nz.items():
        marker = "  ← ALT" if name in ALT_COLS else ""
        print(f"  {name:35s}  {w:+.4f}{marker}")

    kept_alt = [c for c in ALT_COLS if c in nz.index]
    print()
    print(
        f"Alt features kept: {len(kept_alt)} of {len(ALT_COLS)}"
        + (f"  → {kept_alt}" if kept_alt else "")
    )


if __name__ == "__main__":
    main()
