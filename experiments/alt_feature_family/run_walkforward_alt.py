"""Alt-feature walk-forward: engineered baseline vs alt-only vs combined_alt.

Exactly matches the Section 7 protocol:
    walk_forward_eval(initial=60, step=8, test=8, reject_partial=True,
                      impute=True, model_type="logistic",
                      C=0.25, class_weight="balanced")

Three variants per horizon:
    baseline_engineered   — 38 engineered macro features only (Section 7 replica)
    alt_only              — 5 alt features only
    combined_alt          — 38 engineered + 5 alt

Outputs:
    experiments/alt_feature_family/results/walkforward_alt.parquet
        columns: horizon_q, variant, model, auc
    experiments/alt_feature_family/results/preds_cache.parquet
        all per-date predictions, keyed by (horizon_q, variant) — used by the
        paired-bootstrap script.

Also verifies the baseline_engineered AUCs match the existing Section 7 parquet
to 3 decimal places. If they don't, prints an explicit FAIL marker.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.modeling import (  # noqa: E402
    TARGET_COLS,
    aggregate_to_quarterly,
    build_panel_engineered,
    walk_forward_eval,
)


PROCESSED = ROOT / "data" / "processed"
ALT_DIR = ROOT / "experiments" / "alt_feature_family"
ALT_PARQUET = ALT_DIR / "data" / "alt_features.parquet"
WF_OUT = ALT_DIR / "results" / "walkforward_alt.parquet"
PREDS_OUT = ALT_DIR / "results" / "preds_cache.parquet"

EWMA_FEATURES = PROCESSED / "04_network_features_ewma.parquet"
ENGINEERED_REF = PROCESSED / "07_walkforward_engineered.parquet"

# Match Section 7 exactly.
INITIAL = 60
STEP = 8
TEST = 8
C = 0.25
CLASS_WEIGHT = "balanced"

ALT_COLS = [
    "corr_std",
    "corr_skewness",
    "corr_kurtosis",
    "pmfg_sum_sq_corr",
    "pmfg_separators_cliques_ratio",
]
MOMENT_COLS = ["corr_std", "corr_skewness", "corr_kurtosis"]


def build_panel() -> tuple[pd.DataFrame, list[str]]:
    """Build the engineered panel and merge in the monthly alt features."""
    print("Loading cached EWMA network features (for panel index alignment)...")
    net_feats = pd.read_parquet(EWMA_FEATURES)
    print(f"  net_feats shape: {net_feats.shape}")

    print("Building Section 7 engineered panel...")
    base_panel, engineered_cols = build_panel_engineered(net_feats)
    print(f"  base panel shape: {base_panel.shape}")
    print(f"  engineered cols:  {len(engineered_cols)}")
    print(f"  panel range:      {base_panel.index.min().date()} → {base_panel.index.max().date()}")

    print("Loading alt features...")
    alt_monthly = pd.read_parquet(ALT_PARQUET)
    print(f"  alt_monthly shape: {alt_monthly.shape}")
    print(f"  alt cols: {list(alt_monthly.columns)}")
    alt_q = aggregate_to_quarterly(alt_monthly)
    print(f"  alt quarterly shape: {alt_q.shape}")

    panel = base_panel.join(alt_q[ALT_COLS], how="left")
    pre_drop_len = len(panel)
    missing_mask = panel[ALT_COLS].isna().any(axis=1)
    if missing_mask.any():
        missing_dates = panel.index[missing_mask].tolist()
        print(f"  WARN — dropping {missing_mask.sum()} rows with NaN alt features:")
        for d in missing_dates[:6]:
            print(f"    {d.date()}")
        if len(missing_dates) > 6:
            print(f"    … and {len(missing_dates) - 6} more")
    panel = panel.loc[~missing_mask].copy()
    print(f"  final panel shape: {panel.shape} (was {pre_drop_len})")

    return panel, engineered_cols


def run_all(
    panel: pd.DataFrame,
    engineered_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward for all 3 variants × 3 horizons. Return (results, preds)."""
    variants = {
        "baseline_engineered": engineered_cols,
        "alt_only": ALT_COLS,
        "combined_alt": engineered_cols + ALT_COLS,
        "moments_only": MOMENT_COLS,
        "combined_moments": engineered_cols + MOMENT_COLS,
    }

    rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    for horizon, tgt in zip([1, 2, 3], TARGET_COLS):
        y = panel[tgt].astype(int)
        for variant, cols in variants.items():
            X = panel[cols]
            auc, per_fold = walk_forward_eval(
                X,
                y,
                initial=INITIAL,
                step=STEP,
                test=TEST,
                reject_partial=True,
                impute=True,
                model_type="logistic",
                C=C,
                class_weight=CLASS_WEIGHT,
            )
            rows.append(
                {
                    "horizon_q": horizon,
                    "variant": variant,
                    "model": "logistic",
                    "auc": auc,
                }
            )
            print(f"  {tgt:20s} {variant:22s}  AUC = {auc:.4f}")
            per_fold = per_fold.reset_index().rename(columns={"index": "date"})
            per_fold["horizon_q"] = horizon
            per_fold["variant"] = variant
            pred_frames.append(per_fold)
    results = pd.DataFrame(rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    return results, preds


def verify_baseline(results: pd.DataFrame) -> None:
    """Compare our baseline_engineered AUCs against the Section 7 parquet."""
    print()
    print("=== Baseline replication check vs Section 7 ===")
    ref = pd.read_parquet(ENGINEERED_REF)
    ref_base = (
        ref[
            (ref["variant"] == "baseline_engineered")
            & (ref["model"] == "logistic")
        ]
        .set_index("horizon_q")["auc"]
    )
    ours = (
        results[
            (results["variant"] == "baseline_engineered")
            & (results["model"] == "logistic")
        ]
        .set_index("horizon_q")["auc"]
    )
    ok = True
    for h in [1, 2, 3]:
        a, b = float(ref_base.loc[h]), float(ours.loc[h])
        match = round(a, 3) == round(b, 3)
        marker = "OK" if match else "FAIL"
        print(f"  horizon {h}Q  section7={a:.4f}  ours={b:.4f}  [{marker}]")
        ok = ok and match
    if not ok:
        raise SystemExit(
            "baseline_engineered does not match Section 7 to 3dp — "
            "the comparison is invalid. STOP."
        )
    print("  Baseline replication OK ✓")


def main() -> None:
    panel, engineered_cols = build_panel()
    print()
    print("Running walk-forward (logistic, C=0.25) on 3 variants × 3 horizons...")
    results, preds = run_all(panel, engineered_cols)
    verify_baseline(results)

    WF_OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(WF_OUT)
    preds.to_parquet(PREDS_OUT, index=False)
    print()
    print(f"Saved → {WF_OUT.relative_to(ROOT)}")
    print(f"Saved → {PREDS_OUT.relative_to(ROOT)}")

    # Pivoted view
    pivot = results.pivot_table(
        index="horizon_q", columns="variant", values="auc"
    )
    pivot["delta_comb_vs_base"] = (
        pivot["combined_alt"] - pivot["baseline_engineered"]
    )
    pivot = pivot[
        ["baseline_engineered", "alt_only", "combined_alt", "delta_comb_vs_base"]
    ]
    print()
    print("=== ALT-FEATURE AUC COMPARISON ===")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
