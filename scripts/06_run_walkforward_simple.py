"""Section 6 driver: walk-forward evaluation of the simple 6-feature
untransformed macro baseline vs. the 5 network features vs. combined,
across the three correlation estimators.

Writes:
  data/processed/06_walkforward_simple.parquet  — estimator × horizon × variant × model
  reports/tables/06_auc_table.csv                — same, pivoted for the report
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.modeling import build_panel_simple, run_walkforward_simple


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


ESTIMATORS = {
    "Pearson_w60": PROCESSED / "04_network_features_pearson.parquet",
    "Spearman_w60": PROCESSED / "04_network_features_spearman.parquet",
    "EWMA_hl24": PROCESSED / "04_network_features_ewma.parquet",
}


def main() -> None:
    all_results = []
    for name, path in ESTIMATORS.items():
        print(f"\n=== {name} ===")
        feats = pd.read_parquet(path)
        panel = build_panel_simple(feats)
        print(f"  panel shape: {panel.shape}  range: {panel.index.min().date()} → {panel.index.max().date()}")
        print(f"  class balance USRECD: {panel['USRECD'].mean():.3f}")
        res = run_walkforward_simple(panel, initial=60, step=4)
        res.insert(0, "estimator", name)
        all_results.append(res)
        print(res.to_string(index=False))

    full = pd.concat(all_results, ignore_index=True)
    full.to_parquet(PROCESSED / "06_walkforward_simple.parquet")

    pivot = full.pivot_table(
        index=["estimator", "model", "horizon_q"],
        columns="variant",
        values="auc",
    )
    pivot["delta_net_vs_base"] = pivot["network"] - pivot["baseline_simple"]
    pivot["delta_comb_vs_base"] = pivot["combined_simple"] - pivot["baseline_simple"]
    pivot = pivot[[
        "baseline_simple",
        "network",
        "combined_simple",
        "delta_net_vs_base",
        "delta_comb_vs_base",
    ]]
    pivot.to_csv(TABLES / "06_auc_table.csv")
    print("\n=== HEADLINE TABLE ===")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
