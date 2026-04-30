"""Plot Section 6 AUC comparison across estimators, horizons, and variants."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "processed" / "06_walkforward_simple.parquet"
FIG = ROOT / "figures" / "06_auc_comparison.png"


def main() -> None:
    df = pd.read_parquet(RES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    variants = ["baseline_simple", "network", "combined_simple"]
    labels = {"baseline_simple": "baseline", "network": "network", "combined_simple": "combined"}
    colors = {"baseline_simple": "#4C72B0", "network": "#DD8452", "combined_simple": "#55A868"}
    estimators = ["Pearson_w60", "Spearman_w60", "EWMA_hl24"]

    for ax, model in zip(axes, ["logistic", "xgboost"]):
        sub = df[df["model"] == model]
        # Group bars by horizon, one cluster per estimator×variant.
        x_labels = []
        x = []
        y = []
        c = []
        pos = 0
        gap = 0.3
        for h in [1, 2, 3]:
            for est in estimators:
                for v in variants:
                    row = sub[
                        (sub["horizon_q"] == h)
                        & (sub["estimator"] == est)
                        & (sub["variant"] == v)
                    ]
                    if row.empty:
                        continue
                    x.append(pos)
                    y.append(row["auc"].values[0])
                    c.append(colors[v])
                    pos += 1
                pos += gap
                x_labels.append((pos - gap - len(variants) / 2, f"{est}\n{h}Q"))
            pos += gap * 2
        ax.bar(x, y, color=c, edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
        ax.set_ylim(0, 1)
        ax.set_title(f"{model}")
        ax.set_ylabel("pooled out-of-sample AUC")
        ax.set_xticks([lx for lx, _ in x_labels])
        ax.set_xticklabels([lb for _, lb in x_labels], fontsize=7, rotation=0)
        ax.grid(True, axis="y", alpha=0.3)
    # Single legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[v], ec="black") for v in variants
    ]
    axes[0].legend(handles, [labels[v] for v in variants], loc="upper right", framealpha=0.9)
    fig.suptitle("Section 6 — Walk-forward AUC: baseline vs network vs combined", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
