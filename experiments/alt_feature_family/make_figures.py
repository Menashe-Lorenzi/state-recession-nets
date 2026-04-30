"""Alt-feature family figures: alt-feature trajectories + 3-variant AUC bar chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ALT_DIR = ROOT / "experiments" / "alt_feature_family"
ALT_PARQUET = ALT_DIR / "data" / "alt_features.parquet"
WF = ALT_DIR / "results" / "walkforward_alt.parquet"
FIG_DIR = ALT_DIR / "figures"
USREC_PARQUET = ROOT / "data" / "raw" / "nber_usrec.parquet"

ALT_COLS = [
    "corr_std",
    "corr_skewness",
    "corr_kurtosis",
    "pmfg_sum_sq_corr",
    "pmfg_separators_cliques_ratio",
]

PRETTY = {
    "corr_std": "σ(ρ) — cross-sectional std of correlations",
    "corr_skewness": "skew(ρ) — cross-sectional skewness",
    "corr_kurtosis": "exc. kurtosis(ρ)",
    "pmfg_sum_sq_corr": "Σρ² on PMFG edges",
    "pmfg_separators_cliques_ratio": "|min node cuts| / |triangles| (PMFG)",
}


def load_usrec() -> pd.DataFrame:
    if USREC_PARQUET.exists():
        df = pd.read_parquet(USREC_PARQUET)
        if "date" in df.columns:
            df = df.set_index("date")
        return df
    # Fallback: try CSV
    csv = ROOT / "data" / "raw" / "USREC.csv"
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["observation_date"])
        df = df.rename(columns={"observation_date": "date"}).set_index("date")
        return df
    return None


def recession_spans(usrec: pd.DataFrame, start, end) -> list[tuple]:
    if usrec is None:
        return []
    col = "USREC" if "USREC" in usrec.columns else usrec.columns[0]
    s = usrec[col].loc[start:end]
    spans: list[tuple] = []
    in_rec = False
    t0 = None
    for d, v in s.items():
        if v == 1 and not in_rec:
            in_rec = True
            t0 = d
        elif v == 0 and in_rec:
            in_rec = False
            spans.append((t0, d))
    if in_rec:
        spans.append((t0, s.index[-1]))
    return spans


def plot_trajectories() -> None:
    df = pd.read_parquet(ALT_PARQUET)
    usrec = load_usrec()
    spans = recession_spans(usrec, df.index.min(), df.index.max())

    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    for ax, col in zip(axes, ALT_COLS):
        ax.plot(df.index, df[col], color="#1f4e79", lw=1.2)
        for s0, s1 in spans:
            ax.axvspan(s0, s1, color="grey", alpha=0.18, lw=0)
        ax.set_ylabel(PRETTY.get(col, col), fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Alt network features (EWMA hl=24), NBER shaded")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    out = FIG_DIR / "alt_features_trajectories.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out.relative_to(ROOT)}")


def plot_auc_comparison() -> None:
    wf = pd.read_parquet(WF)
    pivot = wf.pivot_table(index="horizon_q", columns="variant", values="auc")
    order = ["baseline_engineered", "alt_only", "combined_alt"]
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(3)
    width = 0.26
    colors = {"baseline_engineered": "#606c6c", "alt_only": "#c66b27", "combined_alt": "#2f7a5a"}
    for i, variant in enumerate(order):
        ax.bar(x + (i - 1) * width, pivot[variant].values, width,
               label=variant, color=colors[variant])
    ax.set_xticks(x)
    ax.set_xticklabels(["1Q", "2Q", "3Q"])
    ax.set_ylabel("Pooled OOS AUC")
    ax.set_ylim(0.4, 0.9)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Alt features — engineered baseline vs alt-only vs combined_alt (sparse-L1 logistic)")
    for i, variant in enumerate(order):
        for j, v in enumerate(pivot[variant].values):
            ax.text(j + (i - 1) * width, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "alt_feature_auc_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out.relative_to(ROOT)}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_trajectories()
    plot_auc_comparison()


if __name__ == "__main__":
    main()
