"""Section 4 plotting — EWMA network feature trajectories and MST snapshots.

Single entry point for the two Section 4 figure scripts previously split
across plot_features.py and plot_mst_snapshots.py. Produces both outputs
in the same run; each plot function is independent so they can be called
individually from a notebook if needed.

Uses the EWMA estimator (halflife = 24 months) throughout, because it is
the best-performing estimator per the Section 5c walk-forward. The
aggregate "US" node is dropped from the MST snapshots — it mechanically
correlates with every state and would hub the tree meaninglessly.

Writes:
  figures/04_features_polished.png   (5-panel + zoom network-feature chart)
  figures/04_mst_snapshots.png       (geographic MST at 4 key dates)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.networks import corr_to_mst  # noqa: E402
from src.plotting import shade_recessions  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"


# ============================================================================
# Network-feature trajectories  (was plot_features.py)
# ============================================================================

FEATURE_ORDER = [
    "mean_corr",
    "largest_eigenvalue",
    "mst_length",
    "network_density",
    "n_communities",
]
FEATURE_LABELS = {
    "mean_corr": "Mean off-diagonal correlation",
    "largest_eigenvalue": "Largest eigenvalue (λ₁)",
    "mst_length": "MST length",
    "network_density": "Network density (|ρ|>0.5)",
    "n_communities": "Louvain community count",
}


def plot_feature_trajectories() -> Path:
    feats = pd.read_parquet(PROCESSED / "04_network_features_ewma.parquet")
    usrec = pd.read_parquet(ROOT / "data" / "raw" / "nber_usrec.parquet")
    if isinstance(usrec, pd.DataFrame):
        usrec = usrec.iloc[:, 0]
    usrec = usrec.reindex(feats.index, method="nearest")
    zoom_end = pd.Timestamp("2020-01-01")

    fig, axes = plt.subplots(
        len(FEATURE_ORDER), 2, figsize=(13, 11), sharex="col",
    )
    for i, col in enumerate(FEATURE_ORDER):
        ax_full = axes[i, 0]
        ax_zoom = axes[i, 1]
        s = feats[col]
        ax_full.plot(s.index, s.values, color="#333333", lw=1.2)
        shade_recessions(ax_full, usrec)
        ax_full.set_ylabel(FEATURE_LABELS[col], fontsize=9)
        ax_full.grid(True, alpha=0.3)
        ax_full.set_title("full sample" if i == 0 else "", fontsize=9)

        s_zoom = s.loc[:zoom_end]
        ax_zoom.plot(s_zoom.index, s_zoom.values, color="#333333", lw=1.2)
        shade_recessions(ax_zoom, usrec.loc[:zoom_end])
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.set_title("pre-2020 zoom" if i == 0 else "", fontsize=9)

    axes[-1, 0].set_xlabel("date")
    axes[-1, 1].set_xlabel("date")
    fig.suptitle(
        "Section 4 — EWMA (halflife=24m) network features with NBER shading",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out = FIGURES / "04_features_polished.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ============================================================================
# Geographic MST snapshots  (was plot_mst_snapshots.py)
# ============================================================================

# Approximate state centroids (lat, lon). Alaska and Hawaii are re-positioned
# to the bottom-left of the canvas for compact plotting.
STATE_COORDS = {
    "AL": (32.8, -86.8),  "AK": (26.0, -119.0),
    "AZ": (34.3, -111.7), "AR": (34.9, -92.4),
    "CA": (37.2, -119.7), "CO": (39.0, -105.5),
    "CT": (41.6, -72.7),  "DE": (39.0, -75.5),
    "FL": (28.6, -82.5),  "GA": (32.7, -83.4),
    "HI": (24.0, -115.0),
    "ID": (44.4, -114.6), "IL": (40.0, -89.2),
    "IN": (39.9, -86.3),  "IA": (42.1, -93.5),
    "KS": (38.5, -98.4),  "KY": (37.5, -85.3),
    "LA": (31.0, -91.8),  "ME": (45.4, -69.2),
    "MD": (39.1, -76.8),  "MA": (42.3, -71.8),
    "MI": (44.3, -85.4),  "MN": (46.3, -94.3),
    "MS": (32.7, -89.7),  "MO": (38.4, -92.5),
    "MT": (46.9, -110.4), "NE": (41.5, -99.8),
    "NV": (39.3, -116.6), "NH": (43.7, -71.6),
    "NJ": (40.2, -74.6),  "NM": (34.4, -106.1),
    "NY": (42.9, -75.5),  "NC": (35.6, -79.4),
    "ND": (47.5, -100.3), "OH": (40.3, -82.8),
    "OK": (35.6, -97.5),  "OR": (44.0, -120.5),
    "PA": (40.9, -77.8),  "RI": (41.7, -71.5),
    "SC": (33.9, -80.9),  "SD": (44.4, -100.2),
    "TN": (35.9, -86.4),  "TX": (31.5, -99.3),
    "UT": (39.3, -111.7), "VT": (44.1, -72.7),
    "VA": (37.5, -78.9),  "WA": (47.4, -120.5),
    "WV": (38.6, -80.6),  "WI": (44.3, -89.6),
    "WY": (43.0, -107.6),
}

SNAPSHOT_DATES = [
    pd.Timestamp("2007-01-31"),
    pd.Timestamp("2009-01-31"),
    pd.Timestamp("2019-12-31"),
    pd.Timestamp("2020-06-30"),
]
SNAPSHOT_TITLES = [
    "2007-01  (GFC run-up, pre-peak)",
    "2009-01  (inside GFC)",
    "2019-12  (calm, pre-COVID)",
    "2020-06  (COVID peak co-movement)",
]


def _load_ewma_cube():
    d = np.load(PROCESSED / "03_rolling_corr_ewma.npz", allow_pickle=True)
    return d["cubes"], pd.DatetimeIndex(d["dates"]), list(d["cols"])


def _nearest_date(target: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    pos = index.get_indexer([target], method="nearest")[0]
    return index[pos]


def _plot_mst_snapshot(ax, corr, cols, title):
    keep = [i for i, c in enumerate(cols) if c != "US"]
    C = corr[np.ix_(keep, keep)]
    names = [cols[i] for i in keep]
    mst = corr_to_mst(C)

    coords = {n: STATE_COORDS[n] for n in names if n in STATE_COORDS}
    xs = [coords[n][1] for n in coords]
    ys = [coords[n][0] for n in coords]

    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            if mst[i, j] > 0 and a in coords and b in coords:
                rho = C[i, j]
                lw = 0.3 + 2.5 * max(0.0, rho)
                alpha = 0.35 + 0.55 * max(0.0, rho)
                ax.plot(
                    [coords[a][1], coords[b][1]],
                    [coords[a][0], coords[b][0]],
                    color=plt.cm.viridis(max(0.0, rho)),
                    lw=lw, alpha=alpha, zorder=1,
                )

    ax.scatter(xs, ys, s=55, c="white", edgecolors="black", lw=0.6, zorder=2)
    for n, (lat, lon) in coords.items():
        ax.annotate(n, (lon, lat), fontsize=6, ha="center", va="center", zorder=3)

    mean_off = (C.sum() - len(C)) / (len(C) * (len(C) - 1))
    mst_len = mst.sum() / 2
    ax.set_title(
        f"{title}\nmean corr {mean_off:.2f}    MST len {mst_len:.1f}",
        fontsize=10,
    )
    ax.set_xlim(-125, -65)
    ax.set_ylim(22, 50)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_mst_snapshots() -> Path:
    cubes, dates, cols = _load_ewma_cube()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, target, title in zip(axes.flat, SNAPSHOT_DATES, SNAPSHOT_TITLES):
        actual = _nearest_date(target, dates)
        pos = dates.get_loc(actual)
        _plot_mst_snapshot(ax, cubes[pos], cols, title)

    fig.suptitle(
        "Section 4 — EWMA (halflife=24m) MST of US state coincident returns",
        fontsize=13, y=1.00,
    )
    cbar_ax = fig.add_axes([0.25, -0.02, 0.5, 0.018])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("edge correlation (ρ)", fontsize=9)
    fig.tight_layout()
    out = FIGURES / "04_mst_snapshots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    plot_feature_trajectories()
    plot_mst_snapshots()


if __name__ == "__main__":
    main()
