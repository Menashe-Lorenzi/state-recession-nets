"""Section 9 — descriptive recession typology (MST fingerprint + PMFG snapshots).

Single entry point for the two Section 9 runs previously split across
09_build_recession_typology.py and 09_pmfg_typology.py. All outputs are
bit-for-bit identical to the old set.

Two steps, in order:

  STEP 1  (MST)        Build the fingerprint table (6 metrics × 4 recessions)
                       from the EWMA hl=24 network features, and emit the
                       per-recession panel figure, the z-scored heatmap, and
                       the markdown typology note.

  STEP 2  (PMFG)       For 8 snapshot dates (pre + during for each of the 4
                       recessions), build the Planar Maximally Filtered Graph
                       on the EWMA correlation matrix and compute clustering /
                       triangles / 4-cliques / Louvain community-size metrics.
                       Append those columns to the typology CSV written in
                       step 1. Run a centrality-stability check across hl =
                       18/24/30 and emit the top-5 centrality tables + the two
                       PMFG figures (geographic + force-directed).

Writes:
  reports/tables/09_recession_typology.csv          (MST metrics + PMFG cols)
  reports/tables/09_pmfg_centrality_top5.csv
  reports/tables/09_pmfg_centrality_robust.csv
  figures/09_typology_panel.png
  figures/09_typology_heatmap.png
  figures/09_pmfg_geographic.png
  figures/09_pmfg_forcedirected.png
  reports/recession_typology.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from networkx.algorithms.community import louvain_communities

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.networks import ewma_corr_cube  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# NBER US business-cycle reference dates (peak = start, trough = end).
RECESSIONS: dict[str, dict[str, pd.Timestamp]] = {
    "1990": {"peak": pd.Timestamp("1990-07-01"), "trough": pd.Timestamp("1991-03-01")},
    "2001": {"peak": pd.Timestamp("2001-03-01"), "trough": pd.Timestamp("2001-11-01")},
    "2008": {"peak": pd.Timestamp("2007-12-01"), "trough": pd.Timestamp("2009-06-01")},
    "2020": {"peak": pd.Timestamp("2020-02-01"), "trough": pd.Timestamp("2020-04-01")},
}

PRE_MONTHS = 18
POST_TROUGH_MONTHS = 6


# ============================================================================
# STEP 1 — MST-based fingerprint, panel, heatmap, markdown
# ============================================================================

def _months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)


def _slice(feats: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    return feats[(feats.index >= lo) & (feats.index <= hi)]


def _nearest_value(feats: pd.DataFrame, date: pd.Timestamp, col: str) -> float:
    idx = feats.index.get_indexer([date], method="nearest")[0]
    return float(feats[col].iloc[idx])


def fingerprint(feats: pd.DataFrame, peak: pd.Timestamp, trough: pd.Timestamp) -> dict:
    window_lo = peak - pd.DateOffset(months=PRE_MONTHS)
    window_hi = trough + pd.DateOffset(months=POST_TROUGH_MONTHS)
    win = _slice(feats, window_lo, window_hi)
    pre = _slice(feats, window_lo, peak)
    rise_win = _slice(feats, window_lo, peak + pd.DateOffset(months=6))

    peak_mean_corr = float(win["mean_corr"].max())
    min_mst_length = float(win["mst_length"].min())
    mean_corr_at_peak = _nearest_value(feats, peak, "mean_corr")

    corr_peak_date = win["mean_corr"].idxmax()
    lead_months = _months_between(corr_peak_date, peak)  # negative ⇒ leads NBER

    pre_disp = float(pre["mean_corr"].std())

    corr_min_date = rise_win["mean_corr"].idxmin()
    collapse_speed = _months_between(corr_peak_date, corr_min_date)

    return {
        "peak_mean_corr": round(peak_mean_corr, 4),
        "min_mst_length": round(min_mst_length, 3),
        "mean_corr_at_nber_peak": round(mean_corr_at_peak, 4),
        "lead_months_to_corr_peak": int(lead_months),
        "pre_window_dispersion": round(pre_disp, 4),
        "corr_collapse_speed": int(collapse_speed),
    }


def build_fingerprint_table(feats: pd.DataFrame) -> pd.DataFrame:
    rows = {
        name: fingerprint(feats, d["peak"], d["trough"])
        for name, d in RECESSIONS.items()
    }
    df = pd.DataFrame(rows).T
    df.index.name = "recession"
    return df


def plot_panel(feats: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row", constrained_layout=True)
    mc_vals: list[float] = []
    mst_vals: list[float] = []
    segs = {}
    for name, d in RECESSIONS.items():
        lo = d["peak"] - pd.DateOffset(months=PRE_MONTHS)
        hi = d["trough"] + pd.DateOffset(months=POST_TROUGH_MONTHS)
        seg = _slice(feats, lo, hi)
        segs[name] = seg
        mc_vals.extend(seg["mean_corr"].tolist())
        mst_vals.extend(seg["mst_length"].tolist())

    mc_min, mc_max = min(mc_vals), max(mc_vals)
    mst_min, mst_max = min(mst_vals), max(mst_vals)
    mc_pad = 0.03 * (mc_max - mc_min)
    mst_pad = 0.03 * (mst_max - mst_min)

    for col_idx, (name, d) in enumerate(RECESSIONS.items()):
        seg = segs[name]
        peak = d["peak"]
        trough = d["trough"]

        ax_top = axes[0, col_idx]
        ax_top.plot(seg.index, seg["mean_corr"], color="tab:blue", linewidth=1.6)
        ax_top.axvline(peak, color="black", linestyle="--", linewidth=1.0)
        ax_top.axvline(trough, color="black", linestyle=":", linewidth=1.0)
        ax_top.axvspan(peak, trough, color="red", alpha=0.08)
        ax_top.set_title(f"{name} recession", fontsize=11)
        ax_top.set_ylim(mc_min - mc_pad, mc_max + mc_pad)
        ax_top.grid(alpha=0.3)
        ax_top.tick_params(axis="x", rotation=30, labelsize=8)
        if col_idx == 0:
            ax_top.set_ylabel("mean pairwise correlation", fontsize=10)

        ax_bot = axes[1, col_idx]
        ax_bot.plot(seg.index, seg["mst_length"], color="tab:orange", linewidth=1.6)
        ax_bot.axvline(peak, color="black", linestyle="--", linewidth=1.0)
        ax_bot.axvline(trough, color="black", linestyle=":", linewidth=1.0)
        ax_bot.axvspan(peak, trough, color="red", alpha=0.08)
        ax_bot.set_ylim(mst_min - mst_pad, mst_max + mst_pad)
        ax_bot.grid(alpha=0.3)
        ax_bot.tick_params(axis="x", rotation=30, labelsize=8)
        if col_idx == 0:
            ax_bot.set_ylabel("MST length", fontsize=10)

    handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", label="NBER peak"),
        plt.Line2D([0], [0], color="black", linestyle=":", label="NBER trough"),
        plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.2, label="recession"),
    ]
    axes[0, -1].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Recession typology — mean correlation and MST length across NBER cycles",
        fontsize=13, y=1.02,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(fp: pd.DataFrame, out: Path) -> None:
    fp_z = fp.apply(lambda c: (c - c.mean()) / c.std(ddof=0))

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    vmax = max(2.0, float(np.nanmax(np.abs(fp_z.values))))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = ax.imshow(fp_z.values, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_yticks(range(len(fp.index)))
    ax.set_yticklabels(fp.index, fontsize=10)
    ax.set_xticks(range(len(fp.columns)))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in fp.columns],
        rotation=0, ha="center", fontsize=8,
    )

    for i in range(fp.shape[0]):
        for j in range(fp.shape[1]):
            raw = fp.iloc[i, j]
            z = fp_z.iloc[i, j]
            text_color = "white" if abs(z) > 1.1 else "black"
            ax.text(j, i, f"{raw:g}", ha="center", va="center", fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("z-score (per column)", fontsize=9)
    ax.set_title(
        "Recession fingerprint — raw values shown, colour is z-score per column",
        fontsize=11,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


NARRATIVES = {
    "1990": (
        "The 1990 recession (Jul 1990 – Mar 1991) is a null event for the network. "
        "The window maximum of mean correlation (0.53) sits at the very start of "
        "the pre-window (i.e. 18 months before the NBER peak) and only decays from "
        "there; MST length never drops below 33, effectively unchanged from its "
        "calm-period level. The network view simply does not see this recession."
    ),
    "2001": (
        "The 2001 recession (Mar – Nov 2001) shows a delayed and modest response. "
        "At the NBER peak mean correlation is near a local minimum (0.30), and the "
        "window maximum of 0.57 is only reached about 10 months later — well after "
        "the NBER trough. MST length barely shortens (minimum 34.1). The dot-com "
        "recession is a sector-specific event that does not fully propagate across "
        "states."
    ),
    "2008": (
        "The 2008-09 recession (Dec 2007 – Jun 2009) is the canonical slow-burn "
        "credit cycle. Mean correlation rises steadily from 0.36 at the NBER peak "
        "to 0.83 fourteen months later, right at the NBER trough (April 2009). MST "
        "length compresses from calm-period values to a low of 18.7. This is the "
        "archetype the network features were designed to detect."
    ),
    "2020": (
        "The 2020 COVID recession (Feb – Apr 2020) is the extreme synchronisation "
        "shock of the sample. Mean correlation jumps from 0.39 at the NBER peak to "
        "0.99 just two months later (coinciding with the trough), and MST length "
        "collapses from calm-period values around 30 to just 3.2. The entire "
        "transition completes inside the peak-to-trough window. This is a "
        "one-in-a-generation event and is structurally different from the other "
        "three."
    ),
}


def write_markdown(fp: pd.DataFrame, out: Path) -> None:
    def fmt(v: float, nd: int = 3) -> str:
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{v:.{nd}f}"

    lines: list[str] = []
    lines.append("# Recession typology — descriptive comparison\n")
    lines.append(
        "Purely descriptive analysis of the EWMA network features around each "
        "NBER recession in the sample. No modelling, no tests. Window for each "
        f"event: `[peak − {PRE_MONTHS} months, trough + {POST_TROUGH_MONTHS} months]`.\n"
    )

    lines.append("## Fingerprint table\n")
    header = "| recession | " + " | ".join(fp.columns) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(fp.columns)) + " |"
    lines.append(header)
    lines.append(sep)
    for rec, row in fp.iterrows():
        vals = [str(rec)] + [fmt(row[c]) for c in fp.columns]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("## Per-recession description\n")
    for rec in ["1990", "2001", "2008", "2020"]:
        row = fp.loc[rec]
        lines.append(f"### {rec}\n")
        lead = int(row["lead_months_to_corr_peak"])
        lead_phrase = (
            f"{abs(lead)} months before the NBER peak" if lead < 0
            else "at the NBER peak" if lead == 0
            else f"{lead} months after the NBER peak"
        )
        lines.append(
            f"Peak mean correlation in the event window is **{row['peak_mean_corr']:.3f}**, "
            f"reached {lead_phrase}. MST length bottoms out at **{row['min_mst_length']:.2f}** "
            f"(smaller = tighter network). At the NBER peak itself, mean correlation is "
            f"**{row['mean_corr_at_nber_peak']:.3f}**. Pre-window dispersion of mean "
            f"correlation is **{row['pre_window_dispersion']:.3f}**, and the rise "
            f"from minimum to peak within the event window takes "
            f"**{int(row['corr_collapse_speed'])} months**."
        )
        lines.append("")
        lines.append(f"_{NARRATIVES[rec]}_\n")

    lines.append("## Proposed qualitative typology\n")
    lines.append(
        "Based purely on the six-metric fingerprint (no clustering), the four "
        "events fall into distinct qualitative groups:\n"
    )
    lines.append(
        "1. **Null events (1990).** The network signature barely moves. Peak "
        "mean correlation is not meaningfully elevated and the MST does not "
        "collapse. These events are effectively invisible to the network view.\n"
    )
    lines.append(
        "2. **Slow-burn recessions (2008-09).** Mean correlation rises gradually "
        "through the event and reaches its maximum deep inside the recession, "
        "months after the NBER peak. The collapse speed is slow, the "
        "pre-window dispersion is moderate — the network is reacting to a "
        "credit cycle, not anticipating it.\n"
    )
    lines.append(
        "3. **Short sector-shock recessions (2001).** Mean correlation peak is "
        "modest, the MST shortens only a little, and the timing of the peak "
        "sits near but not exactly on the NBER peak. The 2001 event is a "
        "sector-specific (dot-com/tech) shock that does not fully propagate "
        "across states.\n"
    )
    lines.append(
        "4. **Synchronisation shocks (2020).** Mean correlation jumps to >0.9, "
        "the MST collapses to a fraction of its calm-period length, and the "
        "whole thing completes within the peak-to-trough window. 2020 is the "
        "only event in the sample where the cross-sectional structure changes "
        "instantaneously.\n"
    )
    lines.append(
        "\nThe practical implication for the recession-prediction modelling: "
        "network features *describe* the heterogeneous geometry of each recession, "
        "but because the four events are qualitatively different rather than "
        "variations on a common template, a single set of network coefficients "
        "cannot capture all of them — which is consistent with the Section 8 "
        "finding that network features do not improve a properly engineered "
        "macro baseline under sparse logistic regression at this sample size.\n"
    )

    out.write_text("\n".join(lines))


def run_step1_mst() -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("STEP 1 — MST fingerprint, panel, heatmap, markdown")
    print("=" * 72)
    feats = pd.read_parquet(PROCESSED / "04_network_features_ewma.parquet").sort_index()
    print(
        f"Loaded EWMA features: {feats.shape}, "
        f"{feats.index.min().date()} → {feats.index.max().date()}"
    )

    fp = build_fingerprint_table(feats)
    print("\n=== Fingerprint table ===")
    print(fp.to_string())

    fp.to_csv(TABLES / "09_recession_typology.csv")
    print(f"\nWrote {TABLES / '09_recession_typology.csv'}")

    panel_out = FIGURES / "09_typology_panel.png"
    plot_panel(feats, panel_out)
    print(f"Wrote {panel_out}")

    heatmap_out = FIGURES / "09_typology_heatmap.png"
    plot_heatmap(fp, heatmap_out)
    print(f"Wrote {heatmap_out}")

    md_out = REPORTS / "recession_typology.md"
    write_markdown(fp, md_out)
    print(f"Wrote {md_out}")

    return fp


# ============================================================================
# STEP 2 — PMFG snapshots, centrality, figures
# ============================================================================

# 8 snapshots: pre + during for each of the 4 NBER recessions.
DATES = [
    pd.Timestamp("1989-06-30"),   # pre-1990
    pd.Timestamp("1990-10-31"),   # during 1990
    pd.Timestamp("2000-06-30"),   # pre-2001
    pd.Timestamp("2001-07-31"),   # during 2001
    pd.Timestamp("2007-01-31"),   # pre-2008
    pd.Timestamp("2009-01-31"),   # during 2008
    pd.Timestamp("2019-12-31"),   # pre-2020
    pd.Timestamp("2020-06-30"),   # during 2020
]
TITLES = [
    "1989-06  (pre-1990)",
    "1990-10  (inside 1990)",
    "2000-06  (pre-2001, dot-com peak)",
    "2001-07  (inside 2001)",
    "2007-01  (GFC run-up)",
    "2009-01  (inside GFC)",
    "2019-12  (calm, pre-COVID)",
    "2020-06  (COVID peak co-movement)",
]
SNAPSHOT_KEY = [
    "1989-06", "1990-10", "2000-06", "2001-07",
    "2007-01", "2009-01", "2019-12", "2020-06",
]
DURING_TO_RECESSION = {
    "1990-10": 1990,
    "2001-07": 2001,
    "2009-01": 2008,
    "2020-06": 2020,
}

DEFAULT_HL = 24
SENSITIVITY_HLS = [18, 24, 30]
LOUVAIN_SEED = 42

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


def load_hl24_cube() -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    d = np.load(PROCESSED / "03_rolling_corr_ewma.npz", allow_pickle=True)
    return d["cubes"], pd.DatetimeIndex(d["dates"]), list(d["cols"])


def compute_cube(returns: pd.DataFrame, halflife: int):
    return ewma_corr_cube(returns, halflife=halflife)


def matrix_at(cube, dates, target):
    pos = dates.get_indexer([target], method="nearest")[0]
    return cube[pos], dates[pos]


def drop_us_aggregate(corr: np.ndarray, cols: list[str]):
    keep = [i for i, c in enumerate(cols) if c != "US"]
    return corr[np.ix_(keep, keep)], [cols[i] for i in keep]


def build_pmfg(corr: np.ndarray, names: list[str]) -> nx.Graph:
    """Planar Maximally Filtered Graph (Tumminello et al. 2005).

    Sort pairs by Mantegna distance d_ij = sqrt(2*(1 - rho)); add edges in
    ascending order; reject any edge that breaks planarity; stop at 3(N-2).
    """
    n = corr.shape[0]
    target_edges = 3 * (n - 2)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            rho = float(corr[i, j])
            d = float(np.sqrt(2.0 * max(0.0, 1.0 - rho)))
            pairs.append((d, i, j, rho))
    pairs.sort(key=lambda t: t[0])

    g = nx.Graph()
    g.add_nodes_from(names)
    for d, i, j, rho in pairs:
        if g.number_of_edges() >= target_edges:
            break
        a, b = names[i], names[j]
        g.add_edge(a, b, corr=rho, dist=d)
        is_planar, _ = nx.check_planarity(g)
        if not is_planar:
            g.remove_edge(a, b)
    return g


def graph_metrics(g: nx.Graph) -> dict:
    avg_clust = float(nx.average_clustering(g))
    triangles = sum(nx.triangles(g).values()) // 3
    four_cliques = sum(1 for c in nx.find_cliques(g) if len(c) >= 4)
    communities = louvain_communities(g, seed=LOUVAIN_SEED)
    largest_comm = max(len(c) for c in communities) if communities else 0
    return {
        "pmfg_avg_clustering": round(avg_clust, 4),
        "pmfg_triangles": int(triangles),
        "pmfg_4cliques": int(four_cliques),
        "pmfg_largest_community": int(largest_comm),
    }


def centrality_top5(g: nx.Graph) -> pd.DataFrame:
    deg = nx.degree_centrality(g)
    bet = nx.betweenness_centrality(g)
    try:
        eig = nx.eigenvector_centrality_numpy(g)
    except Exception:
        eig = nx.eigenvector_centrality(g, max_iter=1000)
    rows = []
    for label, d in [("degree", deg), ("betweenness", bet), ("eigenvector", eig)]:
        ranked = sorted(d.items(), key=lambda kv: -kv[1])[:5]
        for rank, (state, val) in enumerate(ranked, start=1):
            rows.append({
                "centrality_type": label,
                "rank": rank,
                "state": state,
                "value": round(val, 4),
            })
    return pd.DataFrame(rows)


def communities_dict(g: nx.Graph) -> dict[str, int]:
    comms = louvain_communities(g, seed=LOUVAIN_SEED)
    out: dict[str, int] = {}
    for cid, nodes in enumerate(comms):
        for n in nodes:
            out[n] = cid
    return out


def _edge_style(rho: float):
    rho_pos = max(0.0, rho)
    lw = 0.3 + 2.5 * rho_pos
    alpha = 0.30 + 0.55 * rho_pos
    return lw, alpha


def plot_geographic(ax, g: nx.Graph, title: str, label_states: list[str]):
    coords = {n: STATE_COORDS[n] for n in g.nodes if n in STATE_COORDS}
    for a, b, data in g.edges(data=True):
        if a not in coords or b not in coords:
            continue
        rho = data["corr"]
        lw, alpha = _edge_style(rho)
        ax.plot(
            [coords[a][1], coords[b][1]],
            [coords[a][0], coords[b][0]],
            color="#1f4e79", lw=lw, alpha=alpha, zorder=1,
        )
    xs = [coords[n][1] for n in coords]
    ys = [coords[n][0] for n in coords]
    ax.scatter(xs, ys, s=55, c="white", edgecolors="black", lw=0.6, zorder=2)
    for n, (lat, lon) in coords.items():
        weight = "bold" if n in label_states else "normal"
        size = 7 if n in label_states else 6
        color = "#b22222" if n in label_states else "black"
        ax.annotate(n, (lon, lat), fontsize=size, ha="center", va="center",
                    weight=weight, color=color, zorder=3)

    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()
    annot = (
        "top betweenness: " + ", ".join(label_states)
        if label_states else "top centrality nodes not stable across half-lives"
    )
    ax.set_title(f"{title}\nPMFG  |E|={n_edges}  |V|={n_nodes}    {annot}", fontsize=9)
    ax.set_xlim(-125, -65)
    ax.set_ylim(22, 50)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_forcedirected(ax, g: nx.Graph, title: str, label_states: list[str]):
    pos = nx.spring_layout(g, seed=LOUVAIN_SEED, iterations=200, k=None)
    comms = communities_dict(g)
    n_comm = max(comms.values()) + 1 if comms else 1
    cmap = plt.get_cmap("tab10" if n_comm <= 10 else "tab20")

    for a, b, data in g.edges(data=True):
        rho = data["corr"]
        lw, alpha = _edge_style(rho)
        ax.plot(
            [pos[a][0], pos[b][0]],
            [pos[a][1], pos[b][1]],
            color="#888888", lw=lw, alpha=alpha, zorder=1,
        )

    xs = [pos[n][0] for n in g.nodes]
    ys = [pos[n][1] for n in g.nodes]
    cs = [cmap(comms.get(n, 0) % cmap.N) for n in g.nodes]
    ax.scatter(xs, ys, s=130, c=cs, edgecolors="black", lw=0.6, zorder=2)

    for n in g.nodes:
        if n in label_states:
            ax.annotate(n, pos[n], fontsize=8, ha="center", va="center",
                        weight="bold", color="black", zorder=3)

    annot = (
        "top betweenness: " + ", ".join(label_states)
        if label_states else "top centrality nodes not stable across half-lives"
    )
    ax.set_title(f"{title}\nLouvain communities = {n_comm}    {annot}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_pre_vs_during_pmfg(
    g_pre: nx.Graph,
    title_pre: str,
    label_pre: list[str],
    g_during: nx.Graph,
    title_during: str,
    label_during: list[str],
    out_path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_forcedirected(axes[0], g_pre, title_pre, label_pre)
    plot_forcedirected(axes[1], g_during, title_during, label_during)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def run_step2_pmfg() -> None:
    print("\n" + "=" * 72)
    print("STEP 2 — PMFG snapshots, centrality tables, figures")
    print("=" * 72)

    print("Loading EWMA hl=24 cube...")
    cube_24, dates_24, cols_24 = load_hl24_cube()
    print(f"  {cube_24.shape}, {dates_24.min().date()} -> {dates_24.max().date()}")

    print("Loading state returns for sensitivity recompute (hl=18, hl=30)...")
    returns = pd.read_parquet(PROCESSED / "01_state_returns.parquet")

    print("Recomputing EWMA hl=18 cube...")
    cube_18, dates_18, cols_18 = compute_cube(returns, halflife=18)
    print(f"  {cube_18.shape}")

    print("Recomputing EWMA hl=30 cube...")
    cube_30, dates_30, cols_30 = compute_cube(returns, halflife=30)
    print(f"  {cube_30.shape}")

    cubes_by_hl = {
        18: (cube_18, dates_18, cols_18),
        24: (cube_24, dates_24, cols_24),
        30: (cube_30, dates_30, cols_30),
    }

    pmfgs_by_hl: dict[int, list[nx.Graph]] = {hl: [] for hl in SENSITIVITY_HLS}
    snapshot_metrics: list[dict] = []
    top5_rows: list[dict] = []
    robust_rows: list[dict] = []
    robust_betweenness: list[list[str]] = []

    for target, key in zip(DATES, SNAPSHOT_KEY):
        print(f"\n=== Snapshot {key} ===")
        per_hl_top5: dict[int, dict[str, list[str]]] = {}
        for hl in SENSITIVITY_HLS:
            cube, dates, cols = cubes_by_hl[hl]
            corr_full, actual = matrix_at(cube, dates, target)
            corr, names = drop_us_aggregate(corr_full, cols)
            print(
                f"  hl={hl}: nearest date {actual.date()}, "
                f"building PMFG ({len(names)} nodes)..."
            )
            g = build_pmfg(corr, names)
            print(f"    edges={g.number_of_edges()}  expected={3*(len(names)-2)}")
            pmfgs_by_hl[hl].append(g)

            top5 = centrality_top5(g)
            per_hl_top5[hl] = {
                ctype: top5[top5["centrality_type"] == ctype]["state"].tolist()
                for ctype in ["degree", "betweenness", "eigenvector"]
            }

            if hl == DEFAULT_HL:
                metrics = graph_metrics(g)
                metrics["recession"] = key
                snapshot_metrics.append(metrics)
                for _, row in top5.iterrows():
                    top5_rows.append({"snapshot_date": key, **row.to_dict()})

        snapshot_robust: dict[str, list[str]] = {}
        for ctype in ["degree", "betweenness", "eigenvector"]:
            sets = [set(per_hl_top5[hl][ctype]) for hl in SENSITIVITY_HLS]
            robust = sets[0] & sets[1] & sets[2]
            ordered = [s for s in per_hl_top5[DEFAULT_HL][ctype] if s in robust]
            snapshot_robust[ctype] = ordered
            for rank, state in enumerate(ordered, start=1):
                robust_rows.append({
                    "snapshot_date": key,
                    "centrality_type": ctype,
                    "rank": rank,
                    "state": state,
                })
        robust_betweenness.append(snapshot_robust["betweenness"][:3])
        print(f"  robust betweenness top-3: {snapshot_robust['betweenness'][:3]}")

    print("\nWriting CSVs...")
    typo_csv = TABLES / "09_recession_typology.csv"
    typo = pd.read_csv(typo_csv)
    metrics_df = pd.DataFrame(snapshot_metrics)

    metric_cols = [
        "pmfg_avg_clustering", "pmfg_triangles",
        "pmfg_4cliques", "pmfg_largest_community",
    ]
    for col in metric_cols:
        typo[col] = np.nan
    for key, rec in DURING_TO_RECESSION.items():
        row = metrics_df[metrics_df["recession"] == key]
        if row.empty:
            continue
        for col in metric_cols:
            typo.loc[typo["recession"] == rec, col] = float(row.iloc[0][col])
    typo.to_csv(typo_csv, index=False)
    print(f"  appended PMFG cols to {typo_csv}")

    top5_df = pd.DataFrame(
        top5_rows,
        columns=["snapshot_date", "centrality_type", "rank", "state", "value"],
    )
    top5_path = TABLES / "09_pmfg_centrality_top5.csv"
    top5_df.to_csv(top5_path, index=False)
    print(f"  wrote {top5_path}")

    robust_df = pd.DataFrame(
        robust_rows, columns=["snapshot_date", "centrality_type", "rank", "state"]
    )
    robust_path = TABLES / "09_pmfg_centrality_robust.csv"
    robust_df.to_csv(robust_path, index=False)
    print(f"  wrote {robust_path}")

    n_snapshots = len(DATES)
    n_rows = n_snapshots // 2

    print("\nPlotting geographic figure...")
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4 * n_rows))
    for ax, g, title, label in zip(
        axes.flat, pmfgs_by_hl[DEFAULT_HL], TITLES, robust_betweenness
    ):
        plot_geographic(ax, g, title, label)
    fig.suptitle(
        "PMFG snapshots — geographic layout (EWMA hl=24)\n"
        "Tumminello, Aste, Di Matteo & Mantegna (2005), PNAS 102(30)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out_geo = FIGURES / "09_pmfg_geographic.png"
    fig.savefig(out_geo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_geo}")

    print("Plotting force-directed figure...")
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5 * n_rows))
    for ax, g, title, label in zip(
        axes.flat, pmfgs_by_hl[DEFAULT_HL], TITLES, robust_betweenness
    ):
        plot_forcedirected(ax, g, title, label)
    fig.suptitle(
        "PMFG snapshots — force-directed layout, nodes coloured by Louvain community",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out_force = FIGURES / "09_pmfg_forcedirected.png"
    fig.savefig(out_force, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_force}")

    print("Plotting pre-vs-during GFC example figure...")
    pre_idx = SNAPSHOT_KEY.index("2007-01")
    during_idx = SNAPSHOT_KEY.index("2009-01")
    plot_pre_vs_during_pmfg(
        pmfgs_by_hl[DEFAULT_HL][pre_idx],
        TITLES[pre_idx],
        robust_betweenness[pre_idx],
        pmfgs_by_hl[DEFAULT_HL][during_idx],
        TITLES[during_idx],
        robust_betweenness[during_idx],
        FIGURES / "09_pmfg_gfc_single.png",
    )

    print("\n=== PMFG metrics (hl=24, during-recession snapshot) ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Robust top-3 betweenness per snapshot (hl 18/24/30 intersection) ===")
    for key, robust in zip(SNAPSHOT_KEY, robust_betweenness):
        print(f"  {key}: {robust if robust else '(none stable)'}")


# ============================================================================
# Driver
# ============================================================================

def main() -> None:
    run_step1_mst()
    run_step2_pmfg()


if __name__ == "__main__":
    main()
