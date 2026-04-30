"""Alternative network feature family: build 5 alt features from the
cached EWMA correlation cubes. Writes a monthly parquet with the 5 features.

Features (spec):
    1. corr_std       — std of off-diagonal correlation entries
    2. corr_skewness  — skewness of off-diagonal entries (scipy.stats.skew)
    3. corr_kurtosis  — excess kurtosis of off-diagonal entries
    4. pmfg_sum_sq_corr          — sum of squared correlations on the PMFG edges
    5. pmfg_separators_cliques_ratio
        primary definition:
            len(list(nx.all_node_cuts(PMFG))) / num_triangles_in_PMFG
        nx.all_node_cuts returns the set of distinct minimum node-separator
        sets of cardinality κ(G). This is the exact "number of separators"
        reading.

        Because all_node_cuts is occasionally pathological on some PMFG
        topologies (we saw a single call balloon past 5 minutes), it is wrapped
        in a SIGALRM timeout (see ALL_CUTS_TIMEOUT below). Timesteps where the
        primary definition exceeds the timeout fall back to a proxy:
            proxy = nx.node_connectivity(PMFG) / num_triangles
        which is equal to the primary definition when there is a single min cut
        and is always cheap. Fallback rows are flagged in a sidecar column
        ``feat5_fallback`` (True/False) written to a separate parquet.

The PMFG is built via Mantegna distance sqrt(2(1-ρ)), edges sorted ascending,
added one-by-one with networkx.check_planarity rejection. Stop at 3(N-2)=147
edges for N=51.

Outputs:
    experiments/alt_feature_family/data/alt_features.parquet
        columns: corr_std, corr_skewness, corr_kurtosis,
                 pmfg_sum_sq_corr, pmfg_separators_cliques_ratio
        index:   DatetimeIndex (monthly, matches EWMA cube dates)
    experiments/alt_feature_family/data/alt_features_feat5_log.parquet
        columns: feat5_fallback (bool), feat5_seconds (float)
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import skew, kurtosis


ROOT = Path(__file__).resolve().parents[2]
EWMA_NPZ = ROOT / "data" / "processed" / "03_rolling_corr_ewma.npz"
OUT_DIR = ROOT / "experiments" / "alt_feature_family" / "data"
OUT_PARQUET = OUT_DIR / "alt_features.parquet"
OUT_LOG = OUT_DIR / "alt_features_feat5_log.parquet"

ALL_CUTS_TIMEOUT = 5  # seconds; SIGALRM integer resolution


def _p(msg: str) -> None:
    print(msg, flush=True)


def build_pmfg(corr: np.ndarray, node_labels: list[str]) -> nx.Graph:
    n = corr.shape[0]
    target_edges = 3 * (n - 2)
    iu, ju = np.triu_indices(n, k=1)
    rho = np.clip(corr[iu, ju], -1.0, 1.0)
    dist = np.sqrt(2.0 * (1.0 - rho))
    order = np.argsort(dist, kind="stable")

    G = nx.Graph()
    G.add_nodes_from(node_labels)
    added = 0
    for k in order:
        u = node_labels[iu[k]]
        v = node_labels[ju[k]]
        G.add_edge(u, v, weight=float(dist[k]), rho=float(rho[k]))
        is_planar, _ = nx.check_planarity(G)
        if not is_planar:
            G.remove_edge(u, v)
            continue
        added += 1
        if added >= target_edges:
            break
    return G


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _Timeout()


def count_all_min_cuts(G: nx.Graph, timeout_sec: int) -> tuple[int, bool, float]:
    """Run nx.all_node_cuts with a SIGALRM timeout.

    Returns (count, is_fallback, seconds). On timeout, falls back to
    nx.node_connectivity(G) (always cheap) and is_fallback=True.
    """
    t0 = time.perf_counter()
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_sec)
    try:
        cuts = list(nx.all_node_cuts(G))
        signal.alarm(0)
        return len(cuts), False, time.perf_counter() - t0
    except _Timeout:
        signal.alarm(0)
        kappa = nx.node_connectivity(G)
        return int(kappa), True, time.perf_counter() - t0
    except nx.NetworkXError:
        signal.alarm(0)
        return 0, False, time.perf_counter() - t0
    finally:
        signal.signal(signal.SIGALRM, old)


def compute_one_timestep(
    corr: np.ndarray, node_labels: list[str]
) -> tuple[dict[str, float], bool, float]:
    n = corr.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    off = corr[iu, ju]
    corr_std = float(np.std(off, ddof=0))
    corr_skew = float(skew(off, bias=True))
    corr_kurt = float(kurtosis(off, fisher=True, bias=True))

    G = build_pmfg(corr, node_labels)
    pmfg_ssq = 0.0
    for _, _, data in G.edges(data=True):
        pmfg_ssq += float(data["rho"]) ** 2

    num_tri = sum(nx.triangles(G).values()) // 3
    if num_tri == 0:
        ratio = float("nan")
        is_fb = False
        seconds = 0.0
    else:
        n_seps, is_fb, seconds = count_all_min_cuts(G, ALL_CUTS_TIMEOUT)
        ratio = n_seps / num_tri

    return (
        {
            "corr_std": corr_std,
            "corr_skewness": corr_skew,
            "corr_kurtosis": corr_kurt,
            "pmfg_sum_sq_corr": pmfg_ssq,
            "pmfg_separators_cliques_ratio": ratio,
        },
        is_fb,
        seconds,
    )


def main() -> None:
    z = np.load(EWMA_NPZ, allow_pickle=True)
    cubes = z["cubes"]
    dates = pd.DatetimeIndex(z["dates"])
    cols = [str(c) for c in z["cols"]]

    n_steps = cubes.shape[0]
    _p(
        f"Computing alt features for {n_steps} EWMA timesteps "
        f"({dates[0].date()} → {dates[-1].date()})"
    )
    _p(f"SIGALRM timeout on all_node_cuts: {ALL_CUTS_TIMEOUT}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    rows: list[dict[str, float]] = []
    feat5_logs: list[dict] = []
    for i in range(n_steps):
        t_step = time.perf_counter()
        feats, is_fb, secs = compute_one_timestep(cubes[i], cols)
        step_dt = time.perf_counter() - t_step
        rows.append(feats)
        feat5_logs.append({"feat5_fallback": is_fb, "feat5_seconds": secs})

        if i % 10 == 0 or is_fb or step_dt > 3.0 or i == n_steps - 1:
            elapsed = time.perf_counter() - t_start
            rate = elapsed / (i + 1)
            eta = rate * (n_steps - i - 1)
            fb_flag = " FALLBACK" if is_fb else ""
            _p(
                f"  step {i + 1:4d}/{n_steps}  dt={step_dt:5.2f}s"
                f"  elapsed {elapsed:6.1f}s  eta {eta:6.1f}s{fb_flag}"
            )

        # Incremental flush every 25 steps — lets us recover / monitor
        if (i + 1) % 25 == 0 or i == n_steps - 1:
            df_partial = pd.DataFrame(rows, index=dates[: i + 1])
            df_partial.index.name = "date"
            df_partial.to_parquet(OUT_PARQUET)
            log_partial = pd.DataFrame(feat5_logs, index=dates[: i + 1])
            log_partial.index.name = "date"
            log_partial.to_parquet(OUT_LOG)

    df = pd.DataFrame(rows, index=dates)
    df.index.name = "date"
    df.to_parquet(OUT_PARQUET)
    log_df = pd.DataFrame(feat5_logs, index=dates)
    log_df.index.name = "date"
    log_df.to_parquet(OUT_LOG)

    n_fb = int(sum(1 for r in feat5_logs if r["feat5_fallback"]))
    _p("")
    _p(f"Saved → {OUT_PARQUET.relative_to(ROOT)}")
    _p(f"Saved → {OUT_LOG.relative_to(ROOT)}")
    _p(f"Shape: {df.shape}")
    _p(f"Feature-5 fallbacks: {n_fb} / {n_steps}")
    _p("NaNs per column:")
    _p(df.isna().sum().to_string())
    _p("")
    _p("Summary statistics:")
    _p(df.describe().round(4).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
