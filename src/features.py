"""Network feature extractors.

Each function takes either a correlation matrix (dense n×n) or the MST /
thresholded-graph derived from it, and returns a scalar feature. All 5
features are then packaged into ``extract_features_over_time``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain  # python-louvain package

from .networks import corr_to_mst, threshold_graph


def mst_length(corr: np.ndarray) -> float:
    """Sum of MST edge weights (Onnela distances).

    Known result from financial-network literature: MST length *shrinks* as
    the system becomes more synchronised (all distances go to zero as
    correlations go to 1). So a falling mst_length signals rising co-movement.
    """
    mst = corr_to_mst(corr)
    # Each undirected edge appears twice after symmetrisation.
    return float(mst.sum() / 2.0)


def mean_off_diagonal_corr(corr: np.ndarray) -> float:
    """Mean of off-diagonal correlation entries (correlation collapse proxy)."""
    n = corr.shape[0]
    return float((corr.sum() - n) / (n * (n - 1)))


def largest_eigenvalue(corr: np.ndarray) -> float:
    """Largest eigenvalue of the correlation matrix.

    In systemic-risk literature (e.g. Billio et al.), the top eigenvalue of
    the correlation matrix tracks the share of variance explained by the
    first principal component, i.e. the "common factor" dominating all
    states simultaneously. Grows toward n (= 51) as correlations → 1.
    """
    # eigvalsh returns ascending eigenvalues for a symmetric input.
    eigs = np.linalg.eigvalsh(corr)
    return float(eigs[-1])


def network_density(corr: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of off-diagonal pairs with |rho| > threshold.

    With n nodes there are n*(n-1)/2 unordered pairs. Density is the count
    of pairs above the threshold divided by that maximum.
    """
    adj = threshold_graph(corr, threshold=threshold)
    n = adj.shape[0]
    max_edges = n * (n - 1) / 2
    return float(adj.sum() / 2 / max_edges)


def n_communities(corr: np.ndarray, threshold: float = 0.5, seed: int = 0) -> int:
    """Number of Louvain communities in the thresholded graph.

    Isolated nodes each count as their own community. As the system
    synchronises, modular structure erodes and community count drops toward 1.
    """
    adj = threshold_graph(corr, threshold=threshold)
    G = nx.from_numpy_array(adj)
    # Louvain is stochastic; fix the seed for reproducibility.
    partition = community_louvain.best_partition(G, random_state=seed)
    return int(len(set(partition.values())))


def extract_features_over_time(
    cubes: np.ndarray,
    dates: pd.DatetimeIndex,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Apply all 5 feature extractors to each rolling window.

    Parameters
    ----------
    cubes : ndarray (n_windows, n_states, n_states)
    dates : DatetimeIndex — window-end dates
    threshold : float — used for network_density and n_communities

    Returns a DataFrame indexed by date with columns:
        mst_length, mean_corr, largest_eigenvalue, network_density, n_communities
    """
    n_windows = cubes.shape[0]
    out = {
        "mst_length": np.empty(n_windows),
        "mean_corr": np.empty(n_windows),
        "largest_eigenvalue": np.empty(n_windows),
        "network_density": np.empty(n_windows),
        "n_communities": np.empty(n_windows, dtype=int),
    }
    for i in range(n_windows):
        C = cubes[i]
        out["mst_length"][i] = mst_length(C)
        out["mean_corr"][i] = mean_off_diagonal_corr(C)
        out["largest_eigenvalue"][i] = largest_eigenvalue(C)
        out["network_density"][i] = network_density(C, threshold=threshold)
        out["n_communities"][i] = n_communities(C, threshold=threshold)

    return pd.DataFrame(out, index=dates)
