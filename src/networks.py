"""Network construction utilities.

``rolling_corr`` builds 60-month rolling correlation matrices; the
remaining helpers turn those into Onnela-distance MSTs and thresholded
graphs used downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree


def rolling_corr(
    returns: pd.DataFrame,
    window: int = 60,
    min_periods: int | None = None,
    method: str = "pearson",
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """Compute a rolling window of correlation matrices across all columns.

    Parameters
    ----------
    returns : DataFrame
        Monthly returns, rows indexed by date, columns are entities (states).
    window : int
        Rolling window length in months. 60 is the project default.
    min_periods : int, optional
        Minimum non-NaN observations required in the window. Defaults to
        ``window`` (strict: no partial windows).
    method : {"pearson", "spearman", "kendall"}
        Correlation method forwarded to ``DataFrame.corr``. Spearman is
        rank-based and therefore robust to single-observation extremes.

    Returns
    -------
    cubes, dates, cols — as above.
    """
    if min_periods is None:
        min_periods = window

    cols = list(returns.columns)
    values = returns.to_numpy(dtype=float)
    n_rows, n_cols = values.shape

    out_dates: list[pd.Timestamp] = []
    out_mats: list[np.ndarray] = []

    for end in range(window, n_rows + 1):
        start = end - window
        block = values[start:end]
        col_valid = (~np.isnan(block)).sum(axis=0)
        if (col_valid < min_periods).any():
            continue
        mat = pd.DataFrame(block, columns=cols).corr(method=method).to_numpy()
        out_mats.append(mat)
        out_dates.append(returns.index[end - 1])

    cubes = np.stack(out_mats, axis=0) if out_mats else np.empty((0, n_cols, n_cols))
    return cubes, pd.DatetimeIndex(out_dates, name="Date"), cols


def ewma_corr_cube(
    returns: pd.DataFrame,
    halflife: float,
    burn_in_multiplier: float = 3.0,
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """Build a stack of EWMA correlation matrices, one per month after burn-in.

    Uses the same RiskMetrics recursion as ``ewma_mean_corr``; returns the
    full n×n correlation matrix at each step so we can extract the same 5
    network features on top of it. Signature matches ``rolling_corr``.
    """
    lam = 0.5 ** (1.0 / halflife)
    cols = list(returns.columns)
    X = returns.to_numpy(dtype=float)
    n_t, n_s = X.shape

    csum = np.nancumsum(X, axis=0)
    counts = np.arange(1, n_t + 1).reshape(-1, 1)
    running_mean = csum / counts
    prev_mean = np.vstack([np.zeros((1, n_s)), running_mean[:-1]])
    Xc = X - prev_mean

    S = np.zeros((n_s, n_s))
    out_mats: list[np.ndarray] = []
    out_dates: list[pd.Timestamp] = []
    burn_in = int(burn_in_multiplier * halflife)
    for t in range(n_t):
        x = Xc[t]
        if np.any(np.isnan(x)):
            continue
        S = lam * S + (1 - lam) * np.outer(x, x)
        if t < burn_in:
            continue
        v = np.diag(S).copy()
        v[v <= 0] = np.nan
        std = np.sqrt(v)
        corr = S / np.outer(std, std)
        # Numerical safety: force exact symmetry and clip to [-1, 1]
        corr = 0.5 * (corr + corr.T)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        out_mats.append(corr)
        out_dates.append(returns.index[t])

    cubes = np.stack(out_mats, axis=0) if out_mats else np.empty((0, n_s, n_s))
    return cubes, pd.DatetimeIndex(out_dates, name="Date"), cols


def ewma_mean_corr(
    returns: pd.DataFrame,
    halflife: float,
    burn_in_multiplier: float = 3.0,
) -> pd.Series:
    """Mean off-diagonal exponentially-weighted correlation across all columns.

    Implements the RiskMetrics-style EWMA covariance recursion:
        S_t = λ · S_{t-1} + (1 - λ) · x_t x_t^T
    where λ = 0.5^(1/halflife). We skip the first ``burn_in_multiplier * halflife``
    months because the recursion hasn't converged yet. Returns are demeaned by
    their expanding-window mean before being fed in (strictly causal — no
    look-ahead from the full-sample mean).
    """
    lam = 0.5 ** (1.0 / halflife)
    X = returns.to_numpy(dtype=float)
    n_t, n_s = X.shape

    # Expanding (causal) mean: at time t we know returns up to and including t.
    csum = np.nancumsum(X, axis=0)
    counts = np.arange(1, n_t + 1).reshape(-1, 1)
    running_mean = csum / counts
    # Use t-1 mean to demean x_t (strictly causal). At t=0 use 0.
    prev_mean = np.vstack([np.zeros((1, n_s)), running_mean[:-1]])
    Xc = X - prev_mean

    S = np.zeros((n_s, n_s))
    out_vals = []
    out_dates = []
    burn_in = int(burn_in_multiplier * halflife)
    for t in range(n_t):
        x = Xc[t]
        if np.any(np.isnan(x)):
            continue
        S = lam * S + (1 - lam) * np.outer(x, x)
        if t < burn_in:
            continue
        v = np.diag(S).copy()
        v[v <= 0] = np.nan  # avoid div-by-zero / numerical instability
        std = np.sqrt(v)
        corr = S / np.outer(std, std)
        mean_off = (np.nansum(corr) - n_s) / (n_s * (n_s - 1))
        out_vals.append(mean_off)
        out_dates.append(returns.index[t])
    return pd.Series(out_vals, index=pd.DatetimeIndex(out_dates, name="Date"),
                     name=f"ewma_hl{int(halflife)}")


def mean_off_diagonal(cubes: np.ndarray) -> np.ndarray:
    """Mean of the off-diagonal entries of each correlation matrix.

    For an n×n matrix the diagonal (n ones) is excluded. This is the headline
    "mean correlation" used as a systemic co-movement proxy.
    """
    n = cubes.shape[1]
    # Sum all entries, subtract the diagonal (which is n), divide by n*(n-1).
    total = cubes.sum(axis=(1, 2)) - n
    return total / (n * (n - 1))


def corr_to_distance(corr: np.ndarray) -> np.ndarray:
    """Onnela distance: d_ij = sqrt(2 * (1 - rho_ij)).

    Maps Pearson correlation onto a proper metric. High correlation → small
    distance, so an MST built on these distances keeps the strongest
    co-movement links.
    """
    # Clip tiny numerical noise so sqrt is always well-defined.
    arg = np.clip(2.0 * (1.0 - corr), 0.0, 4.0)
    return np.sqrt(arg)


def corr_to_mst(corr: np.ndarray) -> np.ndarray:
    """Minimum spanning tree of the Onnela-distance graph.

    Returns a dense symmetric n×n matrix where non-zero entries are the MST
    edge *distances* (not correlations). Entries not on the MST are zero.
    The MST has exactly n-1 edges.
    """
    d = corr_to_distance(corr)
    # scipy returns an upper-triangular sparse matrix; densify and symmetrise.
    mst_sparse = minimum_spanning_tree(d)
    mst_dense = mst_sparse.toarray()
    return mst_dense + mst_dense.T


def threshold_graph(corr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Boolean adjacency for edges where |rho_ij| > threshold (off-diagonal).

    Returns an n×n 0/1 integer array, symmetric, zero diagonal.
    """
    n = corr.shape[0]
    adj = (np.abs(corr) > threshold).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj
