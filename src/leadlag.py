"""Lead-lag analysis utilities.

We want to know: at what horizon (in months) is each network feature most
strongly related to the NBER recession indicator? A feature that peaks at a
*positive* lag-ahead value is a leading indicator; the lag argmax is its
typical lead time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lead_lag_correlation(
    feature: pd.Series,
    target: pd.Series,
    lags: range | list[int] = range(-12, 13),
) -> pd.Series:
    """Correlation between ``feature`` and ``target`` at each lag in months.

    Convention
    ----------
    For each lag k, we compute corr(feature(t), target(t + k)).
    - k > 0 → target in the FUTURE relative to feature. High corr means the
      feature today moves with a recession k months ahead → LEADING by k.
    - k < 0 → target in the PAST relative to feature. LAGGING by |k|.

    We align on a common DatetimeIndex (month-end) before shifting, so both
    series must share a frequency. NaNs produced by the shift are dropped
    pairwise when computing the correlation.
    """
    feat = feature.copy()
    tgt = target.copy()
    idx = feat.index.intersection(tgt.index)
    feat = feat.loc[idx]
    tgt = tgt.loc[idx]

    out = {}
    for k in lags:
        # k > 0 → target shifted left (future) to align with current feature.
        t_shift = tgt.shift(-k)
        both = pd.concat([feat, t_shift], axis=1).dropna()
        if both.shape[0] < 24:
            out[k] = np.nan
        else:
            out[k] = both.iloc[:, 0].corr(both.iloc[:, 1])
    return pd.Series(out, name=feature.name)


def lead_lag_table(
    features: pd.DataFrame,
    target: pd.Series,
    lags: range | list[int] = range(-12, 13),
) -> pd.DataFrame:
    """Apply ``lead_lag_correlation`` column-wise. Returns a lag × feature DF."""
    cols = {}
    for name in features.columns:
        cols[name] = lead_lag_correlation(features[name], target, lags=lags)
    return pd.DataFrame(cols)


def best_lead(lltable: pd.DataFrame) -> pd.DataFrame:
    """For each feature, return (lag_of_max_abs_corr, corr_at_that_lag).

    Using ``abs`` because features that fall under stress (mst_length,
    n_communities) will have *negative* peak correlation with USREC.
    """
    rows = []
    for name in lltable.columns:
        s = lltable[name]
        k = int(s.abs().idxmax())
        rows.append({"feature": name, "best_lag": k, "corr": float(s.loc[k])})
    return pd.DataFrame(rows).set_index("feature")
