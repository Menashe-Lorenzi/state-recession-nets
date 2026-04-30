"""Paired bootstrap on (combined_alt, combined_moments) − baseline_engineered.

Same protocol as Section 8:
- 2000 resamples with replacement on the pooled OOS test indices
- paired: same resampled indices used for baseline and challenger per resample
- 90 % CI (5th / 95th pct) and P(Δ > 0) reported per horizon

Reads preds_cache.parquet from run_walkforward_alt.py. Writes bootstrap_alt.parquet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
ALT_DIR = ROOT / "experiments" / "alt_feature_family"
PREDS = ALT_DIR / "results" / "preds_cache.parquet"
BOOT_OUT = ALT_DIR / "results" / "bootstrap_alt.parquet"


def main() -> None:
    preds = pd.read_parquet(PREDS)
    print(f"Loaded predictions cache: {preds.shape}")
    print(f"  variants: {sorted(preds['variant'].unique())}")
    print(f"  horizons: {sorted(preds['horizon_q'].unique())}")

    rng = np.random.default_rng(20260413)
    n_resample = 2000

    rows: list[dict] = []
    for challenger in ["combined_alt", "combined_moments"]:
        print(f"\n--- challenger: {challenger} ---")
        for h in [1, 2, 3]:
            base = (
                preds[
                    (preds["variant"] == "baseline_engineered")
                    & (preds["horizon_q"] == h)
                ]
                .set_index("Date")
                .sort_index()
            )
            comb = (
                preds[
                    (preds["variant"] == challenger)
                    & (preds["horizon_q"] == h)
                ]
                .set_index("Date")
                .sort_index()
            )
            assert np.array_equal(base.index.to_numpy(), comb.index.to_numpy())
            assert np.array_equal(base["y"].to_numpy(), comb["y"].to_numpy())
            y = base["y"].to_numpy(dtype=int)
            p_base = base["pred"].to_numpy()
            p_comb = comb["pred"].to_numpy()

            point_base = roc_auc_score(y, p_base)
            point_comb = roc_auc_score(y, p_comb)
            point_delta = point_comb - point_base

            n = len(y)
            # Reset the RNG per challenger so both runs use the *same*
            # bootstrap indices — that is the paired protocol across challengers
            # too, not just baseline-vs-one-challenger.
            local_rng = np.random.default_rng(20260413 + h)
            deltas = np.empty(n_resample, dtype=float)
            for r in range(n_resample):
                idx = local_rng.integers(0, n, size=n)
                yr = y[idx]
                if len(np.unique(yr)) < 2:
                    deltas[r] = np.nan
                    continue
                auc_b = roc_auc_score(yr, p_base[idx])
                auc_c = roc_auc_score(yr, p_comb[idx])
                deltas[r] = auc_c - auc_b
            finite = deltas[np.isfinite(deltas)]
            pct5 = float(np.percentile(finite, 5))
            pct95 = float(np.percentile(finite, 95))
            p_pos = float((finite > 0).mean())

            rows.append(
                {
                    "challenger": challenger,
                    "horizon_q": h,
                    "auc_base": point_base,
                    "auc_challenger": point_comb,
                    "delta_point": point_delta,
                    "delta_pct5": pct5,
                    "delta_pct95": pct95,
                    "p_delta_gt_0": p_pos,
                    "n_oos": n,
                    "n_positives_oos": int(y.sum()),
                    "n_valid_resamples": int(len(finite)),
                }
            )
            print(
                f"  {h}Q  Δ={point_delta:+.4f}  "
                f"[{pct5:+.4f}, {pct95:+.4f}]  P(Δ>0)={p_pos:.3f}  "
                f"(n_oos={n}, positives={int(y.sum())})"
            )

    out = pd.DataFrame(rows)
    BOOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(BOOT_OUT)
    print()
    print(f"Saved → {BOOT_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
