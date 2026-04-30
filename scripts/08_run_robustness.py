"""Section 8: statistical robustness of the Section 7 4-cell comparison.

Tests (all on the engineered 38-feature panel + 5 network features):
  1. Point-estimate recap (logistic baseline / logistic combined /
     xgboost baseline / xgboost combined × 1Q / 2Q / 3Q).
  2. Paired bootstrap CI on each combined-vs-baseline delta (2000 resamples).
  3. Per-recession split: AUC within ±8Q of 2001 / 2008-09 / 2020 anchors.
  4. Drop-COVID sensitivity: re-pool AUC excluding test dates ≥ 2020-01-01.
  5. Logistic C-sensitivity sweep: re-run L1 walk-forward at 6 C values to
     check whether the ~0 delta is robust to regularisation strength.
  6. Logistic coefficient inspection at C=0.25: does L1 zero the network
     features?

All predictions are cached once from a single walk-forward pass, then
bootstrap / per-recession / drop-COVID operate on the cached predictions so
the tests are mutually consistent.

Outputs:
  data/processed/08_bootstrap.parquet
  data/processed/08_per_recession.parquet
  data/processed/08_drop_covid.parquet
  data/processed/08_c_sweep.parquet
  reports/08_robustness_summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.modeling import NETWORK_COLS, TARGET_COLS, build_panel_engineered

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

RNG_SEED = 20260411
N_BOOT = 2000

INITIAL = 60
TEST = 8
STEP = 8

LOGISTIC_C_HEADLINE = 0.25
LOGISTIC_C_SWEEP = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

# XGBoost best hyperparameters from Section 7 run (frozen here to avoid re-tuning).
XGB_BEST = {
    (1, "baseline_engineered"): dict(
        n_estimators=100, max_depth=2, learning_rate=0.10,
        min_child_weight=1, reg_lambda=5.0,
    ),
    (1, "combined_engineered"): dict(
        n_estimators=400, max_depth=2, learning_rate=0.05,
        min_child_weight=1, reg_lambda=1.0,
    ),
    (2, "baseline_engineered"): dict(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_child_weight=1, reg_lambda=5.0,
    ),
    (2, "combined_engineered"): dict(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        min_child_weight=3, reg_lambda=1.0,
    ),
    (3, "baseline_engineered"): dict(
        n_estimators=200, max_depth=2, learning_rate=0.10,
        min_child_weight=3, reg_lambda=5.0,
    ),
    (3, "combined_engineered"): dict(
        n_estimators=400, max_depth=2, learning_rate=0.05,
        min_child_weight=1, reg_lambda=1.0,
    ),
}

RECESSION_ANCHORS = {
    "2001": pd.Timestamp("2001-07-01"),
    "2008-09": pd.Timestamp("2008-12-01"),
    "2020": pd.Timestamp("2020-04-01"),
}
PER_REC_WINDOW_Q = 8


# ------------------------------------------------------------------ models --

def _fit_logistic(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, C: float):
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(imp.fit_transform(X_tr))
    X_te_s = scaler.transform(imp.transform(X_te))
    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_te), float(y_tr.mean())), None, imp, scaler
    clf = LogisticRegression(
        penalty="l1", solver="liblinear",
        C=C, class_weight="balanced", max_iter=5000,
    )
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1], clf, imp, scaler


def _fit_xgb(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, params: dict):
    if XGBClassifier is None:
        raise RuntimeError("xgboost not installed")
    imp = SimpleImputer(strategy="median")
    X_tr_s = imp.fit_transform(X_tr)
    X_te_s = imp.transform(X_te)
    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_te), float(y_tr.mean()))
    pos = max(int(y_tr.sum()), 1)
    spw = float((len(y_tr) - y_tr.sum()) / pos)
    clf = XGBClassifier(
        **params,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0, n_jobs=1,
        scale_pos_weight=spw,
        use_label_encoder=False,
    )
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def walk_forward_predict(
    X: pd.DataFrame, y: pd.Series, model_type: str, **kwargs
) -> pd.DataFrame:
    X = X.sort_index()
    y = y.reindex(X.index)
    n = len(X)
    rows = []
    start = INITIAL
    while start + TEST <= n:
        X_tr = X.iloc[:start].to_numpy(dtype=float)
        y_tr = y.iloc[:start].to_numpy(dtype=int)
        X_te = X.iloc[start : start + TEST].to_numpy(dtype=float)
        y_te = y.iloc[start : start + TEST].to_numpy(dtype=int)
        te_dates = X.index[start : start + TEST]
        if model_type == "logistic":
            p, *_ = _fit_logistic(X_tr, y_tr, X_te, C=kwargs["C"])
        elif model_type == "xgboost":
            p = _fit_xgb(X_tr, y_tr, X_te, kwargs["params"])
        else:
            raise ValueError(model_type)
        for d, pred, yt in zip(te_dates, p, y_te):
            rows.append({"date": d, "pred": float(pred), "y": int(yt)})
        start += STEP
    return pd.DataFrame(rows).set_index("date")


# ------------------------------------------------------------------- tests --

def pooled_auc(df: pd.DataFrame) -> float:
    y = df["y"].to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, df["pred"].to_numpy()))


def paired_bootstrap(
    base: pd.DataFrame, comb: pd.DataFrame, rng: np.random.Generator
) -> dict:
    # Align by index (same OOS dates).
    base = base.sort_index()
    comb = comb.loc[base.index]
    y = base["y"].to_numpy()
    pb = base["pred"].to_numpy()
    pc = comb["pred"].to_numpy()
    n = len(y)
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(
            roc_auc_score(yy, pc[idx]) - roc_auc_score(yy, pb[idx])
        )
    d = np.array(deltas)
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "pct_5": float(np.percentile(d, 5)),
        "pct_95": float(np.percentile(d, 95)),
        "p_positive": float((d > 0).mean()),
        "n_valid": int(len(d)),
    }


def per_recession_auc(df: pd.DataFrame) -> dict:
    out: dict[str, float] = {}
    for name, anchor in RECESSION_ANCHORS.items():
        lo = anchor - pd.DateOffset(months=3 * PER_REC_WINDOW_Q)
        hi = anchor + pd.DateOffset(months=3 * PER_REC_WINDOW_Q)
        window = df[(df.index >= lo) & (df.index <= hi)]
        if len(window) == 0 or len(np.unique(window["y"])) < 2:
            out[name] = float("nan")
        else:
            out[name] = pooled_auc(window)
    return out


def drop_covid_auc(df: pd.DataFrame) -> float:
    mask = df.index < pd.Timestamp("2020-01-01")
    sub = df[mask]
    return pooled_auc(sub)


# ------------------------------------------------------------- main driver --

def _fmt(v: float) -> str:
    if np.isnan(v):
        return "  nan"
    return f"{v: .3f}" if v < 0 else f"{v:.3f}"


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("Loading panel...")
    feats = pd.read_parquet(PROCESSED / "04_network_features_ewma.parquet")
    panel, engineered_cols = build_panel_engineered(feats)
    print(f"  panel: {panel.shape}, {panel.index.min().date()} → {panel.index.max().date()}")
    print(f"  engineered feature count: {len(engineered_cols)}")

    variants = {
        "baseline_engineered": engineered_cols,
        "combined_engineered": engineered_cols + NETWORK_COLS,
    }

    # ------------------------------------------------------- cache predictions
    preds: dict[tuple, pd.DataFrame] = {}
    print("\nCaching walk-forward predictions for all 4 cells × 3 horizons...")
    for horizon, tgt in zip([1, 2, 3], TARGET_COLS):
        y = panel[tgt].astype(int)
        for variant, cols in variants.items():
            X = panel[cols]
            df = walk_forward_predict(X, y, "logistic", C=LOGISTIC_C_HEADLINE)
            preds[("logistic", variant, horizon)] = df
            df = walk_forward_predict(
                X, y, "xgboost", params=XGB_BEST[(horizon, variant)]
            )
            preds[("xgboost", variant, horizon)] = df
    print(f"  cached {len(preds)} prediction sets")

    # -------------------------------------------------- point-estimate recap
    print("\n=== Point estimates (recap of Section 7) ===")
    point_rows = []
    for (model, variant, horizon), df in preds.items():
        point_rows.append(
            {"model": model, "variant": variant, "horizon_q": horizon, "auc": pooled_auc(df)}
        )
    point_df = pd.DataFrame(point_rows)
    pivot = point_df.pivot_table(
        index="horizon_q", columns=["model", "variant"], values="auc"
    )
    print(pivot.round(3).to_string())

    # ---------------------------------------------------- paired bootstrap
    print("\n=== Paired bootstrap CIs on combined-vs-baseline delta ===")
    boot_rows = []
    for model in ["logistic", "xgboost"]:
        for horizon in [1, 2, 3]:
            base = preds[(model, "baseline_engineered", horizon)]
            comb = preds[(model, "combined_engineered", horizon)]
            point = pooled_auc(comb) - pooled_auc(base)
            stats = paired_bootstrap(base, comb, rng)
            boot_rows.append(
                {
                    "model": model, "horizon_q": horizon,
                    "point_delta": point,
                    **stats,
                }
            )
            print(
                f"  {model:8s} {horizon}Q  Δ={point:+.3f}  "
                f"5%={stats['pct_5']:+.3f}  95%={stats['pct_95']:+.3f}  "
                f"P(Δ>0)={stats['p_positive']:.2f}"
            )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_parquet(PROCESSED / "08_bootstrap.parquet")

    # ----------------------------------------------------- per-recession split
    print("\n=== Per-recession AUCs (±8Q around event anchor) ===")
    per_rec_rows = []
    for (model, variant, horizon), df in preds.items():
        rec = per_recession_auc(df)
        for rec_name, auc in rec.items():
            per_rec_rows.append(
                {
                    "model": model, "variant": variant, "horizon_q": horizon,
                    "recession": rec_name, "auc": auc,
                }
            )
    per_rec_df = pd.DataFrame(per_rec_rows)
    per_rec_df.to_parquet(PROCESSED / "08_per_recession.parquet")

    # Print delta table: combined − baseline per recession
    print("\nDelta (combined − baseline) per recession:")
    for model in ["logistic", "xgboost"]:
        for horizon in [1, 2, 3]:
            line = f"  {model:8s} {horizon}Q  "
            for rec_name in RECESSION_ANCHORS:
                base_auc = per_rec_df[
                    (per_rec_df["model"] == model)
                    & (per_rec_df["variant"] == "baseline_engineered")
                    & (per_rec_df["horizon_q"] == horizon)
                    & (per_rec_df["recession"] == rec_name)
                ]["auc"].iloc[0]
                comb_auc = per_rec_df[
                    (per_rec_df["model"] == model)
                    & (per_rec_df["variant"] == "combined_engineered")
                    & (per_rec_df["horizon_q"] == horizon)
                    & (per_rec_df["recession"] == rec_name)
                ]["auc"].iloc[0]
                d = comb_auc - base_auc
                line += f"{rec_name}:{_fmt(d)}  "
            print(line)

    # -------------------------------------------------------- drop-COVID
    print("\n=== Drop-COVID sensitivity (test dates < 2020-01-01) ===")
    dc_rows = []
    for (model, variant, horizon), df in preds.items():
        auc_full = pooled_auc(df)
        auc_dc = drop_covid_auc(df)
        dc_rows.append(
            {
                "model": model, "variant": variant, "horizon_q": horizon,
                "auc_full": auc_full, "auc_drop_covid": auc_dc,
                "delta": auc_dc - auc_full,
            }
        )
    dc_df = pd.DataFrame(dc_rows)
    dc_df.to_parquet(PROCESSED / "08_drop_covid.parquet")

    print("\nDrop-COVID pooled AUCs:")
    dc_pivot = dc_df.pivot_table(
        index="horizon_q",
        columns=["model", "variant"],
        values="auc_drop_covid",
    )
    print(dc_pivot.round(3).to_string())
    print("\nDrop-COVID deltas (combined - baseline):")
    dc_delta = dc_pivot.copy()
    for model in ["logistic", "xgboost"]:
        dc_delta[(model, "delta")] = (
            dc_pivot[(model, "combined_engineered")]
            - dc_pivot[(model, "baseline_engineered")]
        )
    print(dc_delta.round(3).to_string())

    # --------------------------------------------------- logistic C sensitivity
    print("\n=== Logistic C-sensitivity sweep ===")
    c_rows = []
    for C in LOGISTIC_C_SWEEP:
        for horizon, tgt in zip([1, 2, 3], TARGET_COLS):
            y = panel[tgt].astype(int)
            for variant, cols in variants.items():
                X = panel[cols]
                df = walk_forward_predict(X, y, "logistic", C=C)
                auc = pooled_auc(df)
                c_rows.append(
                    {"C": C, "variant": variant, "horizon_q": horizon, "auc": auc}
                )
    c_df = pd.DataFrame(c_rows)
    c_df.to_parquet(PROCESSED / "08_c_sweep.parquet")
    c_pivot = c_df.pivot_table(
        index="C", columns=["horizon_q", "variant"], values="auc"
    )
    c_delta = c_pivot.copy()
    for horizon in [1, 2, 3]:
        c_delta[(horizon, "delta")] = (
            c_pivot[(horizon, "combined_engineered")]
            - c_pivot[(horizon, "baseline_engineered")]
        )
    c_delta = c_delta.sort_index(axis=1)
    print(c_delta.round(3).to_string())

    # --------------------------------------------------- coefficient inspection
    print("\n=== Logistic coefficient inspection (full-panel fit, C=0.25, 3Q) ===")
    y = panel["Target_3Q_ahead"].astype(int)
    cols = engineered_cols + NETWORK_COLS
    X = panel[cols].to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=int)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(imp.fit_transform(X))
    clf = LogisticRegression(
        penalty="l1", solver="liblinear",
        C=0.25, class_weight="balanced", max_iter=5000,
    )
    clf.fit(X_s, y_arr)
    coefs = pd.Series(clf.coef_[0], index=cols).sort_values(key=lambda s: s.abs(), ascending=False)
    nonzero = coefs[coefs.abs() > 1e-8]
    print(f"  Total non-zero features: {len(nonzero)} / {len(cols)}")
    print("  Top 15 by |coef|:")
    for name, val in nonzero.head(15).items():
        marker = " ← NETWORK" if name in NETWORK_COLS else ""
        print(f"    {name:25s} {val:+.4f}{marker}")
    net_nonzero = [c for c in NETWORK_COLS if abs(coefs[c]) > 1e-8]
    print(f"\n  Network features kept non-zero: {net_nonzero}")
    print(f"  Network features zeroed by L1: {[c for c in NETWORK_COLS if c not in net_nonzero]}")

    # ---------------------------------------------------- write summary markdown
    write_summary(boot_df, per_rec_df, dc_df, c_df, coefs, engineered_cols)


def write_summary(
    boot_df: pd.DataFrame,
    per_rec_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    c_df: pd.DataFrame,
    coefs: pd.Series,
    engineered_cols: list[str],
) -> None:
    def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
        fmt = fmt or {}
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        lines = [header, sep]
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if c in fmt and isinstance(v, (int, float)) and not pd.isna(v):
                    vals.append(fmt[c].format(v))
                elif isinstance(v, float) and pd.isna(v):
                    vals.append("nan")
                elif isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    out = REPORTS / "08_robustness_summary.md"
    lines: list[str] = []
    lines.append("# Section 8 — Statistical Robustness of the Section 7 Comparison\n")
    lines.append(
        "All tests run on the same 162-row 1985-2022 panel with 96 OOS predictions "
        "across 12 walk-forward folds (initial=60, test=8, step=8, reject partial). "
        "The engineered 38-feature panel is the baseline; combined adds the 5 "
        "network features. Logistic is sparse L1 with C=0.25 and "
        "class_weight=\"balanced\"; XGBoost uses per-cell tuned hyperparameters "
        "frozen from Section 7.\n"
    )

    # Bootstrap
    lines.append("## 1. Paired bootstrap CI on combined − baseline delta\n")
    lines.append(f"{N_BOOT} resamples of the 96 pooled OOS predictions, delta "
                 f"computed per resample, percentiles reported.\n")
    bt = boot_df[[
        "model", "horizon_q", "point_delta", "pct_5", "pct_95", "p_positive"
    ]].copy()
    bt.columns = ["model", "horizon", "point Δ", "5th pct", "95th pct", "P(Δ>0)"]
    lines.append(_md(bt, fmt={
        "point Δ": "{:+.3f}", "5th pct": "{:+.3f}",
        "95th pct": "{:+.3f}", "P(Δ>0)": "{:.2f}",
    }))
    lines.append("")

    # Per-recession
    lines.append("## 2. Per-recession pooled AUC (±8Q around each event anchor)\n")
    lines.append(
        "Anchors: 2001-07-01, 2008-12-01, 2020-04-01. NaN = window contains a "
        "single class (not scorable).\n"
    )
    pr = per_rec_df.pivot_table(
        index=["model", "horizon_q"],
        columns=["recession", "variant"],
        values="auc",
    ).round(3)
    pr_str = pr.to_string()
    lines.append("```\n" + pr_str + "\n```")
    lines.append("")

    # Drop-COVID
    lines.append("## 3. Drop-COVID sensitivity\n")
    lines.append(
        "Pooled AUC recomputed after dropping OOS dates ≥ 2020-01-01, removing "
        "the COVID shock folds (which dominate on such a short OOS window).\n"
    )
    dc = dc_df[[
        "model", "variant", "horizon_q", "auc_full", "auc_drop_covid", "delta"
    ]].copy()
    dc.columns = ["model", "variant", "horizon", "full AUC", "ex-COVID AUC", "Δ(drop−full)"]
    lines.append(_md(dc, fmt={
        "full AUC": "{:.3f}", "ex-COVID AUC": "{:.3f}", "Δ(drop−full)": "{:+.3f}",
    }))
    lines.append("")

    # C sweep
    lines.append("## 4. Logistic regularisation sensitivity (C sweep)\n")
    lines.append(
        "Re-run the sparse-L1 logistic walk-forward at six C values. The "
        "combined−baseline delta should stay near zero across the sweep if "
        "the network features are robustly redundant with the engineered "
        "baseline.\n"
    )
    c_pivot = c_df.pivot_table(
        index="C", columns=["horizon_q", "variant"], values="auc"
    )
    for horizon in [1, 2, 3]:
        c_pivot[(horizon, "delta")] = (
            c_pivot[(horizon, "combined_engineered")]
            - c_pivot[(horizon, "baseline_engineered")]
        )
    c_pivot = c_pivot.sort_index(axis=1).round(3)
    lines.append("```\n" + c_pivot.to_string() + "\n```")
    lines.append("")

    # Coefficient inspection
    lines.append("## 5. Logistic coefficient inspection (full-panel fit, C=0.25, 3Q)\n")
    nonzero = coefs[coefs.abs() > 1e-8]
    net_nonzero = [c for c in NETWORK_COLS if abs(coefs[c]) > 1e-8]
    net_zero = [c for c in NETWORK_COLS if c not in net_nonzero]
    lines.append(f"- Total non-zero features: **{len(nonzero)} / {len(coefs)}**")
    lines.append(f"- Network features kept: **{net_nonzero or 'none'}**")
    lines.append(f"- Network features zeroed: **{net_zero}**")
    lines.append("")
    lines.append("**Top 15 non-zero coefficients by |weight|:**\n")
    lines.append("```")
    for name, val in nonzero.head(15).items():
        marker = "  ← NETWORK" if name in NETWORK_COLS else ""
        lines.append(f"{name:25s} {val:+.4f}{marker}")
    lines.append("```")
    lines.append("")

    # Verdict
    lines.append("## 6. Verdict\n")
    verdict_lines = _verdict(boot_df, dc_df, c_df, coefs)
    lines.extend(verdict_lines)

    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


def _verdict(
    boot_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    c_df: pd.DataFrame,
    coefs: pd.Series,
) -> list[str]:
    v: list[str] = []
    # Logistic verdict
    lr = boot_df[boot_df["model"] == "logistic"]
    lr_deltas_at_zero = all(
        abs(lr.iloc[i]["point_delta"]) < 0.01 for i in range(len(lr))
    )
    lr_ci_includes_zero = all(
        lr.iloc[i]["pct_5"] < 0 < lr.iloc[i]["pct_95"] for i in range(len(lr))
    )
    net_nonzero = [c for c in NETWORK_COLS if abs(coefs[c]) > 1e-8]

    v.append("### Logistic (sparse L1, C=0.25)")
    v.append("")
    v.append(
        f"- Point deltas are all < |0.01|: **{lr_deltas_at_zero}**."
    )
    v.append(
        f"- 90% bootstrap CI includes zero at every horizon: "
        f"**{lr_ci_includes_zero}**."
    )
    v.append(
        f"- L1 coefficient inspection at C=0.25: network features kept = "
        f"**{net_nonzero or 'none (all zeroed by L1)'}**."
    )
    v.append("")

    if lr_deltas_at_zero and not net_nonzero:
        v.append(
            "**Logistic reading:** network features are mathematically "
            "discarded by L1 regularisation — the combined model IS the "
            "baseline. The ~0 delta is structural, not statistical noise. "
            "Network features are redundant with the engineered baseline "
            "under L1."
        )
    elif lr_deltas_at_zero and lr_ci_includes_zero:
        v.append(
            "**Logistic reading:** tight null — the combined-vs-baseline "
            "delta is indistinguishable from zero even under resampling. "
            "Network features do not improve the logistic baseline at any "
            "horizon."
        )
    else:
        v.append(
            "**Logistic reading:** result is more nuanced than a clean null "
            "— inspect the bootstrap table and coefficient inspection above."
        )
    v.append("")

    # XGBoost verdict
    xgb = boot_df[boot_df["model"] == "xgboost"]
    v.append("### XGBoost (tuned)")
    v.append("")
    for _, row in xgb.iterrows():
        sig = (row["pct_5"] > 0) or (row["pct_95"] < 0)
        tag = "significant" if sig else "not significant"
        v.append(
            f"- {int(row['horizon_q'])}Q: Δ={row['point_delta']:+.3f}, "
            f"90% CI [{row['pct_5']:+.3f}, {row['pct_95']:+.3f}], "
            f"P(Δ>0)={row['p_positive']:.2f} — **{tag}**"
        )
    v.append("")

    # Drop-COVID narrative
    v.append("### Drop-COVID")
    v.append("")
    dc_3q = dc_df[dc_df["horizon_q"] == 3]
    for _, row in dc_3q.iterrows():
        v.append(
            f"- {row['model']} {row['variant']} 3Q: full={row['auc_full']:.3f}, "
            f"ex-COVID={row['auc_drop_covid']:.3f}, Δ={row['delta']:+.3f}"
        )
    v.append("")

    v.append("### Overall")
    v.append("")
    v.append(
        "See each section above. Any claim in the final write-up should "
        "cite the specific test that supports it; results of different "
        "tests are not interchangeable."
    )
    return v


if __name__ == "__main__":
    main()
