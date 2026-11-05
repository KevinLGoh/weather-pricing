"""Does the market still add information against a stronger weather model?

Applies the main experiment's sample rules and chronological split to whichever dataset it
is given, so W and CW can be compared with A, B and C directly. It writes to its own results folder and leaves the app's artifacts
alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import models as base_models
from . import valuation as val
from . import weather_plus
from .experiment import _fmt, chronological_split, prepare_sample
from .guidance import FEATURES

SCORED = [
    "R_gaussian_rule",
    "A_logit",
    "A_xgb",
    "W_rule",
    "W_logit",
    "W_xgb",
    "B_market_norm",
    "C_xgb",
    "CW_logit",
    "CW_xgb",
    "S_blend",
]

COMPARISONS = [
    # (baseline, challenger, question)
    ("A_xgb", "W_xgb", "Stronger weather vs Open-Meteo weather (XGBoost)"),
    ("W_logit", "B_market_norm", "Market alone vs stronger weather (logistic)"),
    ("W_xgb", "B_market_norm", "Market alone vs stronger weather (XGBoost)"),
    ("W_xgb", "CW_xgb", "Market data added to stronger weather (XGBoost)"),
    ("B_market_norm", "CW_logit", "Stronger weather added to market (logistic)"),
    ("B_market_norm", "CW_xgb", "Stronger weather added to market (XGBoost)"),
    ("C_xgb", "CW_xgb", "Stronger weather in the combined model (XGBoost)"),
    ("B_market_norm", "S_blend", "Stronger weather added to market (cross-fitted blend)"),
]

BY_SNAPSHOT = [
    ("A_xgb", "W_xgb", "weather_gain"),
    ("W_xgb", "B_market_norm", "market_vs_weather"),
    ("B_market_norm", "CW_xgb", "weather_added_to_market"),
    ("B_market_norm", "S_blend", "blend_vs_market"),
]

BLEND_BLOCKS = 5


def run(df: pd.DataFrame, out_dir: Path, test_frac: float = 0.3, n_boot: int = 2000, rate: float = 0.04) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_sample(df)
    train, test, val_mask = chronological_split(df, test_frac)

    preds = base_models.fit_predict(train, test, val_mask)
    plus = weather_plus.fit_models(train, val_mask)
    preds.update(weather_plus.predict_models(plus, test))
    y, groups = test["y"].to_numpy(), test["event_id"].to_numpy()
    preds["S_blend"] = _blend(test, y, preds)

    out = test[["city", "event_id", "ticker", "target_date", "snapshot", "lead_hours", "settle_source", "y"]].copy()
    for k, p in preds.items():
        out[k] = p
    out.to_parquet(out_dir / "test_predictions.parquet", index=False)

    scores = ev.score_table(y, {k: preds[k] for k in SCORED})
    scores.to_csv(out_dir / "scores.csv")

    boot = pd.DataFrame(
        [
            {"baseline": a, "challenger": b, "question": q, **ev.bootstrap_diff(y, preds[a], preds[b], groups, n_boot)}
            for a, b, q in COMPARISONS
        ]
    )
    boot.to_csv(out_dir / "bootstrap.csv", index=False)

    enc_rows = []
    for name in ["W_logit", "W_xgb"]:
        enc_rows.append(_encompassing("all", name, y, preds[name], preds["B_market_norm"], groups))
        for snap in sorted(test["snapshot"].unique()):
            m = (test["snapshot"] == snap).to_numpy()
            enc_rows.append(_encompassing(snap, name, y[m], preds[name][m], preds["B_market_norm"][m], groups[m]))
    enc = pd.DataFrame(enc_rows)
    enc.to_csv(out_dir / "encompassing.csv", index=False)

    snap_rows = []
    for snap, g in out.groupby("snapshot"):
        r = {"snapshot": snap, "events": g["event_id"].nunique()}
        for k in ["A_xgb", "W_xgb", "B_market_norm", "CW_xgb", "S_blend"]:
            r[k] = ev.brier(g["y"], g[k])
        for a, b, label in BY_SNAPSHOT:
            d = ev.bootstrap_diff(g["y"], g[a], g[b], g["event_id"], n_boot)
            r[label], r[f"{label}_lo"], r[f"{label}_hi"] = d["diff"], d["ci_lo"], d["ci_hi"]
        snap_rows.append(r)
    by_snapshot = pd.DataFrame(snap_rows)
    by_snapshot.to_csv(out_dir / "by_snapshot.csv", index=False)

    source_rows = []
    for source, g in out.groupby("settle_source"):
        r = {"settle_source": source, "events": g["event_id"].nunique()}
        for k in ["A_xgb", "W_xgb", "B_market_norm", "CW_xgb"]:
            r[k] = ev.brier(g["y"], g[k])
        source_rows.append(r)
    by_source = pd.DataFrame(source_rows)
    by_source.to_csv(out_dir / "by_source.csv", index=False)

    accuracy = forecast_accuracy(test, plus["W_rule"])
    accuracy.to_csv(out_dir / "forecast_accuracy.csv", index=False)

    audit = feature_audit(df, test)
    (out_dir / "feature_audit.json").write_text(json.dumps(audit, indent=2))

    years = val.horizon_years(out["lead_hours"].to_numpy())
    v = {k: val.value(preds[k], years, rate=rate) for k in ["A_xgb", "W_xgb", "B_market_norm"]}
    valuation = {
        "mean_abs_value_gap_A_vs_market": float(np.mean(np.abs(v["A_xgb"] - v["B_market_norm"]))),
        "mean_abs_value_gap_W_vs_market": float(np.mean(np.abs(v["W_xgb"] - v["B_market_norm"]))),
    }
    (out_dir / "valuation.json").write_text(json.dumps(valuation, indent=2))

    md = render(train, test, scores, boot, enc, by_snapshot, by_source, accuracy, audit, valuation)
    (out_dir / "summary.md").write_text(md)
    return md


def _blend(test: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray]) -> np.ndarray:
    """W_xgb and the market blended per snapshot, each date block fitted on the others."""
    dates = np.sort(test["target_date"].unique())
    block_of = {d: i * BLEND_BLOCKS // len(dates) for i, d in enumerate(dates)}
    blocks = test["target_date"].map(block_of).to_numpy()
    p = ev.cross_fit_blend(y, preds["W_xgb"], preds["B_market_norm"], blocks, test["snapshot"].to_numpy())
    return base_models.clip(p)


def _encompassing(snapshot: str, name: str, y, p_weather, p_market, groups) -> dict:
    t = ev.encompassing_test(y, p_weather, p_market, groups)
    return {
        "snapshot": snapshot,
        "weather_model": name,
        "b_weather": t.loc["logit_weather", "coef"],
        "se_weather": t.loc["logit_weather", "se"],
        "p_weather": t.loc["logit_weather", "p_value"],
        "b_market": t.loc["logit_market", "coef"],
        "se_market": t.loc["logit_market", "se"],
        "p_market": t.loc["logit_market", "p_value"],
    }


def forecast_accuracy(test: pd.DataFrame, rule: weather_plus.NowcastRule) -> pd.DataFrame:
    """Mean absolute error (deg F) of each temperature forecast by snapshot, per city-day."""
    events = test.drop_duplicates(["event_ticker", "snapshot"]).copy()
    events["W_centre"] = rule.mu_sigma(events)[0]
    rows = []
    for snap, g in events.groupby("snapshot"):
        r: dict = {"snapshot": snap}
        for col in ["fc_tmax", "gfs_max", "nbm_max", "lamp_max", "W_centre"]:
            err = (g[col] - g["settled_value"]).abs().dropna()
            r[col] = float(err.mean()) if len(err) else float("nan")
            r[f"{col}_n"] = len(err)
        rows.append(r)
    return pd.DataFrame(rows)


def feature_audit(df: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Coverage of the new features, and how often the observed floor was wrong."""
    events = df.drop_duplicates(["event_ticker", "snapshot"])
    test_events = test.drop_duplicates(["event_ticker", "snapshot"])
    floored = events.dropna(subset=["obs_max_sofar", "settled_value"])
    return {
        "coverage_all_rows": {c: float(events[c].notna().mean()) for c in FEATURES},
        "coverage_by_snapshot_test": {
            snap: {c: float(g[c].notna().mean()) for c in ["gfs_max", "nbm_max", "lamp_max", "obs_max_sofar"]}
            for snap, g in test_events.groupby("snapshot")
        },
        "median_age_hours": {
            snap: {c: float(g[c].median()) for c in ["gfs_age_h", "nbm_age_h", "lamp_age_h", "obs_age_h"]}
            for snap, g in test_events.groupby("snapshot")
        },
        "settled_below_observed_floor": float((floored["obs_max_sofar"].round() > floored["settled_value"]).mean()),
        "floor_checks": len(floored),
        "contract_rows_paying_below_floor": _pays_below_floor(df),
    }


def _pays_below_floor(df: pd.DataFrame) -> int:
    """Rows where a contract paid Yes although its whole bracket sits below the floor."""
    d = df.dropna(subset=["obs_max_sofar"])
    below = d["high"] < d["obs_max_sofar"].round()
    return int((below & (d["y"] == 1)).sum())


def render(train, test, scores, boot, enc, by_snapshot, by_source, accuracy, audit, valuation) -> str:
    return f"""# Stronger weather baseline

Cities: {', '.join(sorted(train['city'].unique()))}. Sample rules and chronological split as in the main experiment, so these models compare directly with A, B and C. Train: {train['event_id'].nunique()} city-days ({train['target_date'].min()} to {train['target_date'].max()}).
Test: {test['event_id'].nunique()} city-days ({test['target_date'].min()} to {test['target_date'].max()}), {len(test)} rows.

## Out-of-sample scores (lower is better)

{_fmt(scores.reset_index())}

## Event-clustered bootstrap: Brier(baseline) - Brier(challenger)

Positive means the challenger is better.

{_fmt(boot[['question', 'baseline', 'challenger', 'diff', 'ci_lo', 'ci_hi', 'p_value', 'n_events']], '{:.5f}')}

## Forecast-encompassing regressions against the market

logit P(yes) = a + b_weather * logit(p_weather) + b_market * logit(B_market_norm), clustered by city-day.

{_fmt(enc)}

## Brier and differences by snapshot (95% CIs)

{_fmt(by_snapshot)}

## Brier by settlement source

{_fmt(by_source)}

## Mean absolute error of the temperature forecasts (deg F, per city-day)

{_fmt(accuracy[['snapshot', 'fc_tmax', 'gfs_max', 'nbm_max', 'lamp_max', 'W_centre']], '{:.2f}')}

## Feature audit

```json
{json.dumps(audit, indent=2)}
```

## Value gap to the market (N=$1,000, L=50%)

- Open-Meteo weather (A_xgb): ${valuation['mean_abs_value_gap_A_vs_market']:.2f} mean absolute gap
- Stronger weather (W_xgb): ${valuation['mean_abs_value_gap_W_vs_market']:.2f} mean absolute gap
"""
