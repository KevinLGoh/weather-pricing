"""End-to-end experiment: chronological split, models A/B/C, scoring, inference, valuation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifacts
from . import evaluate as ev
from . import valuation as val
from .config import RESULTS_DIR
from .models import fit_models, predict_models

MAIN = ["R_gaussian_rule", "A_logit", "A_xgb", "B_market_mid", "B_market_norm", "B_logit", "C_logit", "C_xgb"]

COMPARISONS = [
    # (baseline, challenger, question)
    ("A_logit", "C_logit", "Market data added to weather (logistic)"),
    ("A_xgb", "C_xgb", "Market data added to weather (XGBoost)"),
    ("B_market_norm", "C_logit", "Weather data added to market (logistic)"),
    ("B_market_norm", "C_xgb", "Weather data added to market (XGBoost)"),
    ("A_xgb", "B_market_norm", "Market alone vs best weather-only"),
    ("B_market_mid", "B_market_norm", "Renormalising quotes within an event"),
]


# Scored in the robustness cuts and calibration whenever a run includes them.
ROBUSTNESS_EXTRA = ["W_xgb", "X_xgb"]

# Columns of the test-set predictions table, besides one column per model. The app shows
# these, so they are also what app_artifacts/predictions.parquet carries.
APP_COLUMNS = [
    "city",
    "event_id",
    "event_ticker",
    "ticker",
    "target_date",
    "snapshot",
    "lead_hours",
    "strike_type",
    "low",
    "high",
    "bid",
    "ask",
    "mid",
    "mid_norm",
    "spread",
    "stale",
    "settle_source",
    "settled_value",
    "y",
]


@dataclass
class ExtraModels:
    """Further models to score alongside A, B and C, on the same rows and split."""

    fit_predict: Callable  # (train, test, val_mask) -> (fitted models, test probabilities)
    scored: list[str]
    comparisons: list[tuple[str, str, str]]
    encompassing_model: str | None = None
    fitted: dict = field(default_factory=dict)  # filled in by run()


def prepare_sample(df: pd.DataFrame) -> pd.DataFrame:
    """The rows every model is scored on, with the settled temperature filled in."""
    # Kalshi omits the settled temperature on most pre-2025 contracts (and keeps it mainly on
    # the winning bracket), so dropping those rows would bias the sample towards Yes. NOAA's
    # TMAX matches the settled value ~99% of the time; it only feeds the Gaussian error fit,
    # while labels always come from Kalshi's result.
    df = df.assign(settled_value=df["settled_value"].fillna(df["noaa_tmax"]))
    # One common sample for every model, so score differences are not sample differences.
    df = df.dropna(subset=["fc_tmax", "mid", "mid_norm", "settled_value"]).reset_index(drop=True)
    df["event_id"] = df["city"] + ":" + df["target_date"].astype(str)
    return df


def chronological_split(df: pd.DataFrame, test_frac: float, val_frac: float = 0.2):
    dates = np.sort(df["target_date"].unique())
    cut = dates[int(len(dates) * (1 - test_frac))]
    train, test = df[df["target_date"] < cut].copy(), df[df["target_date"] >= cut].copy()
    train_dates = np.sort(train["target_date"].unique())
    val_cut = train_dates[int(len(train_dates) * (1 - val_frac))]
    val_mask = (train["target_date"] >= val_cut).to_numpy()
    return train, test, val_mask


def run(
    df: pd.DataFrame,
    test_frac: float = 0.3,
    n_boot: int = 2000,
    rate: float = 0.04,
    wang_lambda: float = 0.0,
    save_artifacts: bool = False,
    results_dir: Path | None = None,
    extra: ExtraModels | None = None,
) -> str:
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    n0 = len(df)
    data_date_range = [str(df["target_date"].min()), str(df["target_date"].max())]
    cities = sorted(str(c) for c in df["city"].unique())
    df = prepare_sample(df)
    train, test, val_mask = chronological_split(df, test_frac)
    models = fit_models(train, val_mask)
    preds = predict_models(models, test)
    if extra is not None:
        extra.fitted, extra_preds = extra.fit_predict(train, test, val_mask)
        preds.update(extra_preds)
    extra_scored = extra.scored if extra is not None else []
    y, groups = test["y"].to_numpy(), test["event_id"].to_numpy()

    out = test[APP_COLUMNS].copy()
    for k, p in preds.items():
        out[k] = p
    out.to_parquet(results_dir / "test_predictions.parquet", index=False)

    scores = ev.score_table(y, {k: preds[k] for k in [*MAIN, *extra_scored]})
    scores.to_csv(results_dir / "scores.csv")

    boot = pd.DataFrame(
        [
            {"baseline": a, "challenger": b, "question": q, **ev.bootstrap_diff(y, preds[a], preds[b], groups, n_boot)}
            for a, b, q in [*COMPARISONS, *(extra.comparisons if extra is not None else [])]
        ]
    )
    boot.to_csv(results_dir / "bootstrap.csv", index=False)

    enc = ev.encompassing_test(y, preds["A_logit"], preds["B_market_norm"], groups)
    enc.to_csv(results_dir / "encompassing.csv")
    if extra is not None and extra.encompassing_model:
        ev.encompassing_test(y, preds[extra.encompassing_model], preds["B_market_norm"], groups).to_csv(
            results_dir / "encompassing_extended.csv"
        )

    def by(col: str, frame: pd.DataFrame = out) -> pd.DataFrame:
        rows = []
        for key, g in frame.groupby(col, observed=True):
            r = {col: key, "n": len(g), "events": g["event_id"].nunique()}
            for k in ["A_xgb", "B_market_norm", "C_logit", "C_xgb", *[m for m in ROBUSTNESS_EXTRA if m in preds]]:
                r[k] = ev.brier(g["y"], g[k])
            r["A_minus_C_xgb"] = r["A_xgb"] - r["C_xgb"]
            rows.append(r)
        return pd.DataFrame(rows)

    by_lead = by("snapshot")
    out["liquidity"] = np.where(out["stale"], "stale/no quote", pd.cut(out["spread"], [-1, 0.02, 0.05, 1], labels=["spread<=2c", "2-5c", ">5c"]).astype(str))
    by_liq = by("liquidity")
    by_city = by("city")
    by_source = by("settle_source")
    pd.concat([by_lead.assign(cut="lead"), by_liq.assign(cut="liquidity"), by_city.assign(cut="city"), by_source.assign(cut="settle_source")]).to_csv(
        results_dir / "robustness.csv", index=False
    )

    rel = pd.concat(
        [
            ev.reliability(y, preds[k]).assign(model=k)
            for k in ["A_xgb", *[m for m in ROBUSTNESS_EXTRA if m in preds], "B_market_norm", "C_xgb"]
        ]
    )
    rel.to_csv(results_dir / "reliability.csv", index=False)

    # Valuation: the note pays ~8h after the local day ends (Kalshi settles next morning).
    years = val.horizon_years(out["lead_hours"].to_numpy())
    note = val.Note()
    v = {k: val.value(preds[k], years, note, rate, wang_lambda) for k in ["A_xgb", "B_market_norm", "C_xgb"]}
    realised = val.realised_value(y, years, note, rate)
    dv = v["C_xgb"] - v["A_xgb"]
    valuation = {
        "note": {"principal": note.principal, "loss_fraction": note.loss_fraction, "coupon": note.coupon},
        "rate": rate,
        "wang_lambda": wang_lambda,
        "mean_abs_value_change_A_to_C": float(np.mean(np.abs(dv))),
        "p90_abs_value_change_A_to_C": float(np.quantile(np.abs(dv), 0.9)),
        "share_rows_change_over_$5": float(np.mean(np.abs(dv) > 5)),
        "rmse_vs_realised": {k: float(np.sqrt(np.mean((vk - realised) ** 2))) for k, vk in v.items()},
        "mean_value": {k: float(vk.mean()) for k, vk in v.items()},
        "mean_realised": float(realised.mean()),
        "wang_grid_mean_value_C_xgb": {
            str(lam): float(val.value(preds["C_xgb"], years, note, rate, lam).mean()) for lam in [0.0, 0.1, 0.25, 0.5]
        },
    }
    (results_dir / "valuation.json").write_text(json.dumps(valuation, indent=2))

    meta = {
        "rows_in": n0,
        "rows_used": len(df),
        "train": {"rows": len(train), "events": train["event_id"].nunique(), "dates": [str(train["target_date"].min()), str(train["target_date"].max())]},
        "test": {"rows": len(test), "events": test["event_id"].nunique(), "dates": [str(test["target_date"].min()), str(test["target_date"].max())]},
        "test_base_rate": float(y.mean()),
    }
    (results_dir / "split.json").write_text(json.dumps(meta, indent=2))

    md = render_summary(meta, scores, boot, enc, by_lead, by_liq, by_city, by_source, valuation)
    (results_dir / "summary.md").write_text(md)

    if save_artifacts:
        info = {
            "data_date_range": data_date_range,
            "cities": cities,
            "rows_in": meta["rows_in"],
            "rows_used": meta["rows_used"],
            "train": meta["train"],
            "test": meta["test"],
            "settings": {"test_frac": test_frac, "n_boot": n_boot, "rate": rate, "wang_lambda": wang_lambda},
        }
        # Only the app's columns and the model probabilities; not the liquidity helper column.
        artifacts.save(artifacts.ARTIFACTS_DIR, models, out[APP_COLUMNS + list(preds)], results_dir, info)
    return md


def _fmt(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(x) if isinstance(x, (float, np.floating)) else str(x) for x in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_summary(meta, scores, boot, enc, by_lead, by_liq, by_city, by_source, valuation) -> str:
    return f"""# Results summary

Train: {meta['train']['events']} city-days ({meta['train']['dates'][0]} to {meta['train']['dates'][1]}), {meta['train']['rows']} rows.
Test: {meta['test']['events']} city-days ({meta['test']['dates'][0]} to {meta['test']['dates'][1]}), {meta['test']['rows']} rows. Base rate {meta['test_base_rate']:.3f}.
Rows used {meta['rows_used']} of {meta['rows_in']} (dropped rows lack a forecast, quote or settlement value).

## Out-of-sample scores (lower is better)

{_fmt(scores.reset_index())}

## Event-clustered bootstrap: Brier(baseline) - Brier(challenger)

Positive means the challenger is better. 95% CI from resampling whole city-days.

{_fmt(boot[['question', 'baseline', 'challenger', 'diff', 'ci_lo', 'ci_hi', 'p_value', 'n_events']], '{:.5f}')}

## Forecast-encompassing regression (test set, clustered by city-day)

logit P(yes) = a + b1 * logit(p_weather [A_logit]) + b2 * logit(p_market [B_market_norm])

{_fmt(enc.reset_index().rename(columns={'index': 'term'}))}

## Brier by snapshot time

{_fmt(by_lead)}

## Brier by liquidity

{_fmt(by_liq)}

## Brier by city

{_fmt(by_city)}

## Brier by settlement source

{_fmt(by_source)}

## Valuation of the hypothetical note (N=$1,000, L=50%)

- Mean |V(C_xgb) - V(A_xgb)|: ${valuation['mean_abs_value_change_A_to_C']:.2f} (90th pct ${valuation['p90_abs_value_change_A_to_C']:.2f}); share of rows moving > $5: {valuation['share_rows_change_over_$5']:.1%}
- RMSE of model value vs realised payoff: {', '.join(f'{k} ${v:.2f}' for k, v in valuation['rmse_vs_realised'].items())}
- Mean value C_xgb by Wang lambda: {', '.join(f'{k}: ${v:.2f}' for k, v in valuation['wang_grid_mean_value_C_xgb'].items())}
"""
