"""Do individually archived model runs close the market's lead?

HRRR, NBM and ECMWF single runs exist only from April 2026, too few months to retrain the
models on. So this test takes the rolling-origin predictions, which were fitted without
these runs, and asks whether they add anything. Date blocks are cross-fitted: each block's
coefficients come from the other blocks, including later ones, which can only favour the
weather model if the runs help. Everything is scored on the same rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import evaluate as ev
from .experiment import _fmt
from .models import clip
from .weather_plus import MIN_SIGMA, bracket_probability

RUN_FORECASTS = ["hrrr_max", "nbmx_max", "ecmwfx_max"]
OTHER_PREDICTORS = ["lamp_max", "nbm_max", "gfs_max"]
OBS = ["obs_max_sofar", "obs_latest"]
BLOCKS = 5


def date_blocks(dates: pd.Series, blocks: int = BLOCKS) -> np.ndarray:
    ordered = np.sort(dates.unique())
    block_of = {d: i * blocks // len(ordered) for i, d in enumerate(ordered)}
    return dates.map(block_of).to_numpy()


def _design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    consensus = frame[[c for c in columns if c not in OBS]].mean(axis=1)
    return frame[columns].apply(lambda c: c.fillna(consensus)).to_numpy(dtype=float)


def run_probability(df: pd.DataFrame) -> pd.Series:
    """Bracket probability from the runs and guidance, centre and spread fitted on other blocks."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    centre = pd.Series(np.nan, index=df.index, dtype=float)
    for (_, _), g in df.groupby(["city", "snapshot"]):
        columns = [c for c in [*RUN_FORECASTS, *OTHER_PREDICTORS, *OBS] if g[c].notna().any()]
        events = g.drop_duplicates("event_id")
        for block in np.unique(g["block"]):
            fit = events[events["block"] != block]
            model = Ridge(alpha=1.0).fit(_design(fit, columns), fit["settled_value"].to_numpy(dtype=float))
            residual = np.abs(fit["settled_value"].to_numpy(dtype=float) - model.predict(_design(fit, columns)))
            sigma = max(residual.mean() * np.sqrt(np.pi / 2), MIN_SIGMA)
            score = g[g["block"] == block]
            mu = model.predict(_design(score, columns))
            floor = score["obs_max_sofar"].round()
            out.loc[score.index] = bracket_probability(score["low"], score["high"], mu, np.full(len(score), sigma), floor)
            centre.loc[score.index] = mu
    df["R_centre"] = centre
    return pd.Series(clip(out), index=df.index)


def run(oos: pd.DataFrame, features: pd.DataFrame, out_dir: Path, n_boot: int = 2000) -> str:
    """`oos`: rolling-origin predictions. `features`: the same rows with the exact-run columns."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["ticker", "snapshot"]
    columns = [c for c in features.columns if c.startswith(("hrrr_", "nbmx_", "ecmwfx_"))]
    df = oos.merge(features[keys + columns], on=keys, how="inner", validate="one_to_one")
    n_all = df["event_id"].nunique()
    df = df.dropna(subset=RUN_FORECASTS).reset_index(drop=True)
    df["block"] = date_blocks(df["target_date"])

    df["R_runs"] = run_probability(df)
    y, snaps, blocks = df["y"].to_numpy(), df["snapshot"].to_numpy(), df["block"].to_numpy()
    # Stage one adds the runs to the weather model, stage two adds that to the market.
    df["F_weather"] = clip(ev.cross_fit_blend(y, df["X_xgb"], df["R_runs"], blocks, snaps))
    df["F_all"] = clip(ev.cross_fit_blend(y, df["F_weather"], df["B_market_norm"], blocks, snaps))
    df.to_parquet(out_dir / "predictions.parquet", index=False)

    scored = ["A_xgb", "X_xgb", "R_runs", "F_weather", "B_market_norm", "CX_xgb", "F_all"]
    scores = []
    for city, g in [*df.groupby("city"), ("ALL", df)]:
        t = ev.score_table(g["y"], {k: g[k] for k in scored}).reset_index()
        t.insert(0, "city", city)
        scores.append(t)
    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(out_dir / "scores.csv", index=False)

    rows = []
    pairs = [
        ("X_xgb", "F_weather", "Exact runs added to weather"),
        ("X_xgb", "B_market_norm", "Market vs weather without the runs"),
        ("F_weather", "B_market_norm", "Market vs weather with the runs"),
        ("B_market_norm", "F_all", "Weather with the runs added to market"),
    ]
    for city, g, groups in [*[(c, x, "event_id") for c, x in df.groupby("city")], ("ALL", df, "target_date")]:
        for a, b, question in pairs:
            rows.append({"city": city, "question": question, **ev.bootstrap_diff(g["y"], g[a], g[b], g[groups], n_boot)})
    compared = pd.DataFrame(rows)
    compared.to_csv(out_dir / "comparisons.csv", index=False)

    rows = []
    for snap, g in df.groupby("snapshot"):
        lead = ev.bootstrap_diff(g["y"], g["F_weather"], g["B_market_norm"], g["target_date"], n_boot)
        rows.append({"snapshot": snap, "F_weather": ev.brier(g["y"], g["F_weather"]), "B_market_norm": ev.brier(g["y"], g["B_market_norm"]), "market_lead": lead["diff"], "lead_lo": lead["ci_lo"], "lead_hi": lead["ci_hi"]})
    snapshots = pd.DataFrame(rows)
    snapshots.to_csv(out_dir / "by_snapshot.csv", index=False)

    events = df.drop_duplicates(["event_id", "snapshot"])
    rows = []
    for snap, g in events.groupby("snapshot"):
        r: dict = {"snapshot": snap}
        for c in [*RUN_FORECASTS, "lamp_max", "nbm_max", "R_centre"]:
            r[c] = float((g[c] - g["settled_value"]).abs().mean())
        for name in ["hrrr", "nbmx", "ecmwfx"]:
            r[f"{name}_age"] = float(g[f"{name}_age_h"].median())
        rows.append(r)
    accuracy = pd.DataFrame(rows)
    accuracy.to_csv(out_dir / "forecast_accuracy.csv", index=False)

    md = f"""# Exact-run test

{df['event_id'].nunique()} of {n_all} out-of-sample city-days have all three runs ({df['target_date'].min()} to {df['target_date'].max()}), {len(df)} rows. {BLOCKS} date blocks, cross-fitted.

## Scores

{_fmt(scores)}

## Brier(baseline) - Brier(challenger)

Positive means the challenger is better. City rows resample city-days, ALL resamples whole dates.

{_fmt(compared[['city', 'question', 'diff', 'ci_lo', 'ci_hi', 'p_value', 'n_events']], '{:.5f}')}

## Market lead over weather with the runs, by snapshot

{_fmt(snapshots)}

## Mean absolute error (deg F) and median run age (hours)

{_fmt(accuracy, '{:.2f}')}
"""
    (out_dir / "summary.md").write_text(md)
    return md
