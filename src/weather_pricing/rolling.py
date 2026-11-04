"""Rolling-origin evaluation: refit on all earlier days, score the next window, repeat.

One train/test split can flatter or punish a model through the period it happens to test.
Here each city gets its own expanding-window fits, so a result has to hold across seasons,
across market maturity and in a second city. Pooled inference across cities resamples whole
dates, because two cities on one date share weather and are not independent.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import models as base_models
from . import weather_plus
from .experiment import _fmt, prepare_sample

INITIAL_MONTHS = 18
STEP_MONTHS = 6
VAL_FRAC = 0.2
SCORED = ["A_xgb", "W_xgb", "X_xgb", "B_market_norm", "C_xgb", "CX_xgb", "F_blend"]

COMPARISONS = [
    # (baseline, challenger, question)
    ("A_xgb", "X_xgb", "Stronger weather vs Open-Meteo weather"),
    ("W_xgb", "X_xgb", "Extra forecast sources in the weather model"),
    ("A_xgb", "B_market_norm", "Market vs Open-Meteo weather"),
    ("W_xgb", "B_market_norm", "Market vs stronger weather"),
    ("X_xgb", "B_market_norm", "Market vs extended weather"),
    ("B_market_norm", "CX_xgb", "Extended weather added to market"),
]


def make_folds(start: date, end: date, initial_months: int = INITIAL_MONTHS, step_months: int = STEP_MONTHS) -> list[tuple[date, date]]:
    """Test windows [first, next first), the first one after `initial_months` of training."""
    first = pd.Timestamp(start) + pd.DateOffset(months=initial_months)
    folds = []
    while first.date() <= end:
        nxt = first + pd.DateOffset(months=step_months)
        folds.append((first.date(), nxt.date()))
        first = nxt
    return folds


def fit_fold(train: pd.DataFrame, test: pd.DataFrame, val_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Every family, fitted only on `train`."""
    preds = base_models.fit_predict(train, test, val_mask)
    preds.update(weather_plus.predict_models(weather_plus.fit_models(train, val_mask), test))
    extended = weather_plus.fit_models(train, val_mask, weather_plus.EXTENDED_FORECASTS, "X", "CX")
    preds.update(weather_plus.predict_models(extended, test))
    return preds


def run_city(df: pd.DataFrame, fit=fit_fold) -> pd.DataFrame:
    """Out-of-sample predictions for one city, with the fold number on each row."""
    d0, d1 = df["target_date"].min(), df["target_date"].max()
    frames = []
    for k, (first, nxt) in enumerate(make_folds(d0, d1)):
        train = df[df["target_date"] < first].copy()
        test = df[(df["target_date"] >= first) & (df["target_date"] < nxt)].copy()
        if test.empty:
            continue
        train_dates = np.sort(train["target_date"].unique())
        val_cut = train_dates[int(len(train_dates) * (1 - VAL_FRAC))]
        print(f"  [{df['city'].iloc[0]}] fold {k}: train {len(train_dates)} days, test from {first}", flush=True)
        preds = fit(train, test, (train["target_date"] >= val_cut).to_numpy())
        for name, p in preds.items():
            test[name] = p
        test["fold"] = k
        frames.append(test)
    return pd.concat(frames, ignore_index=True)


def forward_blend(out: pd.DataFrame, weather: str = "X_xgb", market: str = "B_market_norm") -> pd.Series:
    """Blend weather and market per snapshot, with coefficients from earlier folds only.

    Unlike a cross-fitted blend, no later data helps, so this could have been run live. The
    first fold has nothing to learn from and stays NaN.
    """
    blended = pd.Series(np.nan, index=out.index, dtype=float)
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    for (_, snap), g in out.groupby(["city", "snapshot"]):
        X = np.column_stack([np.ones(len(g)), lg(g[weather].to_numpy()), lg(g[market].to_numpy())])
        y = g["y"].to_numpy(dtype=float)
        codes = pd.factorize(g["event_id"])[0]
        folds = g["fold"].to_numpy()
        for k in np.unique(folds)[1:]:
            fit, score = folds < k, folds == k
            beta, _ = ev._logit_cluster(X[fit], y[fit], codes[fit])
            blended.loc[g.index[score]] = 1 / (1 + np.exp(-X[score] @ beta))
    return blended


def _compare(frame: pd.DataFrame, groups: str, n_boot: int) -> pd.DataFrame:
    rows = []
    for a, b, question in COMPARISONS:
        d = ev.bootstrap_diff(frame["y"], frame[a], frame[b], frame[groups], n_boot)
        rows.append({"question": question, "baseline": a, "challenger": b, **d})
    return pd.DataFrame(rows)


def _fold_table(out: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for (city, fold), g in out.groupby(["city", "fold"]):
        d = ev.bootstrap_diff(g["y"], g["X_xgb"], g["B_market_norm"], g["event_id"], n_boot)
        added = ev.bootstrap_diff(g["y"], g["B_market_norm"], g["CX_xgb"], g["event_id"], n_boot)
        r = {"city": city, "fold": fold, "from": str(g["target_date"].min()), "to": str(g["target_date"].max()), "city_days": g["event_id"].nunique()}
        for k in ["A_xgb", "W_xgb", "X_xgb", "B_market_norm", "CX_xgb"]:
            r[k] = ev.brier(g["y"], g[k])
        r.update(market_lead=d["diff"], lead_lo=d["ci_lo"], lead_hi=d["ci_hi"], added=added["diff"], added_lo=added["ci_lo"], added_hi=added["ci_hi"])
        rows.append(r)
    return pd.DataFrame(rows)


def _snapshot_table(out: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for (city, snap), g in out.groupby(["city", "snapshot"]):
        d = ev.bootstrap_diff(g["y"], g["X_xgb"], g["B_market_norm"], g["event_id"], n_boot)
        rows.append(
            {
                "city": city,
                "snapshot": snap,
                "X_xgb": ev.brier(g["y"], g["X_xgb"]),
                "B_market_norm": ev.brier(g["y"], g["B_market_norm"]),
                "market_lead": d["diff"],
                "lead_lo": d["ci_lo"],
                "lead_hi": d["ci_hi"],
            }
        )
    return pd.DataFrame(rows)


def _encompassing_row(g: pd.DataFrame, groups: pd.Series, **labels) -> dict:
    t = ev.encompassing_test(g["y"], g["X_xgb"], g["B_market_norm"], groups)
    return {
        **labels,
        "b_weather": t.loc["logit_weather", "coef"],
        "se_weather": t.loc["logit_weather", "se"],
        "p_weather": t.loc["logit_weather", "p_value"],
        "b_market": t.loc["logit_market", "coef"],
        "se_market": t.loc["logit_market", "se"],
    }


def _encompassing(out: pd.DataFrame) -> pd.DataFrame:
    rows = [_encompassing_row(g, g["event_id"], city=city, fold="all") for city, g in out.groupby("city")]
    rows.append(_encompassing_row(out, out["target_date"].astype(str), city="ALL", fold="all"))
    for fold, g in out.groupby("fold"):
        rows.append(_encompassing_row(g, g["target_date"].astype(str), city="ALL", fold=str(fold)))
    return pd.DataFrame(rows)


def liquidity_bucket(frame: pd.DataFrame) -> np.ndarray:
    """The liquidity cut of the main experiment, plus how much had traded by the snapshot."""
    spread = pd.cut(frame["spread"], [-1, 0.02, 0.05, 1], labels=["spread<=2c", "2-5c", ">5c"]).astype(str)
    return np.where(frame["stale"], "stale/no quote", spread)


def _liquidity_table(out: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    frame = out.assign(liquidity=liquidity_bucket(out))
    for bucket, g in frame.groupby("liquidity"):
        lead = ev.bootstrap_diff(g["y"], g["X_xgb"], g["B_market_norm"], g["target_date"], n_boot)
        added = ev.bootstrap_diff(g["y"], g["B_market_norm"], g["CX_xgb"], g["target_date"], n_boot)
        rows.append(
            {
                "liquidity": bucket,
                "rows": len(g),
                "X_xgb": ev.brier(g["y"], g["X_xgb"]),
                "B_market_norm": ev.brier(g["y"], g["B_market_norm"]),
                "market_lead": lead["diff"],
                "lead_lo": lead["ci_lo"],
                "lead_hi": lead["ci_hi"],
                "added": added["diff"],
                "added_lo": added["ci_lo"],
                "added_hi": added["ci_hi"],
            }
        )
    return pd.DataFrame(rows)


def _maturity_table(out: pd.DataFrame) -> pd.DataFrame:
    """Market depth and the market's lead in each fold, across both cities."""
    rows = []
    for fold, g in out.groupby("fold"):
        rows.append(
            {
                "fold": fold,
                "from": str(g["target_date"].min()),
                "median_spread": float(g["spread"].median()),
                "median_cum_volume": float(g["cum_volume"].median()),
                "stale_share": float(g["stale"].mean()),
                "market_lead": ev.brier(g["y"], g["X_xgb"]) - ev.brier(g["y"], g["B_market_norm"]),
                "added": ev.brier(g["y"], g["B_market_norm"]) - ev.brier(g["y"], g["CX_xgb"]),
            }
        )
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, out_dir: Path, n_boot: int = 2000, fit=fit_fold) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_sample(df)
    out = pd.concat([run_city(g.reset_index(drop=True), fit) for _, g in df.groupby("city")], ignore_index=True)
    out["F_blend"] = forward_blend(out)
    out.to_parquet(out_dir / "oos_predictions.parquet", index=False)
    return report(out, out_dir, n_boot)


def report(out: pd.DataFrame, out_dir: Path, n_boot: int = 2000) -> str:
    """Tables and summary from saved predictions, so a new cut does not need a refit."""
    out_dir = Path(out_dir)
    scores = []
    for city, g in [*out.groupby("city"), ("ALL", out)]:
        t = ev.score_table(g["y"], {k: g[k] for k in SCORED if k != "F_blend"}).reset_index()
        t.insert(0, "city", city)
        scores.append(t)
    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(out_dir / "scores.csv", index=False)

    compared = []
    for city, g in out.groupby("city"):
        compared.append(_compare(g, "event_id", n_boot).assign(city=city))
    compared.append(_compare(out, "target_date", n_boot).assign(city="ALL"))
    compared = pd.concat(compared, ignore_index=True)
    compared.to_csv(out_dir / "comparisons.csv", index=False)

    folds, snapshots, encompassing = _fold_table(out, n_boot), _snapshot_table(out, n_boot), _encompassing(out)
    folds.to_csv(out_dir / "by_fold.csv", index=False)
    snapshots.to_csv(out_dir / "by_snapshot.csv", index=False)
    encompassing.to_csv(out_dir / "encompassing.csv", index=False)
    liquidity, maturity = _liquidity_table(out, n_boot), _maturity_table(out)
    liquidity.to_csv(out_dir / "by_liquidity.csv", index=False)
    maturity.to_csv(out_dir / "by_maturity.csv", index=False)

    scored_blend = out.dropna(subset=["F_blend"])
    rows = []
    for city, g, groups in [*[(c, x, "event_id") for c, x in scored_blend.groupby("city")], ("ALL", scored_blend, "target_date")]:
        d = ev.bootstrap_diff(g["y"], g["B_market_norm"], g["F_blend"], g[groups], n_boot)
        rows.append({"city": city, "F_blend": ev.brier(g["y"], g["F_blend"]), "B_market_norm": ev.brier(g["y"], g["B_market_norm"]), **d})
    blend = pd.DataFrame(rows)
    blend.to_csv(out_dir / "forward_blend.csv", index=False)

    md = f"""# Rolling-origin evaluation

Expanding window: {INITIAL_MONTHS} months of first training, then {STEP_MONTHS}-month test windows, refit each time per city. {out['event_id'].nunique()} out-of-sample city-days, {len(out)} rows.

## Scores

{_fmt(scores)}

## Brier(baseline) - Brier(challenger)

Positive means the challenger is better. City rows resample city-days, ALL resamples whole dates.

{_fmt(compared[['city', 'question', 'diff', 'ci_lo', 'ci_hi', 'p_value', 'n_events']], '{:.5f}')}

## By fold

market_lead is Brier(X_xgb) - Brier(market). added is Brier(market) - Brier(CX_xgb).

{_fmt(folds)}

## By snapshot

{_fmt(snapshots)}

## Forecast-encompassing regressions on all out-of-sample rows

Rows for a single fold pool both cities and resample whole dates.

{_fmt(encompassing)}

## By liquidity at the snapshot (both cities, resampling whole dates)

{_fmt(liquidity)}

## Market maturity by fold (both cities)

{_fmt(maturity)}

## Forward blend of X_xgb and the market

Coefficients come from earlier folds only. diff is Brier(market) - Brier(blend).

{_fmt(blend[['city', 'F_blend', 'B_market_norm', 'diff', 'ci_lo', 'ci_hi', 'p_value', 'n_events']], '{:.5f}')}
"""
    (out_dir / "summary.md").write_text(md)
    return md
