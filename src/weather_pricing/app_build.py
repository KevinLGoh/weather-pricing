"""Build app_artifacts/ for every city, with the stronger weather models.

Each city gets its own fits on its own chronological split, in the file formats the single
run writes, so the app reads them the same way. The tables across cities come from the
rolling-origin and exact-run results, which must have been produced first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import artifacts, weather_plus
from . import experiment as exp
from .config import PROCESSED_DIR, RESULTS_DIR
from .guidance import enrich
from .guidance_plus import enrich_extended
from .inference import ALL_MODELS

EXTRA_SCORED = ["W_rule", "W_logit", "W_xgb", "X_logit", "X_xgb", "CW_xgb", "CX_xgb"]
EXTRA_COMPARISONS = [
    ("A_xgb", "W_xgb", "Stronger weather vs Open-Meteo weather"),
    ("W_xgb", "X_xgb", "Extra forecast sources in the weather model"),
    ("W_xgb", "B_market_norm", "Market vs stronger weather"),
    ("X_xgb", "B_market_norm", "Market vs extended weather"),
    ("B_market_norm", "CW_xgb", "Stronger weather added to market"),
    ("B_market_norm", "CX_xgb", "Extended weather added to market"),
]
# Where each table under finding/ comes from.
ROLLING_TABLES = {
    "rolling_comparisons.csv": "comparisons.csv",
    "rolling_by_fold.csv": "by_fold.csv",
    "rolling_by_maturity.csv": "by_maturity.csv",
    "rolling_by_liquidity.csv": "by_liquidity.csv",
    "rolling_by_snapshot.csv": "by_snapshot.csv",
    "rolling_encompassing.csv": "encompassing.csv",
    "rolling_forward_blend.csv": "forward_blend.csv",
}
EXACT_TABLES = {"exact_comparisons.csv": "comparisons.csv", "exact_by_snapshot.csv": "by_snapshot.csv"}
LADDER_NAME = "ladder.csv"


def fit_predict_extra(train: pd.DataFrame, test: pd.DataFrame, val_mask) -> tuple[dict, dict]:
    stronger = weather_plus.fit_models(train, val_mask)
    extended = weather_plus.fit_models(train, val_mask, weather_plus.EXTENDED_FORECASTS, "X", "CX")
    preds = {**weather_plus.predict_models(stronger, test), **weather_plus.predict_models(extended, test)}
    return {"stronger": stronger, "extended": extended}, preds


def _row(test: str, sample: str, weather: str, row: pd.Series) -> dict:
    return {"test": test, "sample": sample, "weather_model": weather, "lead": row["diff"], "ci_lo": row["ci_lo"], "ci_hi": row["ci_hi"]}


def _bootstrap_row(results: Path, baseline: str, challenger: str) -> pd.Series:
    boot = pd.read_csv(results / "bootstrap.csv")
    return boot[(boot["baseline"] == baseline) & (boot["challenger"] == challenger)].iloc[0]


def ladder(city_results: dict[str, Path], rolling_dir: Path, exact_dir: Path) -> pd.DataFrame:
    """The market's lead over the best weather model at each stage of testing."""
    rows = []
    if "NY" in city_results:
        rows.append(_row("Original", "New York, one split", "Open-Meteo", _bootstrap_row(city_results["NY"], "A_xgb", "B_market_norm")))
        rows.append(_row("Stronger baseline", "New York, one split", "NWS guidance and live readings", _bootstrap_row(city_results["NY"], "W_xgb", "B_market_norm")))
    if "CHI" in city_results:
        rows.append(_row("Replication", "Chicago, one split", "NWS guidance and live readings", _bootstrap_row(city_results["CHI"], "W_xgb", "B_market_norm")))
    compared = pd.read_csv(rolling_dir / "comparisons.csv")
    row = compared[(compared["city"] == "ALL") & (compared["baseline"] == "X_xgb") & (compared["challenger"] == "B_market_norm")].iloc[0]
    rows.append(_row("Rolling origin", "Both cities, 5 windows", "Plus NAM and three global models", row))
    exact = pd.read_csv(exact_dir / "comparisons.csv")
    row = exact[(exact["city"] == "ALL") & (exact["question"] == "Market vs weather with the runs")].iloc[0]
    rows.append(_row("Exact runs", "Both cities, from 2026-04", "Plus individual HRRR, NBM and ECMWF runs", row))
    return pd.DataFrame(rows)


def build_artifacts(
    data: list[str],
    test_frac: float = 0.3,
    n_boot: int = 2000,
    rate: float = 0.04,
    out_dir: Path | None = None,
    results_root: Path | None = None,
    enrich_steps=(enrich, enrich_extended),
) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else artifacts.ARTIFACTS_DIR
    results_root = Path(results_root) if results_root is not None else RESULTS_DIR
    rolling_dir, exact_dir = results_root / "rolling", results_root / "exact_runs"
    sources = {
        **{name: rolling_dir / src for name, src in ROLLING_TABLES.items()},
        **{name: exact_dir / src for name, src in EXACT_TABLES.items()},
    }
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run the rolling and exact-runs commands first. Missing: {', '.join(missing)}")

    df = pd.concat([pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in data], ignore_index=True)
    data_date_range = [str(df["target_date"].min()), str(df["target_date"].max())]
    for step in enrich_steps:
        df = step(df)

    frames, fitted, city_results, splits = [], {}, {}, {}
    for city, group in df.groupby("city"):
        print(f"Fitting {city} ...", flush=True)
        results = results_root / "app" / city
        extra = exp.ExtraModels(fit_predict_extra, EXTRA_SCORED, EXTRA_COMPARISONS, encompassing_model="X_xgb")
        exp.run(group.reset_index(drop=True), test_frac=test_frac, n_boot=n_boot, rate=rate, results_dir=results, extra=extra)
        quality = results_root / ("data_quality.json" if city == "NY" else f"data_quality_snapshots_{city.lower()}.json")
        if quality.exists():
            (results / "data_quality.json").write_text(quality.read_text())
        frames.append(pd.read_parquet(results / "test_predictions.parquet"))
        fitted[city] = extra.fitted
        city_results[city] = results
        splits[city] = json.loads((results / "split.json").read_text())

    predictions = pd.concat(frames, ignore_index=True)
    shown = [m for m in ALL_MODELS if m in predictions.columns]
    predictions = predictions[exp.APP_COLUMNS + shown]

    ladder_path = results_root / "app" / LADDER_NAME
    ladder(city_results, rolling_dir, exact_dir).to_csv(ladder_path, index=False)
    info = {
        "data_date_range": data_date_range,
        "cities": sorted(city_results),
        "splits": splits,
        "settings": {"test_frac": test_frac, "n_boot": n_boot, "rate": rate, "wang_lambda": 0.0},
    }
    return artifacts.save_cities(out_dir, fitted, shown, predictions, city_results, {**sources, LADDER_NAME: ladder_path}, info)
