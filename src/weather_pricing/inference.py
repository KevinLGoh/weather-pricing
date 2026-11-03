"""Inference layer for the app: contracts, model probabilities, note values, evidence.

The UI calls only this module. Everything comes from app_artifacts/ (written by
`evaluate --save-artifacts`); nothing is retrained and no API is called. The saved
out-of-sample predictions are used as they are, so models.pkl is never unpickled here.

Every failure raises an InferenceError subclass whose message can be shown to the user
as it is: ArtifactsMissingError (artifacts absent, unreadable or outdated),
InvalidInputError (a value the app does not accept) and DataNotFoundError (a valid
choice with no data).
"""

from __future__ import annotations

import copy
import functools
import json
import math
import numbers
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from . import artifacts
from . import valuation as val
from .config import SNAPSHOTS
from .experiment import APP_COLUMNS, MAIN

BUILD_COMMAND = "python -m weather_pricing.cli evaluate --save-artifacts"

# Snapshot labels as the dataset writes them, e.g. "D-1 14:00".
SNAPSHOT_LABELS = [f"D-{days_before} {hour:02d}:00" for days_before, hour in SNAPSHOTS]

FAMILIES = {
    "rule": ["R_gaussian_rule"],
    "weather": ["A_logit", "A_xgb"],
    "market": ["B_market_mid", "B_market_norm", "B_logit"],
    "combined": ["C_logit", "C_xgb"],
}


class InferenceError(Exception):
    """Base class; the message is meant for the user."""


class ArtifactsMissingError(InferenceError):
    pass


class InvalidInputError(InferenceError):
    pass


class DataNotFoundError(InferenceError):
    pass


# What reading a damaged file can raise: OS errors, bad JSON/CSV/dates (ValueError), and
# pyarrow's ArrowTypeError / ArrowNotImplementedError.
READ_ERRORS = (OSError, ValueError, TypeError, NotImplementedError)


def _unreadable(name: str, folder: Path) -> ArtifactsMissingError:
    """Plain-words error for a file that cannot be used; the cause is chained, not shown."""
    return ArtifactsMissingError(
        f"{name} in {folder} is missing, damaged or from another version. "
        f"Rebuild the artifacts with: {BUILD_COMMAND}"
    )


@functools.lru_cache(maxsize=2)
def _load(folder: str) -> dict:
    """Read the artifacts once per folder. Failures are not cached, so a retry after
    building the artifacts works."""
    path = Path(folder)
    if not (path / "predictions.parquet").exists():
        raise ArtifactsMissingError(f"No app artifacts found in {path}. Build them with: {BUILD_COMMAND}")
    try:
        predictions = pd.read_parquet(path / "predictions.parquet")
    except READ_ERRORS as e:
        raise _unreadable("predictions.parquet", path) from e
    missing = [c for c in APP_COLUMNS + MAIN if c not in predictions.columns]
    if missing:
        raise ArtifactsMissingError(
            f"predictions.parquet lacks the columns {', '.join(missing)}; it was built by an older version. "
            f"Rebuild it with: {BUILD_COMMAND}"
        )
    try:
        predictions["target_date"] = pd.to_datetime(predictions["target_date"]).dt.date
    except READ_ERRORS as e:
        raise _unreadable("predictions.parquet", path) from e
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except READ_ERRORS as e:
        raise _unreadable("manifest.json", path) from e
    if not isinstance(manifest, dict):
        raise _unreadable("manifest.json", path)
    return {"predictions": predictions, "manifest": manifest, "folder": path}


def clear_cache() -> None:
    _load.cache_clear()
    _load_evidence.cache_clear()


_artifacts_dir: Path | None = None  # set by the app; None means artifacts.ARTIFACTS_DIR


def set_artifacts_dir(path: Path | str | None) -> None:
    """Use `path` as the artifacts folder (the app passes the folder next to itself, so
    it works however the package is installed). None returns to the default."""
    global _artifacts_dir
    _artifacts_dir = None if path is None else Path(path)


def _artifacts() -> dict:
    # Caches are keyed by the folder string, so switching folders needs no cache clear.
    return _load(str(_artifacts_dir or artifacts.ARTIFACTS_DIR))


def _parse_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            pass
    raise InvalidInputError(f"{value!r} is not a valid date. Use the form YYYY-MM-DD.")


def _num(x) -> float | None:
    return None if pd.isna(x) else float(x)


def bracket_label(low: float, high: float) -> str:
    """'68°F or below', '69–70°F' or '73°F or above'."""
    if math.isinf(low):
        return f"{high:.0f}°F or below"
    if math.isinf(high):
        return f"{low:.0f}°F or above"
    return f"{low:.0f}–{high:.0f}°F"


def list_cities() -> list[str]:
    return sorted(_artifacts()["predictions"]["city"].unique())


def _city_rows(city: str) -> pd.DataFrame:
    cities = list_cities()
    if city not in cities:
        raise InvalidInputError(f"Unknown city {city!r}. Available: {', '.join(cities)}.")
    preds = _artifacts()["predictions"]
    return preds[preds["city"] == city]


def list_dates(city: str) -> list[date]:
    return sorted(_city_rows(city)["target_date"].unique())


def _date_rows(city: str, target_date) -> tuple[pd.DataFrame, date]:
    rows, day = _city_rows(city), _parse_date(target_date)
    found = rows[rows["target_date"] == day]
    if found.empty:
        dates = sorted(rows["target_date"].unique())
        raise DataNotFoundError(
            f"No contracts for {city} on {day.isoformat()}. The app covers the test period, "
            f"{dates[0].isoformat()} to {dates[-1].isoformat()}."
        )
    return found, day


def list_snapshots(city: str, target_date) -> list[str]:
    present = set(_date_rows(city, target_date)[0]["snapshot"])
    return [s for s in SNAPSHOT_LABELS if s in present]


def _snapshot_rows(city: str, target_date, snapshot: str) -> pd.DataFrame:
    rows, day = _date_rows(city, target_date)
    if snapshot not in SNAPSHOT_LABELS:
        raise InvalidInputError(f"Unknown snapshot {snapshot!r}. Allowed: {', '.join(SNAPSHOT_LABELS)}.")
    found = rows[rows["snapshot"] == snapshot]
    if found.empty:
        available = [s for s in SNAPSHOT_LABELS if s in set(rows["snapshot"])]
        raise DataNotFoundError(
            f"No quotes for {city} on {day.isoformat()} at {snapshot}. Available: {', '.join(available)}."
        )
    return found


def list_contracts(city: str, target_date, snapshot: str) -> pd.DataFrame:
    """Contracts quoted at one snapshot, ordered from the low tail to the high tail."""
    rows = _snapshot_rows(city, target_date, snapshot).sort_values("low")
    out = rows[["ticker", "low", "high", "bid", "ask", "mid"]].reset_index(drop=True)
    out.insert(1, "label", [bracket_label(lo, hi) for lo, hi in zip(out["low"], out["high"])])
    return out


def predict(city: str, target_date, snapshot: str, ticker: str) -> dict:
    """Probability of every model for one contract at one snapshot, and its outcome."""
    rows = _snapshot_rows(city, target_date, snapshot)
    match = rows[rows["ticker"] == ticker]
    if match.empty:
        day = _parse_date(target_date).isoformat()
        raise DataNotFoundError(f"No contract {ticker!r} for {city} on {day} at {snapshot}.")
    r = match.iloc[0]
    return {
        "contract": {
            "ticker": r["ticker"],
            "label": bracket_label(r["low"], r["high"]),
            "low": float(r["low"]),
            "high": float(r["high"]),
            "bid": _num(r["bid"]),
            "ask": _num(r["ask"]),
            "mid": _num(r["mid"]),
            "spread": _num(r["spread"]),
            "stale": bool(r["stale"]),
            "settle_source": r["settle_source"],
            "lead_hours": float(r["lead_hours"]),
        },
        "probabilities": {name: float(r[name]) for name in MAIN},
        "families": {family: list(names) for family, names in FAMILIES.items()},
        "outcome": {"y": int(r["y"]), "settled_value": _num(r["settled_value"])},
    }


def _number(name: str, value, lo: float, hi: float, lo_open: bool = False) -> float:
    """`value` as a float in [lo, hi] (or (lo, hi] if lo_open), else InvalidInputError."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidInputError(f"{name} must be a number, not {value!r}.")
    x = float(value)
    if not math.isfinite(x) or x > hi or x < lo or (lo_open and x == lo):
        low = f"above {_readable(lo)}" if lo_open else f"at least {_readable(lo)}"
        raise InvalidInputError(f"{name} must be a finite number {low} and at most {_readable(hi)}; got {value!r}.")
    return x


def _readable(x: float) -> str:
    """1e9 -> '1,000,000,000'; 0.5 -> '0.5' (never scientific notation for limits)."""
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,}"


def _family_of(model: str) -> str | None:
    return next((family for family, names in FAMILIES.items() if model in names), None)


def price_note(
    probabilities: dict,
    lead_hours,
    principal,
    loss_fraction,
    coupon,
    lam,
    baseline: str = "A_xgb",
) -> pd.DataFrame:
    """Value of the hypothetical note under each model's probability.

    Uses valuation.value with a Wang loading of `lam`; the rate is the one the artifacts
    were evaluated with. `change_vs_baseline` is each model's value minus the baseline's.
    """
    if not isinstance(probabilities, dict) or not probabilities:
        raise InvalidInputError("probabilities must be a non-empty mapping of model name to probability.")
    probs = {}
    for model, p in probabilities.items():
        if _family_of(model) is None:
            raise InvalidInputError(f"probabilities: unknown model {model!r}. Known: {', '.join(MAIN)}.")
        probs[model] = _number(f"probabilities[{model}]", p, 0.0, 1.0)
    if baseline not in probs:
        raise InvalidInputError(f"baseline {baseline!r} is not among the models: {', '.join(probs)}.")
    note = val.Note(
        principal=_number("principal", principal, 0.0, 1e9, lo_open=True),
        loss_fraction=_number("loss_fraction", loss_fraction, 0.0, 1.0),
        coupon=_number("coupon", coupon, 0.0, 1e9),
    )
    lam = _number("lam", lam, -2.0, 2.0)
    years = val.horizon_years(_number("lead_hours", lead_hours, 0.0, 1e4))
    settings = _artifacts()["manifest"].get("settings")
    rate = settings.get("rate") if isinstance(settings, dict) else None
    if isinstance(rate, bool) or not isinstance(rate, numbers.Real) or not math.isfinite(rate):
        raise ArtifactsMissingError(
            f"manifest.json has no valid interest rate (settings.rate = {rate!r}). Rebuild the artifacts with: {BUILD_COMMAND}"
        )
    p = list(probs.values())
    out = pd.DataFrame(
        {
            "model": list(probs),
            "family": [_family_of(m) for m in probs],
            "probability": p,
            "pricing_probability": val.wang(p, lam),
            "value": val.value(p, years, note, rate, lam),
        }
    )
    out["change_vs_baseline"] = out["value"] - out.loc[out["model"] == baseline, "value"].iloc[0]
    return out


EVIDENCE_TABLES = {  # file -> index column (None: no index)
    "scores.csv": 0,
    "bootstrap.csv": None,
    "encompassing.csv": 0,
    "reliability.csv": None,
    "robustness.csv": None,
}
EVIDENCE_JSON = ["valuation.json", "split.json"]


@functools.lru_cache(maxsize=2)
def _load_evidence(folder: str) -> dict:
    path = Path(folder)
    out: dict = {}
    for name, index_col in EVIDENCE_TABLES.items():
        try:
            out[name.removesuffix(".csv")] = pd.read_csv(path / name, index_col=index_col)
        except READ_ERRORS as e:
            raise _unreadable(name, path) from e
    for name in EVIDENCE_JSON:
        try:
            out[name.removesuffix(".json")] = json.loads((path / name).read_text())
        except READ_ERRORS as e:
            raise _unreadable(name, path) from e
    dq = path / "data_quality.json"
    try:
        out["data_quality"] = json.loads(dq.read_text()) if dq.exists() else None
    except READ_ERRORS as e:
        raise _unreadable(dq.name, path) from e
    return out


def load_evidence() -> dict:
    """Scores, bootstrap CIs, encompassing test, calibration, robustness cuts,
    valuation summary, split, manifest and data quality (None if not built)."""
    loaded = _artifacts()
    evidence = copy.deepcopy(_load_evidence(str(loaded["folder"])))
    evidence["manifest"] = copy.deepcopy(loaded["manifest"])
    return evidence
