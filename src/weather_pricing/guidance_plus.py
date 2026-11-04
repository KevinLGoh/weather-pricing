"""Extra weather sources that the archives only offer in part.

Two sets of columns:

- NAM MOS and three more global models from Open-Meteo's previous runs. These feed the X
  family. The global models start in March 2024 and share the age limits of `fc_tmax`.
- Individual HRRR, NBM and ECMWF runs. Open-Meteo keeps these only from April 2026, so they
  feed a separate test over those months. Runs are every 3 or 6 hours, not hourly.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pandas as pd

from .collect import iem, openmeteo
from .config import (
    CITIES,
    EXACT_RUN_MODELS,
    EXACT_RUN_START,
    EXTRA_LAG_HOURS,
    MAX_PREVIOUS_DAY,
    PREVIOUS_RUN_MODELS,
    SNAPSHOTS,
    City,
)
from .dataset import forecast_tmax, snapshot_time
from .guidance import Runs, _age, climate_day_bounds

EXTENDED_COLUMNS = ["nam_max", *[f"om_{m}_max" for m in PREVIOUS_RUN_MODELS]]
EXACT_COLUMNS = [c for m in EXACT_RUN_MODELS for c in (f"{m}_max", f"{m}_age_h")]
MAX_CYCLES_BACK = 4


def _released_cycles(snap: pd.Timestamp, cycle_hours: int, lag_hours: float, limit: int = MAX_CYCLES_BACK) -> list[pd.Timestamp]:
    """Newest first: the model cycles whose release, run time + lag, is at or before snap."""
    newest = (snap - pd.Timedelta(hours=lag_hours)).floor(f"{cycle_hours}h")
    return [newest - pd.Timedelta(hours=cycle_hours * i) for i in range(limit)]


def exact_run_max(snap: pd.Timestamp, target: date, tz: str, cycle_hours: int, lag_hours: float, fetch) -> tuple[float, float]:
    """Highest forecast for the rest of the climate day from the newest released run reaching it.

    A short HRRR run may stop before the warm hours, so older cycles are tried until one
    covers them. `fetch(run)` returns that run's hourly frame. Returns (max, age in hours).
    """
    start, end = climate_day_bounds(target, tz)
    high_time = pd.Timestamp(datetime.combine(target + timedelta(days=1), datetime.min.time()), tz="UTC")
    needed_until = min(high_time, end - pd.Timedelta(hours=1))
    for run in _released_cycles(snap, cycle_hours, lag_hours):
        frame = fetch(run)
        if frame.empty or frame["valid_time"].max() < needed_until:
            continue
        hours = frame[(frame["valid_time"] >= max(start, snap)) & (frame["valid_time"] < end)]
        if len(hours):
            return float(hours["temp_f"].max()), _age(snap, run)
    return math.nan, math.nan


def exact_run_features(city: City, target: date, snap: pd.Timestamp, cache: dict | None = None) -> dict:
    cache = {} if cache is None else cache
    out = {}
    for name, (model, cycle_hours, lag_hours) in EXACT_RUN_MODELS.items():

        def fetch(run, model=model):
            key = (city.key, model, run)
            if key not in cache:
                cache[key] = openmeteo.single_run(city.lat, city.lon, model, run)
            return cache[key]

        out[f"{name}_max"], out[f"{name}_age_h"] = exact_run_max(snap, target, city.tz, cycle_hours, lag_hours, fetch)
    return out


def enrich_exact(df: pd.DataFrame) -> pd.DataFrame:
    """Add the individual-run columns. Rows before EXACT_RUN_START stay NaN."""
    frames = []
    cache: dict = {}
    for key, group in df.groupby("city"):
        city = CITIES[key]
        keys = group[group["target_date"] >= EXACT_RUN_START][["target_date", "snapshot_ts"]].drop_duplicates()
        print(f"Exact runs for {key}: {len(keys)} snapshots ...", flush=True)
        rows = [
            {"city": key, "target_date": k.target_date, "snapshot_ts": k.snapshot_ts, **exact_run_features(city, k.target_date, k.snapshot_ts, cache)}
            for k in keys.itertuples()
        ]
        frames.append(pd.DataFrame(rows, columns=["city", "target_date", "snapshot_ts", *EXACT_COLUMNS]))
    return df.merge(pd.concat(frames, ignore_index=True), on=["city", "target_date", "snapshot_ts"], how="left")


def enrich_extended(df: pd.DataFrame, lag_scale: float = 1.0) -> pd.DataFrame:
    """Add NAM MOS and the extra global models, one lookup per city-day-snapshot."""
    frames = []
    for key, group in df.groupby("city"):
        city = CITIES[key]
        d0, d1 = group["target_date"].min(), group["target_date"].max()
        pad = timedelta(days=3)
        print(f"Extended sources for {key} ({d0} to {d1}) ...", flush=True)
        nam = Runs.from_frame(iem.guidance(city.icao, "NAM", d0 - pad, d1 + pad))
        previous = {}
        for name, model in PREVIOUS_RUN_MODELS.items():
            long = openmeteo.hourly_previous_runs(city.lat, city.lon, d0 - timedelta(days=2), d1 + timedelta(days=1), model)
            wide = long.pivot_table(index="valid_time", columns="lead_day", values="temp_f")
            previous[name] = wide.reindex(columns=range(1, MAX_PREVIOUS_DAY + 1))
        keys = group[["target_date", "snapshot_ts"]].drop_duplicates()
        rows = []
        for k in keys.itertuples():
            row = {"city": key, "target_date": k.target_date, "snapshot_ts": k.snapshot_ts, "nam_max": math.nan}
            high_time = pd.Timestamp(datetime.combine(k.target_date + timedelta(days=1), datetime.min.time()), tz="UTC")
            run, value = nam.latest_value_before(k.snapshot_ts, EXTRA_LAG_HOURS["NAM"] * lag_scale, "n_x", high_time, 48.0)
            if run is not None:
                row["nam_max"] = float(value["n_x"])
            for name, wide in previous.items():
                row[f"om_{name}_max"] = forecast_tmax(wide, k.target_date, city.tz, k.snapshot_ts)[0]
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    return df.merge(pd.concat(frames, ignore_index=True), on=["city", "target_date", "snapshot_ts"], how="left")


def exact_snapshot_keys(city: City, start: date, end: date) -> list[tuple[date, pd.Timestamp]]:
    """Every city-day snapshot in a date range, without needing the market data."""
    days = pd.date_range(start, end, freq="D").date
    return [(d, snapshot_time(d, days_before, hour, city.tz)) for d in days for days_before, hour in SNAPSHOTS]
