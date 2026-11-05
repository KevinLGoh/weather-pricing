"""Point-in-time weather features from NWS guidance and same-day station observations.

These fix the two handicaps of the Open-Meteo baseline: its newest usable forecast is
often most of a day old at the snapshot, and it never sees what the thermometer has
already read. Every value here was public at the snapshot under the release delays in
config.GUIDANCE_LAG_HOURS and config.OBS_PUBLISH_MINUTES.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .collect import iem
from .config import CITIES, GUIDANCE_LAG_HOURS, OBS_PUBLISH_MINUTES

MODELS = ["GFS", "NBS", "LAV"]

# Columns `enrich` adds to the snapshot table.
FEATURES = [
    "gfs_max",
    "gfs_age_h",
    "nbm_max",
    "nbm_spread",
    "nbm_age_h",
    "lamp_max",
    "lamp_age_h",
    "obs_max_sofar",
    "obs_latest",
    "obs_age_h",
    "obs_minus_nbm",
]


def climate_day_bounds(target: date, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC bounds of the NWS climate day: midnight to midnight local standard time.

    Standard time all year, so in summer the day runs 01:00 to 01:00 on the clock. That is
    also why Kalshi closes these markets at 05:00 UTC for New York in summer.
    """
    zone = ZoneInfo(tz)
    noon = datetime.combine(target, datetime.min.time()).replace(hour=12, tzinfo=zone)
    offset = noon.utcoffset() - noon.dst()
    start = pd.Timestamp(datetime.combine(target, datetime.min.time()), tz="UTC") - offset
    return start, start + pd.Timedelta(days=1)


@dataclass
class Runs:
    """One guidance model, with quick access to the rows of each run."""

    times: np.ndarray  # sorted run times, as datetime64
    rows: dict

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Runs:
        rows = {t: g for t, g in frame.groupby("runtime")}
        return cls(np.array(sorted(rows), dtype="datetime64[ns]"), rows)

    def latest_before(self, snap: pd.Timestamp, lag_hours: float) -> tuple[pd.Timestamp | None, pd.DataFrame | None]:
        """Newest run released by the snapshot: run time + lag <= snap."""
        cutoff = (snap - pd.Timedelta(hours=lag_hours)).tz_localize(None).to_datetime64()
        i = int(np.searchsorted(self.times, cutoff, side="right")) - 1
        if i < 0:
            return None, None
        run = pd.Timestamp(self.times[i], tz="UTC")
        return run, self.rows[run]

    def latest_value_before(
        self, snap: pd.Timestamp, lag_hours: float, column: str, ftime: pd.Timestamp, max_age_hours: float
    ) -> tuple[pd.Timestamp | None, pd.Series | None]:
        """Newest released run that forecasts `column` at `ftime`, within the age limit.

        The newest run often has no value: a run from the middle of the day no longer
        forecasts that day's high, so an earlier run is the latest forecast of it.
        """
        cutoff = (snap - pd.Timedelta(hours=lag_hours)).tz_localize(None).to_datetime64()
        for i in range(int(np.searchsorted(self.times, cutoff, side="right")) - 1, -1, -1):
            run = pd.Timestamp(self.times[i], tz="UTC")
            if _age(snap, run) > max_age_hours:
                break
            g = self.rows[run]
            row = g[g["ftime"] == ftime]
            if len(row) and pd.notna(row[column].iloc[0]):
                return run, row.iloc[0]
        return None, None


def _age(snap: pd.Timestamp, run: pd.Timestamp) -> float:
    return (snap - run).total_seconds() / 3600


def snapshot_features(
    snap: pd.Timestamp,
    target: date,
    tz: str,
    runs: dict[str, Runs],
    obs: pd.DataFrame,
    lags: dict[str, float] = GUIDANCE_LAG_HOURS,
    max_run_age_h: float = 48.0,
) -> dict:
    """Weather features for one city-day at one snapshot; NaN where nothing was public."""
    out = dict.fromkeys(FEATURES, math.nan)
    start, end = climate_day_bounds(target, tz)
    # MOS and NBM report the daytime high at 00 UTC of the next date.
    high_time = pd.Timestamp(datetime.combine(target + timedelta(days=1), datetime.min.time()), tz="UTC")

    run, row = runs["GFS"].latest_value_before(snap, lags["GFS"], "n_x", high_time, max_run_age_h)
    if run is not None:
        out["gfs_max"], out["gfs_age_h"] = float(row["n_x"]), _age(snap, run)

    run, row = runs["NBS"].latest_value_before(snap, lags["NBS"], "txn", high_time, max_run_age_h)
    if run is not None:
        out["nbm_max"], out["nbm_age_h"] = float(row["txn"]), _age(snap, run)
        out["nbm_spread"] = float(row["xnd"])

    nbm_now = math.nan
    run, g = runs["NBS"].latest_before(snap, lags["NBS"])
    if run is not None and _age(snap, run) <= max_run_age_h:
        temps = g.dropna(subset=["tmp"])
        if len(temps):
            nearest = temps.iloc[(temps["ftime"] - snap).abs().argsort().iloc[0]]
            if abs((nearest["ftime"] - snap).total_seconds()) <= 2 * 3600:
                nbm_now = float(nearest["tmp"])

    # LAMP is hourly but only reaches about 25 hours ahead, so it counts only when it
    # covers the rest of the climate day's warm hours.
    run, g = runs["LAV"].latest_before(snap, lags["LAV"])
    if run is not None and _age(snap, run) <= 6.0:
        hours = g[(g["ftime"] >= max(start, snap)) & (g["ftime"] < end)].dropna(subset=["tmp"])
        if len(hours) and g["ftime"].max() >= min(high_time, end - pd.Timedelta(hours=1)):
            out["lamp_max"], out["lamp_age_h"] = float(hours["tmp"].max()), _age(snap, run)

    public = snap - pd.Timedelta(minutes=OBS_PUBLISH_MINUTES)
    times = obs["valid"].to_numpy(dtype="datetime64[ns]")
    lo = np.searchsorted(times, start.tz_localize(None).to_datetime64(), side="left")
    hi = np.searchsorted(times, public.tz_localize(None).to_datetime64(), side="right")
    if hi > lo:
        temps = obs["tmpf"].to_numpy()[lo:hi]
        out["obs_max_sofar"], out["obs_latest"] = float(temps.max()), float(temps[-1])
        out["obs_age_h"] = (public - pd.Timestamp(times[hi - 1], tz="UTC")).total_seconds() / 3600
        out["obs_minus_nbm"] = out["obs_latest"] - nbm_now
    return out


def enrich(df: pd.DataFrame, lag_scale: float = 1.0) -> pd.DataFrame:
    """Add the weather features to a snapshot table, one lookup per city-day-snapshot."""
    lags = {k: v * lag_scale for k, v in GUIDANCE_LAG_HOURS.items()}
    frames = []
    for key, group in df.groupby("city"):
        city = CITIES[key]
        d0, d1 = group["target_date"].min(), group["target_date"].max()
        pad = timedelta(days=3)
        print(f"Weather archives for {key} ({d0} to {d1}) ...", flush=True)
        obs = iem.station_temps(city.icao[1:], d0 - pad, d1 + pad)
        runs = {m: Runs.from_frame(iem.guidance(city.icao, m, d0 - pad, d1 + pad)) for m in MODELS}
        keys = group[["city", "target_date", "snapshot_ts"]].drop_duplicates()
        rows = [
            {
                "city": key,
                "target_date": k.target_date,
                "snapshot_ts": k.snapshot_ts,
                **snapshot_features(k.snapshot_ts, k.target_date, city.tz, runs, obs, lags),
            }
            for k in keys.itertuples()
        ]
        frames.append(pd.DataFrame(rows))
    return df.merge(pd.concat(frames, ignore_index=True), on=["city", "target_date", "snapshot_ts"], how="left")
