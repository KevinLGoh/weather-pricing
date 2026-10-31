"""Archived NWP forecasts from Open-Meteo's previous-runs API.

`temperature_2m_previous_dayN` at valid time t is the value from the model run made about
N*24 hours before t. Keeping N explicit lets the dataset builder pick, hour by hour, only
forecasts that had been published before each snapshot.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..config import MAX_PREVIOUS_DAY
from ..http import get_json

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def hourly_previous_runs(
    lat: float, lon: float, start: date, end: date, model: str | None = None, chunk_days: int = 90
) -> pd.DataFrame:
    """Long table: valid_time (UTC), lead_day N, temp_f."""
    variables = [f"temperature_2m_previous_day{n}" for n in range(1, MAX_PREVIOUS_DAY + 1)]
    frames = []
    s = start
    while s <= end:
        e = min(end, s + timedelta(days=chunk_days - 1))
        body = get_json(
            URL,
            {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(variables),
                "start_date": s.isoformat(),
                "end_date": e.isoformat(),
                "temperature_unit": "fahrenheit",
                "timezone": "GMT",
                "models": model,
            },
            namespace="openmeteo",
            min_interval=0.5,
            use_cache=e < date.today() - timedelta(days=8),
        )
        h = body["hourly"]
        wide = pd.DataFrame(h)
        wide["valid_time"] = pd.to_datetime(wide.pop("time"), utc=True)
        frames.append(wide)
        s = e + timedelta(days=1)
    wide = pd.concat(frames, ignore_index=True)
    long = wide.melt(id_vars="valid_time", var_name="var", value_name="temp_f")
    long["lead_day"] = long["var"].str.extract(r"previous_day(\d+)$").astype(int)
    return long.drop(columns="var").dropna(subset=["temp_f"]).reset_index(drop=True)
