"""NOAA NCEI daily summaries (GHCN-Daily) via the token-free Access Data Service."""

from __future__ import annotations

from datetime import date

import pandas as pd

from ..http import get_json

URL = "https://www.ncei.noaa.gov/access/services/data/v1"


def daily_tmax(station: str, start: date, end: date) -> pd.DataFrame:
    """Observed daily max/min temperature (deg F) per local climate day."""
    frames = []
    for year in range(start.year, end.year + 1):
        s, e = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        body = get_json(
            URL,
            {
                "dataset": "daily-summaries",
                "stations": station,
                "startDate": s.isoformat(),
                "endDate": e.isoformat(),
                "dataTypes": "TMAX,TMIN",
                "units": "standard",
                "format": "json",
            },
            namespace="noaa",
            min_interval=0.5,
            # The current year is still being filled in, so do not trust a cached copy.
            use_cache=year < date.today().year,
        )
        frames.append(pd.DataFrame(body))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["date", "tmax_obs", "tmin_obs"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["DATE"]).dt.date,
            "tmax_obs": pd.to_numeric(df.get("TMAX"), errors="coerce"),
            "tmin_obs": pd.to_numeric(df.get("TMIN"), errors="coerce"),
        }
    )
    return out.dropna(subset=["tmax_obs"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
