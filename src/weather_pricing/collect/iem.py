"""Iowa Environmental Mesonet archives: NWS model guidance and station observations.

Each guidance row carries the run time it was issued from, so the dataset can use only
runs that had been released before a snapshot. Times in these archives are UTC.
"""

from __future__ import annotations

from datetime import date, timedelta
from io import StringIO

import pandas as pd

from ..http import get_json

ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
MOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
NS = "iem"
# The archive answers 429 to a few quick requests, so space them well apart.
MIN_INTERVAL = 6.0
MOS_COLUMNS = ["runtime", "ftime", "n_x", "tmp", "txn", "xnd"]


def _month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar months covering start..end, each as (first day, first day of next month)."""
    chunks, first = [], start.replace(day=1)
    while first <= end:
        nxt = (first + timedelta(days=32)).replace(day=1)
        chunks.append((first, nxt))
        first = nxt
    return chunks


def _settled(chunk_end: date) -> bool:
    """Archives are still being filled for recent days, so do not trust a cached copy."""
    return chunk_end < date.today() - timedelta(days=8)


def _read_csv(text: str, required: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text), comment="#")
    if required not in frame.columns:
        raise ValueError(f"unexpected archive response: {text[:200]!r}")
    return frame


def parse_asos(text: str) -> pd.DataFrame:
    frame = _read_csv(text, "valid")
    out = pd.DataFrame(
        {
            "valid": pd.to_datetime(frame["valid"], utc=True),
            "tmpf": pd.to_numeric(frame["tmpf"], errors="coerce"),
        }
    )
    return out.dropna(subset=["tmpf"])


def parse_mos(text: str) -> pd.DataFrame:
    frame = _read_csv(text, "runtime").reindex(columns=MOS_COLUMNS)
    frame["runtime"] = pd.to_datetime(frame["runtime"], utc=True)
    frame["ftime"] = pd.to_datetime(frame["ftime"], utc=True)
    for col in ["n_x", "tmp", "txn", "xnd"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def station_temps(station: str, start: date, end: date) -> pd.DataFrame:
    """Routine and special observations: valid time (UTC) and temperature in deg F."""
    frames = []
    for first, nxt in _month_chunks(start, end):
        text = get_json(
            ASOS_URL,
            {
                "station": station,
                "data": "tmpf",
                "year1": first.year,
                "month1": first.month,
                "day1": first.day,
                "year2": nxt.year,
                "month2": nxt.month,
                "day2": nxt.day,
                "tz": "UTC",
                "format": "onlycomma",
                "latlon": "no",
                "missing": "empty",
                "trace": "empty",
                "report_type": "3,4",
            },
            namespace=NS,
            min_interval=MIN_INTERVAL,
            use_cache=_settled(nxt),
            as_text=True,
        )
        frames.append(parse_asos(text))
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("valid").sort_values("valid").reset_index(drop=True)


def guidance(icao: str, model: str, start: date, end: date) -> pd.DataFrame:
    """Guidance rows of one model: run time, forecast time and the temperature fields."""
    frames = []
    for first, nxt in _month_chunks(start, end):
        text = get_json(
            MOS_URL,
            {
                "station": icao,
                "model": model,
                "sts": f"{first.isoformat()}T00:00:00Z",
                "ets": f"{nxt.isoformat()}T00:00:00Z",
                "format": "csv",
            },
            namespace=NS,
            min_interval=MIN_INTERVAL,
            use_cache=_settled(nxt),
            as_text=True,
        )
        frames.append(parse_mos(text))
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["runtime", "ftime"]).sort_values(["runtime", "ftime"]).reset_index(drop=True)
