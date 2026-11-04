"""Static configuration: cities, paths and experiment defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class City:
    key: str
    name: str
    # Kalshi series ticker for daily max temperature. For NY, series KXHIGHNY also returns
    # the markets listed as HIGHNY-... before 24 Oct 2024; asking for series HIGHNY returns
    # nothing (checked 2026-11-04).
    series: str
    cli_code: str  # NWS climate-report code that Kalshi's rules cite, e.g. CLINYC
    ghcnd: str  # NOAA GHCN-Daily station used for verification
    lat: float
    lon: float
    tz: str


# Coordinates are the settlement stations, not city centres: Kalshi settles on a single
# station, and a forecast for a different grid point would be a different event.
CITIES: dict[str, City] = {
    c.key: c
    for c in [
        City("NY", "New York (Central Park)", "KXHIGHNY", "CLINYC", "USW00094728", 40.7789, -73.9692, "America/New_York"),
        City("CHI", "Chicago (Midway)", "KXHIGHCHI", "CLIMDW", "USW00014819", 41.7861, -87.7522, "America/Chicago"),
        City("MIA", "Miami (MIA)", "KXHIGHMIA", "CLIMIA", "USW00012839", 25.7881, -80.3169, "America/New_York"),
        City("AUS", "Austin (Bergstrom)", "KXHIGHAUS", "CLIAUS", "USW00013904", 30.1831, -97.6799, "America/Chicago"),
        City("DEN", "Denver (DIA)", "KXHIGHDEN", "CLIDEN", "USW00003017", 39.8466, -104.6562, "America/Denver"),
        City("PHL", "Philadelphia (PHL)", "KXHIGHPHIL", "CLIPHL", "USW00013739", 39.8733, -75.2268, "America/New_York"),
        City("LAX", "Los Angeles (LAX)", "KXHIGHLAX", "CLILAX", "USW00023174", 33.9381, -118.3889, "America/Los_Angeles"),
    ]
}

DEFAULT_CITIES = ["NY", "CHI"]

# Snapshot times, as (days before target date, local hour). Kalshi lists these markets at
# about 10:00 ET the day before, so D-1 14:00 is roughly the longest usable lead.
SNAPSHOTS: list[tuple[int, int]] = [(1, 14), (1, 22), (0, 8), (0, 12)]

# Assumed delay between an NWP run's initialisation time and its public availability.
NWP_AVAILABILITY_HOURS = 6
MAX_PREVIOUS_DAY = 5  # Open-Meteo previous-runs exposes previous_day1..7; we use 1..5.

# A market quote older than this (hours) at snapshot time is treated as stale.
STALE_HOURS = 6.0
