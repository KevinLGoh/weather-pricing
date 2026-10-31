from datetime import date

import numpy as np
import pandas as pd

from weather_pricing.config import MAX_PREVIOUS_DAY, NWP_AVAILABILITY_HOURS
from weather_pricing.dataset import forecast_tmax, market_state, snapshot_time

TZ = "America/New_York"


def _fc(start="2026-07-01", days=6):
    """previous_dayN value = 100 + N everywhere, so the max reveals the oldest run used."""
    idx = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    return pd.DataFrame({n: 100.0 + n for n in range(1, MAX_PREVIOUS_DAY + 1)}, index=idx)


def test_snapshot_time_is_local():
    assert snapshot_time(date(2026, 7, 4), 1, 14, TZ) == pd.Timestamp("2026-07-03 18:00", tz="UTC")


def test_forecast_uses_only_released_runs():
    fc = _fc()
    target = date(2026, 7, 4)
    snap = snapshot_time(target, 1, 14, TZ)
    tmax, lead = forecast_tmax(fc, target, TZ, snap)
    # The last hour of the local day is 03:00 UTC on Jul 5, 33h after the snapshot.
    # (33 + availability) / 24 rounded up is the oldest run any hour needs.
    expected_max_n = int(np.ceil((33 + NWP_AVAILABILITY_HOURS) / 24))
    assert tmax == 100 + expected_max_n
    assert 1 <= lead <= expected_max_n


def test_forecast_never_uses_a_run_newer_than_allowed():
    fc = _fc()
    target = date(2026, 7, 4)
    snap = snapshot_time(target, 0, 12, TZ)
    start = pd.Timestamp("2026-07-04", tz=TZ).tz_convert("UTC")
    # Poison previous_day1 for hours whose day-1 run was released after the snapshot.
    late = (fc.index - snap).total_seconds() / 3600 + NWP_AVAILABILITY_HOURS > 24
    fc.loc[late & (fc.index >= start), 1] = 999
    tmax, _ = forecast_tmax(fc, target, TZ, snap)
    assert tmax < 999


def test_forecast_nan_when_runs_too_old_are_needed():
    fc = _fc()
    snap = snapshot_time(date(2026, 7, 4), MAX_PREVIOUS_DAY + 1, 0, TZ)
    assert np.isnan(forecast_tmax(fc, date(2026, 7, 4), TZ, snap)[0])


def test_market_state_ignores_unfinished_candles():
    t = pd.Timestamp("2026-07-03 18:00", tz="UTC")
    candles = pd.DataFrame(
        {
            "end_ts": [t - pd.Timedelta(hours=1), t, t + pd.Timedelta(hours=1)],
            "bid": [0.30, 0.40, 0.90],
            "ask": [0.34, 0.44, 0.95],
            "last": [0.32, 0.42, 0.92],
            "volume": [10.0, 0.0, 50.0],
        }
    )
    s = market_state(candles, t)
    assert np.isclose(s["mid"], 0.42)
    assert np.isclose(s["spread"], 0.04)
    assert s["cum_volume"] == 10.0
    assert s["hours_since_trade"] == 1.0
