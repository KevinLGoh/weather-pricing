from datetime import date

import numpy as np
import pandas as pd

from weather_pricing import rules
from weather_pricing.config import MAX_PREVIOUS_DAY, NWP_AVAILABILITY_HOURS
from weather_pricing.dataset import (
    add_market_columns,
    forecast_tmax,
    local_day_bounds,
    market_state,
    quality_report,
    snapshot_time,
)

TZ = "America/New_York"
INF = float("inf")
# Real-shaped strikes: '<69', '69-70', '71-72', '>72' -> (-inf, 68), (69, 70), (71, 72), (73, inf).
NY_STRIKES = [("less", None, 69), ("between", 69, 70), ("between", 71, 72), ("greater", 72, None)]
NY_SET = [(s, *rules.bracket(s, floor, cap)) for s, floor, cap in NY_STRIKES]


def _event(brackets, mids, snapshot="D 08:00", event="E1"):
    """One row per (strike_type, low, high) bracket with the given mids."""
    return pd.DataFrame(
        {
            "event_ticker": event,
            "snapshot": snapshot,
            "strike_type": [b[0] for b in brackets],
            "low": [float(b[1]) for b in brackets],
            "high": [float(b[2]) for b in brackets],
            "mid": mids,
            "hours_since_trade": 1.0,
            "spread": 0.02,
        }
    )


def _fc(start="2026-07-01", days=6):
    """previous_dayN value = 100 + N everywhere, so the max reveals the oldest run used."""
    idx = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    return pd.DataFrame({n: 100.0 + n for n in range(1, MAX_PREVIOUS_DAY + 1)}, index=idx)


def _report_frame():
    """2 events x 2 contracts x 2 snapshots. Kalshi settles both events as recorded;
    NOAA agrees on event A (72F) but puts event B (settled 82F) in the other bracket (79F)."""
    contracts = [
        # city, event, ticker, target_date, low, high, y, settled_value, noaa_tmax
        ("NY", "A", "A-LO", date(2026, 7, 1), -INF, 70.0, 0, 72.0, 72.0),
        ("NY", "A", "A-HI", date(2026, 7, 1), 71.0, INF, 1, 72.0, 72.0),
        ("CHI", "B", "B-LO", date(2026, 7, 2), -INF, 80.0, 0, 82.0, 79.0),
        ("CHI", "B", "B-HI", date(2026, 7, 2), 81.0, INF, 1, 82.0, 79.0),
    ]
    cols = [
        "city",
        "event_ticker",
        "ticker",
        "target_date",
        "low",
        "high",
        "y",
        "settled_value",
        "noaa_tmax",
    ]
    df = pd.DataFrame([c for c in contracts for _ in range(2)], columns=cols)
    df["settle_source"] = "kalshi"
    df["rule_issues"] = ""
    df["fc_tmax"] = [71.0, np.nan, 71.0, 71.0, 80.0, 80.0, 80.0, 80.0]
    df["stale"] = [False, False, False, True, False, False, False, False]
    df["partition_event"] = True
    # Event B's first snapshot misses a bracket quote: 2 incomplete rows.
    df["brackets_complete"] = [True, True, True, True, False, True, False, True]
    df["event_mid_sum"] = [1.05, 1.02, 1.05, 1.02, np.nan, 1.04, np.nan, 1.04]
    return df


def test_snapshot_time_is_local():
    assert snapshot_time(date(2026, 7, 4), 1, 14, TZ) == pd.Timestamp("2026-07-03 18:00", tz="UTC")


def test_local_day_has_25_hours_when_dst_ends():
    start, end = local_day_bounds(date(2026, 11, 1), TZ)
    assert start == pd.Timestamp("2026-11-01 04:00", tz="UTC")
    assert end == pd.Timestamp("2026-11-02 05:00", tz="UTC")


def test_local_day_has_23_hours_when_dst_starts():
    start, end = local_day_bounds(date(2026, 3, 8), TZ)
    assert start == pd.Timestamp("2026-03-08 05:00", tz="UTC")
    assert end == pd.Timestamp("2026-03-09 04:00", tz="UTC")


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


def test_forecast_nan_when_one_hour_of_the_day_is_missing():
    fc = _fc()
    target = date(2026, 7, 4)
    snap = snapshot_time(target, 1, 14, TZ)
    # The local day runs 04:00 UTC Jul 4 to 04:00 UTC Jul 5; 20:00 UTC is inside it.
    fc = fc.drop(pd.Timestamp("2026-07-04 20:00", tz="UTC"))
    assert np.isnan(forecast_tmax(fc, target, TZ, snap)[0])


def test_forecast_needs_all_25_hours_on_the_autumn_dst_day():
    # DST ends on 2026-11-01: the local day runs 04:00 UTC Nov 1 to 05:00 UTC Nov 2.
    fc = _fc("2026-10-28", days=8)
    target = date(2026, 11, 1)
    snap = snapshot_time(target, 1, 14, TZ)
    assert not np.isnan(forecast_tmax(fc, target, TZ, snap)[0])
    fc = fc.drop(pd.Timestamp("2026-11-02 04:00", tz="UTC"))
    assert np.isnan(forecast_tmax(fc, target, TZ, snap)[0])


def test_forecast_accepts_the_23_hour_spring_dst_day():
    # DST starts on 2026-03-08: the local day runs 05:00 UTC Mar 8 to 04:00 UTC Mar 9.
    fc = _fc("2026-03-04", days=8)
    target = date(2026, 3, 8)
    snap = snapshot_time(target, 1, 14, TZ)
    # 04:00 UTC Mar 9 is the next local midnight: outside the 23-hour day, inside start + 24h.
    fc.loc[pd.Timestamp("2026-03-09 04:00", tz="UTC")] = 200.0
    tmax, _ = forecast_tmax(fc, target, TZ, snap)
    assert not np.isnan(tmax)
    assert tmax < 200


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


def test_complete_bracket_set_is_renormalised_to_one():
    out = add_market_columns(_event(NY_SET, [0.1, 0.3, 0.4, 0.3]))
    assert out["partition_event"].all()
    assert out["brackets_complete"].all()
    assert np.allclose(out["event_mid_sum"], 1.1)
    assert np.isclose(out["mid_norm"].sum(), 1.0)


def test_missing_middle_bracket_keeps_raw_mid():
    brackets = [b for b in NY_SET if b[1] != 69]
    out = add_market_columns(_event(brackets, [0.1, 0.4, 0.3]))
    assert not out["brackets_complete"].any()
    assert out["event_mid_sum"].isna().all()
    assert np.allclose(out["mid_norm"], out["mid"])


def test_missing_tail_keeps_raw_mid():
    full = _event(NY_SET, [0.1, 0.3, 0.4, 0.3], snapshot="D-1 14:00")
    no_tail = _event(NY_SET[:3], [0.1, 0.3, 0.4], snapshot="D 08:00")
    out = add_market_columns(pd.concat([full, no_tail], ignore_index=True))
    rows = out[out["snapshot"] == "D 08:00"]
    assert rows["partition_event"].all()
    assert not rows["brackets_complete"].any()
    assert rows["event_mid_sum"].isna().all()
    assert np.allclose(rows["mid_norm"], rows["mid"])


def test_unquoted_bracket_keeps_raw_mid():
    out = add_market_columns(_event(NY_SET, [0.1, np.nan, 0.4, 0.3]))
    assert not out["brackets_complete"].any()
    assert out["event_mid_sum"].isna().all()
    quoted = out[out["mid"].notna()]
    assert np.allclose(quoted["mid_norm"], quoted["mid"])


def test_threshold_ladder_keeps_raw_mid():
    ladder = [("greater", 46, INF), ("greater", 48, INF)]
    out = add_market_columns(_event(ladder, [0.7, 0.4]))
    assert not out["partition_event"].any()
    assert out["event_mid_sum"].isna().all()
    assert np.allclose(out["mid_norm"], out["mid"])


def test_completeness_is_checked_per_snapshot():
    complete = _event(NY_SET, [0.1, 0.3, 0.4, 0.3], snapshot="D 08:00")
    gap = [b for b in NY_SET if b[1] != 71]
    incomplete = _event(gap, [0.2, 0.3, 0.3], snapshot="D-1 14:00")
    out = add_market_columns(pd.concat([complete, incomplete], ignore_index=True))
    day = out[out["snapshot"] == "D 08:00"]
    before = out[out["snapshot"] == "D-1 14:00"]
    assert day["brackets_complete"].all()
    assert np.isclose(day["mid_norm"].sum(), 1.0)
    assert not before["brackets_complete"].any()
    assert np.allclose(before["mid_norm"], before["mid"])


def test_duplicated_bracket_is_not_complete():
    brackets = NY_SET + [("between", 69, 70)]
    df = _event(brackets, [0.1, 0.3, 0.4, 0.3, 0.3])
    df["ticker"] = ["T-LO", "T-69", "T-71", "T-HI", "T-69-DUP"]
    out = add_market_columns(df)
    assert not out["brackets_complete"].any()
    assert out["event_mid_sum"].isna().all()
    assert np.allclose(out["mid_norm"], out["mid"])


def test_quality_report_counts_rows_contracts_and_events():
    report = quality_report(_report_frame())
    assert report["rows"] == 8
    assert report["contracts"] == 4
    assert report["events"] == 2
    assert report["date_range"] == ["2026-07-01", "2026-07-02"]
    assert report["by_city"] == {"CHI": 1, "NY": 1}
    assert report["missing_forecast_rows"] == 1
    assert report["stale_rows"] == 1


def test_quality_report_measures_agreement_with_noaa_and_settlement():
    report = quality_report(_report_frame())
    assert report["bracket_logic_matches_result"] == 1.0
    # Per contract: event A's 2 match NOAA, event B's 2 do not.
    assert report["noaa_outcome_matches_result"] == 0.5
    assert report["rule_text_mismatches"] == 0


def test_quality_report_counts_incomplete_bracket_rows():
    df = _report_frame()
    # A threshold ladder event is never a complete partition, but that is not a gap.
    ladder = df[df["event_ticker"] == "A"].copy()
    ladder["event_ticker"] = "L"
    ladder["ticker"] = ladder["ticker"].str.replace("A-", "L-")
    ladder["partition_event"] = False
    ladder["brackets_complete"] = False
    ladder["event_mid_sum"] = np.nan
    report = quality_report(pd.concat([df, ladder], ignore_index=True))
    assert report["incomplete_bracket_rows"] == 2
