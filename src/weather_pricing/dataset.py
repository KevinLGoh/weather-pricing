"""Build the point-in-time research dataset.

One row = one Kalshi contract (a temperature bracket for a city-day) observed at one
snapshot time. Every feature on a row uses only information published before that
snapshot: market quotes from candles that had closed, NWP runs that had been released,
and observations of days that had ended.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import rules
from .collect import kalshi, noaa, openmeteo
from .config import CITIES, MAX_PREVIOUS_DAY, NWP_AVAILABILITY_HOURS, PROCESSED_DIR, SNAPSHOTS, STALE_HOURS, City


def snapshot_time(target: date, days_before: int, hour: int, tz: str) -> pd.Timestamp:
    local = datetime.combine(target - timedelta(days=days_before), datetime.min.time()).replace(
        hour=hour, tzinfo=ZoneInfo(tz)
    )
    return pd.Timestamp(local).tz_convert("UTC")


def local_day_bounds(target: date, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(datetime.combine(target, datetime.min.time()), tz=tz)
    return start.tz_convert("UTC"), (start + pd.Timedelta(days=1)).tz_convert("UTC")


def forecast_tmax(fc: pd.DataFrame, target: date, tz: str, snap: pd.Timestamp, extra_lag: int = 0) -> tuple[float, float]:
    """Max forecast temperature over the local target day using only released runs.

    `fc` is indexed by UTC valid time with one column per previous_dayN. For a valid hour
    t, the run behind previous_dayN was initialised ~t - 24N h and released
    NWP_AVAILABILITY_HOURS later, so it is usable at snapshot s when
    N >= (t - s + availability) / 24. `extra_lag` asks for a run that many days older,
    which gives the forecast as it stood a day earlier (used for revision features).
    Returns (tmax, mean N used); NaN if any hour of the day is unavailable.
    """
    start, end = local_day_bounds(target, tz)
    hours = fc.loc[(fc.index >= start) & (fc.index < end)]
    if len(hours) < 20:
        return math.nan, math.nan
    gap_h = (hours.index - snap).total_seconds().to_numpy() / 3600 + NWP_AVAILABILITY_HOURS
    need = np.maximum(1, np.ceil(gap_h / 24).astype(int)) + extra_lag
    if need.max() > MAX_PREVIOUS_DAY:
        return math.nan, math.nan
    vals = hours.to_numpy()[np.arange(len(hours)), need - 1]
    if np.isnan(vals).any():
        return math.nan, math.nan
    return float(vals.max()), float(need.mean())


def market_state(c: pd.DataFrame, snap: pd.Timestamp) -> dict:
    """Quote and activity features from candles that closed at or before the snapshot."""
    past = c[c["end_ts"] <= snap]
    if past.empty:
        return {}
    last = past.iloc[-1]
    bid, ask, lastpx = last["bid"], last["ask"], last["last"]
    has_quote = pd.notna(bid) and pd.notna(ask) and ask > bid
    traded = past[past["volume"] > 0]
    return {
        "bid": bid,
        "ask": ask,
        "last_price": lastpx,
        "mid": (bid + ask) / 2 if has_quote else lastpx,
        "spread": (ask - bid) if has_quote else math.nan,
        "cum_volume": float(past["volume"].sum()),
        "hours_since_trade": (snap - traded["end_ts"].iloc[-1]).total_seconds() / 3600 if len(traded) else math.inf,
        "quote_age_h": (snap - last["end_ts"]).total_seconds() / 3600,
    }


def build_city(city: City, start: date, end: date | None = None, progress: bool = True) -> pd.DataFrame:
    markets = kalshi.settled_markets(city.series)
    if markets.empty:
        return markets
    markets = markets[markets["target_date"] >= start]
    if end:
        markets = markets[markets["target_date"] <= end]
    markets = markets.sort_values(["target_date", "ticker"]).reset_index(drop=True)
    if markets.empty:
        return markets

    # Contract terms: text extraction cross-checked against the structured strike fields.
    specs = markets["rules_primary"].map(rules.extract)
    # Older listings carry no structured strikes; recover them from the rule text.
    missing = markets["strike_type"].isna()
    markets["strikes_from_text"] = missing
    markets.loc[missing, "strike_type"] = [s.comparator for s in specs[missing]]
    markets.loc[missing, "floor_strike"] = [np.nan if s.lo is None else s.lo for s in specs[missing]]
    markets.loc[missing, "cap_strike"] = [np.nan if s.hi is None else s.hi for s in specs[missing]]
    keep = markets["strike_type"].notna()
    markets, specs = markets[keep].reset_index(drop=True), specs[keep].reset_index(drop=True)
    markets["settle_source"] = [s.source for s in specs]
    markets["rule_issues"] = [
        "; ".join(rules.validate(s, r.strike_type, r.floor_strike, r.cap_strike, city.cli_code))
        for s, r in zip(specs, markets.itertuples())
    ]
    bounds = [rules.bracket(r.strike_type, r.floor_strike, r.cap_strike) for r in markets.itertuples()]
    markets["low"], markets["high"] = zip(*bounds)

    d0, d1 = markets["target_date"].min(), markets["target_date"].max()
    obs = noaa.daily_tmax(city.ghcnd, d0 - timedelta(days=7), d1).set_index("date")["tmax_obs"]
    fc_long = openmeteo.hourly_previous_runs(city.lat, city.lon, d0 - timedelta(days=2), d1 + timedelta(days=1))
    fc = fc_long.pivot_table(index="valid_time", columns="lead_day", values="temp_f")
    fc = fc.reindex(columns=range(1, MAX_PREVIOUS_DAY + 1))

    cutoff = kalshi.historical_cutoff()
    records = markets.to_dict("records")
    # Candle downloads dominate run time and are independent, so fetch them concurrently.
    with ThreadPoolExecutor(max_workers=3) as pool:
        all_candles = []
        for i, c in enumerate(pool.map(lambda m: kalshi.candlesticks(m, cutoff), records)):
            if progress and i % 500 == 0:
                print(f"  [{city.key}] candles {i}/{len(records)}", flush=True)
            all_candles.append(c)

    rows = []
    weather_cache: dict[tuple, tuple] = {}
    for m, candles in zip(records, all_candles):
        if candles.empty:
            continue
        target = m["target_date"]
        for days_before, hour in SNAPSHOTS:
            snap = snapshot_time(target, days_before, hour, city.tz)
            if snap < m["open_time"] or snap >= m["close_time"]:
                continue
            ms = market_state(candles, snap)
            if not ms:
                continue
            key = (target, snap)
            if key not in weather_cache:
                fc_now, lead_used = forecast_tmax(fc, target, city.tz, snap)
                fc_prev, _ = forecast_tmax(fc, target, city.tz, snap, extra_lag=1)
                prev_day = snap.tz_convert(city.tz).date() - timedelta(days=1)
                weather_cache[key] = (fc_now, lead_used, fc_prev, obs.get(prev_day, math.nan))
            fc_now, lead_used, fc_prev, persist = weather_cache[key]
            day_end = local_day_bounds(target, city.tz)[1]
            rows.append(
                {
                    "city": city.key,
                    "event_ticker": m["event_ticker"],
                    "ticker": m["ticker"],
                    "target_date": target,
                    "snapshot": f"D-{days_before} {hour:02d}:00",
                    "snapshot_ts": snap,
                    "lead_hours": (day_end - snap).total_seconds() / 3600,
                    "strike_type": m["strike_type"],
                    "low": m["low"],
                    "high": m["high"],
                    "fc_tmax": fc_now,
                    "fc_tmax_prev": fc_prev,
                    "fc_lead_days": lead_used,
                    "obs_tmax_prevday": persist,
                    **ms,
                    "settle_source": m["settle_source"],
                    "rule_issues": m["rule_issues"],
                    "strikes_from_text": m["strikes_from_text"],
                    "settled_value": m["expiration_value"],
                    "noaa_tmax": obs.get(target, math.nan),
                    "y": int(m["result"] == "yes"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # When an event's contracts partition the temperature line (a low tail, a high tail and
    # brackets between), their mids should sum to ~1 and can be renormalised. Early events
    # were threshold ladders instead, where that sum means nothing.
    kinds = df.groupby("event_ticker")["strike_type"].agg(set)
    partition = kinds.map(lambda k: {"less", "greater", "between"} <= k)
    df["partition_event"] = df["event_ticker"].map(partition)
    total = df.groupby(["event_ticker", "snapshot"])["mid"].transform("sum")
    df["event_mid_sum"] = total.where(df["partition_event"])
    df["mid_norm"] = (df["mid"] / total).where(df["partition_event"], df["mid"])
    df["stale"] = (df["hours_since_trade"] > STALE_HOURS) | df["spread"].isna()
    return df


def build(city_keys: list[str], start: date, end: date | None = None) -> pd.DataFrame:
    frames = []
    for key in city_keys:
        print(f"Building {key} ...", flush=True)
        frames.append(build_city(CITIES[key], start, end))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "snapshots.parquet", index=False)
    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Dataset audit: coverage, settlement-source mix and agreement with NOAA."""
    ev = df.drop_duplicates("ticker")
    noaa_ok = ev.dropna(subset=["noaa_tmax"])
    noaa_pays = [rules.pays_yes(v, lo, hi) for v, lo, hi in zip(noaa_ok["noaa_tmax"], noaa_ok["low"], noaa_ok["high"])]
    settled_ok = ev.dropna(subset=["settled_value"])
    settled_pays = [
        rules.pays_yes(v, lo, hi) for v, lo, hi in zip(settled_ok["settled_value"], settled_ok["low"], settled_ok["high"])
    ]
    events = df.drop_duplicates("event_ticker")
    temp_diff = (events["settled_value"] - events["noaa_tmax"]).dropna()
    return {
        "rows": len(df),
        "contracts": int(ev["ticker"].nunique()),
        "events": int(df["event_ticker"].nunique()),
        "date_range": [str(df["target_date"].min()), str(df["target_date"].max())],
        "by_city": df.groupby("city")["event_ticker"].nunique().to_dict(),
        "settlement_sources": ev.groupby("settle_source", dropna=False)["ticker"].count().to_dict(),
        "rule_text_mismatches": int((ev["rule_issues"] != "").sum()),
        "bracket_logic_matches_result": float(np.mean(np.array(settled_pays) == settled_ok["y"].to_numpy())),
        "noaa_outcome_matches_result": float(np.mean(np.array(noaa_pays) == noaa_ok["y"].to_numpy())),
        "settled_minus_noaa_tmax": {
            "share_equal": float((temp_diff == 0).mean()),
            "share_within_1F": float((temp_diff.abs() <= 1).mean()),
            "n": int(len(temp_diff)),
        },
        "missing_forecast_rows": int(df["fc_tmax"].isna().sum()),
        "stale_rows": int(df["stale"].sum()),
        "event_mid_sum_median": float(df["event_mid_sum"].median()),
    }
