from datetime import date

from weather_pricing.collect import kalshi

LEGACY = {
    "ticker": "HIGHNY-23JAN05-B45.5",
    "event_ticker": "HIGHNY-23JAN05",
    "result": "yes",
    "strike_type": None,
    "open_time": "2023-01-04T15:00:00Z",
    "close_time": "2023-01-06T04:59:00Z",
    "settlement_ts": "2023-01-06T15:00:00Z",
    "rules_primary": "If the highest temperature recorded in Central Park, New York for January 05, 2023, is between 45-46°, then the market resolves to Yes.",
}
CURRENT = {
    "ticker": "KXHIGHNY-26OCT30-B70.5",
    "event_ticker": "KXHIGHNY-26OCT30",
    "result": "no",
    "strike_type": "between",
    "floor_strike": 70,
    "cap_strike": 71,
    "open_time": "2026-10-29T14:00:00Z",
    "close_time": "2026-10-31T03:59:00Z",
    "settlement_ts": "2026-10-31T15:00:00Z",
    "rules_primary": "If the maximum temperature recorded at New York City (CLINYC) for Oct 30, 2026, is between 70-71° fahrenheit according to The Weather Company, then the market resolves to Yes.",
}


def fake_get_json(url, params=None, **kwargs):
    # "/historical/markets" also ends in "/markets", so it must be matched first.
    if url.endswith("/historical/markets"):
        return {"markets": [dict(LEGACY)], "cursor": ""}
    if url.endswith("/markets"):
        return {"markets": [dict(CURRENT)], "cursor": ""}
    raise AssertionError(f"unexpected url {url}")


def test_settled_markets_keeps_legacy_highny_events(monkeypatch):
    monkeypatch.setattr(kalshi, "get_json", fake_get_json)
    df = kalshi.settled_markets("KXHIGHNY").set_index("ticker")
    assert list(df.index) == ["HIGHNY-23JAN05-B45.5", "KXHIGHNY-26OCT30-B70.5"]
    assert df.loc["HIGHNY-23JAN05-B45.5", "target_date"] == date(2023, 1, 5)
    assert df.loc["KXHIGHNY-26OCT30-B70.5", "target_date"] == date(2026, 10, 30)
    assert df.loc["HIGHNY-23JAN05-B45.5", "source"] == "historical"
    assert df.loc["KXHIGHNY-26OCT30-B70.5", "source"] == "live"
    assert list(df["series"]) == ["KXHIGHNY", "KXHIGHNY"]


def test_parse_event_date_reads_legacy_ticker():
    assert kalshi.parse_event_date("HIGHNY-23JAN05") == date(2023, 1, 5)
    assert kalshi.parse_event_date("KXHIGHNY-26OCT30") == date(2026, 10, 30)
