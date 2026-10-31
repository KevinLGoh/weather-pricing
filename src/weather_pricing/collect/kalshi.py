"""Kalshi public market data (no authentication needed for these read endpoints).

Settled markets live in two places: those settled after the exchange's historical cutoff
are served by /markets, older ones only by /historical/markets. The same split applies
to candlesticks.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ..http import get_json

BASE = "https://api.elections.kalshi.com/trade-api/v2"
NS = "kalshi"


def _ts(s: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(s) if s else None


def _num(x) -> float | None:
    if x is None or x == "":
        return None
    return float(x)


def historical_cutoff() -> pd.Timestamp:
    body = get_json(f"{BASE}/historical/cutoff", namespace=NS, use_cache=False)
    return pd.Timestamp(body["market_settled_ts"])


def _paginate(path: str, params: dict) -> list[dict]:
    out, cursor = [], None
    while True:
        # Listings grow as new markets settle, so they are always refetched.
        body = get_json(f"{BASE}{path}", {**params, "cursor": cursor}, namespace=NS, use_cache=False)
        out.extend(body.get("markets", []))
        cursor = body.get("cursor")
        if not cursor:
            return out


def settled_markets(series: str) -> pd.DataFrame:
    """All settled markets in a series, live and historical, one row per contract."""
    raw = _paginate("/historical/markets", {"series_ticker": series, "limit": 1000})
    for m in raw:
        m["_source"] = "historical"
    live = _paginate("/markets", {"series_ticker": series, "status": "settled", "limit": 1000})
    for m in live:
        m["_source"] = "live"
    rows = []
    seen = set()
    for m in raw + live:
        if m["ticker"] in seen:
            continue
        seen.add(m["ticker"])
        rows.append(
            {
                "ticker": m["ticker"],
                "event_ticker": m["event_ticker"],
                "series": series,
                "title": m.get("title"),
                "strike_type": m.get("strike_type"),
                "floor_strike": _num(m.get("floor_strike")),
                "cap_strike": _num(m.get("cap_strike")),
                "result": m.get("result"),
                "expiration_value": _num(m.get("expiration_value")),
                "open_time": _ts(m.get("open_time")),
                "close_time": _ts(m.get("close_time")),
                "settlement_ts": _ts(m.get("settlement_ts")),
                "volume": _num(m.get("volume_fp") or m.get("volume")),
                "rules_primary": m.get("rules_primary"),
                "source": m["_source"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["result"].isin(["yes", "no"])].copy()
    df["target_date"] = df["event_ticker"].map(parse_event_date)
    return df.reset_index(drop=True)


def parse_event_date(event_ticker: str):
    """KXHIGHNY-26OCT30 -> date(2026, 10, 30)."""
    code = event_ticker.rsplit("-", 1)[-1]
    return datetime.strptime(code, "%y%b%d").date()


def candlesticks(market: pd.Series | dict, cutoff: pd.Timestamp, period: int = 60) -> pd.DataFrame:
    """Hourly candles over a market's trading life.

    Each candle summarises the period ending at `end_ts`, so a candle is usable at a
    snapshot time t only if end_ts <= t.
    """
    ticker = market["ticker"]
    start = int(market["open_time"].timestamp()) - 3600
    end = int(market["close_time"].timestamp())
    settled = market.get("settlement_ts")
    if settled is not None and not pd.isna(settled) and settled < cutoff:
        url = f"{BASE}/historical/markets/{ticker}/candlesticks"
    else:
        url = f"{BASE}/series/{market['series']}/markets/{ticker}/candlesticks"
    # Kalshi starts returning 429s somewhere above ~5 requests/second.
    body = get_json(url, {"start_ts": start, "end_ts": end, "period_interval": period}, namespace=NS, min_interval=0.2)
    rows = []
    for c in body.get("candlesticks", []):
        bid, ask, price = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
        rows.append(
            {
                "ticker": ticker,
                "end_ts": datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc),
                "bid": _price(bid, "close"),
                "ask": _price(ask, "close"),
                "last": _price(price, "close"),
                "volume": _num(c.get("volume_fp", c.get("volume"))) or 0.0,
                "open_interest": _num(c.get("open_interest_fp", c.get("open_interest"))),
            }
        )
    return pd.DataFrame(rows)


def _price(d: dict, field: str) -> float | None:
    """Live candles use `close_dollars`; historical ones put the same dollar string in
    `close`. Integer values are the legacy cents encoding."""
    if f"{field}_dollars" in d:
        return _num(d[f"{field}_dollars"])
    v = d.get(field)
    if v is None:
        return None
    return float(v) if isinstance(v, str) else float(v) / 100.0
