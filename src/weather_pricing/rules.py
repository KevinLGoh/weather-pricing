"""Structured extraction of contract terms from Kalshi's free-text rules.

This is the deterministic version of the plan's "LLM-assisted extraction" step: it turns
`rules_primary` into fields that can be checked against the exchange's structured strike
fields. An LLM extractor can be dropped in behind the same `RuleSpec` interface; the
validation below is what keeps either extractor honest.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

SOURCES = {
    "weather company": "The Weather Company",
    "national weather service": "NWS",
    "nws": "NWS",
}


@dataclass(frozen=True)
class RuleSpec:
    station_code: str | None
    source: str | None
    comparator: str | None  # greater | less | between
    lo: float | None
    hi: float | None


_STATION = re.compile(r"\((CLI[A-Z]{3,4}|K[A-Z]{3})\)")
_GREATER = re.compile(r"(?:greater than|above|more than)\s+(-?\d+(?:\.\d+)?)\s*°")
_LESS = re.compile(r"(?:less than|below)\s+(-?\d+(?:\.\d+)?)\s*°")
_BETWEEN = re.compile(r"between\s+(-?\d+(?:\.\d+)?)\s*°?\s*(?:-|–|and|to)\s*(-?\d+(?:\.\d+)?)\s*°")


def extract(text: str | None) -> RuleSpec:
    if not text:
        return RuleSpec(None, None, None, None, None)
    station = _STATION.search(text)
    lower = text.lower()
    source = next((v for k, v in SOURCES.items() if k in lower), None)
    if m := _BETWEEN.search(lower):
        comp, lo, hi = "between", float(m[1]), float(m[2])
    elif m := _GREATER.search(lower):
        comp, lo, hi = "greater", float(m[1]), None
    elif m := _LESS.search(lower):
        comp, lo, hi = "less", None, float(m[1])
    else:
        comp = lo = hi = None
    return RuleSpec(station[1] if station else None, source, comp, lo, hi)


def validate(spec: RuleSpec, strike_type: str, floor: float | None, cap: float | None, cli_code: str) -> list[str]:
    """Return a list of disagreements between the text and the structured fields."""
    issues = []
    # Pre-2025 rules name the place ("Central Park, New York") rather than a code.
    if spec.station_code is not None and spec.station_code != cli_code:
        issues.append(f"station {spec.station_code!r} != {cli_code!r}")
    if spec.comparator != strike_type:
        issues.append(f"comparator {spec.comparator!r} != {strike_type!r}")
    if strike_type == "greater" and spec.lo != floor:
        issues.append(f"threshold {spec.lo} != floor {floor}")
    if strike_type == "less" and spec.hi != cap:
        issues.append(f"threshold {spec.hi} != cap {cap}")
    if strike_type == "between" and (spec.lo, spec.hi) != (floor, cap):
        issues.append(f"range {(spec.lo, spec.hi)} != {(floor, cap)}")
    return issues


def bracket(strike_type: str, floor: float | None, cap: float | None) -> tuple[float, float]:
    """Inclusive integer-degree range [low, high] for which the contract pays Yes.

    Kalshi temperatures settle in whole degrees: '>76' is 77 or above, '<69' is 68 or
    below, and 'between 70-71' includes both ends. Infinite ends mark the tails.
    """
    if strike_type == "greater":
        return math.floor(floor) + 1, math.inf
    if strike_type == "less":
        return -math.inf, math.ceil(cap) - 1
    if strike_type == "between":
        return floor, cap
    raise ValueError(f"unknown strike_type {strike_type!r}")


def pays_yes(value: float, low: float, high: float) -> bool:
    return low <= math.floor(value + 0.5) <= high
