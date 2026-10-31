import math

from weather_pricing import rules

NEW = "If the maximum temperature recorded at New York City (CLINYC) for Oct 30, 2026, is greater than 76° fahrenheit according to The Weather Company, then the market resolves to Yes."
OLD = "If the highest temperature recorded in Central Park, New York for February 09, 2025 as reported by the National Weather Service's Daily Climate Report, is between 35-36°, then the market resolves to Yes."
LADDER = "If the highest temperature recorded in Central Park, New York for March 01, 2022, is strictly greater than 47°, then the market resolves to Yes."


def test_extract_current_format():
    s = rules.extract(NEW)
    assert (s.station_code, s.source, s.comparator, s.lo, s.hi) == ("CLINYC", "The Weather Company", "greater", 76, None)


def test_extract_legacy_formats():
    s = rules.extract(OLD)
    assert (s.station_code, s.source, s.comparator, s.lo, s.hi) == (None, "NWS", "between", 35, 36)
    assert rules.extract(LADDER).comparator == "greater"
    assert rules.extract(LADDER).lo == 47
    assert rules.extract("is less than 69° fahrenheit").hi == 69


def test_validate_flags_disagreements():
    s = rules.extract(NEW)
    assert rules.validate(s, "greater", 76, None, "CLINYC") == []
    issues = rules.validate(s, "greater", 75, None, "CLIMDW")
    assert len(issues) == 2


def test_bracket_matches_kalshi_subtitles():
    # '>76' is subtitled '77° or above'; '<69' is '68° or below'.
    assert rules.bracket("greater", 76, None) == (77, math.inf)
    assert rules.bracket("less", None, 69) == (-math.inf, 68)
    assert rules.bracket("between", 70, 71) == (70, 71)
    assert rules.pays_yes(77, 77, math.inf) and not rules.pays_yes(76, 77, math.inf)
    assert rules.pays_yes(70.0, 70, 71) and not rules.pays_yes(72, 70, 71)
