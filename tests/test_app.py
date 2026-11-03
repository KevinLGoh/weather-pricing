import importlib
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from weather_pricing import artifacts, experiment, inference  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = ["contract_explorer.py"]

DAY1, DAY2 = date(2026, 1, 2), date(2026, 1, 3)
SNAPS = {"D-0 08:00": 16.0, "D-1 14:00": 34.0}  # snapshot -> lead_hours (D-0 first on purpose)
# (strike_type, low, high, ticker suffix); written out of order so sorting is tested.
BRACKETS = [
    ("between", 71.0, 72.0, "B71.5"),
    ("less", -math.inf, 68.0, "T68"),
    ("greater", 73.0, math.inf, "T73"),
    ("between", 69.0, 70.0, "B69.5"),
]
WINNER = {DAY1: 69.0, DAY2: 71.0}  # low of the bracket that settled YES
SETTLED = {DAY1: 70.0, DAY2: 71.0}
ATTRIBUTION_WORDS = [
    "Weather data by Open-Meteo.com",
    "https://open-meteo.com/",
    "CC BY 4.0",
    "processed",
    "GHCN-Daily",
    "Menne",
    "no warranty",
    "Kalshi",
]


def _write_artifacts(folder):
    rows = []
    for day in (DAY1, DAY2):
        event = f"KXHIGHNY-{day.strftime('%y%b%d').upper()}"
        for snap, lead in SNAPS.items():
            for b, (strike, low, high, suffix) in enumerate(BRACKETS):
                bid = round(0.05 + 0.2 * b, 2)
                row = {
                    "city": "NY",
                    "event_id": f"{event}|{snap}",
                    "event_ticker": event,
                    "ticker": f"{event}-{suffix}",
                    "target_date": day,
                    "snapshot": snap,
                    "lead_hours": lead,
                    "strike_type": strike,
                    "low": low,
                    "high": high,
                    "bid": bid,
                    "ask": round(bid + 0.04, 2),
                    "mid": round(bid + 0.02, 2),
                    "mid_norm": round(bid + 0.02, 2),
                    "spread": 0.04,
                    "stale": False,
                    "settle_source": "kalshi",
                    "settled_value": SETTLED[day],
                    "y": int(low == WINNER[day]),
                }
                for m, model in enumerate(experiment.MAIN):
                    row[model] = round(0.1 + 0.05 * m + 0.1 * b, 4)
                rows.append(row)
    df = pd.DataFrame(rows)[experiment.APP_COLUMNS + experiment.MAIN]
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / "predictions.parquet", index=False)
    manifest = {
        "settings": {"rate": 0.04, "test_frac": 0.3},
        "test": {"dates": [DAY1.isoformat(), DAY2.isoformat()], "rows": len(df)},
        "cities": ["NY"],
        "models": sorted(experiment.MAIN),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def _shared(monkeypatch):
    monkeypatch.syspath_prepend(str(APP_DIR))
    return importlib.import_module("shared")


def _app(tmp_path, monkeypatch, folder=None):
    """AppTest for the entry script, reading artifacts from `folder` (synthetic by default)."""
    shared = _shared(monkeypatch)
    if folder is None:
        folder = _write_artifacts(tmp_path / "app_artifacts")
    monkeypatch.setattr(shared, "ARTIFACTS_DIR", folder)
    monkeypatch.setattr(inference, "_artifacts_dir", None)  # restored to the default after the test
    inference.clear_cache()
    return AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=30)


def _texts(at):
    return [e.value for kind in (at.markdown, at.warning, at.info, at.caption) for e in kind]


def _selectbox(at, label):
    return next(s for s in at.selectbox if s.label == label)


def test_app_starts_without_exceptions(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception


def test_every_page_shows_the_research_only_notice(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    shared = _shared(monkeypatch)
    for page in PAGES:
        at.switch_page(page).run()
        assert not at.exception, page
        assert any(shared.NOTICE in text for text in _texts(at)), page


def test_attribution_names_open_meteo_cc_by_and_noaa(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    text = "\n".join(_texts(at))
    for word in ATTRIBUTION_WORDS:
        assert word in text, word


def test_missing_artifacts_show_a_readable_error_with_the_build_command(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, folder=tmp_path / "missing")
    at.run()
    assert not at.exception
    assert any("evaluate --save-artifacts" in e.value for e in at.error)


def test_unexpected_error_shows_a_generic_message(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("boom")

    at = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(inference, "list_cities", boom)
    at.run()
    shared = _shared(monkeypatch)
    assert not at.exception
    assert any(e.value == shared.GENERIC_ERROR for e in at.error)
    assert not any("boom" in e.value for e in at.error)


def test_explorer_lists_dates_snapshots_and_contracts(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception
    assert _selectbox(at, "City").options == ["NY"]
    assert [str(d) for d in _selectbox(at, "Date").options] == ["2026-01-03", "2026-01-02"]
    assert _selectbox(at, "Snapshot").options == ["D-1 14:00", "D-0 08:00"]
    labels = ["68°F or below", "69–70°F", "71–72°F", "73°F or above"]
    assert _selectbox(at, "Contract").options == labels
    assert len(at.dataframe) == 1
    assert any(c.value.startswith("Loaded ") and "2" in c.value for c in at.caption)


def test_app_reads_artifacts_next_to_its_own_folder(tmp_path, monkeypatch):
    shared = _shared(monkeypatch)
    assert shared.ARTIFACTS_DIR == APP_DIR.parent / "app_artifacts"
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path / "missing")
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception
    assert not at.error
    assert _selectbox(at, "City").options == ["NY"]
