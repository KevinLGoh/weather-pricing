import importlib
import json
import math
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402
from streamlit.testing.v1.element_tree import Block  # noqa: E402

from weather_pricing import artifacts, config, experiment, inference  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = ["contract_explorer.py", "note_pricer.py", "model_evidence.py"]

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
STALE = "T73"  # suffix of the bracket whose quotes are flagged stale
ONE_SIDED = (DAY1, "D-1 14:00", "B71.5")  # the only contract without a two-sided quote
ATTRIBUTION_WORDS = [
    "Weather data by Open-Meteo.com",
    "https://open-meteo.com/",
    "CC BY 4.0",
    "processed",
    "GHCN-Daily",
    "Menne",
    "10.7289/V5D21VHZ",
    "no warranty",
    "Kalshi",
]
# Parts of the full data sources text (shared.data_sources), with or without a manifest.
DATA_SOURCES_WORDS = [
    "Kalshi",
    '<a href="https://open-meteo.com/">Weather data by Open-Meteo.com</a>',
    '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>',
    "processed",
    "GHCN-Daily",
    "Version 3",
    "doi:10.7289/V5D21VHZ",
    "Menne",
    "(2012)",
    "897–910",  # the Menne et al. (2012) paper
    "no warranty",
]
NOAA_ACCESSED = "2026-10-30"  # noaa_accessed of the synthetic manifest


def _subset_sentence(city):
    c = config.CITIES[city]
    return f"Subset used: station {c.ghcnd}, {c.name}, elements TMAX and TMIN."


def _event(day):
    return f"KXHIGHNY-{day.strftime('%y%b%d').upper()}"


def _probability(m, b):
    """Probability of model number `m` (in experiment.MAIN) for bracket number `b` (in BRACKETS)."""
    return round(0.103 + 0.05 * m + 0.1 * b, 4)


# Results files of the synthetic artifacts, in the formats experiment.run() writes.
SCORES = {  # model -> (brier, log_loss, ece), for every model in experiment.MAIN
    "R_gaussian_rule": (0.10412, 0.33871, 0.04129),
    "A_logit": (0.09233, 0.30544, 0.02518),
    "A_xgb": (0.08871, 0.29312, 0.02246),
    "B_market_mid": (0.08127, 0.27455, 0.03384),
    "B_market_norm": (0.07642, 0.25918, 0.01873),
    "B_logit": (0.07598, 0.25703, 0.01412),
    "C_logit": (0.07415, 0.25129, 0.01306),
    "C_xgb": (0.07361, 0.24987, 0.01155),
}
# (baseline, challenger, question, diff, ci_lo, ci_hi, p_value, n_events); diff > 0: challenger better.
BOOTSTRAP = [
    ("A_logit", "C_logit", "Market data added to weather (logistic)", 0.01818, 0.01242, 0.02391, 0.0, 61),
    ("A_xgb", "C_xgb", "Market data added to weather (XGBoost)", 0.0151, 0.00987, 0.02044, 0.0, 61),
    ("B_market_norm", "C_logit", "Weather data added to market (logistic)", 0.00227, -0.00094, 0.00561, 0.197, 61),
    ("B_market_norm", "C_xgb", "Weather data added to market (XGBoost)", 0.00281, -0.00031, 0.00602, 0.0884, 61),
    ("A_xgb", "B_market_norm", "Market alone vs best weather-only", 0.01229, 0.00618, 0.01853, 0.0004, 61),
    ("B_market_mid", "B_market_norm", "Renormalising quotes within an event", 0.00485, 0.00213, 0.00772, 0.0013, 61),
]
# The weather p-value is above 0.05 on purpose (0.4152, not 0.415: a float 0.415 shows as "0.41").
ENCOMPASSING = {  # term -> (coef, se, z, p_value, ci_lo, ci_hi)
    "const": (-0.1184, 0.1452, -0.8154, 0.4148, -0.4030, 0.1662),
    "logit_weather": (0.0742, 0.0911, 0.8145, 0.4152, -0.1044, 0.2528),
    "logit_market": (0.8846, 0.0743, 11.9058, 0.0, 0.7390, 1.0302),
}
SPLIT = {
    "rows_in": 1312,
    "rows_used": 1240,
    "train": {"rows": 980, "events": 48, "dates": ["2025-10-24", "2025-12-10"]},
    "test": {"rows": 260, "events": 61, "dates": ["2025-12-11", DAY2.isoformat()]},
    "test_base_rate": 0.1846,
}
DATA_QUALITY = {
    "bracket_logic_matches_result": 0.9984,
    "noaa_outcome_matches_result": 0.9921,
    "missing_forecast_rows": 37,
    "incomplete_bracket_rows": 12,
}
# Robustness cuts: (cut, key column, key, events, Brier of A_xgb, B_market_norm, C_logit, C_xgb).
# The lead snapshots and liquidity buckets (labels as experiment.run writes them) are written out
# of order so the page's ordering is tested.
ROBUSTNESS = [
    ("lead", "snapshot", "D-0 12:00", 58, (0.0712, 0.0598, 0.0591, 0.0589)),
    ("lead", "snapshot", "D-1 14:00", 61, (0.0951, 0.0843, 0.0822, 0.0817)),
    ("lead", "snapshot", "D-0 08:00", 60, (0.0834, 0.0702, 0.0688, 0.0684)),
    ("lead", "snapshot", "D-1 22:00", 59, (0.0899, 0.0781, 0.0760, 0.0755)),
    ("liquidity", "liquidity", "2-5c", 57, (0.0866, 0.0741, 0.0729, 0.0722)),
    ("liquidity", "liquidity", "stale/no quote", 33, (0.1032, 0.0954, 0.0921, 0.0915)),
    ("liquidity", "liquidity", "spread<=2c", 52, (0.0811, 0.0698, 0.0684, 0.0679)),
    ("liquidity", "liquidity", ">5c", 41, (0.0943, 0.0857, 0.0831, 0.0826)),
    ("city", "city", "NY", 61, (0.0891, 0.0768, 0.0744, 0.0739)),
    ("settle_source", "settle_source", "NWS", 55, (0.0884, 0.0761, 0.0738, 0.0733)),
    ("settle_source", "settle_source", "The Weather Company", 6, (0.0957, 0.0836, 0.0802, 0.0794)),
]
LIQUIDITY_ORDER = ["spread<=2c", "2-5c", ">5c", "stale/no quote"]  # tightest spread first
LIQUIDITY_LABELS = {  # stored bucket -> page label
    "spread<=2c": "Spread ≤ 2¢",
    "2-5c": "Spread 2–5¢",
    ">5c": "Spread > 5¢",
    "stale/no quote": "Stale or no quote",
}
BRIER_COLUMNS = {  # robustness column -> page column
    "A_xgb": "Weather-only, XGBoost",
    "B_market_norm": "Market mid, renormalised",
    "C_logit": "Combined, logistic",
    "C_xgb": "Combined, XGBoost",
    "A_minus_C_xgb": "Gain from adding the market (XGBoost)",
}
RELIABILITY_MODELS = ["A_xgb", "B_market_norm", "C_xgb"]


def _robustness_rows():
    rows = []
    for cut, key, value, events, (a, b, c_logit, c_xgb) in ROBUSTNESS:
        briers = {
            "A_xgb": a,
            "B_market_norm": b,
            "C_logit": c_logit,
            "C_xgb": c_xgb,
            "A_minus_C_xgb": round(a - c_xgb, 4),
        }
        rows.append({key: value, "n": 4 * events, "events": events, **briers, "cut": cut})
    return rows


def _reliability_rows():
    """Ten bins per model, with slightly different values per model."""
    return [
        {
            "bin": b,
            "n": 20 + b + m,
            "mean_pred": round(0.05 + 0.1 * b + 0.002 * m, 4),
            "obs_freq": round(0.04 + 0.1 * b + 0.003 * m, 4),
            "bin_lo": 0.1 * b,
            "model": model,
        }
        for m, model in enumerate(RELIABILITY_MODELS)
        for b in range(10)
    ]


GIT_COMMIT = "3f9c2a1b7d4e5f60718293a4b5c6d7e8f9012345"
SRC_TREE = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"


def _write_results(folder, skip=()):
    """The results files inference.load_evidence reads; names in `skip` are not written."""
    scores = pd.DataFrame(
        [{"model": m, "brier": b, "log_loss": ll, "ece": e} for m, (b, ll, e) in SCORES.items()]
    ).set_index("model")
    bootstrap = pd.DataFrame(
        BOOTSTRAP,
        columns=["baseline", "challenger", "question", "diff", "ci_lo", "ci_hi", "p_value", "n_events"],
    )
    encompassing = pd.DataFrame.from_dict(
        ENCOMPASSING, orient="index", columns=["coef", "se", "z", "p_value", "ci_lo", "ci_hi"]
    )
    robustness = pd.DataFrame(_robustness_rows())
    reliability = pd.DataFrame(_reliability_rows())
    tables = {
        "scores.csv": (scores, True),
        "bootstrap.csv": (bootstrap, False),
        "encompassing.csv": (encompassing, True),
        "robustness.csv": (robustness, False),
        "reliability.csv": (reliability, False),
    }
    for name, (df, index) in tables.items():
        if name not in skip:
            df.to_csv(folder / name, index=index)
    valuation = {
        "note": {"principal": 1000.0, "loss_fraction": 0.5, "coupon": 0.0},
        "rate": 0.04,
        "wang_lambda": 0.0,
        "mean_abs_value_change_A_to_C": 21.84,
        "mean_value": {"A_xgb": 912.37, "B_market_norm": 908.15, "C_xgb": 909.62},
        "mean_realised": 907.69,
    }
    dicts = {"valuation.json": valuation, "split.json": SPLIT, "data_quality.json": DATA_QUALITY}
    for name, content in dicts.items():
        if name not in skip:
            (folder / name).write_text(json.dumps(content))


def _write_artifacts(folder, skip=(), manifest_changes=None):
    """Synthetic artifacts; `manifest_changes` sets manifest keys (a value of None removes the key)."""
    rows = []
    for day in (DAY1, DAY2):
        event = _event(day)
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
                    "stale": suffix == STALE,
                    "settle_source": "kalshi",
                    "settled_value": SETTLED[day],
                    "y": int(low == WINNER[day]),
                }
                if (day, snap, suffix) == ONE_SIDED:  # last trade only
                    row.update(bid=math.nan, ask=math.nan, spread=math.nan, mid=0.45)
                for m, model in enumerate(experiment.MAIN):
                    row[model] = _probability(m, b)
                rows.append(row)
    df = pd.DataFrame(rows)[experiment.APP_COLUMNS + experiment.MAIN]
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / "predictions.parquet", index=False)
    _write_results(folder, skip)
    manifest = {
        "data_date_range": [SPLIT["train"]["dates"][0], SPLIT["test"]["dates"][1]],
        "cities": ["NY"],
        "rows_in": SPLIT["rows_in"],
        "rows_used": SPLIT["rows_used"],
        "train": SPLIT["train"],
        "test": SPLIT["test"],
        "settings": {"test_frac": 0.3, "n_boot": 2000, "rate": 0.04, "wang_lambda": 0.0},
        "git_commit": GIT_COMMIT,
        "src_tree": SRC_TREE,
        "git_dirty": False,
        "python": "3.12.4",
        "packages": {"numpy": "2.1.3", "pandas": "2.2.3", "scikit-learn": "1.5.2", "xgboost": "2.1.3"},
        "predictions_rows": len(df),
        "models": sorted(experiment.MAIN),
        "noaa_accessed": NOAA_ACCESSED,
    }
    for key, value in (manifest_changes or {}).items():
        if value is None:
            manifest.pop(key, None)
        else:
            manifest[key] = value
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


def _assert_no_unescaped_dollar_pairs(at):
    """Streamlit Markdown reads a `$...$` pair as inline math, so at most one bare `$` per text."""
    for kind in (at.markdown, at.caption, at.info, at.success, at.warning, at.error):
        for e in kind:
            count = len(re.findall(r"(?<!\\)\$", e.value))
            assert count <= 1, f"{count} unescaped '$' in: {e.value!r}"


# A contract of the synthetic data that is not the explorer's default choice.
SELECTION = {"city": "NY", "date": DAY1.isoformat(), "snapshot": "D-0 08:00", "ticker": f"{_event(DAY1)}-B69.5"}
LAMBDA_CAPTION = (
    "λ is illustrative: a Wang-transform risk loading chosen for this example, not calibrated to any market."
)
COMBINED_SENTENCE = "Under the combined XGBoost model the note is worth"


def _note_pricer(tmp_path, monkeypatch, selection=True):
    """AppTest on the Note Pricer page, with SELECTION (or the given dict) in the session, or no selection."""
    at = _app(tmp_path, monkeypatch)
    at.run()  # the explorer stores its own default choice first
    if isinstance(selection, dict):
        at.session_state["selection"] = dict(selection)
    elif selection:
        at.session_state["selection"] = dict(SELECTION)
    else:
        del at.session_state["selection"]
    at.switch_page("note_pricer.py").run()
    return at


def _widget(widgets, text):
    return next(w for w in widgets if text in w.label)


def _change(c):
    return "$0.00" if abs(c) < 0.005 else f"{'+' if c > 0 else '-'}${abs(c):,.2f}"


def _expected_prices(principal=1000.0, loss_fraction=0.5, coupon=0.0, lam=0.0, baseline="A_xgb"):
    """inference.price_note for SELECTION (call after at.run(), which points inference at the artifacts)."""
    s = SELECTION
    result = inference.predict(s["city"], s["date"], s["snapshot"], s["ticker"])
    lead = result["contract"]["lead_hours"]
    return inference.price_note(result["probabilities"], lead, principal, loss_fraction, coupon, lam, baseline=baseline)


def _expected_table(shared, prices):
    return {
        "Model": [shared.MODEL_LABELS[m] for m in prices["model"]],
        "Value": [f"${v:,.2f}" for v in prices["value"]],
        "Change vs baseline": [_change(c) for c in prices["change_vs_baseline"]],
        "Probability": [f"{p:.1%}" for p in prices["probability"]],
        "Pricing probability": [f"{p:.1%}" for p in prices["pricing_probability"]],
        "Family": [shared.FAMILY_LABELS[f] for f in prices["family"]],
    }


def _table(at):
    assert len(at.dataframe) == 1
    return at.dataframe[0].value.to_dict("list")


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
    assert any("app-artifacts" in e.value for e in at.error)
    text = "\n".join(_texts(at))
    for word in ["Weather data by Open-Meteo.com", "CC BY 4.0", "GHCN-Daily"]:
        assert word in text, word


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
    assert _selectbox(at, "City").options == ["New York (Central Park)"]
    assert [str(d) for d in _selectbox(at, "Date").options] == ["2026-01-03", "2026-01-02"]
    assert _selectbox(at, "Snapshot").options == ["D-1 14:00", "D-0 08:00"]
    labels = ["68°F or below", "69–70°F", "71–72°F", "73°F or above"]
    assert _selectbox(at, "Contract").options == labels
    assert len(at.dataframe) == 1
    assert any(c.value.startswith("Loaded 2 test days (2026-01-02 to 2026-01-03)") for c in at.caption)


def test_app_reads_artifacts_next_to_its_own_folder(tmp_path, monkeypatch):
    shared = _shared(monkeypatch)
    assert shared.ARTIFACTS_DIR == APP_DIR.parent / "app_artifacts"
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path / "missing")
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception
    assert not at.error
    assert _selectbox(at, "City").options == ["New York (Central Park)"]


def test_explorer_shows_all_eight_model_probabilities_by_family(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    shared = _shared(monkeypatch)
    assert not at.exception
    # Default choice: newest day, first snapshot, lowest bracket ("68°F or below" = T68).
    b = [suffix for *_, suffix in BRACKETS].index("T68")
    subheaders = [s.value for s in at.subheader]
    for family in ["rule", "weather", "market", "combined"]:  # the families these artifacts have
        assert shared.FAMILY_LABELS[family] in subheaders, family
    assert shared.FAMILY_LABELS["stronger_weather"] not in subheaders
    metrics = {m.label: m.value for m in at.metric}
    for m, model in enumerate(experiment.MAIN):
        assert metrics[shared.MODEL_LABELS[model]] == f"{_probability(m, b) * 100:.1f}%", model
    assert len(at.get("vega_lite_chart")) == 1


def test_explorer_shows_the_outcome_of_the_contract(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    _selectbox(at, "Date").set_value(DAY1).run()
    _selectbox(at, "Contract").set_value("69–70°F").run()  # WINNER[DAY1]
    assert not at.exception
    assert any("paid Yes" in e.value and "70°F" in e.value for e in at.success)
    assert not any("paid No" in e.value for e in at.info)
    _selectbox(at, "Contract").set_value("73°F or above").run()
    assert not at.exception
    assert any("paid No" in e.value and "70°F" in e.value for e in at.info)
    assert not any("paid Yes" in e.value for e in at.success)


def test_explorer_shows_the_quote_as_a_dollar_price(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception
    # Default contract: lowest bracket (T68), same quote on every day and snapshot.
    b = [suffix for *_, suffix in BRACKETS].index("T68")
    bid = round(0.05 + 0.2 * b, 2)
    expected = (
        f"mid price \\${bid + 0.02:.2f} (bid \\${bid:.2f}, ask \\${bid + 0.04:.2f}, spread \\$0.04)"
    )
    assert any(expected in m.value for m in at.markdown), expected


def _expected_contracts_table(one_sided=None):
    """The explorer's contracts table for the synthetic quotes; `one_sided` has no bid or ask."""
    order = ["T68", "B69.5", "B71.5", "T73"]  # lowest bracket first, as the explorer lists them
    labels = ["68°F or below", "69–70°F", "71–72°F", "73°F or above"]
    suffixes = [suffix for *_, suffix in BRACKETS]
    table = {"Bracket": labels, "Bid": [], "Ask": [], "Mid": []}
    for suffix in order:
        bid = round(0.05 + 0.2 * suffixes.index(suffix), 2)
        if suffix == one_sided:
            table["Bid"].append("—")
            table["Ask"].append("—")
            table["Mid"].append("$0.45")
        else:
            table["Bid"].append(f"${bid:.2f}")
            table["Ask"].append(f"${round(bid + 0.04, 2):.2f}")
            table["Mid"].append(f"${round(bid + 0.02, 2):.2f}")
    return table


def test_explorer_table_shows_dollar_prices(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception
    table = _table(at)
    assert table == _expected_contracts_table()
    assert table["Bid"][0] == "$0.25"  # T68, spelled out
    day, snap, suffix = ONE_SIDED
    _selectbox(at, "Date").set_value(day).run()
    _selectbox(at, "Snapshot").set_value(snap).run()
    assert not at.exception
    table = _table(at)
    assert table == _expected_contracts_table(one_sided=suffix)
    assert (table["Bid"][2], table["Ask"][2], table["Mid"][2]) == ("—", "—", "$0.45")


def test_explorer_names_the_last_trade_without_a_two_sided_quote(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    day, snap, _ = ONE_SIDED
    _selectbox(at, "Date").set_value(day).run()
    _selectbox(at, "Snapshot").set_value(snap).run()
    _selectbox(at, "Contract").set_value("71–72°F").run()  # ONE_SIDED
    assert not at.exception
    assert any(
        "last trade price \\$0.45" in m.value and "no two-sided quote at this snapshot" in m.value
        for m in at.markdown
    )
    assert not any("mid price" in m.value for m in at.markdown)


def test_explorer_has_no_unescaped_dollar_pairs(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()  # default contract: two-sided quote
    assert not at.exception
    _assert_no_unescaped_dollar_pairs(at)
    day, snap, _ = ONE_SIDED
    _selectbox(at, "Date").set_value(day).run()
    _selectbox(at, "Snapshot").set_value(snap).run()
    _selectbox(at, "Contract").set_value("71–72°F").run()  # ONE_SIDED: last trade only
    assert not at.exception
    _assert_no_unescaped_dollar_pairs(at)


def test_explorer_warns_about_stale_quotes(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not any("stale" in w.value.lower() for w in at.warning)  # T68 is not stale
    _selectbox(at, "Contract").set_value("73°F or above").run()  # STALE
    assert not at.exception
    assert any(
        "stale" in w.value.lower()
        and "6 hours before the snapshot" in w.value
        and "no two-sided quote at the snapshot" in w.value
        for w in at.warning
    )


def test_explorer_states_the_forecast_age_limitation(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception
    words = [
        "started at least a day before the hour they forecast",
        "published about 6 h after they start",
        "7–23 h",
        "D-0 08:00",
        "D-0 12:00",
    ]
    assert any(all(word in c.value for word in words) for c in at.caption)


def test_explorer_bars_follow_the_family_order(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    shared = _shared(monkeypatch)
    assert not at.exception
    chart = at.get("vega_lite_chart")[0].proto
    # The chart's data rows come in MAIN order (Arrow IPC bytes in the proto) ...
    data = chart.datasets[0].data.data if chart.datasets else chart.data.data
    rows = pa.ipc.open_stream(data).read_pandas()
    labels = [shared.MODEL_LABELS[m] for m in experiment.MAIN]
    names = next(rows[c].tolist() for c in rows.columns if set(rows[c]) == set(labels))
    assert names == labels
    # ... and the bars keep that order: horizontal bars put the models on the y axis, and
    # sort None (st.bar_chart(sort=False)) turns off Vega-Lite's alphabetical sorting.
    spec = json.loads(chart.spec)
    assert "sort" in spec["encoding"]["y"] and spec["encoding"]["y"]["sort"] is None


def test_explorer_stores_the_selection_for_the_note_pricer(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception
    assert at.session_state["selection"] == {
        "city": "NY",
        "date": "2026-01-03",
        "snapshot": "D-1 14:00",
        "ticker": f"{_event(DAY2)}-T68",
    }
    _selectbox(at, "Date").set_value(DAY1).run()
    _selectbox(at, "Snapshot").set_value("D-0 08:00").run()
    _selectbox(at, "Contract").set_value("71–72°F").run()
    assert not at.exception
    assert at.session_state["selection"] == {
        "city": "NY",
        "date": "2026-01-02",
        "snapshot": "D-0 08:00",
        "ticker": f"{_event(DAY1)}-B71.5",
    }


def test_note_pricer_without_a_selection_points_to_the_explorer(tmp_path, monkeypatch):
    at = _note_pricer(tmp_path, monkeypatch, selection=False)
    assert not at.exception
    assert any("Pick a contract in the Contract Explorer first." in e.value for e in at.info)
    assert any("Contract Explorer" in link.value for link in at.page_link)
    assert not at.number_input
    assert not at.dataframe


def test_note_pricer_with_a_stored_choice_that_no_longer_exists_points_back_to_the_explorer(
    tmp_path, monkeypatch
):
    gone = dict(SELECTION, ticker="KXHIGHNY-26JAN02-B99.5")  # not in the synthetic data
    at = _note_pricer(tmp_path, monkeypatch, selection=gone)
    assert not at.exception
    assert any("no longer in the app's data" in e.value for e in at.info)
    assert any("Contract Explorer" in link.value for link in at.page_link)
    assert not at.number_input
    assert not at.dataframe
    assert not at.error


def test_comparison_sentence_says_the_same_for_a_zero_change(monkeypatch):
    shared = _shared(monkeypatch)
    for change in (0.0, 0.004, -0.004):
        sentence = shared.comparison_sentence(change, "Market mid, renormalised")
        assert "the same as under the baseline (Market mid, renormalised)" in sentence, change
        assert "0.00" not in sentence, change


def test_comparison_sentence_names_more_or_less(monkeypatch):
    shared = _shared(monkeypatch)
    label = "Weather-only, XGBoost"
    assert "\\$12.30 more than" in shared.comparison_sentence(12.3, label)
    assert "\\$5.00 less than" in shared.comparison_sentence(-5, label)
    assert "\\$1,234.50 more" in shared.comparison_sentence(1234.5, label)


def test_note_pricer_values_the_selected_contract_with_every_model(tmp_path, monkeypatch):
    at = _note_pricer(tmp_path, monkeypatch)
    shared = _shared(monkeypatch)
    assert not at.exception
    principal = _widget(at.number_input, "Principal")
    assert (principal.value, principal.min, principal.max) == (1000.0, 1, 1e9)
    loss = _widget(at.slider, "Loss if the bracket happens (% of principal)")
    assert (loss.value, loss.min, loss.max) == (50, 0, 100)
    coupon = _widget(at.number_input, "Coupon")
    assert (coupon.value, coupon.min, coupon.max) == (0.0, 0, 1e9)
    lam = _widget(at.slider, "Risk loading λ")
    assert (lam.value, lam.min, lam.max, lam.step) == (0.0, -1.0, 1.0, 0.05)
    baseline = _selectbox(at, "Baseline model")
    assert baseline.options == [shared.MODEL_LABELS[m] for m in experiment.MAIN]
    assert baseline.value == "Weather-only, XGBoost"
    # Summary line: city, date, snapshot and bracket of the selected contract.
    assert any(all(w in t for w in ["New York", "2026-01-02", "D-0 08:00", "69–70°F"]) for t in _texts(at))
    prices = _expected_prices()
    expected = _expected_table(shared, prices)
    table = _table(at)
    assert list(table) == list(expected)
    assert table == expected
    assert table["Model"] == [shared.MODEL_LABELS[m] for m in experiment.MAIN]
    assert len(at.get("vega_lite_chart")) == 1
    change = prices.set_index("model")["change_vs_baseline"]["C_xgb"]
    sentences = [m.value for m in at.markdown if m.value.startswith(COMBINED_SENTENCE)]
    assert len(sentences) == 1
    assert f"\\${abs(change):,.2f} {'more' if change > 0 else 'less'}" in sentences[0]


def test_note_pricer_follows_changed_inputs(tmp_path, monkeypatch):
    at = _note_pricer(tmp_path, monkeypatch)
    shared = _shared(monkeypatch)
    _widget(at.number_input, "Principal").set_value(2000.0)
    _widget(at.slider, "Loss if the bracket happens").set_value(30)
    _widget(at.number_input, "Coupon").set_value(10.0)
    _widget(at.slider, "Risk loading λ").set_value(0.25)
    at.run()
    assert not at.exception
    # The slider shows percent; the page passes percent / 100 as the loss fraction.
    expected = _expected_table(shared, _expected_prices(2000, 0.3, 10, 0.25))
    assert _table(at) == expected


def test_note_pricer_describes_the_note_like_the_valuation(tmp_path, monkeypatch):
    at = _note_pricer(tmp_path, monkeypatch)
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    for words in [
        "pays the principal plus the coupon",
        "loses the loss fraction of the principal",
        "the coupon is paid either way",
        "present value",
    ]:
        assert words in text, words


def test_note_pricer_labels_the_note_hypothetical_and_lambda_illustrative(tmp_path, monkeypatch):
    at = _note_pricer(tmp_path, monkeypatch)
    shared = _shared(monkeypatch)
    assert not at.exception
    assert any(shared.NOTICE in w.value and "hypothetical" in w.value for w in at.warning)
    assert any(c.value == LAMBDA_CAPTION for c in at.caption)
    # The caption comes right after the λ slider.
    elements = [n for n in at.main if not isinstance(n, Block)]
    i = next(i for i, n in enumerate(elements) if n.type == "slider" and "Risk loading λ" in n.label)
    assert elements[i + 1].type == "caption" and elements[i + 1].value == LAMBDA_CAPTION


def test_note_pricer_shows_the_change_against_the_chosen_baseline(tmp_path, monkeypatch):
    at = _note_pricer(tmp_path, monkeypatch)
    shared = _shared(monkeypatch)
    _selectbox(at, "Baseline model").set_value("Market mid, renormalised").run()
    assert not at.exception
    table = _table(at)
    changes = dict(zip(table["Model"], table["Change vs baseline"]))
    assert changes["Market mid, renormalised"] == "$0.00"
    values = _expected_prices(baseline="B_market_norm").set_index("model")["value"]
    assert changes["Combined, XGBoost"] == _change(values["C_xgb"] - values["B_market_norm"])
    assert table == _expected_table(shared, _expected_prices(baseline="B_market_norm"))
    assert any(m.value.startswith(COMBINED_SENTENCE) for m in at.markdown)
    # With the combined XGBoost model as the baseline there is nothing to compare.
    _selectbox(at, "Baseline model").set_value("Combined, XGBoost").run()
    assert not at.exception
    assert not any(m.value.startswith(COMBINED_SENTENCE) for m in at.markdown)


def test_note_pricer_has_no_unescaped_dollar_pairs(tmp_path, monkeypatch):
    at = _note_pricer(tmp_path, monkeypatch)
    _widget(at.number_input, "Principal").set_value(2500.0)
    _widget(at.number_input, "Coupon").set_value(25.0)
    at.run()
    assert not at.exception
    _assert_no_unescaped_dollar_pairs(at)
    # Widget labels are Markdown too.
    for widget in [*at.number_input, *at.slider, *at.selectbox]:
        count = len(re.findall(r"(?<!\\)\$", widget.label))
        assert count <= 1, f"{count} unescaped '$' in: {widget.label!r}"


def test_explorer_button_opens_the_note_pricer(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception
    selection = dict(at.session_state["selection"])
    next(b for b in at.button if b.label == "Price this contract").click().run()
    assert not at.exception
    assert any(t.value == "Note Pricer" for t in at.title)
    assert at.session_state["selection"] == selection


def _evidence(tmp_path, monkeypatch, skip=()):
    """AppTest on the Model Evidence page; results files named in `skip` are not written."""
    folder = _write_artifacts(tmp_path / "app_artifacts", skip=skip)
    at = _app(tmp_path, monkeypatch, folder=folder)
    at.run()
    at.switch_page("model_evidence.py").run()
    return at


def _evidence_table(at, *first_columns):
    """The page's dataframe whose columns start with `first_columns`, as a dict of lists."""
    n = len(first_columns)
    tables = [d.value for d in at.dataframe if list(d.value.columns)[:n] == list(first_columns)]
    assert len(tables) == 1, first_columns
    return tables[0].to_dict("list")


def _brier_columns(rows):
    """The formatted Brier columns of the page's robustness tables for `rows` of ROBUSTNESS."""
    out = {}
    for column, label in BRIER_COLUMNS.items():
        out[label] = [f"{r[column]:.4f}" for r in rows]
    return out


def _cut_rows(cut):
    return [r for r in _robustness_rows() if r["cut"] == cut]


def _all_texts(at):
    kinds = (at.title, at.header, at.subheader, at.markdown, at.caption, at.info, at.success, at.warning, at.error)
    return [e.value for kind in kinds for e in kind]


def _p_text(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def test_evidence_shows_scores_for_every_model(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    shared = _shared(monkeypatch)
    assert not at.exception
    assert any(t.value == "Model Evidence" for t in at.title)
    assert _evidence_table(at, "Model", "Brier") == {
        "Model": [shared.MODEL_LABELS[m] for m in experiment.MAIN],
        "Brier": [f"{SCORES[m][0]:.4f}" for m in experiment.MAIN],
        "Log loss": [f"{SCORES[m][1]:.3f}" for m in experiment.MAIN],
        "ECE": [f"{SCORES[m][2]:.3f}" for m in experiment.MAIN],
    }
    test = SPLIT["test"]
    caption = (
        f"Test period 2025-12-11 to 2026-01-03: {test['events']} city-days, {test['rows']:,} contract snapshots."
    )
    assert any(c.value == caption for c in at.caption), caption


def test_evidence_shows_bootstrap_confidence_intervals(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    shared = _shared(monkeypatch)
    assert not at.exception
    table = _evidence_table(at, "Question")
    assert list(table) == ["Question", "Brier difference", "95% CI", "p-value", "City-days", "Baseline", "Challenger"]
    assert table["Question"] == [row[2] for row in BOOTSTRAP]
    assert table["Baseline"] == [shared.MODEL_LABELS[row[0]] for row in BOOTSTRAP]
    assert table["Challenger"] == [shared.MODEL_LABELS[row[1]] for row in BOOTSTRAP]
    assert table["Brier difference"] == [f"{row[3]:.4f}" for row in BOOTSTRAP]
    assert table["95% CI"] == [f"{row[4]:.4f} to {row[5]:.4f}" for row in BOOTSTRAP]
    assert table["p-value"] == [_p_text(row[6]) for row in BOOTSTRAP]
    assert table["City-days"] == [row[7] for row in BOOTSTRAP]
    # Spelled out for the rows with p = 0.0 and p = 0.197.
    assert table["Brier difference"][0] == "0.0182"
    assert table["95% CI"][0] == "0.0124 to 0.0239"
    assert table["p-value"][0] == "< 0.001"
    assert table["95% CI"][2] == "-0.0009 to 0.0056"
    assert table["p-value"][2] == "0.197"
    assert table["Baseline"][2] == "Market mid, renormalised"
    assert table["Challenger"][2] == "Combined, logistic"


def test_evidence_never_shows_a_negative_zero(tmp_path, monkeypatch):
    """A difference too small for four decimals keeps its digits instead of showing "-0.0000"."""
    folder = _write_artifacts(tmp_path / "app_artifacts")
    boot = pd.read_csv(folder / "bootstrap.csv")
    boot.loc[len(boot) - 1, ["diff", "ci_lo", "ci_hi"]] = [-0.00002, -0.00016, 0.00004]
    boot.to_csv(folder / "bootstrap.csv", index=False)
    at = _app(tmp_path, monkeypatch, folder=folder)
    at.run()
    at.switch_page("model_evidence.py").run()
    assert not at.exception
    table = _evidence_table(at, "Question")
    assert table["Brier difference"][-1] == "-0.00002"
    assert table["95% CI"][-1] == "-0.0002 to 0.00004"
    terms = _evidence_table(at, "Term")
    assert terms == {
        "Term": ["Intercept", "Weather probability (logit)", "Market probability (logit)"],
        "Coefficient": [f"{row[0]:.2f}" for row in ENCOMPASSING.values()],
        "SE": [f"{row[1]:.2f}" for row in ENCOMPASSING.values()],
        "p-value": [_p_text(row[3]) for row in ENCOMPASSING.values()],
    }
    # Spelled out for the weather term.
    assert (terms["Coefficient"][1], terms["SE"][1], terms["p-value"][1]) == ("0.07", "0.09", "0.415")


def test_evidence_explains_how_to_read_the_numbers(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    for words in ["out-of-sample", "Lower is better", "positive difference means the challenger"]:
        assert words in text, words


def test_evidence_shows_where_the_results_come_from(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    expanders = [e for e in at.expander if e.label == "Where these results come from"]
    assert len(expanders) == 1
    text = "\n".join(m.value for m in expanders[0].markdown)
    for words in [
        "2025-10-24",  # data range
        "2026-01-03",
        "2025-12-10",  # last train day
        "2025-12-11",  # first test day
        "1,240 of 1,312",
        GIT_COMMIT[:12],
        SRC_TREE[:12],
    ]:
        assert words in text, words


def test_evidence_limits_conclusions_to_this_weather_baseline(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    assert any(
        "this weather baseline" in e.value
        and "not evidence that prediction markets are better than meteorology in general" in e.value
        for e in at.info
    )
    for text in _all_texts(at):
        assert "better than meteorology" not in text or "not evidence" in text, text


def test_evidence_missing_results_file_shows_a_readable_error(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch, skip=("bootstrap.csv",))
    assert not at.exception
    assert any("bootstrap.csv" in e.value and "app-artifacts" in e.value for e in at.error)
    text = "\n".join(_texts(at))
    for word in ATTRIBUTION_WORDS:
        assert word in text, word


def test_evidence_has_no_unescaped_dollar_pairs(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    _assert_no_unescaped_dollar_pairs(at)


def test_evidence_shows_calibration_for_three_models(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    shared = _shared(monkeypatch)
    assert not at.exception
    assert "Calibration" in [s.value for s in at.subheader]
    charts = at.get("vega_lite_chart")
    assert len(charts) >= 1
    # The chart's series: the three models plus the diagonal.
    chart = charts[0].proto
    data = chart.datasets[0].data.data if chart.datasets else chart.data.data
    points = pa.ipc.open_stream(data).read_pandas()
    series = {shared.MODEL_LABELS[m] for m in RELIABILITY_MODELS} | {"Perfect calibration"}
    assert any(set(points[c]) == series for c in points.columns)
    rows = _reliability_rows()
    assert _evidence_table(at, "Model", "Bin") == {
        "Model": [shared.MODEL_LABELS[r["model"]] for r in rows],
        "Bin": [f"{r['bin_lo']:.1f}–{r['bin_lo'] + 0.1:.1f}" for r in rows],
        "Contracts": [r["n"] for r in rows],
        "Mean predicted": [f"{r['mean_pred']:.3f}" for r in rows],
        "Observed": [f"{r['obs_freq']:.3f}" for r in rows],
    }


def test_evidence_shows_results_by_snapshot(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    assert "By snapshot" in [s.value for s in at.subheader]
    lead = {r["snapshot"]: r for r in _cut_rows("lead")}
    rows = [lead[s] for s in inference.SNAPSHOT_LABELS]
    table = _evidence_table(at, "Snapshot")
    expected = {
        "Snapshot": ["D-1 14:00", "D-1 22:00", "D-0 08:00", "D-0 12:00"],
        "City-days": [r["events"] for r in rows],
        **_brier_columns(rows),
    }
    assert list(table) == list(expected)
    assert table == expected
    assert table["Snapshot"] == inference.SNAPSHOT_LABELS
    # Spelled out for the first row (D-1 14:00).
    assert table["Weather-only, XGBoost"][0] == "0.0951"
    assert table["Gain from adding the market (XGBoost)"][0] == "0.0134"


def test_evidence_shows_robustness_by_liquidity_and_settlement_source(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    expanders = [e for e in at.expander if e.label == "Robustness by liquidity and settlement source"]
    assert len(expanders) == 1
    tables = [d.value.to_dict("list") for d in expanders[0].dataframe]
    assert len(tables) == 2
    by_first = {list(t)[0]: t for t in tables}
    assert set(by_first) == {"Liquidity", "Settlement source"}
    liquidity = {r["liquidity"]: r for r in _cut_rows("liquidity")}
    for first, key, rows in [
        ("Liquidity", "liquidity", [liquidity[label] for label in LIQUIDITY_ORDER]),
        ("Settlement source", "settle_source", _cut_rows("settle_source")),
    ]:
        table = by_first[first]
        shown = [LIQUIDITY_LABELS[r[key]] for r in rows] if first == "Liquidity" else [r[key] for r in rows]
        assert table[first] == shown, first
        assert table["City-days"] == [r["events"] for r in rows], first
        for label, values in _brier_columns(rows).items():
            assert table[label] == values, (first, label)
    assert not any("few test days" in text for text in _all_texts(at))


def _after_subheader(at, title):
    """The element right after the subheader `title` on the page."""
    elements = [n for n in at.main if not isinstance(n, Block)]
    i = next(i for i, n in enumerate(elements) if n.type == "subheader" and n.value == title)
    return elements[i + 1]


def test_evidence_encompassing_text_names_both_models_and_the_weather_p_value(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    shared = _shared(monkeypatch)
    assert not at.exception
    text = _after_subheader(at, "Encompassing regression").value
    assert shared.MODEL_LABELS["A_logit"] in text
    assert shared.MODEL_LABELS["B_market_norm"] in text
    assert "not clearly different from zero (p = 0.415)" in text  # ENCOMPASSING: p = 0.4152
    # With a weather p-value below 0.05 the sentence must not say "not clearly".
    at = _evidence_with_weather_p(tmp_path / "low_p", monkeypatch, 0.01)
    assert not at.exception
    text = _after_subheader(at, "Encompassing regression").value
    assert "is clearly different from zero (p = 0.010)" in text
    assert "not clearly" not in text


def _evidence_with_weather_p(tmp_path, monkeypatch, p):
    """AppTest on the Model Evidence page, with `p` as the weather p-value of the encompassing regression."""
    folder = _write_artifacts(tmp_path / "app_artifacts")
    encompassing = pd.read_csv(folder / "encompassing.csv", index_col=0)
    encompassing.loc["logit_weather", "p_value"] = p
    encompassing.to_csv(folder / "encompassing.csv")
    at = _app(tmp_path, monkeypatch, folder=folder)
    at.run()
    at.switch_page("model_evidence.py").run()
    return at


def test_evidence_p_values_read_well(tmp_path, monkeypatch):
    at = _evidence_with_weather_p(tmp_path / "nan", monkeypatch, math.nan)
    assert not at.exception
    assert not at.error
    assert _evidence_table(at, "Term")["p-value"][1] == "n/a"  # the weather term
    text = _after_subheader(at, "Encompassing regression").value
    assert "The weather coefficient has no p-value in the results." in text
    assert "nan" not in text
    at = _evidence_with_weather_p(tmp_path / "p049", monkeypatch, 0.049)
    assert not at.exception
    text = _after_subheader(at, "Encompassing regression").value
    assert "is clearly different from zero (p = 0.049)" in text
    assert "not clearly" not in text


def test_evidence_lists_liquidity_with_the_tightest_spread_first(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    labels = ["Spread ≤ 2¢", "Spread 2–5¢", "Spread > 5¢", "Stale or no quote"]
    assert _evidence_table(at, "Liquidity")["Liquidity"] == labels


def test_evidence_shows_readable_liquidity_labels(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    assert _evidence_table(at, "Liquidity")["Liquidity"] == [
        "Spread ≤ 2¢",
        "Spread 2–5¢",
        "Spread > 5¢",
        "Stale or no quote",
    ]
    # A bucket the page does not know is shown as stored, after the known ones.
    folder = _write_artifacts(tmp_path / "unknown")
    robustness = pd.read_csv(folder / "robustness.csv")
    unknown = robustness[robustness["liquidity"] == "2-5c"].assign(liquidity="other bucket")
    pd.concat([unknown, robustness]).to_csv(folder / "robustness.csv", index=False)
    at = _app(tmp_path, monkeypatch, folder=folder)
    at.run()
    at.switch_page("model_evidence.py").run()
    assert not at.exception
    assert _evidence_table(at, "Liquidity")["Liquidity"] == [
        "Spread ≤ 2¢",
        "Spread 2–5¢",
        "Spread > 5¢",
        "Stale or no quote",
        "other bucket",
    ]


def test_evidence_counts_city_days(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    train, test = SPLIT["train"], SPLIT["test"]
    caption = (
        f"Test period {test['dates'][0]} to {test['dates'][1]}: {test['events']} city-days, "
        f"{test['rows']:,} contract snapshots."
    )
    assert any(c.value == caption for c in at.caption), caption
    expanders = [e for e in at.expander if e.label == "Where these results come from"]
    assert len(expanders) == 1
    text = "\n".join(m.value for m in expanders[0].markdown)
    assert f"Training period: {train['dates'][0]} to {train['dates'][1]} ({train['events']} city-days)." in text
    assert f"Test period: {test['dates'][0]} to {test['dates'][1]} ({test['events']} city-days)." in text
    for value in [e.value for kind in (at.caption, at.markdown) for e in kind]:
        assert not re.search(r"\d days", value), value


def test_evidence_works_with_an_older_data_quality_file(tmp_path, monkeypatch):
    folder = _write_artifacts(tmp_path / "app_artifacts")
    older = {k: v for k, v in DATA_QUALITY.items() if k != "incomplete_bracket_rows"}
    (folder / "data_quality.json").write_text(json.dumps(older))
    at = _app(tmp_path, monkeypatch, folder=folder)
    at.run()
    at.switch_page("model_evidence.py").run()
    assert not at.exception
    assert not at.error
    subheaders = [s.value for s in at.subheader]
    assert "Calibration" in subheaders
    assert "Limitations" in subheaders
    expanders = [e for e in at.expander if e.label == "Where these results come from"]
    assert len(expanders) == 1
    text = "\n".join(m.value for m in expanders[0].markdown)
    lines = [line for line in text.splitlines() if "Data quality:" in line]
    assert len(lines) == 1, text
    for words in ["99.84%", "99.21%", "37 rows without a complete forecast day"]:
        assert words in lines[0], words
    assert "incomplete bracket" not in lines[0]


def test_evidence_counts_the_weather_company_city_days(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    n = next(r["events"] for r in _cut_rows("settle_source") if r["settle_source"] == "The Weather Company")
    total = SPLIT["test"]["events"]
    assert n < total
    expanders = [e for e in at.expander if e.label == "Robustness by liquidity and settlement source"]
    assert len(expanders) == 1
    caption = f"The Weather Company settles {n} of {total} test city-days"
    assert any(caption in c.value for c in expanders[0].caption), caption
    assert not any("few test days" in text for text in _all_texts(at))


def test_evidence_limitation_on_incomplete_brackets_is_exact(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    text = _after_subheader(at, "Limitations").value
    words = (
        "at a snapshot where not every bracket of an event was quoted, the raw mid is kept instead of "
        "the renormalised one"
    )
    assert words in text


def test_evidence_lists_the_limitations(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    assert "Limitations" in [s.value for s in at.subheader]
    text = "\n".join(m.value for m in at.markdown)
    for words in [
        "7–23 h",
        "D-0 08:00",
        "D-0 12:00",
        "34 h",
        "ensemble",
        "one chronological",
        "stale",
        "not every bracket of an event was quoted",
        "before 2025",
        "hypothetical",
        "illustrative",
        "New York",
    ]:
        assert words in text, words


def test_evidence_does_not_show_rmse(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    texts = _all_texts(at) + [e.label for e in at.expander]
    for d in at.dataframe:
        texts += [str(c) for c in d.value.columns]
        texts += [str(x) for column in d.value.columns for x in d.value[column]]
    texts += [c.proto.spec for c in at.get("vega_lite_chart")]
    for text in texts:
        assert "RMSE" not in text, text


def _footer(at):
    """The footer's markdown text (the only markdown with the Open-Meteo credit outside Model Evidence)."""
    footers = [m.value for m in at.markdown if "Weather data by Open-Meteo.com" in m.value]
    assert len(footers) == 1
    return footers[0]


def _explorer_footer(tmp_path, monkeypatch, manifest_changes):
    folder = _write_artifacts(tmp_path / "app_artifacts", manifest_changes=manifest_changes)
    at = _app(tmp_path, monkeypatch, folder=folder)
    at.run()  # the Contract Explorer is the default page
    assert not at.exception
    return _footer(at)


def test_evidence_has_a_data_sources_section_with_the_full_attribution(tmp_path, monkeypatch):
    at = _evidence(tmp_path, monkeypatch)
    assert not at.exception
    elements = [n for n in at.main if not isinstance(n, Block)]
    i = next(i for i, n in enumerate(elements) if n.type == "subheader" and n.value == "Data sources")
    assert elements[i + 1].type == "markdown"
    text = elements[i + 1].value
    for words in DATA_SOURCES_WORDS + [_subset_sentence("NY"), f"Accessed {NOAA_ACCESSED}."]:
        assert words in text, words


def test_ghcn_citation_names_the_station_from_config_and_the_access_date(tmp_path, monkeypatch):
    footer = _explorer_footer(tmp_path, monkeypatch, {"noaa_accessed": "2026-11-04"})
    assert config.CITIES["NY"].ghcnd in footer
    assert _subset_sentence("NY") in footer
    assert "Accessed 2026-11-04." in footer
    assert "Access date not recorded." not in footer


def test_ghcn_citation_without_access_date_says_so(tmp_path, monkeypatch):
    for value in (None, "unknown"):  # None: the manifest has no noaa_accessed
        footer = _explorer_footer(tmp_path / str(value), monkeypatch, {"noaa_accessed": value})
        assert "Access date not recorded." in footer, value
        assert "Accessed" not in footer, value
        assert _subset_sentence("NY") in footer, value


def test_footer_still_shows_sources_when_artifacts_are_missing(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, folder=tmp_path / "missing")
    at.run()
    assert not at.exception
    footer = _footer(at)
    for words in ["Weather data by Open-Meteo.com", "GHCN-Daily", "no warranty"]:
        assert words in footer, words
    assert "Subset used" not in footer
    assert "Accessed" not in footer
    assert "Access date not recorded." not in footer


def test_footer_ignores_any_error_while_reading_the_manifest(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("boom")

    at = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(inference, "load_manifest", boom)
    at.run()
    assert not at.exception
    footer = _footer(at)
    assert "GHCN-Daily" in footer
    assert "Subset used" not in footer


class _RecordingLog:
    """Stands in for shared.log and records every call as (method, args, kwargs)."""

    def __init__(self):
        self.calls = []

    def _record(self, method, args, kwargs):
        self.calls.append((method, args, kwargs))

    def info(self, *args, **kwargs):
        self._record("info", args, kwargs)

    def warning(self, *args, **kwargs):
        self._record("warning", args, kwargs)

    def exception(self, *args, **kwargs):
        self._record("exception", args, kwargs)

    def of(self, method):
        return [call for call in self.calls if call[0] == method]


def test_footer_logs_one_quiet_line_when_artifacts_are_missing(tmp_path, monkeypatch):
    # Missing artifacts are an expected state: one info line, no stack trace. The page's own
    # guarded() shows the InferenceError without logging it.
    at = _app(tmp_path, monkeypatch, folder=tmp_path / "missing")
    shared = _shared(monkeypatch)
    log = _RecordingLog()
    monkeypatch.setattr(shared, "log", log)
    at.run()
    assert not at.exception
    assert len(log.of("info")) == 1, log.calls
    assert not log.of("info")[0][2].get("exc_info"), log.calls
    assert log.of("exception") == [], log.calls

    # Any other error while reading the manifest is still logged with its stack trace.
    def boom():
        raise RuntimeError("boom")

    at = _app(tmp_path, monkeypatch)
    log = _RecordingLog()
    monkeypatch.setattr(shared, "log", log)
    monkeypatch.setattr(inference, "load_manifest", boom)
    at.run()
    assert not at.exception
    assert len(log.of("exception")) >= 1, log.calls


def test_data_sources_without_a_manifest_has_no_subset_or_date(monkeypatch):
    shared = _shared(monkeypatch)
    text = shared.data_sources(None)
    for words in DATA_SOURCES_WORDS:
        assert words in text, words
    assert "Subset used" not in text
    assert "Accessed" not in text
    assert "Access date not recorded." not in text


def test_data_sources_names_every_city_of_the_manifest(monkeypatch):
    shared = _shared(monkeypatch)
    text = shared.data_sources({"cities": ["NY", "CHI"], "noaa_accessed": "2026-11-04"})
    for words in DATA_SOURCES_WORDS + [_subset_sentence("NY"), _subset_sentence("CHI"), "Accessed 2026-11-04."]:
        assert words in text, words


def test_data_sources_marks_a_city_without_a_configured_station(monkeypatch):
    shared = _shared(monkeypatch)
    assert "ZZ" not in config.CITIES
    text = shared.data_sources({"cities": ["NY", "ZZ"], "noaa_accessed": "2026-11-04"})
    assert _subset_sentence("NY") in text
    assert "Subset used: station not configured for ZZ." in text
    assert "Accessed 2026-11-04." in text
