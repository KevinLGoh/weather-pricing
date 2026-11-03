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

from weather_pricing import artifacts, experiment, inference  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = ["contract_explorer.py", "note_pricer.py"]

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


def _event(day):
    return f"KXHIGHNY-{day.strftime('%y%b%d').upper()}"


def _probability(m, b):
    """Probability of model number `m` (in experiment.MAIN) for bracket number `b` (in BRACKETS)."""
    return round(0.103 + 0.05 * m + 0.1 * b, 4)


def _write_artifacts(folder):
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
        "Family": [shared.FAMILY_LABELS[f] for f in prices["family"]],
        "Probability": [f"{p:.1%}" for p in prices["probability"]],
        "Pricing probability": [f"{p:.1%}" for p in prices["pricing_probability"]],
        "Value": [f"${v:,.2f}" for v in prices["value"]],
        "Change vs baseline": [_change(c) for c in prices["change_vs_baseline"]],
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
    assert any("evaluate --save-artifacts" in e.value for e in at.error)
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
    assert _selectbox(at, "City").options == ["NY"]
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
    assert _selectbox(at, "City").options == ["NY"]


def test_explorer_shows_all_eight_model_probabilities_by_family(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.run()
    shared = _shared(monkeypatch)
    assert not at.exception
    # Default choice: newest day, first snapshot, lowest bracket ("68°F or below" = T68).
    b = [suffix for *_, suffix in BRACKETS].index("T68")
    subheaders = [s.value for s in at.subheader]
    for family in shared.FAMILY_LABELS.values():
        assert family in subheaders, family
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
    assert any(all(w in t for w in ["NY", "2026-01-02", "D-0 08:00", "69–70°F"]) for t in _texts(at))
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
