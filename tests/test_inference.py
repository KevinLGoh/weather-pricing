import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from weather_pricing import artifacts, config, experiment, inference, valuation

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
FAMILIES = {
    "rule": ["R_gaussian_rule"],
    "weather": ["A_logit", "A_xgb"],
    "market": ["B_market_mid", "B_market_norm", "B_logit"],
    "combined": ["C_logit", "C_xgb"],
}


def _ticker(day, suffix):
    return f"KXHIGHNY-{day.strftime('%y%b%d').upper()}-{suffix}"


def _prob(model_index, bracket_index, snapshot):
    return round(0.1 + 0.05 * model_index + 0.1 * bracket_index + (0.01 if snapshot == "D-0 08:00" else 0.0), 4)


SCORES = {"C_xgb": (0.071, 0.231, 0.012), "A_xgb": (0.083, 0.262, 0.019)}  # brier, log_loss, ece
DATA_QUALITY = {"rows": 64, "stale_share": 0.0}


def _write_results(folder, skip=()):
    """Small results files in the formats experiment.run() writes."""
    scores = pd.DataFrame(
        [{"model": m, "brier": b, "log_loss": ll, "ece": e} for m, (b, ll, e) in SCORES.items()]
    ).set_index("model")
    tables = {
        "scores.csv": (scores, True),
        "bootstrap.csv": (
            pd.DataFrame(
                [
                    {
                        "baseline": "A_xgb",
                        "challenger": "C_xgb",
                        "question": "does the market add to weather?",
                        "diff": -0.012,
                        "ci_lo": -0.02,
                        "ci_hi": -0.004,
                        "p_value": 0.01,
                        "n_events": 4,
                    }
                ]
            ),
            False,
        ),
        "encompassing.csv": (
            pd.DataFrame(
                {"coef": [0.1, 0.4, 0.7], "se": [0.05, 0.1, 0.1]},
                index=["const", "logit_weather", "logit_market"],
            ),
            True,
        ),
        "robustness.csv": (
            pd.DataFrame([{"snapshot": "D-1 14:00", "n": 16, "events": 2, "A_xgb": 0.09, "cut": "lead"}]),
            False,
        ),
        "reliability.csv": (
            pd.DataFrame(
                [{"bin": 0, "n": 8, "mean_pred": 0.1, "obs_freq": 0.125, "bin_lo": 0.0, "model": "C_xgb"}]
            ),
            False,
        ),
    }
    for name, (df, index) in tables.items():
        if name not in skip:
            df.to_csv(folder / name, index=index)
    dicts = {
        "valuation.json": {"rate": 0.04, "wang_lambda": 0.0, "mean_value": {"C_xgb": 912.5}},
        "split.json": {"rows_in": 100, "test": {"rows": 32, "dates": [DAY1.isoformat(), DAY2.isoformat()]}},
        "data_quality.json": DATA_QUALITY,
    }
    for name, content in dicts.items():
        if name not in skip:
            (folder / name).write_text(json.dumps(content))


def _write_artifacts(folder, drop=(), skip=()):
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
                    "ticker": _ticker(day, suffix),
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
                    row[model] = _prob(m, b, snap)
                rows.append(row)
    df = pd.DataFrame(rows).drop(columns=list(drop))
    folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(folder / "predictions.parquet", index=False)
    manifest = {
        "settings": {"rate": 0.04, "test_frac": 0.3},
        "test": {"dates": [DAY1.isoformat(), DAY2.isoformat()], "rows": len(df)},
        "cities": ["NY"],
        "models": sorted(experiment.MAIN),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    _write_results(folder, skip)
    return folder


def _use(monkeypatch, folder):
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", folder)
    inference.clear_cache()


def _setup(tmp_path, monkeypatch, drop=(), skip=()):
    _use(monkeypatch, _write_artifacts(tmp_path / "app_artifacts", drop, skip))


def _note_inputs():
    out = inference.predict("NY", DAY1, "D-1 14:00", _ticker(DAY1, "B69.5"))
    return {
        "probabilities": out["probabilities"],
        "lead_hours": out["contract"]["lead_hours"],
        "principal": 1000.0,
        "loss_fraction": 0.5,
        "coupon": 20.0,
        "lam": 0.25,
    }


def test_list_cities_dates_and_snapshots(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert inference.list_cities() == ["NY"]
    assert inference.list_dates("NY") == [DAY1, DAY2]
    assert inference.list_snapshots("NY", DAY1) == ["D-1 14:00", "D-0 08:00"]
    assert inference.list_snapshots("NY", "2026-01-03") == ["D-1 14:00", "D-0 08:00"]


def test_list_contracts_sorted_with_readable_labels(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    df = inference.list_contracts("NY", DAY1, "D-1 14:00")
    assert list(df.columns) == ["ticker", "label", "low", "high", "bid", "ask", "mid"]
    assert list(df["label"]) == ["68°F or below", "69–70°F", "71–72°F", "73°F or above"]
    assert list(df["ticker"]) == [_ticker(DAY1, s) for s in ["T68", "B69.5", "B71.5", "T73"]]
    assert list(df["low"]) == [-math.inf, 69.0, 71.0, 73.0]
    assert list(df["high"]) == [68.0, 70.0, 72.0, math.inf]
    assert df.loc[df["ticker"] == _ticker(DAY1, "B69.5"), "mid"].item() == pytest.approx(0.67)


def test_unknown_city_raises_invalid_input_listing_choices(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with pytest.raises(inference.InvalidInputError, match="NY"):
        inference.list_dates("XX")
    assert issubclass(inference.InvalidInputError, inference.InferenceError)


def test_bad_date_string_raises_invalid_input(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    for bad in ["2026-13-45", "soon"]:
        with pytest.raises(inference.InvalidInputError):
            inference.list_snapshots("NY", bad)


def test_unknown_snapshot_raises_invalid_input(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with pytest.raises(inference.InvalidInputError) as info:
        inference.list_contracts("NY", DAY1, "D-2 09:00")
    message = str(info.value)
    allowed = [f"D-{d} {h:02d}:00" for d, h in config.SNAPSHOTS]
    assert allowed == ["D-1 14:00", "D-1 22:00", "D-0 08:00", "D-0 12:00"]
    positions = [message.find(s) for s in allowed]
    assert all(p >= 0 for p in positions)
    assert positions == sorted(positions)


def test_date_outside_test_period_raises_data_not_found_with_range(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with pytest.raises(inference.DataNotFoundError, match="2026-01-02(.|\n)*2026-01-03"):
        inference.list_snapshots("NY", date(2026, 1, 10))
    assert issubclass(inference.DataNotFoundError, inference.InferenceError)


def test_unknown_contract_raises_data_not_found(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    with pytest.raises(inference.DataNotFoundError):
        inference.predict("NY", DAY1, "D-1 14:00", "KXHIGHNY-26JAN02-B99.5")


def test_predict_returns_all_models_families_and_outcome(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = inference.predict("NY", "2026-01-02", "D-0 08:00", _ticker(DAY1, "B69.5"))
    contract = out["contract"]
    assert contract["label"] == "69–70°F"
    assert (contract["bid"], contract["ask"], contract["mid"]) == pytest.approx((0.65, 0.69, 0.67))
    assert contract["spread"] == pytest.approx(0.04)
    assert not contract["stale"]
    assert contract["settle_source"] == "kalshi"
    assert contract["lead_hours"] == pytest.approx(16.0)
    assert list(out["probabilities"]) == experiment.MAIN
    for m, model in enumerate(experiment.MAIN):
        assert out["probabilities"][model] == pytest.approx(_prob(m, 3, "D-0 08:00"))
    assert out["families"] == FAMILIES
    assert out["outcome"]["y"] == 1
    assert isinstance(out["outcome"]["y"], int)
    assert out["outcome"]["settled_value"] == pytest.approx(70.0)
    loser = inference.predict("NY", DAY2, "D-1 14:00", _ticker(DAY2, "T73"))
    assert loser["outcome"]["y"] == 0


def test_missing_artifacts_folder_names_the_build_command(tmp_path, monkeypatch):
    _use(monkeypatch, tmp_path / "missing")
    with pytest.raises(inference.ArtifactsMissingError, match="evaluate --save-artifacts"):
        inference.list_cities()
    assert issubclass(inference.ArtifactsMissingError, inference.InferenceError)


def test_predictions_missing_a_column_raises_artifacts_missing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, drop=("mid",))
    with pytest.raises(inference.ArtifactsMissingError, match="mid"):
        inference.list_cities()


RAW_ERROR_WORDS = [
    "ArrowInvalid",
    "JSONDecodeError",
    "ValueError",
    "DateParseError",
    "EmptyDataError",
    "FileNotFoundError",
    "Errno",
    "Traceback",
]


def _assert_readable(error, folder, file_name):
    """Plain-words message; the folder path is removed first so tmp_path names cannot match."""
    message = str(error).replace(str(folder), "<folder>")
    if error.__cause__ is not None:
        assert str(error.__cause__).replace(str(folder), "<folder>") not in message, message
    assert "evaluate --save-artifacts" in message, message
    assert file_name in message, message
    assert "damaged" in message, message
    for word in RAW_ERROR_WORDS:
        assert word not in message, (word, message)


def _junk_predictions(folder):
    (folder / "predictions.parquet").write_bytes(b"not a parquet file")


def _delete_manifest(folder):
    (folder / "manifest.json").unlink()


def _truncated_manifest(folder):
    (folder / "manifest.json").write_text("{")


def _manifest_not_an_object(folder):
    (folder / "manifest.json").write_text("[]")


def _unparseable_dates(folder):
    df = pd.read_parquet(folder / "predictions.parquet")
    df["target_date"] = "not-a-date"
    df.to_parquet(folder / "predictions.parquet", index=False)


def test_damaged_artifacts_raise_artifacts_missing(tmp_path, monkeypatch):
    cases = [
        ("predictions.parquet", _junk_predictions),
        ("manifest.json", _delete_manifest),
        ("manifest.json", _truncated_manifest),
        ("manifest.json", _manifest_not_an_object),
        ("predictions.parquet", _unparseable_dates),
    ]
    for file_name, damage in cases:
        folder = _write_artifacts(tmp_path / damage.__name__)
        damage(folder)
        _use(monkeypatch, folder)
        with pytest.raises(inference.ArtifactsMissingError) as info:
            inference.list_cities()
        _assert_readable(info.value, folder, file_name)


def test_failure_is_not_cached_so_a_retry_works(tmp_path, monkeypatch):
    folder = tmp_path / "app_artifacts"
    _use(monkeypatch, folder)
    with pytest.raises(inference.ArtifactsMissingError):
        inference.list_cities()
    _write_artifacts(folder)
    assert inference.list_cities() == ["NY"]


def test_damaged_results_file_raises_artifacts_missing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    (tmp_path / "app_artifacts" / "bootstrap.csv").write_bytes(b"")
    with pytest.raises(inference.ArtifactsMissingError) as info:
        inference.load_evidence()
    _assert_readable(info.value, tmp_path / "app_artifacts", "bootstrap.csv")


def test_artifacts_are_read_once_and_cached(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    real = pd.read_parquet
    calls = []

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(inference.pd, "read_parquet", counting)
    inference.list_cities()
    inference.list_dates("NY")
    inference.list_snapshots("NY", DAY1)
    inference.list_contracts("NY", DAY1, "D-1 14:00")
    inference.predict("NY", DAY1, "D-1 14:00", _ticker(DAY1, "T68"))
    inference.predict("NY", DAY2, "D-0 08:00", _ticker(DAY2, "T73"))
    assert len(calls) == 1


def test_price_note_matches_valuation_value(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    inference.clear_cache()
    inputs = _note_inputs()
    df = inference.price_note(**inputs)
    assert list(df.columns) == ["model", "family", "probability", "pricing_probability", "value", "change_vs_baseline"]
    assert list(df["model"]) == list(inputs["probabilities"])
    family_of = {m: family for family, models in FAMILIES.items() for m in models}
    years = valuation.horizon_years(inputs["lead_hours"])
    note = valuation.Note(1000.0, 0.5, 20.0)
    for _, row in df.iterrows():
        p = inputs["probabilities"][row["model"]]
        assert row["family"] == family_of[row["model"]]
        assert np.isclose(row["probability"], p)
        assert np.isclose(row["pricing_probability"], valuation.wang(p, 0.25))
        assert np.isclose(row["value"], valuation.value(p, years, note, 0.04, 0.25))


def test_price_note_reports_change_from_the_baseline(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    inference.clear_cache()
    inputs = _note_inputs()
    df = inference.price_note(**inputs).set_index("model")
    assert df.loc["A_xgb", "change_vs_baseline"] == 0
    expected = df.loc["C_xgb", "value"] - df.loc["A_xgb", "value"]
    assert np.isclose(df.loc["C_xgb", "change_vs_baseline"], expected)
    market = inference.price_note(**inputs, baseline="B_market_norm").set_index("model")
    assert market.loc["B_market_norm", "change_vs_baseline"] == 0
    expected = market.loc["A_xgb", "value"] - market.loc["B_market_norm", "value"]
    assert np.isclose(market.loc["A_xgb", "change_vs_baseline"], expected)


def test_price_note_rejects_invalid_inputs(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    inference.clear_cache()
    inputs = _note_inputs()
    too_high = {**inputs["probabilities"], "C_xgb": 1.2}
    cases = [
        ("principal", {"principal": 0}),
        ("principal", {"principal": math.inf}),
        ("loss_fraction", {"loss_fraction": 1.5}),
        ("loss_fraction", {"loss_fraction": -0.1}),
        ("coupon", {"coupon": -1}),
        ("lam", {"lam": math.nan}),
        ("lam", {"lam": 3}),
        ("lead_hours", {"lead_hours": -1}),
        ("probabilities", {"probabilities": too_high}),
        ("probabilities", {"probabilities": {}}),
        ("baseline", {"baseline": "nope"}),
        ("principal", {"principal": "abc"}),
    ]
    for name, change in cases:
        with pytest.raises(inference.InvalidInputError) as info:
            inference.price_note(**{**inputs, **change})
        assert name in str(info.value), (name, change)


def test_price_note_without_rate_in_manifest_raises_artifacts_missing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    inference.clear_cache()
    inputs = _note_inputs()
    path = tmp_path / "app_artifacts" / "manifest.json"
    manifest = json.loads(path.read_text())
    without_rate = {k: v for k, v in manifest["settings"].items() if k != "rate"}
    cases = [
        without_rate,
        {**manifest["settings"], "rate": "four"},
        {**manifest["settings"], "rate": math.nan},  # json writes NaN
        {**manifest["settings"], "rate": math.inf},  # json writes Infinity
        None,
    ]
    for settings in cases:
        path.write_text(json.dumps({**manifest, "settings": settings}))
        inference.clear_cache()
        with pytest.raises(inference.ArtifactsMissingError) as info:
            inference.price_note(**inputs)
        assert "evaluate --save-artifacts" in str(info.value), settings


def test_price_note_rejects_unknown_model_and_bools(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    inputs = _note_inputs()
    cases = [
        ("probabilities", {"probabilities": {"nope": 0.5}, "baseline": "nope"}),
        ("principal", {"principal": True}),
    ]
    for name, change in cases:
        with pytest.raises(inference.InvalidInputError) as info:
            inference.price_note(**{**inputs, **change})
        assert name in str(info.value), (name, change)


def test_price_note_accepts_numpy_numbers(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    inputs = _note_inputs()
    plain = inference.price_note(**{**inputs, "lead_hours": 34.0})
    numpy_inputs = {
        **inputs,
        "principal": np.float64(1000.0),
        "coupon": np.int64(20),
        "lam": np.float64(0.25),
        "lead_hours": np.float64(34.0),
    }
    df = inference.price_note(**numpy_inputs)
    assert isinstance(df, pd.DataFrame)
    assert list(df["model"]) == list(plain["model"])
    assert np.isclose(df["value"], plain["value"]).all()


def test_invalid_input_messages_are_readable(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    inputs = _note_inputs()
    with pytest.raises(inference.InvalidInputError) as info:
        inference.price_note(**{**inputs, "principal": float("inf")})
    message = str(info.value)
    assert "1,000,000,000" in message, message
    assert "e+" not in message, message


def test_load_evidence_returns_tables_and_dicts(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    inference.clear_cache()
    evidence = inference.load_evidence()
    tables = ["scores", "bootstrap", "encompassing", "reliability", "robustness"]
    for key in tables:
        assert isinstance(evidence[key], pd.DataFrame), key
    for key in ["valuation", "split", "manifest", "data_quality"]:
        assert isinstance(evidence[key], dict), key
    assert evidence["scores"].loc["C_xgb", "brier"] == pytest.approx(0.071)
    assert evidence["scores"].loc["A_xgb", "log_loss"] == pytest.approx(0.262)
    assert evidence["encompassing"].loc["logit_market", "coef"] == pytest.approx(0.7)
    assert evidence["bootstrap"].loc[0, "challenger"] == "C_xgb"
    assert evidence["robustness"].loc[0, "cut"] == "lead"
    assert evidence["reliability"].loc[0, "model"] == "C_xgb"
    assert evidence["valuation"]["mean_value"]["C_xgb"] == pytest.approx(912.5)
    assert evidence["split"]["rows_in"] == 100
    assert evidence["manifest"]["settings"]["rate"] == pytest.approx(0.04)
    assert evidence["data_quality"] == DATA_QUALITY


def test_load_evidence_without_data_quality_gives_none(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, skip=("data_quality.json",))
    inference.clear_cache()
    evidence = inference.load_evidence()
    assert evidence["data_quality"] is None
    assert isinstance(evidence["scores"], pd.DataFrame)


def test_load_evidence_missing_scores_raises_artifacts_missing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, skip=("scores.csv",))
    inference.clear_cache()
    with pytest.raises(inference.ArtifactsMissingError) as info:
        inference.load_evidence()
    assert "scores.csv" in str(info.value)
    assert "evaluate --save-artifacts" in str(info.value)
