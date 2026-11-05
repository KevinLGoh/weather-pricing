"""The multi-city artifacts, from the build through every app page."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_experiment import _fake_git, _snapshots
from test_rolling import _exact_frame, _oos
from test_weather_plus import _with_guidance

from weather_pricing import app_build, artifacts, exact_runs, inference, rolling

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def _finding_results(root):
    """Real rolling and exact-run tables, written by their own code from synthetic rows."""
    frame = pd.concat([_oos(seed=1, n_days=240), _oos(seed=2, n_days=240).assign(city="CHI")], ignore_index=True)
    frame["event_id"] = frame["city"] + ":" + frame["target_date"].astype(str)
    rng = np.random.default_rng(5)
    for name in rolling.SCORED:
        if name not in frame:
            frame[name] = np.clip(frame["B_market_norm"] + rng.normal(0, 0.02, len(frame)), 0.01, 0.99)
    frame["spread"], frame["stale"], frame["cum_volume"] = 0.02, False, 100.0
    frame.loc[frame["fold"] == 0, "F_blend"] = np.nan
    rolling.report(frame, _mkdir(root / "rolling"), n_boot=20)

    df = _exact_frame()
    columns = [c for c in df.columns if c.startswith(("hrrr_", "nbmx_", "ecmwfx_"))]
    exact_runs.run(df.drop(columns=columns), df[["ticker", "snapshot", *columns]], root / "exact_runs", n_boot=20)


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _enrich(df):
    """Stand-in for the archive lookups: the same columns, on synthetic rows."""
    df = pd.concat([_with_guidance(g, seed=i) for i, (_, g) in enumerate(df.groupby("city"))], ignore_index=True)
    rng = np.random.default_rng(9)
    events = df[["event_ticker", "snapshot", "settled_value"]].drop_duplicates(["event_ticker", "snapshot"]).copy()
    for col in ["nam_max", "om_ecmwf_max", "om_icon_max", "om_gem_max"]:
        events[col] = events["settled_value"] + rng.normal(0, 2.5, len(events))
    return df.merge(events.drop(columns="settled_value"), on=["event_ticker", "snapshot"])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Artifacts from the real build, once for all tests here."""
    root = tmp_path_factory.mktemp("build")
    processed = _mkdir(root / "processed")
    frame = _snapshots(n_days=160)
    frame[frame["city"] == "NY"].to_parquet(processed / "snapshots.parquet", index=False)
    frame[frame["city"] == "CHI"].to_parquet(processed / "snapshots_chi.parquet", index=False)
    results = _mkdir(root / "results")
    (results / "data_quality.json").write_text(json.dumps({"noaa_outcome_matches_result": 0.99}))
    (results / "data_quality_snapshots_chi.json").write_text(json.dumps({"noaa_outcome_matches_result": 1.0}))
    _finding_results(results)
    patch = pytest.MonkeyPatch()
    patch.setattr(app_build, "PROCESSED_DIR", processed)
    patch.setattr(artifacts, "RAW_DIR", root / "raw")
    patch.setattr(artifacts, "_git", _fake_git)
    out = root / "app_artifacts"
    manifest = app_build.build_artifacts(
        ["snapshots", "snapshots_chi"], n_boot=20, out_dir=out, results_root=results, enrich_steps=(_enrich,)
    )
    patch.undo()
    return out, manifest


@pytest.fixture
def use(built, monkeypatch):
    out, manifest = built
    monkeypatch.setattr(inference, "_artifacts_dir", out)
    inference.clear_cache()
    yield out, manifest
    inference.clear_cache()


def test_build_writes_one_evidence_folder_per_city_and_the_finding_tables(built):
    out, manifest = built
    assert manifest["cities"] == ["CHI", "NY"]
    assert manifest["models"] == sorted(inference.ALL_MODELS)
    for city in manifest["cities"]:
        for name in ["scores.csv", "bootstrap.csv", "encompassing.csv", "encompassing_extended.csv", "split.json"]:
            assert (out / "evidence" / city / name).exists(), (city, name)
            assert f"evidence/{city}/{name}" in manifest["files"]
        assert set(manifest["splits"][city]) >= {"train", "test"}
    for name in inference.FINDING_TABLES:
        assert (out / "finding" / f"{name}.csv").exists(), name
    assert not (out / "scores.csv").exists()  # results now live per city


def test_each_city_keeps_its_own_data_quality_report(use):
    assert inference.load_evidence("NY")["data_quality"] == {"noaa_outcome_matches_result": 0.99}
    assert inference.load_evidence("CHI")["data_quality"] == {"noaa_outcome_matches_result": 1.0}


def test_predictions_hold_both_cities_each_on_its_own_test_period(use):
    assert inference.list_cities() == ["CHI", "NY"]
    predictions = pd.read_parquet(use[0] / "predictions.parquet")
    assert set(inference.ALL_MODELS) <= set(predictions.columns)
    for city, split in use[1]["splits"].items():
        rows = predictions[predictions["city"] == city]
        assert len(rows) == split["test"]["rows"]
        assert str(rows["target_date"].min()) >= split["test"]["dates"][0]
    # Every model must be out of sample, a probability, and present on every row.
    values = predictions[inference.ALL_MODELS].to_numpy()
    assert np.isfinite(values).all() and (values > 0).all() and (values < 1).all()


def test_predict_returns_every_family_in_display_order(use):
    day = inference.list_dates("CHI")[0]
    snapshot = inference.list_snapshots("CHI", day)[0]
    ticker = inference.list_contracts("CHI", day, snapshot)["ticker"].iloc[0]
    result = inference.predict("CHI", day, snapshot, ticker)
    assert list(result["families"]) == list(inference.FAMILIES)
    assert list(result["probabilities"]) == inference.ALL_MODELS
    prices = inference.price_note(result["probabilities"], result["contract"]["lead_hours"], 1000, 0.5, 0, 0, baseline="X_xgb")
    assert prices.set_index("model").loc["X_xgb", "change_vs_baseline"] == 0
    assert set(prices["family"]) == set(inference.FAMILIES)


def test_city_evidence_differs_and_a_wrong_city_is_refused(use):
    assert inference.list_evidence_cities() == ["CHI", "NY"]
    ny, chi = inference.load_evidence("NY"), inference.load_evidence("CHI")
    assert not ny["scores"].equals(chi["scores"])
    assert {"W_xgb", "X_xgb", "CX_xgb"} <= set(ny["scores"].index)
    assert ny["encompassing_extended"] is not None
    with pytest.raises(inference.InvalidInputError, match="DEN"):
        inference.load_evidence("DEN")


def test_ladder_reads_the_right_comparisons(use):
    finding = inference.load_finding()
    ladder = finding["ladder"].set_index("test")
    assert list(ladder.index) == ["Original", "Stronger baseline", "Replication", "Rolling origin", "Exact runs"]
    boot = inference.load_evidence("CHI")["bootstrap"]
    row = boot[(boot["baseline"] == "W_xgb") & (boot["challenger"] == "B_market_norm")].iloc[0]
    assert ladder.loc["Replication", "lead"] == pytest.approx(row["diff"])
    assert (ladder["ci_lo"] <= ladder["lead"]).all() and (ladder["lead"] <= ladder["ci_hi"]).all()


def test_build_needs_the_rolling_results_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="rolling"):
        app_build.build_artifacts(["snapshots"], out_dir=tmp_path / "a", results_root=tmp_path / "r")


def test_a_single_run_build_removes_the_city_folders(built, tmp_path, monkeypatch):
    import shutil

    out = tmp_path / "copy"
    shutil.copytree(built[0], out)
    results = _mkdir(tmp_path / "results")
    (results / "scores.csv").write_text("model,brier\nA_xgb,0.1\n")
    monkeypatch.setattr(artifacts, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(artifacts, "_git", _fake_git)
    artifacts.save(out, {}, pd.DataFrame({"a": [1]}), results, {})
    assert not (out / "evidence").exists() and not (out / "finding").exists()
    assert (out / "scores.csv").exists()


def _app(built, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.syspath_prepend(str(APP_DIR))
    import importlib

    shared = importlib.import_module("shared")
    monkeypatch.setattr(shared, "ARTIFACTS_DIR", built[0])
    monkeypatch.setattr(inference, "_artifacts_dir", None)
    inference.clear_cache()
    return AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=60), shared


def _box(at, label):
    return next(s for s in at.selectbox if s.label == label)


def test_explorer_defaults_to_new_york_and_switches_to_chicago(built, monkeypatch):
    at, shared = _app(built, monkeypatch)
    at.run()
    assert not at.exception
    assert _box(at, "City").value == "NY"
    assert any("Central Park" in c.value for c in at.caption)
    labels = {m.label for m in at.metric}
    assert {shared.MODEL_LABELS[m] for m in inference.ALL_MODELS} <= labels
    assert shared.FAMILY_LABELS["extended_weather"] in [s.value for s in at.subheader]
    assert any("Stronger weather adds NWS guidance" in c.value for c in at.caption)

    _box(at, "City").set_value("CHI").run()
    assert not at.exception
    assert any("Midway" in c.value for c in at.caption)
    assert at.session_state["selection"]["city"] == "CHI"


def test_note_pricer_compares_with_the_strongest_weather_model_for_chicago(built, monkeypatch):
    at, shared = _app(built, monkeypatch)
    at.run()
    _box(at, "City").set_value("CHI").run()
    at.switch_page("note_pricer.py").run()
    assert not at.exception
    assert not at.error
    assert _box(at, "Baseline model").value == shared.MODEL_LABELS["X_xgb"]
    table = at.dataframe[0].value
    assert set(table["Model"]) == {shared.MODEL_LABELS[m] for m in inference.ALL_MODELS}
    assert any("extended weather + market" in m.value for m in at.markdown)


def test_evidence_page_shows_the_finding_and_each_city(built, monkeypatch):
    at, shared = _app(built, monkeypatch)
    at.run()
    at.switch_page("model_evidence.py").run()
    assert not at.exception
    assert not at.error
    subheaders = [s.value for s in at.subheader]
    for title in ["Does the market's lead survive better tests?", "Rolling-origin evaluation", "Exact model runs", "Results for one city", "Scores", "Limitations"]:
        assert title in subheaders, title
    assert any("New York and Chicago" in i.value for i in at.info)
    assert any("Two cities" in m.value for m in at.markdown)
    assert not any("New York only" in m.value for m in at.markdown)
    assert _box(at, "City").value == "NY"
    assert _box(at, "City").options == ["Chicago (Midway)", "New York (Central Park)"]

    ny_rows = at.caption
    _box(at, "City").set_value("CHI").run()
    assert not at.exception
    assert any("Test period" in c.value for c in ny_rows)
    scores = next(d.value for d in at.dataframe if "Log loss" in d.value.columns)
    assert set(scores["Model"]) == {shared.MODEL_LABELS[m] for m in inference.ALL_MODELS}


def test_comparison_keeps_its_old_wording_by_default(monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.syspath_prepend(str(APP_DIR))
    import importlib

    shared = importlib.import_module("shared")
    assert shared.comparison_sentence(-5.0, "x").startswith("Under the combined XGBoost model")
    assert shared.comparison_sentence(-5.0, "x", "another model").startswith("Under another model")
