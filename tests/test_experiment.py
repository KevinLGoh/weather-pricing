"""evaluate end to end on a small synthetic snapshot frame, with and without --save-artifacts."""

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from weather_pricing import artifacts, cli, experiment

MODEL_NAMES = [
    "R_gaussian_rule", "A_logit", "A_xgb", "B_market_mid", "B_market_norm", "B_logit", "C_logit", "C_xgb",
]
LEARNED = {"R_gaussian_rule", "A_logit", "A_xgb", "B_logit", "C_logit", "C_xgb"}
APP_COLUMNS = [
    "city", "event_id", "event_ticker", "ticker", "target_date", "snapshot", "lead_hours", "strike_type",
    "low", "high", "bid", "ask", "mid", "mid_norm", "spread", "stale", "settle_source", "settled_value", "y",
]
ARTIFACT_FILES = ["models.pkl", "predictions.parquet", "scores.csv", "manifest.json"]


def _snapshots(n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for city in ["NY", "CHI"]:
        for d in range(n_days):
            target = date(2025, 1, 1) + timedelta(days=d)
            truth = 60 + 15 * np.sin(2 * np.pi * d / 365) + rng.normal(0, 4)
            private = truth + rng.normal(0, 1.5)
            for snap, lead in [("D-1 14:00", 34.0), ("D-0 08:00", 16.0)]:
                fc = truth + rng.normal(0.5, 3.0)
                centre = int(round(fc))
                edges = [centre - 5, centre - 3, centre - 1, centre + 1, centre + 3]
                brackets = [(-np.inf, edges[0] - 1)] + [(a, a + 1) for a in edges[:-1]] + [(edges[-1], np.inf)]
                mids = [norm.cdf((hi + 0.5 - private) / 1.6) - norm.cdf((lo - 0.5 - private) / 1.6) for lo, hi in brackets]
                for (lo, hi), mid in zip(brackets, mids):
                    kind = "less" if np.isinf(lo) else "greater" if np.isinf(hi) else "between"
                    rows.append(
                        dict(city=city, event_ticker=f"{city}{d}", ticker=f"{city}{d}-{lo}", target_date=target,
                             snapshot=snap, lead_hours=lead, strike_type=kind, low=lo, high=hi, fc_tmax=fc,
                             fc_tmax_prev=fc + rng.normal(0, 1), obs_tmax_prevday=truth + rng.normal(0, 5),
                             bid=max(mid - 0.01, 0.0), ask=min(mid + 0.01, 1.0), mid=mid, mid_norm=mid / sum(mids),
                             spread=0.02, cum_volume=1000.0, stale=False, settle_source="nws",
                             settled_value=round(truth), noaa_tmax=round(truth), y=int(lo <= round(truth) <= hi))
                    )
    return pd.DataFrame(rows)


def _fake_git(*args):
    answers = {
        ("rev-parse", "--show-toplevel"): str(artifacts.ROOT),
        ("rev-parse", "HEAD"): "abc123",
        ("rev-parse", "HEAD:src"): "def456",
        ("status", "--porcelain", "--untracked-files=no"): "",
    }
    return answers.get(args, "unknown")


def _run_cli(tmp_path, monkeypatch, *extra):
    frame = _snapshots()
    processed = tmp_path / "processed"
    processed.mkdir()
    frame.to_parquet(processed / "snapshots.parquet", index=False)
    monkeypatch.setattr(cli, "PROCESSED_DIR", processed)
    monkeypatch.setattr(cli, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(experiment, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path / "app_artifacts")
    monkeypatch.setattr(artifacts, "_git", _fake_git)
    cli.main(["evaluate", "--n-boot", "50", *extra])
    return frame


def test_evaluate_with_save_artifacts_writes_app_artifacts(tmp_path, monkeypatch):
    frame = _run_cli(tmp_path, monkeypatch, "--save-artifacts")
    out = tmp_path / "app_artifacts"
    for name in ARTIFACT_FILES:
        assert (out / name).exists(), name

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["rows_in"] == len(frame)
    assert manifest["git_commit"] == "abc123"
    assert manifest["settings"]["n_boot"] == 50
    assert manifest["cities"] == ["CHI", "NY"]
    assert manifest["data_date_range"] == [str(frame["target_date"].min()), str(frame["target_date"].max())]

    predictions = pd.read_parquet(out / "predictions.parquet")
    missing = set(APP_COLUMNS + MODEL_NAMES) - set(predictions.columns)
    assert not missing, missing

    models = artifacts.load_models(out / "models.pkl")
    assert isinstance(models, dict)
    assert set(models) == LEARNED


def test_predictions_cover_only_the_test_period(tmp_path, monkeypatch):
    _run_cli(tmp_path, monkeypatch, "--save-artifacts")
    out = tmp_path / "app_artifacts"
    manifest = json.loads((out / "manifest.json").read_text())
    predictions = pd.read_parquet(out / "predictions.parquet")

    # Both sides as ISO date strings, which sort like dates.
    assert str(predictions["target_date"].min()) >= manifest["test"]["dates"][0]
    assert len(predictions) == manifest["test"]["rows"] == manifest["predictions_rows"]


def test_evaluate_without_flag_writes_no_artifacts(tmp_path, monkeypatch):
    _run_cli(tmp_path, monkeypatch)
    assert not (tmp_path / "app_artifacts").exists()
    assert (tmp_path / "results" / "scores.csv").exists()
