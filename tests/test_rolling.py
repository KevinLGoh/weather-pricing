from datetime import date

import numpy as np
import pandas as pd
from test_models import synthetic

from weather_pricing import exact_runs, guidance_plus, rolling
from weather_pricing.collect import openmeteo
from weather_pricing.config import CITIES, EXACT_RUN_MODELS, EXACT_RUN_START

TZ = "America/New_York"
TARGET = date(2026, 7, 4)
SNAP = pd.Timestamp("2026-07-04 16:00", tz="UTC")  # 12:00 EDT


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_folds_start_after_the_first_training_window_and_do_not_overlap():
    folds = rolling.make_folds(date(2023, 1, 1), date(2024, 8, 15), initial_months=12, step_months=3)
    assert folds[0] == (date(2024, 1, 1), date(2024, 4, 1))
    assert folds[-1] == (date(2024, 7, 1), date(2024, 10, 1))
    assert all(a[1] == b[0] for a, b in zip(folds, folds[1:]))


def test_each_fold_trains_only_on_earlier_days():
    df = synthetic(800)
    df["event_id"] = df["city"] + ":" + df["target_date"].astype(str)
    seen = []

    def fit(train, test, val_mask):
        seen.append((train["target_date"].max(), test["target_date"].min()))
        assert val_mask.any() and not val_mask.all()
        assert train.loc[val_mask, "target_date"].min() > train.loc[~val_mask, "target_date"].max()
        return {"m": np.full(len(test), 0.5)}

    out = rolling.run_city(df, fit)
    assert all(last_train < first_test for last_train, first_test in seen)
    assert out.groupby("ticker")["snapshot"].value_counts().max() == 1
    assert out["target_date"].min() >= date(2025, 7, 1)


def _oos(seed=0, n_days=400):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=n_days).date
    frame = pd.DataFrame({"target_date": np.repeat(days, 6)})
    frame["event_id"] = "NY:" + frame["target_date"].astype(str)
    frame["city"] = "NY"
    frame["snapshot"] = "D-0 08:00"
    frame["fold"] = np.repeat(np.arange(n_days) * 4 // n_days, 6)
    x, m = rng.normal(size=len(frame)), rng.normal(size=len(frame))
    frame["y"] = (rng.random(len(frame)) < 1 / (1 + np.exp(-(x + m)))).astype(int)
    frame["X_xgb"] = 1 / (1 + np.exp(-x))
    frame["B_market_norm"] = 1 / (1 + np.exp(-m))
    return frame


def test_forward_blend_never_uses_later_folds():
    frame = _oos()
    base = rolling.forward_blend(frame)
    assert base[frame["fold"] == 0].isna().all()
    assert base[frame["fold"] > 0].notna().all()
    changed = frame.copy()
    later = changed["fold"] == 3
    changed.loc[later, "y"] = 1 - changed.loc[later, "y"]
    after = rolling.forward_blend(changed)
    assert np.array_equal(after[~later].dropna(), base[~later].dropna())


def test_liquidity_buckets_put_stale_rows_first():
    frame = pd.DataFrame({"spread": [0.01, 0.04, 0.2, 0.01, np.nan], "stale": [False, False, False, True, True]})
    assert rolling.liquidity_bucket(frame).tolist() == ["spread<=2c", "2-5c", ">5c", "stale/no quote", "stale/no quote"]


def test_report_can_be_rebuilt_from_saved_predictions(tmp_path):
    frame = _oos(n_days=240)
    rng = np.random.default_rng(5)
    for name in rolling.SCORED:
        if name not in frame:
            frame[name] = np.clip(frame["B_market_norm"] + rng.normal(0, 0.02, len(frame)), 0.01, 0.99)
    frame["spread"], frame["stale"], frame["cum_volume"] = 0.02, False, 100.0
    frame.loc[frame["fold"] == 0, "F_blend"] = np.nan
    md = rolling.report(frame, tmp_path, n_boot=20)
    assert "Rolling-origin evaluation" in md
    for name in ["scores.csv", "comparisons.csv", "by_fold.csv", "by_liquidity.csv", "by_maturity.csv", "forward_blend.csv"]:
        assert (tmp_path / name).exists()
    assert len(pd.read_csv(tmp_path / "by_maturity.csv")) == 4


def test_released_cycles_exclude_runs_not_out_yet():
    # 16Z with a 2 h delay: the 15Z run is out at 17Z, so 12Z is the newest.
    assert guidance_plus._released_cycles(SNAP, 3, 2.0)[0] == _ts("2026-07-04 12:00")
    assert guidance_plus._released_cycles(SNAP, 6, 7.0)[0] == _ts("2026-07-04 06:00")
    assert guidance_plus._released_cycles(_ts("2026-07-04 17:00"), 3, 2.0)[0] == _ts("2026-07-04 15:00")


def _run_frame(first, last, temp):
    times = pd.date_range(first, last, freq="h", tz="UTC")
    return pd.DataFrame({"valid_time": times, "temp_f": temp})


def test_exact_run_uses_the_newest_run_that_reaches_the_warm_hours():
    runs = {
        _ts("2026-07-04 12:00"): _run_frame("2026-07-04 12:00", "2026-07-04 22:00", 999.0),  # stops too soon
        _ts("2026-07-04 09:00"): _run_frame("2026-07-04 09:00", "2026-07-05 03:00", 80.0),
    }
    high, age = guidance_plus.exact_run_max(SNAP, TARGET, TZ, 3, 2.0, lambda r: runs.get(r, runs[_ts("2026-07-04 12:00")].iloc[0:0]))
    assert (high, age) == (80.0, 7.0)


def test_exact_run_ignores_hours_outside_the_rest_of_the_climate_day():
    frame = _run_frame("2026-07-04 12:00", "2026-07-05 08:00", 70.0)
    frame.loc[frame["valid_time"] == _ts("2026-07-04 13:00"), "temp_f"] = 500.0  # before the snapshot
    frame.loc[frame["valid_time"] == _ts("2026-07-05 06:00"), "temp_f"] = 500.0  # after the climate day
    assert guidance_plus.exact_run_max(SNAP, TARGET, TZ, 3, 2.0, lambda r: frame)[0] == 70.0


def test_exact_run_is_nan_when_nothing_reaches_the_warm_hours():
    empty = pd.DataFrame({"valid_time": pd.to_datetime([], utc=True), "temp_f": []})
    assert np.isnan(guidance_plus.exact_run_max(SNAP, TARGET, TZ, 3, 2.0, lambda r: empty)[0])


def test_previous_runs_from_before_a_model_archive_starts_stay_numeric(monkeypatch):
    def body(*a, **k):
        temps = [None, None] if k.get("use_cache") and a[1]["start_date"] < "2024-03-01" else [60.5, 61.5]
        return {"hourly": {"time": [f"{a[1]['start_date']}T00:00", f"{a[1]['start_date']}T01:00"], "temperature_2m_previous_day1": temps}}

    monkeypatch.setattr(openmeteo, "get_json", body)
    long = openmeteo.hourly_previous_runs(40.0, -73.0, date(2024, 1, 1), date(2024, 6, 1), "ecmwf_ifs025", chunk_days=90)
    assert long["temp_f"].dtype == float
    assert long["valid_time"].min() >= _ts("2024-03-01")


def test_missing_single_run_returns_empty(monkeypatch):
    import requests

    response = requests.Response()
    response.status_code = 400

    def raises(*a, **k):
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(openmeteo, "get_json", raises)
    assert openmeteo.single_run(40.0, -73.0, "ncep_hrrr_conus", SNAP).empty


def test_enrich_exact_leaves_earlier_rows_blank(monkeypatch):
    frame = _run_frame("2026-07-04 00:00", "2026-07-06 00:00", 75.0)
    monkeypatch.setattr(openmeteo, "single_run", lambda *a, **k: frame)
    early = date(2026, 1, 15)
    df = pd.DataFrame(
        {"city": "NY", "target_date": [early, TARGET], "snapshot_ts": [_ts("2026-01-15 17:00"), SNAP], "ticker": ["a", "b"]}
    )
    out = guidance_plus.enrich_exact(df)
    assert early < EXACT_RUN_START
    assert out["hrrr_max"].isna().tolist() == [True, False]
    assert out.loc[1, "hrrr_max"] == 75.0
    assert set(guidance_plus.EXACT_COLUMNS) == {c for m in EXACT_RUN_MODELS for c in (f"{m}_max", f"{m}_age_h")}


def test_every_city_has_an_archive_code():
    assert all(c.icao.startswith("K") and len(c.icao) == 4 for c in CITIES.values())


def _exact_frame(seed=1, n_days=120):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-04-03", periods=n_days).date
    rows = []
    for d in days:
        truth = 70 + rng.normal(0, 6)
        for low, high in [(-np.inf, truth - 2), (truth - 1, truth + 1), (truth + 2, np.inf)]:
            rows.append({"target_date": d, "low": np.round(low), "high": np.round(high), "settled_value": np.round(truth)})
    df = pd.DataFrame(rows)
    df["city"] = "NY"
    df["event_id"] = "NY:" + df["target_date"].astype(str)
    df["snapshot"] = "D-0 08:00"
    df["ticker"] = df.index.astype(str)
    df["y"] = ((df["low"] <= df["settled_value"]) & (df["settled_value"] <= df["high"])).astype(int)
    n = len(df)
    for col, noise in [("hrrr_max", 1.0), ("nbmx_max", 1.5), ("ecmwfx_max", 2.0), ("lamp_max", 1.5), ("nbm_max", 2.0), ("gfs_max", 2.0)]:
        by_day = dict(zip(days, rng.normal(0, noise, len(days))))
        df[col] = df["settled_value"] + df["target_date"].map(by_day)
    df["obs_max_sofar"] = df["settled_value"] - 8
    df["obs_latest"] = df["settled_value"] - 8
    for col in ["hrrr_age_h", "nbmx_age_h", "ecmwfx_age_h"]:
        df[col] = 3.0
    df["A_xgb"] = np.clip(rng.random(n), 0.05, 0.95)
    # Uninformative weather, and a noisy market that does not separate the outcomes perfectly.
    df["X_xgb"] = np.clip(df["y"].mean() + rng.normal(0, 0.03, n), 0.02, 0.98)
    df["CX_xgb"] = df["X_xgb"]
    df["B_market_norm"] = np.clip(0.3 * df["y"] + 0.35 + rng.normal(0, 0.2, n), 0.02, 0.98)
    return df


def test_run_probability_is_fitted_without_the_scored_block():
    df = _exact_frame()
    df["block"] = exact_runs.date_blocks(df["target_date"])
    base = exact_runs.run_probability(df.copy())
    changed = df.copy()
    held_out = changed["block"] == 2
    changed.loc[held_out, "settled_value"] = changed.loc[held_out, "settled_value"] + 40
    after = exact_runs.run_probability(changed)
    assert np.allclose(after[held_out], base[held_out])
    assert not np.allclose(after[~held_out], base[~held_out])


def test_exact_run_test_finds_a_real_gain_and_writes_results(tmp_path):
    df = _exact_frame()
    columns = [c for c in df.columns if c.startswith(("hrrr_", "nbmx_", "ecmwfx_"))]
    oos = df.drop(columns=columns)
    md = exact_runs.run(oos, df[["ticker", "snapshot", *columns]], tmp_path / "x", n_boot=20)
    assert "Exact-run test" in md
    scores = pd.read_csv(tmp_path / "x" / "scores.csv")
    all_rows = scores[scores["city"] == "ALL"].set_index("model")["brier"]
    # X_xgb here is little more than the base rate, so the runs must beat it clearly.
    assert all_rows["F_weather"] < all_rows["X_xgb"] - 0.02
