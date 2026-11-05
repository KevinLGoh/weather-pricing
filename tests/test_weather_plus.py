"""Stronger weather model: the floor logic, and that it finds a real edge on synthetic data."""

import numpy as np
import pandas as pd
from test_models import synthetic

from weather_pricing import evaluate as ev
from weather_pricing import weather_experiment, weather_plus
from weather_pricing.experiment import chronological_split
from weather_pricing.guidance import FEATURES


def test_probabilities_without_a_floor_sum_to_one_across_a_partition():
    lows = [-np.inf, 70, 72, 74]
    highs = [69, 71, 73, np.inf]
    p = weather_plus.bracket_probability(lows, highs, [72] * 4, [3] * 4, [np.nan] * 4)
    assert np.isclose(p.sum(), 1.0)


def test_floor_removes_brackets_already_passed_and_renormalises():
    lows = [-np.inf, 70, 72, 74]
    highs = [69, 71, 73, np.inf]
    p = weather_plus.bracket_probability(lows, highs, [72] * 4, [3] * 4, [72] * 4)
    assert p[0] == 0 and p[1] == 0
    assert np.isclose(p.sum(), 1.0)
    # The floor raises what is left compared with having no floor.
    assert p[2] > weather_plus.bracket_probability([72], [73], [72], [3], [np.nan])[0]


def _with_guidance(df, seed=3):
    rng = np.random.default_rng(seed)
    events = df[["event_ticker", "snapshot", "settled_value"]].drop_duplicates(["event_ticker", "snapshot"]).copy()
    truth = events["settled_value"].to_numpy(dtype=float)
    n = len(events)
    # Guidance is sharper than the fc_tmax of `synthetic`, which has noise of 3 degrees.
    events["nbm_max"] = truth + rng.normal(0, 1.0, n)
    events["gfs_max"] = truth + rng.normal(0, 2.0, n)
    events["nbm_spread"] = 2.0
    today = events["snapshot"].str.startswith("D-0").to_numpy()
    events["lamp_max"] = np.where(today, truth + rng.normal(0, 0.7, n), np.nan)
    events["obs_max_sofar"] = np.where(today, truth - rng.uniform(4, 9, n), np.nan)
    events["obs_latest"] = events["obs_max_sofar"]
    events["obs_minus_nbm"] = np.where(today, rng.normal(0, 1, n), np.nan)
    for col in FEATURES:
        if col not in events:
            events[col] = 1.0
    return df.merge(events.drop(columns="settled_value"), on=["event_ticker", "snapshot"])


def test_stronger_weather_beats_the_weak_baseline_on_synthetic_data():
    df = _with_guidance(synthetic())
    df["event_id"] = df["city"] + ":" + df["target_date"].astype(str)
    train, test, val_mask = chronological_split(df, 0.3)
    models = weather_plus.fit_models(train, val_mask)
    preds = weather_plus.predict_models(models, test)
    y = test["y"].to_numpy()
    assert set(preds) == {"W_rule", "W_logit", "W_xgb", "CW_logit", "CW_xgb"}
    from weather_pricing.models import fit_predict

    base = fit_predict(train, test, val_mask)
    assert ev.brier(y, preds["W_rule"]) < ev.brier(y, base["A_logit"])
    assert ev.brier(y, preds["W_xgb"]) < ev.brier(y, base["A_xgb"])
    for p in preds.values():
        assert np.isfinite(p).all() and (p > 0).all() and (p < 1).all()


def test_rows_keep_their_alignment_when_the_index_is_not_a_range():
    df = _with_guidance(synthetic(120))
    train, test, _ = chronological_split(df, 0.3)
    rule = weather_plus.NowcastRule().fit(train)
    shuffled = test.sample(frac=1.0, random_state=0)
    a = pd.Series(rule.predict(test), index=test.index)
    b = pd.Series(rule.predict(shuffled), index=shuffled.index)
    assert np.allclose(a, b.loc[a.index])


def test_missing_guidance_falls_back_to_the_other_forecasts():
    df = _with_guidance(synthetic(120))
    train, test, _ = chronological_split(df, 0.3)
    rule = weather_plus.NowcastRule().fit(train)
    blank = test.assign(gfs_max=np.nan, nbm_max=np.nan, lamp_max=np.nan, nbm_spread=np.nan)
    p = rule.predict(blank)
    assert np.isfinite(p).all()


def test_experiment_writes_its_own_results(tmp_path):
    df = _with_guidance(synthetic(160))
    df["noaa_tmax"] = df["settled_value"]
    df["settle_source"] = "NWS"
    md = weather_experiment.run(df, tmp_path / "w", n_boot=20)
    assert "Stronger weather baseline" in md
    assert "Cities: NY." in md
    assert "Same rows" not in md  # it must hold for any dataset, not just the main one
    for name in ["scores.csv", "bootstrap.csv", "encompassing.csv", "by_snapshot.csv", "forecast_accuracy.csv", "summary.md"]:
        assert (tmp_path / "w" / name).exists()
