import numpy as np
import pytest

from weather_pricing import evaluate as ev
from weather_pricing import valuation as val


def test_scores():
    y = np.array([1, 0, 1, 0])
    assert ev.brier(y, [1, 0, 1, 0]) == 0
    assert ev.brier(y, [0.5] * 4) == 0.25
    assert np.isclose(ev.log_loss(y, [0.5] * 4), np.log(2))


def test_ece_zero_when_calibrated():
    rng = np.random.default_rng(0)
    p = np.repeat([0.15, 0.55, 0.85], 20000)
    y = rng.random(p.size) < p
    assert ev.ece(y, p) < 0.01


def test_bootstrap_detects_better_model_and_clusters():
    rng = np.random.default_rng(1)
    n_events, per = 300, 6
    groups = np.repeat(np.arange(n_events), per)
    truth = rng.random(n_events * per)
    y = (rng.random(truth.size) < truth).astype(int)
    good, bad = truth, np.full(truth.size, 0.5)
    r = ev.bootstrap_diff(y, bad, good, groups, n_boot=300)
    assert r["diff"] > 0 and r["ci_lo"] > 0 and r["n_events"] == n_events
    null = ev.bootstrap_diff(y, good, good, groups, n_boot=50)
    assert null["diff"] == 0


def test_encompassing_picks_up_informative_market():
    rng = np.random.default_rng(2)
    n = 6000
    x = rng.normal(size=n)
    m = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(x + m)))).astype(int)
    p_weather = 1 / (1 + np.exp(-x))
    p_market = 1 / (1 + np.exp(-(x + m)))
    res = ev.encompassing_test(y, p_weather, p_market, np.arange(n) // 3)
    assert res.loc["logit_market", "p_value"] < 0.01


def test_plan_example_expected_loss():
    # N=$1,000, L=50%, p=10% -> expected principal repayment $950 before discounting.
    assert np.isclose(val.value(0.10, 0.0, val.Note(1000, 0.5), rate=0.04), 950)
    assert np.isclose(val.sensitivity(0.0), -500)


def test_wang_loading_lowers_value():
    p, t = np.array([0.1, 0.5]), 0.01
    assert np.allclose(val.wang(p, 0.0), p)
    assert (val.value(p, t, lam=0.3) < val.value(p, t, lam=0.0)).all()
    assert np.isclose(val.realised_value(1, 0.0), 500)


def test_horizon_years_adds_the_settlement_delay():
    # The note pays about 8 h after the local day ends: 16 h lead -> 24 h horizon.
    year_hours = 24 * 365.25
    assert val.horizon_years(16.0) == pytest.approx(24 / year_hours)
    hours = np.array([24.0, 42.0])
    assert np.allclose(val.horizon_years(np.array([16.0, 34.0])), hours / year_hours)
