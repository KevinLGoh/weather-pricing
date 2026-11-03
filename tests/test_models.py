"""Synthetic end-to-end check: if quotes carry information the forecast lacks, C beats A."""

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from weather_pricing import evaluate as ev
from weather_pricing.experiment import chronological_split
from weather_pricing.models import GaussianRule, fit_predict, weather_features


def synthetic(n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        target = date(2024, 1, 1) + timedelta(days=d)
        truth = 60 + 15 * np.sin(2 * np.pi * d / 365) + rng.normal(0, 4)
        private = truth + rng.normal(0, 1.5)  # what the market knows beyond the NWP model
        for snap, lead in [("D-1 14:00", 34.0), ("D-0 08:00", 16.0)]:
            fc = truth + rng.normal(0.5, 3.0)
            centre = int(round(fc))
            edges = [centre - 5, centre - 3, centre - 1, centre + 1, centre + 3]
            brackets = [(-np.inf, edges[0] - 1)] + [(a, a + 1) for a in edges[:-1]] + [(edges[-1], np.inf)]
            mids = [norm.cdf((hi + 0.5 - private) / 1.6) - norm.cdf((lo - 0.5 - private) / 1.6) for lo, hi in brackets]
            for (lo, hi), mid in zip(brackets, mids):
                kind = "less" if np.isinf(lo) else "greater" if np.isinf(hi) else "between"
                rows.append(
                    dict(city="NY", event_ticker=f"E{d}", ticker=f"E{d}-{lo}", target_date=target, snapshot=snap,
                         lead_hours=lead, strike_type=kind, low=lo, high=hi, fc_tmax=fc, fc_tmax_prev=fc + rng.normal(0, 1),
                         obs_tmax_prevday=truth + rng.normal(0, 5), mid=mid, mid_norm=mid / sum(mids), spread=0.02,
                         cum_volume=1000.0, stale=False, settled_value=round(truth),
                         y=int(lo <= round(truth) <= hi))
                )
    return pd.DataFrame(rows)


def test_combined_beats_weather_only_when_market_is_informed():
    df = synthetic()
    df["event_id"] = df["city"] + ":" + df["target_date"].astype(str)
    train, test, val_mask = chronological_split(df, 0.3)
    preds = fit_predict(train, test, val_mask)
    y = test["y"].to_numpy()
    assert set(preds) >= {"R_gaussian_rule", "A_logit", "A_xgb", "B_market_norm", "C_logit", "C_xgb"}
    assert ev.brier(y, preds["C_logit"]) < ev.brier(y, preds["A_logit"])
    assert ev.brier(y, preds["C_xgb"]) < ev.brier(y, preds["A_xgb"])
    # The rule baseline should be a sensible probability model, not noise.
    assert ev.brier(y, preds["R_gaussian_rule"]) < ev.brier(y, np.full(len(y), y.mean()))


def test_weather_features_emits_no_deprecation_warning():
    df = synthetic(n_days=10)
    rule = GaussianRule().fit(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", FutureWarning)
        features = weather_features(df, rule)
    # The first row has target_date 2024-01-01, which is day 1 of the year.
    assert features["doy_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 1 / 365.25))
