"""Stronger weather-only family (W) and its combination with market data (CW).

W_rule turns all the public guidance into one predictive distribution for the daily high:
a regression gives the centre, the spread grows with NBM's own uncertainty, and the
distribution is cut off below what the thermometer has already read that day. W_logit and
W_xgb recalibrate it with extra features, and CW_* add the market features, mirroring how
the A and C families are compared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, Ridge

from .models import _fit_xgb, _logit_model, clip, market_features, weather_features, GaussianRule

FORECASTS = ["fc_tmax", "gfs_max", "nbm_max", "lamp_max"]
# The X family adds these. The extra models start in March 2024, so earlier rows fill them
# from the consensus of the forecasts that do exist.
EXTENDED_FORECASTS = [*FORECASTS, "nam_max", "om_ecmwf_max", "om_icon_max", "om_gem_max"]
OBS = ["obs_max_sofar", "obs_latest"]
MIN_SIGMA = 0.75
MIN_GROUP_DAYS = 30


def bracket_probability(low, high, mu, sigma, floor) -> np.ndarray:
    """P(low <= T <= high | T >= floor) for T ~ N(mu, sigma) settled in whole degrees.

    `floor` is the rounded high already observed that day, or NaN if there is none yet.
    """
    low, high, mu, sigma = map(lambda a: np.asarray(a, dtype=float), (low, high, mu, sigma))
    floor = np.asarray(floor, dtype=float)
    f = np.where(np.isnan(floor), -np.inf, (floor - 0.5 - mu) / sigma)
    a = np.maximum((low - 0.5 - mu) / sigma, f)
    b = (high + 0.5 - mu) / sigma
    return np.maximum(norm.cdf(b) - norm.cdf(a), 0.0) / np.maximum(1 - norm.cdf(f), 1e-9)


def _design(frame: pd.DataFrame, columns: list[str], forecasts: list[str]) -> np.ndarray:
    """Predictor matrix, with a missing value replaced by the row's forecast consensus."""
    consensus = frame[forecasts].mean(axis=1)
    return frame[columns].apply(lambda c: c.fillna(consensus)).to_numpy(dtype=float)


class NowcastRule:
    """Predictive distribution of the daily high, fitted per city and snapshot."""

    def __init__(self, forecasts: list[str] = FORECASTS):
        self.forecasts = forecasts

    def fit(self, df: pd.DataFrame) -> NowcastRule:
        ev = df.drop_duplicates(["event_ticker", "snapshot"])
        self.spread_median_ = float(ev["nbm_spread"].median()) if ev["nbm_spread"].notna().any() else 3.0
        self.params_: dict = {}
        for key, g in ev.groupby(["city", "snapshot"]):
            if len(g) >= MIN_GROUP_DAYS:
                self.params_[key] = self._fit_group(g)
        self.global_ = self._fit_group(ev)
        return self

    def _fit_group(self, g: pd.DataFrame) -> dict:
        columns = [c for c in [*self.forecasts, *OBS] if g[c].notna().any()]
        X, y = _design(g, columns, self.forecasts), g["settled_value"].to_numpy(dtype=float)
        centre = Ridge(alpha=1.0).fit(X, y)
        abs_err = np.abs(y - centre.predict(X))
        spread = g["nbm_spread"].fillna(self.spread_median_).to_numpy(dtype=float).reshape(-1, 1)
        return {"columns": columns, "centre": centre, "scale": LinearRegression().fit(spread, abs_err)}

    def mu_sigma(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = np.empty(len(df)), np.empty(len(df))
        pos = pd.Series(np.arange(len(df)), index=df.index)
        for key, g in df.groupby(["city", "snapshot"]):
            params = self.params_.get(key, self.global_)
            idx = pos.loc[g.index].to_numpy()
            mu[idx] = params["centre"].predict(_design(g, params["columns"], self.forecasts))
            spread = g["nbm_spread"].fillna(self.spread_median_).to_numpy(dtype=float).reshape(-1, 1)
            # For a normal error, the standard deviation is mean |error| * sqrt(pi / 2).
            sigma[idx] = np.maximum(params["scale"].predict(spread) * np.sqrt(np.pi / 2), MIN_SIGMA)
        return mu, sigma

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        mu, sigma = self.mu_sigma(df)
        return clip(bracket_probability(df["low"], df["high"], mu, sigma, df["obs_max_sofar"].round()))


def weather_plus_features(df: pd.DataFrame, rule: NowcastRule, base_rule: GaussianRule) -> pd.DataFrame:
    mu, sigma = rule.mu_sigma(df)
    low, high = df["low"].to_numpy(dtype=float), df["high"].to_numpy(dtype=float)
    floor = df["obs_max_sofar"].round().to_numpy(dtype=float)
    base = weather_features(df, base_rule)
    return pd.DataFrame(
        {
            "logit_w": logit(rule.predict(df)),
            "w_z_low": np.clip((low - 0.5 - mu) / sigma, -6, 6),
            "w_z_high": np.clip((high + 0.5 - mu) / sigma, -6, 6),
            "w_sigma": sigma,
            "forecast_disagreement": df[rule.forecasts].std(axis=1).fillna(0).to_numpy(),
            "nbm_spread": df["nbm_spread"].fillna(rule.spread_median_).to_numpy(),
            # How far the day's high so far sits from each end of the bracket.
            "floor_above_low": np.nan_to_num(np.clip(floor - low, -20, 20), nan=-20.0),
            "high_above_floor": np.nan_to_num(np.clip(high - floor, -20, 20), nan=20.0),
            "has_obs": (~np.isnan(floor)).astype(float),
            "obs_minus_nbm": df["obs_minus_nbm"].fillna(0).to_numpy(),
            "is_tail": base["is_tail"],
            "lead_hours": base["lead_hours"],
            "fc_revision": base["fc_revision"],
            "persist_anom": base["persist_anom"],
            "doy_sin": base["doy_sin"],
            "doy_cos": base["doy_cos"],
        },
        index=df.index,
    ).astype(float)


def _feature_sets(df: pd.DataFrame, rule: NowcastRule, base_rule: GaussianRule, weather: str, combined: str) -> dict[str, pd.DataFrame]:
    w = weather_plus_features(df, rule, base_rule)
    return {weather: w, combined: pd.concat([w, market_features(df)], axis=1)}


def fit_models(
    train: pd.DataFrame, val_mask: np.ndarray, forecasts: list[str] = FORECASTS, weather: str = "W", combined: str = "CW"
) -> dict:
    y = train["y"].to_numpy()
    rule, base_rule = NowcastRule(forecasts).fit(train), GaussianRule().fit(train)
    X = _feature_sets(train, rule, base_rule, weather, combined)
    models: dict = {f"{weather}_rule": rule, "_base_rule": base_rule, "_names": (weather, combined)}
    for family in [weather, combined]:
        models[f"{family}_logit"] = _logit_model().fit(X[family], y)
        models[f"{family}_xgb"] = _fit_xgb(X[family], y, val_mask)
    return models


def predict_models(models: dict, test: pd.DataFrame) -> dict[str, np.ndarray]:
    weather, combined = models["_names"]
    rule = models[f"{weather}_rule"]
    X = _feature_sets(test, rule, models["_base_rule"], weather, combined)
    preds = {f"{weather}_rule": rule.predict(test)}
    for family in [weather, combined]:
        for kind in ["logit", "xgb"]:
            preds[f"{family}_{kind}"] = clip(models[f"{family}_{kind}"].predict_proba(X[family])[:, 1])
    return preds
