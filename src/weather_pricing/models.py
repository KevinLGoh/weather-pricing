"""Model families for P(contract pays Yes).

A  weather-only:  NWP forecast + recent observations (no market data)
B  market-only:   Kalshi quotes, raw or recalibrated
C  combined:      A's features + B's features

Each family has an interpretable logistic variant and a boosted-tree variant so A-vs-C
comparisons hold the learner fixed. The rule-based baseline turns the forecast into a
bracket probability under a Gaussian error model fitted on the training period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

EPS = 0.005


def clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def lead_bucket(lead_hours: pd.Series) -> pd.Series:
    return pd.cut(lead_hours, [0, 14, 20, 30, 100], labels=["<=14h", "14-20h", "20-30h", ">30h"]).astype(str)


class GaussianRule:
    """Rule-based baseline: T ~ N(forecast + bias, sigma), bias/sigma per city x lead bucket."""

    def fit(self, df: pd.DataFrame) -> GaussianRule:
        ev = df.drop_duplicates(["event_ticker", "snapshot"]).assign(
            err=lambda d: d["settled_value"] - d["fc_tmax"], bucket=lambda d: lead_bucket(d["lead_hours"])
        )
        self.params_ = ev.groupby(["city", "bucket"])["err"].agg(["mean", "std"])
        self.global_ = (ev["err"].mean(), ev["err"].std())
        return self

    def mu_sigma(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        keys = pd.MultiIndex.from_arrays([df["city"], lead_bucket(df["lead_hours"])])
        p = self.params_.reindex(keys)
        bias = p["mean"].fillna(self.global_[0]).to_numpy()
        sigma = p["std"].fillna(self.global_[1]).clip(lower=0.75).to_numpy()
        return df["fc_tmax"].to_numpy() + bias, sigma

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        mu, sigma = self.mu_sigma(df)
        # Whole-degree settlement: [low, high] covers the continuous range [low-0.5, high+0.5).
        hi = norm.cdf((df["high"].to_numpy() + 0.5 - mu) / sigma)
        lo = norm.cdf((df["low"].to_numpy() - 0.5 - mu) / sigma)
        return clip(hi - lo)


def weather_features(df: pd.DataFrame, rule: GaussianRule) -> pd.DataFrame:
    mu, sigma = rule.mu_sigma(df)
    doy = pd.to_datetime(df["target_date"]).dt.day_of_year.to_numpy()
    return pd.DataFrame(
        {
            "logit_gauss": logit(rule.predict(df)),
            "z_low": np.clip((df["low"].to_numpy() - 0.5 - mu) / sigma, -6, 6),
            "z_high": np.clip((df["high"].to_numpy() + 0.5 - mu) / sigma, -6, 6),
            "is_tail": np.isinf(df["low"].to_numpy()) | np.isinf(df["high"].to_numpy()),
            "lead_hours": df["lead_hours"].to_numpy(),
            "fc_revision": (df["fc_tmax"] - df["fc_tmax_prev"]).fillna(0).to_numpy(),
            "persist_anom": (df["obs_tmax_prevday"] - df["fc_tmax"]).fillna(0).to_numpy(),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=df.index,
    ).astype(float)


def market_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "logit_mid": logit(clip(df["mid"])),
            "logit_mid_norm": logit(clip(df["mid_norm"])),
            "spread": df["spread"].fillna(1.0).to_numpy(),
            "log_volume": np.log1p(df["cum_volume"].to_numpy()),
            "stale": df["stale"].astype(float).to_numpy(),
        },
        index=df.index,
    ).astype(float)


def _logit_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def _xgb_model(n_estimators: int = 400):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        eval_metric="logloss",
        n_jobs=4,
    )


def _fit_xgb(X: pd.DataFrame, y: np.ndarray, val_mask: np.ndarray):
    """Pick the number of trees on the last slice of the training period, then refit."""
    probe = _xgb_model(2000)
    probe.set_params(early_stopping_rounds=100)
    probe.fit(X[~val_mask], y[~val_mask], eval_set=[(X[val_mask], y[val_mask])], verbose=False)
    model = _xgb_model(max(50, probe.best_iteration + 1))
    model.fit(X, y)
    return model


def _feature_sets(df: pd.DataFrame, rule: GaussianRule) -> dict[str, pd.DataFrame]:
    """Inputs of each family: A weather-only, B market-only, C both."""
    wf, mf = weather_features(df, rule), market_features(df)
    return {"A": wf, "B": mf, "C": pd.concat([wf, mf], axis=1)}


def fit_models(train: pd.DataFrame, val_mask: np.ndarray) -> dict:
    """Fit the rule baseline and every learned model on `train`."""
    y = train["y"].to_numpy()
    rule = GaussianRule().fit(train)
    X = _feature_sets(train, rule)
    models: dict = {"R_gaussian_rule": rule}
    for name in ["A_logit", "B_logit", "C_logit"]:
        models[name] = _logit_model().fit(X[name[0]], y)
    for name in ["A_xgb", "C_xgb"]:
        models[name] = _fit_xgb(X[name[0]], y, val_mask)
    return models


def predict_models(models: dict, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Test-set probabilities of every model, keyed by model name."""
    rule = models["R_gaussian_rule"]
    X = _feature_sets(test, rule)
    preds: dict[str, np.ndarray] = {
        "R_gaussian_rule": rule.predict(test),
        "B_market_mid": clip(test["mid"]),
        "B_market_norm": clip(test["mid_norm"]),
    }
    for name in ["A_logit", "B_logit", "C_logit", "A_xgb", "C_xgb"]:
        preds[name] = clip(models[name].predict_proba(X[name[0]])[:, 1])
    return preds


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, val_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Fit every model on `train` and return test-set probabilities keyed by model name."""
    return predict_models(fit_models(train, val_mask), test)
