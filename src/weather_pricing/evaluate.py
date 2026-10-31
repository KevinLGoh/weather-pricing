"""Probability scoring, calibration and event-clustered inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p) -> float:
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y})
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), obs_freq=("y", "mean")).reset_index()
    out["bin_lo"] = out["bin"] / bins
    return out


def score_table(y, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"model": k, "brier": brier(y, p), "log_loss": log_loss(y, p), "ece": ece(y, p)} for k, p in preds.items()]
    ).set_index("model")


def _group_index(groups) -> tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(np.asarray(groups))
    return codes, len(uniq)


def bootstrap_diff(y, p_base, p_new, groups, n_boot: int = 2000, seed: int = 0, metric=brier) -> dict:
    """Event-clustered bootstrap of metric(base) - metric(new); positive favours `new`.

    Contracts and snapshots from the same city-day share one outcome, so resampling rows
    would overstate precision; whole events are resampled instead.
    """
    y, p_base, p_new = map(np.asarray, (y, p_base, p_new))
    codes, k = _group_index(groups)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(k + 1))
    members = [order[starts[i] : starts[i + 1]] for i in range(k)]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        rows = np.concatenate([members[i] for i in rng.integers(0, k, k)])
        diffs[b] = metric(y[rows], p_base[rows]) - metric(y[rows], p_new[rows])
    point = metric(y, p_base) - metric(y, p_new)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": point, "ci_lo": lo, "ci_hi": hi, "p_value": min(1.0, p_two), "n_events": k}


def encompassing_test(y, p_weather, p_market, groups) -> pd.DataFrame:
    """Forecast-encompassing regression: logit P(y) = a + b1 logit(p_weather) + b2 logit(p_market).

    b2 != 0 means the market carries information the weather model does not, and vice
    versa for b1. Standard errors are clustered by event.
    """
    def lg(p):
        p = np.clip(np.asarray(p), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    X = np.column_stack([np.ones(len(y)), lg(p_weather), lg(p_market)])
    codes, _ = _group_index(groups)
    beta, se = _logit_cluster(X, np.asarray(y, dtype=float), codes)
    z = beta / se
    return pd.DataFrame(
        {
            "coef": beta,
            "se": se,
            "z": z,
            "p_value": 2 * norm.sf(np.abs(z)),
            "ci_lo": beta - 1.96 * se,
            "ci_hi": beta + 1.96 * se,
        },
        index=["const", "logit_weather", "logit_market"],
    )


def _logit_cluster(X: np.ndarray, y: np.ndarray, codes: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Unpenalised logistic MLE (Newton) with cluster-robust sandwich standard errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    H_inv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    scores = np.zeros((codes.max() + 1, X.shape[1]))
    np.add.at(scores, codes, X * (y - p)[:, None])
    g = len(scores)
    cov = H_inv @ (scores.T @ scores) @ H_inv * g / (g - 1)
    return beta, np.sqrt(np.diag(cov))
