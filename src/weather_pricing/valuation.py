"""Hypothetical one-period weather-linked note.

Each Kalshi contract defines a trigger (the daily high lands in a bracket). The note pays
principal N plus coupon C at settlement and loses a fraction L of principal if triggered:

    V_t = D(t, T) * (C + N * (1 - q_t * L))

q_t is the pricing probability. With lambda = 0 it is the forecast probability p_t, so
V_t is the discounted expected payoff. lambda > 0 applies a Wang transform,
q = Phi(Phi^-1(p) + lambda), the usual way to load catastrophe risk for the premium
investors demand. Forecast probabilities are not risk-neutral ones, so lambda is kept
explicit rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Note:
    principal: float = 1000.0
    loss_fraction: float = 0.5
    coupon: float = 0.0


def discount(rate: float, years) -> np.ndarray:
    return np.exp(-rate * np.asarray(years, dtype=float))


def wang(p, lam: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return p if lam == 0 else norm.cdf(norm.ppf(p) + lam)


def value(p, years, note: Note = Note(), rate: float = 0.04, lam: float = 0.0) -> np.ndarray:
    q = wang(p, lam)
    return discount(rate, years) * (note.coupon + note.principal * (1 - q * note.loss_fraction))


def realised_value(y, years, note: Note = Note(), rate: float = 0.04) -> np.ndarray:
    """Present value of what the note actually paid."""
    return discount(rate, years) * (note.coupon + note.principal * (1 - np.asarray(y) * note.loss_fraction))


def sensitivity(years, note: Note = Note(), rate: float = 0.04) -> np.ndarray:
    """dV/dp: dollars of value per unit of trigger probability (lambda = 0)."""
    return -discount(rate, years) * note.principal * note.loss_fraction
