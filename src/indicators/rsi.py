"""Wilder RSI. No look-ahead, no division by zero, no filled warm-up.

Calculation inputs
------------------
Only the values of ``series`` (close prices) in row order. This function never
reads a timestamp, a calendar month, or a future row.

Warm-up
-------
A period-N RSI needs N price *changes*, which requires N + 1 closes. The first
valid RSI is therefore at index ``period`` (0-based). The preceding ``period``
rows are NaN. Those NaNs are never replaced with 50.

    delta_t = close_t - close_(t-1)
    gain_t  = max(delta_t, 0)
    loss_t  = max(-delta_t, 0)

Seed: SMA of the first ``period`` gains and of the first ``period`` losses.
Then Wilder smoothing:

    avg_gain_t = (avg_gain_(t-1) * (period - 1) + gain_t) / period
    avg_loss_t = (avg_loss_(t-1) * (period - 1) + loss_t) / period

RSI from averages, before any division:

    avg_loss == 0 and avg_gain > 0  ->  100
    avg_gain == 0 and avg_loss > 0  ->  0
    avg_gain == 0 and avg_loss == 0 ->  50
    otherwise                       ->  100 - 100 / (1 + avg_gain / avg_loss)

Every finite RSI satisfies 0 <= RSI <= 100.

NaN policy
----------
A NaN close produces a NaN RSI and **resets** Wilder state. The next delta
that needs a NaN previous close is also unusable. After a hole, ``period``
consecutive valid changes are required before a new seed. No forward-fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ema import IndicatorError, _require_period


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    """Map Wilder averages to RSI without producing inf or a warning."""
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """Return Wilder RSI of ``series`` with the given ``period``."""
    period = _require_period(period)
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype="float64", copy=True)
    n = values.shape[0]
    out = np.full(n, np.nan, dtype="float64")

    gains: list[float] = []
    losses: list[float] = []
    avg_gain: float | None = None
    avg_loss: float | None = None
    previous = np.nan

    for index, price in enumerate(values):
        if np.isnan(price):
            out[index] = np.nan
            gains = []
            losses = []
            avg_gain = None
            avg_loss = None
            previous = np.nan
            continue

        if np.isnan(previous):
            out[index] = np.nan
            previous = price
            continue

        delta = float(price) - float(previous)
        previous = price
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0

        if avg_gain is not None and avg_loss is not None:
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            out[index] = _rsi_from_averages(avg_gain, avg_loss)
            continue

        gains.append(gain)
        losses.append(loss)
        if len(gains) == period:
            avg_gain = float(np.mean(gains))
            avg_loss = float(np.mean(losses))
            out[index] = _rsi_from_averages(avg_gain, avg_loss)

    return pd.Series(out, index=series.index, dtype="float64", name=series.name)


# Re-export so callers can catch a single error type.
__all__ = ["calculate_rsi", "IndicatorError"]
