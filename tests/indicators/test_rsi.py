"""Wilder RSI tests. Expected values are hand-encoded, not taken from calculate_rsi."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.ema import IndicatorError
from src.indicators.rsi import calculate_rsi


def test_first_valid_rsi_is_at_index_equal_to_period():
    # period changes need period + 1 closes; first RSI at index `period`.
    close = pd.Series([10.0, 12.0, 11.0, 13.0, 12.5])
    result = calculate_rsi(close, 2)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert np.isfinite(result.iloc[2])


def test_hand_encoded_wilder_values():
    # close: 10, 12, 11, 13
    # deltas:     +2, -1, +2
    # gains:       2,  0,  2
    # losses:      0,  1,  0
    # seed at index 2: avg_gain = (2+0)/2 = 1, avg_loss = (0+1)/2 = 0.5
    # RS = 2, RSI = 100 - 100/3 = 66.666...
    # index 3: avg_gain = (1 + 2)/2 = 1.5, avg_loss = (0.5 + 0)/2 = 0.25
    # RS = 6, RSI = 100 - 100/7 = 85.714...
    close = pd.Series([10.0, 12.0, 11.0, 13.0])
    result = calculate_rsi(close, 2)

    assert result.iloc[2] == pytest.approx(100.0 - 100.0 / 3.0)
    assert result.iloc[3] == pytest.approx(100.0 - 100.0 / 7.0)


def test_warmup_nans_are_not_replaced_with_fifty():
    close = pd.Series([10.0] * 5)
    result = calculate_rsi(close, 3)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])
    assert result.iloc[3] == pytest.approx(50.0)


def test_increasing_prices_give_rsi_one_hundred():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = calculate_rsi(close, 3)

    assert (result.dropna() == 100.0).all()


def test_decreasing_prices_give_rsi_zero():
    close = pd.Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    result = calculate_rsi(close, 3)

    assert (result.dropna() == 0.0).all()


def test_flat_prices_give_rsi_fifty_after_warmup():
    close = pd.Series([42.0] * 8)
    result = calculate_rsi(close, 4)

    assert result.iloc[:4].isna().all()
    assert (result.iloc[4:] == 50.0).all()


def test_avg_loss_zero_is_one_hundred():
    close = pd.Series([10.0, 11.0, 12.0, 13.0])
    result = calculate_rsi(close, 2)

    assert result.iloc[2] == pytest.approx(100.0)


def test_avg_gain_zero_is_zero():
    close = pd.Series([13.0, 12.0, 11.0, 10.0])
    result = calculate_rsi(close, 2)

    assert result.iloc[2] == pytest.approx(0.0)


def test_both_averages_zero_is_fifty():
    close = pd.Series([10.0, 10.0, 10.0, 10.0])
    result = calculate_rsi(close, 2)

    assert result.iloc[2] == pytest.approx(50.0)


def test_rsi_stays_within_zero_and_one_hundred():
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 3, size=80)
    close = pd.Series(100.0 + np.cumsum(steps))
    result = calculate_rsi(close, 14)

    finite = result.dropna()
    assert (finite >= 0.0).all()
    assert (finite <= 100.0).all()
    assert np.isfinite(finite).all()


def test_rsi_does_not_look_ahead():
    close = pd.Series([10.0, 12.0, 11.0, 13.0, 12.0, 14.0])
    baseline = calculate_rsi(close, 2)

    mutated = close.copy()
    mutated.iloc[5] = 0.0
    after = calculate_rsi(mutated, 2)

    pd.testing.assert_series_equal(baseline.iloc[:5], after.iloc[:5])
    assert after.iloc[5] != baseline.iloc[5]


def test_nan_resets_wilder_state():
    close = pd.Series([10.0, 12.0, 11.0, np.nan, 10.0, 12.0, 11.0])
    result = calculate_rsi(close, 2)

    assert result.iloc[2] == pytest.approx(100.0 - 100.0 / 3.0)
    assert pd.isna(result.iloc[3])
    # After the hole we need two new changes: indices 5 (10->12) and 6 (12->11).
    assert pd.isna(result.iloc[4])
    assert pd.isna(result.iloc[5])
    assert result.iloc[6] == pytest.approx(100.0 - 100.0 / 3.0)


def test_period_must_be_positive():
    with pytest.raises(IndicatorError, match=">= 1"):
        calculate_rsi(pd.Series([1.0, 2.0, 3.0]), 0)
