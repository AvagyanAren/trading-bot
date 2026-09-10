"""EMA tests. Expected values are hand-encoded, not taken from calculate_ema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.ema import IndicatorError, calculate_ema


def test_first_valid_ema_equals_sma_of_period():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    result = calculate_ema(close, 3)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    # SMA(10, 11, 12) = 11
    assert result.iloc[2] == pytest.approx(11.0)


def test_recursive_ema_after_sma_seed():
    # alpha = 2 / (3 + 1) = 0.5
    # EMA[3] = 0.5 * 13 + 0.5 * 11 = 12
    # EMA[4] = 0.5 * 14 + 0.5 * 12 = 13
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    result = calculate_ema(close, 3)

    assert result.iloc[3] == pytest.approx(12.0)
    assert result.iloc[4] == pytest.approx(13.0)


def test_warmup_is_nan_not_zero():
    result = calculate_ema(pd.Series([10.0, 20.0, 30.0, 40.0]), 3)

    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
    assert result.iloc[0] != 0
    assert not np.isfinite(result.iloc[0])


def test_increasing_series_ema_lags_the_last_price():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = calculate_ema(close, 3)

    assert result.iloc[2] == pytest.approx(2.0)  # SMA of 1,2,3
    assert result.dropna().iloc[-1] < close.iloc[-1]


def test_decreasing_series_ema_stays_above_the_last_price():
    close = pd.Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    result = calculate_ema(close, 3)

    assert result.iloc[2] == pytest.approx(5.0)  # SMA of 6,5,4
    assert result.dropna().iloc[-1] > close.iloc[-1]


def test_constant_series_ema_equals_the_constant_after_seed():
    close = pd.Series([7.0, 7.0, 7.0, 7.0, 7.0])
    result = calculate_ema(close, 3)

    assert result.iloc[2:].to_numpy() == pytest.approx(7.0)


def test_period_one_equals_the_input():
    close = pd.Series([4.0, 8.0, 15.0, 16.0, 23.0])
    result = calculate_ema(close, 1)

    pd.testing.assert_series_equal(result, close.astype("float64"), check_names=False)


def test_period_must_be_positive():
    with pytest.raises(IndicatorError, match=">= 1"):
        calculate_ema(pd.Series([1.0, 2.0]), 0)


def test_nan_resets_state_and_is_not_filled():
    # After the hole a new SMA of three consecutive valid prices is required.
    close = pd.Series([10.0, 11.0, 12.0, np.nan, 13.0, 14.0, 15.0])
    result = calculate_ema(close, 3)

    assert result.iloc[2] == pytest.approx(11.0)
    assert pd.isna(result.iloc[3])
    assert pd.isna(result.iloc[4])
    assert pd.isna(result.iloc[5])
    assert result.iloc[6] == pytest.approx(14.0)  # SMA(13, 14, 15)


def test_leading_nans_stay_nan():
    close = pd.Series([np.nan, np.nan, 10.0, 11.0, 12.0])
    result = calculate_ema(close, 3)

    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
    assert result.iloc[4] == pytest.approx(11.0)


def test_ema_does_not_look_ahead():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    baseline = calculate_ema(close, 3)

    mutated = close.copy()
    mutated.iloc[5] = 1_000.0
    after = calculate_ema(mutated, 3)

    pd.testing.assert_series_equal(baseline.iloc[:5], after.iloc[:5])
    assert after.iloc[5] != baseline.iloc[5]


def test_ema_never_uses_a_timestamp_index_as_input():
    """Row order is the series order even if the index is not chronological."""
    close = pd.Series(
        [10.0, 11.0, 12.0, 13.0],
        index=pd.to_datetime(
            ["2024-01-04", "2024-01-03", "2024-01-02", "2024-01-01"]
        ),
    )
    result = calculate_ema(close, 3)

    assert result.iloc[2] == pytest.approx(11.0)
    assert result.iloc[3] == pytest.approx(12.0)
