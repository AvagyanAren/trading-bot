"""Volume SMA tests. Expected values are hand-encoded."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.ema import IndicatorError
from src.indicators.volume import calculate_volume_sma


def test_first_valid_sma_is_mean_of_the_opening_window():
    volume = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = calculate_volume_sma(volume, 3)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(20.0)  # (10+20+30)/3


def test_rolling_windows_are_hand_calculated():
    volume = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = calculate_volume_sma(volume, 3)

    assert result.iloc[3] == pytest.approx(30.0)  # (20+30+40)/3
    assert result.iloc[4] == pytest.approx(40.0)  # (30+40+50)/3


def test_constant_volume_sma_equals_the_constant_after_warmup():
    volume = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
    result = calculate_volume_sma(volume, 3)

    assert result.iloc[2:].to_numpy() == pytest.approx(5.0)


def test_warmup_is_nan_not_zero():
    result = calculate_volume_sma(pd.Series([1.0, 2.0, 3.0]), 3)

    assert pd.isna(result.iloc[0]) and pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)


def test_window_containing_nan_is_nan():
    volume = pd.Series([10.0, 20.0, np.nan, 40.0, 50.0, 60.0])
    result = calculate_volume_sma(volume, 3)

    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])
    assert pd.isna(result.iloc[4])
    assert result.iloc[5] == pytest.approx(50.0)  # (40+50+60)/3


def test_volume_sma_does_not_look_ahead():
    volume = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    baseline = calculate_volume_sma(volume, 3)

    mutated = volume.copy()
    mutated.iloc[4] = 999.0
    after = calculate_volume_sma(mutated, 3)

    pd.testing.assert_series_equal(baseline.iloc[:4], after.iloc[:4])
    assert after.iloc[4] != baseline.iloc[4]


def test_period_must_be_positive():
    with pytest.raises(IndicatorError, match=">= 1"):
        calculate_volume_sma(pd.Series([1.0, 2.0]), 0)
