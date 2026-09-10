"""EMA, volume and RSI predicate tests. Thresholds are passed in, never assumed."""

from __future__ import annotations

import numpy as np

from src.strategy.conditions import ema_trend, rsi_filter, volume_confirmation


def test_ema_trend_true_when_fast_strictly_above_slow():
    assert ema_trend(2.0, 1.0) is True


def test_ema_trend_false_when_equal():
    assert ema_trend(1.0, 1.0) is False


def test_ema_trend_false_when_fast_below_slow():
    assert ema_trend(1.0, 2.0) is False


def test_ema_trend_none_when_nan():
    assert ema_trend(np.nan, 1.0) is None
    assert ema_trend(1.0, np.nan) is None


def test_volume_true_when_strictly_above_threshold():
    assert volume_confirmation(16.0, 10.0, 1.5) is True


def test_volume_false_when_equal_to_threshold():
    assert volume_confirmation(15.0, 10.0, 1.5) is False


def test_volume_false_when_below_threshold():
    assert volume_confirmation(14.0, 10.0, 1.5) is False


def test_volume_none_when_sma_is_nan():
    assert volume_confirmation(20.0, np.nan, 1.5) is None


def test_volume_none_when_volume_is_nan():
    assert volume_confirmation(np.nan, 10.0, 1.5) is None


def test_rsi_true_inside_open_interval():
    assert rsi_filter(60.0, 50.0, 70.0) is True


def test_rsi_false_at_lower_bound():
    assert rsi_filter(50.0, 50.0, 70.0) is False


def test_rsi_false_at_upper_bound():
    assert rsi_filter(70.0, 50.0, 70.0) is False


def test_rsi_false_below_min():
    assert rsi_filter(49.0, 50.0, 70.0) is False


def test_rsi_false_above_max():
    assert rsi_filter(71.0, 50.0, 70.0) is False


def test_rsi_none_when_nan():
    assert rsi_filter(np.nan, 50.0, 70.0) is None
