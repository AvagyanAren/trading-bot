"""Breakout window tests. Expected values are hand-encoded."""

from __future__ import annotations

import numpy as np
import pytest

from src.strategy.breakout import breakout_level, close_breaks_out
from src.strategy.engine import evaluate_closed_candle, generate_signals
from src.strategy.signals import SignalType

from .conftest import entry_ready_frame


def test_breakout_level_is_max_of_previous_highs():
    assert breakout_level([100.0, 101.0, 105.0, 102.0]) == pytest.approx(105.0)


def test_breakout_true_when_close_strictly_above_previous_highs():
    level = breakout_level([100.0, 101.0, 105.0])
    assert close_breaks_out(106.0, level) is True


def test_breakout_false_when_close_equals_previous_high():
    level = breakout_level([100.0, 101.0, 105.0])
    assert close_breaks_out(105.0, level) is False


def test_breakout_false_when_close_below_previous_high():
    level = breakout_level([100.0, 101.0, 105.0])
    assert close_breaks_out(104.0, level) is False


def test_empty_window_is_nan_and_unevaluable():
    level = breakout_level([])
    assert np.isnan(level)
    assert close_breaks_out(106.0, level) is None


def test_nan_in_window_is_nan_not_skipped():
    level = breakout_level([100.0, np.nan, 105.0])
    assert np.isnan(level)
    assert close_breaks_out(106.0, level) is None


def test_current_candle_high_is_not_included_in_breakout_level(
    tiny_indicator, tiny_strategy
):
    frame = entry_ready_frame(6, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    baseline = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    mutated = frame.copy(deep=True)
    mutated.loc[index, "high"] = 10_000.0
    after = generate_signals(
        mutated, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )

    assert after["breakout_level"].iloc[index] == pytest.approx(
        baseline["breakout_level"].iloc[index]
    )
    assert after["breakout"].iloc[index] == baseline["breakout"].iloc[index]


def test_evaluate_closed_candle_ignores_current_high(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    previous = frame["high"].iloc[index - tiny_strategy.breakout_period : index].to_numpy()
    candle = frame.iloc[index].copy()
    first = evaluate_closed_candle(
        candle, previous, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    candle["high"] = 9_999.0
    second = evaluate_closed_candle(
        candle, previous, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert first.breakout_level == second.breakout_level
    assert first.conditions.breakout is True
    assert second.conditions.breakout is True


def test_insufficient_history_is_unevaluable(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(3, tiny_indicator, tiny_strategy)
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert tiny_strategy.breakout_period == 3
    for row in range(3):
        assert result["breakout"].iloc[row] is None
        assert result["signal"].iloc[row] == SignalType.NONE.value
        assert bool(result["warmup"].iloc[row]) is True


def test_previous_bar_high_is_included_in_the_window(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(6, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    mutated = frame.copy(deep=True)
    mutated.loc[index - 1, "high"] = 10_000.0
    after = generate_signals(
        mutated, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert after["breakout"].iloc[index] == False  # noqa: E712
    assert after["breakout_level"].iloc[index] == pytest.approx(10_000.0)


def test_nan_close_makes_breakout_unevaluable():
    assert close_breaks_out(float("nan"), 100.0) is None
