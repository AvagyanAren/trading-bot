"""Strategy engine tests: AND logic, warm-up, timing, look-ahead, continuity."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.strategy.engine import (
    StrategyConfigError,
    StrategyInputError,
    evaluate_closed_candle,
    generate_signals,
    load_strategy_config,
)
from src.strategy.signals import Signal, SignalType

from .conftest import (
    enriched_frame,
    entry_ready_frame,
    indicator_config,
    strategy_config,
)

FORBIDDEN_FIELDS = ("execution_price", "fill_price", "entry_price")


def test_all_four_conditions_true_emits_long_entry(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    index = tiny_strategy.breakout_period
    assert result["signal"].iloc[index] == SignalType.LONG_ENTRY.value
    assert result["ema_trend"].iloc[index] == True  # noqa: E712
    assert result["breakout"].iloc[index] == True
    assert result["volume_confirmation"].iloc[index] == True
    assert result["rsi_filter"].iloc[index] == True
    assert bool(result["warmup"].iloc[index]) is False


def test_false_ema_trend_blocks_entry(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    frame.loc[index, tiny_indicator.ema_fast_column] = 1.0
    frame.loc[index, tiny_indicator.ema_slow_column] = 2.0
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert result["ema_trend"].iloc[index] == False  # noqa: E712
    assert result["signal"].iloc[index] == SignalType.NONE.value


def test_equal_ema_blocks_entry(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    frame.loc[index, tiny_indicator.ema_fast_column] = 1.0
    frame.loc[index, tiny_indicator.ema_slow_column] = 1.0
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert result["ema_trend"].iloc[index] == False  # noqa: E712
    assert result["signal"].iloc[index] == SignalType.NONE.value


def test_false_breakout_blocks_entry(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    frame.loc[index, "close"] = 100.0
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert result["breakout"].iloc[index] == False  # noqa: E712
    assert result["signal"].iloc[index] == SignalType.NONE.value


def test_false_volume_blocks_entry(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    frame.loc[index, "volume"] = 15.0
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert result["volume_confirmation"].iloc[index] == False  # noqa: E712
    assert result["signal"].iloc[index] == SignalType.NONE.value


def test_false_rsi_blocks_entry(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    frame.loc[index, tiny_indicator.rsi_column] = 70.0
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert result["rsi_filter"].iloc[index] == False  # noqa: E712
    assert result["signal"].iloc[index] == SignalType.NONE.value


def test_two_false_conditions_still_no_entry(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    frame.loc[index, tiny_indicator.ema_fast_column] = 0.5
    frame.loc[index, tiny_indicator.rsi_column] = 40.0
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert result["signal"].iloc[index] == SignalType.NONE.value


def test_consecutive_long_entry_is_allowed(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(8, tiny_indicator, tiny_strategy)
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    entries = result["signal"] == SignalType.LONG_ENTRY.value
    assert int(entries.sum()) == 8 - tiny_strategy.breakout_period


def test_warmup_nan_indicator_blocks_entry(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    frame.loc[index, tiny_indicator.ema_slow_column] = np.nan
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert result["ema_trend"].iloc[index] is None
    assert bool(result["warmup"].iloc[index]) is True
    assert result["signal"].iloc[index] == SignalType.NONE.value


def test_slow_ema_nan_is_the_binding_warmup_when_lookback_is_shorter(
    tmp_path,
):
    indicator = indicator_config(tmp_path, ema_fast=3, ema_slow=5, volume_ma_period=4)
    strategy = strategy_config(tmp_path, breakout_period=3)
    n = 8
    frame = entry_ready_frame(n, indicator, strategy)
    ema_slow = np.full(n, 1.0)
    ema_slow[:5] = np.nan
    frame[indicator.ema_slow_column] = ema_slow
    result = generate_signals(
        frame, strategy, indicator, interval="5m", symbol="BTCUSDT"
    )
    assert all(result["signal"].iloc[:5] == SignalType.NONE.value)
    assert result["signal"].iloc[5] == SignalType.LONG_ENTRY.value


def test_signal_timestamp_is_candle_open_not_a_fill(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    index = tiny_strategy.breakout_period
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert result["timestamp"].iloc[index] == frame["timestamp"].iloc[index]
    expected_exec = frame["timestamp"].iloc[index] + np.timedelta64(5, "m")
    assert result["earliest_execution_time"].iloc[index] == expected_exec
    assert result["earliest_execution_time"].iloc[index] > result["timestamp"].iloc[index]
    assert result["reference_close"].iloc[index] == pytest.approx(frame["close"].iloc[index])


def test_signal_schema_has_no_execution_fields():
    names = {item.name for item in fields(Signal)}
    for forbidden in FORBIDDEN_FIELDS:
        assert forbidden not in names
    assert "SHORT" not in SignalType.__members__
    assert set(SignalType) == {SignalType.NONE, SignalType.LONG_ENTRY}


def test_signal_frame_has_no_execution_columns(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    for forbidden in FORBIDDEN_FIELDS:
        assert forbidden not in result.columns
    assert "leverage" not in result.columns


def test_final_historical_candle_may_signal_without_assuming_a_fill(
    tiny_indicator, tiny_strategy
):
    frame = entry_ready_frame(6, tiny_indicator, tiny_strategy)
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    last = result.iloc[-1]
    assert last["signal"] == SignalType.LONG_ENTRY.value
    expected_exec = frame["timestamp"].iloc[-1] + np.timedelta64(5, "m")
    assert last["earliest_execution_time"] == expected_exec
    assert expected_exec not in set(frame["timestamp"])
    assert len(result) == len(frame)
    for forbidden in FORBIDDEN_FIELDS:
        assert forbidden not in result.columns
    assert last["timestamp"] == frame["timestamp"].iloc[-1]


def test_evaluate_final_candle_execution_time_is_a_label(
    tiny_indicator, tiny_strategy
):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    index = len(frame) - 1
    previous = frame["high"].to_numpy()[index - tiny_strategy.breakout_period : index]
    signal = evaluate_closed_candle(
        frame.iloc[index],
        previous,
        tiny_strategy,
        tiny_indicator,
        interval="5m",
        symbol="BTCUSDT",
    )
    assert signal.signal is SignalType.LONG_ENTRY
    assert signal.earliest_execution_time not in set(frame["timestamp"])
    assert not hasattr(signal, "execution_price")
    assert not hasattr(signal, "fill_price")
    assert not hasattr(signal, "entry_price")


def test_input_frame_is_not_mutated(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(6, tiny_indicator, tiny_strategy)
    original = frame.copy(deep=True)
    generate_signals(frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT")
    pd.testing.assert_frame_equal(frame, original)


def test_generate_signals_is_deterministic(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(6, tiny_indicator, tiny_strategy)
    first = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    second = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    pd.testing.assert_frame_equal(first, second)


def test_future_row_mutation_does_not_change_past_signals(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(8, tiny_indicator, tiny_strategy)
    baseline = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    k = 5
    mutated = frame.copy(deep=True)
    mutated.loc[k + 1, "high"] = 50_000.0
    mutated.loc[k + 1, "low"] = 1.0
    mutated.loc[k + 1, "close"] = 50_000.0
    mutated.loc[k + 1, "volume"] = 1_000_000.0
    mutated.loc[k + 1, tiny_indicator.ema_fast_column] = 9_999.0
    mutated.loc[k + 1, tiny_indicator.ema_slow_column] = 0.1
    mutated.loc[k + 1, tiny_indicator.rsi_column] = 99.0
    mutated.loc[k + 1, tiny_indicator.volume_ma_column] = 0.1
    after = generate_signals(
        mutated, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    pd.testing.assert_frame_equal(baseline.iloc[: k + 1], after.iloc[: k + 1])


def test_mutating_current_high_does_not_change_that_rows_breakout(
    tiny_indicator, tiny_strategy
):
    frame = entry_ready_frame(8, tiny_indicator, tiny_strategy)
    k = 5
    baseline = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    mutated = frame.copy(deep=True)
    mutated.loc[k, "high"] = 80_000.0
    after = generate_signals(
        mutated, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert after["breakout_level"].iloc[k] == pytest.approx(baseline["breakout_level"].iloc[k])
    assert after["breakout"].iloc[k] == baseline["breakout"].iloc[k]
    assert after["breakout_level"].iloc[k + 1] != baseline["breakout_level"].iloc[k + 1]


def test_february_uses_january_breakout_history(tmp_path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, breakout_period=3)
    january = entry_ready_frame(10, indicator, strategy, start="2024-01-01 00:00:00")
    february = entry_ready_frame(5, indicator, strategy, start="2024-02-01 00:00:00")
    february["close"] = 101.0
    full = pd.concat([january, february], ignore_index=True)
    full_signals = generate_signals(
        full, strategy, indicator, interval="5m", symbol="BTCUSDT"
    )
    february_only = generate_signals(
        february.reset_index(drop=True),
        strategy,
        indicator,
        interval="5m",
        symbol="BTCUSDT",
    )
    full_feb_level = full_signals["breakout_level"].iloc[len(january)]
    only_feb_level = february_only["breakout_level"].iloc[0]
    assert not pd.isna(full_feb_level)
    assert pd.isna(only_feb_level)
    assert full_signals["signal"].iloc[len(january)] == SignalType.LONG_ENTRY.value
    assert february_only["signal"].iloc[0] == SignalType.NONE.value


def test_breakout_period_is_independent_of_volume_ma_period(tmp_path):
    indicator = indicator_config(tmp_path, volume_ma_period=5)
    strategy = strategy_config(tmp_path, breakout_period=3)
    assert strategy.breakout_period != indicator.volume_ma_period
    frame = entry_ready_frame(4, indicator, strategy)
    assert indicator.volume_ma_column == "volume_ma5"
    result = generate_signals(
        frame, strategy, indicator, interval="5m", symbol="BTCUSDT"
    )
    assert result["breakout"].iloc[3] == True  # noqa: E712
    assert result["signal"].iloc[3] == SignalType.LONG_ENTRY.value
    assert result["breakout_level"].iloc[3] == pytest.approx(100.0)


def test_changing_breakout_period_changes_the_window(tmp_path, tiny_indicator):
    short = strategy_config(tmp_path, breakout_period=2)
    long = strategy_config(tmp_path, breakout_period=4)
    highs = [10.0, 50.0, 10.0, 10.0, 11.0]
    close = [10.0, 10.0, 10.0, 10.0, 11.5]
    frame = enriched_frame(
        5,
        tiny_indicator,
        close=close,
        high=highs,
        volume=[20.0] * 5,
        ema_fast=[2.0] * 5,
        ema_slow=[1.0] * 5,
        rsi=[60.0] * 5,
        volume_ma=[10.0] * 5,
    )
    short_result = generate_signals(
        frame, short, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    long_result = generate_signals(
        frame, long, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert short_result["breakout"].iloc[4] == True  # noqa: E712
    assert long_result["breakout"].iloc[4] == False
    assert long_result["breakout_level"].iloc[4] == pytest.approx(50.0)


def test_changing_volume_multiplier_changes_confirmation(tmp_path, tiny_indicator):
    loose = strategy_config(tmp_path, volume_multiplier=1.0)
    tight = strategy_config(tmp_path, volume_multiplier=3.0)
    frame = entry_ready_frame(5, tiny_indicator, loose)
    index = loose.breakout_period
    loose_result = generate_signals(
        frame, loose, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    tight_result = generate_signals(
        frame, tight, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert loose_result["signal"].iloc[index] == SignalType.LONG_ENTRY.value
    assert tight_result["volume_confirmation"].iloc[index] == False  # noqa: E712
    assert tight_result["signal"].iloc[index] == SignalType.NONE.value


def test_changing_rsi_bounds_changes_filter(tmp_path, tiny_indicator):
    wide = strategy_config(tmp_path, rsi_min=40.0, rsi_max=80.0)
    narrow = strategy_config(tmp_path, rsi_min=61.0, rsi_max=62.0)
    frame = entry_ready_frame(5, tiny_indicator, wide)
    index = wide.breakout_period
    wide_result = generate_signals(
        frame, wide, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    narrow_result = generate_signals(
        frame, narrow, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert wide_result["signal"].iloc[index] == SignalType.LONG_ENTRY.value
    assert narrow_result["rsi_filter"].iloc[index] == False  # noqa: E712
    assert narrow_result["signal"].iloc[index] == SignalType.NONE.value


def test_load_strategy_config_reads_yaml(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "strategy.yaml").write_text(
        "strategy:\n"
        "  name: breakout_volume_ema_rsi\n"
        '  version: "0.3"\n'
        "  breakout_period: 20\n"
        "  volume_multiplier: 1.5\n"
        "  rsi_min: 50\n"
        "  rsi_max: 70\n",
        encoding="utf-8",
    )
    loaded = load_strategy_config(config_dir / "strategy.yaml")
    assert loaded.name == "breakout_volume_ema_rsi"
    assert loaded.breakout_period == 20
    assert loaded.volume_multiplier == pytest.approx(1.5)
    assert loaded.rsi_min == pytest.approx(50.0)
    assert loaded.rsi_max == pytest.approx(70.0)
    assert not hasattr(loaded, "volume_ma_period")
    assert not hasattr(loaded, "ema_fast")


def test_load_strategy_config_rejects_rsi_min_not_less_than_max(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "strategy.yaml").write_text(
        "strategy:\n"
        "  name: x\n"
        "  breakout_period: 3\n"
        "  volume_multiplier: 1.5\n"
        "  rsi_min: 70\n"
        "  rsi_max: 50\n",
        encoding="utf-8",
    )
    with pytest.raises(StrategyConfigError, match="rsi_min"):
        load_strategy_config(config_dir / "strategy.yaml")


def test_missing_indicator_column_fails_without_repair(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy).drop(
        columns=[tiny_indicator.rsi_column]
    )
    with pytest.raises(StrategyInputError, match="rsi"):
        generate_signals(
            frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
        )


def test_integer_close_fails_without_coercion(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(5, tiny_indicator, tiny_strategy)
    frame["close"] = frame["close"].astype("int64")
    with pytest.raises(StrategyInputError, match="floating-point"):
        generate_signals(
            frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
        )


def test_never_emits_short(tiny_indicator, tiny_strategy):
    frame = entry_ready_frame(8, tiny_indicator, tiny_strategy)
    result = generate_signals(
        frame, tiny_strategy, tiny_indicator, interval="5m", symbol="BTCUSDT"
    )
    assert set(result["signal"].unique()) <= {
        SignalType.NONE.value,
        SignalType.LONG_ENTRY.value,
    }
