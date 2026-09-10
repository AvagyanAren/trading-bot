"""Development/test split and lookback behaviour."""

from __future__ import annotations

import pandas as pd
import pytest

from src.research.periods import (
    TEST_CUTOFF,
    ResearchPeriodError,
    align_signals_to_candles,
    assert_development_window,
    prepare_experiment_frames,
    window_mask,
)
from tests.backtest.conftest import ohlcv_frame


def test_development_slice_excludes_2025():
    candles = ohlcv_frame(5, start="2024-12-31 23:45:00")
    # 23:45, 23:50, 23:55, 2025-01-01 00:00, 00:05
    signal, trade = prepare_experiment_frames(
        candles, start=__import__("datetime").date(2024, 12, 31),
        end=__import__("datetime").date(2024, 12, 31),
        lookback_bars=20,
    )
    assert trade["timestamp"].max() < TEST_CUTOFF
    assert signal["timestamp"].max() < TEST_CUTOFF
    assert_development_window(trade)
    assert_development_window(signal)


def test_test_window_uses_prior_highs_for_lookback():
    candles = ohlcv_frame(6, start="2024-12-31 23:40:00")
    from datetime import date

    signal, trade = prepare_experiment_frames(
        candles, start=date(2025, 1, 1), end=date(2025, 1, 1), lookback_bars=2
    )
    assert trade["timestamp"].min() >= TEST_CUTOFF
    assert signal["timestamp"].min() < TEST_CUTOFF
    assert len(signal) == len(trade) + 2


def test_assert_development_window_rejects_2025():
    candles = ohlcv_frame(2, start="2025-01-01 00:00:00")
    with pytest.raises(ResearchPeriodError, match="2025"):
        assert_development_window(candles)


def test_align_signals_follow_candle_order():
    candles = ohlcv_frame(3, start="2024-01-01 00:00:00")
    signals = pd.DataFrame(
        {
            "timestamp": candles["timestamp"],
            "signal": ["NONE", "LONG_ENTRY", "NONE"],
        }
    )
    aligned = align_signals_to_candles(signals, candles)
    assert list(aligned["signal"]) == ["NONE", "LONG_ENTRY", "NONE"]


def test_window_mask_is_end_inclusive_by_calendar_date():
    from datetime import date

    candles = ohlcv_frame(3, start="2024-12-31 23:50:00")
    mask = window_mask(candles["timestamp"], date(2024, 12, 31), date(2024, 12, 31))
    assert int(mask.sum()) == 2
    assert candles.loc[mask, "timestamp"].max() < TEST_CUTOFF
