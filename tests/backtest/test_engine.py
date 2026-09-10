"""Sequential backtest event-loop scenarios."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.fills import (
    long_entry_price,
    long_exit_price,
    stop_loss_price,
    take_profit_price,
)
from src.backtest.ledger import (
    EXIT_END_OF_DATA,
    EXIT_SL,
    EXIT_TP,
    IGNORE_INSUFFICIENT_CASH,
    IGNORE_NO_NEXT_CANDLE,
    IGNORE_POSITION_OPEN,
    STATUS_CLOSED,
    STATUS_OPEN,
)
from src.backtest.portfolio import PendingEntry

from .conftest import make_backtest_config, ohlcv_frame, signal_frame

SYMBOL = "BTCUSDT"


def _run(tmp_path: Path, candles, long_at, **config_kw):
    config = make_backtest_config(tmp_path, **config_kw)
    signals = signal_frame(candles, set(long_at))
    return run_backtest(
        candles, signals, config, symbol=SYMBOL, interval="5m"
    ), config


def test_signal_does_not_fill_at_close(tmp_path: Path):
    candles = ohlcv_frame(3, close=90.0, open_price=100.0, high=101.0, low=99.6)
    result, _ = _run(tmp_path, candles, long_at={0})
    assert result.entries_executed == 1
    trade = result.trades[0]
    assert trade.entry_timestamp == candles["timestamp"].iloc[1]
    assert trade.entry_reference_price == pytest.approx(100.0)
    assert trade.entry_price != pytest.approx(90.0)


def test_entry_price_and_sl_tp_from_slipped_fill(tmp_path: Path):
    open_prices = [100.0, 110.0, 110.0]
    close_prices = [50.0, 110.0, 110.0]
    candles = ohlcv_frame(
        3,
        open_price=open_prices,
        close=close_prices,
        high=[51.0, 111.0, 111.0],
        low=[49.0, 109.4, 109.4],
    )
    result, config = _run(tmp_path, candles, long_at={0})
    trade = result.trades[0]
    expected_ref = 110.0
    expected_fill = long_entry_price(expected_ref, config.slippage)
    assert trade.entry_reference_price == pytest.approx(expected_ref)
    assert trade.entry_price == pytest.approx(expected_fill)
    assert trade.stop_loss == pytest.approx(stop_loss_price(expected_fill, config.stop_loss))
    assert trade.take_profit == pytest.approx(
        take_profit_price(expected_fill, config.take_profit)
    )
    assert trade.stop_loss != pytest.approx(50.0 * (1.0 - config.stop_loss))
    assert trade.stop_loss != pytest.approx(expected_ref * (1.0 - config.stop_loss))
    assert trade.quantity != pytest.approx(
        (20.0 * config.risk_per_trade)
        / (long_entry_price(50.0, config.slippage) * config.stop_loss)
    )


def test_next_open_then_tp(tmp_path: Path):
    candles = ohlcv_frame(
        3,
        open_price=100.0,
        high=[100.5, 103.0, 103.0],
        low=[99.6, 99.6, 99.6],
        close=100.0,
    )
    result, config = _run(tmp_path, candles, long_at={0})
    assert result.trades_closed == 1
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_TP
    assert trade.status == STATUS_CLOSED
    entry = long_entry_price(100.0, config.slippage)
    tp = take_profit_price(entry, config.take_profit)
    assert trade.exit_reference_price == pytest.approx(tp)
    assert trade.exit_price == pytest.approx(long_exit_price(tp, config.slippage))
    assert trade.exit_price < trade.exit_reference_price


def test_next_open_then_sl(tmp_path: Path):
    candles = ohlcv_frame(
        3,
        open_price=100.0,
        high=[100.5, 100.5, 100.5],
        low=[99.6, 98.0, 98.0],
        close=100.0,
    )
    result, config = _run(tmp_path, candles, long_at={0})
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_SL
    entry = long_entry_price(100.0, config.slippage)
    sl = stop_loss_price(entry, config.stop_loss)
    assert trade.exit_price == pytest.approx(long_exit_price(sl, config.slippage))
    assert trade.net_pnl != pytest.approx(-trade.risk_amount)
    assert trade.R == pytest.approx(trade.net_pnl / trade.risk_amount)


def test_entry_candle_hits_both_assumes_sl(tmp_path: Path):
    candles = ohlcv_frame(
        2,
        open_price=100.0,
        high=[100.5, 103.0],
        low=[99.6, 98.0],
        close=100.0,
    )
    result, _ = _run(tmp_path, candles, long_at={0})
    assert result.trades[0].exit_reason == EXIT_SL
    assert result.trades[0].entry_timestamp == result.trades[0].exit_timestamp
    assert result.trades[0].holding_time == pd.Timedelta(0)


def test_entry_candle_tp_only(tmp_path: Path):
    candles = ohlcv_frame(
        2,
        open_price=100.0,
        high=[100.5, 103.0],
        low=[99.6, 99.6],
        close=100.0,
    )
    result, _ = _run(tmp_path, candles, long_at={0})
    assert result.trades[0].exit_reason == EXIT_TP


def test_second_signal_while_open_ignored(tmp_path: Path):
    candles = ohlcv_frame(
        4,
        open_price=100.0,
        high=100.5,
        low=99.6,
        close=100.0,
    )
    result, _ = _run(tmp_path, candles, long_at={0, 1, 2})
    assert result.entries_executed == 1
    reasons = [item.reason for item in result.ignored]
    assert IGNORE_POSITION_OPEN in reasons


def test_signal_after_close_is_accepted(tmp_path: Path):
    candles = ohlcv_frame(
        5,
        open_price=100.0,
        high=[100.5, 100.5, 100.5, 103.0, 100.5],
        low=[99.6, 98.0, 99.6, 99.6, 99.6],
        close=100.0,
    )
    result, _ = _run(tmp_path, candles, long_at={0, 2})
    assert result.entries_executed == 2
    assert result.trades[0].exit_reason == EXIT_SL
    assert result.trades[1].status in {STATUS_CLOSED, STATUS_OPEN}


def test_final_candle_signal_has_no_fill(tmp_path: Path):
    candles = ohlcv_frame(3, open_price=100.0, high=100.5, low=99.6, close=100.0)
    result, _ = _run(tmp_path, candles, long_at={2})
    assert result.entries_executed == 0
    assert result.trades == []
    assert result.ignored[0].reason == IGNORE_NO_NEXT_CANDLE
    last = pd.Timestamp(candles["timestamp"].iloc[2])
    expected_exec = last + pd.Timedelta(5, unit="m")
    candle_times = {pd.Timestamp(ts) for ts in candles["timestamp"].tolist()}
    assert expected_exec not in candle_times


def test_end_of_data_open_isolation(tmp_path: Path):
    candles = ohlcv_frame(
        4,
        open_price=100.0,
        high=[100.5, 100.5, 101.0, 101.8],
        low=[99.6, 99.6, 99.6, 99.6],
        close=[100.0, 100.0, 100.5, 101.5],
    )
    result, config = _run(tmp_path, candles, long_at={0})
    assert result.unresolved_count == 1
    trade = result.trades[0]
    assert trade.status == STATUS_OPEN
    assert trade.exit_reason == EXIT_END_OF_DATA
    assert trade.exit_price is None
    assert trade.exit_fee is None
    assert trade.exit_slippage is None
    assert trade.net_pnl is None
    assert trade.gross_pnl is None
    assert trade.R is None
    assert trade.exit_timestamp is None
    assert trade.exit_reference_price is None
    assert trade.slippage is None
    assert trade.holding_time is None
    assert result.realized_net_pnl == 0.0
    leftover = config.starting_capital - trade.position_value - trade.entry_fee
    assert result.cash == pytest.approx(leftover)
    assert result.unrealized_mark_value == pytest.approx(trade.quantity * 101.5)
    assert result.informational_equity == pytest.approx(
        result.cash + result.unrealized_mark_value
    )
    assert result.informational_equity > result.cash
    assert result.realized_net_pnl != result.informational_equity - config.starting_capital


def test_insufficient_cash_clears_pending(tmp_path: Path):
    candles = ohlcv_frame(
        3,
        open_price=[100.0, 0.0, 100.0],
        high=[100.5, 0.0, 100.5],
        low=[99.6, 0.0, 99.6],
        close=[100.0, 0.0, 100.0],
    )
    result, config = _run(tmp_path, candles, long_at={0})
    assert result.entries_executed == 0
    assert result.trades == []
    assert result.cash == pytest.approx(config.starting_capital)
    assert result.ignored[0].reason == IGNORE_INSUFFICIENT_CASH
    assert result.pending_cleared is True
    names = {item.name for item in fields(PendingEntry)}
    assert "quantity" not in names


def test_boundary_equality_triggers(tmp_path: Path):
    config = make_backtest_config(tmp_path)
    entry = long_entry_price(100.0, config.slippage)
    sl = stop_loss_price(entry, config.stop_loss)
    candles = ohlcv_frame(
        3,
        open_price=100.0,
        high=[100.5, 100.5, 100.5],
        low=[99.6, sl, 99.6],
        close=100.0,
    )
    result, _ = _run(tmp_path, candles, long_at={0})
    assert result.trades[0].exit_reason == EXIT_SL


def test_cash_equals_starting_plus_net_pnl_sl(tmp_path: Path):
    candles = ohlcv_frame(
        3,
        open_price=100.0,
        high=[100.5, 100.5, 100.5],
        low=[99.6, 98.0, 98.0],
        close=100.0,
    )
    result, config = _run(tmp_path, candles, long_at={0})
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_SL
    assert result.cash == pytest.approx(config.starting_capital + trade.net_pnl)
    assert result.cash == pytest.approx(config.starting_capital + result.realized_net_pnl)


def test_cash_equals_starting_plus_net_pnl_tp(tmp_path: Path):
    candles = ohlcv_frame(
        3,
        open_price=100.0,
        high=[100.5, 103.0, 103.0],
        low=[99.6, 99.6, 99.6],
        close=100.0,
    )
    result, config = _run(tmp_path, candles, long_at={0})
    trade = result.trades[0]
    assert trade.exit_reason == EXIT_TP
    assert result.cash == pytest.approx(config.starting_capital + trade.net_pnl)


def test_closed_identities(tmp_path: Path):
    candles = ohlcv_frame(
        3,
        open_price=100.0,
        high=[100.5, 100.5, 100.5],
        low=[99.6, 98.0, 98.0],
        close=100.0,
    )
    result, _ = _run(tmp_path, candles, long_at={0})
    trade = result.trades[0]
    reconstructed = (
        trade.gross_pnl - trade.slippage - trade.entry_fee - trade.exit_fee
    )
    assert trade.net_pnl == pytest.approx(reconstructed)
    assert trade.entry_fee > 0
    assert trade.exit_fee > 0
    assert trade.entry_slippage > 0
    assert trade.exit_slippage > 0


def test_determinism(tmp_path: Path):
    candles = ohlcv_frame(
        4,
        open_price=100.0,
        high=[100.5, 103.0, 100.5, 100.5],
        low=[99.6, 99.6, 99.6, 99.6],
        close=100.0,
    )
    first, _ = _run(tmp_path, candles, long_at={0, 2})
    second, _ = _run(tmp_path, candles, long_at={0, 2})
    pd.testing.assert_frame_equal(first.trade_frame, second.trade_frame)
    pd.testing.assert_frame_equal(first.ignored_frame, second.ignored_frame)
    assert first.cash == second.cash
    assert first.realized_net_pnl == second.realized_net_pnl


def test_lookahead_future_mutation_does_not_change_past(tmp_path: Path):
    candles = ohlcv_frame(
        6,
        open_price=100.0,
        high=[100.5, 100.5, 100.5, 100.5, 103.0, 100.5],
        low=[99.6, 98.0, 99.6, 99.6, 99.6, 99.6],
        close=100.0,
    )
    baseline, _ = _run(tmp_path, candles, long_at={0, 2})
    mutated = candles.copy(deep=True)
    mutated.loc[4, "high"] = 10_000.0
    mutated.loc[4, "low"] = 1.0
    mutated.loc[5, "close"] = 10_000.0
    after, _ = _run(tmp_path, mutated, long_at={0, 2})
    first_exit = baseline.trades[0].exit_timestamp
    pd.testing.assert_series_equal(
        baseline.trade_frame.iloc[0],
        after.trade_frame.iloc[0],
        check_names=False,
    )
    assert after.trades[0].exit_reason == baseline.trades[0].exit_reason
    assert after.trades[0].exit_timestamp == first_exit


def test_month_boundary_position_can_span_months(tmp_path: Path):
    january = ohlcv_frame(
        2,
        start="2024-01-31 23:50:00",
        open_price=100.0,
        high=100.5,
        low=99.6,
        close=100.0,
    )
    february = ohlcv_frame(
        2,
        start="2024-02-01 00:00:00",
        open_price=100.0,
        high=[100.5, 100.5],
        low=[99.6, 98.0],
        close=100.0,
    )
    candles = pd.concat([january, february], ignore_index=True)
    result, _ = _run(tmp_path, candles, long_at={1})
    trade = result.trades[0]
    assert trade.signal_timestamp == pd.Timestamp("2024-01-31 23:55:00", tz="UTC")
    assert trade.entry_timestamp == pd.Timestamp("2024-02-01 00:00:00", tz="UTC")
    assert trade.exit_timestamp == pd.Timestamp("2024-02-01 00:05:00", tz="UTC")
    assert trade.exit_reason == EXIT_SL


def test_does_not_mutate_inputs(tmp_path: Path):
    candles = ohlcv_frame(3, open_price=100.0, high=100.5, low=99.6, close=100.0)
    signals = signal_frame(candles, {0})
    original_c = candles.copy(deep=True)
    original_s = signals.copy(deep=True)
    config = make_backtest_config(tmp_path)
    run_backtest(candles, signals, config, symbol=SYMBOL, interval="5m")
    pd.testing.assert_frame_equal(candles, original_c)
    pd.testing.assert_frame_equal(signals, original_s)


def test_no_time_exit_reason(tmp_path: Path):
    candles = ohlcv_frame(
        3,
        open_price=100.0,
        high=[100.5, 103.0, 100.5],
        low=[99.6, 99.6, 99.6],
        close=100.0,
    )
    result, _ = _run(tmp_path, candles, long_at={0})
    assert all(trade.exit_reason != "TIME_EXIT" for trade in result.trades)
