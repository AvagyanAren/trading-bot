"""Metric definitions on synthetic trades."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.ledger import EXIT_END_OF_DATA, EXIT_SL, EXIT_TP, STATUS_OPEN
from src.research.metrics import compute_metrics, max_consecutive

from .conftest import flat_equity, make_result, make_trade


def test_average_net_and_r_and_expectancy_aliases():
    trades = [
        make_trade(trade_id=1, net_pnl=0.2, R=1.0, exit_reason=EXIT_TP),
        make_trade(
            trade_id=2,
            net_pnl=-0.1,
            R=-0.5,
            exit_reason=EXIT_SL,
            entry_timestamp="2024-01-01 00:20:00",
            exit_timestamp="2024-01-01 00:30:00",
        ),
        make_trade(
            trade_id=3,
            net_pnl=0.4,
            R=2.0,
            exit_reason=EXIT_TP,
            entry_timestamp="2024-01-01 00:40:00",
            exit_timestamp="2024-01-01 00:50:00",
        ),
    ]
    result = make_result(trades, signals_received=5)
    metrics = compute_metrics(result, flat_equity(20.5))
    assert metrics.average_net_pnl_per_closed_trade == pytest.approx(0.5 / 3)
    assert metrics.average_R_per_closed_trade == pytest.approx((1.0 - 0.5 + 2.0) / 3)
    assert metrics.expectancy_per_trade == metrics.average_net_pnl_per_closed_trade
    assert metrics.expectancy_R == metrics.average_R_per_closed_trade
    assert metrics.win_rate == pytest.approx(2 / 3)
    assert metrics.loss_rate == pytest.approx(1 / 3)
    assert metrics.total_R == pytest.approx(2.5)


def test_win_rate_excludes_scratches_from_wins():
    trades = [
        make_trade(trade_id=1, net_pnl=0.2, R=1.0),
        make_trade(
            trade_id=2,
            net_pnl=0.0,
            R=0.0,
            entry_timestamp="2024-01-01 00:20:00",
            exit_timestamp="2024-01-01 00:30:00",
        ),
        make_trade(
            trade_id=3,
            net_pnl=-0.1,
            R=-0.5,
            exit_reason=EXIT_SL,
            entry_timestamp="2024-01-01 00:40:00",
            exit_timestamp="2024-01-01 00:50:00",
        ),
    ]
    result = make_result(trades)
    metrics = compute_metrics(result, flat_equity(20.0 + 0.1))
    assert metrics.win_rate == pytest.approx(1 / 3)
    assert metrics.scratch_count == 1


def test_profit_factor_and_zero_loss_is_na():
    trades = [
        make_trade(trade_id=1, net_pnl=0.2, R=1.0),
        make_trade(
            trade_id=2,
            net_pnl=0.4,
            R=2.0,
            entry_timestamp="2024-01-01 00:20:00",
            exit_timestamp="2024-01-01 00:30:00",
        ),
    ]
    result = make_result(trades)
    metrics = compute_metrics(result, flat_equity(20.6))
    assert metrics.profit_factor is None
    mixed = [
        make_trade(trade_id=1, net_pnl=0.4, R=2.0),
        make_trade(
            trade_id=2,
            net_pnl=-0.2,
            R=-1.0,
            exit_reason=EXIT_SL,
            entry_timestamp="2024-01-01 00:20:00",
            exit_timestamp="2024-01-01 00:30:00",
        ),
    ]
    mixed_result = make_result(mixed)
    mixed_metrics = compute_metrics(mixed_result, flat_equity(20.2))
    assert mixed_metrics.profit_factor == pytest.approx(0.4 / 0.2)


def test_fee_slippage_separation_excludes_open_entry_fee():
    closed = make_trade(
        net_pnl=0.5,
        gross_pnl=2.5,
        entry_fee=0.4,
        exit_fee=0.6,
        slippage=1.0,
    )
    opened = make_trade(
        trade_id=2,
        status=STATUS_OPEN,
        exit_reason=EXIT_END_OF_DATA,
        net_pnl=None,
        gross_pnl=None,
        R=None,
        exit_fee=None,
        slippage=None,
        exit_timestamp=None,
        exit_price=None,
        holding_minutes=None,
        entry_fee=3.0,
        entry_timestamp="2024-01-01 00:40:00",
    )
    result = make_result([closed, opened])
    equity = flat_equity(result.cash)
    equity.loc[:, "realized_equity"] = 20.0 + 0.5
    metrics = compute_metrics(result, equity)
    assert metrics.total_gross_pnl == pytest.approx(2.5)
    assert metrics.total_fees == pytest.approx(1.0)
    assert metrics.total_slippage == pytest.approx(1.0)
    assert metrics.total_net_pnl == pytest.approx(0.5)
    assert metrics.open_entry_fees == pytest.approx(3.0)
    assert metrics.open_end_of_data == 1


def test_open_end_of_data_excluded_from_realized():
    opened = make_trade(
        status=STATUS_OPEN,
        exit_reason=EXIT_END_OF_DATA,
        net_pnl=None,
        gross_pnl=None,
        R=None,
        exit_fee=None,
        slippage=None,
        exit_timestamp=None,
        exit_price=None,
        holding_minutes=None,
        position_value=5.0,
        entry_fee=0.1,
    )
    result = make_result([opened])
    equity = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 00:00", tz="UTC")],
            "cash": [result.cash],
            "position_quantity": [opened.quantity],
            "position_mark_value": [opened.quantity * 110.0],
            "realized_equity": [20.0],
            "mtm_equity": [result.cash + opened.quantity * 110.0],
            "unrealized_pnl": [result.cash + opened.quantity * 110.0 - 20.0],
            "drawdown": [0.0],
            "drawdown_pct": [0.0],
        }
    )
    metrics = compute_metrics(result, equity)
    assert metrics.closed_trades == 0
    assert metrics.total_net_pnl == pytest.approx(0.0)
    assert metrics.ending_realized_equity == pytest.approx(20.0)
    assert metrics.ending_mtm_equity == pytest.approx(result.cash + opened.quantity * 110.0)
    assert metrics.win_rate is None
    assert metrics.average_R_per_closed_trade is None


def test_consecutive_losses_reset_on_scratch():
    trades = [
        make_trade(trade_id=1, net_pnl=0.1, R=1.0),
        make_trade(trade_id=2, net_pnl=-0.1, R=-1.0, exit_reason=EXIT_SL),
        make_trade(trade_id=3, net_pnl=-0.1, R=-1.0, exit_reason=EXIT_SL),
        make_trade(trade_id=4, net_pnl=-0.1, R=-1.0, exit_reason=EXIT_SL),
        make_trade(trade_id=5, net_pnl=0.1, R=1.0),
    ]
    assert max_consecutive(trades) == (1, 3)
    trades[3] = make_trade(trade_id=4, net_pnl=0.0, R=0.0)
    assert max_consecutive(trades)[1] == 2
