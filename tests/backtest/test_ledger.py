"""Trade ledger schema tests."""

from __future__ import annotations

from src.backtest.ledger import TRADE_FRAME_COLUMNS, trades_to_frame


def test_required_columns_present():
    required = {
        "trade_id",
        "status",
        "symbol",
        "side",
        "signal_timestamp",
        "entry_timestamp",
        "entry_reference_price",
        "entry_price",
        "exit_timestamp",
        "exit_price",
        "quantity",
        "position_value",
        "stop_loss",
        "take_profit",
        "exit_reason",
        "gross_pnl",
        "entry_fee",
        "exit_fee",
        "slippage",
        "net_pnl",
        "R",
        "holding_time",
        "risk_amount",
        "entry_slippage",
        "exit_slippage",
        "exit_reference_price",
    }
    assert required.issubset(set(TRADE_FRAME_COLUMNS))
    assert "TIME_EXIT" not in TRADE_FRAME_COLUMNS


def test_empty_frame_has_schema():
    frame = trades_to_frame([])
    assert list(frame.columns) == list(TRADE_FRAME_COLUMNS)
    assert len(frame) == 0
