"""Pending-entry dataclass contract."""

from __future__ import annotations

from dataclasses import fields

from src.backtest.portfolio import PendingEntry


def test_pending_stores_timing_only():
    names = {item.name for item in fields(PendingEntry)}
    assert names == {"signal_timestamp", "execute_at"}
    for forbidden in (
        "quantity",
        "entry_price",
        "risk_amount",
        "stop_loss",
        "take_profit",
        "position_value",
    ):
        assert forbidden not in names
