"""Fill, fee, and slippage formula tests."""

from __future__ import annotations

import pytest

from src.backtest.fills import (
    entry_slippage_cost,
    exit_slippage_cost,
    gross_pnl,
    long_entry_price,
    long_exit_price,
    net_pnl,
    r_multiple,
    stop_loss_price,
    take_profit_price,
    transaction_fee,
)


def test_long_entry_is_worse_than_reference():
    assert long_entry_price(100.0, 0.0005) == pytest.approx(100.05)


def test_long_exit_is_worse_than_reference():
    assert long_exit_price(99.0, 0.0005) == pytest.approx(99.0 * 0.9995)


def test_sl_tp_from_entry_fill_not_reference():
    entry = long_entry_price(100.0, 0.0005)
    sl = stop_loss_price(entry, 0.01)
    tp = take_profit_price(entry, 0.02)
    assert sl == pytest.approx(entry * 0.99)
    assert tp == pytest.approx(entry * 1.02)
    assert sl != pytest.approx(100.0 * 0.99)
    assert tp != pytest.approx(100.0 * 1.02)


def test_fees_and_net_identity():
    qty = 0.2
    entry_ref = 100.0
    entry = long_entry_price(entry_ref, 0.0005)
    sl = stop_loss_price(entry, 0.01)
    exit_px = long_exit_price(sl, 0.0005)
    entry_fee = transaction_fee(qty, entry, 0.001)
    exit_fee = transaction_fee(qty, exit_px, 0.001)
    entry_slip = entry_slippage_cost(qty, entry, entry_ref)
    exit_slip = exit_slippage_cost(qty, sl, exit_px)
    gross = gross_pnl(qty, sl, entry_ref)
    net = net_pnl(qty, exit_px, entry, entry_fee, exit_fee)
    assert net == pytest.approx(gross - entry_slip - exit_slip - entry_fee - exit_fee)
    assert r_multiple(net, qty * entry * 0.01) == pytest.approx(net / (qty * entry * 0.01))
    assert entry_fee > 0
    assert exit_fee > 0
