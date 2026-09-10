"""Simulated fill prices, fees, and dollar slippage. No portfolio state."""

from __future__ import annotations


def long_entry_price(reference: float, slippage: float) -> float:
    """Actual LONG purchase: worse than the reference (next open)."""
    return reference * (1.0 + slippage)


def long_exit_price(reference: float, slippage: float) -> float:
    """Actual LONG sale: worse than the SL/TP reference."""
    return reference * (1.0 - slippage)


def stop_loss_price(entry_price: float, stop_loss: float) -> float:
    """SL reference from the actual entry fill, not the unslipped open."""
    return entry_price * (1.0 - stop_loss)


def take_profit_price(entry_price: float, take_profit: float) -> float:
    """TP reference from the actual entry fill, not the unslipped open."""
    return entry_price * (1.0 + take_profit)


def transaction_fee(quantity: float, fill_price: float, rate: float) -> float:
    return quantity * fill_price * rate


def entry_slippage_cost(
    quantity: float, entry_price: float, entry_reference: float
) -> float:
    return quantity * (entry_price - entry_reference)


def exit_slippage_cost(
    quantity: float, exit_reference: float, exit_price: float
) -> float:
    return quantity * (exit_reference - exit_price)


def gross_pnl(
    quantity: float, exit_reference: float, entry_reference: float
) -> float:
    return quantity * (exit_reference - entry_reference)


def net_pnl(
    quantity: float,
    exit_price: float,
    entry_price: float,
    entry_fee: float,
    exit_fee: float,
) -> float:
    return quantity * (exit_price - entry_price) - entry_fee - exit_fee


def r_multiple(net: float, risk_amount: float) -> float:
    return net / risk_amount
