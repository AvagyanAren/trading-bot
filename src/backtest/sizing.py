"""LONG position sizing at pending-entry execution time (Phase A only)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import BacktestConfig
from .fills import (
    entry_slippage_cost,
    long_entry_price,
    stop_loss_price,
    take_profit_price,
    transaction_fee,
)


@dataclass(frozen=True)
class SizeDecision:
    """Result of sizing against current cash and the execution OPEN."""

    affordable: bool
    quantity: float
    entry_reference_price: float
    entry_price: float
    position_value: float
    entry_fee: float
    entry_slippage: float
    risk_amount: float
    stop_loss: float
    take_profit: float


def size_long_entry(
    cash: float,
    execution_open: float,
    config: BacktestConfig,
) -> SizeDecision:
    """Compute quantity from cash, OPEN, fees, and stop loss.

    Must be called at fill time. Do not freeze the result when a pending
    entry is scheduled.
    """
    entry_reference = float(execution_open)
    entry_price = long_entry_price(entry_reference, config.slippage)
    stop = stop_loss_price(entry_price, config.stop_loss)
    take = take_profit_price(entry_price, config.take_profit)
    empty = SizeDecision(
        affordable=False,
        quantity=0.0,
        entry_reference_price=entry_reference,
        entry_price=entry_price,
        position_value=0.0,
        entry_fee=0.0,
        entry_slippage=0.0,
        risk_amount=0.0,
        stop_loss=stop,
        take_profit=take,
    )
    if cash <= 0 or entry_price <= 0 or config.stop_loss <= 0:
        return empty

    intended_risk = cash * config.risk_per_trade
    qty_risk = intended_risk / (entry_price * config.stop_loss)
    qty_cash = cash / (entry_price * (1.0 + config.entry_rate))
    quantity = min(qty_risk, qty_cash)
    if quantity <= 0:
        return empty

    position_value = quantity * entry_price
    entry_fee = transaction_fee(quantity, entry_price, config.entry_rate)
    needed = position_value + entry_fee
    if needed > cash and needed - cash > 1e-9:
        return empty

    return SizeDecision(
        affordable=True,
        quantity=quantity,
        entry_reference_price=entry_reference,
        entry_price=entry_price,
        position_value=position_value,
        entry_fee=entry_fee,
        entry_slippage=entry_slippage_cost(quantity, entry_price, entry_reference),
        risk_amount=quantity * entry_price * config.stop_loss,
        stop_loss=stop,
        take_profit=take,
    )
