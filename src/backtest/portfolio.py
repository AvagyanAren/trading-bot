"""Account, pending entry, and one LONG position. Strategy stays stateless."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class PendingEntry:
    """Next-open intent. Timing only — no frozen quantity or prices."""

    signal_timestamp: pd.Timestamp
    execute_at: pd.Timestamp


@dataclass
class Position:
    """Open LONG. SL/TP are references from the actual entry fill."""

    trade_id: int
    symbol: str
    side: str
    quantity: float
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    entry_reference_price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_fee: float
    entry_slippage: float
    risk_amount: float
    position_value: float


@dataclass
class Account:
    """Realized cash and at most one position / pending entry."""

    cash: float
    position: Position | None = None
    pending: PendingEntry | None = None
    next_trade_id: int = 1
    realized_net_pnl: float = 0.0
    min_cash: float = field(init=False)

    def __post_init__(self) -> None:
        self.min_cash = self.cash

    def record_cash(self) -> None:
        if self.cash < self.min_cash:
            self.min_cash = self.cash
