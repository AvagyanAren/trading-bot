"""Committed v0.4 full-period baseline totals for optional integrity replay.

These numbers come from reports/backtest_validation_BTCUSDT_5m.txt.
Integrity is not strategy selection and must not write v0.4 report paths.
"""

from __future__ import annotations

from dataclasses import dataclass

RECONCILE_EPS = 1e-8


@dataclass(frozen=True)
class FrozenV04Totals:
    signals: int = 2621
    filled_trades: int = 766
    closed_trades: int = 766
    tp_count: int = 250
    sl_count: int = 516
    total_entry_fees: float = 6.725043036533116
    total_exit_fees: float = 6.72081761892159
    total_slippage: float = 6.722930952104964
    total_gross_pnl: float = 2.4975133405782928
    total_net_pnl: float = -17.67127826698138


FROZEN_V04_FULL_PERIOD = FrozenV04Totals()
