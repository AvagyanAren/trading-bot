"""Trade ledger and ignored-signal records."""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd

STATUS_CLOSED = "CLOSED"
STATUS_OPEN = "OPEN"
SIDE_LONG = "LONG"

EXIT_SL = "SL"
EXIT_TP = "TP"
EXIT_END_OF_DATA = "END_OF_DATA"

IGNORE_POSITION_OPEN = "POSITION_OPEN"
IGNORE_NO_NEXT_CANDLE = "NO_NEXT_CANDLE"
IGNORE_INSUFFICIENT_CASH = "INSUFFICIENT_CASH"

TRADE_FRAME_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "status",
    "symbol",
    "side",
    "signal_timestamp",
    "entry_timestamp",
    "entry_reference_price",
    "entry_price",
    "exit_timestamp",
    "exit_reference_price",
    "exit_price",
    "quantity",
    "position_value",
    "risk_amount",
    "stop_loss",
    "take_profit",
    "exit_reason",
    "gross_pnl",
    "entry_fee",
    "exit_fee",
    "entry_slippage",
    "exit_slippage",
    "slippage",
    "net_pnl",
    "R",
    "holding_time",
)

IGNORED_FRAME_COLUMNS: tuple[str, ...] = (
    "signal_timestamp",
    "reason",
)


@dataclass
class Trade:
    """One simulated LONG. OPEN rows keep realized exit/result fields None."""

    trade_id: int
    status: str
    symbol: str
    side: str
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    entry_reference_price: float
    entry_price: float
    exit_timestamp: pd.Timestamp | None
    exit_reference_price: float | None
    exit_price: float | None
    quantity: float
    position_value: float
    risk_amount: float
    stop_loss: float
    take_profit: float
    exit_reason: str
    gross_pnl: float | None
    entry_fee: float
    exit_fee: float | None
    entry_slippage: float
    exit_slippage: float | None
    slippage: float | None
    net_pnl: float | None
    R: float | None
    holding_time: pd.Timedelta | None


@dataclass(frozen=True)
class IgnoredSignal:
    signal_timestamp: pd.Timestamp
    reason: str


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame({column: [] for column in TRADE_FRAME_COLUMNS})
    rows = [{item.name: getattr(trade, item.name) for item in fields(Trade)} for trade in trades]
    return pd.DataFrame(rows, columns=list(TRADE_FRAME_COLUMNS))


def ignored_to_frame(records: list[IgnoredSignal]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame({column: [] for column in IGNORED_FRAME_COLUMNS})
    return pd.DataFrame(
        [{"signal_timestamp": item.signal_timestamp, "reason": item.reason} for item in records],
        columns=list(IGNORED_FRAME_COLUMNS),
    )
