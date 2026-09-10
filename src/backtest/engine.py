"""Sequential look-ahead-free backtest event loop.

Phase A: execute a pending next-open entry (size now; never from Phase C).
Phase B: if a position is open, evaluate this candle's high/low for SL/TP.
Phase C: read this row's signal only; schedule the next open if flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.strategy.signals import SignalType

from .config import BacktestConfig, BacktestInputError
from .exits import ExitHit, evaluate_ohlcv_exit
from .fills import (
    exit_slippage_cost,
    gross_pnl,
    long_exit_price,
    net_pnl,
    r_multiple,
    transaction_fee,
)
from .ledger import (
    EXIT_END_OF_DATA,
    EXIT_SL,
    EXIT_TP,
    IGNORE_INSUFFICIENT_CASH,
    IGNORE_NO_NEXT_CANDLE,
    IGNORE_POSITION_OPEN,
    IgnoredSignal,
    SIDE_LONG,
    STATUS_CLOSED,
    STATUS_OPEN,
    TRADE_FRAME_COLUMNS,
    Trade,
    ignored_to_frame,
    trades_to_frame,
)
from .portfolio import Account, PendingEntry, Position
from .sizing import size_long_entry

REQUIRED_CANDLE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
)
REQUIRED_SIGNAL_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "signal",
)

_CASH_EPS = 1e-9


@dataclass
class BacktestResult:
    """Outcome of one sequential replay. No analytics beyond engine audit."""

    starting_capital: float
    cash: float
    realized_net_pnl: float
    min_cash: float
    candles_processed: int
    signals_received: int
    entries_executed: int
    trades_closed: int
    unresolved_count: int
    last_close: float | None
    unrealized_mark_value: float | None
    informational_equity: float | None
    pending_cleared: bool
    trades: list[Trade] = field(default_factory=list)
    ignored: list[IgnoredSignal] = field(default_factory=list)

    @property
    def trade_frame(self) -> pd.DataFrame:
        return trades_to_frame(self.trades)

    @property
    def ignored_frame(self) -> pd.DataFrame:
        return ignored_to_frame(self.ignored)


def _as_utc(value: object, field: str) -> pd.Timestamp:
    if value is None or pd.isna(value):
        raise BacktestInputError(f"{field} is missing")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise BacktestInputError(f"{field} must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def _as_float(value: object, field: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise BacktestInputError(f"{field} must be numeric, got {value!r}") from error
    if not np.isfinite(number):
        raise BacktestInputError(f"{field} must be finite, got {value!r}")
    return number


def _is_long_entry(value: object) -> bool:
    if value is SignalType.LONG_ENTRY:
        return True
    if isinstance(value, str) and value == SignalType.LONG_ENTRY.value:
        return True
    return False


def _assert_frames(candles: pd.DataFrame, signals: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_CANDLE_COLUMNS if column not in candles.columns]
    if missing:
        raise BacktestInputError(
            f"Candles missing required column(s): {', '.join(missing)}"
        )
    missing_signals = [
        column for column in REQUIRED_SIGNAL_COLUMNS if column not in signals.columns
    ]
    if missing_signals:
        raise BacktestInputError(
            f"Signals missing required column(s): {', '.join(missing_signals)}"
        )
    if len(candles) != len(signals):
        raise BacktestInputError(
            f"Candle rows {len(candles)} != signal rows {len(signals)}"
        )
    tz = getattr(candles["timestamp"].dtype, "tz", None)
    if str(tz) != "UTC":
        raise BacktestInputError(
            f"candle timestamp dtype must be datetime64[ns, UTC], got {candles['timestamp'].dtype}"
        )
    if not candles["timestamp"].equals(signals["timestamp"]):
        raise BacktestInputError("signal timestamps must match candle timestamps")


def _debit(cash: float, amount: float) -> float | None:
    remaining = cash - amount
    if remaining >= 0:
        return remaining
    if remaining >= -_CASH_EPS:
        return 0.0
    return None


def _close_trade(
    account: Account,
    position: Position,
    *,
    exit_timestamp: pd.Timestamp,
    exit_reason: str,
    exit_reference: float,
    config: BacktestConfig,
) -> Trade:
    exit_price = long_exit_price(exit_reference, config.slippage)
    exit_fee = transaction_fee(position.quantity, exit_price, config.exit_rate)
    exit_slip = exit_slippage_cost(position.quantity, exit_reference, exit_price)
    combined_slip = position.entry_slippage + exit_slip
    gross = gross_pnl(
        position.quantity, exit_reference, position.entry_reference_price
    )
    net = net_pnl(
        position.quantity,
        exit_price,
        position.entry_price,
        position.entry_fee,
        exit_fee,
    )
    proceeds = position.quantity * exit_price - exit_fee
    credited = account.cash + proceeds
    if credited < 0 and credited >= -_CASH_EPS:
        credited = 0.0
    account.cash = credited
    account.realized_net_pnl += net
    account.position = None
    account.record_cash()
    return Trade(
        trade_id=position.trade_id,
        status=STATUS_CLOSED,
        symbol=position.symbol,
        side=position.side,
        signal_timestamp=position.signal_timestamp,
        entry_timestamp=position.entry_timestamp,
        entry_reference_price=position.entry_reference_price,
        entry_price=position.entry_price,
        exit_timestamp=exit_timestamp,
        exit_reference_price=exit_reference,
        exit_price=exit_price,
        quantity=position.quantity,
        position_value=position.position_value,
        risk_amount=position.risk_amount,
        stop_loss=position.stop_loss,
        take_profit=position.take_profit,
        exit_reason=exit_reason,
        gross_pnl=gross,
        entry_fee=position.entry_fee,
        exit_fee=exit_fee,
        entry_slippage=position.entry_slippage,
        exit_slippage=exit_slip,
        slippage=combined_slip,
        net_pnl=net,
        R=r_multiple(net, position.risk_amount),
        holding_time=exit_timestamp - position.entry_timestamp,
    )


def _open_end_of_data(position: Position) -> Trade:
    return Trade(
        trade_id=position.trade_id,
        status=STATUS_OPEN,
        symbol=position.symbol,
        side=position.side,
        signal_timestamp=position.signal_timestamp,
        entry_timestamp=position.entry_timestamp,
        entry_reference_price=position.entry_reference_price,
        entry_price=position.entry_price,
        exit_timestamp=None,
        exit_reference_price=None,
        exit_price=None,
        quantity=position.quantity,
        position_value=position.position_value,
        risk_amount=position.risk_amount,
        stop_loss=position.stop_loss,
        take_profit=position.take_profit,
        exit_reason=EXIT_END_OF_DATA,
        gross_pnl=None,
        entry_fee=position.entry_fee,
        exit_fee=None,
        entry_slippage=position.entry_slippage,
        exit_slippage=None,
        slippage=None,
        net_pnl=None,
        R=None,
        holding_time=None,
    )


def run_backtest(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    config: BacktestConfig,
    *,
    symbol: str,
    interval: str,
) -> BacktestResult:
    """Replay candles sequentially. Does not mutate inputs or use future rows."""
    del interval  # reserved for callers; timestamps are already aligned
    _assert_frames(candles, signals)

    account = Account(cash=float(config.starting_capital))
    trades: list[Trade] = []
    ignored: list[IgnoredSignal] = []
    entries_executed = 0
    signals_received = 0
    n = len(candles)

    for index in range(n):
        timestamp = _as_utc(candles["timestamp"].iloc[index], "timestamp")
        open_price = _as_float(candles["open"].iloc[index], "open")
        high = _as_float(candles["high"].iloc[index], "high")
        low = _as_float(candles["low"].iloc[index], "low")
        close = _as_float(candles["close"].iloc[index], "close")

        if account.pending is not None:
            if account.pending.execute_at != timestamp:
                raise BacktestInputError(
                    "Pending entry execute_at does not match current candle open"
                )
            pending = account.pending
            sized = size_long_entry(account.cash, open_price, config)
            if not sized.affordable:
                ignored.append(
                    IgnoredSignal(
                        signal_timestamp=pending.signal_timestamp,
                        reason=IGNORE_INSUFFICIENT_CASH,
                    )
                )
                account.pending = None
            else:
                remaining = _debit(account.cash, sized.position_value + sized.entry_fee)
                if remaining is None:
                    ignored.append(
                        IgnoredSignal(
                            signal_timestamp=pending.signal_timestamp,
                            reason=IGNORE_INSUFFICIENT_CASH,
                        )
                    )
                    account.pending = None
                else:
                    account.cash = remaining
                    account.position = Position(
                        trade_id=account.next_trade_id,
                        symbol=symbol,
                        side=SIDE_LONG,
                        quantity=sized.quantity,
                        signal_timestamp=pending.signal_timestamp,
                        entry_timestamp=timestamp,
                        entry_reference_price=sized.entry_reference_price,
                        entry_price=sized.entry_price,
                        stop_loss=sized.stop_loss,
                        take_profit=sized.take_profit,
                        entry_fee=sized.entry_fee,
                        entry_slippage=sized.entry_slippage,
                        risk_amount=sized.risk_amount,
                        position_value=sized.position_value,
                    )
                    account.next_trade_id += 1
                    account.pending = None
                    entries_executed += 1
                    account.record_cash()

        if account.position is not None:
            hit = evaluate_ohlcv_exit(
                high,
                low,
                account.position.stop_loss,
                account.position.take_profit,
                same_candle_priority=config.same_candle_priority,
            )
            if hit is ExitHit.SL:
                trades.append(
                    _close_trade(
                        account,
                        account.position,
                        exit_timestamp=timestamp,
                        exit_reason=EXIT_SL,
                        exit_reference=account.position.stop_loss,
                        config=config,
                    )
                )
            elif hit is ExitHit.TP:
                trades.append(
                    _close_trade(
                        account,
                        account.position,
                        exit_timestamp=timestamp,
                        exit_reason=EXIT_TP,
                        exit_reference=account.position.take_profit,
                        config=config,
                    )
                )

        signal_value = signals["signal"].iloc[index]
        if _is_long_entry(signal_value):
            signals_received += 1
            if account.position is not None:
                ignored.append(
                    IgnoredSignal(signal_timestamp=timestamp, reason=IGNORE_POSITION_OPEN)
                )
            elif index == n - 1:
                ignored.append(
                    IgnoredSignal(
                        signal_timestamp=timestamp, reason=IGNORE_NO_NEXT_CANDLE
                    )
                )
            else:
                next_time = _as_utc(
                    candles["timestamp"].iloc[index + 1], "timestamp"
                )
                account.pending = PendingEntry(
                    signal_timestamp=timestamp,
                    execute_at=next_time,
                )

    unresolved = 0
    last_close: float | None = None
    unrealized: float | None = None
    informational: float | None = None
    if n:
        last_close = _as_float(candles["close"].iloc[-1], "close")
    if account.position is not None:
        unresolved = 1
        trades.append(_open_end_of_data(account.position))
        if last_close is not None:
            unrealized = account.position.quantity * last_close
            informational = account.cash + unrealized

    if account.pending is not None:
        raise BacktestInputError("Pending entry survived the end of the dataset")

    return BacktestResult(
        starting_capital=float(config.starting_capital),
        cash=account.cash,
        realized_net_pnl=account.realized_net_pnl,
        min_cash=account.min_cash,
        candles_processed=n,
        signals_received=signals_received,
        entries_executed=entries_executed,
        trades_closed=sum(1 for trade in trades if trade.status == STATUS_CLOSED),
        unresolved_count=unresolved,
        last_close=last_close,
        unrealized_mark_value=unrealized,
        informational_equity=informational,
        pending_cleared=True,
        trades=trades,
        ignored=ignored,
    )


def empty_trade_frame() -> pd.DataFrame:
    return pd.DataFrame({column: [] for column in TRADE_FRAME_COLUMNS})
