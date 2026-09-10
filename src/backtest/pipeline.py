"""Read enriched store, consume v0.3 signals, run the backtest, write reports.

Never writes under data/processed or data/enriched. STATUS PASS means the
simulator matched v0.4 integrity rules, not that the strategy is profitable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data.pipeline import DataConfig
from src.indicators.engine import IndicatorConfig
from src.indicators.pipeline import calendar_month_key, enriched_dataset_dir
from src.strategy.engine import StrategyConfig, StrategyInputError, generate_signals
from src.strategy.pipeline import read_enriched
from src.strategy.signals import SignalType

from .config import BacktestConfig, BacktestInputError
from .engine import BacktestResult, run_backtest
from .ledger import (
    EXIT_END_OF_DATA,
    EXIT_SL,
    EXIT_TP,
    IGNORE_INSUFFICIENT_CASH,
    IGNORE_NO_NEXT_CANDLE,
    IGNORE_POSITION_OPEN,
    STATUS_CLOSED,
    STATUS_OPEN,
    TRADE_FRAME_COLUMNS,
)

ProgressCallback = Callable[[str], None]

REPORT_NAME_TEMPLATE = "backtest_validation_{symbol}_{interval}.txt"
TRADES_CSV_TEMPLATE = "backtest_trades_{symbol}_{interval}.csv"
IGNORED_CSV_TEMPLATE = "backtest_ignored_signals_{symbol}_{interval}.csv"
_RULE = "-" * 78
_RECONCILE_EPS = 1e-8


class BacktestEvaluationError(RuntimeError):
    """The backtest pipeline could not complete."""


@dataclass
class BacktestReport:
    symbol: str
    market: str
    interval: str
    start_date: object
    end_date: object
    strategy_name: str
    strategy_version: str
    starting_capital: float
    risk_per_trade: float
    stop_loss: float
    take_profit: float
    entry_rate: float
    exit_rate: float
    slippage: float
    entry_model: str
    same_candle_priority: str
    candles_processed: int = 0
    signals_received: int = 0
    entries_executed: int = 0
    trades_closed: int = 0
    unresolved_count: int = 0
    ignored_position_open: int = 0
    ignored_no_next_candle: int = 0
    ignored_insufficient_cash: int = 0
    exit_sl: int = 0
    exit_tp: int = 0
    exit_end_of_data: int = 0
    total_entry_fees: float = 0.0
    total_exit_fees: float = 0.0
    total_slippage: float = 0.0
    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0
    final_cash: float = 0.0
    realized_net_pnl: float = 0.0
    unrealized_mark_value: float | None = None
    informational_equity: float | None = None
    input_not_mutated: bool = False
    deterministic: bool = False
    no_negative_cash: bool = False
    identities_ok: bool = False
    cash_reconciles: bool = False
    realized_excludes_open: bool = False
    no_time_exit: bool = False
    last_bar_no_phantom_fill: bool = False
    pending_cleared: bool = False
    findings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class BacktestPipelineResult:
    data_config: DataConfig
    indicator_config: IndicatorConfig
    strategy_config: StrategyConfig
    backtest_config: BacktestConfig
    report: BacktestReport
    first: BacktestResult
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def passed(self) -> bool:
        return self.report.passed

    @property
    def status(self) -> str:
        return self.report.status

    @property
    def report_path(self) -> Path:
        return self.data_config.reports_dir / REPORT_NAME_TEMPLATE.format(
            symbol=self.data_config.symbol,
            interval=self.data_config.interval,
        )

    @property
    def trades_csv_path(self) -> Path:
        return self.data_config.reports_dir / TRADES_CSV_TEMPLATE.format(
            symbol=self.data_config.symbol,
            interval=self.data_config.interval,
        )

    @property
    def ignored_csv_path(self) -> Path:
        return self.data_config.reports_dir / IGNORED_CSV_TEMPLATE.format(
            symbol=self.data_config.symbol,
            interval=self.data_config.interval,
        )


def _closed(trades: list) -> list:
    return [trade for trade in trades if trade.status == STATUS_CLOSED]


def _is_close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=_RECONCILE_EPS, equal_nan=False))


def _validate(
    original_candles: pd.DataFrame,
    candles_after: pd.DataFrame,
    first: BacktestResult,
    second: BacktestResult,
    data_config: DataConfig,
    strategy_config: StrategyConfig,
    backtest_config: BacktestConfig,
    signals: pd.DataFrame,
) -> BacktestReport:
    report = BacktestReport(
        symbol=data_config.symbol,
        market=data_config.market,
        interval=data_config.interval,
        start_date=data_config.start_date,
        end_date=data_config.end_date,
        strategy_name=strategy_config.name,
        strategy_version=strategy_config.version,
        starting_capital=backtest_config.starting_capital,
        risk_per_trade=backtest_config.risk_per_trade,
        stop_loss=backtest_config.stop_loss,
        take_profit=backtest_config.take_profit,
        entry_rate=backtest_config.entry_rate,
        exit_rate=backtest_config.exit_rate,
        slippage=backtest_config.slippage,
        entry_model=backtest_config.entry_model,
        same_candle_priority=backtest_config.same_candle_priority,
        candles_processed=first.candles_processed,
        signals_received=first.signals_received,
        entries_executed=first.entries_executed,
        trades_closed=first.trades_closed,
        unresolved_count=first.unresolved_count,
        final_cash=first.cash,
        realized_net_pnl=first.realized_net_pnl,
        unrealized_mark_value=first.unrealized_mark_value,
        informational_equity=first.informational_equity,
        pending_cleared=first.pending_cleared,
    )

    report.input_not_mutated = bool(original_candles.equals(candles_after))
    if not report.input_not_mutated:
        report.findings.append("Backtest mutated the input candle frame")

    report.deterministic = bool(
        first.trade_frame.equals(second.trade_frame)
        and first.ignored_frame.equals(second.ignored_frame)
        and first.cash == second.cash
        and first.realized_net_pnl == second.realized_net_pnl
    )
    if not report.deterministic:
        report.findings.append("Second backtest pass did not match the first")

    if first.candles_processed != len(original_candles):
        report.findings.append(
            f"Candles processed {first.candles_processed} != input {len(original_candles)}"
        )

    long_count = int((signals["signal"] == SignalType.LONG_ENTRY.value).sum())
    if first.signals_received != long_count:
        report.findings.append(
            f"Signals received {first.signals_received} != LONG_ENTRY rows {long_count}"
        )

    report.no_negative_cash = first.min_cash >= -_RECONCILE_EPS
    if not report.no_negative_cash:
        report.findings.append(f"Cash went negative (min {first.min_cash})")

    if not first.pending_cleared:
        report.findings.append("Pending entry was not cleared at end of data")

    reasons = [item.reason for item in first.ignored]
    report.ignored_position_open = reasons.count(IGNORE_POSITION_OPEN)
    report.ignored_no_next_candle = reasons.count(IGNORE_NO_NEXT_CANDLE)
    report.ignored_insufficient_cash = reasons.count(IGNORE_INSUFFICIENT_CASH)

    report.exit_sl = sum(1 for trade in first.trades if trade.exit_reason == EXIT_SL)
    report.exit_tp = sum(1 for trade in first.trades if trade.exit_reason == EXIT_TP)
    report.exit_end_of_data = sum(
        1 for trade in first.trades if trade.exit_reason == EXIT_END_OF_DATA
    )
    report.no_time_exit = all(
        trade.exit_reason in {EXIT_SL, EXIT_TP, EXIT_END_OF_DATA} for trade in first.trades
    )
    if not report.no_time_exit:
        report.findings.append("TIME_EXIT or unknown exit_reason present")

    closed = _closed(first.trades)
    identities_ok = True
    for trade in closed:
        if (
            trade.net_pnl is None
            or trade.gross_pnl is None
            or trade.slippage is None
            or trade.exit_fee is None
            or trade.R is None
        ):
            identities_ok = False
            break
        reconstructed = (
            trade.gross_pnl - trade.slippage - trade.entry_fee - trade.exit_fee
        )
        if not _is_close(trade.net_pnl, reconstructed):
            identities_ok = False
            break
        if not _is_close(trade.R, trade.net_pnl / trade.risk_amount):
            identities_ok = False
            break
    report.identities_ok = identities_ok and all(
        trade.net_pnl is not None for trade in closed
    )
    if closed and not report.identities_ok:
        report.findings.append("Closed-trade P&L identity failed")

    report.total_entry_fees = float(sum(trade.entry_fee for trade in first.trades))
    report.total_exit_fees = float(
        sum(trade.exit_fee or 0.0 for trade in closed)
    )
    report.total_slippage = float(sum(trade.slippage or 0.0 for trade in closed))
    report.total_gross_pnl = float(sum(trade.gross_pnl or 0.0 for trade in closed))
    report.total_net_pnl = float(sum(trade.net_pnl or 0.0 for trade in closed))

    closed_net_sum = report.total_net_pnl
    report.realized_excludes_open = _is_close(first.realized_net_pnl, closed_net_sum)
    if not report.realized_excludes_open:
        report.findings.append("realized_net_pnl is not the sum of CLOSED net_pnl")

    open_rows = [trade for trade in first.trades if trade.status == STATUS_OPEN]
    if len(open_rows) != first.unresolved_count:
        report.findings.append("OPEN trade count does not match unresolved_count")
    for trade in open_rows:
        if trade.exit_reason != EXIT_END_OF_DATA:
            report.findings.append("OPEN trade missing END_OF_DATA reason")
        none_fields = (
            trade.exit_timestamp,
            trade.exit_reference_price,
            trade.exit_price,
            trade.exit_fee,
            trade.exit_slippage,
            trade.slippage,
            trade.gross_pnl,
            trade.net_pnl,
            trade.R,
            trade.holding_time,
        )
        if any(field is not None for field in none_fields):
            report.findings.append("OPEN / END_OF_DATA row has realized exit fields set")

    reserved = 0.0
    if open_rows:
        reserved = open_rows[0].position_value + open_rows[0].entry_fee
    expected_cash = (
        backtest_config.starting_capital + first.realized_net_pnl - reserved
    )
    report.cash_reconciles = _is_close(first.cash, expected_cash)
    if not report.cash_reconciles:
        report.findings.append(
            f"Cash {first.cash} does not reconcile to {expected_cash}"
        )

    report.last_bar_no_phantom_fill = True
    if not original_candles.empty:
        last_ts = pd.Timestamp(original_candles["timestamp"].iloc[-1])
        last_signal = signals["signal"].iloc[-1]
        if last_signal == SignalType.LONG_ENTRY.value:
            phantom = any(
                trade.entry_timestamp > last_ts for trade in first.trades
            )
            report.last_bar_no_phantom_fill = not phantom
            if phantom:
                report.findings.append("Final-candle signal produced a phantom fill")

    unexpected = [
        column for column in first.trade_frame.columns if column not in TRADE_FRAME_COLUMNS
    ]
    if unexpected:
        report.findings.append(
            "Unexpected trade column(s): " + ", ".join(unexpected)
        )

    if first.entries_executed != len(first.trades):
        report.findings.append(
            f"Entries executed {first.entries_executed} != ledger rows {len(first.trades)}"
        )

    months = calendar_month_key(original_candles["timestamp"]).drop_duplicates()
    if len(months) < 2:
        report.findings.append(
            "Need at least two calendar months to verify concatenated replay"
        )

    return report


def _write_trades_csv(trades: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = trades.copy()
    if "holding_time" in frame.columns:
        frame["holding_time"] = frame["holding_time"].map(
            lambda value: "" if value is None or pd.isna(value) else str(value)
        )
    frame.to_csv(path, index=False)


def run_backtest_pipeline(
    data_config: DataConfig,
    indicator_config: IndicatorConfig,
    strategy_config: StrategyConfig,
    backtest_config: BacktestConfig,
    progress: ProgressCallback | None = None,
) -> BacktestPipelineResult:
    def log(message: str) -> None:
        if progress is not None:
            progress(message)

    dest_dir = enriched_dataset_dir(data_config, indicator_config)

    log("Step 1/5  Reading enriched Parquet (read-only)")
    candles = read_enriched(dest_dir)
    original = candles.copy(deep=True)

    log("Step 2/5  Generating v0.3 signals on the concatenated series")
    try:
        signals = generate_signals(
            candles,
            strategy_config,
            indicator_config,
            interval=data_config.interval,
            symbol=data_config.symbol,
        )
    except StrategyInputError as error:
        raise BacktestEvaluationError(str(error)) from error

    log("Step 3/5  Running sequential backtest")
    try:
        first = run_backtest(
            candles,
            signals,
            backtest_config,
            symbol=data_config.symbol,
            interval=data_config.interval,
        )
        second = run_backtest(
            candles,
            signals,
            backtest_config,
            symbol=data_config.symbol,
            interval=data_config.interval,
        )
    except BacktestInputError as error:
        raise BacktestEvaluationError(str(error)) from error

    log("Step 4/5  Checking integrity (immutability, determinism, accounting)")
    report = _validate(
        original,
        candles,
        first,
        second,
        data_config,
        strategy_config,
        backtest_config,
        signals,
    )
    result = BacktestPipelineResult(
        data_config=data_config,
        indicator_config=indicator_config,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        report=report,
        first=first,
        signals=signals,
    )

    log("Step 5/5  Writing validation report and CSVs")
    result.report_path.parent.mkdir(parents=True, exist_ok=True)
    result.report_path.write_text(render_report(result), encoding="utf-8")
    _write_trades_csv(first.trade_frame, result.trades_csv_path)
    first.ignored_frame.to_csv(result.ignored_csv_path, index=False)
    return result


def render_report(result: BacktestPipelineResult) -> str:
    report = result.report
    lines: list[str] = [
        "=" * 78,
        "Backtest engine validation report",
        "=" * 78,
        "",
        "STATUS: PASS means the simulator matched the specified v0.4",
        "implementation/integrity rules and passed its validation tests.",
        "It does NOT mean the strategy is profitable, has positive",
        "expectancy, is statistically significant, is suitable for live",
        "trading, should be deployed, or has been optimized.",
        "",
        f"Symbol:              {report.symbol}",
        f"Market:              {str(report.market).capitalize()}",
        f"Interval:            {report.interval}",
        f"Period:              {report.start_date} .. {report.end_date} "
        "(UTC, end date inclusive)",
        f"Generated:           {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "Strategy (consumed unchanged from v0.3)",
        _RULE,
        f"Name:                {report.strategy_name}",
        f"Version:             {report.strategy_version}",
        "",
        "Experimental execution assumptions (not live account facts)",
        _RULE,
        f"Starting capital:    {report.starting_capital}",
        f"Risk per trade:      {report.risk_per_trade}",
        f"Stop loss:           {report.stop_loss}",
        f"Take profit:         {report.take_profit}",
        f"Entry model:         {report.entry_model}",
        f"Slippage:            {report.slippage}",
        f"Same-candle:         {report.same_candle_priority}",
        f"Entry fee rate:      {report.entry_rate}",
        f"Exit fee rate:       {report.exit_rate}",
        "",
        "Counts",
        _RULE,
        f"Candles processed:   {report.candles_processed}",
        f"Signals received:    {report.signals_received}",
        f"Entries executed:    {report.entries_executed}",
        f"Trades closed:       {report.trades_closed}",
        f"Unresolved (OPEN):   {report.unresolved_count}",
        f"Ignored POSITION_OPEN:       {report.ignored_position_open}",
        f"Ignored NO_NEXT_CANDLE:      {report.ignored_no_next_candle}",
        f"Ignored INSUFFICIENT_CASH:   {report.ignored_insufficient_cash}",
        "",
        "Exit reasons (closed + unresolved)",
        _RULE,
        f"SL:                  {report.exit_sl}",
        f"TP:                  {report.exit_tp}",
        f"END_OF_DATA:         {report.exit_end_of_data}",
        "",
        "Totals",
        _RULE,
        f"Total entry fees:    {report.total_entry_fees}",
        f"Total exit fees:     {report.total_exit_fees}",
        f"Total slippage:      {report.total_slippage}",
        f"Total gross P&L:     {report.total_gross_pnl}",
        f"Total net P&L:       {report.total_net_pnl}",
        f"realized_net_pnl:    {report.realized_net_pnl}",
        f"Final cash:          {report.final_cash}",
        "",
        "Informational / unrealized (not a fill, not realized)",
        _RULE,
        f"unrealized_mark_value:  {_fmt_optional(report.unrealized_mark_value)}",
        f"informational_equity:   {_fmt_optional(report.informational_equity)}",
        "",
        "Integrity",
        _RULE,
        f"Input not mutated:   {_yn(report.input_not_mutated)}",
        f"Deterministic:       {_yn(report.deterministic)}",
        f"No negative cash:    {_yn(report.no_negative_cash)}",
        f"P&L identities:      {_yn(report.identities_ok)}",
        f"Cash reconciles:     {_yn(report.cash_reconciles)}",
        f"Realized excludes OPEN: {_yn(report.realized_excludes_open)}",
        f"No TIME_EXIT:        {_yn(report.no_time_exit)}",
        f"Last-bar no phantom: {_yn(report.last_bar_no_phantom_fill)}",
        f"Pending cleared:     {_yn(report.pending_cleared)}",
        "",
        "Findings",
        _RULE,
    ]
    if report.findings:
        for finding in report.findings:
            lines.append(f"- {finding}")
    else:
        lines.append("No problems detected.")
    lines.extend(
        [
            "",
            "=" * 78,
            f"STATUS: {report.status}",
            "=" * 78,
            "",
            "STATUS: PASS is simulator integrity only. It is not a claim that",
            "the strategy is profitable or ready for live trading.",
            "",
        ]
    )
    return "\n".join(lines)


def render_console_summary(result: BacktestPipelineResult) -> str:
    report = result.report
    lines = [
        "",
        f"{report.symbol} {report.interval} backtest  "
        f"{report.start_date} .. {report.end_date}",
        f"  strategy           {report.strategy_name} {report.strategy_version}",
        f"  candles            {report.candles_processed}",
        f"  signals            {report.signals_received}",
        f"  entries            {report.entries_executed}",
        f"  closed             {report.trades_closed}",
        f"  unresolved         {report.unresolved_count}",
        f"  realized_net_pnl   {report.realized_net_pnl}",
        f"  final cash         {report.final_cash}",
        f"  report             {result.report_path}",
        f"  trades CSV         {result.trades_csv_path}",
        f"  ignored CSV        {result.ignored_csv_path}",
        "",
        "  STATUS: PASS means simulator integrity, not profitability.",
        "",
    ]
    for finding in report.findings:
        lines.append(f"  [FAIL] {finding}")
    if report.findings:
        lines.append("")
    lines.append(f"STATUS: {report.status}")
    return "\n".join(lines)


def _yn(value: bool) -> str:
    return "OK" if value else "FAIL"


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a (no open position)"
    return str(value)
