"""Read the v0.2 enriched store, evaluate signals, write a validation report.

Monthly Parquet files are concatenated in timestamp order before evaluation.
Month boundaries are storage partitions, not strategy resets. The pipeline
never writes under data/processed or data/enriched.

The report is an integrity check: counts, warm-up, condition failures,
determinism, and timing labels. It does not compute P&L or simulated fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.data.normalizer import interval_to_timedelta
from src.data.pipeline import DataConfig
from src.indicators.engine import IndicatorConfig
from src.indicators.pipeline import calendar_month_key, enriched_dataset_dir

from .engine import (
    REQUIRED_OHLCV,
    SIGNAL_FRAME_COLUMNS,
    StrategyConfig,
    StrategyInputError,
    generate_signals,
)
from .signals import SignalType

ProgressCallback = Callable[[str], None]

REPORT_NAME_TEMPLATE = "strategy_validation_{symbol}_{interval}.txt"
SIGNALS_CSV_TEMPLATE = "strategy_signals_{symbol}_{interval}.csv"
_RULE = "-" * 78
_CSV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "earliest_execution_time",
    "reference_close",
    "breakout_level",
    "ema_trend",
    "breakout",
    "volume_confirmation",
    "rsi_filter",
)


class StrategyEvaluationError(RuntimeError):
    """The strategy evaluation pipeline could not complete."""


@dataclass
class ContinuityCheck:
    """January history vs February-only reset on the first two calendar months."""

    first_month: str
    second_month: str
    carried_forward: bool
    sample_full_series: float | None
    sample_month_only: float | None

    @property
    def passed(self) -> bool:
        return self.carried_forward


@dataclass
class StrategyReport:
    symbol: str
    market: str
    interval: str
    start_date: object
    end_date: object
    strategy_name: str
    strategy_version: str
    breakout_period: int
    volume_multiplier: float
    rsi_min: float
    rsi_max: float
    indicator_columns: tuple[str, ...]
    input_rows: int = 0
    evaluated_rows: int = 0
    warmup_rows: int = 0
    long_entry_count: int = 0
    first_long_entry: pd.Timestamp | None = None
    last_long_entry: pd.Timestamp | None = None
    fail_ema_trend: int = 0
    fail_breakout: int = 0
    fail_volume: int = 0
    fail_rsi: int = 0
    input_not_mutated: bool = False
    deterministic: bool = False
    last_bar_execution_outside_dataset: bool = False
    last_earliest_execution_time: pd.Timestamp | None = None
    continuity: ContinuityCheck | None = None
    findings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class StrategyResult:
    data_config: DataConfig
    indicator_config: IndicatorConfig
    strategy_config: StrategyConfig
    report: StrategyReport
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def passed(self) -> bool:
        return self.report.passed

    @property
    def status(self) -> str:
        return self.report.status

    @property
    def report_path(self) -> Path:
        return (
            self.data_config.reports_dir
            / REPORT_NAME_TEMPLATE.format(
                symbol=self.data_config.symbol,
                interval=self.data_config.interval,
            )
        )

    @property
    def signals_csv_path(self) -> Path:
        return (
            self.data_config.reports_dir
            / SIGNALS_CSV_TEMPLATE.format(
                symbol=self.data_config.symbol,
                interval=self.data_config.interval,
            )
        )


def list_enriched_parquets(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.is_dir():
        raise StrategyEvaluationError(f"Enriched store is missing: {dataset_dir}")
    paths = sorted(dataset_dir.glob("*.parquet"))
    if not paths:
        raise StrategyEvaluationError(f"No enriched Parquet files in {dataset_dir}")
    return paths


def read_enriched(dataset_dir: Path) -> pd.DataFrame:
    """Concatenate monthly enriched Parquet in filename order (chronological)."""
    frames = [
        pd.read_parquet(path, engine="pyarrow")
        for path in list_enriched_parquets(dataset_dir)
    ]
    return pd.concat(frames, ignore_index=True)


def _none_mask(series: pd.Series) -> pd.Series:
    return series.map(_is_none)


def _is_none(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_true(value: object) -> bool:
    if _is_none(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return False


def _is_false(value: object) -> bool:
    if _is_none(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    return False


def _continuity_check(
    frame: pd.DataFrame,
    concatenated_signals: pd.DataFrame,
    strategy_config: StrategyConfig,
    indicator_config: IndicatorConfig,
    *,
    interval: str,
    symbol: str,
) -> ContinuityCheck | None:
    keys = calendar_month_key(frame["timestamp"])
    months = keys.drop_duplicates().tolist()
    if len(months) < 2:
        return None

    first_month, second_month = str(months[0]), str(months[1])
    first_len = int((keys == first_month).sum())
    february = frame.loc[keys == second_month].reset_index(drop=True)
    february_only = generate_signals(
        february, strategy_config, indicator_config, interval=interval, symbol=symbol
    )
    full_level = concatenated_signals["breakout_level"].iloc[first_len]
    only_level = february_only["breakout_level"].iloc[0]
    full_value = None if pd.isna(full_level) else float(full_level)
    only_value = None if pd.isna(only_level) else float(only_level)
    carried = full_value != only_value
    return ContinuityCheck(
        first_month=first_month,
        second_month=second_month,
        carried_forward=bool(carried),
        sample_full_series=full_value,
        sample_month_only=only_value,
    )


def _validate(
    original: pd.DataFrame,
    after: pd.DataFrame,
    first_pass: pd.DataFrame,
    second_pass: pd.DataFrame,
    strategy_config: StrategyConfig,
    indicator_config: IndicatorConfig,
    data_config: DataConfig,
) -> StrategyReport:
    report = StrategyReport(
        symbol=data_config.symbol,
        market=data_config.market,
        interval=data_config.interval,
        start_date=data_config.start_date,
        end_date=data_config.end_date,
        strategy_name=strategy_config.name,
        strategy_version=strategy_config.version,
        breakout_period=strategy_config.breakout_period,
        volume_multiplier=strategy_config.volume_multiplier,
        rsi_min=strategy_config.rsi_min,
        rsi_max=strategy_config.rsi_max,
        indicator_columns=indicator_config.indicator_columns,
        input_rows=len(original),
        evaluated_rows=len(first_pass),
    )

    if first_pass.empty and not original.empty:
        report.findings.append("Signal frame is empty but the input was not")
        return report

    if original.empty:
        report.findings.append("Enriched dataset is empty")
        return report

    report.input_not_mutated = bool(original.equals(after))
    if not report.input_not_mutated:
        report.findings.append("Strategy evaluation mutated the input frame")

    report.deterministic = bool(first_pass.equals(second_pass))
    if not report.deterministic:
        report.findings.append("Second evaluation pass did not match the first")

    if len(first_pass) != len(original):
        report.findings.append(
            f"Row count changed: input {len(original)} vs signals {len(first_pass)}"
        )

    forbidden = {"execution_price", "fill_price", "entry_price"}
    present = forbidden.intersection(first_pass.columns)
    if present:
        report.findings.append(
            "Signal frame contains execution-like column(s): " + ", ".join(sorted(present))
        )

    unexpected = [column for column in first_pass.columns if column not in SIGNAL_FRAME_COLUMNS]
    if unexpected:
        report.findings.append(
            "Unexpected signal column(s): " + ", ".join(unexpected)
        )

    warmup = first_pass["warmup"].to_numpy(dtype=bool)
    report.warmup_rows = int(warmup.sum())
    long_mask = (first_pass["signal"] == SignalType.LONG_ENTRY.value).to_numpy(dtype=bool)
    if bool((warmup & long_mask).any()):
        report.findings.append("LONG_ENTRY was emitted during warm-up")

    unevaluable = (
        _none_mask(first_pass["ema_trend"])
        | _none_mask(first_pass["breakout"])
        | _none_mask(first_pass["volume_confirmation"])
        | _none_mask(first_pass["rsi_filter"])
    ).to_numpy(dtype=bool)
    if not np.array_equal(warmup, unevaluable):
        report.findings.append("warmup flag does not match unevaluable conditions")

    all_true = (
        first_pass["ema_trend"].map(_is_true)
        & first_pass["breakout"].map(_is_true)
        & first_pass["volume_confirmation"].map(_is_true)
        & first_pass["rsi_filter"].map(_is_true)
    ).to_numpy(dtype=bool)
    if not np.array_equal(long_mask, all_true):
        report.findings.append("LONG_ENTRY does not match AND of all four conditions")

    ready = ~warmup
    report.fail_ema_trend = int(
        (ready & first_pass["ema_trend"].map(_is_false).to_numpy(dtype=bool)).sum()
    )
    report.fail_breakout = int(
        (ready & first_pass["breakout"].map(_is_false).to_numpy(dtype=bool)).sum()
    )
    report.fail_volume = int(
        (
            ready
            & first_pass["volume_confirmation"].map(_is_false).to_numpy(dtype=bool)
        ).sum()
    )
    report.fail_rsi = int(
        (ready & first_pass["rsi_filter"].map(_is_false).to_numpy(dtype=bool)).sum()
    )
    report.long_entry_count = int(long_mask.sum())
    if report.long_entry_count:
        entries = first_pass.loc[long_mask, "timestamp"]
        report.first_long_entry = pd.Timestamp(entries.iloc[0])
        report.last_long_entry = pd.Timestamp(entries.iloc[-1])

    step = interval_to_timedelta(data_config.interval)
    expected_exec = first_pass["timestamp"] + step
    exec_ok = first_pass["earliest_execution_time"].eq(expected_exec).all()
    if not bool(exec_ok):
        report.findings.append(
            "earliest_execution_time is not timestamp + interval on every row"
        )

    last_exec = pd.Timestamp(first_pass["earliest_execution_time"].iloc[-1])
    report.last_earliest_execution_time = last_exec
    input_times = set(pd.Timestamp(value) for value in original["timestamp"])
    report.last_bar_execution_outside_dataset = last_exec not in input_times

    tz = getattr(first_pass["timestamp"].dtype, "tz", None)
    if str(tz) != "UTC":
        report.findings.append(
            f"signal timestamp dtype must be datetime64[ns, UTC], got {first_pass['timestamp'].dtype}"
        )

    continuity = _continuity_check(
        original,
        first_pass,
        strategy_config,
        indicator_config,
        interval=data_config.interval,
        symbol=data_config.symbol,
    )
    report.continuity = continuity
    if continuity is None:
        report.findings.append(
            "Need at least two calendar months to verify breakout continuity"
        )
    elif not continuity.carried_forward:
        report.findings.append(
            f"Month-boundary reset detected: breakout_level at {continuity.second_month} "
            f"matches a {continuity.second_month}-only recalculation "
            f"({continuity.sample_month_only})"
        )

    return report


def _write_signals_csv(signals: pd.DataFrame, path: Path) -> None:
    entries = signals.loc[signals["signal"] == SignalType.LONG_ENTRY.value, list(_CSV_COLUMNS)]
    path.parent.mkdir(parents=True, exist_ok=True)
    entries.to_csv(path, index=False)


def run_strategy_evaluation(
    data_config: DataConfig,
    indicator_config: IndicatorConfig,
    strategy_config: StrategyConfig,
    progress: ProgressCallback | None = None,
) -> StrategyResult:
    def log(message: str) -> None:
        if progress is not None:
            progress(message)

    dest_dir = enriched_dataset_dir(data_config, indicator_config)

    log("Step 1/4  Reading enriched Parquet")
    enriched = read_enriched(dest_dir)
    original = enriched.copy(deep=True)

    log("Step 2/4  Evaluating strategy on the concatenated chronological series")
    try:
        first_pass = generate_signals(
            enriched,
            strategy_config,
            indicator_config,
            interval=data_config.interval,
            symbol=data_config.symbol,
        )
        second_pass = generate_signals(
            enriched,
            strategy_config,
            indicator_config,
            interval=data_config.interval,
            symbol=data_config.symbol,
        )
    except StrategyInputError as error:
        raise StrategyEvaluationError(str(error)) from error

    log("Step 3/4  Checking integrity (immutability, determinism, continuity)")
    report = _validate(
        original,
        enriched,
        first_pass,
        second_pass,
        strategy_config,
        indicator_config,
        data_config,
    )

    result = StrategyResult(
        data_config=data_config,
        indicator_config=indicator_config,
        strategy_config=strategy_config,
        report=report,
        signals=first_pass,
    )

    log("Step 4/4  Writing validation report and LONG_ENTRY CSV")
    result.report_path.parent.mkdir(parents=True, exist_ok=True)
    result.report_path.write_text(render_report(result), encoding="utf-8")
    _write_signals_csv(first_pass, result.signals_csv_path)
    return result


def render_report(result: StrategyResult) -> str:
    report = result.report
    config = result.strategy_config
    lines: list[str] = [
        "=" * 78,
        "Strategy engine validation report",
        "=" * 78,
        "",
        f"Symbol:              {report.symbol}",
        f"Market:              {str(report.market).capitalize()}",
        f"Interval:            {report.interval}",
        f"Period:              {report.start_date} .. {report.end_date} "
        "(UTC, end date inclusive)",
        f"Generated:           {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "Strategy",
        _RULE,
        f"Name:                {config.name}",
        f"Version:             {config.version}",
        f"Breakout period:     {config.breakout_period}  "
        "(independent of volume_ma_period)",
        f"Volume multiplier:   {config.volume_multiplier}",
        f"RSI filter:          {config.rsi_min} < rsi < {config.rsi_max}",
        f"Indicator columns:   {', '.join(report.indicator_columns)}",
        "",
        "Counts",
        _RULE,
        f"Candles evaluated:   {report.evaluated_rows}",
        f"Warm-up rows:        {report.warmup_rows}",
        f"LONG_ENTRY signals:  {report.long_entry_count}",
        f"First LONG_ENTRY:    {_fmt(report.first_long_entry)}",
        f"Last LONG_ENTRY:     {_fmt(report.last_long_entry)}",
        "",
        "Condition failures (non-warm-up rows, counted independently)",
        _RULE,
        f"ema_trend false:     {report.fail_ema_trend}",
        f"breakout false:      {report.fail_breakout}",
        f"volume false:        {report.fail_volume}",
        f"rsi_filter false:    {report.fail_rsi}",
        "",
        "Timing",
        _RULE,
        "Signals are evaluated after candle close. reference_close is an",
        "audit value, not a fill. earliest_execution_time is the next",
        "interval boundary — a timing label only. v0.3 does not execute.",
        f"Last-bar label:      {_fmt(report.last_earliest_execution_time)}",
        f"Last-bar outside dataset: {_yn(report.last_bar_execution_outside_dataset)}",
        "",
        "Integrity",
        _RULE,
        f"Input not mutated:   {_yn(report.input_not_mutated)}",
        f"Deterministic:       {_yn(report.deterministic)}",
        f"Required OHLCV:      {', '.join(REQUIRED_OHLCV)}",
        "",
        "Month-boundary continuity",
        _RULE,
        "Breakout history uses previous rows in series order. Calendar",
        "months are storage boundaries, not strategy resets.",
    ]
    continuity = report.continuity
    if continuity is None:
        lines.append("Status:              not checked (fewer than two months)")
    else:
        lines.append(
            f"Months compared:     {continuity.first_month} -> {continuity.second_month}"
        )
        lines.append(
            f"Carried forward:     {_yn(continuity.carried_forward)} "
            f"(breakout_level full-series {continuity.sample_full_series} vs "
            f"month-only {continuity.sample_month_only})"
        )

    lines.extend(["", "Findings", _RULE])
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
        ]
    )
    return "\n".join(lines)


def render_console_summary(result: StrategyResult) -> str:
    report = result.report
    lines = [
        "",
        f"{report.symbol} {report.interval} strategy  "
        f"{report.start_date} .. {report.end_date}",
        f"  strategy           {report.strategy_name} {report.strategy_version}",
        f"  candles            {report.evaluated_rows}",
        f"  warm-up            {report.warmup_rows}",
        f"  LONG_ENTRY         {report.long_entry_count}",
        f"  first signal       {_fmt(report.first_long_entry)}",
        f"  last signal        {_fmt(report.last_long_entry)}",
        f"  input not mutated  {_yn(report.input_not_mutated)}",
        f"  deterministic      {_yn(report.deterministic)}",
        f"  report             {result.report_path}",
        f"  signals CSV        {result.signals_csv_path}",
        "",
    ]
    if report.continuity is not None:
        lines.append(
            f"  continuity         {_yn(report.continuity.passed)} "
            f"({report.continuity.first_month} -> {report.continuity.second_month})"
        )
        lines.append("")
    for finding in report.findings:
        lines.append(f"  [FAIL] {finding}")
    if report.findings:
        lines.append("")
    lines.append(f"STATUS: {report.status}")
    return "\n".join(lines)


def _fmt(timestamp: pd.Timestamp | None) -> str:
    if timestamp is None or pd.isna(timestamp):
        return "n/a"
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")


def _yn(value: bool) -> str:
    return "OK" if value else "FAIL"
