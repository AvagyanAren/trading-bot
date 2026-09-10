"""Read processed Parquet, enrich the full series, write monthly overlays.

Month boundaries are a storage partition. Indicator values are computed once
on the concatenated chronological dataset; timestamps are then used only to
choose which file each already-computed row is written into.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from src.data.pipeline import PARQUET_NAME_TEMPLATE, DataConfig

from .engine import (
    REQUIRED_COLUMNS,
    IndicatorConfig,
    IndicatorInputError,
    calculate_indicators,
)

ProgressCallback = Callable[[str], None]

REPORT_NAME_TEMPLATE = "indicator_validation_{symbol}_{interval}.txt"
OHLCV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
_RULE = "-" * 78


class EnrichmentError(RuntimeError):
    """The enrichment pipeline could not complete."""


@dataclass
class ContinuityCheck:
    """January-state vs February-only reset on the first two calendar months."""

    first_month: str
    second_month: str
    carried_forward: bool
    matches_full_series: bool
    sample_column: str
    full_series_value: float | None
    february_only_value: float | None
    enriched_value: float | None

    @property
    def passed(self) -> bool:
        return self.carried_forward and self.matches_full_series


@dataclass
class EnrichmentReport:
    symbol: str
    market: str
    interval: str
    start_date: object
    end_date: object
    input_rows: int = 0
    output_rows: int = 0
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    ema_fast: int = 0
    ema_slow: int = 0
    rsi_period: int = 0
    volume_ma_period: int = 0
    nan_counts: dict[str, int] = field(default_factory=dict)
    warmup_rows: dict[str, int] = field(default_factory=dict)
    ohlcv_preserved: bool = False
    timestamps_preserved: bool = False
    extra_columns_preserved: bool = False
    duplicates: int = 0
    dtypes_ok: bool = False
    continuity: ContinuityCheck | None = None
    findings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class EnrichmentResult:
    data_config: DataConfig
    indicator_config: IndicatorConfig
    report: EnrichmentReport
    parquet_paths: list[Path] = field(default_factory=list)

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


def calendar_month_key(timestamps: pd.Series) -> pd.Series:
    """UTC YYYY-MM labels used only to partition storage, never to compute."""
    if getattr(timestamps.dtype, "tz", None) is None:
        raise IndicatorInputError(
            "timestamp must be timezone-aware UTC to partition enriched files"
        )
    return timestamps.dt.tz_convert("UTC").dt.strftime("%Y-%m")


def split_by_calendar_month(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group already-computed rows by UTC calendar month, preserving order."""
    if frame.empty:
        return {}
    keys = calendar_month_key(frame["timestamp"])
    parts: dict[str, pd.DataFrame] = {}
    for key in keys.drop_duplicates().tolist():
        parts[str(key)] = frame.loc[keys == key].copy()
    return parts


def list_processed_parquets(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.is_dir():
        raise EnrichmentError(f"Processed store is missing: {dataset_dir}")
    paths = sorted(dataset_dir.glob("*.parquet"))
    if not paths:
        raise EnrichmentError(f"No processed Parquet files in {dataset_dir}")
    return paths


def read_processed(dataset_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path, engine="pyarrow")
        for path in list_processed_parquets(dataset_dir)
    ]
    return pd.concat(frames, ignore_index=True)


def enriched_dataset_dir(data_config: DataConfig, indicator_config: IndicatorConfig) -> Path:
    return (
        indicator_config.enriched_dir
        / data_config.symbol
        / data_config.interval
    )


def write_monthly_parquet(
    parts: dict[str, pd.DataFrame],
    dest_dir: Path,
    symbol: str,
    interval: str,
    staging_dir: Path,
) -> list[Path]:
    """Atomically replace each month file; prune months no longer in ``parts``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    expected_names: set[str] = set()
    for period, part in parts.items():
        name = PARQUET_NAME_TEMPLATE.format(
            symbol=symbol, interval=interval, period=period
        )
        expected_names.add(name)
        final_path = dest_dir / name
        temp_path = staging_dir / (name + ".part")
        part.to_parquet(temp_path, engine="pyarrow", index=False, compression="snappy")
        os.replace(temp_path, final_path)
        written.append(final_path)

    for stale in dest_dir.glob("*.parquet"):
        if stale.name not in expected_names:
            stale.unlink()

    return written


def expected_warmup_rows(config: IndicatorConfig) -> dict[str, int]:
    """Leading NaN counts when the source series itself has no holes."""
    return {
        config.ema_fast_column: config.ema_fast - 1,
        config.ema_slow_column: config.ema_slow - 1,
        config.rsi_column: config.rsi_period,
        config.volume_ma_column: config.volume_ma_period - 1,
    }


def _continuity_check(
    processed: pd.DataFrame,
    enriched: pd.DataFrame,
    config: IndicatorConfig,
) -> ContinuityCheck | None:
    """Compare full-series February values to a February-only reset."""
    keys = calendar_month_key(processed["timestamp"])
    months = keys.drop_duplicates().tolist()
    if len(months) < 2:
        return None

    first_month, second_month = str(months[0]), str(months[1])
    first_slice = processed.loc[keys == first_month]
    second_slice = processed.loc[keys == second_month]
    concat = pd.concat([first_slice, second_slice], ignore_index=True)
    full = calculate_indicators(concat, config)
    february_only = calculate_indicators(second_slice.reset_index(drop=True), config)

    column = config.ema_fast_column
    full_feb = full.iloc[len(first_slice) :].reset_index(drop=True)
    sample_full = float(full_feb[column].iloc[0])
    sample_only = float(february_only[column].iloc[0])
    carried = sample_full != sample_only and not (
        pd.isna(sample_full) and pd.isna(sample_only)
    )

    enriched_keys = calendar_month_key(enriched["timestamp"])
    enriched_feb = enriched.loc[enriched_keys == second_month].reset_index(drop=True)
    matches = False
    sample_enriched: float | None = None
    if len(enriched_feb) == len(full_feb):
        sample_enriched = float(enriched_feb[column].iloc[0])
        indicator_cols = list(config.indicator_columns)
        matches = bool(
            full_feb[indicator_cols]
            .reset_index(drop=True)
            .equals(enriched_feb[indicator_cols].reset_index(drop=True))
        )

    return ContinuityCheck(
        first_month=first_month,
        second_month=second_month,
        carried_forward=bool(carried),
        matches_full_series=matches,
        sample_column=column,
        full_series_value=sample_full,
        february_only_value=sample_only,
        enriched_value=sample_enriched,
    )


def validate_enriched(
    processed: pd.DataFrame,
    enriched: pd.DataFrame,
    config: IndicatorConfig,
) -> EnrichmentReport:
    """Read-back checks. Expected warm-up NaNs are not data-quality failures."""
    report = EnrichmentReport(
        symbol="",
        market="",
        interval="",
        start_date=None,
        end_date=None,
        input_rows=len(processed),
        output_rows=len(enriched),
        ema_fast=config.ema_fast,
        ema_slow=config.ema_slow,
        rsi_period=config.rsi_period,
        volume_ma_period=config.volume_ma_period,
        warmup_rows=expected_warmup_rows(config),
    )

    if enriched.empty:
        report.findings.append("Enriched dataset is empty")
        return report

    report.first_timestamp = enriched["timestamp"].iloc[0]
    report.last_timestamp = enriched["timestamp"].iloc[-1]

    if len(enriched) != len(processed):
        report.findings.append(
            f"Row count changed: processed {len(processed)} vs enriched {len(enriched)}"
        )

    report.timestamps_preserved = bool(
        processed["timestamp"].reset_index(drop=True).equals(
            enriched["timestamp"].reset_index(drop=True)
        )
    )
    if not report.timestamps_preserved:
        report.findings.append("Timestamps do not match the processed store")

    ohlcv_ok = True
    for column in OHLCV_COLUMNS:
        if not processed[column].reset_index(drop=True).equals(
            enriched[column].reset_index(drop=True)
        ):
            ohlcv_ok = False
            report.findings.append(f"Column {column!r} was altered by enrichment")
    report.ohlcv_preserved = ohlcv_ok

    extra_ok = True
    for column in processed.columns:
        if column in OHLCV_COLUMNS:
            continue
        if column not in enriched.columns:
            extra_ok = False
            report.findings.append(f"Pass-through column {column!r} was dropped")
            continue
        if not processed[column].reset_index(drop=True).equals(
            enriched[column].reset_index(drop=True)
        ):
            extra_ok = False
            report.findings.append(f"Pass-through column {column!r} was altered")
    report.extra_columns_preserved = extra_ok

    report.duplicates = int(enriched["timestamp"].duplicated().sum())
    if report.duplicates:
        report.findings.append(f"{report.duplicates} duplicate timestamp(s)")

    missing_indicators = [
        column for column in config.indicator_columns if column not in enriched.columns
    ]
    if missing_indicators:
        report.findings.append(
            f"Missing indicator column(s): {', '.join(missing_indicators)}"
        )

    dtypes_ok = True
    tz = getattr(enriched["timestamp"].dtype, "tz", None)
    if str(tz) != "UTC":
        dtypes_ok = False
        report.findings.append(
            f"timestamp dtype must be datetime64[ns, UTC], got {enriched['timestamp'].dtype}"
        )
    for column in config.indicator_columns:
        if column not in enriched.columns:
            dtypes_ok = False
            continue
        if not pd.api.types.is_float_dtype(enriched[column].dtype):
            dtypes_ok = False
            report.findings.append(
                f"{column} dtype must be float, got {enriched[column].dtype}"
            )
        report.nan_counts[column] = int(enriched[column].isna().sum())
    report.dtypes_ok = dtypes_ok

    expected_nans = expected_warmup_rows(config)
    for column, expected in expected_nans.items():
        actual = report.nan_counts.get(column)
        if actual is None:
            continue
        if actual < expected:
            report.findings.append(
                f"{column}: expected at least {expected} warm-up NaN(s), found {actual}"
            )

    continuity = _continuity_check(processed, enriched, config)
    report.continuity = continuity
    if continuity is None:
        report.findings.append("Need at least two calendar months to verify continuity")
    else:
        if not continuity.carried_forward:
            report.findings.append(
                f"Month-boundary reset detected: {continuity.sample_column} at "
                f"{continuity.second_month} matches a {continuity.second_month}-only "
                f"recalculation ({continuity.february_only_value})"
            )
        if not continuity.matches_full_series:
            report.findings.append(
                f"Enriched {continuity.second_month} does not match full-series values"
            )

    return report


def run_enrichment(
    data_config: DataConfig,
    indicator_config: IndicatorConfig,
    progress: ProgressCallback | None = None,
) -> EnrichmentResult:
    def log(message: str) -> None:
        if progress is not None:
            progress(message)

    dest_dir = enriched_dataset_dir(data_config, indicator_config)

    log("Step 1/4  Reading processed Parquet")
    processed = read_processed(data_config.dataset_dir)

    log("Step 2/4  Calculating indicators on the full chronological series")
    enriched = calculate_indicators(processed, indicator_config)

    log("Step 3/4  Writing month-partitioned enriched Parquet")
    parts = split_by_calendar_month(enriched)
    paths = write_monthly_parquet(
        parts,
        dest_dir=dest_dir,
        symbol=data_config.symbol,
        interval=data_config.interval,
        staging_dir=data_config.downloads_dir,
    )

    log("Step 4/4  Reading enriched store back and validating")
    roundtrip = pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in paths],
        ignore_index=True,
    )
    report = validate_enriched(processed, roundtrip, indicator_config)
    report.symbol = data_config.symbol
    report.market = data_config.market
    report.interval = data_config.interval
    report.start_date = data_config.start_date
    report.end_date = data_config.end_date

    result = EnrichmentResult(
        data_config=data_config,
        indicator_config=indicator_config,
        report=report,
        parquet_paths=paths,
    )
    result.report_path.parent.mkdir(parents=True, exist_ok=True)
    result.report_path.write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: EnrichmentResult) -> str:
    report = result.report
    config = result.indicator_config
    lines: list[str] = [
        "=" * 78,
        "Indicator engine validation report",
        "=" * 78,
        "",
        f"Symbol:              {report.symbol}",
        f"Market:              {str(report.market).capitalize()}",
        f"Interval:            {report.interval}",
        f"Period:              {report.start_date} .. {report.end_date} "
        "(UTC, end date inclusive)",
        f"Generated:           {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "Counts",
        _RULE,
        f"Input rows:          {report.input_rows}",
        f"Output rows:         {report.output_rows}",
        f"First timestamp:     {_fmt(report.first_timestamp)}",
        f"Last timestamp:      {_fmt(report.last_timestamp)}",
        "",
        "Indicator parameters",
        _RULE,
        f"EMA fast:            {config.ema_fast}  -> {config.ema_fast_column}",
        f"EMA slow:            {config.ema_slow}  -> {config.ema_slow_column}",
        f"RSI (Wilder):        {config.rsi_period}  -> {config.rsi_column}",
        f"Volume SMA:          {config.volume_ma_period}  -> {config.volume_ma_column}",
        "",
        "Warm-up NaNs (leading, expected when source has no holes)",
        _RULE,
    ]
    for column, expected in report.warmup_rows.items():
        actual = report.nan_counts.get(column, "n/a")
        lines.append(f"{column:<18} expected {expected:<6} actual {actual}")

    lines.extend(
        [
            "",
            "Schema",
            _RULE,
            f"Required OHLCV:      {', '.join(REQUIRED_COLUMNS)}",
            f"Indicator columns:   {', '.join(config.indicator_columns)}",
            f"Indicator dtypes:    {'float64 OK' if report.dtypes_ok else 'FAIL'}",
            f"Timestamp dtype:     datetime64[ns, UTC]",
            "",
            "Preservation",
            _RULE,
            f"OHLCV preserved:     {_yn(report.ohlcv_preserved)}",
            f"Timestamps preserved: {_yn(report.timestamps_preserved)}",
            f"Extra columns kept:  {_yn(report.extra_columns_preserved)}",
            f"Duplicate timestamps:{report.duplicates}",
            "",
            "Month-boundary continuity",
            _RULE,
            "Calculations use only close/volume in row order. Calendar months",
            "are storage boundaries, not calculation inputs.",
        ]
    )
    continuity = report.continuity
    if continuity is None:
        lines.append("Status:              not checked (fewer than two months)")
    else:
        lines.append(f"Months compared:     {continuity.first_month} -> {continuity.second_month}")
        lines.append(
            f"Carried forward:     {_yn(continuity.carried_forward)} "
            f"({continuity.sample_column} full-series "
            f"{continuity.full_series_value} vs month-only "
            f"{continuity.february_only_value})"
        )
        lines.append(
            f"Enriched matches full series: {_yn(continuity.matches_full_series)}"
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


def render_console_summary(result: EnrichmentResult) -> str:
    report = result.report
    lines = [
        "",
        f"{report.symbol} {report.interval} indicators  "
        f"{report.start_date} .. {report.end_date}",
        f"  input rows         {report.input_rows}",
        f"  output rows        {report.output_rows}",
        f"  OHLCV preserved    {_yn(report.ohlcv_preserved)}",
        f"  timestamps         {_yn(report.timestamps_preserved)}",
        f"  enriched store     {enriched_dataset_dir(result.data_config, result.indicator_config)}",
        f"  report             {result.report_path}",
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
