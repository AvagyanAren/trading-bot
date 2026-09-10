"""Orchestrates the v0.1 data pipeline: download, parse, normalize, store, validate.

Idempotency comes from the storage layout rather than from bookkeeping. Raw
archives are content-addressed by their official SHA-256 sidecar, and processed
Parquet is written one file per source archive. Re-running therefore overwrites
exactly the months it re-derives, so candles can never accumulate twice, and
Parquet files outside the configured range are pruned so the processed store
always mirrors the config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from .downloader import (
    ArchivePeriod,
    BinanceKlineDownloader,
    DownloadReport,
    DEFAULT_BASE_URL,
)
from .normalizer import (
    detect_timestamp_unit,
    interval_to_timedelta,
    normalize_klines,
)
from .parser import read_kline_archive
from .validator import ValidationReport, validate_klines

ProgressCallback = Callable[[str], None]

REPORT_NAME_TEMPLATE = "data_validation_{symbol}_{interval}.txt"
GAPS_NAME_TEMPLATE = "gaps_{symbol}_{interval}.csv"
PARQUET_NAME_TEMPLATE = "{symbol}-{interval}-{period}.parquet"

_RULE = "-" * 78


@dataclass(frozen=True)
class DataConfig:
    """Resolved contents of ``config/data.yaml``."""

    project_root: Path
    symbol: str
    market: str
    data_type: str
    interval: str
    start_date: date
    end_date: date
    base_url: str
    raw_dir: Path
    processed_dir: Path
    downloads_dir: Path
    reports_dir: Path
    max_retries: int
    backoff_seconds: float
    timeout_seconds: float
    chunk_size: int
    verify_existing: bool

    @property
    def dataset_dir(self) -> Path:
        """Processed Parquet lives under ``<processed_dir>/<symbol>/<interval>``."""
        return self.processed_dir / self.symbol / self.interval

    @property
    def report_path(self) -> Path:
        return self.reports_dir / REPORT_NAME_TEMPLATE.format(
            symbol=self.symbol, interval=self.interval
        )

    @property
    def gaps_path(self) -> Path:
        return self.reports_dir / GAPS_NAME_TEMPLATE.format(
            symbol=self.symbol, interval=self.interval
        )


@dataclass(frozen=True)
class ArchiveSummary:
    """Per-archive provenance, surfaced in the report."""

    period: str
    aggregation: str
    had_header: bool
    timestamp_unit: str
    rows: int
    from_cache: bool


@dataclass
class PipelineResult:
    config: DataConfig
    download_report: DownloadReport
    validation_report: ValidationReport
    archives: list[ArchiveSummary] = field(default_factory=list)
    parquet_paths: list[Path] = field(default_factory=list)
    pruned_paths: list[Path] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.validation_report.passed

    @property
    def status(self) -> str:
        return self.validation_report.status


def _as_date(value: object, key: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"Config key {key!r} must be a YYYY-MM-DD date, got {value!r}")


def load_config(config_path: Path) -> DataConfig:
    """Read ``config/data.yaml`` and resolve every path against the project root."""
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    project_root = config_path.parent.parent

    def resolve(key: str, default: str) -> Path:
        return (project_root / str(raw.get(key, default))).resolve()

    download = raw.get("download") or {}

    return DataConfig(
        project_root=project_root,
        symbol=str(raw["symbol"]).upper(),
        market=str(raw.get("market", "spot")),
        data_type=str(raw.get("data_type", "klines")),
        interval=str(raw["interval"]),
        start_date=_as_date(raw["start_date"], "start_date"),
        end_date=_as_date(raw["end_date"], "end_date"),
        base_url=str(raw.get("base_url", DEFAULT_BASE_URL)),
        raw_dir=resolve("raw_dir", "data/raw/binance"),
        processed_dir=resolve("processed_dir", "data/processed"),
        downloads_dir=resolve("downloads_dir", "data/downloads"),
        reports_dir=resolve("reports_dir", "reports"),
        max_retries=int(download.get("max_retries", 4)),
        backoff_seconds=float(download.get("backoff_seconds", 2.0)),
        timeout_seconds=float(download.get("timeout_seconds", 60.0)),
        chunk_size=int(download.get("chunk_size", 262_144)),
        verify_existing=bool(download.get("verify_existing", True)),
    )


def run_pipeline(
    config: DataConfig, progress: ProgressCallback | None = None
) -> PipelineResult:
    """Execute the full pipeline and return a PASS/FAIL result."""

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    downloader = BinanceKlineDownloader(
        symbol=config.symbol,
        interval=config.interval,
        start_date=config.start_date,
        end_date=config.end_date,
        raw_dir=config.raw_dir,
        downloads_dir=config.downloads_dir,
        market=config.market,
        data_type=config.data_type,
        base_url=config.base_url,
        max_retries=config.max_retries,
        backoff_seconds=config.backoff_seconds,
        timeout_seconds=config.timeout_seconds,
        chunk_size=config.chunk_size,
        verify_existing=config.verify_existing,
        progress=progress,
    )

    report("Step 1/5  Ensuring raw archives are present and verified")
    download_report = downloader.download()

    report("Step 2/5  Parsing, normalizing and writing processed Parquet")
    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[ArchiveSummary] = []
    parquet_paths: list[Path] = []

    total = len(download_report.archives)
    for index, archive in enumerate(download_report.archives, start=1):
        parsed = read_kline_archive(archive.path)
        unit = detect_timestamp_unit(parsed.frame["open_time"])
        frame = normalize_klines(
            parsed,
            expected_start=archive.period.start,
            expected_end=archive.period.end,
        )
        parquet_path = _write_parquet(frame, config, archive.period)
        parquet_paths.append(parquet_path)
        summaries.append(
            ArchiveSummary(
                period=archive.period.period,
                aggregation=archive.period.aggregation,
                had_header=parsed.had_header,
                timestamp_unit=unit,
                rows=len(frame),
                from_cache=archive.from_cache,
            )
        )
        report(
            f"[{index}/{total}] {archive.period.period}: {len(frame)} candles, "
            f"header={'yes' if parsed.had_header else 'no'}, unit={unit}"
        )

    pruned = _prune_stale_parquet(config, parquet_paths)
    for path in pruned:
        report(f"Pruned stale processed file outside the configured range: {path.name}")

    report("Step 3/5  Reading the processed store back for validation")
    combined = _read_processed(parquet_paths)

    report("Step 4/5  Validating")
    validation_report = validate_klines(
        combined,
        symbol=config.symbol,
        market=config.market,
        interval=config.interval,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    result = PipelineResult(
        config=config,
        download_report=download_report,
        validation_report=validation_report,
        archives=summaries,
        parquet_paths=parquet_paths,
        pruned_paths=pruned,
    )

    report("Step 5/5  Writing validation report")
    write_reports(result)
    return result


# --- Processed storage ------------------------------------------------------


def _write_parquet(
    frame: pd.DataFrame, config: DataConfig, period: ArchivePeriod
) -> Path:
    """Write one archive's candles to Parquet, replacing any previous copy."""
    name = PARQUET_NAME_TEMPLATE.format(
        symbol=config.symbol, interval=config.interval, period=period.period
    )
    final_path = config.dataset_dir / name

    config.downloads_dir.mkdir(parents=True, exist_ok=True)
    temp_path = config.downloads_dir / (name + ".part")
    frame.to_parquet(temp_path, engine="pyarrow", index=False, compression="snappy")
    os.replace(temp_path, final_path)
    return final_path


def _prune_stale_parquet(config: DataConfig, keep: list[Path]) -> list[Path]:
    """Delete processed files that the current configuration no longer covers."""
    expected = {path.name for path in keep}
    pruned: list[Path] = []
    for path in sorted(config.dataset_dir.glob("*.parquet")):
        if path.name not in expected:
            path.unlink()
            pruned.append(path)
    return pruned


def _read_processed(parquet_paths: list[Path]) -> pd.DataFrame:
    """Concatenate the processed store in chronological archive order.

    Validation runs against what was actually persisted, not the in-memory
    frames, so a dtype or timezone lost in the Parquet round-trip is caught.
    """
    if not parquet_paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path, engine="pyarrow") for path in parquet_paths]
    return pd.concat(frames, ignore_index=True)


# --- Reporting --------------------------------------------------------------


def write_reports(result: PipelineResult) -> None:
    """Write the human-readable report and the machine-readable gap list."""
    config = result.config
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(render_report(result), encoding="utf-8")
    _write_gaps_csv(result)


def _write_gaps_csv(result: PipelineResult) -> None:
    gaps = result.validation_report.gaps
    frame = pd.DataFrame(
        [
            {
                "timestamp_before_gap": _iso(gap.before),
                "timestamp_after_gap": _iso(gap.after),
                "first_missing_candle": _iso(gap.first_missing),
                "last_missing_candle": _iso(gap.last_missing),
                "missing_candles": gap.missing_candles,
                "missing_duration": str(gap.duration),
            }
            for gap in gaps
        ],
        columns=[
            "timestamp_before_gap",
            "timestamp_after_gap",
            "first_missing_candle",
            "last_missing_candle",
            "missing_candles",
            "missing_duration",
        ],
    )
    frame.to_csv(result.config.gaps_path, index=False)


def render_report(result: PipelineResult) -> str:
    """Render the validation report exactly as documented in the README."""
    config = result.config
    report = result.validation_report
    lines: list[str] = []

    lines.append("=" * 78)
    lines.append("Binance historical data validation report")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Symbol:              {report.symbol}")
    lines.append(f"Market:              {config.market.capitalize()}")
    lines.append(f"Interval:            {report.interval}")
    lines.append(
        f"Configured period:   {config.start_date} .. {config.end_date} "
        "(UTC, end date inclusive)"
    )
    lines.append(f"Source:              {config.base_url}")
    lines.append(
        f"Generated:           "
        f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    lines.append("")

    lines.append("Candle counts")
    lines.append(_RULE)
    lines.append(f"Expected candles:    {report.expected_candles}")
    lines.append(f"Actual candles:      {report.actual_candles}")
    lines.append(f"Unique timestamps:   {report.unique_candles}")
    lines.append(f"Missing candles:     {report.missing_candles}")
    lines.append("")

    lines.append("Integrity")
    lines.append(_RULE)
    lines.append(f"Duplicates:          {report.duplicate_candles}")
    lines.append(f"Invalid OHLC:        {report.invalid_ohlc_rows}")
    lines.append(f"Invalid prices:      {report.invalid_price_rows}")
    lines.append(f"Invalid volume:      {report.invalid_volume_rows}")
    lines.append(f"Timestamp errors:    {report.timestamp_errors}")
    lines.append("")

    lines.append("Boundaries")
    lines.append(_RULE)
    lines.append(f"First candle:        {_fmt(report.first_timestamp)}")
    lines.append(f"Last candle:         {_fmt(report.last_timestamp)}")
    lines.append(f"Expected first:      {_fmt(report.expected_first_timestamp)}")
    lines.append(f"Expected last:       {_fmt(report.expected_last_timestamp)}")
    lines.append(
        f"Boundary check:      {'OK' if report.boundary_ok else 'MISMATCH'}"
    )
    lines.append("")

    lines.extend(_render_gap_section(result))
    lines.extend(_render_source_section(result))
    lines.extend(_render_findings_section(report))

    lines.append("=" * 78)
    lines.append(f"STATUS: {report.status}")
    lines.append("=" * 78)
    lines.append("")
    return "\n".join(lines)


def _render_gap_section(result: PipelineResult) -> list[str]:
    report = result.validation_report
    interval_label = _interval_label(result.config.interval)

    lines = ["Gap report", _RULE]
    lines.append(f"Expected interval:   {interval_label}")
    lines.append(f"Total candles:       {report.actual_candles}")
    lines.append(f"Missing intervals:   {len(report.gaps)}")
    lines.append(f"Missing candles:     {report.missing_candles}")

    if report.gaps:
        lines.append("")
        lines.append(
            f"{'Timestamp before gap':<26}{'Timestamp after gap':<26}"
            f"{'Missing':>8}  Duration"
        )
        for gap in report.gaps:
            lines.append(
                f"{_fmt(gap.before):<26}{_fmt(gap.after):<26}"
                f"{gap.missing_candles:>8}  {gap.duration}"
            )
        lines.append("")
        lines.append(
            "Missing candles are reported only. No candle is synthesized, "
            "interpolated or forward-filled."
        )
    lines.append("")
    return lines


def _render_source_section(result: PipelineResult) -> list[str]:
    download = result.download_report
    lines = ["Source archives", _RULE]
    lines.append(f"Archives used:       {len(result.archives)}")
    lines.append(f"Newly downloaded:    {download.downloaded_count}")
    lines.append(f"Already verified:    {download.cached_count}")
    lines.append(f"Not published:       {len(download.missing)}")

    units = sorted({summary.timestamp_unit for summary in result.archives})
    headers = sorted({summary.had_header for summary in result.archives})
    lines.append(f"Epoch units seen:    {', '.join(units) if units else 'n/a'}")
    lines.append(
        "CSV header rows:     "
        + ", ".join("present" if flag else "absent" for flag in headers)
    )

    if download.missing:
        lines.append("")
        lines.append("Periods not published upstream:")
        for period in download.missing:
            lines.append(f"  {period.aggregation} {period.period}")
    lines.append("")
    return lines


def _render_findings_section(report: ValidationReport) -> list[str]:
    lines = ["Findings", _RULE]
    if not report.findings:
        lines.append("No problems detected.")
        lines.append("")
        return lines

    for finding in report.findings:
        lines.append(f"[{finding.severity.value}] {finding.code}")
        lines.append(f"  {finding.message}")
        for sample in finding.samples:
            lines.append(f"    - {sample}")
    lines.append("")
    lines.append(
        f"{len(report.failures)} failure(s), {len(report.warnings)} warning(s). "
        "Missing candles are warnings; all other findings fail the run."
    )
    lines.append("")
    return lines


def _interval_label(interval: str) -> str:
    delta = interval_to_timedelta(interval)
    minutes = delta.total_seconds() / 60
    if minutes.is_integer() and minutes < 60:
        return f"{int(minutes)} minutes ({interval})"
    return f"{delta} ({interval})"


def _fmt(timestamp: pd.Timestamp | None) -> str:
    if timestamp is None or pd.isna(timestamp):
        return "n/a"
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")


def _iso(timestamp: pd.Timestamp | None) -> str:
    if timestamp is None or pd.isna(timestamp):
        return ""
    return pd.Timestamp(timestamp).isoformat()


def render_console_summary(result: PipelineResult) -> str:
    """Compact stdout summary mirroring the written report."""
    report = result.validation_report
    lines = [
        "",
        f"{report.symbol} {report.interval} ({result.config.market})  "
        f"{result.config.start_date} .. {result.config.end_date}",
        f"  expected candles   {report.expected_candles}",
        f"  actual candles     {report.actual_candles}",
        f"  missing candles    {report.missing_candles} "
        f"across {len(report.gaps)} gap(s)",
        f"  duplicates         {report.duplicate_candles}",
        f"  invalid OHLC       {report.invalid_ohlc_rows}",
        f"  invalid prices     {report.invalid_price_rows}",
        f"  invalid volume     {report.invalid_volume_rows}",
        f"  timestamp errors   {report.timestamp_errors}",
        f"  first candle       {_fmt(report.first_timestamp)}",
        f"  last candle        {_fmt(report.last_timestamp)}",
        f"  processed store    {result.config.dataset_dir}",
        f"  validation report  {result.config.report_path}",
        "",
    ]
    for finding in result.validation_report.findings:
        lines.append(f"  [{finding.severity.value}] {finding.message}")
    if result.validation_report.findings:
        lines.append("")
    lines.append(f"STATUS: {report.status}")
    return "\n".join(lines)
