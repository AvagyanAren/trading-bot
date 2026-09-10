"""End-to-end pipeline tests, including idempotency, run entirely offline.

A synthetic raw store is built on disk with real ZIP archives and real SHA-256
sidecars, so the downloader treats every archive as already verified and never
reaches the network. The two archives deliberately straddle Binance's
2025-01-01 epoch-unit change: one publishes milliseconds, the other
microseconds.
"""

from __future__ import annotations

import hashlib
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.downloader import CHECKSUM_SUFFIX
from src.data.normalizer import PROCESSED_COLUMNS, PROCESSED_DTYPES, TIMESTAMP_COLUMN
from src.data.pipeline import load_config, render_report, run_pipeline

CANDLES_PER_DAY = 288
MS_DAY = date(2024, 12, 31)
US_DAY = date(2025, 1, 1)

_UNIT_FACTORS = {"ms": 1_000, "us": 1_000_000}

CONFIG_TEMPLATE = """\
symbol: BTCUSDT
market: spot
data_type: klines
interval: 5m
start_date: {start}
end_date: {end}
base_url: https://data.binance.vision/data
raw_dir: data/raw/binance
processed_dir: data/processed
downloads_dir: data/downloads
reports_dir: reports
download:
  max_retries: 1
  backoff_seconds: 0.0
  verify_existing: true
"""


def _csv_for_day(day: date, unit: str, with_header: bool) -> str:
    factor = _UNIT_FACTORS[unit]
    base_seconds = int(pd.Timestamp(day, tz="UTC").timestamp())
    step_seconds = 300

    lines: list[str] = []
    if with_header:
        lines.append(",".join(PROCESSED_COLUMNS[:6]) + ",close_time,quote_volume,count,taker_base,taker_quote,ignore")

    for index in range(CANDLES_PER_DAY):
        open_epoch = (base_seconds + index * step_seconds) * factor
        close_epoch = open_epoch + step_seconds * factor - 1
        price = 40_000 + index
        lines.append(
            ",".join(
                [
                    str(open_epoch),
                    f"{price}.00000000",
                    f"{price + 5}.00000000",
                    f"{price - 5}.00000000",
                    f"{price + 2}.00000000",
                    "3.50000000",
                    str(close_epoch),
                    f"{price * 3.5:.8f}",
                    str(120 + index),
                    "1.75000000",
                    f"{price * 1.75:.8f}",
                    "0",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _write_archive(raw_dir: Path, day: date, unit: str, with_header: bool) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"BTCUSDT-5m-{day.isoformat()}"
    zip_path = raw_dir / f"{stem}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{stem}.csv", _csv_for_day(day, unit, with_header))

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    zip_path.with_name(zip_path.name + CHECKSUM_SUFFIX).write_text(
        f"{digest}  {zip_path.name}\n", encoding="utf-8"
    )
    return zip_path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A self-contained project root with a pre-verified synthetic raw store."""
    raw_dir = tmp_path / "data" / "raw" / "binance" / "spot" / "BTCUSDT" / "5m"
    _write_archive(raw_dir, MS_DAY, unit="ms", with_header=False)
    _write_archive(raw_dir, US_DAY, unit="us", with_header=True)

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "data.yaml").write_text(
        CONFIG_TEMPLATE.format(start=MS_DAY.isoformat(), end=US_DAY.isoformat()),
        encoding="utf-8",
    )
    return tmp_path


def run(project: Path):
    return run_pipeline(load_config(project / "config" / "data.yaml"))


# --- Configuration ----------------------------------------------------------


def test_config_is_read_from_yaml_not_hardcoded(project: Path):
    config = load_config(project / "config" / "data.yaml")

    assert config.symbol == "BTCUSDT"
    assert config.market == "spot"
    assert config.interval == "5m"
    assert config.start_date == MS_DAY
    assert config.end_date == US_DAY
    assert config.dataset_dir == project / "data" / "processed" / "BTCUSDT" / "5m"
    assert config.report_path.name == "data_validation_BTCUSDT_5m.txt"


# --- Full run ---------------------------------------------------------------


def test_pipeline_passes_across_the_epoch_unit_boundary(project: Path):
    result = run(project)

    assert result.status == "PASS"
    assert result.passed is True
    report = result.validation_report
    assert report.expected_candles == 2 * CANDLES_PER_DAY
    assert report.actual_candles == 2 * CANDLES_PER_DAY
    assert report.missing_candles == 0
    assert report.duplicate_candles == 0
    assert report.invalid_ohlc_rows == 0
    assert report.invalid_price_rows == 0
    assert report.invalid_volume_rows == 0
    assert report.timestamp_errors == 0


def test_both_epoch_units_and_header_styles_are_handled(project: Path):
    result = run(project)

    units = {summary.timestamp_unit for summary in result.archives}
    headers = {summary.had_header for summary in result.archives}

    assert units == {"ms", "us"}
    assert headers == {True, False}


def test_boundary_candles_match_the_configured_period(project: Path):
    report = run(project).validation_report

    assert report.first_timestamp == pd.Timestamp("2024-12-31 00:00:00", tz="UTC")
    assert report.last_timestamp == pd.Timestamp("2025-01-01 23:55:00", tz="UTC")
    assert report.boundary_ok is True


def test_processed_store_is_parquet_partitioned_by_archive(project: Path):
    result = run(project)
    files = sorted(result.config.dataset_dir.glob("*.parquet"))

    assert [path.name for path in files] == [
        "BTCUSDT-5m-2024-12-31.parquet",
        "BTCUSDT-5m-2025-01-01.parquet",
    ]
    assert not list(result.config.dataset_dir.glob("*.csv"))


def test_parquet_round_trip_preserves_explicit_dtypes(project: Path):
    result = run(project)
    frame = pd.read_parquet(result.parquet_paths[0], engine="pyarrow")

    assert list(frame.columns) == list(PROCESSED_COLUMNS)
    for column, dtype in PROCESSED_DTYPES.items():
        assert str(frame[column].dtype) == dtype
    assert str(frame[TIMESTAMP_COLUMN].dt.tz) == "UTC"


def test_validation_report_lists_every_required_field(project: Path):
    result = run(project)
    text = result.config.report_path.read_text(encoding="utf-8")

    for label in (
        "Symbol:",
        "Market:",
        "Interval:",
        "Expected candles:",
        "Actual candles:",
        "Missing candles:",
        "Duplicates:",
        "Invalid OHLC:",
        "Invalid prices:",
        "Invalid volume:",
        "Timestamp errors:",
        "First candle:",
        "Last candle:",
        "Boundary check:",
        "Expected interval:",
        "STATUS: PASS",
    ):
        assert label in text, f"report is missing {label!r}"


def test_gap_report_csv_is_written_even_when_clean(project: Path):
    result = run(project)
    gaps = pd.read_csv(result.config.gaps_path)

    assert list(gaps.columns) == [
        "timestamp_before_gap",
        "timestamp_after_gap",
        "first_missing_candle",
        "last_missing_candle",
        "missing_candles",
        "missing_duration",
    ]
    assert gaps.empty


def test_no_raw_file_is_modified_by_the_pipeline(project: Path):
    raw_dir = project / "data" / "raw" / "binance" / "spot" / "BTCUSDT" / "5m"
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw_dir.iterdir())
    }

    run(project)

    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw_dir.iterdir())
    }
    assert before == after


def test_staging_area_is_left_clean(project: Path):
    run(project)

    assert list((project / "data" / "downloads").glob("*.part")) == []


# --- Idempotency ------------------------------------------------------------


def test_second_run_reproduces_the_same_dataset(project: Path):
    first = run(project)
    first_frames = [
        pd.read_parquet(path, engine="pyarrow") for path in first.parquet_paths
    ]

    second = run(project)
    second_frames = [
        pd.read_parquet(path, engine="pyarrow") for path in second.parquet_paths
    ]

    assert second.status == "PASS"
    assert second.validation_report.actual_candles == (
        first.validation_report.actual_candles
    )
    assert second.validation_report.duplicate_candles == 0
    for before, after in zip(first_frames, second_frames):
        pd.testing.assert_frame_equal(before, after)


def test_second_run_downloads_nothing(project: Path):
    run(project)
    second = run(project)

    assert second.download_report.downloaded_count == 0
    assert second.download_report.cached_count == 2


def test_second_run_does_not_accumulate_parquet_files(project: Path):
    first_files = sorted(p.name for p in run(project).config.dataset_dir.glob("*"))
    second_files = sorted(p.name for p in run(project).config.dataset_dir.glob("*"))

    assert first_files == second_files
    assert len(second_files) == 2


def test_reports_are_stable_between_runs(project: Path):
    first = render_report(run(project))
    second = render_report(run(project))

    def without_timestamp(text: str) -> list[str]:
        return [line for line in text.splitlines() if not line.startswith("Generated:")]

    assert without_timestamp(first) == without_timestamp(second)


def test_shrinking_the_configured_range_prunes_stale_processed_files(project: Path):
    run(project)
    config_path = project / "config" / "data.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(start=US_DAY.isoformat(), end=US_DAY.isoformat()),
        encoding="utf-8",
    )

    result = run(project)

    assert [path.name for path in result.parquet_paths] == [
        "BTCUSDT-5m-2025-01-01.parquet"
    ]
    assert [path.name for path in result.pruned_paths] == [
        "BTCUSDT-5m-2024-12-31.parquet"
    ]
    assert result.status == "PASS"
    assert result.validation_report.actual_candles == CANDLES_PER_DAY


# --- Failure surfacing ------------------------------------------------------


def test_a_gap_in_the_raw_data_warns_but_still_passes(tmp_path: Path):
    raw_dir = tmp_path / "data" / "raw" / "binance" / "spot" / "BTCUSDT" / "5m"
    stem = f"BTCUSDT-5m-{US_DAY.isoformat()}"
    csv_lines = _csv_for_day(US_DAY, "us", with_header=False).splitlines()
    del csv_lines[100:103]

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{stem}.csv", "\n".join(csv_lines) + "\n")
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    zip_path.with_name(zip_path.name + CHECKSUM_SUFFIX).write_text(
        f"{digest}  {zip_path.name}\n", encoding="utf-8"
    )

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "data.yaml").write_text(
        CONFIG_TEMPLATE.format(start=US_DAY.isoformat(), end=US_DAY.isoformat()),
        encoding="utf-8",
    )

    result = run(tmp_path)
    gaps = pd.read_csv(result.config.gaps_path)

    assert result.status == "PASS"
    assert result.validation_report.missing_candles == 3
    assert len(result.validation_report.gaps) == 1
    assert result.validation_report.gaps[0].duration == timedelta(minutes=20)
    assert len(gaps) == 1
    assert gaps.loc[0, "missing_candles"] == 3
