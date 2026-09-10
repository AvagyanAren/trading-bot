"""Downloader tests: archive planning, URL construction and checksum handling.

These are offline. Network behaviour is exercised through a stub session, so
the suite never contacts Binance.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest
import requests

from src.data.downloader import (
    CHECKSUM_SUFFIX,
    DAILY,
    MONTHLY,
    ArchiveNotFound,
    BinanceKlineDownloader,
    ChecksumMismatch,
    DownloadError,
    parse_checksum,
    plan_periods,
    sha256_of_file,
)


# --- Period planning --------------------------------------------------------


def test_whole_months_use_monthly_archives():
    periods = plan_periods(date(2024, 1, 1), date(2024, 3, 31))

    assert [p.aggregation for p in periods] == [MONTHLY] * 3
    assert [p.period for p in periods] == ["2024-01", "2024-02", "2024-03"]


def test_configured_two_year_range_needs_24_monthly_archives():
    periods = plan_periods(date(2024, 1, 1), date(2025, 12, 31))

    assert len(periods) == 24
    assert all(p.aggregation == MONTHLY for p in periods)
    assert periods[0].period == "2024-01"
    assert periods[-1].period == "2025-12"


def test_partial_months_fall_back_to_daily_archives():
    periods = plan_periods(date(2024, 1, 30), date(2024, 2, 2))

    assert [(p.aggregation, p.period) for p in periods] == [
        (DAILY, "2024-01-30"),
        (DAILY, "2024-01-31"),
        (DAILY, "2024-02-01"),
        (DAILY, "2024-02-02"),
    ]


def test_leap_year_february_is_a_whole_month():
    periods = plan_periods(date(2024, 2, 1), date(2024, 2, 29))

    assert [(p.aggregation, p.period) for p in periods] == [(MONTHLY, "2024-02")]


def test_planning_covers_every_day_exactly_once():
    periods = plan_periods(date(2023, 12, 15), date(2024, 2, 3))
    covered = [(p.start, p.end) for p in periods]

    assert covered[0][0] == date(2023, 12, 15)
    assert covered[-1][1] == date(2024, 2, 3)
    for earlier, later in zip(covered, covered[1:]):
        assert (later[0] - earlier[1]).days == 1


def test_reversed_range_is_rejected():
    with pytest.raises(ValueError, match="precedes"):
        plan_periods(date(2024, 2, 1), date(2024, 1, 1))


# --- Checksum sidecars ------------------------------------------------------


def test_parse_checksum_reads_the_official_sidecar_format():
    digest = "a" * 64
    content = f"{digest}  BTCUSDT-5m-2024-01.zip"

    assert parse_checksum(content, "BTCUSDT-5m-2024-01.zip") == digest


def test_parse_checksum_rejects_a_sidecar_for_another_file():
    content = f"{'b' * 64}  BTCUSDT-5m-2024-02.zip"

    with pytest.raises(DownloadError, match="different file"):
        parse_checksum(content, "BTCUSDT-5m-2024-01.zip")


def test_parse_checksum_rejects_content_without_a_digest():
    with pytest.raises(DownloadError, match="No SHA-256 digest"):
        parse_checksum("not a checksum at all")


def test_sha256_of_file_matches_hashlib(tmp_path: Path):
    target = tmp_path / "payload.bin"
    target.write_bytes(b"binance public data")

    assert sha256_of_file(target) == hashlib.sha256(b"binance public data").hexdigest()


# --- URL construction -------------------------------------------------------


def _downloader(tmp_path: Path, session=None, **overrides) -> BinanceKlineDownloader:
    defaults = dict(
        symbol="btcusdt",
        interval="5m",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        raw_dir=tmp_path / "raw",
        downloads_dir=tmp_path / "downloads",
        base_url="https://data.binance.vision/data",
        backoff_seconds=0.0,
        session=session,
        sleep=lambda _seconds: None,
    )
    defaults.update(overrides)
    return BinanceKlineDownloader(**defaults)


def test_archive_url_follows_the_documented_layout(tmp_path: Path):
    downloader = _downloader(tmp_path)
    period = downloader.plan()[0]

    assert downloader.archive_url(period) == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/"
        "BTCUSDT-5m-2024-01.zip"
    )


def test_raw_path_separates_market_symbol_and_interval(tmp_path: Path):
    downloader = _downloader(tmp_path)
    period = downloader.plan()[0]

    assert downloader.raw_path(period) == (
        tmp_path / "raw" / "spot" / "BTCUSDT" / "5m" / "BTCUSDT-5m-2024-01.zip"
    )


# --- Network behaviour via a stub session -----------------------------------


class StubResponse:
    def __init__(self, status_code: int, body: bytes = b""):
        self.status_code = status_code
        self._body = body
        self.closed = False

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    @property
    def text(self) -> str:
        return self._body.decode("utf-8")

    def iter_content(self, chunk_size: int):
        yield self._body

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "StubResponse":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class StubSession:
    """Serves queued responses per URL and records every request."""

    def __init__(self, responses: dict[str, list[StubResponse]]):
        self._responses = responses
        self.requests: list[str] = []

    def get(self, url: str, stream: bool = False, timeout: float | None = None):
        self.requests.append(url)
        queue = self._responses.get(url)
        if not queue:
            raise AssertionError(f"Unexpected request for {url}")
        return queue.pop(0) if len(queue) > 1 else queue[0]


def _single_month_session(payload: bytes, digest: str | None = None) -> StubSession:
    base = (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/"
        "BTCUSDT-5m-2024-01.zip"
    )
    digest = digest or hashlib.sha256(payload).hexdigest()
    return StubSession(
        {
            base: [StubResponse(200, payload)],
            base + CHECKSUM_SUFFIX: [
                StubResponse(200, f"{digest}  BTCUSDT-5m-2024-01.zip".encode())
            ],
        }
    )


def test_verified_archive_lands_in_the_raw_store(tmp_path: Path):
    payload = b"pretend-zip-bytes"
    session = _single_month_session(payload)
    downloader = _downloader(tmp_path, session=session)

    report = downloader.download()

    assert len(report.archives) == 1
    archive = report.archives[0]
    assert archive.path.read_bytes() == payload
    assert archive.from_cache is False
    assert archive.sha256 == hashlib.sha256(payload).hexdigest()


def test_checksum_sidecar_is_persisted_next_to_the_archive(tmp_path: Path):
    payload = b"pretend-zip-bytes"
    downloader = _downloader(tmp_path, session=_single_month_session(payload))

    archive = downloader.download().archives[0]
    sidecar = archive.path.with_name(archive.path.name + CHECKSUM_SUFFIX)

    assert sidecar.exists()
    assert parse_checksum(sidecar.read_text(encoding="utf-8")) == archive.sha256


def test_second_run_reuses_the_verified_archive_without_downloading(tmp_path: Path):
    payload = b"pretend-zip-bytes"
    first_session = _single_month_session(payload)
    _downloader(tmp_path, session=first_session).download()
    request_count = len(first_session.requests)

    second_session = _single_month_session(payload)
    report = _downloader(tmp_path, session=second_session).download()

    assert request_count > 0
    assert second_session.requests == []
    assert report.cached_count == 1
    assert report.downloaded_count == 0


def test_corrupt_local_archive_is_replaced(tmp_path: Path):
    payload = b"pretend-zip-bytes"
    downloader = _downloader(tmp_path, session=_single_month_session(payload))
    archive = downloader.download().archives[0]
    archive.path.write_bytes(b"truncated")

    report = _downloader(tmp_path, session=_single_month_session(payload)).download()

    assert report.downloaded_count == 1
    assert report.archives[0].path.read_bytes() == payload


def test_checksum_mismatch_leaves_the_raw_store_untouched(tmp_path: Path):
    session = _single_month_session(b"pretend-zip-bytes", digest="c" * 64)
    downloader = _downloader(tmp_path, session=session)

    with pytest.raises(ChecksumMismatch):
        downloader.download()

    assert not downloader.raw_path(downloader.plan()[0]).exists()
    assert list((tmp_path / "downloads").glob("*.part")) == []


def test_missing_archive_is_recorded_not_raised(tmp_path: Path):
    base = (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/"
        "BTCUSDT-5m-2024-01.zip"
    )
    session = StubSession(
        {base: [StubResponse(404)], base + CHECKSUM_SUFFIX: [StubResponse(404)]}
    )

    report = _downloader(tmp_path, session=session).download()

    assert report.archives == []
    assert [p.period for p in report.missing] == ["2024-01"]


def test_transient_server_error_is_retried_then_succeeds(tmp_path: Path):
    payload = b"pretend-zip-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    base = (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/"
        "BTCUSDT-5m-2024-01.zip"
    )
    session = StubSession(
        {
            base: [StubResponse(503), StubResponse(200, payload)],
            base + CHECKSUM_SUFFIX: [
                StubResponse(200, f"{digest}  BTCUSDT-5m-2024-01.zip".encode())
            ],
        }
    )

    report = _downloader(tmp_path, session=session).download()

    assert report.downloaded_count == 1
    assert session.requests.count(base) == 2


def test_persistent_network_failure_gives_up_with_a_clear_error(tmp_path: Path):
    class FailingSession:
        def get(self, url: str, stream: bool = False, timeout: float | None = None):
            raise requests.ConnectionError("network unreachable")

    downloader = _downloader(tmp_path, session=FailingSession(), max_retries=3)

    with pytest.raises(DownloadError, match="after 3 attempts"):
        downloader.download()


def test_client_error_other_than_404_is_fatal(tmp_path: Path):
    base = (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/"
        "BTCUSDT-5m-2024-01.zip"
    )
    session = StubSession({base + CHECKSUM_SUFFIX: [StubResponse(403)]})

    with pytest.raises(DownloadError, match="HTTP 403"):
        _downloader(tmp_path, session=session).download()


def test_archive_not_found_is_a_download_error_subclass():
    assert issubclass(ArchiveNotFound, DownloadError)
    assert issubclass(ChecksumMismatch, DownloadError)
