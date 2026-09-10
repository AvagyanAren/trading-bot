"""Downloads Binance Public Data kline archives into an immutable raw store.

The raw store is append-only: a file that lands in ``raw_dir`` has always been
verified against its official SHA-256 sidecar, and is never rewritten in place.
Transfers stage through ``downloads_dir`` as ``*.part`` files and are moved with
:func:`os.replace`, so an interrupted run can never leave a truncated archive
that looks complete.

Source layout reference: https://github.com/binance/binance-public-data
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Sequence

import requests

# The only place in the project that encodes the Binance Public Data URL layout.
ARCHIVE_URL_TEMPLATE = (
    "{base_url}/{market}/{aggregation}/{data_type}/{symbol}/{interval}/{filename}"
)
ARCHIVE_NAME_TEMPLATE = "{symbol}-{interval}-{period}.zip"
CHECKSUM_SUFFIX = ".CHECKSUM"
DEFAULT_BASE_URL = "https://data.binance.vision/data"

MONTHLY = "monthly"
DAILY = "daily"

# A checksum sidecar holds "<64 hex chars>  <filename>".
_SHA256_PATTERN = re.compile(r"\b([0-9a-fA-F]{64})\b")

# Transient HTTP conditions worth another attempt.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

ProgressCallback = Callable[[str], None]


class DownloadError(RuntimeError):
    """Base class for unrecoverable download problems."""


class ArchiveNotFound(DownloadError):
    """The archive does not exist upstream (HTTP 404)."""


class ChecksumMismatch(DownloadError):
    """A transferred archive did not match its official SHA-256 digest."""


@dataclass(frozen=True)
class ArchivePeriod:
    """One Binance archive file's worth of time.

    ``aggregation`` selects the monthly or daily archive family, and ``period``
    is the token Binance puts in the filename (``2024-01`` or ``2024-01-05``).
    """

    aggregation: str
    period: str
    start: date
    end: date

    def filename(self, symbol: str, interval: str) -> str:
        return ARCHIVE_NAME_TEMPLATE.format(
            symbol=symbol, interval=interval, period=self.period
        )


@dataclass(frozen=True)
class DownloadedArchive:
    """A verified archive sitting in the raw store."""

    period: ArchivePeriod
    path: Path
    url: str
    sha256: str
    from_cache: bool


@dataclass
class DownloadReport:
    """Outcome of a download run."""

    archives: list[DownloadedArchive] = field(default_factory=list)
    missing: list[ArchivePeriod] = field(default_factory=list)

    @property
    def downloaded_count(self) -> int:
        return sum(1 for archive in self.archives if not archive.from_cache)

    @property
    def cached_count(self) -> int:
        return sum(1 for archive in self.archives if archive.from_cache)


def plan_periods(start_date: date, end_date: date) -> list[ArchivePeriod]:
    """Break an inclusive date range into the fewest Binance archives needed.

    Whole calendar months map to a single monthly archive; a partially covered
    month falls back to one daily archive per day. Binance publishes both
    families, and preferring monthly keeps a two-year backfill at 24 requests
    instead of 731.
    """
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} precedes start_date {start_date}")

    periods: list[ArchivePeriod] = []
    cursor = start_date

    while cursor <= end_date:
        month_last_day = monthrange(cursor.year, cursor.month)[1]
        month_start = cursor.replace(day=1)
        month_end = cursor.replace(day=month_last_day)

        if cursor == month_start and month_end <= end_date:
            periods.append(
                ArchivePeriod(
                    aggregation=MONTHLY,
                    period=f"{cursor.year:04d}-{cursor.month:02d}",
                    start=month_start,
                    end=month_end,
                )
            )
            cursor = month_end + timedelta(days=1)
            continue

        partial_end = min(month_end, end_date)
        while cursor <= partial_end:
            periods.append(
                ArchivePeriod(
                    aggregation=DAILY,
                    period=cursor.isoformat(),
                    start=cursor,
                    end=cursor,
                )
            )
            cursor += timedelta(days=1)

    return periods


def parse_checksum(content: str, expected_filename: str | None = None) -> str:
    """Extract the SHA-256 digest from a Binance ``.CHECKSUM`` sidecar."""
    match = _SHA256_PATTERN.search(content)
    if match is None:
        raise DownloadError(
            f"No SHA-256 digest found in checksum sidecar: {content!r}"
        )
    if expected_filename and expected_filename not in content:
        raise DownloadError(
            f"Checksum sidecar names a different file than {expected_filename!r}: "
            f"{content!r}"
        )
    return match.group(1).lower()


def sha256_of_file(path: Path, chunk_size: int = 262_144) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BinanceKlineDownloader:
    """Fetches kline archives for one symbol, interval and date range."""

    def __init__(
        self,
        symbol: str,
        interval: str,
        start_date: date,
        end_date: date,
        raw_dir: Path,
        downloads_dir: Path,
        market: str = "spot",
        data_type: str = "klines",
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 60.0,
        chunk_size: int = 262_144,
        verify_existing: bool = True,
        session: requests.Session | None = None,
        progress: ProgressCallback | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.market = market
        self.data_type = data_type
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(1, max_retries)
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self.chunk_size = chunk_size
        self.verify_existing = verify_existing
        self.raw_dir = Path(raw_dir)
        self.downloads_dir = Path(downloads_dir)
        self._session = session or requests.Session()
        self._progress = progress
        self._sleep = sleep

    # --- Layout -------------------------------------------------------------

    @property
    def raw_symbol_dir(self) -> Path:
        """Raw archives live under ``<raw_dir>/<market>/<symbol>/<interval>``."""
        return self.raw_dir / self.market / self.symbol / self.interval

    def archive_url(self, period: ArchivePeriod) -> str:
        return ARCHIVE_URL_TEMPLATE.format(
            base_url=self.base_url,
            market=self.market,
            aggregation=period.aggregation,
            data_type=self.data_type,
            symbol=self.symbol,
            interval=self.interval,
            filename=period.filename(self.symbol, self.interval),
        )

    def raw_path(self, period: ArchivePeriod) -> Path:
        return self.raw_symbol_dir / period.filename(self.symbol, self.interval)

    def plan(self) -> list[ArchivePeriod]:
        return plan_periods(self.start_date, self.end_date)

    # --- Public entry point -------------------------------------------------

    def download(self) -> DownloadReport:
        """Ensure every planned archive is present and verified in the raw store."""
        periods = self.plan()
        self.raw_symbol_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

        report = DownloadReport()
        total = len(periods)
        self._report(f"Planned {total} archive(s) for {self.symbol} {self.interval}")

        for index, period in enumerate(periods, start=1):
            prefix = f"[{index}/{total}] {period.period}"
            try:
                archive = self._ensure_archive(period, prefix)
            except ArchiveNotFound:
                self._report(f"{prefix} not published upstream (HTTP 404)")
                report.missing.append(period)
                continue
            report.archives.append(archive)

        self._report(
            f"Raw store ready: {report.downloaded_count} downloaded, "
            f"{report.cached_count} already verified, {len(report.missing)} missing"
        )
        return report

    # --- Internals ----------------------------------------------------------

    def _ensure_archive(self, period: ArchivePeriod, prefix: str) -> DownloadedArchive:
        zip_path = self.raw_path(period)
        checksum_path = zip_path.with_name(zip_path.name + CHECKSUM_SUFFIX)
        url = self.archive_url(period)

        cached_digest = self._verified_cached_digest(zip_path, checksum_path, prefix)
        if cached_digest is not None:
            self._report(f"{prefix} cached")
            return DownloadedArchive(
                period=period,
                path=zip_path,
                url=url,
                sha256=cached_digest,
                from_cache=True,
            )

        expected = parse_checksum(
            self._fetch_text(url + CHECKSUM_SUFFIX), zip_path.name
        )
        actual = self._stage_and_verify(url, zip_path.name, expected)

        # Persist the sidecar next to the archive so later runs can re-verify
        # the raw store without any network access.
        self._write_atomic_text(checksum_path, f"{expected}  {zip_path.name}\n")

        self._report(f"{prefix} downloaded and verified")
        return DownloadedArchive(
            period=period, path=zip_path, url=url, sha256=actual, from_cache=False
        )

    def _verified_cached_digest(
        self, zip_path: Path, checksum_path: Path, prefix: str
    ) -> str | None:
        """Return the digest of an already-present, trustworthy archive."""
        if not zip_path.exists():
            return None
        if not checksum_path.exists():
            # No local sidecar means the archive was never verified here, so
            # fetch the official digest rather than trusting the local bytes.
            return None

        expected = parse_checksum(checksum_path.read_text(encoding="utf-8"))
        if not self.verify_existing:
            # A sidecar is only written after a successful verification, so its
            # presence alone is evidence; skip re-hashing on the fast path.
            return expected

        actual = sha256_of_file(zip_path, self.chunk_size)
        if actual == expected:
            return actual

        self._report(
            f"{prefix} local copy failed checksum verification, re-downloading"
        )
        return None

    def _stage_and_verify(self, url: str, filename: str, expected: str) -> str:
        """Stream to a ``.part`` file, verify, then atomically publish to raw."""
        part_path = self.downloads_dir / (filename + ".part")
        final_path = self.raw_symbol_dir / filename

        digest = hashlib.sha256()
        try:
            with self._request(url, stream=True) as response:
                with part_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        digest.update(chunk)

            actual = digest.hexdigest()
            if actual != expected:
                raise ChecksumMismatch(
                    f"{filename}: expected SHA-256 {expected}, got {actual}"
                )
            os.replace(part_path, final_path)
        finally:
            part_path.unlink(missing_ok=True)

        return actual

    def _write_atomic_text(self, path: Path, text: str) -> None:
        temp_path = self.downloads_dir / (path.name + ".part")
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)

    def _fetch_text(self, url: str) -> str:
        with self._request(url, stream=False) as response:
            return response.text

    def _request(self, url: str, stream: bool) -> requests.Response:
        """GET with exponential backoff on transient failures only."""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(
                    url, stream=stream, timeout=self.timeout_seconds
                )
            except requests.RequestException as error:
                last_error = error
            else:
                if response.status_code == 404:
                    response.close()
                    raise ArchiveNotFound(url)
                if response.status_code in _RETRYABLE_STATUS:
                    last_error = DownloadError(
                        f"HTTP {response.status_code} for {url}"
                    )
                    response.close()
                elif not response.ok:
                    status = response.status_code
                    response.close()
                    raise DownloadError(f"HTTP {status} for {url}")
                else:
                    return response

            if attempt < self.max_retries:
                delay = self.backoff_seconds * (2 ** (attempt - 1))
                self._report(
                    f"  attempt {attempt}/{self.max_retries} failed "
                    f"({last_error}); retrying in {delay:.1f}s"
                )
                self._sleep(delay)

        raise DownloadError(
            f"Giving up on {url} after {self.max_retries} attempts: {last_error}"
        )

    def _report(self, message: str) -> None:
        if self._progress is not None:
            self._progress(message)


def raw_archives_for(
    raw_dir: Path, market: str, symbol: str, interval: str
) -> Sequence[Path]:
    """List verified raw archives on disk, in filename (chronological) order."""
    directory = Path(raw_dir) / market / symbol.upper() / interval
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{symbol.upper()}-{interval}-*.zip"))
