"""Reads Binance kline archives without trusting their CSV column names.

Binance began shipping a header row in newer archives while older ones open
straight on data, and the header spelling is not guaranteed. The field *order*
is, however, documented and stable, so this module detects whether a header is
present, discards it, and maps fields by position onto an explicit schema.

Everything is read as text here. Type casting is the normalizer's job, which
keeps parsing faithful to the bytes on disk and prevents pandas from silently
inferring a dtype for a column we have not inspected yet.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Documented Binance kline field order. Position is authoritative; the CSV's own
# header, when present, is ignored.
# https://github.com/binance/binance-public-data#klines
KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)

# The field Binance documents as unused. Dropped only after the normalizer has
# confirmed it carries no information.
IGNORED_COLUMN = "ignore"


class ParserError(RuntimeError):
    """The archive did not match the documented Binance kline layout."""


@dataclass(frozen=True)
class ParsedArchive:
    """One archive's rows, still as text, with the schema applied by position."""

    path: Path
    csv_name: str
    had_header: bool
    frame: pd.DataFrame

    @property
    def row_count(self) -> int:
        return len(self.frame)


def _select_csv_member(archive: zipfile.ZipFile, path: Path) -> str:
    members = [
        name for name in archive.namelist() if name.lower().endswith(".csv")
    ]
    if len(members) != 1:
        raise ParserError(
            f"{path.name}: expected exactly one CSV member, found {members!r}"
        )
    return members[0]


def _is_header_row(row: list[str]) -> bool:
    """A data row always opens on an integer epoch; a header row does not."""
    if not row:
        return True
    try:
        int(row[0].strip())
    except ValueError:
        return True
    return False


def read_kline_archive(path: Path) -> ParsedArchive:
    """Read one Binance kline ZIP into a text DataFrame with the internal schema."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            csv_name = _select_csv_member(archive, path)
            payload = archive.read(csv_name)
    except zipfile.BadZipFile as error:
        raise ParserError(f"{path.name}: not a readable ZIP archive ({error})") from error

    if not payload.strip():
        raise ParserError(f"{path.name}: archive member {csv_name!r} is empty")

    text = payload.decode("utf-8")
    first_row = next(csv.reader(io.StringIO(text)))
    had_header = _is_header_row(first_row)

    if len(first_row) != len(KLINE_COLUMNS):
        raise ParserError(
            f"{path.name}: expected {len(KLINE_COLUMNS)} kline fields "
            f"({', '.join(KLINE_COLUMNS)}) but the archive has {len(first_row)}: "
            f"{first_row!r}. Refusing to guess the column mapping."
        )

    frame = pd.read_csv(
        io.StringIO(text),
        header=None,
        names=list(KLINE_COLUMNS),
        skiprows=1 if had_header else 0,
        dtype=str,
        keep_default_na=False,
    )

    if frame.empty:
        raise ParserError(f"{path.name}: archive contains a header but no data rows")

    return ParsedArchive(
        path=path, csv_name=csv_name, had_header=had_header, frame=frame
    )
