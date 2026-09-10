"""Compose EMA, RSI and volume SMA. No formulas, no I/O, no trading logic.

``calculate_indicators`` copies the input frame, checks that the v0.1 schema
is present and already numeric, then appends four indicator columns whose
names are derived from ``config/indicators.yaml``.

Calculation inputs are ``close`` and ``volume`` in row order. Timestamps are
copied through and are never read by the calculators. Invalid input is not
coerced, filled, or repaired: the call fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .ema import IndicatorError, calculate_ema
from .rsi import calculate_rsi
from .volume import calculate_volume_sma

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

OHLCV_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS


class IndicatorConfigError(IndicatorError):
    """The indicator configuration file is missing or invalid."""


class IndicatorInputError(IndicatorError):
    """The input frame does not satisfy the structural contract."""


@dataclass(frozen=True)
class IndicatorConfig:
    """Resolved contents of ``config/indicators.yaml``."""

    ema_fast: int
    ema_slow: int
    rsi_period: int
    volume_ma_period: int
    enriched_dir: Path
    project_root: Path

    @property
    def ema_fast_column(self) -> str:
        return f"ema{self.ema_fast}"

    @property
    def ema_slow_column(self) -> str:
        return f"ema{self.ema_slow}"

    @property
    def rsi_column(self) -> str:
        return f"rsi{self.rsi_period}"

    @property
    def volume_ma_column(self) -> str:
        return f"volume_ma{self.volume_ma_period}"

    @property
    def indicator_columns(self) -> tuple[str, ...]:
        return (
            self.ema_fast_column,
            self.ema_slow_column,
            self.rsi_column,
            self.volume_ma_column,
        )


def _require_positive_int(raw: object, key: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise IndicatorConfigError(
            f"indicators.{key} must be a positive integer, got {raw!r}"
        )
    if raw < 1:
        raise IndicatorConfigError(f"indicators.{key} must be >= 1, got {raw}")
    return raw


def load_indicator_config(config_path: Path) -> IndicatorConfig:
    """Read ``config/indicators.yaml`` and resolve ``enriched_dir``."""
    config_path = Path(config_path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as error:
        raise IndicatorConfigError(f"Cannot read {config_path}: {error}") from error

    block = payload.get("indicators")
    if not isinstance(block, dict):
        raise IndicatorConfigError(
            f"{config_path} must contain an 'indicators' mapping"
        )

    project_root = config_path.parent.parent
    enriched = payload.get("enriched_dir", "data/enriched")
    return IndicatorConfig(
        ema_fast=_require_positive_int(block.get("ema_fast"), "ema_fast"),
        ema_slow=_require_positive_int(block.get("ema_slow"), "ema_slow"),
        rsi_period=_require_positive_int(block.get("rsi_period"), "rsi_period"),
        volume_ma_period=_require_positive_int(
            block.get("volume_ma_period"), "volume_ma_period"
        ),
        enriched_dir=(project_root / str(enriched)).resolve(),
        project_root=project_root,
    )


def _assert_numeric_float(frame: pd.DataFrame, column: str) -> None:
    """Fail if ``column`` is missing or not an already-numeric float dtype.

    v0.1 is responsible for producing float64 close/volume. This check does
    not coerce object/string columns into numbers.
    """
    if column not in frame.columns:
        raise IndicatorInputError(f"Required column {column!r} is missing")
    dtype = frame[column].dtype
    if not pd.api.types.is_float_dtype(dtype):
        raise IndicatorInputError(
            f"Column {column!r} must be a floating-point series "
            f"(v0.1 contract), got dtype {dtype}"
        )


def _assert_structure(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise IndicatorInputError(
            f"Input is missing required column(s): {', '.join(missing)}"
        )
    _assert_numeric_float(frame, "close")
    _assert_numeric_float(frame, "volume")


def calculate_indicators(
    frame: pd.DataFrame, config: IndicatorConfig
) -> pd.DataFrame:
    """Return a copy of ``frame`` with the four configured indicator columns.

    The input is not mutated. Row order is preserved. Indicator math uses
    only ``close`` and ``volume`` in that order; ``timestamp`` is not a
    calculation input.
    """
    _assert_structure(frame)
    out = frame.copy(deep=True)
    out[config.ema_fast_column] = calculate_ema(out["close"], config.ema_fast)
    out[config.ema_slow_column] = calculate_ema(out["close"], config.ema_slow)
    out[config.rsi_column] = calculate_rsi(out["close"], config.rsi_period)
    out[config.volume_ma_column] = calculate_volume_sma(
        out["volume"], config.volume_ma_period
    )
    return out
