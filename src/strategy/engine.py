"""Compose entry conditions into a look-ahead-free signal.

``evaluate_closed_candle`` is the single strategy primitive: it sees one
already-closed candle plus the previous ``breakout_period`` highs. It does
not read the current high for the breakout level, does not read future rows,
and does not treat close as a fill.

``generate_signals`` walks a chronological frame and calls that primitive
with data at rows ``<= i`` only.

``breakout_period`` is read from strategy config. Indicator periods, including
``volume_ma_period``, come from ``IndicatorConfig`` and are used only to
resolve column names. The two lookbacks are never substituted for each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.data.normalizer import interval_to_timedelta
from src.indicators.engine import IndicatorConfig

from .breakout import breakout_level, close_breaks_out
from .conditions import ema_trend, rsi_filter, volume_confirmation
from .signals import ConditionBreakdown, Signal, SignalType

REQUIRED_OHLCV: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

SIGNAL_FRAME_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "candle_close_time",
    "earliest_execution_time",
    "signal",
    "reference_close",
    "breakout_level",
    "warmup",
    "ema_trend",
    "breakout",
    "volume_confirmation",
    "rsi_filter",
    "strategy_name",
    "strategy_version",
    "symbol",
    "interval",
)


class StrategyError(ValueError):
    """Invalid strategy configuration or input."""


class StrategyConfigError(StrategyError):
    """The strategy configuration file is missing or invalid."""


class StrategyInputError(StrategyError):
    """The input frame or candle does not satisfy the structural contract."""


@dataclass(frozen=True)
class StrategyConfig:
    """Resolved contents of ``config/strategy.yaml``.

    Does not contain indicator periods. Those live in ``IndicatorConfig``.
    ``breakout_period`` is independent of ``volume_ma_period``.
    """

    name: str
    version: str
    breakout_period: int
    volume_multiplier: float
    rsi_min: float
    rsi_max: float
    project_root: Path
    rsi_filter_enabled: bool = True


def _require_name(raw: object, key: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise StrategyConfigError(f"strategy.{key} must be a non-empty string")
    return raw.strip()


def _require_positive_int(raw: object, key: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise StrategyConfigError(
            f"strategy.{key} must be a positive integer, got {raw!r}"
        )
    if raw < 1:
        raise StrategyConfigError(f"strategy.{key} must be >= 1, got {raw}")
    return raw


def _require_number(raw: object, key: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise StrategyConfigError(
            f"strategy.{key} must be a number, got {raw!r}"
        )
    return float(raw)


def _require_bool(raw: object, key: str) -> bool:
    if not isinstance(raw, bool):
        raise StrategyConfigError(f"strategy.{key} must be a boolean, got {raw!r}")
    return raw


def load_strategy_config(config_path: Path) -> StrategyConfig:
    """Read ``config/strategy.yaml``. Indicator periods are not loaded here."""
    config_path = Path(config_path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as error:
        raise StrategyConfigError(f"Cannot read {config_path}: {error}") from error

    block = payload.get("strategy")
    if not isinstance(block, dict):
        raise StrategyConfigError(f"{config_path} must contain a 'strategy' mapping")

    rsi_min = _require_number(block.get("rsi_min"), "rsi_min")
    rsi_max = _require_number(block.get("rsi_max"), "rsi_max")
    if not rsi_min < rsi_max:
        raise StrategyConfigError(
            f"strategy.rsi_min must be < strategy.rsi_max, got {rsi_min} >= {rsi_max}"
        )

    volume_multiplier = _require_number(
        block.get("volume_multiplier"), "volume_multiplier"
    )
    if volume_multiplier <= 0:
        raise StrategyConfigError(
            "strategy.volume_multiplier must be > 0, "
            f"got {volume_multiplier}"
        )

    version = block.get("version", "0.3")
    if isinstance(version, bool) or not isinstance(version, (str, int, float)):
        raise StrategyConfigError(
            f"strategy.version must be a string, got {version!r}"
        )

    rsi_filter_raw = block.get("rsi_filter_enabled", True)
    rsi_filter_enabled = _require_bool(rsi_filter_raw, "rsi_filter_enabled")

    return StrategyConfig(
        name=_require_name(block.get("name"), "name"),
        version=str(version).strip(),
        breakout_period=_require_positive_int(
            block.get("breakout_period"), "breakout_period"
        ),
        volume_multiplier=volume_multiplier,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        project_root=config_path.parent.parent,
        rsi_filter_enabled=rsi_filter_enabled,
    )


def _as_float(value: object) -> float:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except (TypeError, ValueError):
        pass
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise StrategyInputError(f"Expected a numeric value, got {value!r}") from error


def _finite_or_none(value: object) -> float | None:
    number = _as_float(value)
    if np.isnan(number) or not np.isfinite(number):
        return None
    return float(number)


def _get(candle: Mapping[str, Any] | pd.Series, key: str) -> object:
    if isinstance(candle, pd.Series):
        if key in candle.index:
            return candle[key]
        return None
    return candle.get(key)


def _as_timestamp(value: object, field: str) -> pd.Timestamp:
    if value is None or pd.isna(value):
        raise StrategyInputError(f"{field} is missing")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise StrategyInputError(f"{field} must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def evaluate_closed_candle(
    candle: Mapping[str, Any] | pd.Series,
    previous_highs: Sequence[float] | np.ndarray,
    strategy_config: StrategyConfig,
    indicator_config: IndicatorConfig,
    *,
    interval: str,
    symbol: str = "",
) -> Signal:
    """Evaluate the hypothesis at the close of one candle.

    ``previous_highs`` must be the highs of the previous ``breakout_period``
    closed candles (N-period through N-1). The current candle's high is not
    used. ``reference_close`` is not a fill. ``earliest_execution_time`` is
    the next interval boundary, which may lie outside a historical dataset.
    """
    timestamp = _as_timestamp(_get(candle, "timestamp"), "timestamp")
    close = _as_float(_get(candle, "close"))
    volume = _as_float(_get(candle, "volume"))
    ema_fast = _as_float(_get(candle, indicator_config.ema_fast_column))
    ema_slow = _as_float(_get(candle, indicator_config.ema_slow_column))
    rsi = _as_float(_get(candle, indicator_config.rsi_column))
    volume_ma = _as_float(_get(candle, indicator_config.volume_ma_column))

    previous = np.asarray(previous_highs, dtype="float64")
    if previous.size != strategy_config.breakout_period:
        level_raw = float("nan")
        breakout = None
    else:
        level_raw = breakout_level(previous)
        breakout = close_breaks_out(close, level_raw)

    conditions = ConditionBreakdown(
        ema_trend=ema_trend(ema_fast, ema_slow),
        breakout=breakout,
        volume_confirmation=volume_confirmation(
            volume, volume_ma, strategy_config.volume_multiplier
        ),
        rsi_filter=rsi_filter(rsi, strategy_config.rsi_min, strategy_config.rsi_max),
    )
    warmup_parts = [
        conditions.ema_trend,
        conditions.breakout,
        conditions.volume_confirmation,
    ]
    if strategy_config.rsi_filter_enabled:
        warmup_parts.append(conditions.rsi_filter)
    warmup = any(value is None for value in warmup_parts)
    fired = (
        conditions.ema_trend is True
        and conditions.breakout is True
        and conditions.volume_confirmation is True
        and (
            (not strategy_config.rsi_filter_enabled)
            or conditions.rsi_filter is True
        )
    )
    earliest = timestamp + interval_to_timedelta(interval)
    return Signal(
        strategy_name=strategy_config.name,
        strategy_version=strategy_config.version,
        symbol=symbol,
        interval=interval,
        timestamp=timestamp,
        candle_close_time=_optional_timestamp(_get(candle, "close_time")),
        earliest_execution_time=earliest,
        signal=SignalType.LONG_ENTRY if fired else SignalType.NONE,
        reference_close=_finite_or_none(close),
        breakout_level=_finite_or_none(level_raw),
        warmup=warmup,
        conditions=conditions,
    )


def _assert_numeric_float(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise StrategyInputError(f"Required column {column!r} is missing")
    dtype = frame[column].dtype
    if not pd.api.types.is_float_dtype(dtype):
        raise StrategyInputError(
            f"Column {column!r} must be a floating-point series "
            f"(v0.2 enriched contract), got dtype {dtype}"
        )


def _assert_structure(frame: pd.DataFrame, indicator_config: IndicatorConfig) -> None:
    missing = [column for column in REQUIRED_OHLCV if column not in frame.columns]
    missing.extend(
        column
        for column in indicator_config.indicator_columns
        if column not in frame.columns
    )
    if missing:
        raise StrategyInputError(
            f"Input is missing required column(s): {', '.join(missing)}"
        )
    tz = getattr(frame["timestamp"].dtype, "tz", None)
    if str(tz) != "UTC":
        raise StrategyInputError(
            f"timestamp dtype must be datetime64[ns, UTC], got {frame['timestamp'].dtype}"
        )
    for column in ("high", "close", "volume", *indicator_config.indicator_columns):
        _assert_numeric_float(frame, column)


def _empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame({column: [] for column in SIGNAL_FRAME_COLUMNS})


def _signal_to_row(signal: Signal) -> dict[str, object]:
    return {
        "timestamp": signal.timestamp,
        "candle_close_time": signal.candle_close_time,
        "earliest_execution_time": signal.earliest_execution_time,
        "signal": signal.signal.value,
        "reference_close": (
            np.nan if signal.reference_close is None else signal.reference_close
        ),
        "breakout_level": (
            np.nan if signal.breakout_level is None else signal.breakout_level
        ),
        "warmup": signal.warmup,
        "ema_trend": signal.conditions.ema_trend,
        "breakout": signal.conditions.breakout,
        "volume_confirmation": signal.conditions.volume_confirmation,
        "rsi_filter": signal.conditions.rsi_filter,
        "strategy_name": signal.strategy_name,
        "strategy_version": signal.strategy_version,
        "symbol": signal.symbol,
        "interval": signal.interval,
    }


def generate_signals(
    frame: pd.DataFrame,
    strategy_config: StrategyConfig,
    indicator_config: IndicatorConfig,
    *,
    interval: str,
    symbol: str = "",
) -> pd.DataFrame:
    """Evaluate every row sequentially. The input frame is not mutated.

    At index ``i`` the breakout window is ``high[i - period : i]`` (current
    high excluded). Row ``i + 1`` is never read. The last row may emit
    LONG_ENTRY; its ``earliest_execution_time`` is still only a timing label.
    """
    _assert_structure(frame, indicator_config)
    if frame.empty:
        return _empty_signal_frame()

    highs = frame["high"].to_numpy(dtype="float64", copy=True)
    period = strategy_config.breakout_period
    rows: list[dict[str, object]] = []
    for index in range(len(frame)):
        previous = highs[index - period : index] if index >= period else highs[0:0]
        rows.append(
            _signal_to_row(
                evaluate_closed_candle(
                    frame.iloc[index],
                    previous,
                    strategy_config,
                    indicator_config,
                    interval=interval,
                    symbol=symbol,
                )
            )
        )
    return pd.DataFrame(rows, columns=list(SIGNAL_FRAME_COLUMNS))
