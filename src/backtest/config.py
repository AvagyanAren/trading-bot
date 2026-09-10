"""Load and validate ``config/backtest.yaml``.

Starting capital, risk, fees, and execution assumptions live here. Strategy
and indicator parameters are not duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ENTRY_MODEL_NEXT_OPEN = "next_open"
SAME_CANDLE_STOP_LOSS = "stop_loss"


class BacktestError(ValueError):
    """Invalid backtest configuration or input."""


class BacktestConfigError(BacktestError):
    """The backtest configuration file is missing or invalid."""


class BacktestInputError(BacktestError):
    """The input frame does not satisfy the structural contract."""


@dataclass(frozen=True)
class BacktestConfig:
    """Resolved contents of ``config/backtest.yaml``."""

    starting_capital: float
    risk_per_trade: float
    stop_loss: float
    take_profit: float
    entry_model: str
    slippage: float
    same_candle_priority: str
    entry_rate: float
    exit_rate: float
    project_root: Path


def _require_mapping(raw: object, key: str, path: Path) -> dict:
    if not isinstance(raw, dict):
        raise BacktestConfigError(f"{path} must contain a {key!r} mapping")
    return raw


def _require_number(raw: object, key: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise BacktestConfigError(f"{key} must be a number, got {raw!r}")
    return float(raw)


def _require_positive(raw: object, key: str) -> float:
    value = _require_number(raw, key)
    if value <= 0:
        raise BacktestConfigError(f"{key} must be > 0, got {value}")
    return value


def _require_non_negative(raw: object, key: str) -> float:
    value = _require_number(raw, key)
    if value < 0:
        raise BacktestConfigError(f"{key} must be >= 0, got {value}")
    return value


def _require_choice(raw: object, key: str, allowed: frozenset[str]) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise BacktestConfigError(f"{key} must be a non-empty string, got {raw!r}")
    value = raw.strip()
    if value not in allowed:
        raise BacktestConfigError(
            f"{key} must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def load_backtest_config(config_path: Path) -> BacktestConfig:
    """Read ``config/backtest.yaml``. No TIME_EXIT / max holding parameters."""
    config_path = Path(config_path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as error:
        raise BacktestConfigError(f"Cannot read {config_path}: {error}") from error
    if not isinstance(payload, dict):
        raise BacktestConfigError(f"{config_path} must contain a mapping")

    if "max_holding_candles" in payload:
        raise BacktestConfigError(
            "max_holding_candles is not part of the v0.4 baseline"
        )

    backtest = _require_mapping(payload.get("backtest"), "backtest", config_path)
    risk = _require_mapping(payload.get("risk"), "risk", config_path)
    execution = _require_mapping(payload.get("execution"), "execution", config_path)
    fees = _require_mapping(payload.get("fees"), "fees", config_path)

    for block, name in (
        (backtest, "backtest"),
        (risk, "risk"),
        (execution, "execution"),
        (fees, "fees"),
    ):
        if "max_holding_candles" in block:
            raise BacktestConfigError(
                f"{name}.max_holding_candles is not part of the v0.4 baseline"
            )

    return BacktestConfig(
        starting_capital=_require_positive(
            backtest.get("starting_capital"), "backtest.starting_capital"
        ),
        risk_per_trade=_require_positive(
            risk.get("risk_per_trade"), "risk.risk_per_trade"
        ),
        stop_loss=_require_positive(risk.get("stop_loss"), "risk.stop_loss"),
        take_profit=_require_positive(risk.get("take_profit"), "risk.take_profit"),
        entry_model=_require_choice(
            execution.get("entry_model"),
            "execution.entry_model",
            frozenset({ENTRY_MODEL_NEXT_OPEN}),
        ),
        slippage=_require_non_negative(
            execution.get("slippage"), "execution.slippage"
        ),
        same_candle_priority=_require_choice(
            execution.get("same_candle_priority"),
            "execution.same_candle_priority",
            frozenset({SAME_CANDLE_STOP_LOSS}),
        ),
        entry_rate=_require_non_negative(fees.get("entry_rate"), "fees.entry_rate"),
        exit_rate=_require_non_negative(fees.get("exit_rate"), "fees.exit_rate"),
        project_root=config_path.parent.parent,
    )
