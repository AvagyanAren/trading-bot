"""Load ``config/research.yaml`` and build isolated A/B/C/D configs.

Default strategy and backtest YAML stay the v0.3/v0.4 baselines. Research
only replaces the variant deltas. Starting capital must remain $20.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

import yaml

from src.backtest.config import BacktestConfig, load_backtest_config
from src.strategy.engine import StrategyConfig, load_strategy_config

VARIANT_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
REQUIRED_STARTING_CAPITAL = 20.0
ROLE_DEVELOPMENT = "DEVELOPMENT"
ROLE_TEST = "OUT-OF-SAMPLE / UNTOUCHED TEST"
FORBIDDEN_VARIANT_KEYS = frozenset(
    {
        "starting_capital",
        "risk_per_trade",
        "stop_loss",
        "slippage",
        "entry_rate",
        "exit_rate",
        "entry_model",
        "same_candle_priority",
        "side",
        "leverage",
        "short",
    }
)
VARIANT_KEYS = frozenset(
    {
        "folder",
        "label",
        "breakout_period",
        "volume_multiplier",
        "rsi_min",
        "rsi_max",
        "rsi_filter_enabled",
        "take_profit",
    }
)


class ResearchError(ValueError):
    """Invalid research configuration or input."""


class ResearchConfigError(ResearchError):
    """The research configuration file is missing or invalid."""


@dataclass(frozen=True)
class PeriodSpec:
    start: date
    end: date
    role: str


@dataclass(frozen=True)
class EligibilityGates:
    """Research eligibility heuristics. Not statistically validated."""

    min_closed_trades: int
    max_drawdown_pct_of_starting_capital: float
    max_largest_win_share_of_wins: float
    min_months_with_trades: int
    max_single_month_net_share: float


@dataclass(frozen=True)
class VariantSpec:
    experiment_id: str
    folder: str
    label: str
    breakout_period: int
    volume_multiplier: float
    rsi_min: float
    rsi_max: float
    rsi_filter_enabled: bool
    take_profit: float


@dataclass(frozen=True)
class ResearchConfig:
    project_root: Path
    output_dir: Path
    development: PeriodSpec
    test: PeriodSpec
    gates: EligibilityGates
    variants: dict[str, VariantSpec]
    base_strategy: StrategyConfig
    base_backtest: BacktestConfig

    def variant(self, experiment_id: str) -> VariantSpec:
        key = str(experiment_id).strip().upper()
        if key not in self.variants:
            raise ResearchConfigError(
                f"Unknown experiment id {experiment_id!r}; "
                f"expected one of {list(VARIANT_ORDER)}"
            )
        return self.variants[key]

    def strategy_for(self, experiment_id: str) -> StrategyConfig:
        variant = self.variant(experiment_id)
        return replace(
            self.base_strategy,
            breakout_period=variant.breakout_period,
            volume_multiplier=variant.volume_multiplier,
            rsi_min=variant.rsi_min,
            rsi_max=variant.rsi_max,
            rsi_filter_enabled=variant.rsi_filter_enabled,
        )

    def backtest_for(self, experiment_id: str) -> BacktestConfig:
        variant = self.variant(experiment_id)
        return replace(self.base_backtest, take_profit=variant.take_profit)


def _require_mapping(raw: object, key: str) -> dict:
    if not isinstance(raw, dict):
        raise ResearchConfigError(f"{key} must be a mapping")
    return raw


def _require_str(raw: object, key: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ResearchConfigError(f"{key} must be a non-empty string, got {raw!r}")
    return raw.strip()


def _require_bool(raw: object, key: str) -> bool:
    if not isinstance(raw, bool):
        raise ResearchConfigError(f"{key} must be a boolean, got {raw!r}")
    return raw


def _require_int(raw: object, key: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ResearchConfigError(f"{key} must be an integer, got {raw!r}")
    return raw


def _require_number(raw: object, key: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ResearchConfigError(f"{key} must be a number, got {raw!r}")
    return float(raw)


def _require_positive_int(raw: object, key: str) -> int:
    value = _require_int(raw, key)
    if value < 1:
        raise ResearchConfigError(f"{key} must be >= 1, got {value}")
    return value


def _require_positive(raw: object, key: str) -> float:
    value = _require_number(raw, key)
    if value <= 0:
        raise ResearchConfigError(f"{key} must be > 0, got {value}")
    return value


def _as_date(raw: object, key: str) -> date:
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return date.fromisoformat(raw.strip())
        except ValueError as error:
            raise ResearchConfigError(f"{key} must be YYYY-MM-DD, got {raw!r}") from error
    raise ResearchConfigError(f"{key} must be a date, got {raw!r}")


def _load_period(raw: object, key: str, expected_role: str) -> PeriodSpec:
    block = _require_mapping(raw, key)
    start = _as_date(block.get("start"), f"{key}.start")
    end = _as_date(block.get("end"), f"{key}.end")
    if start > end:
        raise ResearchConfigError(f"{key}.start must be <= {key}.end")
    role = _require_str(block.get("role"), f"{key}.role")
    if role != expected_role:
        raise ResearchConfigError(
            f"{key}.role must be {expected_role!r}, got {role!r}"
        )
    return PeriodSpec(start=start, end=end, role=role)


def _load_gates(raw: object) -> EligibilityGates:
    block = _require_mapping(raw, "gates")
    share = _require_number(block.get("max_largest_win_share_of_wins"), "gates.max_largest_win_share_of_wins")
    month_share = _require_number(
        block.get("max_single_month_net_share"), "gates.max_single_month_net_share"
    )
    dd_frac = _require_number(
        block.get("max_drawdown_pct_of_starting_capital"),
        "gates.max_drawdown_pct_of_starting_capital",
    )
    if not 0 < share <= 1:
        raise ResearchConfigError(
            "gates.max_largest_win_share_of_wins must be in (0, 1]"
        )
    if not 0 < month_share <= 1:
        raise ResearchConfigError(
            "gates.max_single_month_net_share must be in (0, 1]"
        )
    if not 0 < dd_frac <= 1:
        raise ResearchConfigError(
            "gates.max_drawdown_pct_of_starting_capital must be in (0, 1]"
        )
    return EligibilityGates(
        min_closed_trades=_require_positive_int(
            block.get("min_closed_trades"), "gates.min_closed_trades"
        ),
        max_drawdown_pct_of_starting_capital=dd_frac,
        max_largest_win_share_of_wins=share,
        min_months_with_trades=_require_positive_int(
            block.get("min_months_with_trades"), "gates.min_months_with_trades"
        ),
        max_single_month_net_share=month_share,
    )


def _load_variant(experiment_id: str, raw: object) -> VariantSpec:
    block = _require_mapping(raw, f"variants.{experiment_id}")
    unexpected = set(block) - VARIANT_KEYS
    forbidden = unexpected & FORBIDDEN_VARIANT_KEYS
    if forbidden:
        raise ResearchConfigError(
            f"variants.{experiment_id} cannot override {sorted(forbidden)}"
        )
    if unexpected:
        raise ResearchConfigError(
            f"variants.{experiment_id} has unknown key(s): {sorted(unexpected)}"
        )
    rsi_min = _require_number(block.get("rsi_min"), f"variants.{experiment_id}.rsi_min")
    rsi_max = _require_number(block.get("rsi_max"), f"variants.{experiment_id}.rsi_max")
    if not rsi_min < rsi_max:
        raise ResearchConfigError(
            f"variants.{experiment_id}.rsi_min must be < rsi_max"
        )
    return VariantSpec(
        experiment_id=experiment_id,
        folder=_require_str(block.get("folder"), f"variants.{experiment_id}.folder"),
        label=_require_str(block.get("label"), f"variants.{experiment_id}.label"),
        breakout_period=_require_positive_int(
            block.get("breakout_period"), f"variants.{experiment_id}.breakout_period"
        ),
        volume_multiplier=_require_positive(
            block.get("volume_multiplier"),
            f"variants.{experiment_id}.volume_multiplier",
        ),
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        rsi_filter_enabled=_require_bool(
            block.get("rsi_filter_enabled"),
            f"variants.{experiment_id}.rsi_filter_enabled",
        ),
        take_profit=_require_positive(
            block.get("take_profit"), f"variants.{experiment_id}.take_profit"
        ),
    )


def load_research_config(
    config_path: Path,
    *,
    strategy_config_path: Path | None = None,
    backtest_config_path: Path | None = None,
) -> ResearchConfig:
    config_path = Path(config_path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as error:
        raise ResearchConfigError(f"Cannot read {config_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ResearchConfigError(f"{config_path} must contain a mapping")

    project_root = config_path.parent.parent
    research = _require_mapping(payload.get("research"), "research")
    output_dir = project_root / Path(
        _require_str(research.get("output_dir"), "research.output_dir")
    )

    periods = _require_mapping(payload.get("periods"), "periods")
    development = _load_period(
        periods.get("development"), "periods.development", ROLE_DEVELOPMENT
    )
    test = _load_period(periods.get("test"), "periods.test", ROLE_TEST)
    if development.end >= test.start:
        raise ResearchConfigError(
            "development period must end before the test period starts"
        )

    variants_raw = _require_mapping(payload.get("variants"), "variants")
    if set(variants_raw) != set(VARIANT_ORDER):
        raise ResearchConfigError(
            f"variants must be exactly {list(VARIANT_ORDER)}, got {sorted(variants_raw)}"
        )
    variants = {
        experiment_id: _load_variant(experiment_id, variants_raw[experiment_id])
        for experiment_id in VARIANT_ORDER
    }

    strategy_path = (
        Path(strategy_config_path).resolve()
        if strategy_config_path is not None
        else project_root / "config" / "strategy.yaml"
    )
    backtest_path = (
        Path(backtest_config_path).resolve()
        if backtest_config_path is not None
        else project_root / "config" / "backtest.yaml"
    )
    base_strategy = load_strategy_config(strategy_path)
    base_backtest = load_backtest_config(backtest_path)
    if base_backtest.starting_capital != REQUIRED_STARTING_CAPITAL:
        raise ResearchConfigError(
            "backtest.starting_capital must be "
            f"{REQUIRED_STARTING_CAPITAL}, got {base_backtest.starting_capital}"
        )

    return ResearchConfig(
        project_root=project_root,
        output_dir=output_dir,
        development=development,
        test=test,
        gates=_load_gates(payload.get("gates")),
        variants=variants,
        base_strategy=base_strategy,
        base_backtest=base_backtest,
    )
