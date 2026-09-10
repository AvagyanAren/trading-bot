"""CLI entry point for the v0.4 backtest engine.

Reads the v0.2 enriched store, consumes v0.3 signals unchanged, simulates
next-open fills with configured fees and slippage, and writes an integrity
report. Does not download data, does not talk to Binance, and does not
optimize the strategy.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.config import (  # noqa: E402
    BacktestConfigError,
    load_backtest_config,
)
from src.backtest.pipeline import (  # noqa: E402
    BacktestEvaluationError,
    render_console_summary,
    run_backtest_pipeline,
)
from src.data.pipeline import load_config  # noqa: E402
from src.indicators.engine import (  # noqa: E402
    IndicatorConfigError,
    load_indicator_config,
)
from src.strategy.engine import (  # noqa: E402
    StrategyConfigError,
    StrategyInputError,
    load_strategy_config,
)

DEFAULT_DATA_CONFIG = PROJECT_ROOT / "config" / "data.yaml"
DEFAULT_INDICATOR_CONFIG = PROJECT_ROOT / "config" / "indicators.yaml"
DEFAULT_STRATEGY_CONFIG = PROJECT_ROOT / "config" / "strategy.yaml"
DEFAULT_BACKTEST_CONFIG = PROJECT_ROOT / "config" / "backtest.yaml"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_backtest",
        description=(
            "Replay Strategy #1 on the enriched historical store with the "
            "configured execution simulator and write an integrity report."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Path to the data pipeline configuration (default: config/data.yaml)",
    )
    parser.add_argument(
        "--indicators-config",
        type=Path,
        default=DEFAULT_INDICATOR_CONFIG,
        help="Path to the indicator configuration (default: config/indicators.yaml)",
    )
    parser.add_argument(
        "--strategy-config",
        type=Path,
        default=DEFAULT_STRATEGY_CONFIG,
        help="Path to the strategy configuration (default: config/strategy.yaml)",
    )
    parser.add_argument(
        "--backtest-config",
        type=Path,
        default=DEFAULT_BACKTEST_CONFIG,
        help="Path to the backtest configuration (default: config/backtest.yaml)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step progress output; print only the final summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        data_config = load_config(args.config)
        indicator_config = load_indicator_config(args.indicators_config)
        strategy_config = load_strategy_config(args.strategy_config)
        backtest_config = load_backtest_config(args.backtest_config)
    except (
        OSError,
        KeyError,
        ValueError,
        IndicatorConfigError,
        StrategyConfigError,
        BacktestConfigError,
    ) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return EXIT_ERROR

    progress = None if args.quiet else lambda message: print(message, flush=True)

    try:
        result = run_backtest_pipeline(
            data_config,
            indicator_config,
            strategy_config,
            backtest_config,
            progress=progress,
        )
    except (BacktestEvaluationError, StrategyInputError) as error:
        print(f"\nBacktest aborted: {error}", file=sys.stderr)
        return EXIT_ERROR

    print(render_console_summary(result))
    return EXIT_PASS if result.passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
